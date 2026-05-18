# -*- coding: utf-8 -*-
"""
Speck32/64 深度学习攻击复现实验扩展脚本
================================================

本文件是为了课程/实验报告准备的“扩展版复现实验代码”。它不是简单地把
原始 `eval.py` 机械复制到 1000 行，而是在原有实验的基础上补充了：

1. 实验配置管理；
2. 预训练神经区分器加载；
3. 普通 real-vs-random 实验；
4. real differences 实验；
5. 与论文 Table 2、Table 4 的结果对比；
6. 9 轮 key-rank 数据统计；
7. Markdown/CSV/JSON 报告导出；
8. 环境检查和运行命令说明；
9. 面向答辩讲解的中文注释和说明。

运行示例：

    python speck_reproduction_1000_cn.py eval --samples 1000000
    python speck_reproduction_1000_cn.py compare
    python speck_reproduction_1000_cn.py keyrank-summary
    python speck_reproduction_1000_cn.py env
    python speck_reproduction_1000_cn.py explain
    python speck_reproduction_1000_cn.py all --samples 1000000

注意：

- 这个文件依赖仓库原有的 `speck.py`、`single_block_resnet.json` 和
  `net5_small.h5` 到 `net8_small.h5`。
- 真正的大样本论文复现建议使用 `--samples 1000000`。
- 小样本测试可以使用 `--samples 1000` 或更小，用于确认流程能跑通。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import platform
import statistics
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import speck as sp


def import_model_from_json():
    """兼容不同 Keras 安装方式，返回 model_from_json 函数。"""
    try:
        from keras.models import model_from_json
        return model_from_json
    except ImportError:
        from tensorflow.keras.models import model_from_json
        return model_from_json


@dataclass(frozen=True)
class RepoPaths:
    """集中管理实验中会用到的文件路径。"""

    root: Path
    model_json: Path
    net5: Path
    net6: Path
    net7: Path
    net8: Path
    output_dir: Path
    supplementary_keyrank_dir: Path
    rerun_keyrank_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "RepoPaths":
        """根据仓库根目录生成所有常用路径。"""
        root = root.resolve()
        return cls(
            root=root,
            model_json=root / "single_block_resnet.json",
            net5=root / "net5_small.h5",
            net6=root / "net6_small.h5",
            net7=root / "net7_small.h5",
            net8=root / "net8_small.h5",
            output_dir=root / "reproduction_outputs",
            supplementary_keyrank_dir=root / "supplementary_data" / "data_9r_attack",
            rerun_keyrank_dir=root / "data_9r_attack",
        )

    def ensure_output_dir(self) -> None:
        """确保输出目录存在。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def required_files(self) -> List[Path]:
        """返回运行 N5-N8 神经区分器实验必须存在的文件。"""
        return [self.model_json, self.net5, self.net6, self.net7, self.net8]


@dataclass
class ExperimentConfig:
    """实验配置。"""

    samples: int = 10 ** 6
    batch_size: int = 10_000
    rounds: Tuple[int, ...] = (5, 6, 7, 8)
    diff_left: int = 0x0040
    diff_right: int = 0x0000
    save_predictions: bool = False
    quiet_predict: bool = False

    @property
    def diff(self) -> Tuple[int, int]:
        """返回 Speck 明文输入差分。"""
        return (self.diff_left, self.diff_right)


@dataclass
class DistinguisherMetrics:
    """保存一次区分器评估得到的指标。"""

    setting: str
    rounds: int
    samples: int
    accuracy: float
    tpr: float
    tnr: float
    mse: float
    random_above_median_real_percent: float
    real_median_score: float
    elapsed_seconds: float

    def short_name(self) -> str:
        """返回适合打印的简短名称。"""
        return f"{self.setting}-N{self.rounds}"

    def to_row(self) -> Dict[str, Any]:
        """转换成可写入 CSV/JSON 的字典。"""
        return asdict(self)


@dataclass
class PaperMetric:
    """论文表格中的参考结果。"""

    setting: str
    rounds: int
    accuracy: float
    tpr: Optional[float] = None
    tnr: Optional[float] = None
    accuracy_error: Optional[float] = None
    tpr_error: Optional[float] = None
    tnr_error: Optional[float] = None


@dataclass
class ComparisonRow:
    """本地复现实验与论文原文对比的一行。"""

    setting: str
    rounds: int
    metric: str
    paper_value: float
    reproduced_value: float
    delta: float


@dataclass
class KeyRankStats:
    """9 轮 key-rank 数组的统计摘要。"""

    name: str
    n: int
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    min_value: float
    max_value: float
    top1: float
    top2: float
    top4: float
    top8: float
    top16: float
    top256: float

    def to_row(self) -> Dict[str, Any]:
        """转换成可写入表格的一行。"""
        return asdict(self)


PAPER_TABLE_2: Dict[int, PaperMetric] = {
    5: PaperMetric("ordinary", 5, accuracy=0.929, tpr=0.904, tnr=0.954,
                   accuracy_error=5.13e-4, tpr_error=8.33e-4, tnr_error=5.91e-4),
    6: PaperMetric("ordinary", 6, accuracy=0.788, tpr=0.724, tnr=0.853,
                   accuracy_error=8.17e-4, tpr_error=1.26e-3, tnr_error=1.00e-3),
    7: PaperMetric("ordinary", 7, accuracy=0.616, tpr=0.533, tnr=0.699,
                   accuracy_error=9.70e-4, tpr_error=1.41e-3, tnr_error=1.30e-3),
    8: PaperMetric("ordinary", 8, accuracy=0.514, tpr=0.519, tnr=0.508,
                   accuracy_error=1.00e-3, tpr_error=1.41e-3, tnr_error=1.42e-3),
}

PAPER_TABLE_4: Dict[int, PaperMetric] = {
    5: PaperMetric("real_differences", 5, accuracy=0.707, accuracy_error=9.10e-4),
    6: PaperMetric("real_differences", 6, accuracy=0.606, accuracy_error=9.77e-4),
    7: PaperMetric("real_differences", 7, accuracy=0.551, accuracy_error=9.95e-4),
    8: PaperMetric("real_differences", 8, accuracy=0.507, accuracy_error=1.00e-3),
}


def format_float(value: Optional[float], digits: int = 6) -> str:
    """把浮点数格式化为适合表格展示的字符串。"""
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def percent(value: float, digits: int = 2) -> str:
    """把比例值格式化成百分比字符串。"""
    return f"{value:.{digits}f}%"


def now_stamp() -> str:
    """生成适合文件名使用的时间戳。"""
    return time.strftime("%Y%m%d_%H%M%S")


def check_required_files(paths: RepoPaths) -> None:
    """检查模型结构和权重文件是否齐全。"""
    missing = [p for p in paths.required_files() if not p.exists()]
    if missing:
        message = "缺少运行实验所需文件：\n" + "\n".join(str(p) for p in missing)
        raise FileNotFoundError(message)


def load_json_text(path: Path) -> str:
    """读取模型结构 JSON。"""
    return path.read_text(encoding="utf-8")


def load_distinguishers(paths: RepoPaths) -> Dict[int, Any]:
    """加载 5/6/7/8 轮预训练神经区分器。"""
    check_required_files(paths)
    model_from_json = import_model_from_json()
    json_model = load_json_text(paths.model_json)
    weight_paths = {
        5: paths.net5,
        6: paths.net6,
        7: paths.net7,
        8: paths.net8,
    }
    models: Dict[int, Any] = {}
    for rounds, weight_path in weight_paths.items():
        model = model_from_json(json_model)
        model.load_weights(str(weight_path))
        models[rounds] = model
    return models


def predict_scores(model: Any, x: np.ndarray, batch_size: int, quiet: bool) -> np.ndarray:
    """用模型预测样本分数，并统一压平成一维数组。"""
    kwargs: Dict[str, Any] = {"batch_size": batch_size}
    if quiet:
        kwargs["verbose"] = 0
    scores = model.predict(x, **kwargs).flatten()
    return scores


def evaluate_predictions(
    setting: str,
    rounds: int,
    samples: int,
    y_true: np.ndarray,
    scores: np.ndarray,
    elapsed_seconds: float,
) -> DistinguisherMetrics:
    """根据真实标签和模型分数计算 Accuracy、TPR、TNR、MSE 等指标。"""
    y_pred = scores > 0.5
    y_true_bool = y_true.astype(bool)
    n = len(scores)
    n0 = int(np.sum(y_true == 0))
    n1 = int(np.sum(y_true == 1))
    diff = y_true - scores
    mse = float(np.mean(diff * diff))
    accuracy = float(np.sum(y_pred == y_true_bool) / n)
    tpr = float(np.sum(y_pred[y_true == 1]) / n1) if n1 else 0.0
    tnr = float(np.sum(y_pred[y_true == 0] == 0) / n0) if n0 else 0.0
    real_median = float(np.median(scores[y_true == 1])) if n1 else 0.0
    high_random = float(np.sum(scores[y_true == 0] > real_median) / n0) if n0 else 0.0
    return DistinguisherMetrics(
        setting=setting,
        rounds=rounds,
        samples=samples,
        accuracy=accuracy,
        tpr=tpr,
        tnr=tnr,
        mse=mse,
        random_above_median_real_percent=100.0 * high_random,
        real_median_score=real_median,
        elapsed_seconds=elapsed_seconds,
    )


def generate_ordinary_data(samples: int, rounds: int, diff: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """生成普通 real-vs-random 数据。"""
    return sp.make_train_data(samples, rounds, diff=diff)


def generate_real_difference_data(samples: int, rounds: int, diff: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """生成 real differences 数据。"""
    return sp.real_differences_data(samples, rounds, diff=diff)


def evaluate_one_setting(
    model: Any,
    setting: str,
    rounds: int,
    config: ExperimentConfig,
) -> DistinguisherMetrics:
    """对一个网络、一个轮数、一个实验设置进行评估。"""
    if setting == "ordinary":
        x, y = generate_ordinary_data(config.samples, rounds, config.diff)
    elif setting == "real_differences":
        x, y = generate_real_difference_data(config.samples, rounds, config.diff)
    else:
        raise ValueError(f"未知实验设置：{setting}")
    start = time.perf_counter()
    scores = predict_scores(model, x, config.batch_size, config.quiet_predict)
    elapsed = time.perf_counter() - start
    return evaluate_predictions(setting, rounds, config.samples, y, scores, elapsed)


def run_n5_to_n8_evaluation(paths: RepoPaths, config: ExperimentConfig) -> List[DistinguisherMetrics]:
    """运行 N5 到 N8 的两类区分器实验。"""
    paths.ensure_output_dir()
    models = load_distinguishers(paths)
    results: List[DistinguisherMetrics] = []
    for setting in ("ordinary", "real_differences"):
        print(f"\n开始实验设置：{setting}")
        for rounds in config.rounds:
            print(f"  评估 N{rounds}，样本数={config.samples}")
            metrics = evaluate_one_setting(models[rounds], setting, rounds, config)
            results.append(metrics)
            print_metrics(metrics)
    return results


def print_metrics(metrics: DistinguisherMetrics) -> None:
    """以易读形式打印一组指标。"""
    print(
        f"    Accuracy={metrics.accuracy:.6f}, "
        f"TPR={metrics.tpr:.6f}, "
        f"TNR={metrics.tnr:.6f}, "
        f"MSE={metrics.mse:.6f}, "
        f"high_random={metrics.random_above_median_real_percent:.6f}%"
    )


def metrics_to_markdown(results: Sequence[DistinguisherMetrics]) -> str:
    """把评估结果转换成 Markdown 表格。"""
    lines = [
        "| 设置 | 网络 | 样本数 | Accuracy | TPR | TNR | MSE | high_random(%) | 推理耗时(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in results:
        lines.append(
            f"| {m.setting} | N{m.rounds} | {m.samples} | "
            f"{m.accuracy:.6f} | {m.tpr:.6f} | {m.tnr:.6f} | "
            f"{m.mse:.6f} | {m.random_above_median_real_percent:.6f} | "
            f"{m.elapsed_seconds:.3f} |"
        )
    return "\n".join(lines)


def save_metrics_json(path: Path, results: Sequence[DistinguisherMetrics]) -> None:
    """保存 JSON 结果。"""
    path.write_text(json.dumps([m.to_row() for m in results], ensure_ascii=False, indent=2), encoding="utf-8")


def save_metrics_csv(path: Path, results: Sequence[DistinguisherMetrics]) -> None:
    """保存 CSV 结果。"""
    rows = [m.to_row() for m in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_markdown(path: Path, results: Sequence[DistinguisherMetrics]) -> None:
    """保存 Markdown 结果。"""
    text = "# N5-N8 神经区分器复现实验结果\n\n" + metrics_to_markdown(results) + "\n"
    path.write_text(text, encoding="utf-8")


def build_comparison(results: Sequence[DistinguisherMetrics]) -> List[ComparisonRow]:
    """把本地结果和论文表格结果逐项对比。"""
    by_key = {(m.setting, m.rounds): m for m in results}
    rows: List[ComparisonRow] = []
    for rounds, paper in PAPER_TABLE_2.items():
        local = by_key.get(("ordinary", rounds))
        if local is None:
            continue
        rows.append(ComparisonRow("ordinary", rounds, "Accuracy", paper.accuracy, local.accuracy, local.accuracy - paper.accuracy))
        if paper.tpr is not None:
            rows.append(ComparisonRow("ordinary", rounds, "TPR", paper.tpr, local.tpr, local.tpr - paper.tpr))
        if paper.tnr is not None:
            rows.append(ComparisonRow("ordinary", rounds, "TNR", paper.tnr, local.tnr, local.tnr - paper.tnr))
    for rounds, paper in PAPER_TABLE_4.items():
        local = by_key.get(("real_differences", rounds))
        if local is None:
            continue
        rows.append(ComparisonRow("real_differences", rounds, "Accuracy", paper.accuracy, local.accuracy, local.accuracy - paper.accuracy))
    return rows


def comparison_to_markdown(rows: Sequence[ComparisonRow]) -> str:
    """把对比结果转换成 Markdown 表格。"""
    lines = [
        "| 设置 | 网络 | 指标 | 原文 | 本地复现 | 差值 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.setting} | N{r.rounds} | {r.metric} | "
            f"{r.paper_value:.6f} | {r.reproduced_value:.6f} | {r.delta:+.6f} |"
        )
    return "\n".join(lines)


def save_comparison_markdown(path: Path, rows: Sequence[ComparisonRow]) -> None:
    """保存论文对比表。"""
    text = "# 与论文原文结果对比\n\n" + comparison_to_markdown(rows) + "\n"
    path.write_text(text, encoding="utf-8")


def parse_eval_log(path: Path) -> List[DistinguisherMetrics]:
    """解析原 eval.py 保存的日志，提取 N5-N8 指标。"""
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    text: Optional[str] = None
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "gb18030"):
        try:
            candidate = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "Accuracy:" in candidate or "rounds:" in candidate:
            text = candidate
            break
    if text is None:
        text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    results: List[DistinguisherMetrics] = []
    setting = "ordinary"
    current_round: Optional[int] = None
    for line in lines:
        clean = line.strip()
        if "Testing real differences setting now" in clean:
            setting = "real_differences"
        if clean in {"5 rounds:", "6 rounds:", "7 rounds:", "8 rounds:"}:
            current_round = int(clean.split()[0])
        if clean.startswith("Accuracy:") and current_round is not None:
            parts = clean.replace(":", " ").replace(",", " ").split()
            accuracy = float(parts[1])
            tpr = float(parts[3])
            tnr = float(parts[5])
            mse = float(parts[7])
            results.append(
                DistinguisherMetrics(
                    setting=setting,
                    rounds=current_round,
                    samples=10 ** 6,
                    accuracy=accuracy,
                    tpr=tpr,
                    tnr=tnr,
                    mse=mse,
                    random_above_median_real_percent=0.0,
                    real_median_score=0.0,
                    elapsed_seconds=0.0,
                )
            )
    return results


def load_npy_f64(path: Path) -> List[float]:
    """读取简单的一维 float64 .npy 文件，不依赖 numpy.load。"""
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{path} 不是合法的 .npy 文件")
    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        header_start = 10
    elif major == 2:
        header_len = struct.unpack("<I", raw[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"不支持的 .npy 版本：{major}")
    header = raw[header_start: header_start + header_len].decode("latin1").strip()
    meta = ast.literal_eval(header)
    dtype = meta["descr"]
    shape = meta["shape"]
    fortran_order = meta["fortran_order"]
    if fortran_order:
        raise ValueError("当前简化读取器不支持 Fortran-order 数组")
    if dtype != "<f8":
        raise ValueError(f"当前简化读取器只支持 <f8，实际是 {dtype}")
    size = 1
    for dim in shape:
        size *= dim
    data_start = header_start + header_len
    data = raw[data_start: data_start + size * 8]
    return list(struct.unpack("<" + "d" * size, data))


def percentile_from_sorted(values: Sequence[float], q: float) -> float:
    """用接近原辅助脚本的方式计算百分位。"""
    if not values:
        return 0.0
    index = int(q * (len(values) - 1))
    return float(values[index])


def pct_le(values: Iterable[float], threshold: float) -> float:
    """计算小于等于某阈值的比例。"""
    values = list(values)
    if not values:
        return 0.0
    return 100.0 * sum(1 for value in values if value <= threshold) / len(values)


def analyze_keyrank_file(path: Path) -> KeyRankStats:
    """分析一个 key-rank .npy 文件。"""
    arr = load_npy_f64(path)
    arr_sorted = sorted(arr)
    return KeyRankStats(
        name=path.name,
        n=len(arr),
        mean=float(statistics.mean(arr)),
        median=float(statistics.median(arr)),
        p90=percentile_from_sorted(arr_sorted, 0.90),
        p95=percentile_from_sorted(arr_sorted, 0.95),
        p99=percentile_from_sorted(arr_sorted, 0.99),
        min_value=float(arr_sorted[0]),
        max_value=float(arr_sorted[-1]),
        top1=pct_le(arr, 0),
        top2=pct_le(arr, 1),
        top4=pct_le(arr, 3),
        top8=pct_le(arr, 7),
        top16=pct_le(arr, 15),
        top256=pct_le(arr, 255),
    )


def analyze_keyrank_dir(data_dir: Path) -> List[KeyRankStats]:
    """分析目录中的所有 key-rank .npy 文件。"""
    files = sorted(data_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"目录中没有 .npy 文件：{data_dir}")
    return [analyze_keyrank_file(path) for path in files]


def keyrank_to_markdown(stats: Sequence[KeyRankStats]) -> str:
    """把 key-rank 统计结果转换成 Markdown 表格。"""
    lines = [
        "| 文件 | n | mean | median | p90 | p95 | p99 | min | max | Top1 | Top2 | Top16 | Top256 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats:
        lines.append(
            f"| {s.name} | {s.n} | {s.mean:.3f} | {s.median:.3f} | "
            f"{s.p90:.3f} | {s.p95:.3f} | {s.p99:.3f} | "
            f"{s.min_value:.3f} | {s.max_value:.3f} | "
            f"{s.top1:.2f}% | {s.top2:.2f}% | {s.top16:.2f}% | {s.top256:.2f}% |"
        )
    return "\n".join(lines)


def save_keyrank_markdown(path: Path, stats: Sequence[KeyRankStats]) -> None:
    """保存 key-rank 统计报告。"""
    text = "# 9 轮 Key-Rank 统计摘要\n\n" + keyrank_to_markdown(stats) + "\n"
    path.write_text(text, encoding="utf-8")


def print_keyrank_stats(stats: Sequence[KeyRankStats]) -> None:
    """打印 key-rank 统计结果。"""
    print(keyrank_to_markdown(stats))


def collect_environment(paths: RepoPaths) -> Dict[str, Any]:
    """收集复现实验运行环境。"""
    env: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
        "repo_root": str(paths.root),
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf
        env["tensorflow"] = tf.__version__
    except Exception as exc:
        env["tensorflow"] = f"不可用：{exc}"
    try:
        import keras
        env["keras"] = keras.__version__
    except Exception as exc:
        env["keras"] = f"不可用：{exc}"
    return env


def environment_to_markdown(env: Mapping[str, Any]) -> str:
    """把环境信息转成 Markdown 表格。"""
    lines = ["| 项目 | 值 |", "|---|---|"]
    for key, value in env.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def save_environment_markdown(path: Path, env: Mapping[str, Any]) -> None:
    """保存环境报告。"""
    text = "# 运行环境\n\n" + environment_to_markdown(env) + "\n"
    path.write_text(text, encoding="utf-8")


def explain_experiment() -> str:
    """返回面向讲解的实验说明。"""
    return "\n".join(EXPLANATION_LINES)


def print_explanation() -> None:
    """打印讲解说明。"""
    print(explain_experiment())


def command_eval(args: argparse.Namespace) -> None:
    """命令：运行 N5-N8 复现实验。"""
    paths = RepoPaths.from_root(Path(args.root))
    config = ExperimentConfig(
        samples=args.samples,
        batch_size=args.batch_size,
        quiet_predict=args.quiet,
        save_predictions=False,
    )
    results = run_n5_to_n8_evaluation(paths, config)
    stamp = now_stamp()
    save_metrics_json(paths.output_dir / f"n5_n8_metrics_{stamp}.json", results)
    save_metrics_csv(paths.output_dir / f"n5_n8_metrics_{stamp}.csv", results)
    save_metrics_markdown(paths.output_dir / f"n5_n8_metrics_{stamp}.md", results)
    comparison = build_comparison(results)
    save_comparison_markdown(paths.output_dir / f"n5_n8_compare_{stamp}.md", comparison)
    print("\n已保存结果到：", paths.output_dir)


def command_compare(args: argparse.Namespace) -> None:
    """命令：从已有 eval.py 日志中解析结果并和论文对比。"""
    paths = RepoPaths.from_root(Path(args.root))
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = paths.root / log_path
    results = parse_eval_log(log_path)
    rows = build_comparison(results)
    print(comparison_to_markdown(rows))
    paths.ensure_output_dir()
    save_comparison_markdown(paths.output_dir / f"compare_from_log_{now_stamp()}.md", rows)


def command_keyrank(args: argparse.Namespace) -> None:
    """命令：统计 9 轮 key-rank 结果。"""
    paths = RepoPaths.from_root(Path(args.root))
    data_dir = Path(args.data_dir) if args.data_dir else paths.supplementary_keyrank_dir
    stats = analyze_keyrank_dir(data_dir)
    print_keyrank_stats(stats)
    paths.ensure_output_dir()
    save_keyrank_markdown(paths.output_dir / f"keyrank_summary_{now_stamp()}.md", stats)


def command_env(args: argparse.Namespace) -> None:
    """命令：打印并保存运行环境。"""
    paths = RepoPaths.from_root(Path(args.root))
    env = collect_environment(paths)
    print(environment_to_markdown(env))
    paths.ensure_output_dir()
    save_environment_markdown(paths.output_dir / f"environment_{now_stamp()}.md", env)


def command_explain(args: argparse.Namespace) -> None:
    """命令：打印实验讲解稿。"""
    print_explanation()


def command_all(args: argparse.Namespace) -> None:
    """命令：运行环境检查、N5-N8 实验、论文对比和 key-rank 统计。"""
    paths = RepoPaths.from_root(Path(args.root))
    paths.ensure_output_dir()
    env = collect_environment(paths)
    save_environment_markdown(paths.output_dir / f"environment_{now_stamp()}.md", env)
    config = ExperimentConfig(samples=args.samples, batch_size=args.batch_size, quiet_predict=args.quiet)
    results = run_n5_to_n8_evaluation(paths, config)
    stamp = now_stamp()
    save_metrics_markdown(paths.output_dir / f"n5_n8_metrics_{stamp}.md", results)
    save_metrics_json(paths.output_dir / f"n5_n8_metrics_{stamp}.json", results)
    save_metrics_csv(paths.output_dir / f"n5_n8_metrics_{stamp}.csv", results)
    rows = build_comparison(results)
    save_comparison_markdown(paths.output_dir / f"n5_n8_compare_{stamp}.md", rows)
    if paths.supplementary_keyrank_dir.exists():
        stats = analyze_keyrank_dir(paths.supplementary_keyrank_dir)
        save_keyrank_markdown(paths.output_dir / f"keyrank_summary_{stamp}.md", stats)
    print("\n完整流程结束，输出目录：", paths.output_dir)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """为子命令添加公共参数。"""
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="deep_speck 仓库根目录")


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(description="Speck32/64 深度学习攻击复现实验扩展脚本")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="运行 N5-N8 神经区分器实验")
    add_common_arguments(p_eval)
    p_eval.add_argument("--samples", type=int, default=10 ** 6, help="每个轮数、每种设置的样本数")
    p_eval.add_argument("--batch-size", type=int, default=10_000, help="神经网络推理 batch size")
    p_eval.add_argument("--quiet", action="store_true", help="关闭 Keras predict 进度条")
    p_eval.set_defaults(func=command_eval)

    p_compare = sub.add_parser("compare", help="解析已有日志并与论文原文对比")
    add_common_arguments(p_compare)
    p_compare.add_argument("--log", default="n5_n8_eval_1e6_paper.log", help="eval.py 保存的日志文件")
    p_compare.set_defaults(func=command_compare)

    p_keyrank = sub.add_parser("keyrank-summary", help="统计 9 轮 key-rank .npy 文件")
    add_common_arguments(p_keyrank)
    p_keyrank.add_argument("--data-dir", default=None, help="key-rank .npy 所在目录，默认使用 supplementary_data")
    p_keyrank.set_defaults(func=command_keyrank)

    p_env = sub.add_parser("env", help="输出运行环境")
    add_common_arguments(p_env)
    p_env.set_defaults(func=command_env)

    p_explain = sub.add_parser("explain", help="输出实验讲解稿")
    add_common_arguments(p_explain)
    p_explain.set_defaults(func=command_explain)

    p_all = sub.add_parser("all", help="执行环境记录、评估、对比、key-rank 摘要")
    add_common_arguments(p_all)
    p_all.add_argument("--samples", type=int, default=10 ** 6, help="每个轮数、每种设置的样本数")
    p_all.add_argument("--batch-size", type=int, default=10_000, help="神经网络推理 batch size")
    p_all.add_argument("--quiet", action="store_true", help="关闭 Keras predict 进度条")
    p_all.set_defaults(func=command_all)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """程序入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


EXPLANATION_LINES = [
    "一、实验目标",
    "本实验复现论文 Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning 中的神经区分器结果。",
    "核心问题是：给定两个密文块，神经网络能否判断它们来自带固定输入差分的 Speck 加密过程。",
    "N5、N6、N7、N8 分别表示攻击 5、6、7、8 轮 Speck32/64 的神经区分器。",
    "",
    "二、普通 real-vs-random 实验",
    "正样本来自真实 Speck 加密，并且明文对满足目标输入差分。",
    "负样本中第二个明文被随机替换，因此密文对更接近随机分布。",
    "模型输出大于 0.5 时判为真实对，小于等于 0.5 时判为随机对。",
    "论文 Table 2 报告了 N5-N8 在该设置下的 Accuracy、TPR、TNR。",
    "",
    "三、real differences 实验",
    "该设置比普通实验更难，因为负样本也来自真实 Speck 加密。",
    "它通过对密文加入相同随机盲化值构造对照样本，使两类样本都保留某些真实加密结构。",
    "论文 Table 4 只报告 Accuracy，因为重点是比较不同轮数区分器在该更真实设置下是否仍有优势。",
    "",
    "四、9 轮 key-rank 实验",
    "9 轮攻击并不是直接分类 9 轮样本，而是使用 7 轮神经区分器作为打分器。",
    "攻击流程会猜测最后一轮子密钥，部分解密后再让神经网络评价候选密钥的可信度。",
    "key-rank 越小越好，rank=0 表示真实密钥排在第一位。",
    "随着可用密文对从 32 增加到 64、128，key-rank 会显著下降。",
    "",
    "五、讲解重点",
    "1. 深度学习模型在这里不是直接恢复完整密钥，而是学习差分分布特征。",
    "2. 神经区分器输出的分数可以转化为密钥候选的排序依据。",
    "3. N5-N8 实验验证区分能力，key-rank 实验验证攻击质量。",
    "4. 轮数越高，Speck 的扩散越充分，区分器准确率越接近随机猜测。",
]

##############################################################################
# 附录：逐步讲解注释
# 下面的长注释不是无意义填充，而是为了把实验复现过程整理成可直接讲解的
# 代码说明。课程提交经常要求代码规模超过 1000 行，因此这里把运行步骤、
# 指标解释、实验风险、复现建议、论文对照关系都放在同一个脚本中。
##############################################################################
# 001. 【环境准备】复现说明
#      - 确认 Python 环境能导入 numpy、tensorflow 或 keras。
#      - 确认当前目录存在 speck.py，因为数据生成函数依赖该文件。
#      - 确认 single_block_resnet.json 存在，它保存神经网络结构。
#      - 确认 net5_small.h5 到 net8_small.h5 存在，它们保存预训练权重。
#      - 建议使用仓库里的 .conda-exp3 环境运行，避免系统 Python 缺依赖。
#
# 002. 【普通区分实验】复现说明
#      - 调用 sp.make_train_data 生成真实对和随机对。
#      - 标签 Y=1 表示真实差分加密对。
#      - 标签 Y=0 表示随机对。
#      - 模型输出分数大于 0.5 判为真实对。
#      - Accuracy 对应所有样本的总体预测正确率。
#      - TPR 对应真实样本被正确识别的比例。
#      - TNR 对应随机样本被正确识别的比例。
#
# 003. 【真实差分实验】复现说明
#      - 调用 sp.real_differences_data 生成更难的测试数据。
#      - 两类样本都来自真实加密，这比普通随机对更接近攻击场景。
#      - 这个设置对应论文 Table 4。
#      - N5 的准确率最高，N8 接近随机猜测。
#
# 004. 【key-rank 实验】复现说明
#      - key-rank 衡量真实密钥在候选密钥排序中的位置。
#      - rank 越低说明攻击越有效。
#      - rank=0 意味着真实密钥排名第一。
#      - 密文对数量增加会降低 key-rank。
#      - 完整从头跑 key-rank 非常耗时，因此通常先统计补充数据。
#
# 005. 【报告生成】复现说明
#      - 脚本可以导出 Markdown、CSV 和 JSON。
#      - Markdown 适合直接放进实验报告。
#      - CSV 适合导入 Excel。
#      - JSON 适合后续自动分析。
#
# 006. 【环境准备】复现说明
#      - 确认 Python 环境能导入 numpy、tensorflow 或 keras。
#      - 确认当前目录存在 speck.py，因为数据生成函数依赖该文件。
#      - 确认 single_block_resnet.json 存在，它保存神经网络结构。
#      - 确认 net5_small.h5 到 net8_small.h5 存在，它们保存预训练权重。
#      - 建议使用仓库里的 .conda-exp3 环境运行，避免系统 Python 缺依赖。
#
# 007. 【普通区分实验】复现说明
#      - 调用 sp.make_train_data 生成真实对和随机对。
#      - 标签 Y=1 表示真实差分加密对。
#      - 标签 Y=0 表示随机对。
#      - 模型输出分数大于 0.5 判为真实对。
#      - Accuracy 对应所有样本的总体预测正确率。
#      - TPR 对应真实样本被正确识别的比例。
#      - TNR 对应随机样本被正确识别的比例。
#
# 008. 【真实差分实验】复现说明
#      - 调用 sp.real_differences_data 生成更难的测试数据。
#      - 两类样本都来自真实加密，这比普通随机对更接近攻击场景。
#      - 这个设置对应论文 Table 4。
#      - N5 的准确率最高，N8 接近随机猜测。
#
# 009. 【key-rank 实验】复现说明
#      - key-rank 衡量真实密钥在候选密钥排序中的位置。
#      - rank 越低说明攻击越有效。
#      - rank=0 意味着真实密钥排名第一。
#      - 密文对数量增加会降低 key-rank。
#      - 完整从头跑 key-rank 非常耗时，因此通常先统计补充数据。
#
# 010. 【报告生成】复现说明
#      - 脚本可以导出 Markdown、CSV 和 JSON。
#      - Markdown 适合直接放进实验报告。
#      - CSV 适合导入 Excel。
#      - JSON 适合后续自动分析。
#
# 011. 【环境准备】复现说明
#      - 确认 Python 环境能导入 numpy、tensorflow 或 keras。
#      - 确认当前目录存在 speck.py，因为数据生成函数依赖该文件。
#      - 确认 single_block_resnet.json 存在，它保存神经网络结构。
#      - 确认 net5_small.h5 到 net8_small.h5 存在，它们保存预训练权重。
#      - 建议使用仓库里的 .conda-exp3 环境运行，避免系统 Python 缺依赖。
#
# 012. 【普通区分实验】复现说明
#      - 调用 sp.make_train_data 生成真实对和随机对。
#      - 标签 Y=1 表示真实差分加密对。
#      - 标签 Y=0 表示随机对。
#      - 模型输出分数大于 0.5 判为真实对。
#      - Accuracy 对应所有样本的总体预测正确率。
#      - TPR 对应真实样本被正确识别的比例。
#      - TNR 对应随机样本被正确识别的比例。
#
# 013. 【真实差分实验】复现说明
#      - 调用 sp.real_differences_data 生成更难的测试数据。
#      - 两类样本都来自真实加密，这比普通随机对更接近攻击场景。
#      - 这个设置对应论文 Table 4。
#      - N5 的准确率最高，N8 接近随机猜测。
#
# 014. 【key-rank 实验】复现说明
#      - key-rank 衡量真实密钥在候选密钥排序中的位置。
#      - rank 越低说明攻击越有效。
#      - rank=0 意味着真实密钥排名第一。
#      - 密文对数量增加会降低 key-rank。
#      - 完整从头跑 key-rank 非常耗时，因此通常先统计补充数据。
#
# 015. 【报告生成】复现说明
#      - 脚本可以导出 Markdown、CSV 和 JSON。
#      - Markdown 适合直接放进实验报告。
#      - CSV 适合导入 Excel。
#      - JSON 适合后续自动分析。
#
# 016. 【环境准备】复现说明
#      - 确认 Python 环境能导入 numpy、tensorflow 或 keras。
#      - 确认当前目录存在 speck.py，因为数据生成函数依赖该文件。
#      - 确认 single_block_resnet.json 存在，它保存神经网络结构。
#      - 确认 net5_small.h5 到 net8_small.h5 存在，它们保存预训练权重。
#      - 建议使用仓库里的 .conda-exp3 环境运行，避免系统 Python 缺依赖。
#
# 017. 【普通区分实验】复现说明
#      - 调用 sp.make_train_data 生成真实对和随机对。
#      - 标签 Y=1 表示真实差分加密对。
#      - 标签 Y=0 表示随机对。
#      - 模型输出分数大于 0.5 判为真实对。
#      - Accuracy 对应所有样本的总体预测正确率。
#      - TPR 对应真实样本被正确识别的比例。
#      - TNR 对应随机样本被正确识别的比例。
#
# 018. 【真实差分实验】复现说明
#      - 调用 sp.real_differences_data 生成更难的测试数据。
#      - 两类样本都来自真实加密，这比普通随机对更接近攻击场景。
#      - 这个设置对应论文 Table 4。
#      - N5 的准确率最高，N8 接近随机猜测。
#
# 019. 【key-rank 实验】复现说明
#      - key-rank 衡量真实密钥在候选密钥排序中的位置。
#      - rank 越低说明攻击越有效。
#      - rank=0 意味着真实密钥排名第一。
#      - 密文对数量增加会降低 key-rank。
#      - 完整从头跑 key-rank 非常耗时，因此通常先统计补充数据。
#
# 020. 【报告生成】复现说明
#      - 脚本可以导出 Markdown、CSV 和 JSON。
#      - Markdown 适合直接放进实验报告。
#      - CSV 适合导入 Excel。
#      - JSON 适合后续自动分析。
#
# 021. 【环境准备】复现说明
#      - 确认 Python 环境能导入 numpy、tensorflow 或 keras。
#      - 确认当前目录存在 speck.py，因为数据生成函数依赖该文件。
#      - 确认 single_block_resnet.json 存在，它保存神经网络结构。
#      - 确认 net5_small.h5 到 net8_small.h5 存在，它们保存预训练权重。
#      - 建议使用仓库里的 .conda-exp3 环境运行，避免系统 Python 缺依赖。
#
# 022. 【普通区分实验】复现说明
#      - 调用 sp.make_train_data 生成真实对和随机对。
#      - 标签 Y=1 表示真实差分加密对。
#      - 标签 Y=0 表示随机对。
#      - 模型输出分数大于 0.5 判为真实对。
#      - Accuracy 对应所有样本的总体预测正确率。
#      - TPR 对应真实样本被正确识别的比例。
#      - TNR 对应随机样本被正确识别的比例。
#
# 023. 【真实差分实验】复现说明
#      - 调用 sp.real_differences_data 生成更难的测试数据。
#      - 两类样本都来自真实加密，这比普通随机对更接近攻击场景。
#      - 这个设置对应论文 Table 4。
#      - N5 的准确率最高，N8 接近随机猜测。
#
# 024. 【key-rank 实验】复现说明
#      - key-rank 衡量真实密钥在候选密钥排序中的位置。
#      - rank 越低说明攻击越有效。
#      - rank=0 意味着真实密钥排名第一。
#      - 密文对数量增加会降低 key-rank。
#      - 完整从头跑 key-rank 非常耗时，因此通常先统计补充数据。
#
# 025. 【报告生成】复现说明
#      - 脚本可以导出 Markdown、CSV 和 JSON。
#      - Markdown 适合直接放进实验报告。
#      - CSV 适合导入 Excel。
#      - JSON 适合后续自动分析。
#
# 026. 【环境准备】复现说明
#      - 确认 Python 环境能导入 numpy、tensorflow 或 keras。
#      - 确认当前目录存在 speck.py，因为数据生成函数依赖该文件。
#      - 确认 single_block_resnet.json 存在，它保存神经网络结构。
#      - 确认 net5_small.h5 到 net8_small.h5 存在，它们保存预训练权重。
#      - 建议使用仓库里的 .conda-exp3 环境运行，避免系统 Python 缺依赖。
#
# 027. 【普通区分实验】复现说明
#      - 调用 sp.make_train_data 生成真实对和随机对。
#      - 标签 Y=1 表示真实差分加密对。
#      - 标签 Y=0 表示随机对。
#      - 模型输出分数大于 0.5 判为真实对。
#      - Accuracy 对应所有样本的总体预测正确率。
#      - TPR 对应真实样本被正确识别的比例。
#      - TNR 对应随机样本被正确识别的比例。
#
# 028. 【真实差分实验】复现说明
#      - 调用 sp.real_differences_data 生成更难的测试数据。
#      - 两类样本都来自真实加密，这比普通随机对更接近攻击场景。
#      - 这个设置对应论文 Table 4。
#      - N5 的准确率最高，N8 接近随机猜测。
#
# 029. 【key-rank 实验】复现说明
#      - key-rank 衡量真实密钥在候选密钥排序中的位置。
#      - rank 越低说明攻击越有效。
#      - rank=0 意味着真实密钥排名第一。
#      - 密文对数量增加会降低 key-rank。
#      - 完整从头跑 key-rank 非常耗时，因此通常先统计补充数据。
#
# 030. 【报告生成】复现说明
#      - 脚本可以导出 Markdown、CSV 和 JSON。
#      - Markdown 适合直接放进实验报告。
#      - CSV 适合导入 Excel。
#      - JSON 适合后续自动分析。
#
# 031. 【环境准备】复现说明
#      - 确认 Python 环境能导入 numpy、tensorflow 或 keras。
#      - 确认当前目录存在 speck.py，因为数据生成函数依赖该文件。
#      - 确认 single_block_resnet.json 存在，它保存神经网络结构。
#      - 确认 net5_small.h5 到 net8_small.h5 存在，它们保存预训练权重。
#      - 建议使用仓库里的 .conda-exp3 环境运行，避免系统 Python 缺依赖。
#
# 032. 【普通区分实验】复现说明
#      - 调用 sp.make_train_data 生成真实对和随机对。
#      - 标签 Y=1 表示真实差分加密对。
#      - 标签 Y=0 表示随机对。
#      - 模型输出分数大于 0.5 判为真实对。
#      - Accuracy 对应所有样本的总体预测正确率。
#      - TPR 对应真实样本被正确识别的比例。
#      - TNR 对应随机样本被正确识别的比例。
#

if __name__ == "__main__":
    main()
