# Speck32/64 论文复现指南（中文）

对应论文：**Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning** (CRYPTO 2019)。

## 1. 代码来源结论

这篇论文有作者公开的官方补充代码仓库：`agohr/deep_speck`。  
当前目录即该仓库代码（含预训练模型与部分补充数据）。

## 2. 运行环境

官方论文时代环境（兼容性差，但最贴近原文）：
- `requirements-legacy-paper.txt`

现代可运行环境（本仓库已做最小兼容修改）：
- `requirements-modern.txt`
- 修改点仅涉及：
  - `keras` / `tensorflow.keras` 双兼容导入；
  - 训练日志字段兼容（`val_acc` vs `val_accuracy`）；
  - 脚本参数化，避免默认一启动即长时间任务。

## 3. 实验脚本与论文内容映射

- `eval.py`  
  评估 5~8 轮神经区分器（real-vs-random + real-differences）。
- `train_5_rounds.py` + `train_nets.py`  
  训练 5 轮区分器（论文中的训练流程）。
- `test_key_recovery.py`  
  11 轮密钥恢复 PoC（论文核心攻击），并含 12 轮参数（扩展示例）。
- `key_rank.py`  
  9 轮攻击的 key-rank 统计（神经区分器版）。
- `key_rank_ddt.py`  
  9 轮攻击的 key-rank 统计（DDT 版，需额外加载 DDT）。
- `neural_difference_search.py`  
  few-shot 输入差分搜索实验。
- `cpp/`  
  论文中差分分布相关验证与 DDT 计算（极高计算/存储成本）。

## 4. 一键运行

快速烟雾测试（建议先跑）：

```powershell
pwsh .\run_repro.ps1 -Mode smoke -EvalSamples 200000
```

论文级默认参数（耗时很高）：

```powershell
pwsh .\run_repro.ps1 -Mode paper
```

## 5. 关键计算成本提醒

- `test_key_recovery.py` 的 12 轮攻击在 CPU 上可能单次数小时到十余小时。
- `cpp/speck_ddt.cpp` 对内存和时长要求极高（原仓库说明约 70GB RAM、数百 CPU 天）。

## 6. 常用单脚本命令

```powershell
python .\eval.py --samples 1000000
python .\test_key_recovery.py --runs-11r 100 --skip-12r
python .\test_key_recovery.py --quick
python .\key_rank.py --start-exp 5 --end-exp 8
python .\neural_difference_search.py --runs 10 --search-steps 2000
```
