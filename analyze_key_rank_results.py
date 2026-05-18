"""Analyze 9-round key-rank supplementary results without numpy dependency.

This helper parses .npy files from:
  supplementary_data/data_9r_attack
and prints summary statistics plus success rates.
"""

from __future__ import annotations

import ast
import statistics
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class KeyRankStats:
    name: str
    n: int
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    min_v: float
    max_v: float
    top1: float
    top2: float
    top4: float
    top8: float
    top16: float
    top256: float


def load_npy_f64(path: Path) -> List[float]:
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{path} is not a .npy file.")

    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        header_start = 10
    elif major == 2:
        header_len = struct.unpack("<I", raw[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"Unsupported .npy version in {path}.")

    header = raw[header_start : header_start + header_len].decode("latin1").strip()
    meta = ast.literal_eval(header)
    dtype = meta["descr"]
    shape = meta["shape"]
    fortran_order = meta["fortran_order"]
    if fortran_order:
        raise ValueError(f"Fortran-order arrays are not supported: {path}")
    if dtype != "<f8":
        raise ValueError(f"Only <f8 is supported, got {dtype} in {path}.")

    size = 1
    for dim in shape:
        size *= dim

    data_start = header_start + header_len
    data = raw[data_start : data_start + size * 8]
    return list(struct.unpack("<" + ("d" * size), data))


def pct_le(values: Iterable[float], threshold: float) -> float:
    values = list(values)
    return 100.0 * sum(1 for v in values if v <= threshold) / len(values)


def analyze_one(path: Path) -> KeyRankStats:
    arr = load_npy_f64(path)
    arr_sorted = sorted(arr)
    n = len(arr)
    return KeyRankStats(
        name=path.name,
        n=n,
        mean=statistics.mean(arr),
        median=statistics.median(arr),
        p90=arr_sorted[int(0.90 * (n - 1))],
        p95=arr_sorted[int(0.95 * (n - 1))],
        p99=arr_sorted[int(0.99 * (n - 1))],
        min_v=arr_sorted[0],
        max_v=arr_sorted[-1],
        top1=pct_le(arr, 0),
        top2=pct_le(arr, 1),
        top4=pct_le(arr, 3),
        top8=pct_le(arr, 7),
        top16=pct_le(arr, 15),
        top256=pct_le(arr, 255),
    )


def main() -> None:
    data_dir = Path("supplementary_data/data_9r_attack")
    files = sorted(data_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    print("9-round key-rank summary (supplementary data):")
    for fp in files:
        s = analyze_one(fp)
        print(f"\n{s.name}")
        print(
            f"  n={s.n}, mean={s.mean:.3f}, median={s.median:.3f}, "
            f"p90={s.p90:.3f}, p95={s.p95:.3f}, p99={s.p99:.3f}, "
            f"min={s.min_v:.3f}, max={s.max_v:.3f}"
        )
        print(
            f"  top1={s.top1:.2f}%, top2={s.top2:.2f}%, top4={s.top4:.2f}%, "
            f"top8={s.top8:.2f}%, top16={s.top16:.2f}%, top256={s.top256:.2f}%"
        )


if __name__ == "__main__":
    main()
