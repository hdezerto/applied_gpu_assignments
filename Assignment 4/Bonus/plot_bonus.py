import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                m = int(row["m"])
                n = int(row["n"])
                cpu = float(row["cpu_ms"])
                gemm = float(row["gemm_ms"])
                wmma = float(row["wmma_ms"])
            except (KeyError, ValueError):
                continue
            rows.append((m, n, cpu, gemm, wmma))
    rows.sort(key=lambda r: r[0])
    return rows


def plot(rows: List[tuple], output: Path):
    labels = [f"{m}x{n}" for m, n, _, _, _ in rows]
    cpu = [r[2] for r in rows]
    gemm = [r[3] for r in rows]
    wmma = [r[4] for r in rows]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_cpu = ax.bar(x - width, cpu, width, label="CPU (double)")
    bars_gemm = ax.bar(x, gemm, width, label="gemm (float)")
    bars_wmma = ax.bar(x + width, wmma, width, label="wmma (tensor core)")

    ax.bar_label(bars_cpu, padding=2, fmt="%.2g")
    ax.bar_label(bars_gemm, padding=2, fmt="%.2g")
    ax.bar_label(bars_wmma, padding=2, fmt="%.2g")

    ax.set_yscale("log")
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("CPU vs gemm vs wmma runtime")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot runtimes for bonus WMMA experiment")
    parser.add_argument("csv", type=Path, help="CSV file produced by wmma_bonus")
    parser.add_argument("--out", type=Path, default=Path("bonus_runtime.png"),
                        help="Output image path")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit("No valid rows found in CSV")

    plot(rows, args.out)


if __name__ == "__main__":
    main()
