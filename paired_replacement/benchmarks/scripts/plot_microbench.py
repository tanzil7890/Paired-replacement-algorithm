#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: str):
    rows = []
    with open(path, "r", newline="") as f:
        filtered = [line for line in f if not line.startswith("#")]
        if not filtered:
            return rows
        r = csv.DictReader(filtered)
        for row in r:
            if "k" in row and row["k"]:
                try:
                    row["k"] = int(row["k"]) 
                except Exception:
                    continue
            if "speedup" in row and row["speedup"]:
                try:
                    row["speedup"] = float(row["speedup"]) 
                except Exception:
                    row["speedup"] = None
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy_csv", type=str, default="benchmarks/results_microbench.csv")
    ap.add_argument("--torch_csv", type=str, default="benchmarks/results_microbench_torch.csv")
    ap.add_argument("--out", type=str, default="benchmarks/microbench_speedup.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4))

    if Path(args.numpy_csv).exists():
        rows = load_csv(args.numpy_csv)
        ks = [r["k"] for r in rows]
        sp = [r["speedup"] for r in rows]
        ax.plot(ks, sp, marker="o", label="NumPy ref")

    if Path(args.torch_csv).exists():
        rows = load_csv(args.torch_csv)
        ks = [r["k"] for r in rows]
        sp = [r["speedup"] for r in rows]
        ax.plot(ks, sp, marker="s", label="C++/PyTorch ext")

    ax.set_xlabel("Mask delta k")
    ax.set_ylabel("Speedup vs full rebuild")
    ax.set_title("Paired Replacement Microbench Speedup")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
