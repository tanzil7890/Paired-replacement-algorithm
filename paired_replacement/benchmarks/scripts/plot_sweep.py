#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_rows(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["N"] = int(row["N"]) ; row["ratio"] = float(row["ratio"]) ; row["m"] = int(row["m"]) ; row["k"] = int(row["k"]) ; row["speedup_idx"] = float(row["speedup_idx"]) ; row["speedup_mask"] = float(row["speedup_mask"]) ; row["pattern"] = row["pattern"].strip()
                rows.append(row)
            except Exception:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser(description="Plot sweep results: speedup vs k for each ratio and pattern")
    ap.add_argument("--csv", type=str, default="benchmarks/results_sweep.csv")
    ap.add_argument("--out", type=str, default="benchmarks/sweep_plots.png")
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    if not rows:
        raise SystemExit("No rows in sweep CSV")
    ratios = sorted({r["ratio"] for r in rows})
    patterns = sorted({r["pattern"] for r in rows})

    fig, axs = plt.subplots(len(ratios), len(patterns), figsize=(5*len(patterns), 3.5*len(ratios)), squeeze=False)
    for i, ratio in enumerate(ratios):
        for j, pattern in enumerate(patterns):
            subset = [r for r in rows if r["ratio"] == ratio and r["pattern"] == pattern]
            subset.sort(key=lambda x: x["k"])
            ks = [r["k"] for r in subset]
            sp = [r["speedup_idx"] for r in subset]
            ax = axs[i][j]
            ax.plot(ks, sp, marker='o')
            ax.set_xscale('log') if max(ks) / max(1, min(ks)) > 32 else None
            ax.set_title(f"ratio={ratio:.2f}, pattern={pattern}")
            ax.set_xlabel("k")
            ax.set_ylabel("Speedup vs index_select")
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

