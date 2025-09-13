#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: str):
    rows = []
    with open(path, "r", newline="") as f:
        # Skip comment banner lines starting with '#'
        filtered = [line for line in f if not line.startswith("#")]
        r = csv.DictReader(filtered)
        for row in r:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch_csv", type=str, default="benchmarks/results_microbench_torch.csv")
    ap.add_argument("--out", type=str, default="benchmarks/microbench_speedup_overlays.png")
    args = ap.parse_args()

    if not Path(args.torch_csv).exists():
        raise SystemExit(f"Missing {args.torch_csv}")

    rows = read_csv(args.torch_csv)
    rows.sort(key=lambda x: int(x["k"]))

    ks = [int(r["k"]) for r in rows]
    sp_idx = [float(r.get("speedup", "nan")) for r in rows]
    sp_mask = [float(r.get("speedup_mask", "nan")) for r in rows]
    sp_hybrid = [float(r.get("speedup_hybrid_vs_idx", "nan")) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, sp_idx, marker="o", label="Paired vs index_select")
    ax.plot(ks, sp_hybrid, marker="^", label="Hybrid vs index_select")
    ax.plot(ks, sp_mask, marker="s", label="Paired vs boolean mask")
    ax.set_xlabel("Mask delta k")
    ax.set_ylabel("Speedup vs baseline")
    ax.set_title("Paired Replacement Speedup (Torch Ext)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
