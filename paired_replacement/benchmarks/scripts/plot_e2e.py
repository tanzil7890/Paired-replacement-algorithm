#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path):
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    return rows


def main():
    ap = argparse.ArgumentParser(description="Plot end-to-end MoE-like speedups")
    ap.add_argument("--csv", type=str, default="benchmarks/data/results_e2e_torch.csv")
    ap.add_argument("--out", type=str, default="benchmarks/plots/e2e_speedup.png")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--up_cols", type=int, default=None)
    ap.add_argument("--sticky_beta", type=float, default=None)
    ap.add_argument("--mode", type=str, choices=["paired","hybrid"], default="paired",
                    help="Which speedup to plot vs index_select")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    rows = read_rows(csv_path)
    # Optional filtering for consistent presentation
    def keep(r):
        try:
            if args.N is not None and int(r.get("N", -1)) != args.N:
                return False
            if args.hidden_dim is not None and int(r.get("hidden_dim", -1)) != args.hidden_dim:
                return False
            if args.up_cols is not None and int(r.get("up_cols", -1)) != args.up_cols:
                return False
            if args.sticky_beta is not None and float(r.get("sticky_beta", "nan")) != args.sticky_beta:
                return False
            return True
        except Exception:
            return False
    rows = [r for r in rows if keep(r)]
    # Organize by batch size
    by_batch = {}
    for r in rows:
        try:
            b = int(r["batch"]) ; m = int(r["m"]) ; k = float(r.get("avg_k", "nan"))
            sp = float(r["speedup"]) if args.mode == "paired" else float(r.get("hybrid_speedup", "nan"))
        except Exception:
            continue
        by_batch.setdefault(b, []).append((m, sp, k))

    fig, ax = plt.subplots(figsize=(6, 4))
    for b, data in sorted(by_batch.items()):
        data.sort(key=lambda t: t[0])
        ms = [t[0] for t in data]
        sp = [t[1] for t in data]
        ax.plot(ms, sp, marker="o", label=f"batch={b}")

    ax.set_xlabel("Active rows m")
    ax.set_ylabel("Speedup vs index_select")
    title_mode = "Paired" if args.mode == "paired" else "Hybrid"
    ax.set_title(f"End-to-End MoE-like Speedup (CPU, {title_mode})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
