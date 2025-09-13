#!/usr/bin/env python3
import argparse
import csv
import statistics as stats
from pathlib import Path

import torch

from run_microbench_torch import run_microbench


def aggregate(results_list):
    # results_list: list of lists of dicts (one per run)
    by_k = {}
    for results in results_list:
        for r in results:
            k = int(r["k"]) if isinstance(r["k"], (int, str)) else r["k"]
            by_k.setdefault(k, {"paired_us": [], "rebuild_us": [], "rebuild_mask_us": []})
            by_k[k]["paired_us"].append(float(r["paired_us"]))
            by_k[k]["rebuild_us"].append(float(r["rebuild_us"]))
            by_k[k]["rebuild_mask_us"].append(float(r["rebuild_mask_us"]))
    rows = []
    for k in sorted(by_k.keys()):
        d = by_k[k]
        def mean_ci(vals):
            mu = stats.mean(vals)
            sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
            ci95 = 1.96 * (sd / (len(vals) ** 0.5)) if len(vals) > 1 else 0.0
            return mu, sd, ci95
        p_mu, p_sd, p_ci = mean_ci(d["paired_us"]) 
        i_mu, i_sd, i_ci = mean_ci(d["rebuild_us"]) 
        m_mu, m_sd, m_ci = mean_ci(d["rebuild_mask_us"]) 
        rows.append({
            "k": k,
            "paired_us_mean": p_mu,
            "paired_us_sd": p_sd,
            "paired_us_ci95": p_ci,
            "index_us_mean": i_mu,
            "index_us_sd": i_sd,
            "index_us_ci95": i_ci,
            "mask_us_mean": m_mu,
            "mask_us_sd": m_sd,
            "mask_us_ci95": m_ci,
            "speedup_idx_mean": i_mu / p_mu if p_mu > 0 else float('inf'),
            "speedup_mask_mean": m_mu / p_mu if p_mu > 0 else float('inf'),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Repeated microbench to compute mean/CI")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--ks", type=str, default="16,64,256,1024")
    ap.add_argument("--steps_per_k", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outfile", type=str, default="benchmarks/results_timing_ci.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    ks = [int(x) for x in args.ks.split(",") if x]

    all_runs = []
    for r in range(args.repeats):
        print(f"Run {r+1}/{args.repeats}")
        res = run_microbench(
            args.N, args.up_cols, args.hidden_dim, args.gate_cols, args.m,
            ks, args.steps_per_k, device, outfile=None
        )
        all_runs.append(res)

    rows = aggregate(all_runs)

    Path(Path(args.outfile).parent).mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {args.outfile}")


if __name__ == "__main__":
    main()

