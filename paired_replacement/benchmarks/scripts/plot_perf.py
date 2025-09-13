#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_perf_csv(path: str):
    rows = []
    with open(path, "r", newline="") as f:
        filtered = [line for line in f if not line.startswith("#")]
        if not filtered:
            return rows
        r = csv.DictReader(filtered)
        for row in r:
            rows.append(row)
    return rows


def to_float(row, key):
    try:
        return float(row.get(key, float('nan')))
    except Exception:
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf_csv", type=str, default="benchmarks/results_perf_torch.csv")
    ap.add_argument("--out", type=str, default="benchmarks/perf_plots.png")
    args = ap.parse_args()

    if not Path(args.perf_csv).exists():
        raise SystemExit(f"Missing {args.perf_csv}")

    rows = read_perf_csv(args.perf_csv)
    by_mode = {"paired": [], "rebuild": []}
    for r in rows:
        if r.get("mode") in by_mode:
            by_mode[r["mode"]].append(r)

    for mode in by_mode:
        by_mode[mode].sort(key=lambda x: int(x["k"]))

    ks = [int(r["k"]) for r in by_mode["paired"]]

    def miss_rate(rows, loads_key, misses_key):
        rates = []
        for r in rows:
            loads = to_float(r, loads_key)
            miss = to_float(r, misses_key)
            if loads and loads > 0:
                rates.append(miss / loads)
            else:
                rates.append(float('nan'))
        return rates

    def cpi(rows):
        cps = []
        for r in rows:
            cycles = to_float(r, "cycles")
            inst = to_float(r, "instructions")
            if inst and inst > 0:
                cps.append(cycles / inst)
            else:
                cps.append(float('nan'))
        return cps

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # LLC load miss rate
    paired_llc = miss_rate(by_mode["paired"], "LLC-loads", "LLC-load-misses")
    rebuild_llc = miss_rate(by_mode["rebuild"], "LLC-loads", "LLC-load-misses")
    axs[0].plot(ks, paired_llc, marker='o', label='paired')
    axs[0].plot(ks, rebuild_llc, marker='s', label='rebuild')
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('LLC load miss rate')
    axs[0].set_title('LLC Miss Rate vs k')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # CPI
    paired_cpi = cpi(by_mode["paired"]) 
    rebuild_cpi = cpi(by_mode["rebuild"]) 
    axs[1].plot(ks, paired_cpi, marker='o', label='paired')
    axs[1].plot(ks, rebuild_cpi, marker='s', label='rebuild')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('cycles / instruction')
    axs[1].set_title('CPI vs k')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    # DRAM bandwidth (derived)
    def bw(rows):
        vals = []
        for r in rows:
            try:
                v = float(r.get('dram_bw_bytes_per_sec', 'nan'))
            except Exception:
                v = float('nan')
            vals.append(v / 1e9 if v == v else float('nan'))
        return vals
    paired_bw = bw(by_mode['paired'])
    rebuild_bw = bw(by_mode['rebuild'])
    axs[2].plot(ks, paired_bw, marker='o', label='paired')
    axs[2].plot(ks, rebuild_bw, marker='s', label='rebuild')
    axs[2].set_xlabel('k')
    axs[2].set_ylabel('GB/s (derived)')
    axs[2].set_title('DRAM BW vs k (derived)')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
