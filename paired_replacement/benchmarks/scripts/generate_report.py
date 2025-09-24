#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_rows_skip_comments(path: Path):
    if not path or not path.exists():
        return []
    with open(path, "r", newline="") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
        if not lines:
            return []
        r = csv.DictReader(lines)
        return list(r)


def summarize_microbench(torch_csv: Path):
    rows = read_rows_skip_comments(torch_csv)
    if not rows:
        return None
    # Convert relevant fields
    data = []
    for r in rows:
        try:
            k = int(r["k"]) ; sp = float(r.get("speedup", "nan")); spm = float(r.get("speedup_mask", "nan"))
        except Exception:
            continue
        data.append((k, sp, spm))
    data.sort(key=lambda x: x[0])
    if not data:
        return None
    best_small_k = min(data, key=lambda t: t[0])
    max_speedup = max(data, key=lambda t: t[1] if t[1] == t[1] else -1)
    return {
        "count": len(data),
        "min_k_speedup": {"k": best_small_k[0], "speedup_idx": best_small_k[1], "speedup_mask": best_small_k[2]},
        "max_speedup": {"k": max_speedup[0], "speedup_idx": max_speedup[1]},
    }


def read_timing_ci(ci_csv: Path):
    rows = read_rows_skip_comments(ci_csv)
    out = []
    for r in rows:
        try:
            k = int(r["k"])
            pmu = float(r["paired_us_mean"]) ; imu = float(r["index_us_mean"]) ; mmu = float(r["mask_us_mean"]) ; sp = float(r["speedup_idx_mean"]) ; spm = float(r["speedup_mask_mean"]) ; pci = float(r.get("paired_us_ci95", 0.0)) ; ici = float(r.get("index_us_ci95", 0.0)) ; mci = float(r.get("mask_us_ci95", 0.0))
        except Exception:
            continue
        out.append({
            "k": k, "paired_us_mean": pmu, "index_us_mean": imu, "mask_us_mean": mmu,
            "speedup_idx_mean": sp, "speedup_mask_mean": spm,
            "paired_us_ci95": pci, "index_us_ci95": ici, "mask_us_ci95": mci,
        })
    return out


def read_perf_summary(perf_csv: Path):
    rows = read_rows_skip_comments(perf_csv)
    by_mode = {"paired": [], "rebuild": []}
    for r in rows:
        if r.get("mode") in by_mode:
            by_mode[r["mode"]].append(r)
    for mode in by_mode:
        by_mode[mode].sort(key=lambda x: int(x["k"]))
    return by_mode


def read_instruments_json(path: Path):
    if not path or not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Generate a concise markdown report from benchmarks")
    ap.add_argument("--torch_csv", type=str, default="benchmarks/results_microbench_torch.csv")
    ap.add_argument("--timing_ci_csv", type=str, default="benchmarks/results_timing_ci.csv")
    ap.add_argument("--overlay_png", type=str, default="benchmarks/microbench_speedup_overlays.png")
    ap.add_argument("--perf_csv", type=str, default="")
    ap.add_argument("--perf_png", type=str, default="")
    ap.add_argument("--e2e_csv", type=str, default="")
    ap.add_argument("--e2e_png", type=str, default="")
    ap.add_argument("--instruments_json", type=str, default="")
    ap.add_argument("--out_md", type=str, default="benchmarks/report.md")
    args = ap.parse_args()

    torch_csv = Path(args.torch_csv)
    timing_ci_csv = Path(args.timing_ci_csv)
    overlay_png = Path(args.overlay_png)
    perf_csv = Path(args.perf_csv) if args.perf_csv else None
    perf_png = Path(args.perf_png) if args.perf_png else None
    e2e_csv = Path(args.e2e_csv) if args.e2e_csv else None
    e2e_png = Path(args.e2e_png) if args.e2e_png else None
    instruments_json = Path(args.instruments_json) if args.instruments_json else None

    micro = summarize_microbench(torch_csv)
    timing = read_timing_ci(timing_ci_csv)
    perf = read_perf_summary(perf_csv) if perf_csv and perf_csv.exists() else None
    inst = read_instruments_json(instruments_json) if instruments_json and instruments_json.exists() else None

    Path(Path(args.out_md).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("# Paired Replacement Benchmark Report\n\n")
        f.write("## Microbench Overlays\n")
        if micro:
            f.write(f"- Data points: {micro['count']}\n")
            f.write(f"- Min-k speedup: k={micro['min_k_speedup']['k']}, idx={micro['min_k_speedup']['speedup_idx']:.2f}x, mask={micro['min_k_speedup']['speedup_mask']:.2f}x\n")
            f.write(f"- Max speedup vs index_select: {micro['max_speedup']['speedup_idx']:.2f}x at k={micro['max_speedup']['k']}\n\n")
        if overlay_png.exists():
            f.write(f"![Speedup overlays]({overlay_png})\n\n")

        f.write("## Timing (mean and 95% CI)\n")
        if timing:
            for row in timing:
                f.write(
                    f"- k={row['k']}: paired={row['paired_us_mean']:.1f}±{row['paired_us_ci95']:.1f}us, "
                    f"index={row['index_us_mean']:.1f}±{row['index_us_ci95']:.1f}us, "
                    f"mask={row['mask_us_mean']:.1f}±{row['mask_us_ci95']:.1f}us, "
                    f"speedup idx={row['speedup_idx_mean']:.2f}x, mask={row['speedup_mask_mean']:.2f}x\n"
                )
            f.write("\n")

        if perf:
            f.write("## Linux perf summary (per-step normalized)\n")
            ks = [int(r["k"]) for r in perf["paired"]]
            f.write(f"- k values: {ks}\n")
            # Show derived BW if present
            paired_bw = [r.get("dram_bw_bytes_per_sec") for r in perf["paired"]]
            if any(paired_bw):
                f.write("- DRAM BW present (derived from IMC CAS counters).\n")
            if perf_png and perf_png.exists():
                f.write(f"\n![perf plots]({perf_png})\n\n")

        # E2E summary
        if e2e_csv and e2e_csv.exists():
            f.write("## End-to-End (MoE-like, CPU)\n")
            try:
                rows = read_rows_skip_comments(e2e_csv)
                # find best speedup and show typical config
                best = None
                for r in rows:
                    try:
                        sp = float(r.get("speedup", "nan"))
                        m = int(r.get("m", -1))
                        b = int(r.get("batch", -1))
                        ak = float(r.get("avg_k", "nan"))
                    except Exception:
                        continue
                    if sp == sp:  # not NaN
                        if best is None or sp > best[0]:
                            best = (sp, m, b, ak)
                if best:
                    f.write(
                        f"- Best speedup: {best[0]:.2f}× at m={best[1]}, batch={best[2]} (avg k≈{best[3]:.1f}).\n"
                    )
            except Exception:
                pass
            if e2e_png and e2e_png.exists():
                f.write(f"\n![E2E speedup]({e2e_png})\n\n")

        if inst:
            f.write("## Instruments summary (macOS)\n")
            p = inst.get("paired", {}) ; r = inst.get("rebuild", {}) ; d = inst.get("diff", {})
            if p and r:
                def getf(obj, key):
                    v = obj.get(key)
                    return f"{v:.2g}" if isinstance(v, (int, float)) else "NA"
                f.write(
                    f"- CPI paired={getf(p,'cpi')} vs rebuild={getf(r,'cpi')} (diff={getf(d,'cpi')})\n"
                )
                f.write("- Selected counters (sums over export):\n")
                for key in sorted(set(list(p.keys()) + list(r.keys()))):
                    if key in ("cpi",):
                        continue
                    f.write(f"  - {key}: paired={getf(p,key)}, rebuild={getf(r,key)}, diff={getf(d,key)}\n")

    print(f"Saved {args.out_md}")


if __name__ == "__main__":
    main()
