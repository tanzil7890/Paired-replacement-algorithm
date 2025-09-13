#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


def read_json(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def read_csv_rows(path: Path) -> List[dict]:
    try:
        with open(path, "r", newline="") as f:
            # skip banner comments if present
            lines = [ln for ln in f if not ln.startswith("#")]
            if not lines:
                return []
            r = csv.DictReader(lines)
            return list(r)
    except Exception:
        return []


def pick_cpu_name(sysinfo: dict) -> str:
    # Try macOS keys
    details = sysinfo.get("details", {})
    if "machdep.cpu.brand_string" in details:
        return details["machdep.cpu.brand_string"]
    if "system_profiler" in details:
        import re
        m = re.search(r"\n\s*Chip:\s*(.+)\n", details["system_profiler"])  # e.g., "Apple M3"
        if m:
            return m.group(1).strip()
    # Try Linux lscpu
    if "Model name" in details:
        return details["Model name"]
    # Fallbacks
    plat = sysinfo.get("platform") or ""
    mach = sysinfo.get("machine") or ""
    return f"{plat} {mach}".strip()


def pick_core_mem(sysinfo: dict) -> Tuple[str, str]:
    # Core count
    cores = sysinfo.get("cpu_physical") or sysinfo.get("details", {}).get("hw.physicalcpu")
    cores_str = str(cores) if cores else "?"
    # Memory
    details = sysinfo.get("details", {})
    mem = None
    if "hw.memsize" in details:  # macOS bytes
        try:
            mem_gb = int(details["hw.memsize"]) / (1024**3)
            mem = f"{mem_gb:.1f} GB"
        except Exception:
            pass
    if not mem and "mem_MemTotal" in details:  # Linux kB
        import re
        m = details["mem_MemTotal"]
        ks = None
        try:
            ks = int(m.split()[0])
        except Exception:
            try:
                ks = int(re.sub(r"[^0-9]", "", m))
            except Exception:
                ks = None
        if ks:
            mem = f"{ks/1024/1024:.1f} GB"
    return cores_str, mem or "?"


def collect_hosts(roots: List[Path]) -> List[Path]:
    host_dirs: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.iterdir():
            if p.is_dir() and (p / "results_timing_ci.csv").exists() and (p / "sysinfo.json").exists():
                host_dirs.append(p)
    return sorted(host_dirs)


def merge(roots: List[Path], out_csv: Path, out_md: Path) -> None:
    host_dirs = collect_hosts(roots)
    if not host_dirs:
        raise SystemExit("No host result folders found under: " + ", ".join(map(str, roots)))

    # Flatten rows per host
    merged_rows: List[dict] = []
    host_meta: Dict[str, dict] = {}

    for hd in host_dirs:
        sysinfo = read_json(hd / "sysinfo.json")
        cpu = pick_cpu_name(sysinfo)
        cores, mem = pick_core_mem(sysinfo)
        host_meta[hd.name] = {"cpu": cpu, "cores": cores, "mem": mem}

        rows = read_csv_rows(hd / "results_timing_ci.csv")
        for r in rows:
            try:
                k = int(r["k"]) if "k" in r else None
                if k is None:
                    continue
                merged_rows.append({
                    "host": hd.name,
                    "cpu": cpu,
                    "cores": cores,
                    "mem": mem,
                    "k": k,
                    "paired_us_mean": float(r.get("paired_us_mean", "nan")),
                    "paired_us_ci95": float(r.get("paired_us_ci95", 0.0)),
                    "index_us_mean": float(r.get("index_us_mean", "nan")),
                    "index_us_ci95": float(r.get("index_us_ci95", 0.0)),
                    "mask_us_mean": float(r.get("mask_us_mean", "nan")),
                    "mask_us_ci95": float(r.get("mask_us_ci95", 0.0)),
                    "speedup_idx_mean": float(r.get("speedup_idx_mean", "nan")),
                    "speedup_mask_mean": float(r.get("speedup_mask_mean", "nan")),
                })
            except Exception:
                continue

    # Write combined CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "host", "cpu", "cores", "mem", "k",
            "paired_us_mean", "paired_us_ci95",
            "index_us_mean", "index_us_ci95",
            "mask_us_mean", "mask_us_ci95",
            "speedup_idx_mean", "speedup_mask_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(merged_rows, key=lambda r: (r["k"], r["host"])) )

    # Make a concise markdown summary with per-k tables and overall averages
    ks = sorted({r["k"] for r in merged_rows})
    hosts = sorted({r["host"] for r in merged_rows})

    def host_row_for_k(host: str, k: int):
        for r in merged_rows:
            if r["host"] == host and r["k"] == k:
                return r
        return None

    lines: List[str] = []
    lines.append("# Cross-Machine Comparison\n")
    lines.append("## Hosts\n")
    for h in hosts:
        meta = host_meta[h]
        lines.append(f"- {h}: {meta['cpu']} | cores={meta['cores']} | mem={meta['mem']}")
    lines.append("")

    for k in ks:
        lines.append(f"## k={k}\n")
        # Table header
        lines.append("host | cpu | speedup_idx_mean | speedup_mask_mean | paired_us_mean | index_us_mean | mask_us_mean")
        lines.append("--- | --- | ---:| ---:| ---:| ---:| ---:")
        sp_list = []
        for h in hosts:
            r = host_row_for_k(h, k)
            if not r:
                continue
            sp_list.append(r["speedup_idx_mean"]) if r["speedup_idx_mean"] == r["speedup_idx_mean"] else None
            cpu = host_meta[h]["cpu"]
            lines.append(
                f"{h} | {cpu} | {r['speedup_idx_mean']:.2f} | {r['speedup_mask_mean']:.2f} | "
                f"{r['paired_us_mean']:.1f} | {r['index_us_mean']:.1f} | {r['mask_us_mean']:.1f}"
            )
        if sp_list:
            mu = mean(sp_list)
            sd = pstdev(sp_list) if len(sp_list) > 1 else 0.0
            lines.append(f"\n- Average speedup vs index_select across hosts: {mu:.2f}x (sd {sd:.2f})\n")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_csv} and {out_md}")


def main():
    ap = argparse.ArgumentParser(description="Merge multi-CPU host results into a single report")
    ap.add_argument("--roots", type=str, default="benchmarks/results",
                    help="Comma-separated directories containing host result folders")
    ap.add_argument("--out_csv", type=str, default="benchmarks/merged_results_timing_ci.csv")
    ap.add_argument("--out_md", type=str, default="benchmarks/merged_report.md")
    args = ap.parse_args()

    roots = [Path(p.strip()) for p in args.roots.split(",") if p.strip()]
    merge(roots, Path(args.out_csv), Path(args.out_md))


if __name__ == "__main__":
    main()

