#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List


interesting = {
    "cpu_cycles": [r"cpu\s*cycles"],
    "instructions": [r"instructions?(\s*retired)?"],
    "l1d_misses": [r"l1d.*miss"],
    "l2_misses": [r"l2.*miss"],
    "l3_misses": [r"l3.*miss", r"llc.*miss"],
    "branch_misses": [r"branch.*miss"],
    "cache_misses": [r"cache.*miss"],
    "cache_refs": [r"cache.*ref"],
}


def normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def match_key(col: str) -> str | None:
    coln = normalize(col)
    for key, pats in interesting.items():
        for p in pats:
            if re.search(p, coln):
                return key
    return None


def parse_csv_file(path: Path) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    try:
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                for col, val in row.items():
                    if val is None or val == "":
                        continue
                    key = match_key(col)
                    if key is None:
                        continue
                    try:
                        x = float(val)
                    except Exception:
                        continue
                    agg[key] = agg.get(key, 0.0) + x
    except Exception:
        pass
    return agg


def parse_export_dir(export_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, float] = {}
    files = list(export_dir.rglob("*.csv"))
    for fp in files:
        part = parse_csv_file(fp)
        for k, v in part.items():
            summary[k] = summary.get(k, 0.0) + v
    # Derived
    res: Dict[str, Any] = dict(summary)
    cycles = summary.get("cpu_cycles")
    instr = summary.get("instructions")
    if cycles and instr and instr > 0:
        res["cpi"] = cycles / instr
    return res


def main():
    ap = argparse.ArgumentParser(description="Summarize Instruments xctrace CSV export")
    ap.add_argument("--paired_export", type=str, required=True)
    ap.add_argument("--rebuild_export", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="benchmarks/instruments_summary.json")
    ap.add_argument("--out_md", type=str, default="benchmarks/instruments_summary.md")
    args = ap.parse_args()

    paired_dir = Path(args.paired_export)
    rebuild_dir = Path(args.rebuild_export)
    paired = parse_export_dir(paired_dir)
    rebuild = parse_export_dir(rebuild_dir)

    summary = {
        "paired": paired,
        "rebuild": rebuild,
        "diff": {k: (rebuild.get(k, 0) - paired.get(k, 0)) for k in set(paired) | set(rebuild)},
    }

    Path(Path(args.out_json).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown summary
    def fmt(v):
        return f"{v:.3g}" if isinstance(v, (int, float)) else str(v)
    lines: List[str] = []
    lines.append("# Instruments Summary\n")
    lines.append("## Paired\n")
    for k in sorted(paired.keys()):
        lines.append(f"- {k}: {fmt(paired[k])}")
    lines.append("\n## Rebuild\n")
    for k in sorted(rebuild.keys()):
        lines.append(f"- {k}: {fmt(rebuild[k])}")
    lines.append("\n## Rebuild - Paired (difference)\n")
    for k in sorted(summary["diff"].keys()):
        lines.append(f"- {k}: {fmt(summary['diff'][k])}")
    with open(args.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()

