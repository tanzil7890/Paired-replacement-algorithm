#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def have_xctrace() -> bool:
    return shutil.which("xcrun") is not None


def list_templates() -> list[str]:
    try:
        out = subprocess.check_output(["xcrun", "xctrace", "list", "templates"], text=True)
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # Lines look like: "Time Profiler" (macOS)
        if line.startswith("\"") and line.endswith("\""):
            names.append(line.strip('\"'))
    return names


def run_xctrace(template: str, out_trace: Path, cmd: list[str]) -> int:
    out_trace.parent.mkdir(parents=True, exist_ok=True)
    if out_trace.exists():
        try:
            out_trace.unlink()
        except Exception:
            pass
    full_cmd = [
        "xcrun",
        "xctrace",
        "record",
        "--template",
        template,
        "--output",
        str(out_trace),
        "--",
    ] + cmd
    print("Running:", " ".join(full_cmd))
    return subprocess.call(full_cmd)


def export_trace(in_trace: Path, out_dir: Path, fmt: str = "csv") -> bool:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            str(in_trace),
            "--output",
            str(out_dir),
            "--format",
            fmt,
        ]
        print("Exporting:", " ".join(cmd))
        subprocess.check_call(cmd)
        return True
    except Exception as e:
        print("Export failed:", e)
        return False


def main():
    ap = argparse.ArgumentParser(description="Run xctrace around paired vs rebuild loops (macOS)")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--steps_per_k", type=int, default=50)
    ap.add_argument("--outdir", type=str, default="benchmarks/xctrace")
    args = ap.parse_args()

    if not have_xctrace():
        print("xcrun/xctrace not found. Install Xcode Command Line Tools and try again.")
        sys.exit(1)

    templates = list_templates()
    # Prefer Counters if available; else fall back to Time Profiler
    template = "Counters" if "Counters" in templates else "Time Profiler"
    print(f"Using template: {template}")

    script = Path(__file__).parent / "run_microbench_torch.py"

    # Build base command
    base = [
        sys.executable,
        str(script),
        "--N", str(args.N),
        "--hidden_dim", str(args.hidden_dim),
        "--up_cols", str(args.up_cols),
        "--gate_cols", str(args.gate_cols),
        "--m", str(args.m),
        "--steps_per_k", str(args.steps_per_k),
        "--device", "cpu",
    ]

    outdir = Path(args.outdir)
    # Paired
    trace1 = outdir / f"paired_k{args.k}.trace"
    cmd1 = base + ["--mode", "paired", "--k", str(args.k)]
    code = run_xctrace(template, trace1, cmd1)
    if code != 0:
        print(f"xctrace for paired failed with exit code {code}")
    else:
        export_trace(trace1, outdir / f"paired_k{args.k}_export", fmt="csv")

    # Rebuild
    trace2 = outdir / f"rebuild_k{args.k}.trace"
    cmd2 = base + ["--mode", "rebuild", "--k", str(args.k)]
    code = run_xctrace(template, trace2, cmd2)
    if code != 0:
        print(f"xctrace for rebuild failed with exit code {code}")
    else:
        export_trace(trace2, outdir / f"rebuild_k{args.k}_export", fmt="csv")

    print("Done. Open the .trace files in Instruments to inspect counters and timelines.")


if __name__ == "__main__":
    main()

