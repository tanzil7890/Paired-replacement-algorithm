#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=cwd)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Run a standardized suite on this CPU and package outputs")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--ks", type=str, default="16,64,256,1024")
    ap.add_argument("--steps_per_k", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--perf", action="store_true", help="Include Linux perf run if available")
    ap.add_argument("--bw_preset", type=str, default="intel")
    ap.add_argument("--outdir", type=str, default="benchmarks/results")
    # E2E sweep knobs
    ap.add_argument("--e2e_m_list", type=str, default="512,1024,2048")
    ap.add_argument("--e2e_batch_list", type=str, default="8,16,32")
    ap.add_argument("--e2e_steps", type=int, default=50)
    args = ap.parse_args()

    # Destination directory tagged by host and timestamp
    host = socket.gethostname()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    root = Path(args.outdir) / f"{host}_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    # 1) sysinfo
    sysinfo_path = root / "sysinfo.json"
    code = run(["python3", "benchmarks/scripts/sysinfo.py"]) 
    if (Path("benchmarks/configs/sysinfo.json").exists()):
        shutil.copy2("benchmarks/configs/sysinfo.json", sysinfo_path)

    # 2) autotune
    at_path = root / "autotune_result.json"
    run(["python3", "benchmarks/scripts/run_autotune.py",
         "--N", str(args.N), "--hidden_dim", str(args.hidden_dim), "--up_cols", str(args.up_cols),
         "--gate_cols", str(args.gate_cols), "--m", str(args.m), "--steps", str(args.steps_per_k),
         "--outfile", str(at_path)])

    # 3) torch microbench + overlays
    torch_csv = root / "results_microbench_torch.csv"
    run(["python3", "benchmarks/scripts/run_microbench_torch.py",
         "--N", str(args.N), "--hidden_dim", str(args.hidden_dim), "--up_cols", str(args.up_cols),
         "--gate_cols", str(args.gate_cols), "--m", str(args.m),
         "--ks", args.ks, "--steps_per_k", str(args.steps_per_k), "--device", args.device,
         "--outfile", str(torch_csv)])
    overlay_png = root / "microbench_speedup_overlays.png"
    run(["python3", "benchmarks/scripts/plot_microbench_overlays.py",
         "--torch_csv", str(torch_csv), "--out", str(overlay_png)])

    # 4) timing CI
    ci_csv = root / "results_timing_ci.csv"
    run(["python3", "benchmarks/scripts/run_timing_ci.py",
         "--N", str(args.N), "--hidden_dim", str(args.hidden_dim), "--up_cols", str(args.up_cols),
         "--gate_cols", str(args.gate_cols), "--m", str(args.m),
         "--ks", args.ks, "--steps_per_k", str(args.steps_per_k),
         "--repeats", str(args.repeats), "--device", args.device,
         "--outfile", str(ci_csv)])

    # 5) E2E sweep + plot
    e2e_csv = root / "results_e2e_torch.csv"
    e2e_png = root / "e2e_speedup.png"
    run(["python3", "benchmarks/scripts/run_e2e_sweep.py",
         "--N", str(args.N), "--hidden_dim", str(args.hidden_dim), "--up_cols", str(args.up_cols),
         "--gate_cols", str(args.gate_cols), "--m_list", args.e2e_m_list, "--batch_list", args.e2e_batch_list,
         "--steps", str(args.e2e_steps), "--device", args.device, "--outfile", str(e2e_csv)])
    if e2e_csv.exists():
        run(["python3", "benchmarks/scripts/plot_e2e.py", "--csv", str(e2e_csv), "--out", str(e2e_png)])

    # 6) perf (Linux only)
    perf_csv = None
    perf_png = None
    if args.perf and platform.system().lower() == "linux":
        perf_csv = root / "results_perf_torch.csv"
        run(["python3", "benchmarks/scripts/run_perf_torch.py",
             "--N", str(args.N), "--hidden_dim", str(args.hidden_dim), "--up_cols", str(args.up_cols),
             "--gate_cols", str(args.gate_cols), "--m", str(args.m),
             "--ks", args.ks, "--steps_per_k", str(args.steps_per_k), "--device", args.device,
             "--bw_preset", args.bw_preset, "--outfile", str(perf_csv)])
        perf_png = root / "perf_plots.png"
        if perf_csv.exists():
            run(["python3", "benchmarks/scripts/plot_perf.py", "--perf_csv", str(perf_csv), "--out", str(perf_png)])

    # 7) report
    report_md = root / "report.md"
    cmd = ["python3", "benchmarks/scripts/generate_report.py",
           "--torch_csv", str(torch_csv), "--timing_ci_csv", str(ci_csv),
           "--overlay_png", str(overlay_png), "--e2e_csv", str(e2e_csv), "--e2e_png", str(e2e_png),
           "--out_md", str(report_md)]
    if perf_csv and Path(perf_csv).exists():
        cmd += ["--perf_csv", str(perf_csv)]
        if perf_png and Path(perf_png).exists():
            cmd += ["--perf_png", str(perf_png)]
    run(cmd)

    print(f"Multi-CPU suite completed. Results at {root}")


if __name__ == "__main__":
    main()
