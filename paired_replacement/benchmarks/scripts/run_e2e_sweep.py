#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import subprocess


def run(cmd):
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Run end-to-end MoE-like sweeps and save CSV")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m_list", type=str, default="512,1024,2048")
    ap.add_argument("--batch_list", type=str, default="8,16,32")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outfile", type=str, default="benchmarks/data/results_e2e_torch.csv")
    ap.add_argument("--reset", action="store_true", help="Overwrite outfile and write a fresh header")
    ap.add_argument("--hybrid_tau", type=float, default=0.5)
    ap.add_argument("--sticky_beta", type=float, default=0.8)
    args = ap.parse_args()

    outp = Path(args.outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # Optionally start fresh
    if args.reset and outp.exists():
        outp.unlink()
    # Seed CSV header if not present
    if not outp.exists():
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "N","hidden_dim","up_cols","gate_cols","m","batch","steps",
                    "paired_ms_per_step","rebuild_ms_per_step","speedup","hybrid_ms_per_step","hybrid_speedup","avg_k","device","sticky_beta","hybrid_tau",
                ],
            )
            w.writeheader()

    ms = [int(x) for x in args.m_list.split(",") if x]
    batches = [int(x) for x in args.batch_list.split(",") if x]

    for m in ms:
        for b in batches:
            cmd = [
                "python3", "paired_replacement/benchmarks/scripts/run_e2e_torch.py",
                "--N", str(args.N), "--hidden_dim", str(args.hidden_dim), "--up_cols", str(args.up_cols),
                "--gate_cols", str(args.gate_cols), "--m", str(m), "--batch", str(b), "--steps", str(args.steps),
                "--device", args.device, "--sticky_beta", str(args.sticky_beta), "--hybrid_tau", str(args.hybrid_tau), "--outfile", str(outp),
            ]
            rc = run(cmd)
            if rc != 0:
                print(f"Warning: command failed with code {rc}: {' '.join(cmd)}")

    print(f"E2E sweep completed -> {outp}")


if __name__ == "__main__":
    main()
