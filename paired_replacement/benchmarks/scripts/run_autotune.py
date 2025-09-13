#!/usr/bin/env python3
import argparse
import json
import os
import platform
from pathlib import Path

import torch

from run_microbench_torch import autotune_parallel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--grain_candidates", type=str, default="32,64,128,256")
    ap.add_argument("--threshold_candidates", type=str, default="32,64,128,256")
    default_out = str((Path(__file__).parent.parent / "configs" / "autotune_result.json").resolve())
    ap.add_argument("--outfile", type=str, default=default_out)
    args = ap.parse_args()

    device = torch.device(args.device)
    grains = [int(x) for x in args.grain_candidates.split(",") if x]
    thrs = [int(x) for x in args.threshold_candidates.split(",") if x]

    g, th, score = autotune_parallel(
        args.N, args.up_cols, args.hidden_dim, args.gate_cols, args.m, device, steps=args.steps,
        grains=grains, thresholds=thrs
    )
    # Persist suggestion
    result = {
        "grain": g,
        "par_threshold": th,
        "avg_us_per_step": score * 1e6,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__ if torch else None,
        "N": args.N,
        "hidden_dim": args.hidden_dim,
        "up_cols": args.up_cols,
        "gate_cols": args.gate_cols,
        "m": args.m,
        "steps": args.steps,
    }
    Path(Path(args.outfile).parent).mkdir(parents=True, exist_ok=True)
    with open(args.outfile, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Autotune selected grain={g}, threshold={th} (avg {result['avg_us_per_step']:.1f} us/step); saved {args.outfile}")


if __name__ == "__main__":
    main()
