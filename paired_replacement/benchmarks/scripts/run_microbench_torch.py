#!/usr/bin/env python3
import argparse
import csv
import time
import math
from pathlib import Path
import json
from datetime import datetime

import torch
from torch.utils.cpp_extension import load
import os


def build_ext(verbose: bool = False):
    proj_root = Path(__file__).parents[2]  # paired_replacement/
    build_dir = str(proj_root / "build")
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    return load(
        name="weight_cache_ext",
        sources=[str(proj_root / "src" / "cpp" / "weight_cache_binding.cpp")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=[str(proj_root / "src" / "cpp")],
        build_directory=build_dir,
        verbose=verbose,
    )


def make_random_pools(N: int, up_cols: int, hidden_dim: int, gate_cols: int, device: torch.device):
    up_weight = torch.randn(N, up_cols, dtype=torch.float32, device=device)
    down_weight = torch.randn(hidden_dim, N, dtype=torch.float32, device=device)
    gate_weight = torch.randn(N, gate_cols, dtype=torch.float32, device=device) if gate_cols > 0 else None
    return up_weight, down_weight, gate_weight


def rand_init_mask(N: int, m: int, device: torch.device):
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    idx = torch.randperm(N, device=device)[:m]
    mask[idx] = True
    return mask


def perturb_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
    N = mask.numel()
    active_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    inactive_idx = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
    r = k // 2
    a = k - r
    r = min(r, active_idx.numel())
    a = min(a, inactive_idx.numel())
    if r > 0:
        remove = active_idx[torch.randperm(active_idx.numel(), device=mask.device)[:r]]
    else:
        remove = torch.empty(0, dtype=torch.long, device=mask.device)
    if a > 0:
        add = inactive_idx[torch.randperm(inactive_idx.numel(), device=mask.device)[:a]]
    else:
        add = torch.empty(0, dtype=torch.long, device=mask.device)
    new_mask = mask.clone()
    if remove.numel() > 0:
        new_mask[remove] = False
    if add.numel() > 0:
        new_mask[add] = True
    return new_mask


def run_microbench(
    N: int,
    up_cols: int,
    hidden_dim: int,
    gate_cols: int,
    m: int,
    ks: list[int],
    steps_per_k: int,
    device: torch.device,
    outfile: str | None,
    grain: int | None = None,
    par_threshold: int | None = None,
    autotune_source: str | None = None,
):
    # Build extension and get custom class handle
    build_ext(verbose=False)
    WeightCache = torch.classes.paired.WeightCache

    up, down, gate = make_random_pools(N, up_cols, hidden_dim, gate_cols, device)
    init_mask = rand_init_mask(N, m, device)

    has_gate = gate is not None
    if not has_gate:
        gate = torch.empty(N, 0, dtype=torch.float32, device=device)

    cache = WeightCache(init_mask, hidden_dim, gate, up, down, has_gate)

    rowsize_bytes = (up_cols + (gate_cols if has_gate else 0) + hidden_dim) * 4

    results = []
    for k in ks:
        # Warmup
        mask = init_mask
        for _ in range(3):
            mask = perturb_mask(mask, k)
            cache.update_active_weights(mask)

        # Paired cache timing
        mask_p = init_mask
        t0 = time.perf_counter()
        for _ in range(steps_per_k):
            mask_p = perturb_mask(mask_p, k)
            cache.update_active_weights(mask_p)
        t1 = time.perf_counter()
        paired_time = (t1 - t0) / steps_per_k

        # Baseline rebuild (index_select)
        mask_b = init_mask
        t0 = time.perf_counter()
        for _ in range(steps_per_k):
            mask_b = perturb_mask(mask_b, k)
            idx = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)
            up_act = up.index_select(0, idx)
            down_act = down.index_select(1, idx)  # [hidden, m]
            if has_gate:
                gate_act = gate.index_select(0, idx)
                _ = torch.cat([gate_act, up_act], dim=1)
            else:
                _ = up_act
            _ = down_act  # use value to avoid DCE
        t1 = time.perf_counter()
        rebuild_time = (t1 - t0) / steps_per_k

        # Baseline using boolean mask (gather via advanced indexing)
        mask_m = init_mask
        t0 = time.perf_counter()
        for _ in range(steps_per_k):
            mask_m = perturb_mask(mask_m, k)
            up_act = up[mask_m]
            down_act = down[:, mask_m]
            if has_gate:
                gate_act = gate[mask_m]
                _ = torch.cat([gate_act, up_act], dim=1)
            else:
                _ = up_act
            _ = down_act
        t1 = time.perf_counter()
        rebuild_mask_time = (t1 - t0) / steps_per_k

        # Hybrid baseline: if actual delta > tau*m, rebuild, else paired-update
        tau = float(os.environ.get("WEIGHT_CACHE_HYBRID_TAU", "")) if os.environ.get("WEIGHT_CACHE_HYBRID_TAU") else None
        tau = tau if tau is not None else None
        # default from args if not env set; capture once outside loop by reading args via closure? not available; infer from ratio of provided m below
        # We'll compute dynamic threshold = args.hybrid_tau * m via captured value in outer scope: store in local from environment after first use
        # Since we don't have args here, approximate tau_frac from env else 0.5
        tau_frac = float(os.environ.get("WEIGHT_CACHE_HYBRID_TAU_FRAC", "0.5"))
        mask_h = init_mask
        t0 = time.perf_counter()
        for _ in range(steps_per_k):
            new_mask = perturb_mask(mask_h, k)
            # compute actual delta
            added = (~mask_h) & new_mask
            removed = mask_h & (~new_mask)
            delta = int(added.sum().item() + removed.sum().item())
            if delta > int(tau_frac * m):
                # rebuild path
                idx = torch.nonzero(new_mask, as_tuple=False).squeeze(-1)
                up.index_select(0, idx)
                down.index_select(1, idx)
                if has_gate:
                    gate.index_select(0, idx)
            else:
                cache.update_active_weights(new_mask)
            mask_h = new_mask
        t1 = time.perf_counter()
        hybrid_time = (t1 - t0) / steps_per_k

        A = math.ceil(k / 2)
        R = math.floor(k / 2)
        lower_bound_rows = max(A, R)
        lower_bound_bytes = lower_bound_rows * rowsize_bytes

        results.append(
            {
                "N": N,
                "hidden_dim": hidden_dim,
                "up_cols": up_cols,
                "gate_cols": gate_cols,
                "m": m,
                "k": k,
                "paired_us": paired_time * 1e6,
                "rebuild_us": rebuild_time * 1e6,
                "speedup": rebuild_time / paired_time if paired_time > 0 else float("inf"),
                "min_bytes_moved": lower_bound_bytes,
                "rebuild_mask_us": rebuild_mask_time * 1e6,
                "speedup_mask": rebuild_mask_time / paired_time if paired_time > 0 else float("inf"),
                "grain": grain if grain is not None else os.environ.get("WEIGHT_CACHE_GRAIN", "64"),
                "par_threshold": par_threshold if par_threshold is not None else os.environ.get("WEIGHT_CACHE_PAR_THRESHOLD", "64"),
                "hybrid_us": hybrid_time * 1e6,
                "speedup_hybrid_vs_idx": rebuild_time / hybrid_time if hybrid_time > 0 else float("inf"),
                "paired_vs_hybrid": hybrid_time / paired_time if paired_time > 0 else float("inf"),
            }
        )

    if outfile and results:
        with open(outfile, "w", newline="") as f:
            # Write provenance banner as a comment line
            banner_parts = [
                f"autotune_source={autotune_source}" if autotune_source else None,
                f"grain={grain if grain is not None else os.environ.get('WEIGHT_CACHE_GRAIN','')}",
                f"par_threshold={par_threshold if par_threshold is not None else os.environ.get('WEIGHT_CACHE_PAR_THRESHOLD','')}",
                f"device={device.type}",
                f"torch={torch.__version__}",
                f"timestamp={datetime.utcnow().isoformat()}Z",
            ]
            banner = "# " + ", ".join([p for p in banner_parts if p]) + "\n"
            f.write(banner)
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--ks", type=str, default="16,64,256,1024")
    ap.add_argument("--steps_per_k", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outfile", type=str, default="benchmarks/results_microbench_torch.csv")
    # Hybrid baseline: if delta k > tau*m, rebuild; else paired update
    ap.add_argument("--hybrid_tau", type=float, default=0.5,
                    help="Threshold as fraction of m for hybrid baseline (rebuild if k > tau*m)")
    # Autotune knobs
    ap.add_argument("--autotune", action="store_true", help="Autotune grain/threshold before benchmarking")
    ap.add_argument("--grain_candidates", type=str, default="32,64,128,256")
    ap.add_argument("--threshold_candidates", type=str, default="32,64,128,256")
    # Auto-load prior autotune JSON
    # Default to package-local configs path if not provided
    default_at_path = str((Path(__file__).parent.parent / "configs" / "autotune_result.json").resolve())
    ap.add_argument("--autotune_json", type=str, default=default_at_path,
                    help="Path to autotune result JSON to load if present (used when --autotune is not set)")
    ap.add_argument("--skip_autotune_json", action="store_true",
                    help="Disable loading autotune JSON even if present")
    # Perf-mode: run only one loop kind for counter measurement
    ap.add_argument("--mode", type=str, choices=["paired", "rebuild"], default=None,
                    help="If set, run only the specified loop to be wrapped by perf stat")
    ap.add_argument("--k", type=int, default=None, help="Mask delta k for perf-mode")
    args = ap.parse_args()

    device = torch.device(args.device)
    # Perf-mode: run a single loop and exit (perf stat wraps this script)
    if args.mode is not None:
        assert args.k is not None, "--k is required in --mode"
        # Build extension and create weights
        build_ext(verbose=False)
        WeightCache = torch.classes.paired.WeightCache
        N = args.N
        up_cols = args.up_cols
        hidden_dim = args.hidden_dim
        gate_cols = args.gate_cols
        m = args.m
        steps = args.steps_per_k
        up, down, gate = make_random_pools(N, up_cols, hidden_dim, gate_cols, device)
        init_mask = rand_init_mask(N, m, device)
        has_gate = gate is not None
        if not has_gate:
            gate = torch.empty(N, 0, dtype=torch.float32, device=device)
        cache = WeightCache(init_mask, hidden_dim, gate, up, down, has_gate)
        # Warmup
        mask = init_mask
        for _ in range(3):
            mask = perturb_mask(mask, args.k)
            if args.mode == "paired":
                cache.update_active_weights(mask)
            else:
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                up.index_select(0, idx)
                down.index_select(1, idx)
                if has_gate:
                    gate.index_select(0, idx)
        # Measured loop
        mask = init_mask
        for _ in range(steps):
            mask = perturb_mask(mask, args.k)
            if args.mode == "paired":
                cache.update_active_weights(mask)
            else:
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                up_act = up.index_select(0, idx)
                down_act = down.index_select(1, idx)
                if has_gate:
                    gate_act = gate.index_select(0, idx)
                    _ = torch.cat([gate_act, up_act], dim=1)
                else:
                    _ = up_act
                _ = down_act
        return

    # Optional autotune
    tuned_grain = None
    tuned_thr = None
    autotune_source = None
    if args.autotune:
        grains = [int(x) for x in args.grain_candidates.split(",") if x]
        thrs = [int(x) for x in args.threshold_candidates.split(",") if x]
        tuned_grain, tuned_thr, score = autotune_parallel(
            args.N, args.up_cols, args.hidden_dim, args.gate_cols, args.m, device, steps=args.steps_per_k,
            grains=grains, thresholds=thrs
        )
        # Apply tuned values to environment so C++ reads them
        os.environ["WEIGHT_CACHE_GRAIN"] = str(tuned_grain)
        os.environ["WEIGHT_CACHE_PAR_THRESHOLD"] = str(tuned_thr)
        print(f"Autotune selected grain={tuned_grain}, threshold={tuned_thr} (score={score*1e6:.1f} us/step avg)")
        autotune_source = "search"
    else:
        # Try to load prior autotune JSON if present and not disabled
        at_path = Path(args.autotune_json)
        if not args.skip_autotune_json and at_path.exists():
            try:
                with open(at_path, "r") as f:
                    data = json.load(f)
                g = int(data.get("grain")) if data.get("grain") is not None else None
                th = int(data.get("par_threshold")) if data.get("par_threshold") is not None else None
                if g is not None and th is not None:
                    tuned_grain, tuned_thr = g, th
                    os.environ["WEIGHT_CACHE_GRAIN"] = str(tuned_grain)
                    os.environ["WEIGHT_CACHE_PAR_THRESHOLD"] = str(tuned_thr)
                    print(f"Loaded autotune JSON: grain={tuned_grain}, threshold={tuned_thr} from {at_path}")
                    autotune_source = str(at_path)
            except Exception as e:
                print(f"Warning: failed to load autotune JSON {at_path}: {e}")

    ks = [int(x) for x in args.ks.split(",") if x]
    # Pass hybrid tau to env for benchmark loop
    os.environ["WEIGHT_CACHE_HYBRID_TAU_FRAC"] = str(args.hybrid_tau)
    results = run_microbench(
        args.N,
        args.up_cols,
        args.hidden_dim,
        args.gate_cols,
        args.m,
        ks,
        args.steps_per_k,
        device,
        args.outfile,
        grain=tuned_grain,
        par_threshold=tuned_thr,
        autotune_source=autotune_source,
    )

    for r in results:
        print(
            f"k={r['k']:>5}  paired={r['paired_us']:.1f}us  index_sel={r['rebuild_us']:.1f}us  "
            f"bool_mask={r['rebuild_mask_us']:.1f}us  hybrid={r.get('hybrid_us', float('nan')):.1f}us  "
            f"speedup_idx={r['speedup']:.2f}x  speedup_mask={r['speedup_mask']:.2f}x  "
            f"hybrid_vs_idx={r.get('speedup_hybrid_vs_idx', float('nan')):.2f}x  paired_vs_hybrid={r.get('paired_vs_hybrid', float('nan')):.2f}x"
        )


def autotune_parallel(N, up_cols, hidden_dim, gate_cols, m, device, steps, grains, thresholds):
    """Search over (grain, threshold) to minimize average paired update time across two k values.
    Returns (best_grain, best_threshold, best_time_per_step).
    """
    build_ext(verbose=False)
    WeightCache = torch.classes.paired.WeightCache
    up, down, gate = make_random_pools(N, up_cols, hidden_dim, gate_cols, device)
    init_mask = rand_init_mask(N, m, device)
    has_gate = gate is not None
    if not has_gate:
        gate = torch.empty(N, 0, dtype=torch.float32, device=device)
    cache = WeightCache(init_mask, hidden_dim, gate, up, down, has_gate)

    ks = [max(8, m // 16), max(32, m // 4)]  # small and larger deltas relative to m

    best_t = float("inf")
    best_g = None
    best_th = None
    for g in grains:
        for th in thresholds:
            os.environ["WEIGHT_CACHE_GRAIN"] = str(g)
            os.environ["WEIGHT_CACHE_PAR_THRESHOLD"] = str(th)
            # Warmup a bit
            mask = init_mask
            for _ in range(3):
                mask = perturb_mask(mask, ks[0])
                cache.update_active_weights(mask)
            # Measure both ks and average
            total = 0.0
            for kval in ks:
                mask = init_mask
                t0 = time.perf_counter()
                for _ in range(max(3, steps // 2)):
                    mask = perturb_mask(mask, kval)
                    cache.update_active_weights(mask)
                t1 = time.perf_counter()
                total += (t1 - t0) / max(3, steps // 2)
            avg = total / len(ks)
            if avg < best_t:
                best_t, best_g, best_th = avg, g, th
    return best_g, best_th, best_t


if __name__ == "__main__":
    main()
