#!/usr/bin/env python3
import argparse
import csv
import time
import math
import numpy as np
from typing import Optional

from paired_replacement.src.python.paired_cache import PairedCache, baseline_full_rebuild


def make_random_pools(N: int, up_cols: int, hidden_dim: int, gate_cols: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    up_pool = rng.standard_normal((N, up_cols), dtype=np.float32)
    down_pool = rng.standard_normal((hidden_dim, N), dtype=np.float32)
    gate_pool = (
        rng.standard_normal((N, gate_cols), dtype=np.float32) if gate_cols > 0 else None
    )
    return up_pool, down_pool, gate_pool


def rand_init_mask(N: int, m: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(N, dtype=bool)
    idx = rng.choice(N, size=m, replace=False)
    mask[idx] = True
    return mask


def perturb_mask(mask: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Flip k/2 actives to inactive and k/2 inactives to active to keep cardinality.
    If k is odd, favor additions by one.
    """
    N = mask.shape[0]
    m = int(mask.sum())
    r = k // 2
    a = k - r
    active_idx = np.flatnonzero(mask)
    inactive_idx = np.flatnonzero(~mask)
    r = min(r, active_idx.shape[0])
    a = min(a, inactive_idx.shape[0])
    remove = rng.choice(active_idx, size=r, replace=False) if r > 0 else np.array([], dtype=int)
    add = rng.choice(inactive_idx, size=a, replace=False) if a > 0 else np.array([], dtype=int)
    new_mask = mask.copy()
    new_mask[remove] = False
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
    seed: int,
    warmup: int = 3,
    outfile: Optional[str] = None,
):
    rng = np.random.default_rng(seed)
    up_pool, down_pool, gate_pool = make_random_pools(N, up_cols, hidden_dim, gate_cols, seed)

    mask = rand_init_mask(N, m, rng)

    cache = PairedCache(up_pool, down_pool, gate_pool, init_mask=mask)

    rowsize_bytes = (up_cols + (gate_cols if gate_pool is not None else 0) + hidden_dim) * 4

    results = []

    for k in ks:
        # Warmup perturbations
        tmp_mask = mask
        for _ in range(warmup):
            tmp_mask = perturb_mask(tmp_mask, k, rng)
            cache.update(tmp_mask)

        # Measure paired cache
        mask_p = mask
        t_start = time.perf_counter()
        for _ in range(steps_per_k):
            mask_p = perturb_mask(mask_p, k, rng)
            cache.update(mask_p)
        t_end = time.perf_counter()
        paired_time = (t_end - t_start) / steps_per_k

        # Measure baseline full rebuild
        mask_b = mask
        t_start = time.perf_counter()
        for _ in range(steps_per_k):
            mask_b = perturb_mask(mask_b, k, rng)
            _ = baseline_full_rebuild(up_pool, down_pool, gate_pool, mask_b)
        t_end = time.perf_counter()
        rebuild_time = (t_end - t_start) / steps_per_k

        # Theoretical minimal bytes moved per step = max(A,R)*row_bytes; here we keep |S| constant, so A=R=k/2 (ceil/floor)
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
            }
        )

        # Next round starts from last mask
        mask = mask_p

    if outfile:
        with open(outfile, "w", newline="") as f:
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outfile", type=str, default="benchmarks/results_microbench.csv")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x]
    results = run_microbench(
        args.N,
        args.up_cols,
        args.hidden_dim,
        args.gate_cols,
        args.m,
        ks,
        args.steps_per_k,
        args.seed,
        outfile=args.outfile,
    )

    # Print compact summary
    for r in results:
        print(
            f"k={r['k']:>5}  paired={r['paired_us']:.1f}us  rebuild={r['rebuild_us']:.1f}us  "
            f"speedup={r['speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
