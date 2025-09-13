#!/usr/bin/env python3
import argparse
import time
import numpy as np
from typing import Optional

from paired_replacement.src.python.paired_cache import PairedCache, baseline_full_rebuild


def make_random_pools(N: int, up_cols: int, hidden_dim: int, gate_cols: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    up_pool = rng.standard_normal((N, up_cols), dtype=np.float32)
    down_pool = rng.standard_normal((hidden_dim, N), dtype=np.float32)
    gate_scores_W = rng.standard_normal((N, hidden_dim), dtype=np.float32)  # for gating logits
    gate_pool = (
        rng.standard_normal((N, gate_cols), dtype=np.float32) if gate_cols > 0 else None
    )
    return up_pool, down_pool, gate_pool, gate_scores_W


def topk_mask_from_logits(logits: np.ndarray, k: int) -> np.ndarray:
    # logits: [batch, N]
    # Reduce across batch (max pooling) to choose a shared active set
    pooled = logits.max(axis=0)  # [N]
    idx = np.argpartition(pooled, -k)[-k:]
    mask = np.zeros_like(pooled, dtype=bool)
    mask[idx] = True
    return mask


def run_e2e(
    N: int,
    up_cols: int,
    hidden_dim: int,
    gate_cols: int,
    m: int,
    batch: int,
    steps: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    up_pool, down_pool, gate_pool, gate_scores_W = make_random_pools(
        N, up_cols, hidden_dim, gate_cols, seed
    )

    # Initial input and mask
    X = rng.standard_normal((batch, hidden_dim), dtype=np.float32)
    logits = X @ gate_scores_W.T  # [batch, N]
    mask0 = topk_mask_from_logits(logits, m)

    # Paired cache path
    cache = PairedCache(up_pool, down_pool, gate_pool, init_mask=mask0)
    t_paired_total = 0.0
    for _ in range(steps):
        # New random input each step to shift active set
        X = rng.standard_normal((batch, hidden_dim), dtype=np.float32)
        logits = X @ gate_scores_W.T
        mask = topk_mask_from_logits(logits, m)
        t0 = time.perf_counter()
        cache.update(mask)
        # Forward: Y = (X @ Down_active^T) @ Up_active
        down = cache.get_active_down_weight()  # [hidden, m]
        up = cache.get_concat_weight()  # [m, up_cols + gate_cols]
        Y = (X @ down) @ up  # [batch, up_cols + gate_cols]
        t_paired_total += time.perf_counter() - t0

    # Baseline rebuild path
    t_rebuild_total = 0.0
    mask = mask0
    for _ in range(steps):
        X = rng.standard_normal((batch, hidden_dim), dtype=np.float32)
        logits = X @ gate_scores_W.T
        mask = topk_mask_from_logits(logits, m)
        t0 = time.perf_counter()
        concat, down = baseline_full_rebuild(up_pool, down_pool, gate_pool, mask)
        Yb = (X @ down) @ concat
        t_rebuild_total += time.perf_counter() - t0

    return {
        "paired_ms_per_step": (t_paired_total / steps) * 1e3,
        "rebuild_ms_per_step": (t_rebuild_total / steps) * 1e3,
        "speedup": t_rebuild_total / t_paired_total if t_paired_total > 0 else float("inf"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = run_e2e(
        args.N,
        args.up_cols,
        args.hidden_dim,
        args.gate_cols,
        args.m,
        args.batch,
        args.steps,
        args.seed,
    )

    print(
        f"paired={res['paired_ms_per_step']:.3f} ms/step  "
        f"rebuild={res['rebuild_ms_per_step']:.3f} ms/step  "
        f"speedup={res['speedup']:.2f}x"
    )


if __name__ == "__main__":
    main()

