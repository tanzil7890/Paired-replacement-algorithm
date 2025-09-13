#!/usr/bin/env python3
import argparse
import csv
import math
import time
from pathlib import Path
from typing import List, Tuple

import torch

from run_microbench_torch import build_ext


def parse_list_numbers(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # Support 1e6 style
        try:
            v = int(float(tok))
        except Exception:
            continue
        out.append(v)
    return out


def parse_list_floats(s: str) -> List[float]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except Exception:
            continue
        out.append(v)
    return out


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


def perturb_mask_random(mask: torch.Tensor, k: int) -> torch.Tensor:
    N = mask.numel()
    active_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    inactive_idx = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
    r = min(k // 2, active_idx.numel())
    a = min(k - r, inactive_idx.numel())
    remove = active_idx[torch.randperm(active_idx.numel(), device=mask.device)[:r]] if r > 0 else torch.empty(0, dtype=torch.long, device=mask.device)
    add = inactive_idx[torch.randperm(inactive_idx.numel(), device=mask.device)[:a]] if a > 0 else torch.empty(0, dtype=torch.long, device=mask.device)
    new_mask = mask.clone()
    if remove.numel() > 0:
        new_mask[remove] = False
    if add.numel() > 0:
        new_mask[add] = True
    return new_mask


def perturb_mask_block(mask: torch.Tensor, k: int) -> torch.Tensor:
    # Remove a contiguous block of size r from current actives; add a contiguous block of size a elsewhere
    N = mask.numel()
    actives = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    inactives = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
    if actives.numel() == 0 or inactives.numel() == 0:
        return perturb_mask_random(mask, k)
    r = min(k // 2, actives.numel())
    a = min(k - r, inactives.numel())
    rng = torch.randint(0, max(1, actives.numel()), (1,), device=mask.device).item()
    # Pick block starting index in actives (approximate contiguous by sorting indices)
    actives_sorted = torch.sort(actives).values
    start_r = min(rng, max(0, actives_sorted.numel() - r))
    rem_block = actives_sorted[start_r:start_r + r]
    inact_sorted = torch.sort(inactives).values
    rng2 = torch.randint(0, max(1, inact_sorted.numel()), (1,), device=mask.device).item()
    start_a = min(rng2, max(0, inact_sorted.numel() - a))
    add_block = inact_sorted[start_a:start_a + a]
    new_mask = mask.clone()
    if rem_block.numel() > 0:
        new_mask[rem_block] = False
    if add_block.numel() > 0:
        new_mask[add_block] = True
    return new_mask


def measure_once(N: int, up_cols: int, hidden_dim: int, gate_cols: int, m: int, k: int, pattern: str, device: torch.device):
    WeightCache = torch.classes.paired.WeightCache
    up, down, gate = make_random_pools(N, up_cols, hidden_dim, gate_cols, device)
    init_mask = rand_init_mask(N, m, device)
    has_gate = gate is not None
    if not has_gate:
        gate = torch.empty(N, 0, dtype=torch.float32, device=device)
    cache = WeightCache(init_mask, hidden_dim, gate, up, down, has_gate)
    # choose perturb function
    perturb = perturb_mask_random if pattern == "random" else perturb_mask_block
    # warmup
    mask = init_mask
    for _ in range(3):
        mask = perturb(mask, k)
        cache.update_active_weights(mask)
    # paired
    mask_p = init_mask
    t0 = time.perf_counter()
    for _ in range(5):
        mask_p = perturb(mask_p, k)
        cache.update_active_weights(mask_p)
    t1 = time.perf_counter()
    paired_time = (t1 - t0) / 5
    # index_select
    mask_b = init_mask
    t0 = time.perf_counter()
    for _ in range(5):
        mask_b = perturb(mask_b, k)
        idx = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)
        up.index_select(0, idx)
        down.index_select(1, idx)
        if has_gate:
            gate.index_select(0, idx)
    t1 = time.perf_counter()
    rebuild_time = (t1 - t0) / 5
    # boolean-mask
    mask_m = init_mask
    t0 = time.perf_counter()
    for _ in range(5):
        mask_m = perturb(mask_m, k)
        _ = up[mask_m]
        _ = down[:, mask_m]
        if has_gate:
            _ = gate[mask_m]
    t1 = time.perf_counter()
    mask_time = (t1 - t0) / 5

    return paired_time, rebuild_time, mask_time


def run_sweep(Ns: List[int], ratios: List[float], ks: List[int], patterns: List[str], repeats: int,
              up_cols: int, hidden_dim: int, gate_cols: int, device: torch.device, outfile: Path):
    build_ext(verbose=False)
    rows = []
    for N in Ns:
        for ratio in ratios:
            m = max(1, int(N * ratio))
            # adapt ks if any too large: cap at 2*min(m, N-m)
            kcap = 2 * min(m, N - m)
            ks_eff = [k for k in ks if k <= kcap and k > 0]
            if not ks_eff:
                continue
            for pattern in patterns:
                for k in ks_eff:
                    # repeats and average
                    pt, rt, mt = 0.0, 0.0, 0.0
                    for _ in range(max(1, repeats)):
                        p, r, mtime = measure_once(N, up_cols, hidden_dim, gate_cols, m, k, pattern, device)
                        pt += p; rt += r; mt += mtime
                    pt /= max(1, repeats); rt /= max(1, repeats); mt /= max(1, repeats)
                    rows.append({
                        "N": N,
                        "ratio": ratio,
                        "m": m,
                        "k": k,
                        "pattern": pattern,
                        "paired_us": pt * 1e6,
                        "index_us": rt * 1e6,
                        "mask_us": mt * 1e6,
                        "speedup_idx": rt / pt if pt > 0 else float('inf'),
                        "speedup_mask": mt / pt if pt > 0 else float('inf'),
                    })
                    print(f"N={N} ratio={ratio:.3f} m={m} k={k} pattern={pattern} paired={pt*1e6:.1f}us idx={rt*1e6:.1f}us mask={mt*1e6:.1f}us speedup_idx={rt/pt if pt>0 else float('inf'):.2f}x")
    # write CSV
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(outfile, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Parameter sweep for N, m/N, k, and mask patterns")
    ap.add_argument("--Ns", type=str, default="100000,200000")
    ap.add_argument("--ratios", type=str, default="0.01,0.05,0.1")
    ap.add_argument("--ks", type=str, default="8,32,128,512,2048")
    ap.add_argument("--patterns", type=str, default="random,block")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--up_cols", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--outfile", type=str, default="benchmarks/results_sweep.csv")
    args = ap.parse_args()

    Ns = parse_list_numbers(args.Ns)
    ratios = parse_list_floats(args.ratios)
    ks = parse_list_numbers(args.ks)
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip() in ("random", "block")]
    device = torch.device(args.device)

    run_sweep(Ns, ratios, ks, patterns, args.repeats, args.up_cols, args.hidden_dim, args.gate_cols, device, Path(args.outfile))


if __name__ == "__main__":
    main()

