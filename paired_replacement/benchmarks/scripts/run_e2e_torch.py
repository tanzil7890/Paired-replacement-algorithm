#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from contextlib import nullcontext


def build_ext(verbose: bool = False):
    proj_root = Path(__file__).parents[2]
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


def topk_union_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    # logits: [batch, N]; union the per-sample top-k to form a shared mask
    batch, N = logits.shape
    topk_idx = torch.topk(logits, k=min(k, N), dim=1).indices  # [batch, k]
    mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
    mask[topk_idx.reshape(-1)] = True
    # Cap to at most k by selecting the k largest pooled logits
    # Pool by max across batch
    pooled = torch.zeros(N, dtype=logits.dtype, device=logits.device)
    pooled.index_put_((topk_idx.reshape(-1),), logits.gather(1, topk_idx).reshape(-1), accumulate=True)
    # Keep at most k highest pooled (if union > k)
    if mask.sum().item() > k:
        vals, idxs = torch.topk(pooled, k)
        new_mask = torch.zeros_like(mask)
        new_mask[idxs] = True
        return new_mask
    return mask


def run_e2e(N: int, hidden_dim: int, up_cols: int, gate_cols: int, m: int, batch: int, steps: int, device: torch.device,
            profile: bool = False, profile_out: str | None = None):
    build_ext(verbose=False)
    WeightCache = torch.classes.paired.WeightCache

    # Random fixed weights
    up = torch.randn(N, up_cols, dtype=torch.float32, device=device)
    down = torch.randn(hidden_dim, N, dtype=torch.float32, device=device)
    gate = torch.randn(N, gate_cols, dtype=torch.float32, device=device) if gate_cols > 0 else torch.empty(N, 0, dtype=torch.float32, device=device)
    gate_scores_W = torch.randn(N, hidden_dim, dtype=torch.float32, device=device)

    # Initialize with first batch
    X = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)
    logits = X @ gate_scores_W.T
    mask0 = topk_union_mask(logits, m)
    cache = WeightCache(mask0, hidden_dim, gate, up, down, gate_cols > 0)

    # Paired path (optionally profile)
    t_paired = 0.0
    if profile:
        try:
            import torch.profiler as prof
            with prof.profile(activities=[prof.ProfilerActivity.CPU], record_shapes=True) as p:
                for _ in range(steps):
                    X = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)
                    logits = X @ gate_scores_W.T
                    mask = topk_union_mask(logits, m)
                    t0 = time.perf_counter()
                    cache.update_active_weights(mask)
                    down_act = cache.get_active_down_weight()
                    up_act = cache.get_concat_weight()
                    _ = (X @ down_act) @ up_act
                    t_paired += time.perf_counter() - t0
            if profile_out:
                Path(profile_out).parent.mkdir(parents=True, exist_ok=True)
                p.export_chrome_trace(profile_out)
        except Exception:
            # Fallback to normal timing if profiler not available
            profile = False
    if not profile:
        for _ in range(steps):
            X = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)
            logits = X @ gate_scores_W.T
            mask = topk_union_mask(logits, m)
            t0 = time.perf_counter()
            cache.update_active_weights(mask)
            down_act = cache.get_active_down_weight()
            up_act = cache.get_concat_weight()
            _ = (X @ down_act) @ up_act
            t_paired += time.perf_counter() - t0

    # Rebuild path
    t_rebuild = 0.0
    mask = mask0
    for _ in range(steps):
        X = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)
        logits = X @ gate_scores_W.T
        mask = topk_union_mask(logits, m)
        t0 = time.perf_counter()
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        up_act = up.index_select(0, idx)
        if gate_cols > 0:
            gate_act = gate.index_select(0, idx)
            up_concat = torch.cat([gate_act, up_act], dim=1)
        else:
            up_concat = up_act
        down_act = down.index_select(1, idx)
        _ = (X @ down_act) @ up_concat
        t_rebuild += time.perf_counter() - t0

    return {
        "paired_ms_per_step": (t_paired / steps) * 1e3,
        "rebuild_ms_per_step": (t_rebuild / steps) * 1e3,
        "speedup": (t_rebuild / t_paired) if t_paired > 0 else float("inf"),
    }


def main():
    ap = argparse.ArgumentParser(description="End-to-end Torch MoE-like benchmark (CPU)")
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--hidden_dim", type=int, default=1024)
    ap.add_argument("--up_cols", type=int, default=1024)
    ap.add_argument("--gate_cols", type=int, default=0)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile_out", type=str, default="benchmarks/e2e_torch_profile.json")
    args = ap.parse_args()

    res = run_e2e(
        args.N, args.hidden_dim, args.up_cols, args.gate_cols, args.m, args.batch, args.steps, torch.device(args.device),
        profile=args.profile, profile_out=args.profile_out
    )
    print(
        f"paired={res['paired_ms_per_step']:.3f} ms/step  rebuild={res['rebuild_ms_per_step']:.3f} ms/step  speedup={res['speedup']:.2f}x"
    )


if __name__ == "__main__":
    main()
