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
            profile: bool = False, profile_out: str | None = None, sticky_beta: float = 0.0, hybrid_tau: float = 0.5):
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
    if sticky_beta > 0:
        sticky = logits.clone()
    mask0 = topk_union_mask(logits, m)
    cache = WeightCache(mask0, hidden_dim, gate, up, down, gate_cols > 0)

    # Paired path (optionally profile)
    t_paired = 0.0
    # Track realized mask deltas to relate to microbench 'k'
    k_sum = 0
    k_cnt = 0
    if profile:
        try:
            import torch.profiler as prof
            with prof.profile(activities=[prof.ProfilerActivity.CPU], record_shapes=True) as p:
                prev_mask = mask0.clone()
                for _ in range(steps):
                    X = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)
                    logits_new = X @ gate_scores_W.T
                    logits = sticky_beta * sticky + (1 - sticky_beta) * logits_new if sticky_beta > 0 else logits_new
                    if sticky_beta > 0:
                        sticky = logits
                    mask = topk_union_mask(logits, m)
                    # measure realized delta k
                    added = (~prev_mask) & mask
                    removed = prev_mask & (~mask)
                    k_sum += int((added.sum() + removed.sum()).item())
                    k_cnt += 1
                    t0 = time.perf_counter()
                    cache.update_active_weights(mask)
                    down_act = cache.get_active_down_weight()
                    up_act = cache.get_concat_weight()
                    _ = (X @ down_act) @ up_act
                    t_paired += time.perf_counter() - t0
                    prev_mask = mask
            if profile_out:
                Path(profile_out).parent.mkdir(parents=True, exist_ok=True)
                p.export_chrome_trace(profile_out)
        except Exception:
            # Fallback to normal timing if profiler not available
            profile = False
    if not profile:
        prev_mask = mask0.clone()
        for _ in range(steps):
            X = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)
            logits_new = X @ gate_scores_W.T
            logits = sticky_beta * sticky + (1 - sticky_beta) * logits_new if sticky_beta > 0 else logits_new
            if sticky_beta > 0:
                sticky = logits
            mask = topk_union_mask(logits, m)
            # realized delta k this step
            added = (~prev_mask) & mask
            removed = prev_mask & (~mask)
            k_sum += int((added.sum() + removed.sum()).item())
            k_cnt += 1
            t0 = time.perf_counter()
            cache.update_active_weights(mask)
            down_act = cache.get_active_down_weight()
            up_act = cache.get_concat_weight()
            _ = (X @ down_act) @ up_act
            t_paired += time.perf_counter() - t0
            prev_mask = mask

    # Rebuild path
    t_rebuild = 0.0
    mask = mask0
    for _ in range(steps):
        X = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)
        logits_new = X @ gate_scores_W.T
        logits = sticky_beta * logits + (1 - sticky_beta) * logits_new if sticky_beta > 0 else logits_new
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

    # Hybrid path: rebuild if delta > tau*m else paired update
    t_hybrid = 0.0
    prev_mask_h = mask0.clone()
    cache_h = WeightCache(mask0, hidden_dim, gate, up, down, gate_cols > 0)
    for _ in range(steps):
        X = torch.randn(batch, hidden_dim, dtype=torch.float32, device=device)
        logits_new = X @ gate_scores_W.T
        logits = sticky_beta * logits + (1 - sticky_beta) * logits_new if sticky_beta > 0 else logits_new
        mask_h = topk_union_mask(logits, m)
        added = (~prev_mask_h) & mask_h
        removed = prev_mask_h & (~mask_h)
        delta = int((added.sum() + removed.sum()).item())
        if delta > int(hybrid_tau * m):
            t0 = time.perf_counter()
            idx = torch.nonzero(mask_h, as_tuple=False).squeeze(-1)
            up_act = up.index_select(0, idx)
            if gate_cols > 0:
                gate_act = gate.index_select(0, idx)
                up_concat = torch.cat([gate_act, up_act], dim=1)
            else:
                up_concat = up_act
            down_act = down.index_select(1, idx)
            _ = (X @ down_act) @ up_concat
            t_hybrid += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            cache_h.update_active_weights(mask_h)
            down_act = cache_h.get_active_down_weight()
            up_act = cache_h.get_concat_weight()
            _ = (X @ down_act) @ up_act
            t_hybrid += time.perf_counter() - t0
        prev_mask_h = mask_h

    avg_k = (k_sum / k_cnt) if k_cnt > 0 else 0.0
    return {
        "paired_ms_per_step": (t_paired / steps) * 1e3,
        "rebuild_ms_per_step": (t_rebuild / steps) * 1e3,
        "speedup": (t_rebuild / t_paired) if t_paired > 0 else float("inf"),
        "hybrid_ms_per_step": (t_hybrid / steps) * 1e3,
        "hybrid_speedup": (t_rebuild / t_hybrid) if t_hybrid > 0 else float("inf"),
        "avg_k": avg_k,
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
    ap.add_argument("--sticky_beta", type=float, default=0.0,
                    help="EMA smoothing factor for logits to induce sticky masks (0=no smoothing)")
    ap.add_argument("--hybrid_tau", type=float, default=0.5,
                    help="Hybrid threshold as fraction of m; rebuild if delta k > tau*m")
    ap.add_argument("--outfile", type=str, default="")
    args = ap.parse_args()

    res = run_e2e(
        args.N, args.hidden_dim, args.up_cols, args.gate_cols, args.m, args.batch, args.steps, torch.device(args.device),
        profile=args.profile, profile_out=args.profile_out, sticky_beta=args.sticky_beta, hybrid_tau=args.hybrid_tau
    )
    # Optional CSV output (single row)
    if args.outfile:
        import csv
        outp = Path(args.outfile)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "N","hidden_dim","up_cols","gate_cols","m","batch","steps",
            "paired_ms_per_step","rebuild_ms_per_step","speedup",
            "hybrid_ms_per_step","hybrid_speedup",
            "avg_k","device","sticky_beta","hybrid_tau"
        ]
        write_header = not outp.exists()
        with open(outp, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow({
                "N": args.N, "hidden_dim": args.hidden_dim, "up_cols": args.up_cols, "gate_cols": args.gate_cols,
                "m": args.m, "batch": args.batch, "steps": args.steps,
                "paired_ms_per_step": f"{res['paired_ms_per_step']:.6f}",
                "rebuild_ms_per_step": f"{res['rebuild_ms_per_step']:.6f}",
                "speedup": f"{res['speedup']:.6f}",
                "hybrid_ms_per_step": f"{res['hybrid_ms_per_step']:.6f}",
                "hybrid_speedup": f"{res['hybrid_speedup']:.6f}",
                "avg_k": f"{res['avg_k']:.6f}",
                "device": args.device,
                "sticky_beta": f"{args.sticky_beta:.3f}",
                "hybrid_tau": f"{args.hybrid_tau:.3f}",
            })
        print(f"Appended results to {outp}")

    print(
        f"paired={res['paired_ms_per_step']:.3f} ms/step  rebuild={res['rebuild_ms_per_step']:.3f} ms/step  speedup={res['speedup']:.2f}x  avg_k≈{res['avg_k']:.1f}"
    )


if __name__ == "__main__":
    main()
