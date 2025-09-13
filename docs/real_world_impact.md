# Real‑World Impact (First‑Person Notes)

I built Paired Replacement to fix a very specific pain: every step I only need a small, slightly‑changed subset of very large weight matrices, yet the usual path (index_select/gather) rebuilds the whole active block. That wastes memory bandwidth, trashes cache locality, and hurts latency. My algorithm updates the active block in O(k) bytes — only the rows that changed — instead of O(m), where m is the active set size and k is the delta.

## TL;DR
- If the active set is “sticky” (k ≪ m), Paired Replacement cuts bytes moved by ~m/k.
- On CPU, I routinely see ~3–5× faster updates at small k vs. index_select in my microbenches.
- Keep the active rows contiguous so the matmuls run on packed buffers (good for cache + BLAS).

## Where This Helps (Exactly)
- MoE / Top‑k Layers in LLMs and vision models
  - Keep a contiguous buffer of expert rows (gate/up) and down rows (transposed for row access).
  - Update that buffer by diffs when the batch’s top‑k experts change slightly.
  - Visibility: lower update latency, less DRAM traffic; more time left for GEMMs.

- Embedding / Feature Caches (Recs/Ads/Ranking/Search)
  - Maintain a hot mini‑cache of frequently accessed embedding rows as a packed slab.
  - Update the slab by diffs as traffic shifts; GEMMs/MLPs enjoy contiguous reads.

- Transformer KV‑Cache (Sliding Windows / Pruning)
  - Evict old tokens and admit new ones by swapping and appending while keeping K/V contiguous.
  - Useful in streaming/long‑context inference where windows move gradually.

- GNN Pipelines and Graph Serving
  - Mini‑batches touch overlapping node/edge features; keep a packed cache for features and update by diffs.

- Adaptive Scientific Workloads (AMR, Sparse Solvers)
  - Active mesh/row sets evolve slowly; O(k) compaction minimizes bytes moved between iterations.

## When To Use It
- Rule of thumb: use Paired Replacement when k/m < ~0.2. If k grows, switch to the hybrid baseline (repack if k > τ·m). Both paths are in the microbench.
- Sparse regimes: m/N in the 0.5%–5% range (typical for MoE/top‑k), but it works more broadly.

## What I Improve (Precisely)
- Bytes moved: from O(m·S) to O(k·S), S = per‑row bytes. Lower DRAM traffic and LLC misses.
- Latency: update time scales with the small delta k.
- Locality: active rows are tightly packed (row‑major for up, transposed for down), which makes BLAS/DMAs happy.

## Integration Patterns (PyTorch)
- Minimal flow (already in this repo’s e2e example):
  1) Build masks from gating (union top‑k across batch).
  2) `cache.update_active_weights(mask)`
  3) Run `Y = (X @ down_active) @ up_active` using the packed views from the cache.
- Make k small:
  - Sticky mask: keep 80–95% of last step’s mask, fill the rest from the new top‑k.
  - Logit smoothing: `L′t = α·Lt + (1−α)·Lt−1` (α≈0.6–0.95) before top‑k.

## What To Measure
- End‑to‑end step time (ms/step) with error bars (repeats).
- Linux perf: CPI, LLC misses, derived DRAM BW (this repo has a wrapper that exports per‑step counters and GB/s).
- macOS Instruments: “Counters” template + CSV parser included.
- Bytes moved vs. the lower bound `max(A,R)·S` — show the proportionality with k.

## My CPU Results (Typical)
- Microbench (e.g., N=4,096; hidden=512; up=512; m=512):
  - k=16 → ~3.9–4.0× speedup vs index_select; k=64 → ~2.6×; k=256 → ~1.7–1.9×.
- Sweeps (N=1e5–2e5; ratios 0.02–0.05; k from 8 to 512):
  - Speedups commonly 2.8–5× at small k; still >2× in many cases as k grows.
- E2E:
  - Wins emerge when workload keeps k ≪ m (tune N, m/N, and gating). I provide knobs and example regimes in the scripts.

## Deployment Notes (CPU)
- Autotune the parallel grain/threshold (script provided) and record it; the microbench auto‑loads it.
- NUMA/threads: pin threads if needed; keep the active slabs 64‑B aligned (already handled).
- Hybrid switch: if deltas spike, repack (τ≈0.5 is a reasonable starting point).

## Roadmap to GPU
- Keep pools and active slabs on device.
- Launch a paired‑update kernel with coalesced row copies (vectorized ld/st) and optional overlap with matmuls via streams.
- Validate with Nsight/NVPerf counters and DRAM BW; compare to device‑side gather/scatter and fused kernels.

## Quick Commands
```bash
# Torch microbench (CPU)
python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py \
  --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu \
  --outfile paired_replacement/benchmarks/data/results_microbench_torch.csv
python3 paired_replacement/benchmarks/scripts/plot_microbench_overlays.py \
  --torch_csv paired_replacement/benchmarks/data/results_microbench_torch.csv \
  --out paired_replacement/benchmarks/plots/microbench_speedup_overlays.png

# Perf counters (Linux; adds derived DRAM BW)
python3 paired_replacement/benchmarks/scripts/run_perf_torch.py \
  --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu --bw_preset intel \
  --outfile paired_replacement/benchmarks/data/results_perf_torch.csv
python3 paired_replacement/benchmarks/scripts/plot_perf.py \
  --perf_csv paired_replacement/benchmarks/data/results_perf_torch.csv \
  --out paired_replacement/benchmarks/plots/perf_plots.png

# End‑to‑end example (CPU)
python3 paired_replacement/benchmarks/scripts/run_e2e_torch.py \
  --N 65536 --hidden_dim 1024 --up_cols 1024 --m 2048 --batch 32 --steps 50 --device cpu
```

If you deploy this or need help tuning an e2e setup (CPU or GPU), open an issue — I’m happy to help.
