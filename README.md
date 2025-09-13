# Paired Replacement: A Differential Caching Algorithm for Dynamic Sparse Neural Network Inference


## Overview

This folder contains the complete implementation and research materials for the **Paired Replacement Algorithm**, a novel cache replacement technique for dynamic sparse neural network inference.

## Why I Built This (Real‑World Impact)

In practice, the “active rows” of big matrices (experts, embeddings, features) change slowly step‑to‑step. I designed Paired Replacement to update that active block in O(k) bytes (only what changed) instead of O(m) (rebuild everything). On CPU this consistently cuts memory traffic and latency when k ≪ m.

Where this helps me (and you):
- Mixture‑of‑Experts / top‑k layers: keep a packed expert buffer; updates scale with the small mask delta k. I see ~3–5× faster updates for small k (vs index_select) in my microbenches.
- Embedding/feature caches (recs/ads/ranking): maintain a hot, contiguous mini‑cache of rows so GEMMs stay fast and DRAM traffic drops.
- KV‑cache/windowed attention: remove/add token rows by diffs; swap‑with‑last keeps buffers compact.
- GNN mini‑batches & adaptive scientific codes (AMR, sparse solvers): active row sets evolve gradually; O(k) compaction reduces copies.

When to use it: if k/m < ~0.2 (i.e., the active set is sticky). If k grows, I provide a hybrid baseline (repack if k > τ·m) you can enable.

Status: CPU artifact is ready and reproducible (scripts + plots). A device‑native GPU path is the next milestone.

## Core Innovation

The Paired Replacement Algorithm introduces a fundamentally new approach to cache management through:

1. **XOR-based Differential Updates**: Uses bitwise XOR operations to compute mask differences efficiently
2. **Paired Substitution Strategy**: Matches removals with additions for direct memory replacement
3. **Cache-line Aware Memory Layout**: Optimized for modern CPU cache hierarchies
4. **Zero-copy Tensor Integration**: Seamless integration with PyTorch through `torch::from_blob`

## Performance Results

- **6.7× faster cache updates**: 29.89ms → 4.46ms compared to naive `index_select`
- **Better cache locality**: Row-major for up projection, column-major for down projection
- **Contiguous memory access**: Single memcpy operations for cache updates
- **O(1) lookup complexity**: Through hash map-based index tracking

## Algorithm Description

### Section 1: XOR-based Mask Difference Computation
```cpp
auto added_mask = new_mask & (~old_mask);     // new & ~old = added
auto removed_mask = old_mask & (~new_mask);   // old & ~new = removed
```

### Section 2: Paired Replacement Strategy
```cpp
const size_t pairs_to_process = std::min(num_removals, num_additions);

// Direct replacement - most cache-efficient operation
for (size_t i = 0; i < pairs_to_process; ++i) {
    int64_t removed_idx = diff.removed_indices[i];
    int64_t added_idx = diff.added_indices[i];
    
    // Single memcpy per matrix - optimal memory usage
    std::memcpy(active_buffer + pos * row_size,
                memory_pool + added_idx * row_size,
                row_size * sizeof(float));
}
```

### Section 3: Handle Remaining Operations
- **Excess additions**: Append to end of active buffer
- **Excess removals**: Move last element to fill gaps

For a more complete walkthrough, invariants, and structured steps, see `docs/algorithm_pseudocode.md`.

## Key Technical Features

### Memory Management
- **64-byte aligned allocations** for optimal cache line usage
- **Pre-allocated buffers** to avoid dynamic allocation overhead
- **Smart pointer management** for automatic cleanup

### Data Structures
- `std::vector<int64_t> active_indices`: Maintains order for contiguous access
- `std::unordered_map<int64_t, size_t> index_to_position`: O(1) position lookup
- Cache-aligned memory pools for all weight matrices

### Integration
- **PyTorch compatibility**: Uses `torch::from_blob` for zero-copy tensor views on CPU
- **Device support**: CPU returns zero-copy views; non-CPU devices materialize tensors via `.to(device)`
- **Dtype support (current)**: Float32 only (templating planned)


## Installation

### Quick Start
```bash
# Install in development mode
python install_dev.py

# Or manually
pip install -e .
```

### Using the Package
```python
from paired_replacement.src.python.paired_cache import PairedCache, baseline_full_rebuild
```

### Running Scripts
```bash
python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --device cpu
python3 paired_replacement/benchmarks/scripts/run_autotune.py --N 8192 --outfile autotune.json
```

See `docs/installation.md` for detailed installation instructions.

## File Structure

- `src/cpp/paired/weight_cache.h`: Complete algorithm implementation
- `src/cpp/weight_cache_binding.cpp`: PyTorch extension bindings
- `src/python/paired_cache.py`: NumPy reference implementation
- `paired_replacement/`: Python package structure for clean imports
- `benchmarks/scripts/`: Performance evaluation and plotting scripts
- `benchmarks/data/`: Benchmark results and CSV files
- `benchmarks/plots/`: Generated performance visualizations
- `docs/`: Technical documentation and algorithm details
- `paper/`: Research paper materials and publication plan
- `results/`: Archived experiment results
- `pyproject.toml`: Modern Python packaging configuration

## Invariants and Constraints

- **Shapes**: `up_weight` `[sparse_dim, up_cols]`; `down_weight` `[hidden_size, sparse_dim]`; optional `gate_weight` `[sparse_dim, gate_cols]`
- **Mask**: Boolean tensor of length `sparse_dim` (internally normalized to the current device)
- **Outputs**: `get_concat_weight()` -> `[num_active, up_cols + (gate_cols if gate present)]` via column-wise concat; `get_active_down_weight()` -> `[hidden_size, num_active]`
- **Dtype**: Currently float32 only
- **Empty mask**: Supported; returns empty, correctly shaped tensors

## Next Steps for Publication

1. **Theoretical Analysis**: Formal complexity analysis and cache behavior modeling
2. **Comprehensive Evaluation**: Comparison with traditional cache replacement algorithms
3. **Dtype Templating**: Add fp16/bf16/int8 specialization paths
4. **Broader Applicability**: Extension to general sparse computation scenarios
5. **Paper Writing**: Full research paper with experimental validation

## Benchmarks

- NumPy reference and microbench: `python3 paired_replacement/benchmarks/scripts/run_microbench.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --outfile benchmarks/data/results_microbench.csv`
- C++/PyTorch extension microbench (builds JIT): `python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --device cpu --outfile benchmarks/data/results_microbench_torch.csv`
- Plot speedup vs k: `python3 paired_replacement/benchmarks/scripts/plot_microbench.py --numpy_csv benchmarks/data/results_microbench.csv --torch_csv benchmarks/data/results_microbench_torch.csv --out benchmarks/plots/microbench_speedup.png`
- Plot torch overlays (index_select vs boolean-mask baselines): `python3 paired_replacement/benchmarks/scripts/plot_microbench_overlays.py --torch_csv benchmarks/data/results_microbench_torch.csv --out benchmarks/plots/microbench_speedup_overlays.png`

- Parameter sweeps (N, m/N, k, random vs block patterns):
  - Run: `python3 paired_replacement/benchmarks/scripts/run_sweep.py --Ns 100000,1000000 --ratios 0.01,0.05,0.1 --ks 8,32,128,512,2048 --patterns random,block --repeats 3 --up_cols 256 --hidden_dim 256 --outfile benchmarks/data/results_sweep.csv`
  - Plot: `python3 paired_replacement/benchmarks/scripts/plot_sweep.py --csv benchmarks/data/results_sweep.csv --out benchmarks/plots/sweep_plots.png`

Notes:
- The PyTorch extension compiles locally (CPU) and exposes the custom class as `torch.classes.paired.WeightCache`.
- For Apple Silicon (ARM64), SIMD headers are guarded; no SIMD is used.
- Parallelization knobs (environment variables):
  - `WEIGHT_CACHE_GRAIN` (default: 64) — chunk size used in parallel_for
  - `WEIGHT_CACHE_PAR_THRESHOLD` (default: 64) — minimum work items to enable parallel_for
  - `WEIGHT_CACHE_HYBRID_TAU_FRAC` (default: 0.5) — hybrid baseline threshold (rebuild if k > tau*m)

## Perf counters (Linux)

Collect microarchitectural counters and derived memory bandwidth:

```
# core counters
python3 paired_replacement/benchmarks/scripts/run_perf_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu \
  --events cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,cycles,instructions,task-clock \
  --outfile benchmarks/data/results_perf_torch.csv

# DRAM bandwidth via IMC CAS counters (Intel preset)
python3 paired_replacement/benchmarks/scripts/run_perf_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu --bw_preset intel --bytes_per_cas 64 \
  --outfile benchmarks/data/results_perf_torch.csv

python3 paired_replacement/benchmarks/scripts/plot_perf.py --perf_csv benchmarks/data/results_perf_torch.csv --out benchmarks/plots/perf_plots.png
```

## Autotuning and Provenance

```
# Autotune once and record result
python3 paired_replacement/benchmarks/scripts/run_autotune.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --steps 50 \
  --outfile benchmarks/configs/autotune_result.json

# Microbench will auto-load autotune JSON unless --skip_autotune_json is set
python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu --outfile benchmarks/data/results_microbench_torch.csv
```

Each CSV begins with a banner comment that records device, torch version, autotune source, and timestamp.


## Usage Example

```cpp
// Initialize with sparsity mask
WeightCache cache(init_mask, hidden_size, gate_weight, up_weight, down_weight);

// Update with new sparsity pattern - uses paired replacement algorithm
cache.update_active_weights(new_mask);

// Get optimized weight tensors (zero-copy)
auto active_weights = cache.get_concat_weight();
auto active_down = cache.get_active_down_weight();
```

## Performance Characteristics

- **Time Complexity**: O(k) where k = number of changes in mask
- **Space Complexity**: O(n) where n = total number of weights
- **Cache Efficiency**: Optimal for sequential access patterns
- **Memory Bandwidth**: Minimized through paired substitution strategy

## Minimal Data Movement Guarantee

- **Model**: Let A be additions and R be removals in a mask-delta update; let S be the per-row byte size across all concatenated matrices (omit gate if absent). We consider contiguous active buffers and whole-row copies (from pool→active or within active) as the primitive operation.
- **Lower bound**: Any algorithm must write at least `max(A, R)` rows, i.e., `max(A, R)·S` bytes, to transform one contiguous active set into another.
- **Optimality**: The paired replacement algorithm writes exactly `max(A, R)` rows by first overwriting `min(A, R)` holes with additions, then handling the residual `|A−R|` via appends (if A>R) or move-last-to-hole (if R>A).
- **Cache lines**: With B-byte cache lines, B-aligned rows, and S a multiple of B, the algorithm attains the minimal `max(A, R)·(S/B)` cache-line writes (and reads).
- **Ordering**: No stable order is enforced; if stability is required, the lower bound increases toward `A + R` row writes in the worst case.

---


## End-to-End (Torch, CPU)

```
python3 paired_replacement/benchmarks/scripts/run_e2e_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --batch 16 --steps 50 --device cpu
# Optional profiling
python3 paired_replacement/benchmarks/scripts/run_e2e_torch.py ... --profile --profile_out benchmarks/configs/e2e_torch_profile.json
```

## Multi-CPU Coverage and Merge

```
# Per-host suite
python3 paired_replacement/benchmarks/scripts/run_multi_cpu_suite.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --repeats 5 --device cpu --outdir results

# Merge across hosts
python3 paired_replacement/benchmarks/scripts/merge_multi_cpu.py --roots results --out_csv benchmarks/data/merged_results_timing_ci.csv --out_md benchmarks/data/merged_report.md
```

## Figures

- Overlays (Paired vs baselines): `benchmarks/plots/microbench_speedup_overlays.png`
- Perf (LLC miss rate, CPI, DRAM BW): `benchmarks/plots/perf_plots.png`
- Sweeps (speedup vs k across ratios/patterns): `benchmarks/plots/sweep_plots.png`

## Algorithm Pseudocode and Theory

- Full algorithm walkthrough and invariants: `docs/algorithm_pseudocode.md`
- Paper plan, theory, and evaluation methodology: `paper/paper_outline.md`

If you deploy this or need help tuning an e2e setup (CPU or GPU), open an issue — I’m happy to help.

