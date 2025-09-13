# Benchmarks

This folder provides a Python/Numpy reference implementation of the Paired Replacement algorithm and benchmark runners.

## Files

- `../src/python/paired_cache.py`: Numpy implementation and the full-rebuild baseline
- `run_microbench.py`: Microbenchmarks that vary mask delta `k` and report update latency and speedup
- `run_e2e_moe.py`: End-to-end simulation with dynamic top-k gating producing changing masks each step

## Requirements

- Python 3.9+
- NumPy

## Usage

Microbenchmarks (varying k):

```
python3 benchmarks/scripts/run_microbench.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --outfile benchmarks/data/results_microbench.csv
```

End-to-end MoE-like simulation:

```
python3 benchmarks/scripts/run_e2e_moe.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --batch 16 --steps 50
```

Both scripts print a concise summary and (for the microbench) write a CSV of results.

## Perf counters (Linux)

Collect LLC misses, cache references, cycles, and instructions using `perf stat`:

```
# Core microarchitectural counters
python3 benchmarks/scripts/run_perf_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu \
  --events cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,cycles,instructions,task-clock \
  --outfile benchmarks/data/results_perf_torch.csv

# Add DRAM bandwidth estimation via IMC CAS counters (Intel)
python3 benchmarks/scripts/run_perf_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu \
  --events cache-references,cache-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,cycles,instructions,task-clock \
  --bw_preset intel --bytes_per_cas 64 \
  --outfile benchmarks/data/results_perf_torch.csv

# Auto-discover IMC events (best-effort)
python3 benchmarks/scripts/run_perf_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu \
  --bw_preset auto --outfile benchmarks/data/results_perf_torch.csv
```

Notes:
- Requires Linux with `perf` installed and accessible in PATH.
- The wrapper runs each k and mode (paired vs rebuild) as isolated processes to measure counters precisely for the update loop.
- Values in the CSV are normalized per step (divide by `steps_per_k`).
- When IMC events are enabled, derived metrics are added: `dram_read_bytes_per_step`, `dram_write_bytes_per_step`, and `dram_bw_bytes_per_sec`.
- Plot perf results (including derived BW):

```
python3 benchmarks/scripts/plot_perf.py --perf_csv benchmarks/data/results_perf_torch.csv --out benchmarks/plots/perf_plots.png
```

## Autotuning parallel knobs

You can autotune the parallel grain and threshold to your CPU and workload, and record the selection:

```
python3 benchmarks/scripts/run_autotune.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --grain_candidates 32,64,128,256 --threshold_candidates 32,64,128,256 \
  --steps 50 --outfile benchmarks/configs/autotune_result.json
```

To benchmark with autotuned values and include them in the CSV:

```
python3 benchmarks/scripts/run_microbench_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu --autotune \
  --grain_candidates 32,64,128,256 --threshold_candidates 32,64,128,256 \
  --outfile benchmarks/data/results_microbench_torch.csv
```

Alternatively, set the environment variables directly:

```
WEIGHT_CACHE_GRAIN=128 WEIGHT_CACHE_PAR_THRESHOLD=128 python3 benchmarks/scripts/run_microbench_torch.py ...
```

Automatic reuse of prior autotune result:

```
# If benchmarks/configs/autotune_result.json exists, the microbench will auto-load it
python3 benchmarks/scripts/run_microbench_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 \
  --ks 16,64,256,1024 --steps_per_k 50 --device cpu --outfile benchmarks/data/results_microbench_torch.csv

# To disable auto-loading, pass:
python3 benchmarks/scripts/run_microbench_torch.py ... --skip_autotune_json

# To specify a custom path:
python3 benchmarks/scripts/run_microbench_torch.py ... --autotune_json path/to/result.json
```
