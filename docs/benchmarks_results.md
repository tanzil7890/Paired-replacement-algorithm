# Benchmarks Results (How I reproduce them)

Here’s how I run the benchmarks on my machine and what I typically see.

## Setup

I first install the package for clean imports and console scripts:
```bash
python install_dev.py
```

This provides both console scripts and traditional script execution methods. All scripts live under `benchmarks/scripts/` and emit CSVs with provenance banners and plots.

## Microbench (Torch extension)

- Command:
  - `python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --device cpu --outfile benchmarks/data/results_microbench_torch.csv`
- Plots:
  - `python3 paired_replacement/benchmarks/scripts/plot_microbench_overlays.py --torch_csv benchmarks/data/results_microbench_torch.csv --out benchmarks/plots/microbench_speedup_overlays.png`
- Representative small run (Apple M3) I see:
  - k=16: paired≈57 µs, index≈315 µs, mask≈169 µs → speedup vs index≈5.5×
  - k=64: paired≈58 µs, index≈93 µs, mask≈113 µs → speedup vs index≈1.6×
- Figure:
  - `benchmarks/plots/microbench_speedup_overlays.png`

## Timing with Confidence Intervals

- Command:
  - `python3 paired_replacement/benchmarks/scripts/run_timing_ci.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --repeats 5 --device cpu --outfile benchmarks/data/results_timing_ci.csv`
- Representative small run (Apple M3) I see:
  - k=16: paired≈54.5±2.5 µs, index≈152.3±30.2 µs → speedup≈2.79×
  - k=64: paired≈75.6±8.4 µs, index≈100.8±11.2 µs → speedup≈1.33×

## Parameter Sweeps (N, m/N, k, patterns)

- Command:
  - `python3 paired_replacement/benchmarks/scripts/run_sweep.py --Ns 100000,1000000 --ratios 0.01,0.05,0.1 --ks 8,32,128,512,2048 --patterns random,block --repeats 3 --up_cols 256 --hidden_dim 256 --outfile benchmarks/data/results_sweep.csv`
  - `python3 paired_replacement/benchmarks/scripts/plot_sweep.py --csv benchmarks/data/results_sweep.csv --out benchmarks/plots/sweep_plots.png`
- What I observe:
  - Paired updates consistently outperform rebuilds for small k; advantage narrows as k grows.
  - Structured block deltas can shift crossover; hybrid baseline is competitive at larger k.
- Figure:
  - `benchmarks/plots/sweep_plots.png`

## End‑to‑End Torch (CPU)

- Command:
  - `python3 paired_replacement/benchmarks/scripts/run_e2e_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --batch 16 --steps 50 --device cpu`
  - Optional profile: `--profile --profile_out benchmarks/configs/e2e_torch_profile.json`
- Notes: end‑to‑end crossover depends on parameter choices. Larger N and smaller m/N generally improve the paired advantage; I also enforce “sticky” masks to keep k small.

## Perf (Linux) and Instruments (macOS)

- Linux perf:
  - `python3 paired_replacement/benchmarks/scripts/run_perf_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --device cpu --bw_preset intel --bytes_per_cas 64 --outfile benchmarks/data/results_perf_torch.csv`
  - `python3 paired_replacement/benchmarks/scripts/plot_perf.py --perf_csv benchmarks/data/results_perf_torch.csv --out benchmarks/plots/perf_plots.png`
- macOS Instruments:
  - `python3 paired_replacement/benchmarks/scripts/run_xctrace.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --k 64 --steps_per_k 50 --outdir benchmarks/data/xctrace`
  - Parse summary: `python3 paired_replacement/benchmarks/scripts/parse_xctrace_csv.py --paired_export benchmarks/data/xctrace/paired_k64_export --rebuild_export benchmarks/data/xctrace/rebuild_k64_export --out_json benchmarks/configs/instruments_summary.json --out_md benchmarks/data/instruments_summary.md`

## Multi‑CPU Suite and Merge

- Per-host: `python3 paired_replacement/benchmarks/scripts/run_multi_cpu_suite.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --repeats 5 --device cpu --outdir results`
- Merge across hosts: `python3 paired_replacement/benchmarks/scripts/merge_multi_cpu.py --roots results --out_csv benchmarks/data/merged_results_timing_ci.csv --out_md benchmarks/data/merged_report.md`

## Provenance and Autotune

- I record autotune knobs per host: `python3 paired_replacement/benchmarks/scripts/run_autotune.py --N ... --outfile paired_replacement/benchmarks/configs/autotune_result.json`
- The microbench auto‑loads autotune JSON unless `--skip_autotune_json` is set.
- CSVs include banner comments with device, autotune source, and timestamp; perf adds a JSON sidecar.
