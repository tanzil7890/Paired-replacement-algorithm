# Project Structure (How I organized the repo)

This is how I structure the project so I can iterate quickly, keep imports clean, and reproduce results reliably.
## Installation and Usage

### Quick Setup
```bash
# Install as Python package (editable)
python install_dev.py

# Use clean imports
from paired_replacement.src.python.paired_cache import PairedCache

# Run scripts directly
python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --device cpu
```


## Directory Layout (current)

```
paired_replacement_algorithm_paper/
├── README.md                         # Main project documentation
├── CLAUDE.md                         # Project guidance
├── pyproject.toml                    # Python packaging config
├── setup.py                          # Legacy setup for compatibility
├── install_dev.py                    # Development installation script
├── .gitignore                        # Git ignore patterns
│
├── paired_replacement/               # Python package
│   ├── __init__.py
│   ├── src/
│   │   ├── __init__.py
│   │   ├── cpp/
│   │   │   ├── weight_cache_binding.cpp
│   │   │   └── paired/weight_cache.h
│   │   └── python/
│   │       ├── __init__.py
│   │       └── paired_cache.py
│   └── benchmarks/
│       ├── __init__.py
│       ├── scripts/                  # Runner and plotter scripts
│       ├── data/                     # CSVs and reports
│       ├── plots/                    # Generated figures
│       └── configs/                  # Autotune/sysinfo/profile JSON
│
├── docs/                             # Documentation
│   ├── installation.md
│   ├── algorithm_pseudocode.md
│   ├── benchmarks_results.md
│   └── project_structure.md          # This file
│
├── paper/                            # Research paper materials
│   └── paper_outline.md
│
├── build/                            # JIT build artifacts (PyTorch extension)
└── results/                          # Archived experiment results
```

## Primary Commands

### Benchmarking
- Torch microbench:
  - `python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --device cpu`
- NumPy microbench:
  - `python3 paired_replacement/benchmarks/scripts/run_microbench.py --N 8192`
- Overlays:
  - `python3 paired_replacement/benchmarks/scripts/plot_microbench_overlays.py --torch_csv paired_replacement/benchmarks/data/results_microbench_torch.csv --out paired_replacement/benchmarks/plots/microbench_speedup_overlays.png`
- Parameter sweeps:
  - `python3 paired_replacement/benchmarks/scripts/run_sweep.py --Ns 100000,1000000 --ratios 0.01,0.05,0.1 --ks 8,32,128,512,2048 --patterns random,block --repeats 3 --up_cols 256 --hidden_dim 256 --outfile paired_replacement/benchmarks/data/results_sweep.csv`
- Plot sweeps:
  - `python3 paired_replacement/benchmarks/scripts/plot_sweep.py --csv paired_replacement/benchmarks/data/results_sweep.csv --out paired_replacement/benchmarks/plots/sweep_plots.png`

### Performance Counters (Linux)
- Perf:
  - `python3 paired_replacement/benchmarks/scripts/run_perf_torch.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --device cpu --bw_preset intel --bytes_per_cas 64 --outfile paired_replacement/benchmarks/data/results_perf_torch.csv`
- Plot perf:
  - `python3 paired_replacement/benchmarks/scripts/plot_perf.py --perf_csv paired_replacement/benchmarks/data/results_perf_torch.csv --out paired_replacement/benchmarks/plots/perf_plots.png`

### Autotune and Provenance
- Autotune:
  - `python3 paired_replacement/benchmarks/scripts/run_autotune.py --N 8192 --outfile paired_replacement/benchmarks/configs/autotune_result.json`
- Microbench (auto-load autotune):
  - `python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --device cpu`

### End-to-End and Reporting
- E2E Torch sweep + plot:
  - `python3 paired_replacement/benchmarks/scripts/run_e2e_sweep.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m_list 512,1024,2048 --batch_list 8,16,32 --steps 50 --device cpu --outfile paired_replacement/benchmarks/data/results_e2e_torch.csv`
  - `python3 paired_replacement/benchmarks/scripts/plot_e2e.py --csv paired_replacement/benchmarks/data/results_e2e_torch.csv --out paired_replacement/benchmarks/plots/e2e_speedup.png`
- Profile single run:
  - `python3 paired_replacement/benchmarks/scripts/run_e2e_torch.py ... --profile --profile_out paired_replacement/benchmarks/configs/e2e_torch_profile.json`
- Multi-CPU suite:
  - `python3 paired_replacement/benchmarks/scripts/run_multi_cpu_suite.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024 --steps_per_k 50 --repeats 5 --device cpu --outdir results`
- Merge hosts:
  - `python3 paired_replacement/benchmarks/scripts/merge_multi_cpu.py --roots results --out_csv paired_replacement/benchmarks/data/merged_results_timing_ci.csv --out_md paired_replacement/benchmarks/data/merged_report.md`

## Notes on Packaging

### Source Code Organization
- C++: `paired_replacement/src/cpp/` — Core algorithm and PyTorch bindings
- Python: `paired_replacement/src/python/` — NumPy reference and helpers
- Imports: `from paired_replacement.src.python.paired_cache import PairedCache`

### Benchmark Organization
- Scripts: `paired_replacement/benchmarks/scripts/`
- Data: `paired_replacement/benchmarks/data/`
- Plots: `paired_replacement/benchmarks/plots/`
- Configs: `paired_replacement/benchmarks/configs/`

## Development Workflow

1. Install: `python install_dev.py` (or `pip install -e .`)
2. Edit: change code in `paired_replacement/src/` and scripts in `paired_replacement/benchmarks/scripts/`
3. Run: execute benchmarks and plots via the commands above
4. Artifacts: CSVs in `benchmarks/data/`, plots in `benchmarks/plots/`, reports in MD files

All file paths and commands reflect the current package layout.
