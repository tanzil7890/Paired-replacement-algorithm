# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing the **Paired Replacement Algorithm**, a differential caching algorithm for dynamic sparse neural network inference. The algorithm uses XOR-based mask differences and paired substitution strategy to achieve O(k) cache updates where k is the number of changes in the sparsity mask.

## Installation and Setup

```bash
# Quick development installation
python install_dev.py

# Or manually
pip install -e .

# Install with benchmark dependencies
pip install -e ".[benchmarks,dev]"
```

## Architecture

### Core Components

- **C++ Implementation**: `paired_replacement/src/cpp/paired/weight_cache.h` - Main algorithm implementation with cache-line aligned memory management
- **PyTorch Binding**: `paired_replacement/src/cpp/weight_cache_binding.cpp` - PyTorch extension providing `torch.classes.paired.WeightCache`
- **Python Reference**: `paired_replacement/src/python/paired_cache.py` - NumPy-based reference implementation
- **Benchmarking Suite**: `paired_replacement/benchmarks/scripts/` - Comprehensive performance evaluation tools

### Package Structure

```
paired_replacement/
├── src/
│   ├── cpp/          # C++ core algorithm and PyTorch bindings
│   └── python/       # Python reference implementation
└── benchmarks/
    ├── scripts/      # Benchmark execution scripts
    ├── data/         # Results CSV files
    └── plots/        # Generated visualizations
```

## Key Commands

### Development
```bash
# Install package in development mode
python install_dev.py

# Format code (if black/isort installed)
black .
isort .

# Run tests (if pytest installed)
pytest
```

### Benchmarking

#### Microbenchmarks
```bash
# NumPy reference implementation
python3 paired_replacement/benchmarks/scripts/run_microbench.py --N 8192 --ks 16,64,256,1024

# PyTorch C++ extension (JIT compilation)
python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --device cpu

# Performance counters (Linux only)
python3 paired_replacement/benchmarks/scripts/run_perf_torch.py --N 8192 --events cache-references,cache-misses
```

#### End-to-End Evaluation
```bash
# Single configuration
python3 paired_replacement/benchmarks/scripts/run_e2e_torch.py --N 8192 --hidden_dim 1024 --batch 16

# Parameter sweep
python3 paired_replacement/benchmarks/scripts/run_e2e_sweep.py --N 8192 --m_list 512,1024,2048 --batch_list 8,16,32
```

#### Visualization
```bash
# Plot microbenchmark results
python3 paired_replacement/benchmarks/scripts/plot_microbench_overlays.py --torch_csv benchmarks/data/results_microbench_torch.csv

# Plot end-to-end results
python3 paired_replacement/benchmarks/scripts/plot_e2e.py --csv benchmarks/data/results_e2e_torch.csv
```

### Autotuning
```bash
# Generate autotune configuration
python3 paired_replacement/benchmarks/scripts/run_autotune.py --N 8192 --outfile benchmarks/configs/autotune_result.json

# Benchmarks automatically load autotune configs unless --skip_autotune_json is specified
```

### Multi-Host Testing
```bash
# Run comprehensive test suite
python3 paired_replacement/benchmarks/scripts/run_multi_cpu_suite.py --N 8192 --outdir results

# Merge results across hosts
python3 paired_replacement/benchmarks/scripts/merge_multi_cpu.py --roots results --out_csv merged_results.csv
```

## Algorithm Details

### Core Innovation
The algorithm uses XOR operations to compute mask differences:
```cpp
auto added_mask = new_mask & (~old_mask);     // new & ~old = added
auto removed_mask = old_mask & (~new_mask);   // old & ~new = removed
```

Then applies paired substitution to minimize memory movement:
- Pairs removals with additions for direct replacement (most cache-efficient)
- Handles excess additions by appending to buffer
- Handles excess removals by moving last element to fill gaps

### Performance Characteristics
- **Time Complexity**: O(k) where k = number of mask changes
- **Memory Movement**: Optimal - exactly max(A,R) row writes where A=additions, R=removals
- **Cache Efficiency**: 64-byte aligned allocations, contiguous memory access patterns

## Environment Variables

Parallelization controls:
- `WEIGHT_CACHE_GRAIN` (default: 64) - Chunk size for parallel operations
- `WEIGHT_CACHE_PAR_THRESHOLD` (default: 64) - Minimum work items to enable parallelization
- `WEIGHT_CACHE_HYBRID_TAU_FRAC` (default: 0.5) - Hybrid baseline threshold

## Usage Patterns

### C++ API (via PyTorch)
```python
# Initialize cache with sparsity mask
cache = torch.classes.paired.WeightCache(mask, hidden_size, gate_weight, up_weight, down_weight)

# Update with new mask (uses paired replacement)
cache.update_active_weights(new_mask)

# Get optimized tensors (zero-copy on CPU)
active_weights = cache.get_concat_weight()
active_down = cache.get_active_down_weight()
```

### Python Reference
```python
from paired_replacement.src.python.paired_cache import PairedCache, baseline_full_rebuild

cache = PairedCache(mask, up_weight, down_weight, gate_weight)
cache.update_active_weights(new_mask)
```

## Device Support

- **CPU**: Zero-copy tensor views via `torch::from_blob`
- **GPU**: Tensor materialization via `.to(device)` (device-native path planned)
- **ARM64**: SIMD headers guarded, fallback to scalar operations

## Data Types

Currently supports float32 only. Templating for fp16/bf16/int8 is planned.

## Testing Strategy

Benchmarks include:
- Microbenchmarks comparing against `index_select` and boolean masking baselines
- End-to-end MoE-like workloads
- Parameter sweeps across problem sizes and sparsity patterns
- Performance counter analysis (Linux perf)
- Multi-CPU validation for reproducibility