# Installation Guide (How I run it)

Here’s how I install and use the Paired Replacement package locally. I favor editable installs so I can iterate fast and keep imports clean.

## Development Installation

I install in development mode during research:

```bash
# Option 1: Using the provided script
python3 install_dev.py

# Option 2: Manual installation
pip install -e .

# Option 3: With extra dependencies
pip install -e ".[benchmarks,dev]"
```

## Package Structure

After install, I use these clean imports:

```python
# Import the core algorithm
from paired_replacement.src.python.paired_cache import PairedCache, baseline_full_rebuild

# Import benchmark utilities (if needed)
from paired_replacement.benchmarks.scripts import run_microbench_torch
```

## Running Scripts

I typically run the scripts like this:

```bash
# Direct Python execution using package structure
python3 paired_replacement/benchmarks/scripts/run_microbench.py --N 8192 --hidden_dim 1024 --up_cols 1024 --m 1024 --ks 16,64,256,1024

# Run PyTorch extension benchmarks  
python3 paired_replacement/benchmarks/scripts/run_microbench_torch.py --N 8192 --device cpu --outfile results.csv

# Auto-tune parallelization parameters
python3 paired_replacement/benchmarks/scripts/run_autotune.py --N 8192 --outfile autotune.json
```

## Alternative Execution Methods

You can also run scripts this way:

### 1. Using Python Module Execution
```bash
python -m paired_replacement.benchmarks.scripts.run_microbench_torch --help
```

### 2. Direct Script Execution (legacy path)
```bash
python benchmarks/scripts/run_microbench_torch.py --help
```

Note: I recommend the package‑structure method (`python3 paired_replacement/benchmarks/scripts/...`) because it keeps imports consistent.

## Dependencies

### Core (what I actually need)
- `numpy>=1.20.0` - NumPy reference implementation
- `torch>=1.12.0` - PyTorch extension integration

### Optional (what I use for plots/analysis)
- `matplotlib>=3.5.0` - For plotting benchmarks
- `pandas>=1.3.0` - For data analysis
- `pytest>=6.0` - For testing (dev)
- `black>=22.0` - For code formatting (dev)

## Verification

I quickly verify installs like this:

```python
# Test core import
from paired_replacement.src.python.paired_cache import PairedCache
print("✅ Core algorithm import successful")

# Test package version
import paired_replacement
print(f"📦 Package version: {paired_replacement.__version__}")
```

## Development Workflow

1. Install in dev mode: `python install_dev.py`
2. Make changes in `paired_replacement/src/` or `paired_replacement/benchmarks/scripts/`
3. Run scripts from the project root
4. No reinstall needed — changes are live

## Troubleshooting

### Import Errors
If I hit import errors, I:
1. Make sure you've installed the package: `pip install -e .`
2. Check you're running from the project root directory
3. Verify package installation: `pip list | grep paired-replacement`

### Console Scripts Not Found
If console scripts aren’t available, I:
1. Reinstall the package: `pip install -e .`
2. Check your PATH includes pip's script directory
3. Use module execution as fallback: `python -m paired_replacement.benchmarks.scripts.run_microbench`

### Permission Issues
On some systems I need:
```bash
python -m pip install --user -e .
```
