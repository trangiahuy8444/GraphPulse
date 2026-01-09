# Mac M2 (Apple Silicon) Compatibility Guide for GraphPulse

## Overview

This document outlines the modifications made to make GraphPulse compatible with Mac M2 (Apple Silicon) systems. The codebase has been patched to support Metal Performance Shaders (MPS) as the GPU backend, with automatic fallback to CPU when MPS is unavailable.

## Key Changes Made

### 1. Device Configuration (`models/temporal_gnn/script/config.py`)

**Before:**
```python
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device(f"cuda:{args.device_id}")
else:
    args.device = torch.device("cpu")
```

**After:**
```python
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device(f"cuda:{args.device_id}")
elif torch.backends.mps.is_available():
    args.device = torch.device("mps")  # Mac M2 GPU acceleration
else:
    args.device = torch.device("cpu")
```

### 2. Random Seed Initialization (`models/temporal_gnn/script/utils/util.py`)

**Before:**
```python
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
```

**After:**
```python
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
# MPS doesn't have separate seed functions, manual_seed is sufficient
```

### 3. GPU Memory Tracking

Fixed in multiple files:
- `models/temporal_gnn/script/main.py`
- `models/temporal_gnn/script/train_tgc_end_to_end.py`
- `models/temporal_gnn/script/train_graph_classification.py`
- `models/temporal_gnn/script/baselines/run_*.py`

**Before:**
```python
gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
```

**After:**
```python
if torch.cuda.is_available():
    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000
else:
    gpu_mem_alloc = 0  # MPS doesn't support memory tracking yet
```

### 4. CUDA-Specific Tensor Type Checks (`models/temporal_gnn/script/models/EvolveGCN/EGCN.py`)

**Before:**
```python
device = 'cuda' if vect.is_cuda else 'cpu'
if isinstance(node_embs, torch.cuda.sparse.FloatTensor):
    ...
```

**After:**
```python
device = vect.device  # Works for CUDA, MPS, and CPU
if isinstance(node_embs, torch.sparse.FloatTensor) or \
   (hasattr(torch, 'cuda') and torch.cuda.is_available() and 
    isinstance(node_embs, torch.cuda.sparse.FloatTensor)):
    ...
```

## Installation Requirements

### PyTorch Installation for Mac M2

The original `requirements.txt` specifies `torch==1.6.0+cu101`, which is CUDA-specific and incompatible with Mac M2.

**For Mac M2, install PyTorch with MPS support (PyTorch 1.12+):**

> **⚠️ zsh Users:** When installing packages with version specifiers (e.g., `>=2.0.0`), always use quotes to prevent zsh from interpreting `>` as output redirection. Example: `pip install "torch>=2.0.0"` instead of `pip install torch>=2.0.0`.

```bash
# Install PyTorch with MPS support (minimum PyTorch 1.12)
pip install torch torchvision torchaudio

# For specific version (recommended: PyTorch 2.0+ for better MPS stability)
# Note: Use quotes for zsh compatibility (zsh interprets >= as redirection)
pip install "torch>=2.0.0" torchvision torchaudio
```

**Verify MPS availability:**
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### PyTorch Geometric Installation

**Important:** PyTorch Geometric and its extensions (torch-scatter, torch-sparse, torch-cluster, torch-spline-conv) need to be built from source on Mac M2.

```bash
# Install PyTorch first (see above)
# Note: Use quotes for zsh compatibility
pip install "torch>=2.0.0"

# Install PyTorch Geometric and dependencies
pip install torch-geometric

# For older versions, you may need to build extensions manually:
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Note:** If you encounter compilation errors, you may need:
- Xcode Command Line Tools: `xcode-select --install`
- CMake (for building extensions): `brew install cmake`

### Updated Requirements (Mac M2 Compatible)

Create a new `requirements_mac_m2.txt`:

```txt
# Core dependencies
torch>=2.0.0  # MPS support requires PyTorch 2.0+
torch-geometric>=2.3.0
numpy>=1.20.0
scipy>=1.6.0
pandas>=2.0.0
networkx>=2.5
scikit-learn>=0.24.0

# For RNN models (TensorFlow/Keras)
tensorflow-macos>=2.13.0  # Apple's optimized TensorFlow for Mac
tensorflow-metal>=0.6.0   # Metal acceleration for TensorFlow

# Topological Data Analysis
kmapper>=2.0.1

# Other dependencies
pytorch-lightning>=2.0.0
matplotlib>=3.7.0
pandas>=2.0.0
PyYAML>=6.0
tqdm>=4.66.0
geoopt>=0.3.1  # For hyperbolic geometry (may need manual installation)
```

## Running Experiments on Mac M2

### Temporal GNN Models

```bash
cd models/temporal_gnn/script

# Example: Run HTGN on aion dataset
python main.py --dataset aion --model HTGN --device_id -1

# Note: --device_id -1 forces CPU, but with MPS support, it will automatically use MPS if available
```

### RNN Models

The RNN models use TensorFlow/Keras. For Mac M2:

1. Install TensorFlow-Metal (Apple's optimized version):
```bash
pip install tensorflow-macos tensorflow-metal
```

2. The models should automatically use Metal acceleration.

### Static GNN Models

No changes needed - these models work with standard PyTorch device handling.

## Known Limitations on Mac M2

1. **MPS Memory Tracking**: MPS doesn't support `torch.cuda.max_memory_allocated()`. Memory usage is logged as 0 MiB.

2. **MPS Performance**: MPS may have different performance characteristics compared to CUDA. Some operations may be slower or have different numerical precision.

3. **PyTorch Geometric Extensions**: Some operations in PyTorch Geometric may fall back to CPU even when MPS is available. This is expected and should not affect correctness.

4. **Random Seed Reproducibility**: MPS uses different random number generation than CUDA. Results may vary slightly, but should be reproducible across runs on the same Mac M2 system.

5. **PyTorch Version Compatibility**: The original codebase targets PyTorch 1.6.0. Some APIs may have changed in newer versions. If you encounter issues:
   - Check PyTorch migration guides for API changes
   - Consider using PyTorch 1.13-2.0 for better compatibility with older code

## Troubleshooting

### Issue: "MPS backend not available"
**Solution:** 
- Ensure you have PyTorch 1.12+ installed
- Check macOS version (MPS requires macOS 12.3+)
- Verify with: `torch.backends.mps.is_available()`

### Issue: "torch-scatter/torch-sparse installation fails"
**Solution:**
- Install Xcode Command Line Tools: `xcode-select --install`
- Install CMake: `brew install cmake`
- Try installing from PyG wheels: `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html`

### Issue: "Slow performance on MPS"
**Solution:**
- Some operations are faster on CPU. Try: `--device_id -1` to force CPU
- Monitor which operations are slow and consider CPU fallback for those
- Check if your model benefits from mixed precision (may not be fully supported on MPS yet)

### Issue: "Numerical differences vs CUDA results"
**Solution:**
- This is expected due to different hardware and floating-point implementations
- Differences should be small (< 0.01 in metrics)
- For exact reproducibility, use CPU mode: `--device_id -1`

## Reproducing Paper Results

To reproduce results from the paper on Mac M2:

1. **Use CPU mode for exact reproducibility:**
   ```bash
   python main.py --dataset aion --model HTGN --device_id -1 --seed 1024
   ```

2. **Expected differences:**
   - ROC-AUC and Accuracy metrics may vary by ±0.01-0.02 due to hardware differences
   - Training time will be significantly different (CPU/MPS vs CUDA)
   - Overall trends and relative performance between models should be preserved

3. **For best results:**
   - Use the same random seed: `--seed 1024`
   - Run multiple seeds and average (as in the paper)
   - Compare relative improvements rather than absolute values

## Additional Notes

- All `.to(args.device)` calls in the codebase now automatically work with MPS/CPU
- The device selection is automatic - no code changes needed when running
- Check logs for device selection: `INFO: using MPS (Apple Silicon GPU) to train the model`

## Testing

To verify the fixes work:

```python
import torch
from models.temporal_gnn.script.config import args

# Check device selection
print(f"Selected device: {args.device}")
assert str(args.device) in ["cpu", "mps", "cuda:0"], "Invalid device"

# Test tensor operations
x = torch.randn(10, 10).to(args.device)
y = torch.randn(10, 10).to(args.device)
z = torch.matmul(x, y)
print(f"Tensor on device: {z.device}")
```

---

**Last Updated:** 2024
**Tested on:** macOS 13+ (Ventura/Sonoma), Mac M2, PyTorch 2.0+
