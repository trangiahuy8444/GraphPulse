# Analysis: Remaining Benchmark Models for 'dgd' Dataset

## Overview

This document provides analysis and instructions for running the remaining benchmark models to complete the GraphPulse comparison table for the `dgd` dataset on Mac M2.

## Models to Run

1. **GRUGCN** (Temporal GNN)
2. **GIN** (Static Baseline - Raw Graphs)
3. **TDA-GIN** (Static Baseline - TDA-Enhanced Graphs)

---

## 1. GRUGCN (Temporal GNN)

### Location
- **Model Definition**: `models/temporal_gnn/script/models/DynModels.py` (class `DGCN`)
- **Model Loader**: `models/temporal_gnn/script/models/load_model.py` (line 9-10)
- **Main Script**: `models/temporal_gnn/script/main.py`

### Command Structure
GRUGCN uses the same command structure as HTGN and EvolveGCN:

```bash
cd models/temporal_gnn/script
python main.py \
    --dataset dgd \
    --model GRUGCN \
    --device cpu \
    --device_id -1 \
    --seed 1024 \
    --max_epoch 500 \
    --patience 50 \
    --lr 0.01 \
    --nfeat 128 \
    --nhid 16 \
    --nout 16
```

### Notes
- GRUGCN is registered in `load_model.py` as `args.model in ['GRUGCN', 'DynGCN']`
- Uses the same data loading pipeline as HTGN (from `data_util.py`)
- Results saved to `../saved_models/dgd/` and result logs

---

## 2. GIN (Static Baseline - Raw Graphs)

### Location
- **Model Definition**: `models/static_gnn/static_graph_methods.py` (class `GIN`)
- **Main Script**: `models/static_gnn/static_graph_methods.py`

### Current Implementation
The script currently:
- Hardcodes a `networkList` with multiple datasets (line 259-261)
- Reads data from `PygGraphs/TimeSeries/{network}/TDA_Tuned/{variable}/` (line 265)
- Uses TDA-enhanced graphs by default

### Required Modifications for GIN (Raw)

**Option A: Modify `read_torch_time_series_data` function**

The function `read_torch_time_series_data(network, variable)` currently reads from:
```python
file_path_different_TDA_Tuned = "PygGraphs/TimeSeries/{}/TDA_Tuned/{}/".format(network, variable)
```

For **GIN (Raw)**, we need to read from raw graphs instead. Based on the directory structure:
- `PygGraphs/TimeSeries/networkdgd.txt/TemporalVectorizedGraph_Tuned/` contains raw-like graph snapshots
- `PygGraphs/TimeSeries/networkdgd.txt/RawGraph/` may not exist

**Solution**: Create a helper function or modify the main block:

```python
def read_torch_time_series_data_raw(network):
    """Read raw graph data (without TDA features) for GIN baseline"""
    file_path_temporal = "PygGraphs/TimeSeries/{}/TemporalVectorizedGraph_Tuned/".format(network)
    GraphDataList = []
    import os
    import pickle
    
    if os.path.exists(file_path_temporal):
        files = sorted([f for f in os.listdir(file_path_temporal) if f.endswith(('.txt', '.pkl'))])
        for file in files:
            with open(file_path_temporal + file, 'rb') as f:
                data = pickle.load(f)
                GraphDataList.append(data)
    else:
        raise FileNotFoundError(f"TemporalVectorizedGraph_Tuned not found for {network}")
    return GraphDataList
```

Then in `__main__`, change:
```python
# OLD:
data = read_torch_time_series_data(network, "Overlap_xx_Ncube_x")

# NEW (for GIN Raw):
data = read_torch_time_series_data_raw(network)
```

**Option B: Manual Edit Instructions**

1. Open `models/static_gnn/static_graph_methods.py`
2. At line 258, modify `networkList` to:
   ```python
   networkList = ["networkdgd.txt"]
   ```
3. Add the helper function `read_torch_time_series_data_raw` (see Option A above) before `if __name__ == "__main__":`
4. At line 265, change:
   ```python
   data = read_torch_time_series_data_raw(network)  # Instead of read_torch_time_series_data(...)
   ```

### Command
```bash
cd models/static_gnn
python static_graph_methods.py
```

### Results
- Saved to: `models/static_gnn/GnnResults/GIN_TimeSeries_Result.txt`
- Format: Network, Duplicate, Epoch, Train/Test Accuracy, AUC scores

---

## 3. TDA-GIN (Static Baseline - TDA-Enhanced Graphs)

### Location
- **Model Definition**: Same as GIN (`models/static_gnn/static_graph_methods.py`)
- **Main Script**: Same as GIN

### Current Implementation
The script reads from:
```python
file_path_different_TDA_Tuned = "PygGraphs/TimeSeries/{}/TDA_Tuned/{}/".format(network, variable)
```

Where `variable = "Overlap_xx_Ncube_x"` (line 265).

### Required Modifications for TDA-GIN

**Issue**: The hardcoded variable `"Overlap_xx_Ncube_x"` doesn't match the actual folder name.

**Actual folder structure**:
```
PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/Overlap_0.3_Ncube_2/
```

**Solution**: Update line 265 to use the correct variable name:

```python
# OLD:
data = read_torch_time_series_data(network, "Overlap_xx_Ncube_x")

# NEW:
data = read_torch_time_series_data(network, "Overlap_0.3_Ncube_2")
```

Also modify `networkList` to only include `dgd`:
```python
networkList = ["networkdgd.txt"]
```

### Command
```bash
cd models/static_gnn
python static_graph_methods.py
```

### Results
- Saved to: `models/static_gnn/GnnResults/GIN_TimeSeries_Result.txt`
- Format: Network, Duplicate, Epoch, Train/Test Accuracy, AUC scores

---

## Automated Script

A shell script `run_remaining_dgd_m2.sh` has been created to automate all three models:

```bash
# Make executable (if needed)
chmod +x run_remaining_dgd_m2.sh

# Run all remaining benchmarks
./run_remaining_dgd_m2.sh
```

The script:
1. Runs GRUGCN using `main.py` with appropriate arguments
2. Automatically modifies `static_graph_methods.py` for TDA-GIN, runs it, then restores the original
3. Automatically modifies `static_graph_methods.py` for GIN (Raw), runs it, then restores the original

---

## Summary Table

| Model | Type | Script | Data Source | Modifications Required |
|-------|------|--------|-------------|------------------------|
| **GRUGCN** | Temporal GNN | `main.py` | `data/all_network/networkdgd.txt` | None (command-line args) |
| **GIN (Raw)** | Static Baseline | `static_graph_methods.py` | `TemporalVectorizedGraph_Tuned/` | Modify data reading function |
| **TDA-GIN** | Static Baseline | `static_graph_methods.py` | `TDA_Tuned/Overlap_0.3_Ncube_2/` | Update variable name + networkList |

---

## Expected Results Format

All models should output metrics compatible with the GraphPulse paper tables:
- **ROC-AUC** (primary metric)
- **Accuracy**
- **Train/Test splits** (chronological 80-20)

Results will be logged to:
- GRUGCN: `models/temporal_gnn/script/result_txt/` (or similar)
- GIN/TDA-GIN: `models/static_gnn/GnnResults/GIN_TimeSeries_Result.txt`

---

## Troubleshooting

### Issue: `FileNotFoundError` for TDA folder
**Solution**: Verify the actual TDA folder name:
```bash
ls PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/
```
Update the variable name in `static_graph_methods.py` accordingly.

### Issue: `FileNotFoundError` for RawGraph
**Solution**: Use `TemporalVectorizedGraph_Tuned/` as the raw data source (as implemented in the script).

### Issue: Results overwriting each other
**Solution**: The script appends to result files. For separate runs, consider renaming output files or using timestamps.

---

## Next Steps

After running all benchmarks:
1. Collect results from all three models
2. Compare with paper tables (Table 1, 2, 3)
3. Document any discrepancies (hardware differences, random seed, etc.)
