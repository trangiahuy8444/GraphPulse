# TDA Features Generation Guide

## Issue Summary

The `TDA-GIN` model failed with a `FileNotFoundError` because `static_graph_methods.py` uses **relative paths** that are expected to be run from the **project root**, but the script was running from `models/static_gnn/`.

## Root Cause

1. **Directory EXISTS**: The TDA features directory `PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/Overlap_0.3_Ncube_2/` already exists (722 files confirmed).

2. **Path Resolution Issue**: `static_graph_methods.py` uses relative paths like:
   ```python
   file_path_different_TDA_Tuned = "PygGraphs/TimeSeries/{}/TDA_Tuned/{}/".format(network, variable)
   ```
   These paths are relative to the **project root**, not `models/static_gnn/`.

3. **Working Directory Mismatch**: The script `run_remaining_dgd_m2.sh` changed to `models/static_gnn/` before running, causing path resolution to fail.

## Solution Applied

The script has been updated to **change to project root** before running `static_graph_methods.py`:

```bash
# Run TDA-GIN from project root (paths in script are relative to project root)
cd "$PROJECT_ROOT"
python "$STATIC_GNN_DIR/static_graph_methods.py"
```

## TDA Feature Generation (If Needed)

If you need to **regenerate** TDA features (e.g., if they don't exist or are corrupted), use the provided script:

### Script: `generate_tda_features_dgd.py`

**Purpose**: Generates TDA features specifically for the `dgd` dataset with parameters `overlap=0.3` and `n_cubes=2`.

**Usage**:
```bash
# From project root
python generate_tda_features_dgd.py
```

**What it does**:
1. Calls `parser.create_time_series_graphs("networkdgd.txt")`
2. Internally calls `create_TDA_graph()` with hardcoded parameters:
   - `per_overlap = [0.3]`
   - `n_cubes = [2]`
3. Creates directory: `PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/Overlap_0.3_Ncube_2/`
4. Generates PyTorch Geometric data objects for each time window

**Note**: This process can take a while (potentially hours) as it processes all time windows (720+ snapshots for dgd dataset).

### When to Regenerate

You should **NOT** need to regenerate if:
- ✅ The directory `PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/Overlap_0.3_Ncube_2/` exists
- ✅ It contains files (check with `ls -la`)

You **SHOULD** regenerate if:
- ❌ The directory doesn't exist
- ❌ The directory is empty
- ❌ You need different TDA parameters
- ❌ The preprocessing was incomplete

## How TDA Features Are Generated

### Code Location

**Main Function**: `analyzer/network_parser.py`
- **Method**: `create_TDA_graph(self, data, label, htmlPath="", timeWindow=0, network="")`
- **Line**: 1150-1202

**Called From**: `create_time_series_graphs()`
- **Method**: `analyzer/network_parser.py::create_time_series_graphs()`
- **Line**: 306

### Parameters (Hardcoded)

```python
per_overlap = [0.3]  # Line 1152
n_cubes = [2]        # Line 1153
cls = 2              # Line 1162 (KMeans clusters)
```

### Process Flow

1. **Node Features Extraction**: Extract node features from graph snapshots
2. **TDA Mapper**: Apply KeplerMapper with T-SNE projection
3. **Clustering**: Use KMeans clustering (2 clusters)
4. **Cover Generation**: Create cover with `n_cubes=2`, `perc_overlap=0.3`
5. **Graph Construction**: Build TDA graph from mapper output
6. **Conversion**: Convert to PyTorch Geometric Data object
7. **Save**: Save to `PygGraphs/TimeSeries/{network}/TDA_Tuned/Overlap_{overlap}_Ncube_{n_cube}/`

### Output Format

- **Directory**: `PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/Overlap_0.3_Ncube_2/`
- **Files**: `networkdgd.txt_TDA_graph(cube-2,overlap-0.3)_{timeWindow}`
- **Format**: Pickle files containing PyTorch Geometric `Data` objects
- **Features**: `cluster_size` (node attribute)

## Verification

To verify TDA features exist:

```bash
# From project root
ls -la PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/Overlap_0.3_Ncube_2/

# Count files
ls PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/Overlap_0.3_Ncube_2/ | wc -l
```

Expected: 720 files (one per time window for dgd dataset)

## Updated Script Behavior

The `run_remaining_dgd_m2.sh` script now:
1. ✅ Changes to project root before running `static_graph_methods.py`
2. ✅ Uses absolute path to the script: `python "$STATIC_GNN_DIR/static_graph_methods.py"`
3. ✅ Ensures paths resolve correctly

## Next Steps

1. **Re-run the benchmark script**:
   ```bash
   ./run_remaining_dgd_m2.sh
   ```

2. **If TDA-GIN still fails**, check:
   - Working directory is project root
   - TDA directory exists and has files
   - Python path is correct

3. **If regeneration is needed**:
   ```bash
   python generate_tda_features_dgd.py
   ```
   (This may take several hours)
