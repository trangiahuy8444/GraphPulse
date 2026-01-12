# config_GIN.yml Setup Guide

## Issue Summary

The `static_graph_methods.py` script expects `config_GIN.yml` to be in the **current working directory**, but when running from project root (after path fixes), the file is in `models/static_gnn/`.

## Solution Applied

The `run_remaining_dgd_m2.sh` script now:
1. Copies `config_GIN.yml` from `models/static_gnn/` to project root temporarily
2. Runs the Python script from project root
3. Cleans up the copied config file afterward

## Config File Location

**Original Location**: `models/static_gnn/config_GIN.yml`

**Content**: The file already exists and contains the required configuration:
- `hidden_units`: `[[64, 64, 64, 64]]` (4 layers with 64 units each)
- `dropout`: `[0.5]`
- `train_eps`: `[true]`
- `aggregation`: `[mean]` (and `[sum]` as alternative)

## Config File Structure

The config file uses a list format because the code accesses elements with `[0]`:

```python
self.dropout = config['dropout'][0]          # Takes first element
self.embeddings_dim = [config['hidden_units'][0][0]] + config['hidden_units'][0]
train_eps = config['train_eps'][0]           # Takes first element
if config['aggregation'][0] == 'sum':        # Takes first element
```

### Required Fields

Based on code analysis (`static_graph_methods.py` lines 30-42):

1. **`hidden_units`**: List of lists
   - Format: `[[64, 64, 64, 64]]` for 4 layers with 64 units each
   - Used: `config['hidden_units'][0]` gives `[64, 64, 64, 64]`

2. **`dropout`**: List
   - Format: `[0.5]`
   - Used: `config['dropout'][0]` gives `0.5`

3. **`train_eps`**: List
   - Format: `[true]` or `[false]`
   - Used: `config['train_eps'][0]` gives boolean

4. **`aggregation`**: List
   - Format: `['mean']` or `['sum']`
   - Used: `config['aggregation'][0]` gives string

### Paper Specifications (Section 6)

According to GraphPulse paper:
- **Hidden units**: 64 (4 layers implied by "four middle layers")
- **Learning rate**: 0.0001 (1×10⁻⁴) - *Note: Hardcoded in code, not from config*
- **Optimizer**: Adam - *Note: Hardcoded in code, not from config*

The existing `config_GIN.yml` matches these specifications:
- `hidden_units: [[64, 64, 64, 64]]` ✅
- Learning rate is hardcoded to `0.0001` in line 165 of `static_graph_methods.py`
- Optimizer is hardcoded to `Adam` in line 165

## Minimal Config File

If you need a minimal config file, use:

```yaml
hidden_units:
  - [64, 64, 64, 64]

dropout:
  - 0.5

train_eps:
  - true

aggregation:
  - mean
```

**Note**: The existing `config_GIN.yml` has additional fields (model, device, batch_size, etc.) that are not used by the code but don't cause issues.

## Verification

To verify the config file is correct:

```bash
# Check if file exists
ls -la models/static_gnn/config_GIN.yml

# Validate YAML syntax (if you have yq or python-yaml installed)
python3 -c "import yaml; yaml.safe_load(open('models/static_gnn/config_GIN.yml'))"
```

## Manual Fix (If Needed)

If you need to create the config file manually:

```bash
# From project root
cat > models/static_gnn/config_GIN.yml << 'EOF'
hidden_units:
  - [64, 64, 64, 64]
dropout:
  - 0.5
train_eps:
  - true
aggregation:
  - mean
EOF
```

## Alternative Solution (Code Modification)

For a cleaner solution, you could modify `static_graph_methods.py` to use an absolute path:

```python
# Line 152: Change from
with open("config_GIN.yml", "r") as f:

# To:
import os
config_path = os.path.join(os.path.dirname(__file__), "config_GIN.yml")
with open(config_path, "r") as f:
```

This would allow the script to find the config file regardless of the working directory.

However, the current solution (copying the file temporarily) is simpler and doesn't require code changes.
