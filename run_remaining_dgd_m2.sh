#!/bin/bash

# =============================================================================
# Script: run_remaining_dgd_m2.sh
# Purpose: Run remaining benchmark models for 'dgd' dataset on Mac M2
# Models: GRUGCN, GIN (Raw), TDA-GIN
# =============================================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root (assuming script is run from project root)
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/models/temporal_gnn/script"
STATIC_GNN_DIR="$PROJECT_ROOT/models/static_gnn"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GraphPulse Remaining Benchmarks (dgd)${NC}"
echo -e "${BLUE}Mac M2 (Apple Silicon)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# =============================================================================
# 1. GRUGCN (Temporal GNN)
# =============================================================================
echo -e "${GREEN}[1/3] Running GRUGCN...${NC}"
cd "$SCRIPT_DIR"
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

echo -e "${GREEN}✓ GRUGCN completed${NC}"
echo ""

# =============================================================================
# 2. TDA-GIN (Static Baseline - TDA Graphs)
# =============================================================================
echo -e "${GREEN}[2/3] Running TDA-GIN...${NC}"
# Ensure GnnResults directory exists (script writes results here)
mkdir -p "$PROJECT_ROOT/GnnResults"
cd "$STATIC_GNN_DIR"

# Create a backup
cp static_graph_methods.py static_graph_methods.py.backup

# Modify for TDA-GIN (dgd only, correct TDA folder)
python3 << 'PYTHON_EOF'
import re
import os

# Get absolute path to static_graph_methods.py
script_dir = os.path.dirname(os.path.abspath('static_graph_methods.py'))

with open('static_graph_methods.py', 'r') as f:
    content = f.read()

# Modify networkList to only include dgd
content = re.sub(
    r'networkList = \[.*?\]',
    'networkList = ["networkdgd.txt"]',
    content,
    flags=re.DOTALL
)

# Update TDA variable to match actual folder name (Overlap_0.3_Ncube_2)
content = re.sub(
    r'read_torch_time_series_data\(network, "Overlap_xx_Ncube_x"\)',
    'read_torch_time_series_data(network, "Overlap_0.3_Ncube_2")',
    content
)

with open('static_graph_methods.py', 'w') as f:
    f.write(content)

print("✓ Modified static_graph_methods.py for TDA-GIN (dgd only)")
PYTHON_EOF

# Run TDA-GIN from project root (paths in script are relative to project root)
# Copy config to project root temporarily (script looks for config_GIN.yml in CWD)
cp "$STATIC_GNN_DIR/config_GIN.yml" "$PROJECT_ROOT/config_GIN.yml"
cd "$PROJECT_ROOT"
python "$STATIC_GNN_DIR/static_graph_methods.py"
# Cleanup
rm -f "$PROJECT_ROOT/config_GIN.yml"

# Restore original
cd "$STATIC_GNN_DIR"
mv static_graph_methods.py.backup static_graph_methods.py

echo -e "${GREEN}✓ TDA-GIN completed${NC}"
echo ""

# =============================================================================
# 3. GIN (Static Baseline - Raw Graphs)
# =============================================================================
echo -e "${GREEN}[3/3] Running GIN (Raw Graphs)...${NC}"
# Ensure GnnResults directory exists (script writes results here)
mkdir -p "$PROJECT_ROOT/GnnResults"
cd "$STATIC_GNN_DIR"

# Create a backup
cp static_graph_methods.py static_graph_methods.py.backup

# Modify for GIN (Raw) - use TemporalVectorizedGraph_Tuned instead of TDA
python3 << 'PYTHON_EOF'
import re

with open('static_graph_methods.py', 'r') as f:
    content = f.read()

# Add helper function to read raw/temporal graphs
helper_function = '''
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

'''

# Insert helper function before the main block
content = content.replace('if __name__ == "__main__":', helper_function + '\nif __name__ == "__main__":')

# Modify networkList to only include dgd
content = re.sub(
    r'networkList = \[.*?\]',
    'networkList = ["networkdgd.txt"]',
    content,
    flags=re.DOTALL
)

# Change to use raw data function instead of TDA
content = re.sub(
    r'data = read_torch_time_series_data\(network, "Overlap_xx_Ncube_x"\)',
    'data = read_torch_time_series_data_raw(network)',
    content
)

with open('static_graph_methods.py', 'w') as f:
    f.write(content)

print("✓ Modified static_graph_methods.py for GIN (Raw) - using TemporalVectorizedGraph_Tuned")
PYTHON_EOF

# Run GIN (Raw) from project root (paths in script are relative to project root)
# Copy config to project root temporarily (script looks for config_GIN.yml in CWD)
cp "$STATIC_GNN_DIR/config_GIN.yml" "$PROJECT_ROOT/config_GIN.yml"
cd "$PROJECT_ROOT"
python "$STATIC_GNN_DIR/static_graph_methods.py"
# Cleanup
rm -f "$PROJECT_ROOT/config_GIN.yml"

# Restore original
cd "$STATIC_GNN_DIR"
mv static_graph_methods.py.backup static_graph_methods.py

echo -e "${GREEN}✓ GIN (Raw) completed${NC}"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All benchmarks completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results location:"
echo "  - GRUGCN: $SCRIPT_DIR/../saved_models/dgd/ and result logs"
echo "  - GIN (Raw): $STATIC_GNN_DIR/GnnResults/GIN_TimeSeries_Result.txt"
echo "  - TDA-GIN: $STATIC_GNN_DIR/GnnResults/GIN_TimeSeries_Result.txt"
echo ""
