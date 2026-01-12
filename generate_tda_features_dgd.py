#!/usr/bin/env python3
"""
Script to generate TDA features for 'dgd' dataset with specific parameters.
This script specifically generates TDA_Tuned folder with Overlap_0.3_Ncube_2.

Usage:
    python generate_tda_features_dgd.py
"""

import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from analyzer.network_parser import NetworkParser

def main():
    """Generate TDA features for networkdgd.txt dataset"""
    
    print("=" * 60)
    print("TDA Features Generator for 'dgd' Dataset")
    print("Parameters: overlap=0.3, n_cubes=2")
    print("=" * 60)
    print()
    
    # Initialize parser
    parser = NetworkParser()
    
    # Configure paths (adjust if needed)
    parser.file_path = "./data/all_network/"
    parser.timeseries_file_path = "./data/all_network/TimeSeries/"
    
    network_name = "networkdgd.txt"
    
    # Check if time series file exists
    timeseries_file = os.path.join(parser.timeseries_file_path, network_name)
    if not os.path.exists(timeseries_file):
        print(f"ERROR: Time series file not found: {timeseries_file}")
        print(f"Please ensure you have run data preprocessing first.")
        print(f"Expected location: {os.path.abspath(timeseries_file)}")
        return 1
    
    print(f"✓ Found time series file: {timeseries_file}")
    print()
    
    # Check if graph features have been created
    networkx_graph_dir = "NetworkxGraphs"
    if not os.path.exists(networkx_graph_dir):
        print(f"WARNING: NetworkxGraphs directory not found.")
        print(f"This suggests create_graph_features() has not been run.")
        print(f"Proceeding anyway, but TDA generation requires graph features.")
        print()
    
    # Target directory that will be created
    target_dir = f"PygGraphs/TimeSeries/{network_name}/TDA_Tuned/Overlap_0.3_Ncube_2/"
    print(f"Target output directory: {os.path.abspath(target_dir)}")
    print()
    
    print("Starting TDA feature generation...")
    print("NOTE: This process calls create_time_series_graphs() which:")
    print("  - Processes all time windows (can take a while)")
    print("  - Generates TDA graphs with overlap=0.3, n_cubes=2")
    print("  - Saves to TDA_Tuned/Overlap_0.3_Ncube_2/")
    print()
    
    try:
        # This function internally calls create_TDA_graph with the correct parameters
        parser.create_time_series_graphs(network_name)
        
        # Verify output was created
        if os.path.exists(target_dir):
            file_count = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
            print()
            print("=" * 60)
            print(f"✓ SUCCESS: TDA features generated!")
            print(f"  Directory: {os.path.abspath(target_dir)}")
            print(f"  Files created: {file_count}")
            print("=" * 60)
            return 0
        else:
            print()
            print("=" * 60)
            print("WARNING: Process completed but target directory not found.")
            print(f"  Expected: {os.path.abspath(target_dir)}")
            print("  Check error messages above for issues.")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR: TDA feature generation failed!")
        print(f"Error: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
