import os
import numpy as np
import pandas as pd
import networkx as nx
import time
import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
import pickle
from tqdm import tqdm
from script.utils.make_edges_orign import mask_edges_det, mask_edges_prd, mask_edges_prd_new_by_marlin
from script.utils.make_edges_new import get_edges, get_prediction_edges, get_prediction_edges_modified, get_new_prediction_edges, get_new_prediction_edges_modified


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def load_vgrnn_dataset(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset(dataset):
    print('>> loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1
    pedges, nedges = get_prediction_edges(undirected_edges)  # list
    new_pedges, new_nedges = get_new_prediction_edges(undirected_edges, num_nodes)

    data['edge_index_list'] = undirected_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> INFO: data: {}'.format(dataset))
    print('>> INFO: total length: {}'.format(len(edge_index_list)))
    print('>> INFO: number nodes: {}'.format(data['num_nodes']))
    return data


def load_vgrnn_dataset_det(dataset):
    assert dataset in ['enron10', 'dblp']  # using vgrnn dataset
    print('>> loading on vgrnn dataset')
    with open('../data/input/raw/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pickle.load(handle, encoding='iso-8859-1')
    print('>> generating edges, negative edges and new edges, wait for a while ...')
    data = {}
    edges, biedges = mask_edges_det(adj_time_list)  # list
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    print('>> processing finished!')
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in range(len(biedges)):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> data: {}'.format(dataset))
    print('>> total length:{}'.format(len(edge_index_list)))
    print('>> number nodes: {}'.format(data['num_nodes']))
    return data


def load_new_dataset_det(dataset):
    print('>> INFO: loading on new dataset')
    data = {}
    rawfile = '../data/input/processed/{}/{}.pt'.format(dataset, dataset)
    edge_index_list = torch.load(rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    undirected_edges = get_edges(edge_index_list)
    num_nodes = int(np.max(np.hstack(undirected_edges))) + 1

    gdata_list = []
    for edge_index in undirected_edges:
        gdata = Data(x=None, edge_index=edge_index, num_nodes=num_nodes)
        gdata_list.append(train_test_split_edges(gdata, 0.1, 0.4))

    data['gdata'] = gdata_list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('>> INFO: data: {}'.format(dataset))
    print('>> INFO: total length: {}'.format(len(edge_index_list)))
    print('>> INFO: number nodes: {}'.format(data['num_nodes']))
    return data


def load_continuous_time_dataset(dataset, neg_sample):
    print("INFO: Loading a continuous-time dataset: {}".format(dataset))
    data = {}
    p_rawfile = '../data/input/continuous_time/{}_pedges_{}.pt'.format(dataset, neg_sample)  # positive edges
    n_rawfile = '../data/input/continuous_time/{}_nedges_{}.pt'.format(dataset, neg_sample)  # negative edges

    # positive edges
    pedge_index_list = torch.load(p_rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    p_undirected_edges = get_edges(pedge_index_list)
    # negative edges
    nedge_index_list = torch.load(n_rawfile)  # format: list:[[[1,2],[2,3],[3,4]]]
    n_undirected_edges = get_edges(nedge_index_list)

    num_nodes = int(np.max(np.hstack(p_undirected_edges))) + 1  # only care about positive number of nodes

    pedges = get_prediction_edges_modified(p_undirected_edges)  # list
    nedges = get_prediction_edges_modified(n_undirected_edges)  # list

    new_pedges = get_new_prediction_edges_modified(p_undirected_edges, num_nodes)
    new_nedges = get_new_prediction_edges_modified(n_undirected_edges, num_nodes)

    data['edge_index_list'] = p_undirected_edges
    data['pedges'], data['nedges'] = pedges, nedges
    data['new_pedges'], data['new_nedges'] = new_pedges, new_nedges  # list
    data['num_nodes'] = num_nodes
    data['time_length'] = len(pedge_index_list)
    data['weights'] = None
    print('>> INFO: Data: {}'.format(dataset))
    print('>> INFO: Total length: {}'.format(len(pedge_index_list)))
    print('>> INFO: Number nodes: {}'.format(data['num_nodes']))
    return data


def load_TGC_dataset(dataset):
    print("INFO: Loading a Graph from `Temporal Graph Classification (TGC)` Category: {}".format(dataset))
    data = {}
    
    # Get current script's absolute path and find project root
    current_file = os.path.abspath(__file__)  # data_util.py absolute path
    current_dir = os.path.dirname(current_file)  # utils/ directory
    
    # Traverse up to find project root (look for GraphPulse directory or go up max 5 levels)
    project_root = None
    search_dir = current_dir
    max_levels = 5
    for level in range(max_levels):
        # Check if this directory contains project root indicators
        if os.path.basename(search_dir).lower() == 'graphpulse' or \
           os.path.exists(os.path.join(search_dir, 'data', 'all_network')) or \
           os.path.exists(os.path.join(search_dir, 'analyzer')):
            project_root = search_dir
            break
        parent = os.path.dirname(search_dir)
        if parent == search_dir:  # Reached filesystem root
            break
        search_dir = parent
    
    # If project root not found, use current script's directory and go up 4 levels as fallback
    if project_root is None:
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
        print("INFO: Project root not found, using fallback: {}".format(project_root))
    else:
        print("INFO: Detected project root: {}".format(project_root))
    
    # Define target filenames to search for
    target_filenames = [
        'network{}.txt'.format(dataset),  # Preprocessing format: networkdgd.txt
        '{}_edgelist.txt'.format(dataset),  # Expected format: dgd_edgelist.txt
    ]
    
    # Directories to exclude (processed/binary data folders)
    excluded_dirs = {'NetworkxGraphs', 'cached', 'checkpoints', '__pycache__', '.git', 
                     'saved_models', 'output', 'PygGraphs', 'Sequences', 'Processed', 
                     'Invalid', 'node_modules', '.pytest_cache'}
    
    def is_text_file(filepath):
        """Check if file is a text file (not binary/pickle)"""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(512)  # Read first 512 bytes
                # Check for binary indicators
                if b'\x00' in chunk:  # Null bytes indicate binary
                    return False
                # Check for pickle magic bytes
                if chunk.startswith(b'\x80') or chunk.startswith(b'PK'):  # Pickle or ZIP
                    return False
                # Try to decode as text
                try:
                    chunk.decode('utf-8', errors='strict')
                    return True
                except UnicodeDecodeError:
                    return False
        except Exception:
            return False
    
    print("INFO: Searching for dataset file with names: {}".format(', '.join(target_filenames)))
    
    # STEP 1: Prioritize specific raw data locations first
    prioritized_paths = [
        os.path.join(project_root, 'data', 'all_network', 'network{}.txt'.format(dataset)),
        os.path.join(project_root, 'data', 'input', 'raw', dataset, '{}_edgelist.txt'.format(dataset)),
        os.path.join(project_root, 'data', 'all_network', 'TimeSeries', 'network{}.txt'.format(dataset)),
    ]
    
    print("INFO: Checking prioritized raw data locations first...")
    edgelist_rawfile = None
    for path in prioritized_paths:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            if is_text_file(abs_path):
                edgelist_rawfile = abs_path
                print("INFO: ✓ Found dataset file in prioritized location: {}".format(edgelist_rawfile))
                break
            else:
                print("INFO: ⚠ Found file at {} but it appears to be binary (skipped)".format(abs_path))
    
    # STEP 2: If not found in prioritized locations, do recursive search (excluding processed folders)
    if edgelist_rawfile is None:
        print("INFO: Not found in prioritized locations. Starting recursive search...")
        print("INFO: Excluding processed folders: {}".format(', '.join(excluded_dirs)))
        
        files_searched = 0
        for root, dirs, files in os.walk(project_root):
            # Exclude processed directories and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_dirs]
            
            # Skip if current directory is in excluded list
            if any(excluded in root for excluded in excluded_dirs):
                dirs[:] = []  # Don't recurse into this branch
                continue
            
            for filename in files:
                files_searched += 1
                if filename in target_filenames:
                    full_path = os.path.join(root, filename)
                    # Make sure it's a file and it's a text file (not binary/pickle)
                    if os.path.isfile(full_path) and is_text_file(full_path):
                        edgelist_rawfile = os.path.abspath(full_path)
                        print("INFO: ✓ Found dataset file at: {}".format(edgelist_rawfile))
                        break
                if files_searched % 1000 == 0:
                    print("INFO: Searched {} files...".format(files_searched))
            
            if edgelist_rawfile is not None:
                break
        
        print("INFO: Searched {} files in total during recursive search".format(files_searched))
    
    if edgelist_rawfile is None:
        raise FileNotFoundError(
            "ERROR: Could not find dataset file for '{}'.\n"
            "Searched for files: {} in project root: {}\n"
            "Checked prioritized locations:\n  {}\n"
            "Please ensure the raw dataset file (text format) exists in data/all_network/ or data/input/raw/".format(
                dataset, ', '.join(target_filenames), project_root,
                '\n  '.join([os.path.abspath(p) for p in prioritized_paths])
            )
        )
    
    # Check if file has expected format (snapshot, source, destination) or raw format (from, to, timestamp, value)
    print("INFO: Attempting to read dataset file...")
    
    # First, try reading as CSV with header (expected format)
    try:
        edgelist_df = pd.read_csv(edgelist_rawfile)
        if 'snapshot' in edgelist_df.columns and 'source' in edgelist_df.columns and 'destination' in edgelist_df.columns:
            print("INFO: Detected expected format (snapshot, source, destination)")
        else:
            raise ValueError("File does not have expected CSV format with 'snapshot', 'source', 'destination' columns")
    except (ValueError, KeyError, pd.errors.EmptyDataError, pd.errors.ParserError):
        # If CSV format fails, try raw format (space-separated, no header)
        print("INFO: CSV format not detected, trying raw format...")
        try:
            edgelist_df = pd.read_csv(edgelist_rawfile, sep=' ', names=["from", "to", "timestamp", "value"])
            
            # Validate that we got valid data (check if timestamp column can be converted)
            try:
                test_timestamp = pd.to_datetime(edgelist_df['timestamp'].iloc[0], unit='s')
            except (ValueError, IndexError):
                raise ValueError("File does not appear to be in raw format (from to timestamp value)")
            
            # Raw format detected - need to convert to snapshot format
            print("INFO: Detected raw format (from, to, timestamp, value). Converting to snapshot format...")
            
            # Convert timestamps to snapshots using daily grouping
            edgelist_df['timestamp'] = pd.to_datetime(edgelist_df['timestamp'], unit='s')
            edgelist_df = edgelist_df.sort_values('timestamp')
            
            # Create daily snapshots (group by date)
            edgelist_df['date'] = edgelist_df['timestamp'].dt.date
            edgelist_df['snapshot'] = edgelist_df.groupby('date').ngroup()
            
            # Rename columns to match expected format
            edgelist_df = edgelist_df.rename(columns={'from': 'source', 'to': 'destination'})
            edgelist_df = edgelist_df[['snapshot', 'source', 'destination']]
            
            print("INFO: Converted to snapshot format. Total snapshots: {}".format(edgelist_df['snapshot'].nunique()))
        except Exception as e:
            raise ValueError(
                "ERROR: Could not parse dataset file '{}'. "
                "Expected format: CSV with columns (snapshot, source, destination) or raw format (from to timestamp value). "
                "Error: {}".format(edgelist_rawfile, str(e))
            )
    
    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    print("INFO: Number of unique snapshots: {}".format(len(uniq_ts_list)))
    adj_time_list = []
    
    # Progress bar for snapshot processing
    print("INFO: Processing {} snapshots...".format(len(uniq_ts_list)))
    for ts in tqdm(uniq_ts_list, desc="Building adjacency matrices", unit="snapshot"):
        # NOTE: this code does not use any node or edge features
        ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination']]
        ts_G = nx.from_pandas_edgelist(ts_edges, 'source', 'destination')
        ts_A = nx.to_scipy_sparse_array(ts_G)
        adj_time_list.append(ts_A)

    # Now, exactly like "load_vgrnn_dataset_det"
    print('INFO: Generating edges, negative edges and new edges, wait for a while ...')
    edge_proc_start = time.time()
    data = {}
    
    # Add progress feedback for expensive negative sampling operations
    print('INFO: Step 1/3: Masking edges (deterministic)...')
    edges, biedges = mask_edges_det(adj_time_list)  # list
    
    print('INFO: Step 2/3: Generating prediction edges (this may take a while)...')
    pedges, nedges = mask_edges_prd(adj_time_list)  # list
    
    print('INFO: Step 3/3: Generating new prediction edges by Marlin (this may take a while)...')
    new_pedges, new_nedges = mask_edges_prd_new_by_marlin(adj_time_list)  # list
    
    print('INFO: Processing finished! Elapsed time (sec.): {:.2f}'.format(time.time() - edge_proc_start))
    assert len(edges) == len(biedges) == len(pedges) == len(nedges) == len(new_nedges) == len(new_pedges)
    
    print('INFO: Converting edge lists to tensors...')
    edge_index_list, pedges_list, nedges_list, new_nedges_list, new_pedges_list = [], [], [], [], []
    for t in tqdm(range(len(biedges)), desc="Converting to tensors", unit="snapshot"):
        edge_index_list.append(torch.tensor(np.transpose(biedges[t]), dtype=torch.long))
        pedges_list.append(torch.tensor(np.transpose(pedges[t]), dtype=torch.long))
        nedges_list.append(torch.tensor(np.transpose(nedges[t]), dtype=torch.long))
        new_pedges_list.append(torch.tensor(np.transpose(new_pedges[t]), dtype=torch.long))
        new_nedges_list.append(torch.tensor(np.transpose(new_nedges[t]), dtype=torch.long))

    data['edge_index_list'] = edge_index_list
    data['pedges'], data['nedges'] = pedges_list, nedges_list
    data['new_pedges'], data['new_nedges'] = new_pedges_list, new_nedges_list  # list
    data['num_nodes'] = int(np.max(np.vstack(edges))) + 1

    data['time_length'] = len(edge_index_list)
    data['weights'] = None
    print('INFO: Data: {}'.format(dataset))
    print('INFO: Total length:{}'.format(len(edge_index_list)))
    print('INFO: Number nodes: {}'.format(data['num_nodes']))
    return data


def loader(dataset='enron10', neg_sample=''):
    # if cached, load directly
    data_root = '../data/input/cached/{}/'.format(dataset)
    filepath = mkdirs(data_root) + '{}.data'.format(dataset)  # the data will be saved here after generation.
    print("INFO: Dataset: {}".format(dataset))
    print("DEBUG: Look for data at {}.".format(filepath))
    if os.path.isfile(filepath):
        print('INFO: Loading {} directly.'.format(dataset))
        return torch.load(filepath)
    
    # if not cached, to process and cached
    print('INFO: data does not exits, processing ...')
    if dataset in ['enron10', 'dblp']:
        data = load_vgrnn_dataset(dataset)
    elif dataset in ['as733', 'fbw', 'HepPh30', 'disease']:
        data = load_new_dataset(dataset)
    elif dataset in ['canVote', 'LegisEdgelist', 'wikipedia', 'UNtrade']:
        print("INFO: Loading a continuous-time dynamic graph dataset: {}".format(dataset))
        data = load_continuous_time_dataset(dataset, neg_sample)
    elif dataset in ['adex', 'aeternity', 'aion', 'aragon', 'bancor', 'centra', 'cindicator', 
                     'coindash', 'dgd', 'iconomi',  'mathoverflow', 'RedditB', 'CollegeMsg']:
        print("INFO: Loading a dynamic graph datasets for TG-Classification: {}".format(dataset))
        data = load_TGC_dataset(dataset)
    else:
        raise ValueError("ERROR: Undefined dataset!")
    torch.save(data, filepath)
    print('INFO: Dataset is saved!')
    return data
