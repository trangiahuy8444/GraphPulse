import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def mask_edges_prd(adjs_list):
    pos_edges_l, false_edges_l = [], []
    edges_list = []
    # Progress bar for processing snapshots
    for i in tqdm(range(0, len(adjs_list)), desc="mask_edges_prd", unit="snapshot"):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_false = int(edges.shape[0])

        pos_edges_l.append(edges)

        # Optimized ismember using set for faster lookups
        def ismember_set(edge_tuple, edge_set):
            """Check if edge is in set (much faster than array comparison)"""
            return edge_tuple in edge_set
        
        # Convert edges_all to set for O(1) lookup
        edges_all_set = set(map(tuple, edges_all))
        edges_false_set = set()
        
        edges_false = []
        max_attempts = num_false * 100  # Safety limit to prevent infinite loops
        attempts = 0
        
        # Progress bar for negative sampling (only show if it's taking a while)
        pbar = None
        if num_false > 100:  # Only show progress bar for larger samples
            pbar = tqdm(total=num_false, desc=f"  Negative sampling (snapshot {i+1}/{len(adjs_list)})", 
                       unit="edge", leave=False)
        
        while len(edges_false) < num_false and attempts < max_attempts:
            attempts += 1
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            
            edge_tuple = (idx_i, idx_j)
            # Fast set-based lookup instead of slow array comparison
            if edge_tuple in edges_all_set or edge_tuple in edges_false_set:
                continue
            
            edges_false.append([idx_i, idx_j])
            edges_false_set.add(edge_tuple)
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        if len(edges_false) < num_false:
            print(f"WARNING: Only sampled {len(edges_false)}/{num_false} negative edges after {max_attempts} attempts. Graph may be too dense.")

        # Verify no false edges are in positive edges (using set for speed)
        edges_false_array = np.asarray(edges_false)
        for edge in edges_false_array:
            assert tuple(edge) not in edges_all_set, f"False edge {edge} found in positive edges!"

        false_edges_l.append(edges_false_array)

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l


def test_adj(adjs_list, adj_orig_dense_list):
    # this method is to test the adj_list and adj_orig_dense_list is same or not
    for i, a in enumerate(adj_orig_dense_list):
        a = sp.csr_matrix(a)
        a = a - sp.dia_matrix((a.diagonal()[np.newaxis, :], [0]), shape=a.shape)
        a.eliminate_zeros()
        assert np.diag(a.todense()).sum() == 0
        team1 = sp.csr_matrix(a).todok().tocoo()
        print(len(list(team1.col.reshape(-1))))

        adj = adjs_list[i]
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        assert np.diag(adj.todense()).sum() == 0
        team2 = sp.csr_matrix(adj).todok().tocoo()

        print(len(list(team2.col.reshape(-1))))
        print('==')


def mask_edges_prd_new(adjs_list, adj_orig_dense_list):
    # produce new edge index
    pos_edges_l, false_edges_l = [], []

    # the first snapshots
    adj = adjs_list[0]
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    edges = sparse_to_tuple(adj_triu)[0]  # single direction
    edges_all = sparse_to_tuple(adj)[0]  # all

    pos_edges_l.append(edges)

    num_false = int(edges.shape[0])

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edges_false = []
    while len(edges_false) < num_false:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if edges_false:
            if ismember([idx_j, idx_i], np.array(edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(edges_false)):
                continue
        edges_false.append([idx_i, idx_j])

    assert ~ismember(edges_false, edges_all)
    false_edges_l.append(np.asarray(edges_false))

    # the next snapshots
    for i in range(1, len(adjs_list)):
        edges_pos = np.transpose(np.asarray(np.where((adj_orig_dense_list[i] - adj_orig_dense_list[i - 1]) > 0)))
        num_false = int(edges_pos.shape[0])
        adj = adjs_list[i]  # current adj
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        assert np.diag(adj.todense()).sum() == 0

        edges_all = sparse_to_tuple(adj)[0]

        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:  # filter self-loop
                continue
            if ismember([idx_i, idx_j], edges_all):  # filter old edges
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])

        assert ~ismember(edges_false, edges_all)

        false_edges_l.append(np.asarray(edges_false))
        pos_edges_l.append(edges_pos)

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l


def mask_edges_prd_new_by_marlin(adjs_list):
    # This code is same with the previous one but only need to one spare adj matrix

    pos_edges_l, false_edges_l = [], []

    # 1. the first snapshots
    adj = adjs_list[0]
    # 1.1 Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # 1.2 Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    edges = sparse_to_tuple(adj_triu)[0]  # single direction
    edges_all = sparse_to_tuple(adj)[0]  # all

    pos_edges_l.append(edges)

    num_false = int(edges.shape[0])

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    # 1.3 negative sampling (optimized with set-based lookup)
    edges_all_set = set(map(tuple, edges_all))
    edges_false_set = set()
    edges_false = []
    max_attempts = num_false * 100
    attempts = 0
    
    pbar = None
    if num_false > 100:
        pbar = tqdm(total=num_false, desc="  Negative sampling (first snapshot)", unit="edge", leave=False)
    
    while len(edges_false) < num_false and attempts < max_attempts:
        attempts += 1
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        edge_tuple = (idx_i, idx_j)
        if edge_tuple in edges_all_set or edge_tuple in edges_false_set:
            continue
        edges_false.append([idx_i, idx_j])
        edges_false_set.add(edge_tuple)
        if pbar:
            pbar.update(1)
    
    if pbar:
        pbar.close()
    
    if len(edges_false) < num_false:
        print(f"WARNING: Only sampled {len(edges_false)}/{num_false} negative edges after {max_attempts} attempts.")
    
    # Verify
    for edge in edges_false:
        assert tuple(edge) not in edges_all_set
    false_edges_l.append(np.asarray(edges_false))

    # 2. the next snapshots (with progress bar)
    for i in tqdm(range(1, len(adjs_list)), desc="mask_edges_prd_new_by_marlin", unit="snapshot"):
        # 2.1 get new edge_index
        edges = sparse_to_tuple(adjs_list[i])[0]  # current edges
        last_edges = sparse_to_tuple(adjs_list[i - 1])[0]  # last edges
        edges_perm = edges[:, 0] * 1e5 + edges[:, 1]  # hash current edges
        last_edges_perm = last_edges[:, 0] * 1e5 + last_edges[:, 1]  # hash last edges
        perm = np.setdiff1d(edges_perm, np.intersect1d(edges_perm, last_edges_perm))  # new edges: edge-edge^last_edge
        edges_pos = np.vstack(np.divmod(perm, 1e5)).transpose().astype(np.longlong)  # convert perm to indices
        num_false = int(edges_pos.shape[0])

        # 2.2 get all pos edge to avoid being sampled
        adj = adjs_list[i]  # current adj
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        assert np.diag(adj.todense()).sum() == 0
        edges_all = sparse_to_tuple(adj)[0]

        # 2.3 sample equal size of neg edges (optimized with set-based lookup)
        edges_all_set = set(map(tuple, edges_all))
        edges_false_set = set()
        edges_false = []
        max_attempts = num_false * 100
        attempts = 0
        
        pbar = None
        if num_false > 100:
            pbar = tqdm(total=num_false, desc=f"  Negative sampling (snapshot {i+1}/{len(adjs_list)})", 
                       unit="edge", leave=False)
        
        while len(edges_false) < num_false and attempts < max_attempts:
            attempts += 1
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:  # filter self-loop
                continue
            edge_tuple = (idx_i, idx_j)
            if edge_tuple in edges_all_set or edge_tuple in edges_false_set:
                continue
            edges_false.append([idx_i, idx_j])
            edges_false_set.add(edge_tuple)
            if pbar:
                pbar.update(1)
        
        if pbar:
            pbar.close()
        
        if len(edges_false) < num_false:
            print(f"WARNING: Only sampled {len(edges_false)}/{num_false} negative edges after {max_attempts} attempts.")
        
        # Verify
        for edge in edges_false:
            assert tuple(edge) not in edges_all_set
        
        false_edges_l.append(np.asarray(edges_false))
        pos_edges_l.append(edges_pos)

    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l


def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))
    return out


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_edges_det(adjs_list):
    '''
    produce edge_index in np format
    '''
    edges_list = []
    biedges_list = []
    # Progress bar for deterministic edge masking
    for i in tqdm(range(0, len(adjs_list)), desc="mask_edges_det", unit="snapshot"):
        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        edges = sparse_to_tuple(adj_triu)[0]  # single directional
        np.random.shuffle(edges)
        edges_list.append(edges)
        biedges = sparse_to_tuple(adj)[0]  # bidirectional edges
        np.random.shuffle(biedges)
        biedges_list.append(biedges)

    return edges_list, biedges_list
