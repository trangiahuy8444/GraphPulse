# Giải Thích Luồng Dữ Liệu và Mô Hình: Từ Raw Files Đến Model Input

## Tổng Quan

Tài liệu này mô tả chi tiết luồng xử lý dữ liệu từ raw network files đến input của mô hình machine learning, sử dụng dataset `dgd` (networkdgd.txt) làm ví dụ minh họa. Pipeline này áp dụng cho tất cả các transaction network datasets trong GraphPulse.

## Luồng Xử Lý Dữ Liệu: Dataset DGD

### Giai Đoạn 1: Raw Data Files

**Input**: File raw network `data/all_network/networkdgd.txt`

**Format dữ liệu**:
```
from_node to_node timestamp value
node1 node2 1234567890 100.5
node2 node3 1234567900 200.3
...
```

**Đặc điểm**:
- Mỗi dòng đại diện cho một transaction
- `from_node`, `to_node`: Định danh nodes trong network
- `timestamp`: Unix timestamp (đơn vị: giây)
- `value`: Giá trị transaction (numeric)

**Thống kê dataset DGD**:
- Tổng số snapshots: 720 (theo cấu hình trong `config.py`)
- Test snapshots: 144 (20% cho testing)
- Train snapshots: 576 (80% cho training)
- Duration: Khoảng 720 ngày dữ liệu liên tục

### Giai Đoạn 2: Time Series Data Preprocessing

**Module**: `analyzer/network_parser.py` → `NetworkParser.create_time_series_graphs()`

**Quá trình xử lý**:

1. **Đọc và Parse Raw File**:
   - Đọc file `networkdgd.txt` sử dụng pandas
   - Chuyển đổi timestamp từ Unix seconds sang datetime
   - Sắp xếp dữ liệu theo thời gian (chronological order)

2. **Sliding Window Generation**:
   - **Window Size**: 7 ngày (tham số `windowSize`)
   - **Gap**: 3 ngày (khoảng cách giữa data window và label window)
   - **Label Window Size**: 7 ngày (để tạo labels)

3. **Snapshot Creation**:
   ```
   Snapshot 1: Days 0-6    → Label từ Days 10-16
   Snapshot 2: Days 1-7    → Label từ Days 11-17
   Snapshot 3: Days 2-8    → Label từ Days 12-18
   ...
   Snapshot 720: Days 713-719 → Label tương ứng
   ```

4. **Label Generation**:
   - So sánh số lượng transactions trong data window và label window
   - Label = 1 (live network): Nếu số transactions tăng từ data window → label window
   - Label = 0 (dead network): Nếu số transactions giảm hoặc bằng

5. **Normalization**:
   - Normalize edge weights về khoảng [1, 10]
   - Công thức: `normalized_value = 1 + 9 * ((value - min) / (max - min))`

**Output**: Tập hợp các temporal graph snapshots, mỗi snapshot là một NetworkX MultiDiGraph

### Giai Đoạn 3: Topological Data Analysis (TDA) Feature Extraction

**Module**: `analyzer/network_parser.py` → `NetworkParser.process_TDA_extracted_temporal_features()`

**Quá trình trích xuất đặc trưng TDA**:

1. **Node Feature Calculation**:
   - `outgoing_edge_weight_sum`: Tổng trọng số các edges đi ra từ node
   - `incoming_edge_weight_sum`: Tổng trọng số các edges đi vào node
   - `outgoing_edge_count`: Số lượng edges đi ra
   - `incoming_edge_count`: Số lượng edges đi vào

2. **Daily Temporal Features**:
   - Tính toán features cho từng ngày trong window (7 ngày)
   - Mỗi node có 2 features mỗi ngày: `dailyClusterID` và `dailyClusterSize`

3. **Mapper Method Application**:
   - Sử dụng KeplerMapper (kmapper library)
   - Áp dụng Topological Data Analysis để nhóm các nodes dựa trên features
   - Tạo mapper graph với các parameters:
     - `n_cubes`: Số lượng hypercubes cho coverage
     - `per_overlap`: Phần trăm overlap giữa các cubes
     - `cls`: Clustering algorithm (thường dùng DBSCAN hoặc KMeans)

4. **Feature Vector Generation**:
   - Mỗi node được gán với cluster ID và cluster size
   - Features cuối cùng: [outgoing_weight_sum, incoming_weight_sum, outgoing_count, incoming_count, dailyClusterID, dailyClusterSize]

**Output**: 
- PyTorch Geometric Data objects với node features đầy đủ
- Lưu trong `PygGraphs/TimeSeries/networkdgd.txt/TDA_Tuned/`

### Giai Đoạn 4: Sequence Generation cho RNN Models

**Module**: `analyzer/network_parser.py` → `NetworkParser.create_time_series_rnn_sequence()`

**Quá trình tạo sequences**:

1. **Temporal Window Aggregation**:
   - Mỗi snapshot tạo một feature vector đại diện cho toàn bộ graph
   - Aggregation: Tính mean, max, hoặc sum của node features

2. **Sequence Construction**:
   - **TDA Sequences** (`seq_tda.txt`): 5 features per timestep
     - Features từ TDA-extracted graph (5 features)
   - **Raw Sequences** (`seq_raw.txt`): 3 features per timestep
     - Basic graph statistics (nodes, edges, density)
   - **Combined Sequences**: 8 features per timestep (GraphPulse approach)
     - Kết hợp TDA features + Raw features

3. **Sequence Format**:
   ```python
   {
       "sequence": {
           "Overlap_xx_Ncube_x": [
               [[feat1, feat2, ..., feat5], ...],  # Timestep 1
               [[feat1, feat2, ..., feat5], ...],  # Timestep 2
               ...
           ]
       },
       "label": [0, 1, 0, 1, ...]  # Binary labels
   }
   ```

**Output**: 
- Sequence files trong `data/Sequences/networkdgd.txt/`
- `seq_tda.txt`: TDA features
- `seq_raw.txt`: Raw features

### Giai Đoạn 5: Data Loading cho Temporal GNN Models

**Module**: `models/temporal_gnn/script/utils/data_util.py` → `loader()`

**Quá trình load dữ liệu**:

1. **Edge List Generation**:
   - Đọc edgelist từ processed data
   - Tạo `edge_index_list`: Danh sách edge indices cho mỗi snapshot
   - Mỗi snapshot có dạng `torch.Tensor` shape `(2, num_edges)`

2. **Negative Sampling**:
   - Tạo negative edges cho link prediction task
   - `pedges`: Positive edges (existing edges)
   - `nedges`: Negative edges (non-existing edges)
   - `new_pedges`, `new_nedges`: Inductive test edges

3. **Train-Test Split**:
   - **Sequential Split**: 80-20 split theo thời gian
   - Train: Snapshots 0 đến (total - testlength)
   - Test: Snapshots (total - testlength) đến total
   - Đối với DGD: Train = 576 snapshots, Test = 144 snapshots

4. **Data Structure**:
   ```python
   data = {
       'edge_index_list': [tensor, ...],  # Edge indices cho mỗi timestep
       'pedges': [tensor, ...],            # Positive edges
       'nedges': [tensor, ...],            # Negative edges
       'new_pedges': [tensor, ...],        # Inductive positive edges
       'new_nedges': [tensor, ...],        # Inductive negative edges
       'num_nodes': int,                   # Tổng số nodes
       'time_length': int                  # Tổng số timesteps (720)
   }
   ```

### Giai Đoạn 6: Model Input Preparation

**Module**: `models/temporal_gnn/script/inits.py` → `prepare()`

**Input cho HTGN Model**:

1. **Node Features**:
   - **One-hot encoding** (default): `torch.eye(num_nodes)` → `(num_nodes, num_nodes)`
   - **Trainable features** (cho DGD): `Parameter(torch.ones(num_nodes, nfeat))` → `(num_nodes, nfeat)`
   - `nfeat = 128` (theo config)

2. **Edge Indices**:
   - Cho mỗi timestep `t`: `edge_index = data['edge_index_list'][t]`
   - Shape: `(2, num_edges_at_t)`
   - Được move lên device (CUDA/MPS/CPU)

3. **Temporal Context**:
   - **Hidden States**: Window của hidden states từ các timesteps trước
   - **Window Size**: `nb_window = 5` (default)
   - Hidden states được lưu trữ và cập nhật qua time

4. **Forward Pass Input**:
   ```python
   # Cho mỗi timestep t:
   x = node_features.to(device)           # (num_nodes, nfeat)
   edge_index = edge_indices[t].to(device) # (2, num_edges)
   hiddens = [h_prev_1, h_prev_2, ..., h_prev_5]  # Temporal window
   ```

### Giai Đoạn 7: Model Processing (HTGN)

**Module**: `models/temporal_gnn/script/models/HTGN.py`

**Forward Pass Flow**:

1. **Feature Projection**:
   ```
   Input: x (num_nodes, nfeat=128)
   ↓
   Linear: x → h0 (num_nodes, nout=16)
   ```

2. **Hyperbolic Graph Convolution**:
   ```
   h0 → HGCNConv Layer 1 → h1 (num_nodes, 2*nhid=32)
   ↓
   h1 → HGCNConv Layer 2 → h2 (num_nodes, nout=16)
   ```
   - Ánh xạ features vào hyperbolic space (PoincareBall)
   - Thực hiện graph convolution trong hyperbolic space
   - Capture hierarchical structures

3. **Temporal Modeling**:
   ```
   h2 + h_prev → GRU → h_current (num_nodes, nout=16)
   ```
   - GRU cập nhật hidden state dựa trên current embedding và previous state
   - Capture temporal dynamics

4. **Temporal Attention**:
   ```
   [h_prev_1, h_prev_2, ..., h_prev_5, h_current]
   ↓
   Weighted Aggregation (HTA) → h_aggregated
   ```
   - Hyperbolic Temporal Attention (HTA) tính trọng số cho các hidden states
   - Aggregate thành final representation

5. **Output**:
   - Node embeddings: `z (num_nodes, nout)` trong hyperbolic space
   - Sử dụng cho link prediction hoặc graph classification

### Giai Đoạn 8: Training Loop

**Module**: `models/temporal_gnn/script/main.py` → `Runner.run()`

**Quá trình training**:

1. **Training Phase** (576 snapshots cho DGD):
   ```python
   for epoch in range(max_epochs):
       for t in train_shots:  # t từ 0 đến 575
           # Forward pass
           z = model(edge_index, x)
           
           # Loss calculation
           loss = loss_function(z, pos_edges, neg_edges)
           
           # Backpropagation
           loss.backward()
           optimizer.step()
           
           # Update hidden states
           model.update_hiddens(z)
   ```

2. **Validation Phase** (144 snapshots cuối):
   ```python
   model.eval()
   with torch.no_grad():
       for t in test_shots:  # t từ 576 đến 719
           z = model(edge_index, x)
           auc, ap = evaluate(z, pos_edges, neg_edges)
   ```

3. **Early Stopping**:
   - Monitor validation loss
   - Stop nếu không cải thiện sau `patience` epochs (default: 50)

## Tóm Tắt Luồng Dữ Liệu

```
Raw File (networkdgd.txt)
    ↓
[Preprocessing: Parse, Sort, Validate]
    ↓
Time Series Data (720 days)
    ↓
[Sliding Window: 7 days, gap 3 days]
    ↓
720 Graph Snapshots (NetworkX graphs)
    ↓
[TDA Feature Extraction]
    ↓
720 PyG Data Objects (với node features)
    ↓
[Sequence Generation - RNN path]
    ↓
Sequence Files (seq_tda.txt, seq_raw.txt)
    ↓
OR
    ↓
[Data Loading - Temporal GNN path]
    ↓
Data Dictionary (edge_index_list, pedges, nedges, ...)
    ↓
[Train-Test Split: 576 train, 144 test]
    ↓
Model Input (mỗi timestep)
    ↓
HTGN Forward Pass
    ↓
Node Embeddings
    ↓
Link Prediction / Graph Classification
```

## Tham Số Quan Trọng cho Dataset DGD

- **Total Snapshots**: 720
- **Train Snapshots**: 576 (80%)
- **Test Snapshots**: 144 (20%)
- **Window Size**: 7 ngày
- **Gap**: 3 ngày
- **Label Window**: 7 ngày
- **Node Features Dimension**: 128 (trainable) hoặc num_nodes (one-hot)
- **Hidden Dimension**: 16 (default)
- **Temporal Window**: 5 snapshots

## Lưu Ý về Mac M2

Khi chạy trên Mac M2 với MPS:
- Tất cả tensors tự động được move lên MPS device
- Data loading và preprocessing giống hệt như trên CUDA
- Performance có thể khác một chút do hardware differences
- Để reproducibility: Sử dụng `--device_id -1` để force CPU mode
