# Thư Mục Models

## Giới thiệu

Thư mục `models/` chứa tất cả các implementation của các mô hình machine learning được sử dụng trong dự án GraphPulse. Đây là phần core của dự án, bao gồm các mô hình RNN cho sequence processing, static GNN cho graph classification, và temporal GNN cho temporal graph property prediction.

Thư mục này được tổ chức thành ba phần chính, mỗi phần giải quyết một approach khác nhau cho bài toán dự đoán thuộc tính của temporal graphs:

1. **RNN Models**: Xử lý sequences được extract từ temporal graphs
2. **Static GNN**: Xử lý từng graph snapshot độc lập
3. **Temporal GNN**: Xử lý temporal information một cách explicit

## Cấu trúc

```
models/
├── rnn/                    # RNN models cho sequence processing
│   ├── rnn_methods.py     # LSTM/GRU implementation
│   └── README.md
├── static_gnn/            # Static Graph Neural Networks
│   ├── static_graph_methods.py  # GIN implementation
│   ├── config_GIN.yml     # GIN configuration
│   └── README.md
└── temporal_gnn/          # Temporal Graph Neural Networks (chính)
    ├── script/            # Main scripts và implementations
    ├── requirements.txt   # Dependencies
    ├── requirements_mac_m2.txt  # Mac M2 compatible dependencies
    └── README.md
```

## Tổng Quan Các Mô Hình

### 1. RNN Models (`rnn/`)

**Mục đích**: Xử lý sequences được extract từ temporal graphs, sử dụng TDA features hoặc raw features.

**Models**:
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

**Input**: Sequences từ `data/Sequences/` với:
- TDA features: 5 features per timestep
- Raw features: 3 features per timestep
- Combined: 8 features per timestep

**Output**: Binary classification predictions (live/dead networks)

**Ưu điểm**:
- Dễ sử dụng và nhanh để test
- Hiệu quả với sequence data
- Không cần GPU (có thể chạy trên CPU)

**Nhược điểm**:
- Không capture structural information của graph
- Chỉ sử dụng features đã được extract

### 2. Static GNN (`static_gnn/`)

**Mục đích**: Graph classification trên static graph snapshots, xử lý từng snapshot độc lập.

**Models**:
- GIN (Graph Isomorphism Network): Model chính
- GCN (Graph Convolutional Network): Baseline

**Input**: Static graph snapshots từ temporal graphs

**Output**: Graph-level predictions

**Ưu điểm**:
- Capture structural information
- Strong baseline để so sánh

**Nhược điểm**:
- Không sử dụng temporal information
- Xử lý từng snapshot độc lập, không có memory

### 3. Temporal GNN (`temporal_gnn/`)

**Mục đích**: Temporal graph property prediction với explicit temporal modeling.

**Models**:
- **HTGN (Hyperbolic Temporal Graph Network)**: Mô hình chính của GraphPulse
  - Sử dụng hyperbolic geometry để capture hierarchical structures
  - Temporal reasoning qua recurrent layers
  - Cho kết quả tốt nhất trong experiments

- **EvolveGCN**: Baseline temporal GNN
  - Evolves GCN parameters theo thời gian
  - State-of-the-art baseline

- **Static Baselines**: GAE, VGAE
  - Graph autoencoders cho comparison

**Input**: Temporal graph sequences với node embeddings qua time

**Output**: Predictions cho future graph properties

**Ưu điểm**:
- Best performance (10.2% improvement over baselines)
- Capture cả structural và temporal information
- Explicit modeling của temporal dynamics

**Nhược điểm**:
- Phức tạp hơn, cần nhiều computational resources
- Cần GPU (CUDA hoặc MPS) cho training hiệu quả

## So Sánh Các Approaches

| Approach | Input | Output | Temporal Info | Best For |
|----------|-------|--------|---------------|----------|
| RNN | Sequences (TDA/Raw features) | Binary classification | Implicit (qua sequences) | Quick experiments, feature-based |
| Static GNN | Single graph snapshot | Graph property | Không | Baseline, static analysis |
| Temporal GNN | Temporal graph sequence | Future properties | Explicit (recurrent + temporal edges) | Best performance, production |

## Workflow Đề Xuất

### Bắt đầu với RNN
```bash
cd models/rnn
python rnn_methods.py
```
- Nhanh để test và validate pipeline
- Dễ debug
- Không cần GPU

### Nâng cấp lên Temporal GNN
```bash
cd models/temporal_gnn/script
python main.py --dataset aion --model HTGN
```
- Cho kết quả tốt nhất
- Sử dụng cả structural và temporal information
- Cần GPU cho training hiệu quả

### So sánh với Static GNN
```bash
cd models/static_gnn
python static_graph_methods.py
```
- Baseline để đánh giá improvement
- Hiểu contribution của temporal information

## Chi Tiết Từng Model

Xem các README riêng trong từng thư mục con:
- [`rnn/README.md`](rnn/README.md) - Chi tiết về RNN models
- [`static_gnn/README.md`](static_gnn/README.md) - Chi tiết về Static GNN
- [`temporal_gnn/README.md`](temporal_gnn/README.md) - Chi tiết về Temporal GNN

## Dependencies

Mỗi model type có dependencies khác nhau:

**RNN Models**:
- TensorFlow/Keras
- NumPy, Pandas
- scikit-learn

**Static GNN**:
- PyTorch
- PyTorch Geometric
- PyYAML (cho config)

**Temporal GNN**:
- PyTorch 2.0+ (hoặc 1.6.0+ cho CUDA)
- PyTorch Geometric
- geoopt (cho hyperbolic geometry)
- Xem `temporal_gnn/requirements.txt`

## Lưu ý

### Hardware Requirements

- **RNN**: Có thể chạy trên CPU, nhưng sẽ chậm hơn
- **Static GNN**: Nên có GPU cho training nhanh hơn
- **Temporal GNN**: **Bắt buộc cần GPU** (CUDA hoặc MPS) cho training hiệu quả

### Mac M2 Compatibility

Temporal GNN đã được patch để hỗ trợ Mac M2 với MPS:
- Xem `temporal_gnn/requirements_mac_m2.txt`
- Xem `MAC_M2_COMPATIBILITY.md` ở root

### Model Selection

- **Nếu cần quick results**: Dùng RNN
- **Nếu cần best performance**: Dùng Temporal GNN (HTGN)
- **Nếu cần baseline**: Dùng Static GNN

### Reproducibility

Tất cả models hỗ trợ random seed setting:
- RNN: `reset_random_seeds()` function
- Temporal GNN: `--seed` argument
- Static GNN: Random seed trong code

### Evaluation Metrics

Tất cả models đều sử dụng:
- **ROC-AUC**: Metric chính để so sánh
- **Accuracy**: Metric phụ
- **AP (Average Precision)**: Cho Temporal GNN
