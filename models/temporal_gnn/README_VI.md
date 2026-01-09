# Thư Mục Temporal GNN

## Giới thiệu

Thư mục `models/temporal_gnn/` chứa implementation của các mô hình Temporal Graph Neural Network - phần quan trọng nhất của dự án GraphPulse. Đây là nơi chứa mô hình chính **HTGN (Hyperbolic Temporal Graph Network)** và các baseline methods như EvolveGCN, GAE, VGAE.

Temporal GNN models trong GraphPulse:
- Xử lý temporal graph sequences với explicit temporal modeling
- Capture cả structural và temporal information
- Sử dụng hyperbolic geometry để model hierarchical structures
- Cho kết quả tốt nhất (10.2% improvement over baselines)

## Cấu trúc

```
temporal_gnn/
├── script/                    # Scripts chính và implementations
│   ├── main.py               # Entry point chính cho temporal GNN training
│   ├── config.py             # Configuration với Mac M2 support
│   ├── models/               # Model implementations
│   │   ├── HTGN.py          # Hyperbolic Temporal Graph Network (main model)
│   │   ├── BaseModel.py     # Base class cho tất cả models
│   │   ├── DynModels.py     # Dynamic graph models
│   │   ├── EvolveGCN/       # EvolveGCN implementation
│   │   └── static_baselines/ # GAE, VGAE baselines
│   ├── hgcn/                 # Hyperbolic GCN components
│   │   ├── layers/          # Hyperbolic layers (HGCNConv, HypGRU, HGATConv)
│   │   ├── manifolds/       # Hyperbolic manifolds (PoincareBall)
│   │   └── utils/           # Hyperbolic utilities
│   ├── baselines/            # Baseline scripts
│   │   ├── run_evolvegcn_baselines.py
│   │   ├── run_evolvegcn_baselines_TGC.py
│   │   └── run_static_baselines.py
│   ├── utils/                # Utility functions
│   │   ├── data_util.py     # Data loading và preprocessing
│   │   ├── util.py          # General utilities (với Mac M2 fixes)
│   │   └── ...
│   ├── example/              # Example run scripts
│   │   ├── run_htgn.sh
│   │   ├── run_evolvegcn.sh
│   │   └── ...
│   ├── train_tgc_end_to_end.py  # End-to-end training cho TGC
│   ├── train_graph_classification.py  # Graph classification training
│   ├── loss.py               # Loss functions
│   └── inits.py              # Initialization functions
├── data_sample/              # Sample data để test
├── requirements.txt          # Dependencies (CUDA version)
└── requirements_mac_m2.txt   # Mac M2 compatible dependencies
```

## Các file chính

### `script/main.py`

Entry point chính cho training temporal GNN models. Chứa `Runner` class để:
- Load data và prepare train/test splits
- Initialize model (HTGN, EvolveGCN, etc.)
- Training loop với early stopping
- Evaluation và logging
- Model saving

### `script/config.py`

Configuration file với argparse, định nghĩa tất cả hyperparameters và arguments. **Đã được patch cho Mac M2** với MPS support:
- Device selection: CUDA → MPS → CPU (automatic)
- Dataset-specific configurations
- Model-specific parameters

### `script/models/HTGN.py`

**Mô hình chính của GraphPulse** - Hyperbolic Temporal Graph Network.

**Key Components:**
- **Hyperbolic Geometry**: Sử dụng PoincareBall manifold
- **HGCN Layers**: Hyperbolic Graph Convolutional layers
- **GRU**: Gated Recurrent Unit cho temporal modeling
- **HTA (Hyperbolic Temporal Attention)**: Attention mechanism cho temporal aggregation
- **HTC (Hyperbolic Temporal Consistency)**: Regularization term

**Architecture:**
```
Input features → Linear → HGCN Layer 1 → HGCN Layer 2 → GRU → Temporal Attention → Output
```

### `script/models/BaseModel.py`

Base class cho tất cả temporal models, cung cấp:
- Hidden state management
- Temporal window handling
- Weighted hidden aggregation
- GRU integration

### `script/models/EvolveGCN/`

Implementation của EvolveGCN baseline:
- **EGCN.py**: EvolveGCN model với GRCU layers
- **GCNCONV.py**: Custom GCN convolution layer
- Support cho EGCNO và EGCNH variants

### `script/models/static_baselines/`

Static baseline models:
- **GAE.py**: Graph Autoencoder
- **VGAE.py**: Variational Graph Autoencoder

### `script/hgcn/`

Hyperbolic GCN components:
- **manifolds/**: Hyperbolic manifolds (PoincareBall, Euclidean)
- **layers/**: Hyperbolic layers (HGCNConv, HypGRU, HGATConv)
- **utils/**: Math utilities cho hyperbolic operations

### `script/utils/util.py`

General utilities với **Mac M2 compatibility fixes**:
- Random seed setting (conditional CUDA calls)
- Logger initialization
- Learning rate scheduling
- Negative sampling
- Device-agnostic tensor operations

### `script/loss.py`

Loss functions:
- **ReconLoss**: Reconstruction loss cho link prediction
- **VGAEloss**: Variational loss cho VAE models
- Prediction functions cho evaluation (AUC, AP)

## Cách sử dụng

### Quick Start

```bash
cd models/temporal_gnn/script

# Chạy HTGN với example script
bash example/run_htgn.sh

# Hoặc chạy trực tiếp
python main.py --dataset aion --model HTGN --seed 1024
```

### Configuration

**Các arguments chính:**
```bash
python main.py \
    --dataset aion \              # Dataset name
    --model HTGN \                # Model (HTGN, EvolveGCN, GAE, VGAE)
    --device_id -1 \              # -1 for CPU, 0+ for GPU
    --seed 1024 \                 # Random seed
    --lr 0.01 \                   # Learning rate
    --nhid 16 \                   # Hidden dimension
    --nb_window 5 \               # Temporal window size
    --max_epoch 500 \             # Max epochs
    --patience 50                 # Early stopping patience
```

### Mac M2 Users

**Quan trọng**: Code đã được patch để hỗ trợ Mac M2:
- Tự động detect và sử dụng MPS nếu available
- Fallback về CPU nếu MPS không available
- GPU memory tracking sẽ hiển thị 0 (MPS limitation)

**Install dependencies:**
```bash
pip install "torch>=2.0.0" torchvision torchaudio
pip install -r requirements_mac_m2.txt
```

### Example Scripts

**run_htgn.sh:**
```bash
#!/bin/bash
python main.py --dataset aion --model HTGN --device_id -1 --seed 1024
```

**run_evolvegcn.sh:**
```bash
python main.py --dataset aion --model EvolveGCN --egcn_type EGCNH
```

### Dataset-Specific Configurations

Config.py tự động set parameters cho từng dataset:
- `aion`: testlength=38, trainable_feat=1
- `dgd`: testlength=144, trainable_feat=1
- `adex`: testlength=59, trainable_feat=1
- Và nhiều datasets khác...

### Training Modes

**1. Link Prediction (default):**
```bash
python main.py --dataset aion --model HTGN
```

**2. Temporal Graph Classification (TGC):**
```bash
python train_tgc_end_to_end.py --dataset aion --model HTGN
```

**3. Graph Classification:**
```bash
python train_graph_classification.py --dataset aion --model HTGN
```

## Model Details

### HTGN Architecture

**Input Processing:**
- Node features (one-hot hoặc trainable)
- Edge indices qua time
- Temporal window của historical states

**Hyperbolic Layers:**
- **HGCNConv**: Hyperbolic graph convolution
  - Projects features vào hyperbolic space
  - Performs graph convolution trong hyperbolic space
  - Projects back (nếu cần)

**Temporal Modeling:**
- **GRU**: Gated Recurrent Unit để update hidden states
- **Temporal Attention**: Weighted aggregation của hidden states trong window
- **HTC**: Hyperbolic Temporal Consistency regularization

**Output:**
- Node embeddings trong hyperbolic space
- Used cho link prediction hoặc graph classification

### EvolveGCN

**Variants:**
- **EGCNO**: Evolves GCN weights only
- **EGCNH**: Evolves GCN weights và hidden states

**Architecture:**
- GRCU (Graph Recurrent Convolutional Unit) layers
- Matrix GRU để evolve GCN weights
- TopK selection cho EGCNH

## Data Format

### Expected Input

Data được load từ `script/utils/data_util.py` với format:
```python
{
    'edge_index_list': [tensor, ...],  # Edge indices cho mỗi timestep
    'pedges': [tensor, ...],            # Positive edges
    'nedges': [tensor, ...],            # Negative edges
    'new_pedges': [tensor, ...],        # New positive edges (inductive)
    'new_nedges': [tensor, ...],        # New negative edges (inductive)
    'num_nodes': int,                   # Number of nodes
    'time_length': int                  # Number of timesteps
}
```

### Output Format

Results được log vào:
- Console: Training progress, metrics
- File: `../data/output/log/{dataset}/{model}/{dataset}_seed_{seed}.txt`
- Model: Saved vào `../saved_models/{dataset}/{dataset}_{model}_seed_{seed}.pth`

## Evaluation Metrics

**Link Prediction:**
- **AUC (Area Under ROC Curve)**: Transductive và Inductive
- **AP (Average Precision)**: Transductive và Inductive

**Graph Classification:**
- **Accuracy**: Classification accuracy
- **F1 Score**: F1 score (nếu applicable)

## Lưu ý

### Dependencies

**Core:**
- PyTorch 2.0+ (hoặc 1.6.0+ cho CUDA)
- PyTorch Geometric
- geoopt (hyperbolic geometry)
- NetworkX, NumPy, Pandas

**Mac M2:**
- Xem `requirements_mac_m2.txt`
- Cần PyTorch 2.0+ cho MPS support

### Hardware Requirements

**Bắt buộc:**
- **GPU**: Khuyến nghị mạnh (CUDA hoặc MPS)
- **Memory**: Tối thiểu 8GB GPU memory (16GB+ recommended)
- **CPU**: Nhiều cores cho data loading

**Mac M2:**
- Đã được optimize cho MPS
- Có thể chạy trên CPU nhưng rất chậm
- Memory tracking không available (hiển thị 0 MiB)

### Configuration

**Device Selection** (automatic):
1. CUDA (nếu `device_id >= 0` và CUDA available)
2. MPS (nếu Mac M2 và PyTorch 2.0+)
3. CPU (fallback)

**Random Seeds:**
- Set trong `util.py` với Mac M2 compatibility
- CUDA seeds chỉ được set nếu CUDA available

### Performance

**Training Time:**
- Tùy dataset size và model complexity
- HTGN: Có thể mất vài giờ cho large datasets
- EvolveGCN: Nhanh hơn HTGN

**Memory Usage:**
- Phụ thuộc vào số nodes và timesteps
- Có thể cần giảm batch size hoặc window size

### Troubleshooting

**MPS Issues (Mac M2):**
- Ensure PyTorch 2.0+
- Check: `torch.backends.mps.is_available()`
- Fallback to CPU: `--device_id -1`

**CUDA Out of Memory:**
- Giảm `nb_window`
- Giảm `nhid`
- Process smaller datasets

**Low Performance:**
- Tăng `nhid` (hidden dimension)
- Tăng `nb_window` (temporal window)
- Adjust learning rate
- Check data quality

**Import Errors:**
- Đảm bảo đã install tất cả dependencies
- Check Python path và imports
- Verify PyTorch Geometric installation

### Reproducibility

- Sử dụng `--seed` argument
- Results có thể khác một chút giữa CUDA, MPS, và CPU
- Để exact reproducibility, sử dụng CPU: `--device_id -1`

### Best Practices

1. **Start với small dataset**: Test với `aion` trước
2. **Monitor training**: Check logs và loss curves
3. **Early stopping**: Sử dụng patience để avoid overfitting
4. **Hyperparameter tuning**: Experiment với nhid, lr, nb_window
5. **Multiple runs**: Run với nhiều seeds và average results
