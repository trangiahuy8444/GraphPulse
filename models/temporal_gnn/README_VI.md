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

# Ví dụ 1: Chạy HTGN với example script
bash example/run_htgn.sh

# Ví dụ 2: Chạy trực tiếp với dataset aion
python main.py --dataset aion --model HTGN --seed 1024

# Ví dụ 3: Huấn luyện trên dataset dgd (Mac M2 tự động sử dụng MPS)
python main.py --dataset dgd --model HTGN --seed 1024
```

### Configuration

**Các arguments chính:**
```bash
python main.py \
    --dataset aion \              # Tên dataset (aion, dgd, adex, etc.)
    --model HTGN \                # Mô hình (HTGN, EvolveGCN, GAE, VGAE)
    --device_id -1 \              # -1 cho CPU, 0+ cho GPU, để trống để auto-detect
    --seed 1024 \                 # Random seed cho reproducibility
    --lr 0.01 \                   # Learning rate
    --nhid 16 \                   # Hidden dimension
    --nb_window 5 \               # Kích thước temporal window
    --max_epoch 500 \             # Số epochs tối đa
    --patience 50                 # Early stopping patience
```

**Lưu ý về device_id trên Mac M2**:
- **Không chỉ định `--device_id`**: Tự động detect và sử dụng MPS nếu available (khuyến nghị)
- **`--device_id -1`**: Force sử dụng CPU (cho exact reproducibility hoặc debugging)
- **`--device_id 0`**: Sẽ fallback về MPS nếu CUDA không available (Mac M2 behavior)

### Mac M2 Users

**Quan trọng**: Code đã được patch để hỗ trợ Mac M2 với MPS (Metal Performance Shaders):
- Tự động detect và sử dụng MPS nếu available (PyTorch 2.0+ required)
- Fallback về CPU nếu MPS không available
- GPU memory tracking sẽ hiển thị 0 MiB (MPS limitation, không phải bug)

**Cài đặt dependencies cho Mac M2:**
```bash
# Bước 1: Cài đặt PyTorch với MPS support (yêu cầu PyTorch 2.0+)
# Lưu ý: Sử dụng quotes cho zsh compatibility
pip install "torch>=2.0.0" torchvision torchaudio

# Bước 2: Xác minh MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Bước 3: Cài đặt dependencies tương thích Mac M2
pip install -r requirements_mac_m2.txt
```

**Ví dụ huấn luyện trên Mac M2:**
```bash
cd models/temporal_gnn/script

# Ví dụ 1: Huấn luyện HTGN trên dataset aion (tự động dùng MPS)
python main.py --dataset aion --model HTGN --seed 1024

# Ví dụ 2: Huấn luyện trên dataset dgd với cấu hình cụ thể
python main.py --dataset dgd --model HTGN --nhid 16 --lr 0.01 --seed 1024

# Ví dụ 3: Force sử dụng CPU cho exact reproducibility
python main.py --dataset aion --model HTGN --device_id -1 --seed 1024

# Ví dụ 4: Huấn luyện với EvolveGCN baseline
python main.py --dataset aion --model EvolveGCN --egcn_type EGCNH
```

**Kiểm tra device đang sử dụng:**
Khi chạy training, logs sẽ hiển thị:
```
INFO: using MPS (Apple Silicon GPU) to train the model
```
hoặc
```
INFO: using cpu to train the model
```

### Example Scripts

**run_htgn.sh:**
```bash
#!/bin/bash
# Example script cho HTGN training
python main.py --dataset aion --model HTGN --device_id -1 --seed 1024
```

**run_evolvegcn.sh:**
```bash
#!/bin/bash
# Example script cho EvolveGCN baseline
python main.py --dataset aion --model EvolveGCN --egcn_type EGCNH
```

**Lưu ý**: Trên Mac M2, có thể bỏ `--device_id -1` để tự động sử dụng MPS.

### Dataset-Specific Configurations

File `config.py` tự động thiết lập các parameters phù hợp cho từng dataset:
- **`aion`**: testlength=38 snapshots, trainable_feat=1 (trainable node features)
- **`dgd`**: testlength=144 snapshots (20% của 720 total), trainable_feat=1
- **`adex`**: testlength=59 snapshots (20% của 293 total), trainable_feat=1
- **`aragon`**: testlength=67 snapshots, trainable_feat=1
- Và nhiều datasets khác được cấu hình tương tự...

**Lưu ý**: `trainable_feat=1` có nghĩa là sử dụng trainable node features thay vì one-hot encoding. Điều này thường cho kết quả tốt hơn cho large networks.

### Training Modes

**1. Link Prediction (default mode):**
```bash
# Huấn luyện cho link prediction task
python main.py --dataset dgd --model HTGN

# Với Mac M2 (tự động dùng MPS):
python main.py --dataset dgd --model HTGN --seed 1024
```

**2. Temporal Graph Classification (TGC):**
```bash
# Huấn luyện cho temporal graph classification
python train_tgc_end_to_end.py --dataset aion --model HTGN

# Mac M2 example:
python train_tgc_end_to_end.py --dataset aion --model HTGN --seed 1024
```

**3. Graph Classification:**
```bash
# Huấn luyện cho graph-level classification
python train_graph_classification.py --dataset aion --model HTGN

# Mac M2 example:
python train_graph_classification.py --dataset aion --model HTGN --seed 1024
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

**Core Dependencies:**
- **PyTorch**: 2.0+ (khuyến nghị cho Mac M2 với MPS) hoặc 1.6.0+ cho CUDA systems
- **PyTorch Geometric**: Graph neural network library
- **geoopt**: Hyperbolic geometry library (có thể cần build từ source)
- **NetworkX**: Graph manipulation và analysis
- **NumPy, Pandas**: Data processing

**Mac M2 Specific:**
- Xem `requirements_mac_m2.txt` cho danh sách đầy đủ
- **Bắt buộc**: PyTorch 2.0+ cho MPS support
- PyTorch Geometric extensions có thể cần build từ source

### Hardware Requirements

**Yêu cầu phần cứng:**
- **GPU**: Khuyến nghị mạnh (CUDA hoặc MPS) cho training hiệu quả
- **Memory**: Tối thiểu 8GB GPU memory (khuyến nghị 16GB+ cho large datasets như dgd)
- **CPU**: Nhiều cores cho data loading và preprocessing

**Mac M2 (Apple Silicon):**
- Đã được optimize cho MPS (Metal Performance Shaders)
- MPS sẽ tự động được sử dụng nếu PyTorch 2.0+ được cài đặt
- Có thể chạy trên CPU nhưng sẽ rất chậm (không khuyến nghị cho production)
- Memory tracking không available trên MPS (sẽ hiển thị 0 MiB trong logs - đây là limitation của MPS, không phải bug)

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
- Tùy thuộc vào kích thước dataset và độ phức tạp của model
- **HTGN**: Có thể mất vài giờ cho large datasets (ví dụ: dgd với 720 snapshots)
- **EvolveGCN**: Nhanh hơn HTGN do architecture đơn giản hơn
- **Mac M2 với MPS**: Thời gian training tương đương hoặc nhanh hơn CPU, nhưng có thể chậm hơn CUDA một chút

**Memory Usage:**
- Phụ thuộc vào số nodes, số timesteps, và kích thước temporal window
- Large datasets như dgd có thể cần giảm `nb_window` hoặc `nhid` nếu gặp memory issues
- Mac M2: Memory tracking không available, nhưng MPS có thể quản lý memory tốt hơn CPU mode

### Troubleshooting

**MPS Issues (Mac M2):**
- Đảm bảo đã cài đặt PyTorch 2.0+: `pip install "torch>=2.0.0"`
- Kiểm tra MPS availability: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Nếu MPS không available, check macOS version (yêu cầu macOS 12.3+)
- Fallback to CPU: Sử dụng `--device_id -1` nếu gặp vấn đề với MPS

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

- **Random Seeds**: Luôn sử dụng `--seed` argument để đảm bảo reproducibility
- **Hardware Differences**: Kết quả có thể khác một chút giữa CUDA, MPS, và CPU (thường < 0.01-0.02 trong metrics)
- **Exact Reproducibility**: Để đạt exact reproducibility, sử dụng CPU mode: `--device_id -1 --seed 1024`
- **Mac M2**: Sử dụng cùng seed trên cùng hệ thống sẽ cho kết quả reproducible, nhưng có thể khác với CUDA systems

### Best Practices

1. **Start với small dataset**: Test với `aion` trước
2. **Monitor training**: Check logs và loss curves
3. **Early stopping**: Sử dụng patience để avoid overfitting
4. **Hyperparameter tuning**: Experiment với nhid, lr, nb_window
5. **Multiple runs**: Run với nhiều seeds và average results
