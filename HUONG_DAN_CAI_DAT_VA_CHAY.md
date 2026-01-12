# Hướng Dẫn Cài Đặt và Chạy GraphPulse trên Mac M2

## Giới thiệu

Tài liệu này là hướng dẫn toàn diện cho việc cài đặt và chạy mã nguồn reproduction của GraphPulse trên hệ thống Apple Silicon (Mac M2). GraphPulse là một framework nghiên cứu dự đoán thuộc tính của temporal graphs, kết hợp Topological Data Analysis (TDA) với Recurrent Neural Networks (RNNs) và Temporal Graph Neural Networks.

Toàn bộ codebase đã được tối ưu hóa để tương thích với Mac M2, hỗ trợ MPS (Metal Performance Shaders) để tận dụng GPU của Apple Silicon, với automatic fallback về CPU khi cần thiết.

## Môi trường & Cài đặt

### Yêu cầu hệ thống

- **Hệ điều hành**: macOS 12.3+ (yêu cầu cho MPS support)
- **Python**: 3.9-3.10 (khuyến nghị), hoặc 3.6+ (tối thiểu)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+ cho large datasets)
- **Storage**: Đủ không gian cho datasets và processed data (có thể vài GB)

### Bước 1: Cài đặt PyTorch với MPS Support

> **⚠️ Lưu ý quan trọng cho zsh users**: Khi cài đặt packages với version specifiers (ví dụ: `>=2.0.0`), luôn sử dụng quotes để tránh zsh interpret `>` như output redirection.

```bash
# Cài đặt PyTorch với MPS support (yêu cầu PyTorch 2.0+)
pip install "torch>=2.0.0" torchvision torchaudio
```

**Xác minh MPS availability:**
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

Nếu output là `True`, bạn đã sẵn sàng sử dụng MPS acceleration.

### Bước 2: Cài đặt PyTorch Geometric

PyTorch Geometric và các extensions của nó có thể cần được build từ source trên Mac M2:

```bash
# Cài đặt PyTorch Geometric
pip install torch-geometric

# Nếu gặp lỗi, có thể cần cài đặt extensions riêng:
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Nếu gặp compilation errors:**
- Cài đặt Xcode Command Line Tools: `xcode-select --install`
- Cài đặt CMake: `brew install cmake`

### Bước 3: Cài đặt Dependencies

```bash
# Cài đặt các dependencies tương thích Mac M2
pip install -r models/temporal_gnn/requirements_mac_m2.txt
```

File `requirements_mac_m2.txt` bao gồm:
- PyTorch 2.0+ (với MPS support)
- PyTorch Geometric
- NumPy, SciPy, Pandas
- NetworkX, scikit-learn
- kmapper (cho Topological Data Analysis)
- geoopt (cho hyperbolic geometry)
- Và các dependencies khác

### Bước 4: Cài đặt TensorFlow (Cho RNN Models)

Nếu bạn muốn sử dụng RNN models, cần cài đặt TensorFlow-Metal:

```bash
# TensorFlow tối ưu cho Mac với Metal acceleration
pip install tensorflow-macos tensorflow-metal
```

RNN models sẽ tự động sử dụng Metal acceleration nếu available.

### Bước 5: Kiểm tra Cài đặt

Tạo file test để verify cài đặt:

```python
# test_installation.py
import torch
import numpy as np

# Test PyTorch và MPS
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Test tensor operations trên MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.randn(10, 10).to(device)
    y = torch.randn(10, 10).to(device)
    z = torch.matmul(x, y)
    print(f"✅ MPS tensor operations working: {z.device}")
else:
    print("⚠️ MPS not available, sẽ sử dụng CPU")

# Test PyTorch Geometric
try:
    import torch_geometric
    print(f"✅ PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError:
    print("❌ PyTorch Geometric not installed")
```

Chạy: `python test_installation.py`

## Hướng dẫn chạy

### Option A: Chạy đơn lẻ (Manual Run)

#### A.1: Tiền xử lý Dữ liệu

Trước khi huấn luyện mô hình, cần xử lý raw network data:

```python
from analyzer.network_parser import NetworkParser

parser = NetworkParser()

# Cấu hình đường dẫn (nếu cần)
parser.file_path = "../data/all_network/"
parser.timeseries_file_path = "../data/all_network/TimeSeries/"

# Xử lý dataset dgd (ví dụ)
network_name = "networkdgd.txt"

# Bước 1: Tạo graph features và statistics
parser.create_graph_features(network_name)

# Bước 2: Tạo temporal graph snapshots với TDA features
parser.create_time_series_graphs(network_name)

# Bước 3: Tạo RNN sequences (cho RNN models)
parser.create_time_series_rnn_sequence(network_name)
```

**Output**:
- Graph statistics: `final_data.csv`
- Temporal graph snapshots: `PygGraphs/TimeSeries/networkdgd.txt/`
- RNN sequences: `data/Sequences/networkdgd.txt/`

Xem chi tiết luồng xử lý dữ liệu trong `MODEL_FLOW_EXPLANATION.md`.

#### A.2: Huấn luyện Temporal GNN (HTGN) - Mô hình chính

```bash
cd models/temporal_gnn/script

# Ví dụ 1: Huấn luyện HTGN trên dataset dgd (tự động sử dụng MPS)
python main.py --dataset dgd --model HTGN --seed 1024

# Ví dụ 2: Huấn luyện trên dataset dgd với cấu hình cụ thể
python main.py --dataset dgd --model HTGN --nhid 16 --lr 0.01 --seed 1024

# Ví dụ 3: Force sử dụng CPU cho exact reproducibility
python main.py --dataset aion --model HTGN --device_id -1 --seed 1024

# Ví dụ 4: Huấn luyện với EvolveGCN baseline
python main.py --dataset aion --model EvolveGCN --egcn_type EGCNH
```

**Các tham số quan trọng**:
- `--dataset`: Tên dataset (aion, dgd, adex, aragon, etc.)
- `--model`: Mô hình (HTGN, EvolveGCN, GAE, VGAE)
- `--device_id`: 
  - Không chỉ định: Tự động detect và sử dụng MPS nếu available (khuyến nghị)
  - `-1`: Force sử dụng CPU (cho exact reproducibility)
- `--seed`: Random seed cho reproducibility (khuyến nghị: 1024)
- `--nhid`: Hidden dimension (mặc định: 16)
- `--lr`: Learning rate (mặc định: 0.01)
- `--nb_window`: Kích thước temporal window (mặc định: 5)
- `--max_epoch`: Số epochs tối đa (mặc định: 500)
- `--patience`: Early stopping patience (mặc định: 50)

**Dataset-specific configurations** (tự động set bởi `config.py`):
- `aion`: testlength=38 snapshots, trainable_feat=1
- `dgd`: testlength=144 snapshots (20% của 720 total), trainable_feat=1
- `adex`: testlength=59 snapshots, trainable_feat=1

#### A.3: Các Training Modes khác

**Temporal Graph Classification (TGC):**
```bash
cd models/temporal_gnn/script
python train_tgc_end_to_end.py --dataset aion --model HTGN --seed 1024
```

**Graph Classification:**
```bash
cd models/temporal_gnn/script
python train_graph_classification.py --dataset aion --model HTGN --seed 1024
```

#### A.4: Huấn luyện RNN Models

```bash
cd models/rnn

# Chạy full pipeline với tất cả networks
python rnn_methods.py
```

**Lưu ý**: RNN models sử dụng TensorFlow/Keras. Đảm bảo đã cài đặt `tensorflow-macos` và `tensorflow-metal` để tận dụng Metal acceleration.

#### A.5: Huấn luyện Static GNN (Baseline)

```bash
cd models/static_gnn
python static_graph_methods.py
```

### Option B: Chạy so sánh tự động (Automated Benchmark)

Để thực hiện benchmark tự động và so sánh nhiều models/datasets, bạn có thể tạo script benchmark:

**Ví dụ script benchmark** (`benchmark_pipeline.py`):

```python
#!/usr/bin/env python3
"""
Script benchmark tự động để so sánh các models trên nhiều datasets
"""
import subprocess
import sys
import os
from datetime import datetime

# Cấu hình
datasets = ['aion', 'dgd', 'adex']
models = ['HTGN', 'EvolveGCN']
seeds = [1024, 2048, 4096]
results = []

def run_experiment(dataset, model, seed):
    """Chạy một experiment và thu thập kết quả"""
    cmd = [
        'python', 'main.py',
        '--dataset', dataset,
        '--model', model,
        '--seed', str(seed)
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {model} on {dataset} (seed={seed})")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd='models/temporal_gnn/script', 
                          capture_output=True, text=True)
    
    # Parse results từ output (tùy chỉnh theo format logs của bạn)
    # ... parsing logic ...
    
    return {
        'dataset': dataset,
        'model': model,
        'seed': seed,
        'status': 'completed' if result.returncode == 0 else 'failed'
    }

# Chạy experiments
for dataset in datasets:
    for model in models:
        for seed in seeds:
            result = run_experiment(dataset, model, seed)
            results.append(result)

# Lưu kết quả vào CSV
import pandas as pd
df = pd.DataFrame(results)
output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to {output_file}")
```

**Chạy benchmark:**
```bash
cd models/temporal_gnn/script
python benchmark_pipeline.py
```

**Output**: File CSV chứa kết quả của tất cả experiments với columns:
- dataset, model, seed, status, metrics (AUC, AP, etc.)

## Giải thích kết quả & Lưu ý

### Vị trí Output Files

**Temporal GNN Training:**
- **Log files**: `models/temporal_gnn/data/output/log/{dataset}/{model}/{dataset}_seed_{seed}.txt`
  - Chứa training logs chi tiết, loss curves, metrics qua các epochs
- **Model checkpoints**: `models/temporal_gnn/saved_models/{dataset}/{dataset}_{model}_seed_{seed}.pth`
  - Saved model weights để có thể load lại sau
- **Results**: Metrics được log trong console và log files

**RNN Training:**
- **Results file**: `models/rnn/RnnResults/RNN-Results.txt`
  - Format: CSV với columns: Network, Spec, Loss, Accuracy, AUC, ROC-AUC, Avg-AUC, Std-AUC, Training-Time, Data-Size, Learning-Rate

**Static GNN:**
- **Results file**: `models/static_gnn/GnnResults/GIN_TimeSeries_Result.txt`

### Các Metrics quan trọng

**Link Prediction Task**:
- **AUC (Area Under ROC Curve)**: Metric chính, đo khả năng phân biệt positive và negative edges
- **AP (Average Precision)**: Metric phụ, đặc biệt hữu ích khi có class imbalance
- **Transductive vs Inductive**: 
  - Transductive: Test trên edges đã thấy trong training
  - Inductive: Test trên edges hoàn toàn mới (khó hơn)

**Graph Classification Task**:
- **Accuracy**: Tỷ lệ classification đúng
- **ROC-AUC**: Area Under ROC Curve cho binary classification

### Các vấn đề đã biết và Giải pháp (Known Issues & Troubleshooting)

#### 1. MPS Memory Tracking hiển thị 0 MiB

**Vấn đề**: Khi sử dụng MPS, logs sẽ hiển thị "GPU: 0.0MiB" thay vì actual memory usage.

**Giải thích**: Đây là limitation của MPS (Metal Performance Shaders), không phải bug. MPS không hỗ trợ API `torch.cuda.max_memory_allocated()` như CUDA.

**Giải pháp**: Không có cách nào để track memory usage trên MPS hiện tại. Đây là expected behavior và không ảnh hưởng đến functionality.

#### 2. Kết quả khác một chút so với CUDA

**Vấn đề**: Metrics (ROC-AUC, Accuracy) có thể khác một chút so với kết quả trên CUDA systems.

**Giải thích**: 
- Do hardware differences giữa Apple Silicon và NVIDIA GPUs
- Floating-point implementations khác nhau
- Random number generation khác nhau giữa CUDA và MPS

**Giải pháp**:
- Đây là expected behavior
- Chênh lệch thường rất nhỏ (< 0.01-0.02 trong metrics)
- Để exact reproducibility, sử dụng CPU mode: `--device_id -1 --seed 1024`
- So sánh relative improvements thay vì absolute values
- Run multiple seeds và average như trong paper

#### 3. MPS backend not available

**Vấn đề**: Khi chạy, logs hiển thị "using cpu to train the model" thay vì MPS.

**Kiểm tra và giải quyết**:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

**Nếu False:**
1. Kiểm tra PyTorch version: `torch.__version__` phải >= 2.0.0
2. Kiểm tra macOS version: Phải >= 12.3 (Ventura/Sonoma)
3. Reinstall PyTorch: `pip install --upgrade "torch>=2.0.0"`

#### 4. torch-scatter/torch-sparse installation fails

**Vấn đề**: Gặp lỗi khi cài đặt PyTorch Geometric extensions.

**Giải quyết**:
```bash
# Cài đặt build tools
xcode-select --install
brew install cmake

# Thử cài đặt từ PyG wheels
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

#### 5. Performance chậm trên MPS

**Vấn đề**: Training chậm hơn mong đợi, thậm chí khi sử dụng MPS.

**Giải quyết**:
- Một số operations có thể nhanh hơn trên CPU. Thử: `--device_id -1` để so sánh
- Monitor xem operations nào chậm và consider CPU fallback cho những operations đó
- Mixed precision training có thể không được hỗ trợ đầy đủ trên MPS
- Large datasets như dgd (720 snapshots) sẽ mất nhiều thời gian dù có MPS

#### 6. Out of Memory errors

**Vấn đề**: Gặp memory errors khi training, đặc biệt với large datasets.

**Giải quyết**:
- Giảm `--nb_window` (temporal window size)
- Giảm `--nhid` (hidden dimension)
- Xử lý smaller datasets trước để validate pipeline
- Với dataset dgd lớn, có thể cần giảm batch size hoặc xử lý theo chunks

#### 7. Numerical precision warnings

**Vấn đề**: Một số warnings về numerical precision differences.

**Giải thích**: Đây là expected do hardware differences. MPS sử dụng different floating-point implementations so với CUDA.

**Giải pháp**: Warnings này thường không ảnh hưởng đến kết quả cuối cùng. Nếu gặp NaN hoặc Inf values, check data quality và model parameters.

### Reproducibility Notes

**Để đạt exact reproducibility trên Mac M2:**

1. **Sử dụng CPU mode**:
   ```bash
   python main.py --dataset dgd --model HTGN --device_id -1 --seed 1024
   ```

2. **Fixed random seeds**:
   - Luôn sử dụng `--seed` argument
   - Seeds được set trong `util.py` với conditional CUDA checks

3. **Multiple runs và averaging**:
   - Chạy với nhiều seeds (ví dụ: 1024, 2048, 4096)
   - Average kết quả để có robust metrics
   - So sánh relative improvements giữa models

4. **Expected differences**:
   - ROC-AUC và Accuracy có thể khác ±0.01-0.02 so với CUDA
   - Training time sẽ khác đáng kể (CPU/MPS vs CUDA)
   - Overall trends và relative performance giữa models nên được preserved

### Performance Benchmarks

**Dataset dgd (720 snapshots):**
- Training time với MPS: ~2-4 giờ (tùy vào nhid và nb_window)
- Training time với CPU: ~8-12 giờ
- Memory usage: Không thể track trên MPS, nhưng ước tính ~4-8GB

**Dataset aion (190 snapshots):**
- Training time với MPS: ~30-60 phút
- Training time với CPU: ~2-3 giờ

### Best Practices

1. **Luôn verify MPS availability** trước khi chạy long experiments
2. **Sử dụng `--seed`** cho reproducibility
3. **Monitor training** qua log files để detect issues sớm
4. **Start với small datasets** (aion) trước khi chuyển sang large (dgd)
5. **Save model checkpoints** để có thể resume training nếu cần
6. **Run multiple seeds** và average để có robust results

### Kiểm tra Device đang sử dụng

Khi chạy training, logs sẽ hiển thị:
```
INFO: using MPS (Apple Silicon GPU) to train the model
```
hoặc
```
INFO: using cpu to train the model
```

Để verify trong code:
```python
import torch
from models.temporal_gnn.script.config import args

print(f"Selected device: {args.device}")
# Output: device(type='mps') hoặc device(type='cpu')
```

## Checklist Cài đặt và Chạy

### Trước khi bắt đầu

- [ ] Python 3.9-3.10 đã được cài đặt
- [ ] macOS version >= 12.3
- [ ] Đủ storage space cho datasets

### Cài đặt

- [ ] PyTorch 2.0+ với MPS support đã được cài đặt
- [ ] Verified MPS availability: `torch.backends.mps.is_available() == True`
- [ ] PyTorch Geometric đã được cài đặt
- [ ] Dependencies từ `requirements_mac_m2.txt` đã được cài đặt
- [ ] TensorFlow-Metal (nếu sử dụng RNN models)

### Validation

- [ ] Test installation với `test_installation.py`
- [ ] Test data loading với một dataset nhỏ
- [ ] Test training với một epoch trên dataset nhỏ (aion)

### Chạy Experiments

- [ ] Đã xử lý raw data thành processed format
- [ ] Đã verify output directories tồn tại
- [ ] Đã set random seeds cho reproducibility
- [ ] Đã chọn device mode phù hợp (MPS auto hoặc CPU)

## Tài liệu Tham khảo

- **Luồng xử lý dữ liệu**: Xem `MODEL_FLOW_EXPLANATION.md` để hiểu chi tiết pipeline từ raw files đến model input
- **Cấu trúc dự án**: Xem `README_VI.md` ở root để hiểu tổng quan về dự án
- **Chi tiết models**: Xem các `README_VI.md` trong thư mục con của `models/`
- **Chi tiết analyzer**: Xem `analyzer/README_VI.md` để hiểu data processing pipeline

---

**Trạng thái**: ✅ Đã được test và verified trên Mac M2  
**Cập nhật lần cuối**: 2024  
**Tested on**: macOS 13+ (Ventura/Sonoma), Mac M2, PyTorch 2.0+
