# GraphPulse: Topological Representations for Temporal Graph Property Prediction

## Giới thiệu

GraphPulse là một framework nghiên cứu dự đoán thuộc tính của temporal graphs (đồ thị thời gian). Framework này kết hợp Topological Data Analysis (TDA) với Recurrent Neural Networks (RNNs) để dự đoán sự tiến hóa của các mạng lưới biến đổi theo thời gian.

### Mục đích dự án

Nhiều mạng lưới trong thực tế phát triển theo thời gian, và việc dự đoán sự tiến hóa của các mạng này vẫn là một thách thức. Graph Neural Networks (GNNs) đã cho thấy thành công thực nghiệm trên static graphs, nhưng chúng thiếu khả năng học hiệu quả từ các nodes và edges với timestamps khác nhau. GraphPulse nhằm giải quyết vấn đề này bằng cách:

1. **Sử dụng Mapper method** (từ Topological Data Analysis) để trích xuất thông tin clustering quan trọng từ graph nodes
2. **Tận dụng khả năng sequential modeling** của RNNs để suy luận temporal về sự tiến hóa của graph

### Kết quả

Thông qua thử nghiệm rộng rãi, mô hình của chúng tôi đã cải thiện metric ROC-AUC thêm **10.2%** so với phương pháp state-of-the-art tốt nhất trên các temporal networks khác nhau.

## Cấu trúc dự án

```
GraphPulse/
├── analyzer/              # Xử lý và phân tích dữ liệu mạng
├── config.py             # Cấu hình chung cho dự án
├── data/                 # Dữ liệu mạng và sequences đã xử lý
├── image/                # Hình ảnh, biểu đồ cho documentation
├── models/               # Các mô hình machine learning
│   ├── rnn/             # Mô hình RNN cho sequence processing
│   ├── static_gnn/      # Static Graph Neural Networks
│   └── temporal_gnn/    # Temporal Graph Neural Networks (chính)
├── util/                 # Các utility functions
└── requirements.txt      # Dependencies Python
```

## Các file chính

### `config.py`
File cấu hình chung định nghĩa các đường dẫn dataset và tham số validation. Chứa các biến:
- `file_path`: Đường dẫn đến thư mục chứa network files
- `timeseries_file_path`: Đường dẫn cho time series data
- `time_window`: Cửa sổ thời gian (mặc định: [7] ngày)
- `network_validation_duration`: Thời gian tối thiểu để validate network (20 ngày)
- `label_treshhold_percentage`: Ngưỡng phần trăm để phân loại live/dead networks (10%)

### `requirements.txt`
Danh sách các thư viện Python cần thiết cho dự án, bao gồm:
- PyTorch, TensorFlow
- PyTorch Geometric
- NetworkX, Pandas, NumPy
- kmapper (cho TDA)
- scikit-learn, matplotlib

### `MAC_M2_COMPATIBILITY.md`
Tài liệu hướng dẫn tương thích với Mac M2 (Apple Silicon), bao gồm các sửa đổi cho MPS (Metal Performance Shaders) support.

### `MAC_M2_FIXES_SUMMARY.md`
Tóm tắt các sửa đổi đã thực hiện để tương thích với Mac M2.

## Cách sử dụng

### 1. Cài đặt Dependencies

```bash
# Cài đặt các thư viện cơ bản
pip install -r requirements.txt

# Đối với Mac M2, sử dụng:
pip install "torch>=2.0.0" torchvision torchaudio
pip install -r models/temporal_gnn/requirements_mac_m2.txt
```

### 2. Xử lý Dữ liệu

```python
from analyzer.network_parser import NetworkParser

parser = NetworkParser()
# Tạo graph features từ network file
parser.create_graph_features("networkadex.txt")
# Tạo time series graphs với TDA
parser.create_time_series_graphs("networkadex.txt")
# Tạo RNN sequences
parser.create_time_series_rnn_sequence("networkadex.txt")
```

### 3. Chạy Mô hình

**RNN Models:**
```bash
cd models/rnn
python rnn_methods.py
```

**Temporal GNN:**
```bash
cd models/temporal_gnn/script
python main.py --dataset aion --model HTGN
```

**Static GNN:**
```bash
cd models/static_gnn
python static_graph_methods.py
```

## Workflow đề xuất

1. **Bước 1: Xử lý dữ liệu** - Sử dụng `analyzer/` để parse networks và tạo features
2. **Bước 2: Tạo sequences** - Extract TDA features và tạo sequences cho RNN
3. **Bước 3: Training** - Chọn mô hình phù hợp (RNN cho quick test, Temporal GNN cho best results)
4. **Bước 4: Evaluation** - So sánh kết quả với baselines

## Lưu ý

### Yêu cầu hệ thống
- **Python**: 3.6+ (khuyến nghị 3.9-3.10)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+ cho large networks)
- **Storage**: Đủ không gian cho datasets và processed data

### Hardware Compatibility
- **Mac M2**: Đã được patch để hỗ trợ MPS (xem `MAC_M2_COMPATIBILITY.md`)
- **CUDA**: Hỗ trợ đầy đủ cho NVIDIA GPUs
- **CPU**: Có thể chạy trên CPU nhưng sẽ chậm hơn đáng kể

### Dependencies quan trọng
- **PyTorch Geometric**: Có thể cần build từ source trên Mac M2
- **kmapper**: Cần cho Topological Data Analysis
- **TensorFlow**: Cần cho RNN models (có thể dùng tensorflow-macos trên Mac)

### Data Paths
Trước khi chạy, cần cập nhật các đường dẫn trong `config.py`:
- `file_path`: Đường dẫn đến thư mục chứa network files
- `timeseries_file_path`: Đường dẫn cho time series data

### Performance Notes
- TDA processing có thể mất nhiều thời gian cho large networks
- Temporal GNN training tốn nhiều memory hơn RNN
- Khuyến nghị sử dụng GPU (CUDA hoặc MPS) cho training

## Trích dẫn

Nếu sử dụng GraphPulse trong nghiên cứu của bạn, vui lòng trích dẫn:

```bibtex
@inproceedings{shamsi2024graphpulse,
    title={GraphPulse: Topological Representations for Temporal Graph Property Prediction},
    author={Shamsi, Kiarash and Poursafaei, Farimah and Huang, Shenyang and Ngo, Bao Tran Gia and Coskunuzer, Baris and Akcora, Cuneyt Gurcan},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```
