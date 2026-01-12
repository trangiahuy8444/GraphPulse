# GraphPulse: Topological Representations for Temporal Graph Property Prediction

## Giới thiệu

GraphPulse là một framework nghiên cứu dự đoán thuộc tính của temporal graphs (đồ thị thời gian). Framework này kết hợp Topological Data Analysis (TDA) với Recurrent Neural Networks (RNNs) để dự đoán sự tiến hóa của các mạng lưới biến đổi theo thời gian.

### Mục đích nghiên cứu

Nhiều mạng lưới trong thực tế phát triển theo thời gian, và việc dự đoán sự tiến hóa của các mạng này vẫn là một thách thức trong lĩnh vực machine learning và graph analysis. Graph Neural Networks (GNNs) đã chứng minh thành công thực nghiệm trên static graphs, tuy nhiên chúng thiếu khả năng học hiệu quả từ các nodes và edges với timestamps khác nhau. GraphPulse nhằm giải quyết vấn đề này thông qua hai phương pháp chính:

1. **Trích xuất đặc trưng bằng Mapper method**: Áp dụng Topological Data Analysis (TDA) để trích xuất thông tin clustering quan trọng từ graph nodes, giúp capture cấu trúc topo học của mạng.

2. **Mô hình hóa chuỗi thời gian**: Tận dụng khả năng sequential modeling của Recurrent Neural Networks (RNNs) để suy luận temporal về sự tiến hóa của graph.

### Kết quả thực nghiệm

Thông qua các thử nghiệm rộng rãi trên nhiều temporal networks khác nhau, mô hình GraphPulse đã đạt được cải thiện metric ROC-AUC thêm **10.2%** so với phương pháp state-of-the-art tốt nhất hiện tại.

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
├── MODEL_FLOW_EXPLANATION.md     # Giải thích chi tiết luồng dữ liệu
├── HUONG_DAN_CAI_DAT_VA_CHAY.md  # Hướng dẫn cài đặt và chạy (Mac M2)
└── requirements.txt               # Dependencies Python
```

## Các file chính

### `config.py`
File cấu hình chung định nghĩa các đường dẫn dataset và tham số validation cho toàn bộ dự án. Các biến quan trọng:
- `file_path`: Đường dẫn đến thư mục chứa raw network files
- `timeseries_file_path`: Đường dẫn cho time series data đã được tổ chức
- `time_window`: Cửa sổ thời gian cho việc tạo snapshots (mặc định: `[7]` ngày)
- `network_validation_duration`: Thời gian tối thiểu để validate network (20 ngày)
- `label_treshhold_percentage`: Ngưỡng phần trăm để phân loại live/dead networks (10%)

### `requirements.txt`
Danh sách các thư viện Python cần thiết cho dự án, bao gồm:
- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **Graph Processing**: PyTorch Geometric, NetworkX
- **Data Processing**: Pandas, NumPy
- **Topological Analysis**: kmapper (cho TDA)
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib

### `MODEL_FLOW_EXPLANATION.md`
Tài liệu chi tiết giải thích luồng xử lý dữ liệu từ raw files đến model input, sử dụng dataset `dgd` làm ví dụ minh họa. Xem tài liệu này để hiểu rõ pipeline hoàn chỉnh.

### `HUONG_DAN_CAI_DAT_VA_CHAY.md`
**Tài liệu chính** - Hướng dẫn toàn diện về cài đặt và chạy GraphPulse trên Mac M2 (Apple Silicon). Bao gồm:
- Hướng dẫn cài đặt chi tiết (PyTorch với MPS, dependencies)
- Cách chạy models (Manual và Automated)
- Troubleshooting và known issues
- Reproducibility guidelines

## Cách sử dụng

> **📖 Hướng dẫn chi tiết**: Xem `HUONG_DAN_CAI_DAT_VA_CHAY.md` để có hướng dẫn cài đặt và chạy đầy đủ, bao gồm troubleshooting và best practices.

### Quick Start - Cài đặt cơ bản

**Cho Mac M2 (Apple Silicon):**
```bash
# Cài đặt PyTorch với MPS support
pip install "torch>=2.0.0" torchvision torchaudio

# Cài đặt dependencies
pip install -r models/temporal_gnn/requirements_mac_m2.txt
```

### Quick Start - Chạy Models

**Temporal GNN (Mô hình chính):**
```bash
cd models/temporal_gnn/script
python main.py --dataset aion --model HTGN --seed 1024
```

Xem `HUONG_DAN_CAI_DAT_VA_CHAY.md` cho:
- Hướng dẫn xử lý dữ liệu chi tiết
- Tất cả các training options và parameters
- Automated benchmarking
- Troubleshooting guides

**Static GNN (Baseline):**
```bash
cd models/static_gnn
python static_graph_methods.py
```

## Workflow nghiên cứu đề xuất

1. **Giai đoạn 1: Tiền xử lý dữ liệu**
   - Sử dụng `analyzer/network_parser.py` để parse raw network files
   - Trích xuất graph features và statistics
   - Xem chi tiết trong `MODEL_FLOW_EXPLANATION.md`

2. **Giai đoạn 2: Trích xuất đặc trưng TDA**
   - Áp dụng Topological Data Analysis để tạo TDA-extracted features
   - Tạo temporal graph snapshots với PyTorch Geometric format
   - Generate sequences cho RNN models

3. **Giai đoạn 3: Huấn luyện và đánh giá**
   - **Quick validation**: Sử dụng RNN models để test pipeline nhanh
   - **Best performance**: Sử dụng Temporal GNN (HTGN) cho kết quả tốt nhất
   - **Baseline comparison**: So sánh với Static GNN để đánh giá contribution của temporal information

4. **Giai đoạn 4: Phân tích kết quả**
   - So sánh metrics (ROC-AUC, Accuracy) với baselines
   - Analyze model performance trên các datasets khác nhau
   - Visualize results và training curves

## Lưu ý

### Yêu cầu hệ thống
- **Python**: 3.6+ (khuyến nghị 3.9-3.10)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+ cho large networks như dgd)
- **Storage**: Đủ không gian cho datasets và processed data (có thể vài GB)

### Tương thích phần cứng

**Mac M2 (Apple Silicon)**:
- Đã được patch để hỗ trợ MPS (Metal Performance Shaders)
- Xem chi tiết trong `HUONG_DAN_CAI_DAT_VA_CHAY.md`
- PyTorch 2.0+ required cho MPS support
- Memory tracking sẽ hiển thị 0 MiB (limitation của MPS, không phải bug) - xem troubleshooting trong hướng dẫn chính

**CUDA (NVIDIA GPUs)**:
- Hỗ trợ đầy đủ cho training với GPU acceleration
- Original requirements sử dụng `torch==1.6.0+cu101`

**CPU-only**:
- Có thể chạy trên CPU nhưng sẽ chậm hơn đáng kể
- Khuyến nghị cho small datasets hoặc debugging

### Dependencies quan trọng

**PyTorch Geometric**:
- Có thể cần build từ source trên Mac M2
- Xem hướng dẫn trong `HUONG_DAN_CAI_DAT_VA_CHAY.md`

**kmapper**:
- Cần thiết cho Topological Data Analysis
- Phụ thuộc vào scikit-learn

**TensorFlow**:
- Cần cho RNN models
- Mac M2: Sử dụng `tensorflow-macos` và `tensorflow-metal`

### Cấu hình đường dẫn dữ liệu

Trước khi chạy, cần cập nhật các đường dẫn trong `config.py` hoặc trong `NetworkParser` class:
- `file_path`: Đường dẫn đến thư mục chứa raw network files
- `timeseries_file_path`: Đường dẫn cho time series data
- Đảm bảo các thư mục output có quyền ghi

### Ghi chú về hiệu suất

**Xử lý TDA**:
- Có thể mất nhiều thời gian cho large networks (vài giờ cho networks với nhiều nodes)
- Sử dụng multiprocessing để tăng tốc khi có thể

**Huấn luyện Temporal GNN**:
- Tốn nhiều memory hơn RNN (đặc biệt với temporal window)
- Khuyến nghị sử dụng GPU (CUDA hoặc MPS) cho training hiệu quả
- Large datasets như dgd (720 snapshots) có thể cần GPU memory lớn

**Reproducibility**:
- Trên Mac M2, kết quả có thể khác một chút so với CUDA (do hardware differences)
- Sử dụng `--device_id -1` và `--seed` để đảm bảo reproducibility
- Chênh lệch metrics thường < 0.01-0.02

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
