# Thư Mục RNN Models

## Giới thiệu

Thư mục `models/rnn/` chứa implementation của các mô hình Recurrent Neural Network (RNN) cho sequence processing trong GraphPulse. Các mô hình này xử lý sequences được extract từ temporal graphs, sử dụng TDA (Topological Data Analysis) features hoặc raw features để dự đoán evolution của networks.

RNN models trong GraphPulse được thiết kế để:
- Xử lý sequences với TDA-extracted features (5 features)
- Xử lý sequences với raw features (3 features)
- Kết hợp cả hai loại features (8 features) - GraphPulse approach
- Thực hiện binary classification (live/dead networks)

## Các file chính

### `rnn_methods.py`

File Python chứa toàn bộ implementation của RNN models, bao gồm:

**Class AUCCallback:**
- Custom callback cho Keras để track ROC-AUC score trong quá trình training
- Lưu trữ AUC scores qua các epochs
- Cung cấp methods để tính average và standard deviation của AUC

**Functions chính:**

1. **`read_seq_data(network)`**: Đọc sequence data từ file pickle trong thư mục `Sequence/{network}/seq.txt`.

2. **`read_seq_data_by_file_name(network, file)`**: Đọc sequence data từ file cụ thể (seq_tda.txt hoặc seq_raw.txt) cho một network. File path được hardcode trong function.

3. **`train_test_split_sequential(*arrays, train_size)`**: Sequential train-test split (80-20) thay vì random split để preserve temporal order. Quan trọng cho time series data.

4. **`reset_random_seeds()`**: Reset random seeds cho reproducibility (Python hash seed, TensorFlow seed, NumPy seed).

5. **`LSTM_classifier(data, labels, spec, network)`**: Function chính để train và evaluate LSTM model.
   - Tạo LSTM model với architecture phù hợp dựa trên spec (TDA5, Raw, Combined, GraphPulse, etc.)
   - Architecture: LSTM(64) → LSTM(32) → GRU(32) → GRU(32) → Dense(100) → Dense(1)
   - Sử dụng Adam optimizer với learning rate tùy chỉnh cho từng network
   - Training 100 epochs với early stopping
   - Trả về ROC-AUC score
   - Lưu results vào `RnnResults/RNN-Results.txt`

**Main execution:**
- Chạy trên nhiều networks khác nhau
- Support multiple runs (5 runs) để tính average
- Xử lý cả TDA và raw sequences
- Tạo combined sequences (GraphPulse approach)
- Support ablation studies
- Normalization options: "all" hoặc "per_column"

## Cách sử dụng

### Import Module

```python
from models.rnn.rnn_methods import LSTM_classifier, read_seq_data_by_file_name, reset_random_seeds
```

### Ví dụ Sử Dụng Cơ Bản

```python
import numpy as np
from models.rnn.rnn_methods import LSTM_classifier, read_seq_data_by_file_name

# Đọc sequence data từ processed files
network = "networkdgd.txt"
data = read_seq_data_by_file_name(network, "seq_tda.txt")
data_raw = read_seq_data_by_file_name(network, "seq_raw.txt")

# Chuẩn bị dữ liệu cho training
np_data = np.array(data["sequence"]["Overlap_xx_Ncube_x"])
np_labels = np.array(data["label"])

# Normalization dữ liệu (Min-Max scaling)
min_val = np.min(np_data)
max_val = np.max(np_data)
normalized_data = (np_data - min_val) / (max_val - min_val)
normalized_data = np.nan_to_num(normalized_data, nan=0)

# Huấn luyện và đánh giá mô hình
auc_score = LSTM_classifier(normalized_data, np_labels, "TDA5", network)
print(f"ROC-AUC Score: {auc_score:.4f}")
```

### Sử dụng với GraphPulse (Combined Features)

```python
# Combine TDA và raw features
concatenated = np.concatenate((normalized_tda_data, normalized_raw_data), axis=2)

# Train với combined features
auc = LSTM_classifier(concatenated, np_labels, "GraphPulse", network)
```

### Chạy Full Pipeline (như trong main)

```bash
# Chạy file trực tiếp để thực hiện full pipeline
cd models/rnn
python rnn_methods.py
```

**Lưu ý Mac M2**: RNN models sử dụng TensorFlow/Keras. Trên Mac M2:
```bash
# Đảm bảo đã cài đặt TensorFlow-Metal
pip install tensorflow-macos tensorflow-metal

# Sau đó chạy training
python rnn_methods.py
```

Script sẽ tự động thực hiện:
- Xử lý tất cả networks trong `networkList`
- Chạy 5 runs (repetitions) cho mỗi network để tính average metrics
- Test các approaches: TDA5, Raw, Combined, GraphPulse
- Lưu results vào `RnnResults/RNN-Results.txt`

## Model Architecture

### Input Shape

- **TDA5**: `(batch_size, 7, 5)` - 5 TDA features cho 7 timesteps
- **Raw**: `(batch_size, 7, 3)` - 3 raw features cho 7 timesteps
- **Combined/GraphPulse**: `(batch_size, 7, 8)` - 8 combined features
- **TDA3**: `(batch_size, 7, 3)` - 3 TDA features (ablation)
- **GraphPulse ablation**: `(batch_size, 7, 4)` - 4 features (removed one)

### Architecture Details

```
Input (7 timesteps, n_features)
  ↓
LSTM(64, return_sequences=True)
  ↓
LSTM(32, activation='relu', return_sequences=True)
  ↓
GRU(32, activation='relu', return_sequences=True)
  ↓
GRU(32, activation='relu', return_sequences=False)
  ↓
Dense(100, activation='relu')
  ↓
Dense(1, activation='sigmoid')
  ↓
Output (binary classification)
```

### Hyperparameters

**Learning Rates** (tùy chỉnh cho từng network):
- `networkiconomi.txt`: 0.00001
- `networkcentra.txt`: 0.00004
- `networkbancor.txt`: 0.00005
- `networkaragon.txt`: 0.00001
- `Reddit_B.tsv`: 0.00005
- Default: 0.0001

**Training Parameters**:
- Epochs: 100
- Loss: binary_crossentropy
- Optimizer: Adam
- Metrics: AUC, accuracy
- Validation: 20% của data (sequential split)

## Data Format

### Sequence Data Structure

Sequence files (pickle format) chứa dictionary:
```python
{
    "sequence": {
        "Overlap_xx_Ncube_x": [
            [[feat1, feat2, ..., feat5], ...],  # Timestep 1
            [[feat1, feat2, ..., feat5], ...],  # Timestep 2
            ...
        ],
        "raw": [...]
    },
    "label": [0, 1, 0, 1, ...]  # Binary labels
}
```

### Normalization

Hai loại normalization:
1. **"all"**: Normalize toàn bộ data cùng lúc
2. **"per_column"**: Normalize từng feature column riêng biệt
3. **"auto"**: Tự động chọn dựa trên network (centra và Reddit dùng per_column)

## Output

### Results File

Results được lưu vào `RnnResults/RNN-Results.txt` với format:
```
Network,Spec,Loss,Accuracy,AUC,ROC-AUC,Avg-AUC,Std-AUC,Training-Time,Data-Size,Learning-Rate
```

### Metrics

- **Loss**: Binary crossentropy loss
- **Accuracy**: Classification accuracy
- **AUC**: Area Under Curve từ Keras
- **ROC-AUC**: ROC-AUC từ scikit-learn (sử dụng trong paper)
- **Avg-AUC**: Average AUC qua các epochs
- **Std-AUC**: Standard deviation của AUC

## Lưu ý

### Dependencies

- **TensorFlow/Keras**: Framework cho RNN implementation (LSTM, GRU)
- **NumPy**: Data manipulation và numerical operations
- **scikit-learn**: Metrics evaluation (ROC-AUC calculation)
- **Pickle**: Đọc sequence data từ processed files

**Mac M2 Specific:**
- **TensorFlow-Metal**: Apple's optimized TensorFlow cho Mac với Metal acceleration
- Cài đặt: `pip install tensorflow-macos tensorflow-metal`
- Models sẽ tự động sử dụng Metal acceleration nếu available

### Hardware Requirements

- **CPU**: Có thể chạy trên CPU nhưng sẽ chậm hơn đáng kể
- **GPU**: Không bắt buộc nhưng sẽ nhanh hơn đáng kể (CUDA hoặc Metal trên Mac)
- **Memory**: Cần đủ RAM để load sequences vào memory (có thể vài GB cho large networks như dgd)
- **Mac M2**: TensorFlow-Metal sẽ tận dụng GPU của Apple Silicon để accelerate training

### Data Paths

**Quan trọng**: Function `read_seq_data_by_file_name` có hardcoded path:
```python
file_path = "C:/Users/kiara/Desktop/MyProject/GraphPulse/data/Sequences/{}/"
```

Cần cập nhật path này trong code trước khi sử dụng!

### Performance

- **Training time**: Tùy thuộc vào data size, có thể vài phút đến vài giờ
- **Memory usage**: Sequences được load toàn bộ vào memory
- **Early stopping**: Model sẽ stop nếu không cải thiện (thực tế không có early stopping trong code, train full 100 epochs)

### Reproducibility

- Sử dụng `reset_random_seeds()` để đảm bảo reproducibility
- Tuy nhiên, TensorFlow/Keras có thể có non-deterministic operations
- Đảm bảo set `TF_DETERMINISTIC_OPS=1` environment variable nếu cần exact reproducibility

### Ablation Studies

Code hỗ trợ ablation studies bằng cách:
- Remove một feature tại một thời điểm
- Test với TDA3 (remove 2 features)
- Test với GraphPulse ablation (remove từng feature)

### Troubleshooting

**Memory Error**:
- Giảm batch size (mặc định là toàn bộ data)
- Process từng network một
- Giảm sequence length nếu có thể

**Low AUC**:
- Kiểm tra data quality
- Thử different learning rates
- Thử different normalization methods
- Check if labels are balanced

**FileNotFoundError**:
- Update hardcoded path trong `read_seq_data_by_file_name`
- Đảm bảo sequence files đã được tạo bởi analyzer
