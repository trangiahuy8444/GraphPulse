# Thư Mục Data

## Giới thiệu

Thư mục `data/` chứa tất cả dữ liệu mạng lưới (network data) và các sequences đã được xử lý sẵn cho dự án GraphPulse. Đây là nơi lưu trữ:

1. **Raw network data**: Các file mạng lưới thô từ các nguồn khác nhau (Ethereum ERC20 tokens, MathOverflow, Reddit)
2. **Time series data**: Dữ liệu mạng lưới đã được tổ chức theo thời gian
3. **Processed sequences**: Các sequences đã được extract từ TDA và raw features, sẵn sàng cho RNN training

Thư mục này đóng vai trò trung tâm trong pipeline của GraphPulse, cung cấp dữ liệu đầu vào cho cả data processing (analyzer) và model training (models).

## Cấu trúc

```
data/
├── all_network/                    # Raw network data files
│   ├── *.txt                       # Network files (transaction networks)
│   └── TimeSeries/                 # Time series organized data
│       ├── *.txt                   # Time series network files
│       └── Other/                  # Other datasets (MathOverflow, Reddit)
│           ├── mathoverflow.txt
│           └── Reddit_B.rar
└── Sequences/                      # Processed sequences for RNN
    ├── {network_name}/
    │   ├── seq_tda.txt            # TDA-extracted sequences
    │   └── seq_raw.txt            # Raw feature sequences
```

## Các file và thư mục chính

### `all_network/`

Thư mục chứa các file mạng lưới thô (raw network files) từ các nguồn khác nhau.

**Transaction Networks** (Ethereum ERC20 tokens):
- `networkadex.txt`, `networkaragon.txt`, `networkbancor.txt`
- `networkcentra.txt`, `networkcoindash.txt`, `networkdgd.txt`
- `networkiconomi.txt`

**Format**: Mỗi dòng là một transaction với format:
```
from_node to_node timestamp value
```

**Other Networks**:
- `mathoverflow.txt`: MathOverflow Q&A network
- Format: `from_node to_node timestamp`

### `all_network/TimeSeries/`

Thư mục chứa dữ liệu mạng lưới đã được tổ chức theo thời gian, sẵn sàng cho time series processing.

**Time Series Network Files**:
- Các file tương ứng với networks trong `all_network/` nhưng đã được sắp xếp và tổ chức theo timeline
- Được sử dụng bởi `analyzer/network_parser.py` để tạo temporal graph snapshots

**TimeSeries/Other/**:
- `mathoverflow.txt`: MathOverflow time series data
- `Reddit_B.rar`: Reddit network data (nén) - cần giải nén trước khi sử dụng

### `Sequences/`

Thư mục chứa các sequences đã được xử lý sẵn từ TDA và raw features, được lưu dưới dạng pickle files.

**Cấu trúc**: Mỗi network có một thư mục riêng:
- `{network_name}/seq_tda.txt`: Sequences với TDA features (5 features per timestep)
- `{network_name}/seq_raw.txt`: Sequences với raw features (3 features per timestep)

**Format của sequence files**:
- Là pickle files chứa dictionary với keys:
  - `"sequence"`: Dictionary chứa sequences với các keys là parameter configurations
  - `"label"`: List các labels tương ứng (binary classification)

**Networks có sẵn sequences**:
- `mathoverflow.txt/`
- `networkadex.txt/`
- `networkaragon.txt/`
- `networkbancor.txt/`
- `networkcentra.txt/`
- `networkcoindash.txt/`
- `networkdgd.txt/`
- `networkiconomi.txt/`
- `Reddit_B.tsv/`

## Cách sử dụng

### 1. Sử dụng Raw Network Data

**Cho analyzer/network_parser.py:**

```python
from analyzer.network_parser import NetworkParser

parser = NetworkParser()
# Đảm bảo file_path trỏ đúng đến data/all_network/
parser.file_path = "../data/all_network/"

# Xử lý một network file
parser.create_graph_features("networkadex.txt")
```

### 2. Sử dụng Time Series Data

```python
parser = NetworkParser()
parser.timeseries_file_path = "../data/all_network/TimeSeries/"

# Tạo temporal graph snapshots
parser.create_time_series_graphs("networkadex.txt")
```

### 3. Sử dụng Processed Sequences

**Cho RNN models:**

```python
import pickle
import os

# Đọc sequence data
network_name = "networkadex.txt"
file_path = f"../data/Sequences/{network_name}/"

# Đọc TDA sequences
with open(file_path + "seq_tda.txt", 'rb') as f:
    tda_data = pickle.load(f)

# Đọc raw sequences
with open(file_path + "seq_raw.txt", 'rb') as f:
    raw_data = pickle.load(f)

# Sử dụng sequences
sequences = tda_data["sequence"]
labels = tda_data["label"]
```

**Trong rnn_methods.py:**

```python
from models.rnn.rnn_methods import read_seq_data_by_file_name

# Đọc sequences (đường dẫn được hardcode trong function)
data = read_seq_data_by_file_name("networkadex.txt", "seq_tda.txt")
data_raw = read_seq_data_by_file_name("networkadex.txt", "seq_raw.txt")
```

### 4. Xử lý Reddit Data

```python
# Giải nén file RAR trước
# Sau đó sử dụng parser
parser.create_time_series_reddit_graphs("Reddit_B.tsv")
```

## Lưu ý

### Data Paths

- **Relative paths**: Nhiều scripts sử dụng relative paths. Đảm bảo chạy từ đúng thư mục hoặc cập nhật paths trong `config.py`
- **Hardcoded paths**: Một số functions có hardcoded paths (như trong `rnn_methods.py`). Cần kiểm tra và cập nhật nếu cần

### Data Format Requirements

**Transaction Networks:**
- Format: `from_node to_node timestamp value`
- Timestamp: Unix timestamp (seconds)
- Value: Numeric value (transaction amount)

**Other Networks:**
- Format: `from_node to_node timestamp`
- Hoặc TSV format cho Reddit với columns: `SOURCE_SUBREDDIT`, `TARGET_SUBREDDIT`, `TIMESTAMP`, `LINK_SENTIMENT`

### Data Validation

- Networks phải có ít nhất 20 ngày dữ liệu để được coi hợp lệ
- Data sẽ được tự động validate bởi `NetworkParser`
- Invalid networks sẽ được di chuyển vào `Invalid/` folder

### Storage Requirements

- **Raw data**: Có thể chiếm vài GB tùy vào số lượng networks
- **Processed sequences**: Thường nhỏ hơn raw data nhưng vẫn có thể lớn với networks lớn
- **Temporal graphs**: Có thể rất lớn, đặc biệt với nhiều snapshots

### Data Organization

- **Chronological order**: Sequences và time series data được sắp xếp theo thời gian
- **Train-test split**: Mặc định 80-20 split, train data đến trước, test data đến sau
- **Labels**: Binary classification (0 = dead network, 1 = live network)

### Performance Notes

- **Reading sequences**: Sử dụng pickle nên đọc nhanh, nhưng files có thể lớn
- **Memory**: Loading toàn bộ sequences vào memory có thể tốn nhiều RAM
- **Processing time**: Xử lý raw data thành sequences có thể mất vài giờ cho networks lớn

### Data Preprocessing

Trước khi sử dụng data:
1. Kiểm tra format của files
2. Đảm bảo timestamps hợp lệ
3. Validate data quality (check for missing values, outliers)
4. Xử lý Reddit data: giải nén RAR file trước

### Backup và Version Control

- **Không commit large files**: Các file data lớn thường không nên commit vào git
- **Sử dụng .gitignore**: Đảm bảo data files được ignore
- **Backup**: Nên có backup của processed sequences vì quá trình tạo lại tốn thời gian
