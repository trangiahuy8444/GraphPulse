# Thư Mục Analyzer

## Giới thiệu

Thư mục `analyzer/` chứa các công cụ phân tích và xử lý dữ liệu mạng lưới cho dự án GraphPulse. Module chính `NetworkParser` cung cấp các chức năng toàn diện để parse, xử lý và chuyển đổi dữ liệu mạng lưới thô thành các định dạng phù hợp cho machine learning, bao gồm cả việc áp dụng Topological Data Analysis (TDA) để trích xuất features.

Thư mục này đóng vai trò quan trọng trong pipeline xử lý dữ liệu của GraphPulse, chuyển đổi raw network data thành:
- Graph features và statistics
- Temporal graph snapshots với PyTorch Geometric format
- TDA-extracted features cho RNN models
- Sequences cho temporal modeling

## Các file chính

### `network_parser.py`

File Python chứa class `NetworkParser` với các phương thức xử lý dữ liệu mạng lưới. Đây là module trung tâm của analyzer package.

**Class NetworkParser** bao gồm các phương thức chính:

1. **`create_graph_features(file)`**: Parse network files và tính toán các graph features (density, diameter, centrality measures, etc.), tạo labels cho classification (live/dead networks), và lưu graphs dưới dạng NetworkX pickle files.

2. **`create_time_series_graphs(file)`**: Xử lý dữ liệu time series cho transaction networks, tạo temporal graph snapshots với sliding window, áp dụng TDA để extract features, và tạo PyTorch Geometric data objects.

3. **`create_time_series_rnn_sequence(file)`**: Tạo sequences từ time series data cho RNN models, extract TDA features cho mỗi time window, và lưu sequences dưới dạng pickle files.

4. **`create_time_series_other_graphs(file)`**: Xử lý các dataset không phải transaction networks (như MathOverflow) với format dữ liệu khác nhau.

5. **`create_time_series_reddit_graphs(file)`**: Xử lý Reddit dataset với format TSV đặc biệt.

**Các phương thức hỗ trợ:**
- `process_TDA_extracted_temporal_features()`: Áp dụng TDA Mapper method để trích xuất topological features
- `create_TDA_graph()`: Tạo TDA-extracted graphs với các parameters khác nhau
- Các phương thức validate và filter data

## Cách sử dụng

### Import Module

```python
from analyzer.network_parser import NetworkParser
```

### Khởi tạo Parser

```python
parser = NetworkParser()
```

**Lưu ý:** Trước khi sử dụng, cần cấu hình các đường dẫn trong class:
- `file_path`: Đường dẫn đến thư mục chứa network files (mặc định: `"../data/all_network/"`)
- `timeseries_file_path`: Đường dẫn cho time series data (mặc định: `"../data/all_network/TimeSeries/"`)
- `timeseries_file_path_other`: Đường dẫn cho datasets khác (mặc định: `"../data/all_network/TimeSeries/Other/"`)
- `timeWindow`: Kích thước cửa sổ thời gian (mặc định: `[7]` ngày)

### Ví dụ Sử Dụng

#### 1. Tạo Graph Features

```python
from analyzer.network_parser import NetworkParser

parser = NetworkParser()
# Parse file và tạo graph features
parser.create_graph_features("networkadex.txt")
```

Output: Tạo file `final_data.csv` với graph statistics và lưu NetworkX graph vào `NetworkxGraphs/`

#### 2. Tạo Time Series Graphs với TDA

```python
parser = NetworkParser()
# Tạo temporal graph snapshots với TDA features
parser.create_time_series_graphs("networkadex.txt")
```

Output: Tạo PyTorch Geometric data objects trong `PygGraphs/TimeSeries/networkadex.txt/` với các thư mục:
- `RawGraph/`: Raw graph snapshots
- `TDA/`: TDA-extracted graphs với các parameters khác nhau
- `TDA_Tuned/`: TDA graphs đã được tune parameters

#### 3. Tạo RNN Sequences

```python
parser = NetworkParser()
# Tạo sequences cho RNN models
parser.create_time_series_rnn_sequence("networkadex.txt")
```

Output: Tạo sequences trong `data/Sequences/networkadex.txt/`:
- `seq_tda.txt`: Sequences với TDA features
- `seq_raw.txt`: Sequences với raw features

#### 4. Xử lý Multiple Files

```python
import os
from analyzer.network_parser import NetworkParser

parser = NetworkParser()
files = os.listdir(parser.file_path)

for file in files:
    if file.endswith(".txt"):
        print(f"Processing {file}...")
        
        # Tạo graph features
        parser.create_graph_features(file)
        
        # Tạo time series graphs
        parser.create_time_series_graphs(file)
        
        # Tạo RNN sequences
        parser.create_time_series_rnn_sequence(file)
```

#### 5. Xử lý Other Networks

```python
# Xử lý MathOverflow dataset
parser.create_time_series_other_graphs("mathoverflow.txt")

# Xử lý Reddit dataset
parser.create_time_series_reddit_graphs("Reddit_B.tsv")
```

## Tham số Quan trọng

Các tham số có thể điều chỉnh trong `NetworkParser`:

- **`windowSize`**: Kích thước cửa sổ thời gian (mặc định: 7 ngày)
- **`gap`**: Khoảng cách giữa data window và label window (mặc định: 3 ngày)
- **`lableWindowSize`**: Kích thước cửa sổ để tạo label (mặc định: 7 ngày)
- **`maxDuration`**: Thời gian tối thiểu của data để được coi là hợp lệ (mặc định: 180 ngày cho time series, 20 ngày cho graph features)
- **`networkValidationDuration`**: Thời gian tối thiểu để validate network (mặc định: 20 ngày)
- **`labelTreshholdPercentage`**: Ngưỡng phần trăm để phân loại live/dead (mặc định: 10%)

## Định Dạng Dữ Liệu Đầu Vào

### Transaction Networks
Format: `from_node to_node timestamp value`
```
node1 node2 1234567890 100.5
node2 node3 1234567900 200.3
```

### Other Networks (MathOverflow, etc.)
Format: `from_node to_node timestamp`
```
user1 user2 1234567890
user2 user3 1234567900
```

### Reddit Networks
Format TSV với columns:
- `SOURCE_SUBREDDIT`
- `TARGET_SUBREDDIT`
- `TIMESTAMP`
- `LINK_SENTIMENT`

## Output Files

### Graph Features Output
1. **`final_data.csv`**: Statistics về các graphs với columns:
   - network, timeframe, start_date
   - num_nodes, num_edges, density, diameter
   - centrality measures (degree, closeness, betweenness)
   - assortativity, clique_number
   - peak, last_dates_trans, label_factor_percentage, label

2. **`NetworkxGraphs/`**: Pickled NetworkX graphs

### Time Series Graphs Output
1. **`PygGraphs/TimeSeries/{network}/RawGraph/`**: Raw PyTorch Geometric data objects
2. **`PygGraphs/TimeSeries/{network}/TDA/`**: TDA-extracted graphs với parameters cơ bản
3. **`PygGraphs/TimeSeries/{network}/TDA_Tuned/`**: TDA-extracted graphs với parameters đã tune

### RNN Sequences Output
1. **`data/Sequences/{network}/seq_tda.txt`**: Sequences với TDA features (5 features)
2. **`data/Sequences/{network}/seq_raw.txt`**: Sequences với raw features (3 features)

## Lưu ý

### Memory và Performance
- **Memory**: Xử lý large networks có thể cần nhiều RAM. Nên xử lý từng file một thay vì batch processing cho networks lớn.
- **Time**: Quá trình TDA extraction có thể mất nhiều thời gian, đặc biệt với networks lớn. Có thể mất vài phút đến vài giờ tùy vào kích thước network.
- **Multiprocessing**: Một số phương thức sử dụng multiprocessing để tăng tốc. Đảm bảo có đủ CPU cores.

### File Management
- Files sẽ được tự động di chuyển vào `Processed/` hoặc `Invalid/` sau khi xử lý
- Đảm bảo các thư mục output tồn tại hoặc có quyền tạo files
- Kiểm tra lại đường dẫn trong `file_path`, `timeseries_file_path` trước khi chạy

### TDA Processing
- Đảm bảo dữ liệu có đủ nodes/edges cho TDA processing
- Có thể cần điều chỉnh các tham số TDA (n_cubes, per_overlap, cls) nếu kết quả không tốt
- TDA features sẽ được extract cho mỗi node trong graph

### Dependencies
Module này yêu cầu:
- `networkx`: Để xử lý graphs
- `pandas`: Để xử lý dữ liệu
- `numpy`: Các thao tác số học
- `kmapper`: Cho Topological Data Analysis
- `sklearn`: Machine learning utilities (MinMaxScaler)
- `torch_geometric`: Để tạo PyTorch Geometric data objects
- `pickle`: Để lưu graphs và sequences

### Troubleshooting

**Lỗi FileNotFoundError:**
- Kiểm tra lại đường dẫn trong `file_path`, `timeseries_file_path`
- Đảm bảo các thư mục output tồn tại hoặc có quyền tạo

**Lỗi Memory:**
- Giảm `maxDuration` hoặc `windowSize`
- Xử lý từng file một thay vì batch processing
- Tăng swap memory nếu có thể

**Lỗi TDA Processing:**
- Kiểm tra dữ liệu có đủ nodes/edges không (tối thiểu vài chục nodes)
- Điều chỉnh các tham số TDA (n_cubes, per_overlap, cls)
- Thử với networks nhỏ hơn trước

**Network không hợp lệ:**
- Networks với duration < `networkValidationDuration` sẽ bị di chuyển vào `Invalid/`
- Kiểm tra format dữ liệu đầu vào
- Đảm bảo timestamps hợp lệ
