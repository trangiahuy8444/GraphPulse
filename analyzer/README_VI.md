# Thư Mục Analyzer

## Giới thiệu

Thư mục `analyzer/` chứa các công cụ phân tích và tiền xử lý dữ liệu mạng lưới cho dự án GraphPulse. Module chính `NetworkParser` cung cấp một tập hợp các phương thức toàn diện để parse, xử lý và chuyển đổi dữ liệu mạng lưới thô (raw network data) thành các định dạng phù hợp cho machine learning, bao gồm việc áp dụng Topological Data Analysis (TDA) để trích xuất các đặc trưng topo học.

Thư mục này đóng vai trò quan trọng trong pipeline xử lý dữ liệu của GraphPulse, thực hiện chuyển đổi từ raw network data sang:
- Graph features và các thống kê mạng (network statistics)
- Temporal graph snapshots với định dạng PyTorch Geometric
- TDA-extracted features cho các mô hình RNN
- Sequences cho temporal modeling

## Các file chính

### `network_parser.py`

File Python chứa class `NetworkParser` với các phương thức xử lý dữ liệu mạng lưới. Đây là module trung tâm của analyzer package, thực hiện toàn bộ quá trình tiền xử lý từ raw files đến dữ liệu sẵn sàng cho training.

**Class NetworkParser** bao gồm các phương thức chính:

1. **`create_graph_features(file)`**: Parse các network files và tính toán các đặc trưng graph (density, diameter, các measures về centrality, etc.), tạo labels cho classification task (live/dead networks), và lưu các graphs dưới dạng NetworkX pickle files. Phương thức này tạo ra các thống kê tổng hợp cho từng network.

2. **`create_time_series_graphs(file)`**: Xử lý dữ liệu time series cho transaction networks, tạo các temporal graph snapshots sử dụng sliding window approach, áp dụng TDA để trích xuất features, và tạo các PyTorch Geometric data objects. Đây là bước quan trọng để chuẩn bị dữ liệu cho Temporal GNN models.

3. **`create_time_series_rnn_sequence(file)`**: Tạo các sequences từ time series data cho RNN models, trích xuất TDA features cho mỗi time window, và lưu sequences dưới dạng pickle files. Sequences này được sử dụng trực tiếp cho việc huấn luyện các mô hình RNN.

4. **`create_time_series_other_graphs(file)`**: Xử lý các dataset không phải transaction networks (như MathOverflow) với format dữ liệu khác nhau, không có trường `value` trong edges.

5. **`create_time_series_reddit_graphs(file)`**: Xử lý Reddit dataset với format TSV đặc biệt, bao gồm thông tin về sentiment của links.

**Các phương thức hỗ trợ:**
- `process_TDA_extracted_temporal_features()`: Áp dụng TDA Mapper method để trích xuất topological features từ các node embeddings
- `create_TDA_graph()`: Tạo các TDA-extracted graphs với các parameters khác nhau (n_cubes, per_overlap, clustering algorithms)
- Các phương thức validate và filter data để đảm bảo chất lượng dữ liệu

## Cách sử dụng

### Import Module

```python
from analyzer.network_parser import NetworkParser
```

### Khởi tạo Parser

```python
parser = NetworkParser()
```

**Lưu ý quan trọng:** Trước khi sử dụng, cần cấu hình các đường dẫn trong class để đảm bảo module có thể truy cập đúng các thư mục dữ liệu:
- `file_path`: Đường dẫn đến thư mục chứa raw network files (mặc định: `"../data/all_network/"`)
- `timeseries_file_path`: Đường dẫn cho time series data (mặc định: `"../data/all_network/TimeSeries/"`)
- `timeseries_file_path_other`: Đường dẫn cho các datasets khác như MathOverflow, Reddit (mặc định: `"../data/all_network/TimeSeries/Other/"`)
- `timeWindow`: Kích thước cửa sổ thời gian cho việc tạo snapshots (mặc định: `[7]` ngày)

### Ví dụ Sử Dụng

#### 1. Trích xuất Graph Features

```python
from analyzer.network_parser import NetworkParser

parser = NetworkParser()

# Parse file và tạo graph features
# Ví dụ với dataset dgd
parser.create_graph_features("networkdgd.txt")
```

**Output**: 
- File `final_data.csv` chứa các graph statistics (density, diameter, centrality measures, etc.)
- NetworkX graph objects được lưu trong `NetworkxGraphs/` dưới dạng pickle files

#### 2. Tạo Temporal Graph Snapshots với TDA Features

```python
parser = NetworkParser()

# Tạo temporal graph snapshots với TDA features
# Quá trình này bao gồm:
# - Sliding window generation
# - TDA feature extraction
# - PyTorch Geometric data object creation
parser.create_time_series_graphs("networkdgd.txt")
```

**Output**: Tạo các PyTorch Geometric data objects trong `PygGraphs/TimeSeries/networkdgd.txt/` với cấu trúc:
- `RawGraph/`: Raw graph snapshots chưa qua TDA processing
- `TDA/`: TDA-extracted graphs với các parameters cơ bản
- `TDA_Tuned/`: TDA graphs đã được tune parameters (sử dụng cho training)

#### 3. Tạo RNN Sequences

```python
parser = NetworkParser()

# Tạo sequences cho RNN models
# Sequences này được sử dụng trực tiếp trong models/rnn/
parser.create_time_series_rnn_sequence("networkdgd.txt")
```

**Output**: Tạo sequence files trong `data/Sequences/networkdgd.txt/`:
- `seq_tda.txt`: Sequences với TDA features (5 features per timestep)
- `seq_raw.txt`: Sequences với raw features (3 features per timestep)

#### 4. Xử lý Batch Files

```python
import os
from analyzer.network_parser import NetworkParser

parser = NetworkParser()
files = os.listdir(parser.file_path)

# Xử lý từng file một (khuyến nghị cho large networks)
for file in files:
    if file.endswith(".txt"):
        print(f"Đang xử lý {file}...")
        
        # Bước 1: Tạo graph features và statistics
        parser.create_graph_features(file)
        
        # Bước 2: Tạo temporal graph snapshots với TDA
        parser.create_time_series_graphs(file)
        
        # Bước 3: Tạo RNN sequences
        parser.create_time_series_rnn_sequence(file)
        
        print(f"Hoàn thành xử lý {file}")
```

#### 5. Xử lý các Datasets Đặc Biệt

```python
# Xử lý MathOverflow dataset (không có value field)
parser.create_time_series_other_graphs("mathoverflow.txt")

# Xử lý Reddit dataset với format TSV
# Lưu ý: Cần giải nén file RAR trước nếu cần
parser.create_time_series_reddit_graphs("Reddit_B.tsv")
```

## Tham số Quan trọng

Các tham số có thể điều chỉnh trong `NetworkParser` để tùy chỉnh quá trình xử lý:

- **`windowSize`**: Kích thước cửa sổ thời gian để tạo mỗi snapshot (mặc định: 7 ngày). Giá trị này ảnh hưởng đến độ chi tiết temporal information.

- **`gap`**: Khoảng cách thời gian giữa data window và label window (mặc định: 3 ngày). Gap này giúp tránh data leakage trong quá trình prediction.

- **`lableWindowSize`**: Kích thước cửa sổ thời gian để tạo labels (mặc định: 7 ngày). Label được tạo bằng cách so sánh số lượng transactions giữa data window và label window.

- **`maxDuration`**: Thời gian tối thiểu của data để được coi là hợp lệ (mặc định: 180 ngày cho time series processing, 20 ngày cho graph features). Networks không đáp ứng điều kiện này sẽ bị đánh dấu invalid.

- **`networkValidationDuration`**: Thời gian tối thiểu để validate một network (mặc định: 20 ngày). Đây là threshold để xác định network có đủ dữ liệu để phân tích.

- **`labelTreshholdPercentage`**: Ngưỡng phần trăm để phân loại live/dead networks (mặc định: 10%). Networks với tỷ lệ transactions cuối so với peak < threshold được phân loại là "dead".

## Định Dạng Dữ Liệu Đầu Vào

### Transaction Networks (Ethereum ERC20 tokens)
Format: `from_node to_node timestamp value`
```
node1 node2 1234567890 100.5
node2 node3 1234567900 200.3
```

**Lưu ý**: 
- Timestamp là Unix timestamp (seconds)
- Value là numeric, có thể được normalize trong quá trình xử lý
- Ví dụ datasets: `networkdgd.txt`, `networkadex.txt`, `networkaragon.txt`

### Other Networks (MathOverflow, etc.)
Format: `from_node to_node timestamp`
```
user1 user2 1234567890
user2 user3 1234567900
```

**Đặc điểm**: Không có trường `value`, mỗi edge có weight = 1.

### Reddit Networks
Format TSV với các columns:
- `SOURCE_SUBREDDIT`: Subreddit nguồn
- `TARGET_SUBREDDIT`: Subreddit đích
- `TIMESTAMP`: Timestamp của interaction
- `LINK_SENTIMENT`: Sentiment của link (1 hoặc -1)

## Output Files

### Graph Features Output

1. **`final_data.csv`**: File CSV chứa các thống kê về graphs với các columns:
   - `network`: Tên network file
   - `timeframe`: Time window được sử dụng
   - `start_date`: Ngày bắt đầu của timeframe
   - `num_nodes`, `num_edges`: Số lượng nodes và edges
   - `density`: Mật độ của graph
   - `diameter`: Đường kính của graph
   - `avg_shortest_path_length`: Độ dài đường đi ngắn nhất trung bình
   - Centrality measures: `max_degree_centrality`, `min_degree_centrality`, `max_closeness_centrality`, `min_closeness_centrality`, `max_betweenness_centrality`, `min_betweenness_centrality`
   - `assortativity`: Hệ số assortativity
   - `clique_number`: Số lượng clique lớn nhất
   - `peak`: Peak transactions count
   - `last_dates_trans`: Tổng transactions trong các ngày cuối
   - `label_factor_percentage`: Tỷ lệ phần trăm cho labeling
   - `label`: Binary label (0 = dead, 1 = live)

2. **`NetworkxGraphs/`**: Thư mục chứa các NetworkX graph objects được serialize dưới dạng pickle files, có thể được load lại để phân tích tiếp.

### Time Series Graphs Output

1. **`PygGraphs/TimeSeries/{network}/RawGraph/`**: 
   - Raw PyTorch Geometric data objects chưa qua TDA processing
   - Mỗi file đại diện cho một snapshot trong timeline

2. **`PygGraphs/TimeSeries/{network}/TDA/`**: 
   - TDA-extracted graphs với các parameters cơ bản
   - Được tạo với các cấu hình TDA mặc định

3. **`PygGraphs/TimeSeries/{network}/TDA_Tuned/`**: 
   - TDA-extracted graphs với parameters đã được tune
   - **Lưu ý**: Đây là version được sử dụng cho training trong `models/static_gnn/`

### RNN Sequences Output

1. **`data/Sequences/{network}/seq_tda.txt`**: 
   - Sequences với TDA features (5 features per timestep)
   - Format: Dictionary với keys là TDA parameters và values là sequences

2. **`data/Sequences/{network}/seq_raw.txt`**: 
   - Sequences với raw features (3 features per timestep)
   - Bao gồm các basic graph statistics

**Format của sequence files**:
```python
{
    "sequence": {
        "Overlap_xx_Ncube_x": [
            [[feat1, feat2, feat3, feat4, feat5], ...],  # Timestep 1
            [[feat1, feat2, feat3, feat4, feat5], ...],  # Timestep 2
            ...
        ],
        "raw": [...]
    },
    "label": [0, 1, 0, 1, ...]  # Binary labels
}
```

## Lưu ý

### Memory và Performance

- **Memory Requirements**: Xử lý large networks như `dgd` (720 snapshots) có thể cần nhiều RAM (khuyến nghị 16GB+). Nên xử lý từng file một thay vì batch processing cho networks lớn để tránh memory overflow.

- **Processing Time**: Quá trình TDA extraction có thể mất nhiều thời gian, đặc biệt với networks lớn. Có thể mất vài phút đến vài giờ tùy vào:
  - Số lượng nodes trong network
  - Kích thước window và số lượng snapshots
  - Cấu hình TDA parameters (n_cubes, clustering algorithm)

- **Multiprocessing**: Một số phương thức sử dụng multiprocessing để tăng tốc. Đảm bảo hệ thống có đủ CPU cores để tận dụng parallel processing.

**Ví dụ với dataset dgd**:
- Tổng snapshots: 720
- Processing time: Có thể mất vài giờ cho toàn bộ pipeline
- Memory peak: Có thể đạt 8-12GB trong quá trình TDA extraction

### File Management

- Files sẽ được tự động di chuyển vào `Processed/` hoặc `Invalid/` sau khi xử lý để tránh xử lý trùng lặp
- Đảm bảo các thư mục output (`NetworkxGraphs/`, `PygGraphs/`, `data/Sequences/`) tồn tại hoặc có quyền tạo files
- Kiểm tra lại đường dẫn trong `file_path`, `timeseries_file_path` trước khi chạy, đặc biệt quan trọng trên các hệ thống khác nhau

### TDA Processing

- **Data Requirements**: Đảm bảo dữ liệu có đủ nodes/edges cho TDA processing (tối thiểu vài chục nodes). Networks quá nhỏ có thể không tạo được meaningful TDA features.

- **Parameter Tuning**: Có thể cần điều chỉnh các tham số TDA nếu kết quả không tốt:
  - `n_cubes`: Số lượng hypercubes trong Mapper (ảnh hưởng đến độ chi tiết)
  - `per_overlap`: Phần trăm overlap giữa các cubes (ảnh hưởng đến connectivity)
  - `cls`: Clustering algorithm (DBSCAN, KMeans, etc.)

- **Feature Extraction**: TDA features sẽ được extract cho mỗi node trong graph, bao gồm cluster membership và cluster properties.

### Dependencies

Module này yêu cầu các thư viện sau:
- `networkx`: Để xử lý và thao tác graphs
- `pandas`: Để xử lý và manipulate dữ liệu tabular
- `numpy`: Các thao tác số học và array operations
- `kmapper`: Cho Topological Data Analysis (KeplerMapper implementation)
- `sklearn`: Machine learning utilities (MinMaxScaler cho normalization)
- `torch_geometric`: Để tạo PyTorch Geometric data objects
- `pickle`: Để serialize và deserialize graph objects và sequences

**Lưu ý Mac M2**: Tất cả dependencies này đều tương thích với Mac M2. Không có yêu cầu đặc biệt nào cho việc xử lý dữ liệu trên Mac M2, vì phần lớn các operations là CPU-based.

### Troubleshooting

**Lỗi FileNotFoundError:**
- Kiểm tra lại đường dẫn trong `file_path`, `timeseries_file_path`
- Đảm bảo các thư mục output tồn tại hoặc có quyền tạo
- Trên Mac M2, đảm bảo sử dụng relative paths đúng hoặc absolute paths

**Lỗi Memory (Out of Memory):**
- Giảm `maxDuration` hoặc `windowSize` để giảm số lượng snapshots
- Xử lý từng file một thay vì batch processing
- Tăng swap memory nếu có thể
- Với dataset lớn như dgd, có thể cần xử lý theo batches của snapshots

**Lỗi TDA Processing:**
- Kiểm tra dữ liệu có đủ nodes/edges không (tối thiểu vài chục nodes)
- Điều chỉnh các tham số TDA (n_cubes, per_overlap, cls) để phù hợp với network size
- Thử với networks nhỏ hơn trước để validate pipeline
- Kiểm tra logs để xem có warnings về clustering failures không

**Network không hợp lệ:**
- Networks với duration < `networkValidationDuration` sẽ bị di chuyển vào `Invalid/`
- Kiểm tra format dữ liệu đầu vào (columns, separators)
- Đảm bảo timestamps hợp lệ và theo đúng format Unix timestamp
- Verify rằng network có đủ edges để tạo meaningful graph structure

**Performance Issues trên Mac M2:**
- TDA processing có thể chậm hơn một chút so với CUDA systems (do chủ yếu là CPU-bound)
- Multiprocessing sẽ tận dụng được nhiều cores trên Mac M2
- Consider xử lý trong background nếu cần
