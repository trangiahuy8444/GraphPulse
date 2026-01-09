# Thư Mục Util

## Giới thiệu

Thư mục `util/` chứa các utility functions và helper modules được sử dụng xuyên suốt dự án GraphPulse. Các modules này cung cấp các chức năng hỗ trợ cho data processing, graph manipulation, visualization, và data preparation cho các mô hình khác nhau.

Thư mục này đóng vai trò là thư viện các công cụ tiện ích, được import và sử dụng bởi:
- `analyzer/`: Xử lý và phân tích dữ liệu
- `models/`: Chuẩn bị dữ liệu cho training
- Các scripts khác trong dự án

## Các file chính

### `file_util.py`

Module chứa các utility functions để làm việc với files và dictionaries.

**Các functions chính:**

1. **`write_list_to_csv(filename, data_list)`**: Ghi một list vào file CSV, mỗi item một dòng.

2. **`read_dicts_from_files(directory_path)`**: Đọc các dictionary từ nhiều files trong một thư mục và merge chúng lại. Sử dụng pickle để đọc/write.

3. **`merge_dicts_with_same_keys(dict_list)`**: Merge nhiều dictionaries có cùng keys thành một dictionary duy nhất. Nếu key đã tồn tại, append value vào list.

4. **`find_indices_of_items_with_incorrect_size(list_3d)`**: Tìm các indices của items có kích thước không đúng (thường dùng để validate data shape).

5. **`filter_valid_sublists(list_3d)`**: Lọc các sublists hợp lệ (có length = 5) từ một list 3D.

6. **`merge_dicts(list_of_dicts)`**: Merge dictionaries với validation, loại bỏ outliers. Trả về merged dictionary và list các outlier indices.

7. **`merge_dicts_old(list_of_dicts)`**: Version cũ của merge_dicts, merge đơn giản hơn.

8. **`output_cleaner()`**: Clean và format output từ RNN results file, chuyển đổi sang CSV format.

9. **`save_fixed_seq(data, path)`**: Lưu sequence data đã được fix (loại bỏ outliers) vào file pickle.

### `graph_util.py`

Module chứa các utility functions để làm việc với graphs, network analysis, và data processing.

**Các functions chính:**

1. **`read_and_merge_node_network_count(file_path)`**: Đọc và merge node network counts từ nhiều files trong thư mục `NodeExistenceMatrix/`.

2. **`process_motifs(file)`**: Xử lý motifs từ một network file, tính toán và lưu statistics.

3. **`from_networkx(G, label, ...)`**: Chuyển đổi một NetworkX graph sang PyTorch Geometric `Data` object. Hỗ trợ:
   - Node attributes grouping
   - Edge attributes grouping
   - Directed/undirected graphs
   - MultiGraph/MultiDiGraph

4. **`get_daily_node_avg(file)`**: Tính toán số lượng nodes trung bình mỗi ngày cho một network file.

5. **`get_daily_avg(file)`**: Tính toán các statistics trung bình hàng ngày (nodes, edges, transactions) cho networks khác (MathOverflow, etc.).

6. **`get_daily_avg_reddit(file)`**: Tính toán statistics trung bình hàng ngày cho Reddit dataset với format TSV đặc biệt.

7. **`process_data_duration(self, file)`**: Xử lý và tính toán duration của data cho một network file.

8. **`create_node_token_network_count(file, bucket)`**: Tạo hash map đếm số lần một node xuất hiện trong các networks khác nhau.

9. **`multiprocess_node_network_count()`**: Sử dụng multiprocessing để xử lý node network count cho nhiều buckets.

10. **`process_bucket_node_network_count(bucket)`**: Xử lý node network count cho một bucket cụ thể.

### `temporal_gnn_data_util.py`

Module chứa các utility functions đặc biệt cho việc chuẩn bị dữ liệu cho Temporal GNN models.

**Các functions chính:**

1. **`creatBaselineDatasets(self, file)`**: Tạo baseline datasets cho temporal GNN với sliding window approach. Tạo:
   - Graph data windows (7 ngày)
   - Label windows (7 ngày, cách 3 ngày sau data window)
   - Labels: 1 nếu transactions tăng, 0 nếu giảm
   - Lưu vào CSV với snapshot indices

2. **`creatBaselineDatasetsOther(self, file)`**: Tương tự `creatBaselineDatasets` nhưng cho other networks (MathOverflow, etc.) với format khác.

3. **`creatBaselineDatasetsReddit(self, file)`**: Tạo baseline datasets cho Reddit network với format TSV đặc biệt và offset time (800 ngày).

**Parameters cho baseline datasets:**
- `windowSize`: 7 ngày (kích thước data window)
- `gap`: 3 ngày (khoảng cách giữa data và label window)
- `lableWindowSize`: 7 ngày (kích thước label window)
- `maxDuration`: 20 ngày (thời gian tối thiểu)

### `visualization.py`

Module chứa các functions để visualize data, results, và statistics.

**Các functions visualization:**

1. **`create_hist_for_node_network_count(file_path)`**: Tạo histogram cho node network count distribution.

2. **`visualize_time_exp()`**: Visualize thời gian training của RNN với scatter plot và regression line.

3. **`visualize_time_exp_bar()`**: Visualize thời gian training RNN với bar chart, sắp xếp từ cao đến thấp.

4. **`visualize_time_exp_bar_method()`**: So sánh thời gian training giữa các methods khác nhau.

5. **`visualize_time_exp_scatter()`**: Scatter plot với logarithmic scale cho TDA processing time vs số nodes.

6. **`visualize_labels(file)`**: Visualize distribution của labels theo thời gian với line plot và color-coded points.

7. **Data distribution visualizations**: Nhiều functions để visualize data distribution theo các features khác nhau:
   - `data_by_edge_visualization(data)`: Distribution theo số edges
   - `data_by_node_visualization(data)`: Distribution theo số nodes
   - `data_by_density_visualization(data)`: Distribution theo density
   - `data_by_peak_visualization(data)`: Distribution theo peak transactions
   - `data_by_data_duration_visualization(data)`: Distribution theo data duration
   - `data_by_Avg_shortest_path_length_visualization(data)`: Distribution theo average shortest path length
   - `data_by_max_degree_centrality_visualization(data)`: Distribution theo degree centrality
   - `data_by_max_closeness_centrality_visualization(data)`: Distribution theo closeness centrality
   - `data_by_max_betweenness_centrality_visualization(data)`: Distribution theo betweenness centrality
   - `data_by_clique_visualization(data)`: Distribution theo clique number
   - `data_by_avg_daily_trans_visualization(data)`: Distribution theo average daily transactions

Tất cả các visualization functions đều:
- Tạo bar charts với grouped data (theo label)
- Save plots với high resolution (300 DPI)
- Sử dụng pandas và matplotlib

## Cách sử dụng

### Import Modules

```python
# Import specific functions
from util.file_util import write_list_to_csv, merge_dicts
from util.graph_util import from_networkx, get_daily_avg
from util.temporal_gnn_data_util import creatBaselineDatasets
from util.visualization import visualize_labels, data_by_node_visualization

# Hoặc import toàn bộ module
import util.file_util as file_util
import util.graph_util as graph_util
```

### Ví dụ Sử Dụng

#### 1. File Utilities

```python
from util.file_util import write_list_to_csv, merge_dicts

# Ghi list vào CSV
data_list = ['item1', 'item2', 'item3']
write_list_to_csv('output.csv', data_list)

# Merge dictionaries
dicts = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
merged, outliers = merge_dicts(dicts)
```

#### 2. Graph Utilities

```python
from util.graph_util import from_networkx
import networkx as nx

# Tạo NetworkX graph
G = nx.Graph()
G.add_edge(0, 1, weight=1.0)
G.nodes[0]['feature'] = 0.5

# Chuyển sang PyTorch Geometric
data = from_networkx(G, label=1)
print(data.edge_index)  # Edge indices
print(data.x)  # Node features
```

#### 3. Temporal GNN Data Utilities

```python
from util.temporal_gnn_data_util import creatBaselineDatasets

# Tạo baseline datasets
parser = NetworkParser()
creatBaselineDatasets(parser, "networkadex.txt")
# Tạo file CSV trong ../data/all_network/TimeSeries/Baseline/
```

#### 4. Visualization

```python
from util.visualization import visualize_labels, data_by_node_visualization
import pandas as pd

# Visualize labels
visualize_labels("networkadex_labels.txt")

# Visualize data distribution
data = pd.read_csv('final_data.csv')
data_by_node_visualization(data)
data_by_edge_visualization(data)
```

## Dependencies

Các modules trong `util/` yêu cầu:

- **file_util.py**:
  - `csv`: CSV file operations
  - `os`, `pickle`: File operations
  - `numpy`: Numerical operations

- **graph_util.py**:
  - `networkx`: Graph operations
  - `pandas`: Data processing
  - `numpy`: Numerical operations
  - `torch`: PyTorch tensors
  - `torch_geometric`: PyTorch Geometric Data objects
  - `config`: Project configuration

- **temporal_gnn_data_util.py**:
  - `pandas`: Data processing
  - `datetime`: Date operations
  - `os`, `shutil`: File operations

- **visualization.py**:
  - `matplotlib`: Plotting
  - `seaborn`: Statistical visualizations
  - `pandas`: Data processing
  - `numpy`: Numerical operations
  - `datetime`: Date operations

## Lưu ý

### Performance

- **Multiprocessing**: Một số functions sử dụng multiprocessing (như `multiprocess_node_network_count()`). Đảm bảo có đủ CPU cores.
- **Memory**: Functions như `merge_dicts` có thể tốn memory với large datasets. Xem xét batch processing.

### File Paths

- **Relative paths**: Nhiều functions sử dụng relative paths từ project root. Đảm bảo chạy từ đúng directory.
- **Hardcoded paths**: Một số paths được hardcode (như trong `output_cleaner()`). Cần cập nhật nếu cần.

### Data Format

- **Pickle files**: Nhiều functions sử dụng pickle. Đảm bảo Python version tương thích khi đọc/write.
- **CSV format**: Các CSV functions expect specific formats. Kiểm tra format trước khi sử dụng.

### Visualization

- **Display**: Visualization functions sử dụng `plt.show()` - đảm bảo có display hoặc comment out nếu chạy trên server.
- **File saving**: Tất cả plots được save với 300 DPI. Files có thể lớn.
- **Dependencies**: Cần matplotlib backend phù hợp (không phải headless nếu dùng `show()`).

### Error Handling

- **Exception handling**: Một số functions có try-except blocks nhưng không comprehensive. Kiểm tra errors.
- **Validation**: Validate input data trước khi sử dụng các utility functions.

### Compatibility

- **Python version**: Đảm bảo Python 3.6+ (do sử dụng f-strings trong một số places)
- **Library versions**: Kiểm tra compatibility giữa networkx, torch_geometric versions
- **Platform**: Hầu hết functions cross-platform, nhưng một số file operations có thể cần adjust cho Windows
