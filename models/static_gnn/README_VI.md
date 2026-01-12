# Thư Mục Static GNN

## Giới thiệu

Thư mục `models/static_gnn/` chứa implementation của các mô hình Graph Neural Network (GNN) cho static graph classification trong GraphPulse. Các mô hình này xử lý từng graph snapshot độc lập, không xem xét temporal information, và được sử dụng chủ yếu như baseline methods để so sánh với temporal approaches.

Static GNN models trong GraphPulse:
- Xử lý single graph snapshots từ temporal graphs
- Capture structural information của graphs
- Thực hiện graph-level classification (live/dead networks)
- Sử dụng GIN (Graph Isomorphism Network) - một mô hình mạnh cho graph classification

## Các file chính

### `static_graph_methods.py`

File Python chứa implementation của Static GNN models.

**Class GIN (Graph Isomorphism Network):**

Mô hình chính cho static graph classification, implementation của Graph Isomorphism Network.

**Architecture:**
- Multi-layer GIN với MLP trong mỗi layer
- Batch normalization sau mỗi linear layer
- Global pooling (sum hoặc mean) để aggregate node features thành graph-level representation
- Skip connections (residual-like) giữa các layers
- Final classification layer với dropout

**Methods:**
- `__init__(dim_features, dim_target, config)`: Khởi tạo model với configuration từ YAML file
- `forward(x, edge_index, batch)`: Forward pass, nhận node features, edge indices, và batch information

**Functions:**

1. **`read_data()`**: Đọc và merge các CSV files:
   - `final_data.csv`: Graph statistics
   - `GnnResults/final_data_date.csv`: Data duration
   - `GnnResults/average_transaction.csv`: Average daily transactions
   - Merge và tạo final dataset với đầy đủ features

2. **`read_torch_data()`**: Đọc PyTorch Geometric data objects từ thư mục `PygGraphs/`. Đọc tất cả files `.txt` (pickle format) trong thư mục.

3. **`read_torch_time_series_data(network, variable)`**: Đọc temporal graph data từ thư mục `PygGraphs/TimeSeries/{network}/TDA_Tuned/{variable}/`. Đọc tất cả graph snapshots cho một network với TDA parameters cụ thể.

4. **`GIN_classifier(data, network)`**: Function chính để train và evaluate GIN model:
   - Chronological train-test split (80-20)
   - Tạo GIN model với config từ YAML
   - Training với Adam optimizer
   - Evaluation trên train và test sets
   - Logging results mỗi 10 epochs
   - Lưu results vào `GnnResults/GIN_TimeSeries_Result.txt`

5. **`train(train_loader, model, criterion, optimizer)`**: Training function:
   - Iterate qua batches
   - Forward pass, backward pass, và update parameters
   - Error handling cho từng batch

6. **`test(test_loader, model)`**: Evaluation function:
   - Tính accuracy và ROC-AUC
   - Collect predictions và true labels
   - Return accuracy và AUC score

### `config_GIN.yml`

File cấu hình YAML cho GIN model với các hyperparameters:
- `hidden_units`: List các hidden dimensions cho mỗi layer (ví dụ: `[[64, 64]]`)
- `dropout`: Dropout rate (ví dụ: `[0.2]`)
- `train_eps`: Learnable epsilon parameter cho GIN layers
- `aggregation`: Aggregation method (`sum` hoặc `mean`)

## Cách sử dụng

### Import Module

```python
from models.static_gnn.static_graph_methods import GIN, GIN_classifier, read_torch_time_series_data
import yaml
```

### Load Configuration

```python
with open('models/static_gnn/config_GIN.yml', 'r') as f:
    config = yaml.safe_load(f)
```

### Tạo Model

```python
# Tạo GIN model
model = GIN(
    dim_features=1,      # Node feature dimension
    dim_target=2,        # Number of classes (binary classification)
    config=config
)
```

### Forward Pass

```python
from torch_geometric.loader import DataLoader

# data là PyTorch Geometric Data object hoặc list of Data objects
loader = DataLoader(data, batch_size=32)

for batch in loader:
    output = model(batch.x, batch.edge_index, batch.batch)
    # output shape: (batch_size, num_classes)
```

### Train Full Pipeline

```python
from models.static_gnn.static_graph_methods import GIN_classifier, read_torch_time_series_data

# Đọc temporal graph data từ processed TDA snapshots
network = "networkdgd.txt"
data = read_torch_time_series_data(network, "Overlap_xx_Ncube_x")

# Huấn luyện và đánh giá mô hình
GIN_classifier(data, network)
```

### Chạy Full Pipeline (như trong main)

```bash
# Chạy file trực tiếp để thực hiện full pipeline
cd models/static_gnn
python static_graph_methods.py
```

**Lưu ý Mac M2**: Static GNN models sử dụng PyTorch và PyTorch Geometric. Đảm bảo đã cài đặt PyTorch với MPS support:
```bash
# Cài đặt PyTorch với MPS (nếu chưa có)
pip install "torch>=2.0.0" torchvision torchaudio

# Models sẽ tự động sử dụng MPS nếu available
python static_graph_methods.py
```

Script sẽ tự động thực hiện:
- Xử lý tất cả networks trong `networkList`
- Chạy 5 runs (repetitions) cho mỗi network
- Đọc temporal graph snapshots từ `TDA_Tuned/` folder
- Huấn luyện và đánh giá GIN model trên từng snapshot
- Lưu results vào `GnnResults/GIN_TimeSeries_Result.txt`

## Model Architecture

### GIN Layer Structure

Mỗi GIN layer bao gồm:
1. **MLP (Multi-Layer Perceptron)**:
   - Linear → BatchNorm → ReLU
   - Linear → BatchNorm → ReLU

2. **GINConv**: Graph Isomorphism Network Convolution
   - Aggregates neighbor features
   - Applies MLP transformation
   - Uses learnable epsilon parameter

3. **Global Pooling**: Aggregate node features thành graph representation
   - Sum pooling hoặc Mean pooling

4. **Skip Connection**: Residual-like connection từ các layers

### Input/Output

**Input**:
- `x`: Node features tensor `(num_nodes, dim_features)`
- `edge_index`: Edge connectivity tensor `(2, num_edges)`
- `batch`: Batch assignment tensor `(num_nodes,)`

**Output**:
- Graph-level logits `(batch_size, num_classes)`

## Data Format

### PyTorch Geometric Data Objects

Mỗi graph snapshot là một `torch_geometric.data.Data` object với:
- `x`: Node features `(num_nodes, feature_dim)`
- `edge_index`: Edge indices `(2, num_edges)`
- `y`: Graph label (binary: 0 hoặc 1)

### Data Loading

Data được đọc từ:
```
PygGraphs/TimeSeries/{network}/TDA_Tuned/{variable}/
```

Mỗi file trong thư mục này là một pickled PyTorch Geometric Data object đại diện cho một graph snapshot.

## Training Details

### Train-Test Split

- **Chronological split**: 80% train, 20% test
- Không shuffle để preserve temporal order
- Train data đến trước, test data đến sau

### Training Parameters

- **Optimizer**: Adam với learning rate 0.0001
- **Loss**: CrossEntropyLoss (multi-class classification)
- **Epochs**: 101 epochs (logging mỗi 10 epochs)
- **Batch size**: 1 (mỗi graph là một batch)

### Evaluation Metrics

- **Accuracy**: Classification accuracy
- **ROC-AUC**: Area Under ROC Curve (weighted, one-vs-rest)

## Output

### Results File

Results được lưu vào `GnnResults/GIN_TimeSeries_Result.txt` với format:
```
Network,Duplicate,Epoch,Train Accuracy,Train AUC Score,Test Accuracy,Test AUC Score,Unseen AUC Score,Number of Zero labels,Number of one labels
```

### Model Output

- Model predictions: Logits `(batch_size, 2)` cho binary classification
- Probabilities: Softmax của logits
- Predictions: Argmax của logits

## Lưu ý

### Dependencies

- **PyTorch**: Deep learning framework (2.0+ cho Mac M2 với MPS, hoặc 1.6.0+ cho CUDA)
- **PyTorch Geometric**: Graph neural network library
- **PyYAML**: Đọc và parse config files (YAML format)
- **scikit-learn**: Metrics evaluation (ROC-AUC calculation)
- **pytorch-lightning**: Optional, cho advanced training features (không được sử dụng trong implementation hiện tại)

**Mac M2 Specific:**
- PyTorch 2.0+ với MPS support được khuyến nghị
- PyTorch Geometric extensions có thể cần build từ source
- Models sẽ tự động sử dụng MPS nếu available

### Hardware Requirements

- **GPU**: Không bắt buộc nhưng khuyến nghị cho training nhanh hơn (CUDA hoặc MPS)
- **CPU**: Có thể chạy trên CPU nhưng sẽ chậm hơn đáng kể
- **Memory**: Cần đủ RAM để load graph snapshots vào memory (có thể vài GB cho large networks)
- **Mac M2**: MPS sẽ tự động được sử dụng nếu PyTorch 2.0+ được cài đặt

### Configuration

File `config_GIN.yml` cần được cấu hình đúng:
- `hidden_units`: List of lists, mỗi sublist là hidden dimensions cho một layer
- `dropout`: Dropout rate (0.0 đến 1.0)
- `train_eps`: Boolean, có train epsilon parameter không
- `aggregation`: `sum` hoặc `mean`

### Data Paths

**Quan trọng**: Các functions sử dụng relative paths:
- `PygGraphs/`: PyTorch Geometric graphs
- `GnnResults/`: Results output
- `config_GIN.yml`: Config file

Đảm bảo chạy từ đúng directory hoặc cập nhật paths.

### Limitations

1. **Không có Temporal Information**: Chỉ xử lý single snapshot, không có memory qua time
2. **Baseline Purpose**: Chủ yếu dùng để so sánh, không phải main model
3. **Performance**: Thường thấp hơn temporal methods do thiếu temporal information
4. **Single Snapshot**: Mỗi snapshot được xử lý độc lập

### Performance Notes

- Training time: Tùy thuộc vào số lượng snapshots và graph size
- Memory usage: Cần đủ memory cho tất cả snapshots
- Evaluation: Được thực hiện trên cả train và test sets

### Troubleshooting

**Dimension Mismatch**:
- Kiểm tra `dim_features` có đúng với node features không
- Kiểm tra `dim_target` có đúng với số classes không
- Verify graph data format

**Low Accuracy**:
- Đây là baseline model, kỳ vọng performance thấp hơn temporal methods
- Thử điều chỉnh hyperparameters trong config
- Kiểm tra data quality và labels

**Memory Issues**:
- Giảm số lượng snapshots
- Process từng network một
- Giảm hidden units trong config

**File Not Found**:
- Kiểm tra đường dẫn đến PyG graphs
- Đảm bảo TDA processing đã được thực hiện
- Verify network name và variable parameter
