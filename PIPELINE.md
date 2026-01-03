# Pipeline chạy & so sánh các model trong repo này

File này mô tả **pipeline thực nghiệm** để chạy các nhóm model trong repo và **so sánh kết quả** giữa chúng một cách nhất quán.

Repo hiện có 3 nhánh chính:

- **`models/rnn/` (GraphPulse-style)**: dùng **chuỗi đặc trưng theo thời gian** (TDA + raw) rồi train RNN (LSTM/GRU).
- **`models/static_gnn/` (baseline static)**: baseline GNN (ví dụ GIN) chạy trên **graph snapshot đã được chuyển sang PyG** (pickle).
- **`models/temporal_gnn/` (SOTA temporal GNN)**: thư viện riêng cho các model temporal/dynamic GNN (HTGN, EvolveGCN, GAE/VGAE, …) + pipeline chạy.

> **Quan trọng để so sánh công bằng**: phải thống nhất **(a) task**, **(b) dataset**, **(c) cách chia train/test theo thời gian**, và **(d) metric**. Nếu chạy “mỗi nhánh theo cách riêng”, kết quả thường không so sánh trực tiếp được.

---

## 1) Chọn task & metric so sánh

Trong repo có 2 kiểu task phổ biến:

### A. Temporal graph classification (TGC)
- **Input**: chuỗi snapshot (edge list theo thời gian) + **label cho mỗi snapshot**.
- **Metric khuyến nghị**: ROC-AUC, AP (average precision).
- **Nhánh phù hợp**: `models/temporal_gnn/script/train_tgc_end_to_end.py`

### B. Sequence classification (chuỗi đặc trưng → label)
- **Input**: sequence feature (TDA/raw/combined) + label.
- **Metric khuyến nghị**: ROC-AUC.
- **Nhánh phù hợp**: `models/rnn/rnn_methods.py`

### C. Link prediction (dự đoán cạnh tương lai)
- **Input**: chuỗi snapshot, negative sampling.
- **Metric khuyến nghị**: ROC-AUC, AP (transductive/inductive).
- **Nhánh phù hợp**: `models/temporal_gnn/script/main.py`

**Khuyến nghị để so sánh giữa nhiều model**: ưu tiên **Task A (TGC)** vì `temporal_gnn` hỗ trợ pipeline rõ ràng; còn `rnn`/`static_gnn` cần đảm bảo label/chuỗi tương thích.

---

## 2) Chuẩn hoá dataset (để mọi model đọc cùng một nguồn)

### 2.1. Dữ liệu gốc trong repo
Repo có sẵn:

- **Raw network (toàn bộ)**: `data/all_network/*.txt`
- **TimeSeries**: `data/all_network/TimeSeries/*.txt` và `data/all_network/TimeSeries/Other/*`
- **Pre-extracted sequences (đã có sẵn)**: `data/Sequences/<network>/{seq_raw.txt, seq_tda.txt}`

### 2.2. Quy ước input cho từng nhánh

#### `models/rnn/` (GraphPulse-style)
Kỳ vọng các file pickle:

- `data/Sequences/<network>/seq_tda.txt`
- `data/Sequences/<network>/seq_raw.txt`

Trong đó đối tượng pickle có dạng:
- dict keys: `sequence`, `label`
- `sequence`: dict (mỗi key là spec) → list sample (mỗi sample là chuỗi 7 ngày)
- `label`: list nhãn nhị phân theo sample

#### `models/temporal_gnn/` (TGC)
Kỳ vọng dữ liệu theo cấu trúc (tương đối với `models/temporal_gnn/script/`):

```
models/temporal_gnn/data/input/raw/<dataset>/
  <dataset>_edgelist.txt
  <dataset>_labels.csv
```

Trong `*_edgelist.txt` thường có các cột như:
- `source`, `destination`, `snapshot` (+ có thể có `weight`)

`*_labels.csv`: 1 cột label theo thứ tự snapshot.

#### `models/static_gnn/`
Các hàm trong `models/static_gnn/static_graph_methods.py` đang đọc dữ liệu PyG pickle ở các thư mục kiểu:
- `PygGraphs/TimeSeries/<network>/...`

Những file này **thường phải generate lại** bằng `analyzer/network_parser.py` (hoặc pipeline tương đương).

---

## 3) Pipeline chạy model: lệnh thực thi

### 3.1. Temporal GNN (SOTA) — chạy được ngay bằng data mẫu

Repo có data mẫu ở:
`models/temporal_gnn/data_sample/`

Vì code đọc từ `../data/...` (tính từ `models/temporal_gnn/script/`), bạn tạo symlink:

```bash
ln -s data_sample models/temporal_gnn/data
```

#### A) Temporal Graph Classification (khuyến nghị để benchmark)

```bash
python3 models/temporal_gnn/script/train_tgc_end_to_end.py \
  --dataset=aion \
  --model=HTGN \
  --seed=710 \
  --max_epoch=50
```

Kết quả/log sẽ nằm dưới:
`models/temporal_gnn/data/output/log/<dataset>/<model>/...`

#### B) Link prediction

```bash
python3 models/temporal_gnn/script/main.py \
  --dataset=aion \
  --model=HTGN \
  --seed=710
```

---

### 3.2. RNN (GraphPulse-style) — chạy trên sequences trong `data/Sequences/`

File chạy chính: `models/rnn/rnn_methods.py`

**Lưu ý quan trọng**: hàm `read_seq_data_by_file_name(...)` đang hard-code đường dẫn Windows. Để chạy trên Linux, bạn cần sửa đường dẫn về repo, ví dụ:

- `/workspace/data/Sequences/{network}/`

Sau đó chạy:

```bash
python3 models/rnn/rnn_methods.py
```

Output/metric hiện đang được append vào:
- `RnnResults/RNN-Results.txt` (nếu thư mục tồn tại)

---

### 3.3. Static GNN baseline (GIN)

File chạy chính: `models/static_gnn/static_graph_methods.py`

Pipeline của nhánh này thường là:

1) **Generate PyG graphs** (snapshot / TDA graphs) từ dữ liệu TimeSeries bằng `analyzer/network_parser.py`
2) Chạy `static_graph_methods.py` để train/eval GIN trên các graph đã được serialize.

Chạy:

```bash
python3 models/static_gnn/static_graph_methods.py
```

> Nếu bạn chưa có thư mục `PygGraphs/TimeSeries/...` ở root (cùng cấp với file python khi chạy), thì nhánh này sẽ không chạy được cho tới khi generate dữ liệu.

---

## 4) Pipeline so sánh kết quả (benchmark protocol)

Để so sánh giữa model, bạn nên chốt một “protocol” như sau:

- **Dataset**: chọn 1 tên dataset/network duy nhất (ví dụ `aion`, `adex`, …)
- **Task**: chọn 1 task duy nhất (khuyến nghị: TGC)
- **Split**: chronological split (train = 80% snapshot đầu, test = 20% snapshot cuối)
- **Seed**: chạy nhiều seed (ví dụ 5 seed) và báo cáo mean±std
- **Metric**: ROC-AUC + AP

### 4.1. Tại sao hiện tại có thể “khó so sánh trực tiếp”?

- `models/rnn/` đang học trên **sequence feature** và label theo “window/sample”.
- `models/temporal_gnn/` (TGC) học trên **snapshot graph** và label theo snapshot.
- `models/static_gnn/` học trên **pickle PyG graph** (cần pipeline generate riêng).

Vì vậy, để so sánh “cùng thang đo”, bạn cần:

- **Hoặc** đưa tất cả về **Task A (TGC)** bằng cách chuẩn hoá input cho `rnn`/`static_gnn` theo snapshot label,
- **Hoặc** đưa tất cả về **Task B (sequence classification)** bằng cách trích sequence feature tương thích cho `temporal_gnn` (khó hơn).

Trong thực tế, dễ nhất là chốt TGC và:
- dùng `temporal_gnn` làm SOTA/baseline,
- xây pipeline để tạo sequence từ snapshot rồi cho `rnn` học (nhưng label phải đúng theo snapshot).

---

## 5) Checklist khi báo cáo kết quả

- **Cùng dataset + cùng task** (đừng so ROC-AUC của 2 task khác nhau).
- **Cùng split** (cùng `testlength` hoặc cùng tỉ lệ 80/20 theo thời gian).
- **Cùng cách tạo label**.
- **Cùng seed set** (ví dụ: 43–47).
- **Ghi rõ version** (Python, PyTorch, PyG, TF nếu dùng).

---

## 6) Gợi ý cấu trúc bảng kết quả (so sánh)

Bạn có thể tổng hợp theo format:

| Dataset | Task | Model | Seed | AUC | AP | Notes |
|---|---|---|---:|---:|---:|---|
| aion | TGC | HTGN | 710 | ... | ... | temporal_gnn |
| aion | TGC | EGCN | 710 | ... | ... | temporal_gnn |
| aion | SeqCls | GraphPulse (TDA+Raw→RNN) | 42 | ... | - | rnn |

Nếu mục tiêu là **1 bảng duy nhất**, hãy ép mọi model về **cùng Task** trước khi benchmark.

