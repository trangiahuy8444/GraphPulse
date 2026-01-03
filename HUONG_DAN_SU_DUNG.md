# Hướng dẫn sử dụng GraphPulse (dành cho người mới)

Tài liệu này hướng dẫn bạn **cách chạy các mô hình trong repo** và **so sánh kết quả** giữa các nhóm model.

Repo này thực tế gồm 3 “nhánh” khác nhau:

- **GraphPulse-style RNN**: `models/rnn/` (dùng sequence TDA + Raw → RNN/LSTM/GRU).
- **Static GNN baseline**: `models/static_gnn/` (ví dụ GIN chạy trên graph đã được chuyển sang PyTorch Geometric).
- **Temporal/Dynamic GNN (SOTA/baselines)**: `models/temporal_gnn/` (HTGN/GRUGCN/EvolveGCN/GAE/VGAE…).

Ngoài file này, bạn có thể đọc thêm `PIPELINE.md` (bản tóm tắt pipeline & benchmark).

---

## Mục lục

1. [Bạn muốn chạy nhanh cái gì?](#bạn-muốn-chạy-nhanh-cái-gì)
2. [Chuẩn bị môi trường Python](#chuẩn-bị-môi-trường-python)
3. [Chạy Temporal GNN (dễ nhất vì có data mẫu)](#chạy-temporal-gnn-dễ-nhất-vì-có-data-mẫu)
4. [Chạy GraphPulse RNN (TDA/Raw/GraphPulse) trên sequences có sẵn](#chạy-graphpulse-rnn-tdarawgraphpulse-trên-sequences-có-sẵn)
5. [Static GNN (GIN/TDA-GIN) – phần nâng cao](#static-gnn-gintda-gin--phần-nâng-cao)
6. [So sánh kết quả giữa các model (benchmark)](#so-sánh-kết-quả-giữa-các-model-benchmark)
7. [Troubleshooting](#troubleshooting)

---

## Tổng quan cực ngắn (để bạn không bị rối)

### Repo này đang làm gì?

Mục tiêu chung là học/dự đoán trên **temporal graph** (đồ thị thay đổi theo thời gian). Nhưng repo có nhiều cách biểu diễn:

- **Cách 1 (GraphPulse-style)**: biến đồ thị theo thời gian thành **chuỗi đặc trưng** (TDA + raw) theo từng ngày/tuần → đưa vào **RNN** để dự đoán label.
- **Cách 2 (Temporal GNN)**: đưa trực tiếp chuỗi snapshot graph vào **temporal/dynamic GNN** (HTGN/GRUGCN/EvolveGCN…) để làm link prediction hoặc graph classification.

### Thuật ngữ bạn sẽ gặp

- **Network / dataset**: một tập dữ liệu (ví dụ `networkdgd.txt` hoặc `aion`).
- **Snapshot**: đồ thị tại một thời điểm (ví dụ 1 ngày/1 tuần).
- **Sequence**: chuỗi vector đặc trưng theo thời gian, thường có dạng `(n_samples, 7, n_features)`.
- **TDA**: topological data analysis. Trong repo này, TDA features đã được trích xuất sẵn thành `seq_tda.txt`.

### Cấu trúc thư mục bạn nên biết

Ngay trong repo (thư mục `/workspace`) có các phần chính:

- `data/`
  - `data/all_network/TimeSeries/`: dữ liệu time series dạng text (nguồn gốc)
  - `data/Sequences/`: sequences đã trích xuất sẵn cho RNN (dễ chạy nhất)
- `models/`
  - `models/rnn/`: GraphPulse-style RNN (TensorFlow/Keras)
  - `models/static_gnn/`: static GNN baseline (PyTorch Geometric)
  - `models/temporal_gnn/`: temporal/dynamic GNN library (PyTorch Geometric)
- `scripts/`: script helper cho người mới (mình đã thêm)
- `RnnResults/`: nơi lưu kết quả RNN (tự tạo khi chạy)

---

## Bạn muốn chạy nhanh cái gì?

### Nếu bạn “chưa biết gì” và muốn thấy chạy ra kết quả ngay

1) **Chạy Temporal GNN với data mẫu `aion`** (nhánh `models/temporal_gnn/`)  
→ Đây là đường chạy “ít phụ thuộc nhất”, có data mẫu đi kèm.

2) **Chạy GraphPulse-style RNN trên 1 network** (nhánh `models/rnn/`)  
→ Repo đã có sẵn sequence ở `data/Sequences/...`, bạn không cần trích xuất lại.

---

## Chuẩn bị môi trường Python

Trong workspace này, lệnh đúng là **`python3`** (không phải `python`).

### 0) Kiểm tra nhanh máy có Python3 chưa

```bash
python3 --version
```

Nếu ra dạng `Python 3.x.y` là ổn.

### Khuyến nghị: tạo 2 môi trường riêng

Vì:
- `models/rnn` dùng **TensorFlow/Keras** (nặng).
- `models/temporal_gnn` dùng **PyTorch/PyG** (cũng nặng).
- Cài chung vẫn được, nhưng hay gặp xung đột phiên bản hoặc cài rất lâu.

### Môi trường 1: cho GraphPulse RNN

```bash
cd /workspace
python3 -m venv .venv_rnn
source .venv_rnn/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Ghi chú:
- `requirements.txt` của repo này pin khá nhiều gói (TensorFlow + PyTorch + PyG). Cài sẽ lâu lần đầu.

### Môi trường 2: cho Temporal GNN

```bash
cd /workspace
python3 -m venv .venv_tgnn
source .venv_tgnn/bin/activate
pip install -U pip

# Cài tối thiểu (thường đủ để chạy demo aion)
pip install torch numpy pandas scikit-learn matplotlib geoopt
pip install torch-geometric
```

Ghi chú:
- File `models/temporal_gnn/requirements.txt` trong repo đang pin version rất cũ (torch 1.6 + CUDA). Trên máy CPU/Linux mới có thể không phù hợp. Với người mới, bạn nên dùng cách cài “tối thiểu” ở trên trước.

---

## Chạy Temporal GNN (dễ nhất vì có data mẫu)

Nhánh `models/temporal_gnn` đã có data mẫu ở `models/temporal_gnn/data_sample/`.
Code trong `models/temporal_gnn/script/` lại đọc data theo đường dẫn `../data/...`, vì vậy bạn cần tạo symlink:

```bash
cd /workspace
ln -s models/temporal_gnn/data_sample models/temporal_gnn/data
```

Nếu bạn chạy lại mà báo “File exists” thì là bạn đã tạo rồi (không sao).

### 1) Chạy Temporal Graph Classification (khuyến nghị để benchmark)

```bash
python3 models/temporal_gnn/script/train_tgc_end_to_end.py \
  --dataset=aion \
  --model=HTGN \
  --seed=710 \
  --max_epoch=50
```

Bạn sẽ thấy log in ra loss/AUC/AP theo epoch. Output/log thường nằm dưới:

- `models/temporal_gnn/data/output/log/aion/HTGN/`

### 1.1) Chạy nhiều seed (để lấy mean ± std)

Ví dụ chạy 5 seed:

```bash
for seed in 43 44 45 46 47; do
  python3 models/temporal_gnn/script/train_tgc_end_to_end.py \
    --dataset=aion \
    --model=HTGN \
    --seed=$seed \
    --max_epoch=50
done
```

### 1.2) Chạy model khác (GRUGCN / EvolveGCN)

Ví dụ GRUGCN:

```bash
python3 models/temporal_gnn/script/train_tgc_end_to_end.py \
  --dataset=aion \
  --model=GRUGCN \
  --seed=710 \
  --max_epoch=50
```

EvolveGCN (baseline script khác):

```bash
python3 models/temporal_gnn/script/baselines/run_evolvegcn_baselines_TGC.py \
  --dataset=aion \
  --model=EGCN \
  --seed=710
```

Ghi chú:
- Các script baseline trong `baselines/` có thể có tham số khác nhau, nên nếu lỗi, hãy xem phần `--help` bằng cách thêm `-h`.

### 2) Chạy link prediction (nếu bạn cần)

```bash
python3 models/temporal_gnn/script/main.py \
  --dataset=aion \
  --model=HTGN \
  --seed=710
```

### 3) Nếu bạn muốn dùng dataset của bạn (nâng cao)

`models/temporal_gnn` (task TGC) mong dữ liệu nằm ở:

```
models/temporal_gnn/data/input/raw/<dataset>/
  <dataset>_edgelist.txt
  <dataset>_labels.csv
```

Trong đó:
- `*_edgelist.txt` cần có tối thiểu các cột `source`, `destination`, `snapshot` (có thể có thêm `weight`)
- `*_labels.csv` là 1 cột label theo thứ tự snapshot

Với người mới, hãy chạy demo `aion` trước để hiểu pipeline, rồi mới convert dữ liệu của bạn sang format này.

### 4) Bảng “mapping tên dataset” (khi bạn đọc paper/dữ liệu)

Trong repo có 2 kiểu tên:

- Kiểu “file network”: `networkdgd.txt`, `networkadex.txt`, `mathoverflow.txt`, `Reddit_B.tsv` (thường dùng ở `data/Sequences`)
- Kiểu “dataset short name”: `dgd`, `adex`, `mathoverflow`, `CollegeMsg`, `aion` (thường dùng ở `models/temporal_gnn`)

Ví dụ mapping hay gặp:

| File name trong `data/` | Dataset name trong `temporal_gnn` |
|---|---|
| `networkdgd.txt` | `dgd` |
| `networkadex.txt` | `adex` |
| `networkaion.txt` | `aion` |
| `networkaragon.txt` | `aragon` |
| `networkbancor.txt` | `bancor` |
| `networkcentra.txt` | `centra` |
| `networkcoindash.txt` | `coindash` |
| `networkiconomi.txt` | `iconomi` |
| `mathoverflow.txt` | `mathoverflow` |
| `CollegeMsg.txt` | `CollegeMsg` |

Ghi chú:
- Riêng Reddit, trong code `models/temporal_gnn/script/utils/data_util.py` có nhắc `RedditB` (khác với `Reddit_B.tsv`). Nếu bạn muốn chạy Reddit trong `temporal_gnn`, bạn cần đảm bảo folder + filename theo đúng quy ước của `temporal_gnn`.

---

## Chạy GraphPulse RNN (TDA/Raw/GraphPulse) trên sequences có sẵn

### 1) Bạn cần biết “sequence” nằm ở đâu

Repo đã có sẵn:

```
data/Sequences/<network>/
  seq_tda.txt   # TDA features
  seq_raw.txt   # Raw features
```

Ví dụ:
- `data/Sequences/networkdgd.txt/seq_tda.txt`
- `data/Sequences/networkdgd.txt/seq_raw.txt`

Mỗi file là **pickle** (đọc bằng Python) có dạng:
- `{"sequence": {...}, "label": [...] }`

### 2) Các “model type” trong nhánh RNN

| Tên bạn chạy | Nguồn dữ liệu | Số feature | Ý nghĩa |
|---|---|---:|---|
| `TDA5` | `seq_tda.txt` | 5 | Chỉ TDA features |
| `Raw` | `seq_raw.txt` | 3 | Chỉ raw graph features |
| `GraphPulse` | `seq_tda.txt` + `seq_raw.txt` | 8 | Nối (concatenate) TDA5 + Raw |

### 3) Cách chạy dễ nhất (1 network)

Repo đã có script helper cho người mới:

```bash
python3 scripts/run_graphpulse_rnn_one.py --network networkdgd.txt --model GraphPulse
```

Các ví dụ khác:

```bash
python3 scripts/run_graphpulse_rnn_one.py --network networkdgd.txt --model TDA5
python3 scripts/run_graphpulse_rnn_one.py --network networkdgd.txt --model Raw
```

Nếu bạn muốn chạy “không normalize” (để so sánh ablation):

```bash
python3 scripts/run_graphpulse_rnn_one.py --network networkdgd.txt --model GraphPulse --no_normalize
```

Kết quả sẽ được append vào:
- `RnnResults/RNN-Results.txt`

### 3.1) Script này đang làm gì?

Nó sẽ:

- đọc `data/Sequences/<network>/seq_tda.txt` và/hoặc `seq_raw.txt`
- chọn key đầu tiên trong `sequence` (thường là dạng `overlap...` cho TDA hoặc `raw` cho Raw)
- (tuỳ chọn) normalize về [0,1]
- train RNN bằng hàm `LSTM_classifier(...)`
- ghi kết quả ra `RnnResults/RNN-Results.txt`

### 3.2) Chạy nhiều seed (để so sánh ổn định hơn)

Ví dụ chạy 5 seed cho GraphPulse trên `networkdgd.txt`:

```bash
for seed in 43 44 45 46 47; do
  python3 scripts/run_graphpulse_rnn_one.py --network networkdgd.txt --model GraphPulse --seed $seed --epochs 50
done
```

Ghi chú:
- `--epochs 50` giúp chạy nhanh hơn để test. Muốn “paper-like” thì tăng lên (ví dụ 100).

### 4) Chạy “full” như code gốc (rất lâu)

File `models/rnn/rnn_methods.py` có sẵn vòng lặp chạy nhiều network × nhiều cấu hình × nhiều run. Chạy:

```bash
python3 models/rnn/rnn_methods.py
```

Lưu ý:
- Đây có thể chạy từ **vài giờ đến vài ngày** tùy máy.
- Nếu bạn chỉ muốn chạy nhanh, nên dùng `scripts/run_graphpulse_rnn_one.py` trước.

### 4.1) Muốn chạy nhanh hơn nữa?

Trong `models/rnn/rnn_methods.py`, số epoch đang cố định là `epochs=100` trong `model_LSTM.fit(...)`.
Nếu bạn muốn chạy nhanh để “test pipeline”, bạn có thể giảm xuống (ví dụ 5–10 epoch). Đây là thay đổi code (không bắt buộc).

### 5) Đọc & tổng hợp kết quả RNN

Repo có script parse đơn giản:

```bash
python3 scripts/parse_rnn_results.py
```

Nó sẽ tạo:
- `RnnResults/RNN-Results.csv`
và in ra top dòng theo `ROC_AUC`.

---

## Static GNN (GIN/TDA-GIN) – phần nâng cao

Nhánh `models/static_gnn/static_graph_methods.py` **không chạy “ngay lập tức”** nếu bạn chưa có dữ liệu PyG graphs ở thư mục `PygGraphs/...`.

### Vì sao khó hơn?

Static GNN ở repo này đọc các file pickle PyTorch Geometric do pipeline trích xuất tạo ra (TDA graph / temporal vectorized graph). Các file đó **không nằm sẵn trong repo**.

### Bạn cần làm gì nếu vẫn muốn chạy?

1) Tạo PyG graphs bằng `analyzer/network_parser.py`
2) Đảm bảo tạo ra đúng thư mục `PygGraphs/TimeSeries/<network>/...`
3) Chạy `models/static_gnn/static_graph_methods.py`

Ghi chú quan trọng:
- `analyzer/network_parser.py` khi chạy “theo kiểu batch” có thể **move** file dữ liệu sang thư mục `Processed/` → dễ làm bạn “mất file” (thực ra là bị dời đi). Nếu bạn là người mới, hãy **copy dữ liệu ra chỗ khác** trước khi chạy batch.

Nếu mục tiêu của bạn là **so sánh model nhanh**, hãy ưu tiên:
- `models/temporal_gnn` (có pipeline rõ và có data mẫu)
- `models/rnn` (đã có sequence sẵn)

---

## So sánh kết quả giữa các model (benchmark)

### 1) Quy tắc vàng: phải so sánh “cùng task”

Trong repo, `models/rnn` và `models/temporal_gnn` không mặc định làm cùng 1 task:

- `models/rnn`: sequence classification (chuỗi feature → label)
- `models/temporal_gnn`: có link prediction và temporal graph classification (TGC)

Nếu bạn muốn **1 bảng so sánh trực tiếp**, bạn cần quyết định:

- **Option A (dễ nhất)**: benchmark theo **task mà bạn đang chạy**, ví dụ:
  - so sánh các model trong `models/temporal_gnn` với nhau (HTGN vs GRUGCN vs EGCN…)
  - so sánh các spec trong `models/rnn` với nhau (TDA5 vs Raw vs GraphPulse…)

- **Option B (khó hơn nhưng “paper-like”)**: chuẩn hoá để mọi model cùng chung task/dataset/split/metric (cần thêm bước convert/align data).

### 2) Protocol benchmark tối thiểu (khuyến nghị)

- **Dataset**: chọn 1 dataset (ví dụ `aion` cho temporal_gnn demo; `networkdgd.txt` cho rnn)
- **Seeds**: 5 seed (ví dụ 43–47)
- **Split**: theo thời gian (train trước, test sau)
- **Metric**: ROC-AUC, AP (hoặc ROC-AUC cho RNN nếu bạn chỉ log AUC)
- **Báo cáo**: mean ± std

### 3) Lưu ý quan trọng về “Table 10/11 (node count / density)”

Trong nội dung bạn đưa có nhắc các script như `create_node_count_density_sequences.py`, `train_node_count_density.py`.
Hiện tại trong repo này **không có sẵn** các file đó.

Nếu bạn muốn làm các task kiểu “node count tăng/giảm” hoặc “density tăng/giảm”, hướng đúng là:

- tạo label mới ở bước trích xuất (thay vì label theo số giao dịch tăng/giảm)
- lưu sequences ra file mới (ví dụ `seq_tda_node_count.txt`)
- viết script train đọc đúng file mới đó

Chỗ dễ nhất để bắt đầu xem cách label đang được tạo là trong:
- `analyzer/network_parser.py` (các hàm tạo sequence/graph theo time window)

---

## Troubleshooting

### 1) `python: command not found`
Dùng `python3` thay vì `python`.

### 2) `FileNotFoundError` khi chạy RNN
Kiểm tra bạn có file:
- `data/Sequences/<network>/seq_tda.txt`
- `data/Sequences/<network>/seq_raw.txt`

Ví dụ:
```bash
ls data/Sequences/networkdgd.txt/
```

### 3) Temporal GNN báo không tìm thấy data
Bạn quên tạo symlink:

```bash
ln -s models/temporal_gnn/data_sample models/temporal_gnn/data
```

### 4) Cài `torch-geometric` bị lỗi
`torch-geometric` phụ thuộc phiên bản PyTorch. Nếu lỗi, hãy cài đúng theo hướng dẫn của PyG tương ứng với PyTorch bạn đang dùng.

