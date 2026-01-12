# Khuyến Nghị Quy Trình: Python Scripts vs. Jupyter Notebooks

## Tóm Tắt Thực Thi

**Khuyến nghị**: **Phương pháp kết hợp (Hybrid Approach)** - Sử dụng các script `.py` cho việc thực thi training, các notebook `.ipynb` cho phân tích và visualization.

- **Training**: Python scripts (`.py`) - Sẵn sàng cho production, ổn định cho các chạy dài
- **Phân tích**: Jupyter notebooks (`.ipynb`) - Khám phá tương tác, visualization, so sánh kết quả

---

## Phân Tích Chi Tiết

### 1. Tính Ổn Định & Quản Lý Bộ Nhớ trên Mac M2

#### Python Scripts (`.py`) ✅ **THẮNG CUỘC**

**Ưu điểm:**
- **Không bị crash kernel**: Scripts chạy trong một Python process riêng biệt, không phụ thuộc vào Jupyter kernel
- **Quản lý bộ nhớ tốt hơn**: Clean process lifecycle - bộ nhớ được giải phóng hoàn toàn sau khi thực thi
- **Không có vấn đề ngắt kết nối**: Có thể chạy ở background (`nohup`, `screen`, `tmux`) hoặc như system services
- **Khôi phục**: Dễ dàng resume với checkpointing (code của bạn đã lưu models)
- **Cô lập tài nguyên**: Mỗi lần chạy script là độc lập, ngăn ngừa memory leaks tích lũy

**Đặc thù Mac M2:**
- Quản lý bộ nhớ MPS ổn định hơn với standalone scripts
- Jupyter kernels đôi khi mất MPS context trên các chạy dài (nhiều giờ)
- Xử lý tốt hơn việc reset MPS device trong các session training dài

**Ví dụ Pattern Ổn Định:**
```python
# main.py - Đã implement tốt
if __name__ == '__main__':
    # Clean process start
    runner = Runner()
    runner.run()
    # Clean process end - memory fully released
```

#### Jupyter Notebooks (`.ipynb`) ⚠️ **RỦI RO**

**Nhược điểm:**
- **Kernel crashes**: Trên Mac M2, Jupyter kernels có thể crash sau 2-4 giờ, đặc biệt với MPS
- **Tích lũy bộ nhớ**: Biến tồn tại trong kernel memory, có thể dẫn đến lỗi OOM (Out of Memory)
- **Vấn đề kết nối**: Browser disconnection có thể interrupt kernel (mặc dù execution vẫn tiếp tục)
- **Mất MPS context**: Đôi khi mất MPS device context sau khi restart kernel

**Khi Nào Notebooks Hoạt Động Tốt:**
- Các chạy training ngắn (< 1 giờ)
- Các session debugging tương tác
- Kiểm tra model (forward pass, gradient checks)

**Vấn Đề Đặc Thù Mac M2:**
- Jupyter + MPS + chạy dài = Xác suất crash cao hơn
- Vấn đề memory tracking (đã là 0 MiB trên MPS) tồi tệ hơn trong notebooks

---

### 2. Tự Động Hóa & Sequential Benchmarks

#### Python Scripts (`.py`) ✅ **THẮNG CUỘC**

**Ưu điểm:**
- **Tự động hóa native**: Có thể được gọi từ bash scripts, cron jobs, CI/CD pipelines
- **Xử lý lỗi**: Dễ dàng implement try/except, retry logic, graceful failures
- **Logging**: Code của bạn đã sử dụng `logger` - hoàn hảo cho các chạy không giám sát
- **Exit codes**: Exit codes đúng (0=thành công, 1=lỗi) cho các automation tools
- **Thực thi song song**: Có thể dễ dàng chạy nhiều models song song với `multiprocessing` hoặc `subprocess`

**Ví dụ Automation Script:**
```python
# benchmark_pipeline.py (như đã đề xuất trong HUONG_DAN_CAI_DAT_VA_CHAY.md)
import subprocess
import sys
from datetime import datetime

datasets = ['dgd', 'aion', 'adex']
models = ['HTGN', 'EvolveGCN']
seeds = [1024, 2048, 4096]

results = []
for dataset in datasets:
    for model in models:
        for seed in seeds:
            cmd = [
                'python', 'main.py',
                '--dataset', dataset,
                '--model', model,
                '--seed', str(seed)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Dễ dàng theo dõi success/failure, parse logs, retry on error
```

**Chạy Overnight:**
```bash
# Chạy và detach, logs được lưu tự động
nohup python benchmark_pipeline.py > benchmark.log 2>&1 &

# Hoặc sử dụng screen/tmux để kiểm soát tốt hơn
screen -S graphpulse
python benchmark_pipeline.py
# Ctrl+A, D để detach
```

#### Jupyter Notebooks (`.ipynb`) ❌ **KHÔNG PHÙ HỢP**

**Nhược điểm:**
- **Thực thi thủ công**: Mỗi cell phải được chạy thủ công hoặc qua `nbconvert` (cồng kềnh)
- **Không có tự động hóa native**: Yêu cầu external tools (`papermill`, `jupyter nbconvert`)
- **Khôi phục lỗi**: Khó khăn hơn để implement robust error handling
- **Overnight runs**: Yêu cầu giữ Jupyter server chạy (rủi ro)

**Workaround (Không Khuyến Nghị):**
```bash
# Convert notebook sang script (mất tính tương tác)
jupyter nbconvert --to script notebook.ipynb
python notebook.py  # Bây giờ nó về cơ bản là một .py script
```

---

### 3. Version Control với Git

#### Python Scripts (`.py`) ✅ **THẮNG CUỘC**

**Ưu điểm:**
- **Clean diffs**: Text-based, dễ dàng review changes
- **Không có merge conflicts**: Standard diff/merge workflow
- **Kích thước file nhỏ**: Thường < 500 dòng, dễ dàng theo dõi
- **Lịch sử rõ ràng**: Git blame hoạt động hoàn hảo
- **Thân thiện với branch**: Dễ dàng duy trì nhiều experiment branches

**Ví dụ Git Workflow:**
```bash
# Clean, readable diffs
git diff main.py
# Hiển thị chính xác những gì đã thay đổi, từng dòng một

# Dễ dàng theo dõi thay đổi experiments
git checkout -b experiment/htgn-hyperparams
# Edit main.py
git commit -m "Tune HTGN hyperparameters for dgd"
```

#### Jupyter Notebooks (`.ipynb`) ⚠️ **CÓ VẤN ĐỀ**

**Nhược điểm:**
- **Định dạng JSON**: Notebooks là các file JSON - diffs rối rắm
- **Merge conflicts**: Rất khó khăn để resolve conflicts trong cấu trúc JSON
- **Diffs lớn**: Bao gồm output cells, images, metadata - làm phình git history
- **Hidden state**: Không thể thấy code nào thực sự đã chạy vs. những gì được hiển thị

**Chiến Lược Giảm Thiểu (Vẫn Không Lý Tưởng):**
```bash
# Strip outputs trước khi commit (mất visualization)
jupyter nbconvert --ClearOutputPreprocessor.enabled=True notebook.ipynb

# Sử dụng nbstripout pre-commit hook
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

**Ví dụ Git Diff Rối Rắm:**
```json
{
   "cells": [
      {
         "cell_type": "code",
         "execution_count": 42,
         "metadata": {"collapsed": false},
         "outputs": [{"data": {"text/plain": "..."}, ...}],  // Rối rắm!
         "source": ["model.train()\n"]
      }
   ]
}
```

---

### 4. Visualization & Phân Tích Tương Tác

#### Jupyter Notebooks (`.ipynb`) ✅ **THẮNG CUỘC**

**Ưu điểm:**
- **Inline plots**: Matplotlib/Plotly plots hiển thị trực tiếp trong notebook
- **Interactive widgets**: Có thể điều chỉnh parameters và re-run cells
- **Rich outputs**: Có thể hiển thị DataFrames, images, HTML, LaTeX một cách đẹp mắt
- **Khám phá lặp lại**: Hoàn hảo cho data exploration, so sánh kết quả

**Ví dụ Visualization Workflow:**
```python
# Trong notebook: results_analysis.ipynb
import pandas as pd
import matplotlib.pyplot as plt

# Load tất cả kết quả
results = []
for log_file in glob('data/output/log/**/*.txt'):
    metrics = parse_log(log_file)
    results.append(metrics)

df = pd.DataFrame(results)

# So sánh tương tác
plt.figure(figsize=(12, 6))
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    plt.plot(model_data['epoch'], model_data['auc'], label=model)
plt.legend()
plt.show()  # Hiển thị inline

# Tạo bảng so sánh (hiển thị như HTML table đẹp)
comparison = df.groupby('model').agg({'auc': ['mean', 'std']})
display(comparison)  # Rich formatting
```

#### Python Scripts (`.py`) ⚠️ **HẠN CHẾ**

**Nhược điểm:**
- **Visualization riêng biệt**: Cần lưu plots vào file, sau đó mở riêng
- **Không có tính tương tác**: Không thể điều chỉnh parameters và re-run nhanh chóng
- **Chi phí lặp lại**: Edit code → Run → Kiểm tra file → Lặp lại

**Workaround:**
```python
# Trong script: visualize_results.py
import matplotlib.pyplot as plt
import pandas as pd

# ... load và process data ...

plt.figure(figsize=(12, 6))
# ... plotting code ...
plt.savefig('results_comparison.png')  # Lưu vào file
plt.close()  # Phải close hoặc sử dụng non-interactive backend
print("Plot saved to results_comparison.png")  # Kiểm tra thủ công
```

---

## Quy Trình Kết Hợp Được Khuyến Nghị

### Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    Giai Đoạn Training (.py)                 │
│                                                              │
│  benchmark_pipeline.py  →  Chạy tất cả models tuần tự      │
│        ↓                                                     │
│  main.py, train_*.py  →  Training scripts                   │
│        ↓                                                     │
│  Logs lưu tại: data/output/log/{dataset}/{model}/          │
│  Models lưu tại: saved_models/{dataset}/                   │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Giai Đoạn Phân Tích (.ipynb)               │
│                                                              │
│  results_analysis.ipynb  →  Load logs, visualize, compare  │
│        ↓                                                     │
│  Tạo tables, plots, so sánh metrics                         │
│        ↓                                                     │
│  Export sang định dạng sẵn sàng cho paper (PDF, LaTeX)      │
└─────────────────────────────────────────────────────────────┘
```

### Hướng Dẫn Triển Khai

#### Bước 1: Training với Python Scripts

**Cấu Trúc File:**
```
models/temporal_gnn/script/
├── main.py                    # Hiện có - link prediction
├── train_tgc_end_to_end.py   # Hiện có - TGC task
├── train_graph_classification.py  # Hiện có - graph classification
├── benchmark_pipeline.py      # MỚI - Automated benchmark script
└── utils/
    └── result_parser.py       # MỚI - Parse log files
```

**benchmark_pipeline.py** (Phiên bản nâng cao):
```python
#!/usr/bin/env python3
"""
Automated benchmark pipeline cho GraphPulse reproduction
"""
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
import json

# Cấu hình
EXPERIMENTS = [
    {'dataset': 'dgd', 'model': 'HTGN', 'seeds': [1024, 2048, 4096]},
    {'dataset': 'dgd', 'model': 'EvolveGCN', 'egcn_type': 'EGCNH', 'seeds': [1024, 2048, 4096]},
    # Thêm experiments khác...
]

def run_experiment(config, seed):
    """Chạy một experiment đơn lẻ và trả về result metadata"""
    script_dir = Path(__file__).parent
    
    # Build command
    cmd = ['python', 'main.py',
           '--dataset', config['dataset'],
           '--model', config['model'],
           '--seed', str(seed)]
    
    if 'egcn_type' in config:
        cmd.extend(['--egcn_type', config['egcn_type']])
    
    print(f"\n{'='*60}")
    print(f"Running: {config['model']} on {config['dataset']} (seed={seed})")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # Chạy với timeout (biện pháp an toàn)
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=7200  # Tối đa 2 giờ mỗi experiment
        )
        
        return {
            'dataset': config['dataset'],
            'model': config['model'],
            'seed': seed,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'timestamp': datetime.now().isoformat()
        }
    except subprocess.TimeoutExpired:
        return {
            'dataset': config['dataset'],
            'model': config['model'],
            'seed': seed,
            'success': False,
            'error': 'Timeout',
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Chạy tất cả experiments"""
    results = []
    
    for config in EXPERIMENTS:
        for seed in config['seeds']:
            result = run_experiment(config, seed)
            results.append(result)
            
            # Lưu intermediate results (checkpoint)
            with open('benchmark_progress.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            if not result['success']:
                print(f"⚠️ Experiment failed: {result.get('error', 'Unknown error')}")
    
    # Lưu kết quả cuối cùng
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark complete. Results saved to {output_file}")
    print(f"   Total experiments: {len(results)}")
    print(f"   Successful: {sum(r['success'] for r in results)}")
    print(f"   Failed: {sum(not r['success'] for r in results)}")
    
    return results

if __name__ == '__main__':
    results = main()
    sys.exit(0 if all(r['success'] for r in results) else 1)
```

**Chạy Overnight:**
```bash
cd models/temporal_gnn/script

# Lựa chọn 1: Chạy trực tiếp với nohup
nohup python benchmark_pipeline.py > benchmark.log 2>&1 &
tail -f benchmark.log  # Theo dõi tiến trình

# Lựa chọn 2: Sử dụng screen (khuyến nghị)
screen -S graphpulse_benchmark
python benchmark_pipeline.py
# Ctrl+A, D để detach
# Reattach: screen -r graphpulse_benchmark

# Lựa chọn 3: Sử dụng tmux
tmux new -s benchmark
python benchmark_pipeline.py
# Ctrl+B, D để detach
# Reattach: tmux attach -t benchmark
```

#### Bước 2: Phân Tích với Jupyter Notebooks

**Cấu Trúc File:**
```
notebooks/
├── 01_load_results.ipynb        # Load và parse log files
├── 02_compare_models.ipynb      # So sánh HTGN vs EvolveGCN
├── 03_reproduce_tables.ipynb    # Tạo Tables 1, 2, 3 từ paper
└── utils/
    └── log_parser.py            # Helper functions để parse logs
```

**01_load_results.ipynb** (Ví dụ):
```python
# Cell 1: Setup
import pandas as pd
import numpy as np
from pathlib import Path
import re

# Cell 2: Parse log files
def parse_training_log(log_file):
    """Parse một training log file và trích xuất metrics"""
    with open(log_file) as f:
        content = f.read()
    
    # Trích xuất final metrics sử dụng regex
    auc_match = re.search(r'Test AUC: ([\d.]+)', content)
    ap_match = re.search(r'AP: ([\d.]+)', content)
    
    return {
        'auc': float(auc_match.group(1)) if auc_match else None,
        'ap': float(ap_match.group(1)) if ap_match else None,
        # ... trích xuất thêm metrics
    }

# Cell 3: Load tất cả kết quả
log_dir = Path('../models/temporal_gnn/data/output/log')
results = []

for log_file in log_dir.rglob('*.txt'):
    metrics = parse_training_log(log_file)
    metrics['file'] = str(log_file)
    results.append(metrics)

df = pd.DataFrame(results)
display(df.head())

# Cell 4: Thống kê tổng hợp
summary = df.groupby(['dataset', 'model']).agg({
    'auc': ['mean', 'std', 'count'],
    'ap': ['mean', 'std']
})
display(summary)
```

**03_reproduce_tables.ipynb**:
```python
# Tạo tables sẵn sàng cho paper
import pandas as pd

# Cell 1: Định dạng kết quả như Table 1 (từ paper)
table1 = df.pivot_table(
    values='auc',
    index='dataset',
    columns='model',
    aggfunc=['mean', 'std']
)

# Định dạng cho LaTeX
latex_table = table1.to_latex(
    float_format="%.4f",
    caption="Link Prediction Results (ROC-AUC)"
)
print(latex_table)

# Cell 2: Tạo comparison plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code ...
plt.savefig('paper_figures/table1_comparison.png', dpi=300)
plt.show()
```

---

## Khuyến Nghị Cuối Cùng

### Sử Dụng Python Scripts (`.py`) Cho:

1. ✅ **Thực thi training** (tất cả `main.py`, `train_*.py`)
2. ✅ **Automated benchmarking** (`benchmark_pipeline.py`)
3. ✅ **Tiền xử lý dữ liệu** (đã có trong `analyzer/`)
4. ✅ **Experiments chạy dài** (> 1 giờ)
5. ✅ **Overnight/unattended runs**
6. ✅ **Production code** đi vào git

### Sử Dụng Jupyter Notebooks (`.ipynb`) Cho:

1. ✅ **Phân tích kết quả** (loading logs, tính toán statistics)
2. ✅ **Visualization** (plots, comparison charts)
3. ✅ **Tạo tables cho paper** (Tables 1, 2, 3)
4. ✅ **Interactive debugging** (kiểm tra model, gradient checks)
5. ✅ **Khám phá dữ liệu** (EDA trước khi training)
6. ✅ **One-off experiments** (test hyperparameters nhanh)

### Best Practices

**Cho Training Scripts:**
```python
# Luôn bao gồm:
- Proper logging (✅ bạn đã có điều này)
- Checkpoint saving (✅ bạn đã có điều này)
- Error handling với try/except
- Command-line arguments (✅ bạn đã có điều này)
- Exit codes cho automation
```

**Cho Analysis Notebooks:**
```python
# Luôn bao gồm:
- Clear markdown explanations
- Cell output clearing trước khi commit (sử dụng nbstripout)
- Reproducible imports (pin versions)
- Lưu plots như separate files (không chỉ inline)
```

**Git Workflow:**
```bash
# .gitignore
*.ipynb_checkpoints
.pytest_cache
*.pyc
__pycache__

# Giữ notebooks trong thư mục riêng
notebooks/
  *.ipynb

# Sử dụng nbstripout cho notebooks (tùy chọn nhưng khuyến nghị)
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

---

## Checklist Khởi Động Nhanh

- [ ] Tạo `benchmark_pipeline.py` cho automated training
- [ ] Test một training run: `python main.py --dataset dgd --model HTGN --seed 1024`
- [ ] Thiết lập screen/tmux cho overnight runs
- [ ] Tạo thư mục `notebooks/` cho phân tích
- [ ] Tạo `01_load_results.ipynb` để parse logs
- [ ] Tạo `03_reproduce_tables.ipynb` cho paper tables
- [ ] Cài đặt nbstripout nếu commit notebooks vào git

---

## Tóm Tắt

**Phương pháp kết hợp (Hybrid approach) mang lại lợi ích tốt nhất của cả hai:**

- **Training ổn định, tự động hóa** với Python scripts
- **Phân tích phong phú, tương tác** với Jupyter notebooks
- **Version control sạch sẽ** bằng cách tách biệt concerns
- **Workflow sẵn sàng cho production** phù hợp với paper reproduction

Codebase hiện tại của bạn đã được cấu trúc tốt cho approach này - bạn chỉ cần thêm analysis notebooks và benchmark automation script.
