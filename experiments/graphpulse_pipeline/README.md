### GraphPulse pipeline (run + export tables)

This folder provides an end-to-end pipeline to run the **sequence-based models** used in GraphPulse:

- **Fsnapshot (raw)**: `seq_raw.txt` (3 features per day)
- **FMapper**: `seq_tda.txt` (5 Mapper features per day)
- **GraphPulse (raw+mapper)**: concatenation of both (8 features per day)

It exports **paper-style tables**:
- **Table 1**: ROC-AUC for the graph property prediction task (best is **bold**, second-best is <u>underlined</u>)
- **Table 2**: dataset statistics computed from `data/all_network/TimeSeries/`
- **Table 3**: ablation (Fsnapshot vs FMapper vs both)
- **Table 4**: ablation removing one Mapper feature (implemented for `networkaragon.txt`)
- **Table 10**: ROC-AUC for the *node count prediction* task (run with `--task node_count`)
- **Table 11**: ROC-AUC for the *density prediction* task (run with `--task density`)

#### Prerequisites

Create a python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Run (single dataset)

From repo root:

```bash
python3 -m experiments.graphpulse_pipeline.run_pipeline --dataset networkdgd.txt --task tx_count --seed 42 --epochs 100
```

For Table 10 (node count prediction):

```bash
python3 -m experiments.graphpulse_pipeline.run_pipeline --dataset networkdgd.txt --task node_count --seed 42 --epochs 100
```

For Table 11 (density prediction):

```bash
python3 -m experiments.graphpulse_pipeline.run_pipeline --dataset networkdgd.txt --task density --seed 42 --epochs 100
```

Outputs are written under:
- `experiments/graphpulse_pipeline/out/<dataset>/<task>/...`

#### Notes on alignment (important)

Some datasets have outlier windows removed in `seq_tda.txt` (e.g., Mapper failures).
To align `Fsnapshot` with `FMapper` for the **combined GraphPulse** representation, the pipeline:
- treats `y_tda` as a chronological **subsequence** of `y_raw`
- recovers the kept indices via subsequence matching
- then concatenates features on matching windows only

