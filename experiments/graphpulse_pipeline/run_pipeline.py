from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from experiments.graphpulse_pipeline.data import (
    SequenceBundle,
    compute_dataset_stats,
    compute_window_labels,
    filter_sequences_with_fixed_window,
    load_raw_sequence,
    load_tda_sequence,
    make_combined_sequence,
    align_raw_to_tda_by_labels,
)
from experiments.graphpulse_pipeline.modeling import TrainConfig, train_and_eval_auc
from experiments.graphpulse_pipeline.tables import make_table_1, make_table_2, make_table_3, make_table_4


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_one_task(
    *,
    dataset: str,
    task: str,
    tda_key: str | None,
    sequences_root: Path,
    timeseries_root: Path,
    out_dir: Path,
    cfg: TrainConfig,
) -> Dict[str, float]:
    """
    Runs the sequence-based models for a single dataset+task and returns model->AUC.
    """
    # Load sequences
    X_raw, y_tx_raw = load_raw_sequence(dataset, sequences_root)
    used_tda_key, X_tda, y_tx_tda = load_tda_sequence(dataset, sequences_root, tda_key=tda_key)

    # Filter fixed window (robustness against occasional malformed samples)
    X_raw, y_tx_raw, keep_raw = filter_sequences_with_fixed_window(X_raw, y_tx_raw, window=7)
    X_tda, y_tx_tda, keep_tda = filter_sequences_with_fixed_window(X_tda, y_tx_tda, window=7)

    # For tasks other than default tx_count, compute labels from raw timeseries and map to kept indices.
    if task == "tx_count":
        y_raw = y_tx_raw
        y_tda = y_tx_tda
    else:
        y_all = compute_window_labels(dataset, timeseries_root, task=task)
        # Align: y_all corresponds to the original raw window index; then apply filtering and TDA alignment.
        y_raw = y_all[keep_raw]

        # Use tx-label subsequence alignment to infer which raw indices were retained for tda.
        idx_raw_kept_for_tda = align_raw_to_tda_by_labels(y_raw=y_tx_raw, y_tda=y_tx_tda)
        # idx_raw_kept_for_tda is in the *filtered raw* index space (because y_tx_raw is already filtered),
        # therefore we can index y_raw directly.
        y_tda = y_raw[idx_raw_kept_for_tda]

    # Build bundles
    raw_bundle = SequenceBundle(name="Fsnapshot (raw)", X=X_raw, y=y_raw)
    tda_bundle = SequenceBundle(name=f"FMapper ({used_tda_key})", X=X_tda, y=y_tda)
    combined_bundle = make_combined_sequence(X_raw=X_raw, y_raw=y_tx_raw, X_tda=X_tda, y_tda=y_tx_tda)
    # overwrite labels for combined when task != tx_count
    if task != "tx_count":
        idx_raw_kept_for_tda = align_raw_to_tda_by_labels(y_raw=y_tx_raw, y_tda=y_tx_tda)
        combined_bundle = SequenceBundle(
            name="GraphPulse (raw+mapper)",
            X=combined_bundle.X,
            y=y_raw[idx_raw_kept_for_tda],
        )
    else:
        combined_bundle = SequenceBundle(name="GraphPulse (raw+mapper)", X=combined_bundle.X, y=combined_bundle.y)

    results: Dict[str, float] = {}
    for bundle in [raw_bundle, tda_bundle, combined_bundle]:
        res = train_and_eval_auc(bundle.X, bundle.y, cfg)
        results[bundle.name] = res.roc_auc

    _write_json(
        out_dir / dataset / task / "results.json",
        {
            "dataset": dataset,
            "task": task,
            "tda_key": used_tda_key,
            "train": cfg.__dict__,
            "results": results,
        },
    )
    return results


def run_ablation_remove_mapper_features(
    *,
    dataset: str,
    sequences_root: Path,
    out_dir: Path,
    cfg: TrainConfig,
    tda_key: str | None,
) -> Dict[str, float]:
    """
    Table 4: remove one Mapper feature (dimension) from the combined model.
    """
    X_raw, y_raw = load_raw_sequence(dataset, sequences_root)
    used_tda_key, X_tda, y_tda = load_tda_sequence(dataset, sequences_root, tda_key=tda_key)

    X_raw, y_raw, _ = filter_sequences_with_fixed_window(X_raw, y_raw, window=7)
    X_tda, y_tda, _ = filter_sequences_with_fixed_window(X_tda, y_tda, window=7)
    combined = make_combined_sequence(X_raw=X_raw, y_raw=y_raw, X_tda=X_tda, y_tda=y_tda)

    # combined.X = [raw(3) | mapper(5)]
    raw_dim = X_raw.shape[-1]
    mapper_dim = X_tda.shape[-1]
    if combined.X.shape[-1] != raw_dim + mapper_dim:
        raise ValueError("Unexpected combined dim; expected raw_dim + mapper_dim.")

    feature_results: Dict[str, float] = {}
    for j in range(mapper_dim):
        keep_mapper = [k for k in range(mapper_dim) if k != j]
        X_new = np.concatenate([combined.X[:, :, :raw_dim], combined.X[:, :, raw_dim:][:, :, keep_mapper]], axis=-1)
        name = f"remove_mapper_feat_{j}"
        res = train_and_eval_auc(X_new, combined.y, cfg)
        feature_results[name] = res.roc_auc

    _write_json(
        out_dir / dataset / "ablation_remove_mapper_features" / "results.json",
        {"dataset": dataset, "tda_key": used_tda_key, "train": cfg.__dict__, "results": feature_results},
    )
    return feature_results


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphPulse: run sequence-model pipeline and export tables.")
    parser.add_argument("--dataset", type=str, default="networkdgd.txt", help="Dataset filename under data/Sequences/")
    parser.add_argument(
        "--task",
        type=str,
        default="tx_count",
        choices=["tx_count", "node_count", "density"],
        help="Prediction task (affects labels).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--tda_key", type=str, default=None, help="Optional: specific TDA sequence key to use.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="experiments/graphpulse_pipeline/out",
        help="Output directory (relative to repo root).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sequences_root = repo_root / "data" / "Sequences"
    timeseries_root = repo_root / "data" / "all_network" / "TimeSeries"
    out_dir = (repo_root / args.out_dir).resolve()

    cfg = TrainConfig(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, patience=args.patience)

    # Table 2 stats
    stats = compute_dataset_stats(args.dataset, timeseries_root)

    # total snapshots from raw labels length
    X_raw, y_raw = load_raw_sequence(args.dataset, sequences_root)
    stats_row = {
        "dataset": args.dataset.replace(".txt", ""),
        "Edaily": f"{stats['Edaily']:.2f}",
        "Vdaily": f"{stats['Vdaily']:.2f}",
        "G": str(len(y_raw)),
        "duration": f"{stats['duration_ymd'][0]},{stats['duration_ymd'][1]},{stats['duration_ymd'][2]}",
    }
    table2 = make_table_2([stats_row])
    _write_text(out_dir / args.dataset / "Table_2_dataset_stats.md", table2.markdown)
    _write_json(out_dir / args.dataset / "Table_2_dataset_stats.json", table2.rows)

    # Table 1: property prediction ROC-AUC (for this dataset+task)
    results = run_one_task(
        dataset=args.dataset,
        task=args.task,
        tda_key=args.tda_key,
        sequences_root=sequences_root,
        timeseries_root=timeseries_root,
        out_dir=out_dir,
        cfg=cfg,
    )
    table1 = make_table_1(results, dataset=args.dataset.replace(".txt", ""))
    if args.task == "tx_count":
        table_name = "Table_1_roc_auc_graph_property_prediction"
    elif args.task == "node_count":
        table_name = "Table_10_roc_auc_node_count_prediction"
    else:  # density
        table_name = "Table_11_roc_auc_density_prediction"
    _write_text(out_dir / args.dataset / args.task / f"{table_name}.md", table1.markdown)
    _write_json(out_dir / args.dataset / args.task / f"{table_name}.json", table1.rows)

    # Table 3: ablation (Fsnapshot vs FMapper vs both) - same as Table 1 content, but named as ablation
    table3 = make_table_3(results, dataset=args.dataset.replace(".txt", ""))
    _write_text(out_dir / args.dataset / args.task / "Table_3_ablation_fs_fm.md", table3.markdown)
    _write_json(out_dir / args.dataset / args.task / "Table_3_ablation_fs_fm.json", table3.rows)

    # Table 4: only meaningful for Aragon in the paper, but runnable for any dataset.
    if args.dataset.startswith("networkaragon"):
        ab4 = run_ablation_remove_mapper_features(
            dataset=args.dataset,
            sequences_root=sequences_root,
            out_dir=out_dir,
            cfg=cfg,
            tda_key=args.tda_key,
        )
        table4 = make_table_4(ab4, dataset=args.dataset.replace(".txt", ""))
        _write_text(out_dir / args.dataset / "Table_4_ablation_remove_mapper_feature.md", table4.markdown)
        _write_json(out_dir / args.dataset / "Table_4_ablation_remove_mapper_feature.json", table4.rows)

    # Table 10/11 hooks: just rerun with different tasks (node_count, density)
    # This script already supports `--task node_count` and `--task density` to produce those tables.


if __name__ == "__main__":
    main()

