"""
Run 1 GraphPulse-style RNN experiment on 1 network.

This is a beginner-friendly wrapper around `models/rnn/rnn_methods.py`.

Examples:
  # Train GraphPulse (TDA + Raw, normalized) on one network
  python3 scripts/run_graphpulse_rnn_one.py --network networkdgd.txt --model GraphPulse

  # Train TDA5 only
  python3 scripts/run_graphpulse_rnn_one.py --network networkdgd.txt --model TDA5

  # Train Raw only
  python3 scripts/run_graphpulse_rnn_one.py --network networkdgd.txt --model Raw
"""

from __future__ import annotations

import argparse
from typing import Literal

import numpy as np

from models.rnn.rnn_methods import LSTM_classifier, read_seq_data_by_file_name

ModelName = Literal["TDA5", "Raw", "GraphPulse"]


def _minmax_normalize_all(x: np.ndarray) -> np.ndarray:
    """Normalize entire tensor to [0, 1] using global min/max."""
    min_v = np.min(x)
    max_v = np.max(x)
    if max_v == min_v:
        return np.zeros_like(x)
    x = (x - min_v) / (max_v - min_v)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _pick_first_key(d: dict) -> str:
    if not d:
        raise ValueError("sequence dict is empty")
    return next(iter(d.keys()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, help="e.g. networkdgd.txt, networkadex.txt, Reddit_B.tsv")
    parser.add_argument("--model", required=True, choices=["TDA5", "Raw", "GraphPulse"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed passed to the RNN trainer.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default matches original code).")
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="If set, do not normalize features (for fair comparison keep this consistent).",
    )
    args = parser.parse_args()

    network: str = args.network
    model: ModelName = args.model

    if model == "TDA5":
        data = read_seq_data_by_file_name(network, "seq_tda.txt")
        key = _pick_first_key(data["sequence"])
        x = np.array(data["sequence"][key])
        y = np.array(data["label"])
        if not args.no_normalize:
            x = _minmax_normalize_all(x)
        print(f"Loaded TDA5: key={key} X={x.shape} y={y.shape}")
        auc = LSTM_classifier(x, y, spec="TDA5", network=network, seed=args.seed, epochs=args.epochs)
        print(f"Done. AUC={auc:.4f}")
        return

    if model == "Raw":
        data = read_seq_data_by_file_name(network, "seq_raw.txt")
        key = _pick_first_key(data["sequence"])  # usually "raw"
        x = np.array(data["sequence"][key])
        y = np.array(data["label"])
        if not args.no_normalize:
            x = _minmax_normalize_all(x)
        print(f"Loaded Raw: key={key} X={x.shape} y={y.shape}")
        auc = LSTM_classifier(x, y, spec="Raw", network=network, seed=args.seed, epochs=args.epochs)
        print(f"Done. AUC={auc:.4f}")
        return

    # GraphPulse = concatenate TDA5 (5) + Raw (3) => 8 features
    tda = read_seq_data_by_file_name(network, "seq_tda.txt")
    raw = read_seq_data_by_file_name(network, "seq_raw.txt")
    tda_key = _pick_first_key(tda["sequence"])
    raw_key = _pick_first_key(raw["sequence"])

    x_tda = np.array(tda["sequence"][tda_key])
    x_raw = np.array(raw["sequence"][raw_key])
    y = np.array(tda["label"])

    if x_tda.shape[0] != x_raw.shape[0]:
        raise ValueError(f"Sample count mismatch: TDA={x_tda.shape[0]} Raw={x_raw.shape[0]}")
    if x_tda.shape[1] != x_raw.shape[1]:
        raise ValueError(f"Time length mismatch: TDA={x_tda.shape[1]} Raw={x_raw.shape[1]}")

    if not args.no_normalize:
        x_tda = _minmax_normalize_all(x_tda)
        x_raw = _minmax_normalize_all(x_raw)

    x = np.concatenate((x_tda, x_raw), axis=2)
    print(f"Loaded GraphPulse: tda_key={tda_key} raw_key={raw_key} X={x.shape} y={y.shape}")
    auc = LSTM_classifier(x, y, spec="GraphPulse", network=network, seed=args.seed, epochs=args.epochs)
    print(f"Done. AUC={auc:.4f}")


if __name__ == "__main__":
    main()

