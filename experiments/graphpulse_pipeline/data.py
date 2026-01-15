from __future__ import annotations

import dataclasses
import datetime as dt
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class SequenceBundle:
    """A single (X, y) bundle for one representation."""

    name: str
    X: np.ndarray  # shape: [N, T, F]
    y: np.ndarray  # shape: [N]


def _read_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def load_raw_sequence(dataset: str, sequences_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Fsnapshot sequences from `seq_raw.txt` pickle.

    Returns:
        (X_raw, y_raw)
        - X_raw shape: [N, 7, 3]
        - y_raw shape: [N]
    """
    p = sequences_root / dataset / "seq_raw.txt"
    d = _read_pickle(p)
    X = np.array(d["sequence"]["raw"], dtype=float)
    y = np.array(d["label"], dtype=int)
    return X, y


def load_tda_sequence(dataset: str, sequences_root: Path, tda_key: str | None = None) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Load FMapper sequences from `seq_tda.txt` pickle.

    Args:
        tda_key: if None, uses the only key present.

    Returns:
        (used_key, X_tda, y_tda)
        - X_tda shape: [N, 7, 5] (in provided datasets)
        - y_tda shape: [N]
    """
    p = sequences_root / dataset / "seq_tda.txt"
    d = _read_pickle(p)
    keys = list(d["sequence"].keys())
    if tda_key is None:
        if len(keys) != 1:
            raise ValueError(f"Expected exactly one tda key in {p}, found: {keys}")
        tda_key = keys[0]
    if tda_key not in d["sequence"]:
        raise KeyError(f"tda_key={tda_key!r} not found in {p}. Available: {keys}")
    X = np.array(d["sequence"][tda_key], dtype=float)
    y = np.array(d["label"], dtype=int)
    return tda_key, X, y


def _subsequence_indices(longer: List[int], shorter: List[int]) -> List[int]:
    """
    Find indices `idx` such that longer[idx[i]] == shorter[i] and idx is increasing.
    Greedy subsequence match. Works when `shorter` is a subsequence of `longer`.
    """
    idx: List[int] = []
    j = 0
    for i, v in enumerate(longer):
        if j >= len(shorter):
            break
        if v == shorter[j]:
            idx.append(i)
            j += 1
    if j != len(shorter):
        raise ValueError(
            "Could not align label subsequence (shorter is not a subsequence of longer). "
            f"matched={j} expected={len(shorter)}"
        )
    return idx


def align_raw_to_tda_by_labels(
    y_raw: np.ndarray,
    y_tda: np.ndarray,
) -> np.ndarray:
    """
    Compute raw indices kept in the TDA sequence file.

    Rationale:
    In this repo, `seq_tda.txt` is typically created by filtering/removing some time windows
    (outliers / failures) while preserving chronological order. Therefore, `y_tda` is a
    subsequence of `y_raw` and we can align by subsequence matching.

    Returns:
        idx_raw_kept: np.ndarray of shape [len(y_tda)]
    """
    idx = _subsequence_indices(y_raw.tolist(), y_tda.tolist())
    return np.array(idx, dtype=int)


def make_combined_sequence(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    X_tda: np.ndarray,
    y_tda: np.ndarray,
) -> SequenceBundle:
    """
    Create GraphPulse combined representation by concatenating [Fsnapshot | FMapper].

    Uses label-based subsequence alignment to map TDA samples back to raw indices.
    """
    idx_raw_kept = align_raw_to_tda_by_labels(y_raw, y_tda)
    X_raw_aligned = X_raw[idx_raw_kept]
    if X_raw_aligned.shape[0] != X_tda.shape[0]:
        raise ValueError(f"Alignment mismatch: X_raw_aligned={X_raw_aligned.shape[0]} X_tda={X_tda.shape[0]}")
    X_combined = np.concatenate([X_raw_aligned, X_tda], axis=-1)
    return SequenceBundle(name="combined", X=X_combined, y=y_tda.copy())


def filter_sequences_with_fixed_window(X: np.ndarray, y: np.ndarray, window: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out samples whose time dimension != `window`.

    Returns:
        (X_filt, y_filt, kept_indices)
    """
    keep = np.array([i for i in range(X.shape[0]) if X[i].shape[0] == window], dtype=int)
    return X[keep], y[keep], keep


def compute_dataset_stats(
    dataset_file: str,
    timeseries_root: Path,
) -> Dict[str, object]:
    """
    Compute Table-2 style dataset stats from `data/all_network/TimeSeries/<dataset_file>`.

    Returns keys:
      - dataset
      - Edaily (avg daily edges)
      - Vdaily (avg daily nodes)
      - G (total number of snapshots)  -> caller should fill from sequences
      - duration_ymd (years, months, days) as tuple[int,int,int]
      - start_date / end_date (ISO strings)
    """
    p = timeseries_root / dataset_file
    df = pd.read_csv(p, sep=" ", names=["src", "dst", "ts", "value"])
    df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date

    start = df["date"].min()
    end = df["date"].max()
    if start is None or end is None:
        raise ValueError(f"Could not infer start/end date from {p}")

    # daily edges: raw count per day (as in paper: "average number of daily edges")
    daily_edges = df.groupby("date").size()

    # daily nodes: unique nodes appearing per day
    daily_nodes = (
        df.groupby("date")
        .apply(lambda g: pd.unique(pd.concat([g["src"], g["dst"]], ignore_index=True)).size)
    )

    # duration as (years, months, days) - approximate months as 30-day chunks
    delta_days = (dt.date.fromisoformat(str(end)) - dt.date.fromisoformat(str(start))).days
    years = delta_days // 365
    rem = delta_days % 365
    months = rem // 30
    days = rem % 30

    return {
        "dataset": dataset_file,
        "Edaily": float(daily_edges.mean()),
        "Vdaily": float(daily_nodes.mean()),
        "duration_ymd": (int(years), int(months), int(days)),
        "start_date": str(start),
        "end_date": str(end),
    }


def compute_window_labels(
    dataset_file: str,
    timeseries_root: Path,
    task: str,
    window_days: int = 7,
    gap_days: int = 3,
    label_window_days: int = 7,
) -> np.ndarray:
    """
    Compute binary labels per time-window for alternative tasks (Table 10/11).

    Tasks:
      - 'tx_count': 1 if future tx count increases vs current window tx count.
      - 'node_count': 1 if future unique node count increases vs current window node count.
      - 'density': 1 if future density increases vs current density.

    Notes:
      - This follows the sliding-window logic in `analyzer/network_parser.py`:
        advance window_start_date by 1 day each step.
    """
    if task not in {"tx_count", "node_count", "density"}:
        raise ValueError(f"Unknown task={task}. Expected one of: tx_count, node_count, density")

    p = timeseries_root / dataset_file
    df = pd.read_csv(p, sep=" ", names=["src", "dst", "ts", "value"])
    df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date
    df = df.sort_values("date")

    start = df["date"].min()
    end = df["date"].max()
    if start is None or end is None:
        raise ValueError(f"Could not infer start/end date from {p}")

    labels: List[int] = []

    window_start = start
    # loop condition mirrors: while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize)
    while (end - window_start).days > (window_days + gap_days + label_window_days):
        window_end = window_start + dt.timedelta(days=window_days)
        label_start = window_end + dt.timedelta(days=gap_days)
        label_end = label_start + dt.timedelta(days=label_window_days)

        cur = df[(df["date"] >= window_start) & (df["date"] < window_end)]
        fut = df[(df["date"] >= label_start) & (df["date"] < label_end)]

        if task == "tx_count":
            cur_v = int(len(cur))
            fut_v = int(len(fut))
        elif task == "node_count":
            cur_nodes = pd.unique(pd.concat([cur["src"], cur["dst"]], ignore_index=True))
            fut_nodes = pd.unique(pd.concat([fut["src"], fut["dst"]], ignore_index=True))
            cur_v = int(cur_nodes.size)
            fut_v = int(fut_nodes.size)
        else:  # density
            cur_nodes = pd.unique(pd.concat([cur["src"], cur["dst"]], ignore_index=True))
            fut_nodes = pd.unique(pd.concat([fut["src"], fut["dst"]], ignore_index=True))

            def density(nodes: np.ndarray, edges_df: pd.DataFrame) -> float:
                n = int(nodes.size)
                if n <= 1:
                    return 0.0
                # Use unique directed edges for density
                e = int(edges_df[["src", "dst"]].drop_duplicates().shape[0])
                return float(e) / float(n * (n - 1))

            cur_v = density(cur_nodes, cur)
            fut_v = density(fut_nodes, fut)

        labels.append(1 if (fut_v - cur_v) > 0 else 0)
        window_start = window_start + dt.timedelta(days=1)

    return np.array(labels, dtype=int)

