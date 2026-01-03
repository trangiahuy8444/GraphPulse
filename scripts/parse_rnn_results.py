"""
Parse `RnnResults/RNN-Results.txt` into a clean CSV and print summaries.

The file is written by `models/rnn/rnn_methods.py` and contains rows like:
  network,spec,loss,accuracy,auc,roc_auc,avg_auc,std_auc,time,data_size,lr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="RnnResults/RNN-Results.txt")
    parser.add_argument("--out", default="RnnResults/RNN-Results.csv")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=[
            "Network",
            "Spec",
            "Loss",
            "Accuracy",
            "AUC_metric",
            "ROC_AUC",
            "Avg_AUC",
            "Std_AUC",
            "Time",
            "Data_Size",
            "Learning_Rate",
        ],
    )
    df = df.dropna(subset=["Network", "Spec"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote: {out} (rows={len(df)})")
    print()
    print("Top-10 rows by ROC_AUC:")
    try:
        top = df.sort_values("ROC_AUC", ascending=False).head(10)
        print(top[["Network", "Spec", "ROC_AUC", "Accuracy", "Learning_Rate"]].to_string(index=False))
    except Exception:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

