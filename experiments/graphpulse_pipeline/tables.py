from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def _rank_1st_2nd(values: Dict[str, float]) -> Tuple[str, str]:
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    best = items[0][0] if items else ""
    second = items[1][0] if len(items) > 1 else ""
    return best, second


def format_value(name: str, value: float, best: str, second: str, digits: int = 4) -> str:
    s = f"{value:.{digits}f}"
    if name == best:
        return f"**{s}**"
    if name == second:
        # underline is not standard markdown, but matches many paper repos
        return f"<u>{s}</u>"
    return s


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    sep = "|"
    lines = []
    lines.append(sep + sep.join(headers) + sep)
    lines.append(sep + sep.join(["---"] * len(headers)) + sep)
    for r in rows:
        lines.append(sep + sep.join(r) + sep)
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class TableArtifacts:
    markdown: str
    rows: List[Dict[str, str]]


def make_table_1(results: Dict[str, float], *, dataset: str) -> TableArtifacts:
    """
    Table 1: ROC-AUC results for graph property prediction.
    """
    best, second = _rank_1st_2nd(results)
    headers = ["Model", "ROC-AUC"]
    rows = []
    row_dicts: List[Dict[str, str]] = []
    for name, auc in sorted(results.items()):
        rows.append([name, format_value(name, auc, best, second)])
        row_dicts.append({"dataset": dataset, "model": name, "roc_auc": f"{auc:.6f}"})
    return TableArtifacts(markdown=markdown_table(headers, rows), rows=row_dicts)


def make_table_2(stats_rows: List[Dict[str, str]]) -> TableArtifacts:
    """
    Table 2: Dataset statistics.

    Expected each row dict to already contain formatted string fields:
      dataset, Edaily, Vdaily, G, duration
    """
    headers = ["Dataset", "|E_daily|", "|V_daily|", "|G|", "Duration*"]
    rows = []
    for r in stats_rows:
        rows.append([r["dataset"], r["Edaily"], r["Vdaily"], r["G"], r["duration"]])
    return TableArtifacts(markdown=markdown_table(headers, rows), rows=stats_rows)


def make_table_3(results: Dict[str, float], *, dataset: str) -> TableArtifacts:
    """
    Table 3: Ablation study (Fsnapshot vs FMapper vs both).
    """
    best, second = _rank_1st_2nd(results)
    headers = ["Setting", "ROC-AUC"]
    rows = []
    row_dicts: List[Dict[str, str]] = []
    for name, auc in sorted(results.items()):
        rows.append([name, format_value(name, auc, best, second)])
        row_dicts.append({"dataset": dataset, "setting": name, "roc_auc": f"{auc:.6f}"})
    return TableArtifacts(markdown=markdown_table(headers, rows), rows=row_dicts)


def make_table_4(feature_results: Dict[str, float], *, dataset: str) -> TableArtifacts:
    """
    Table 4: Ablation study removing one Mapper feature (Aragon).
    """
    best, second = _rank_1st_2nd(feature_results)
    headers = ["Removed Mapper feature", "ROC-AUC"]
    rows = []
    row_dicts: List[Dict[str, str]] = []
    for name, auc in sorted(feature_results.items()):
        rows.append([name, format_value(name, auc, best, second)])
        row_dicts.append({"dataset": dataset, "removed_feature": name, "roc_auc": f"{auc:.6f}"})
    return TableArtifacts(markdown=markdown_table(headers, rows), rows=row_dicts)

