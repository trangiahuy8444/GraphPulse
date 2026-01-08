"""
GraphPulse experiment pipeline.

This package provides a reproducible pipeline to:
- Load pre-extracted sequence features (Fsnapshot / FMapper) from `data/Sequences/<dataset>/`.
- Compute dataset statistics from `data/all_network/TimeSeries/<dataset>`.
- Run sequence-based models (LSTM/GRU) for graph property prediction and ablations.
- Export paper-style tables (Markdown/CSV/JSON).
"""

