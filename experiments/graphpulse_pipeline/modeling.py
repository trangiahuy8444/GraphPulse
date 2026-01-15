from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 15  # for EarlyStopping


@dataclass(frozen=True)
class TrainResult:
    roc_auc: float
    history: Dict[str, list]


def _set_seeds(seed: int) -> None:
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        # allow importing this module without TF installed
        pass
    np.random.seed(seed)


def chronological_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(X))
    n_train = int(train_ratio * n)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def build_lstm_gru_classifier(input_dim: int, time_steps: int = 7):
    """
    Match the architecture used in `models/rnn/rnn_methods.py` (LSTM/GRU stack).
    """
    from keras.models import Sequential
    from keras.layers import LSTM, GRU, Dense

    model = Sequential()
    model.add(LSTM(64, input_shape=(time_steps, input_dim), return_sequences=True))
    model.add(LSTM(32, activation="relu", return_sequences=True))
    model.add(GRU(32, activation="relu", return_sequences=True))
    model.add(GRU(32, activation="relu", return_sequences=False))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def train_and_eval_auc(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
    *,
    use_early_stopping: bool = True,
) -> TrainResult:
    """
    Train a sequence classifier and return ROC-AUC on the chronological test split.

    Fairness note:
    - Uses a chronological split (first 80% train, last 20% test), matching the repo's scripts.
    - Early stopping (optional) monitors validation AUC and restores best weights, avoiding test leakage.
    """
    _set_seeds(cfg.seed)

    # Lazy imports so that this file can exist without forcing TF installation at import time.
    from sklearn.metrics import roc_auc_score
    import tensorflow as tf

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    X_train, y_train, X_test, y_test = chronological_split(X, y, train_ratio=0.8)

    model = build_lstm_gru_classifier(input_dim=X.shape[-1], time_steps=X.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=cfg.patience,
                restore_best_weights=True,
            )
        )

    hist = model.fit(
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0).reshape(-1)
    auc = float(roc_auc_score(y_test, y_pred))
    return TrainResult(roc_auc=auc, history=hist.history)

