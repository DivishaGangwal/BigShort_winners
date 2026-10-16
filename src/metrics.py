"""Common metrics on natural, unweighted validation data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

EPS = 1e-15
THRESHOLD = 0.5


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def log_loss(y, p) -> float:
    y, p = np.asarray(y, dtype=float), _clip(p)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def accuracy(y, p) -> float:
    return float(np.mean((np.asarray(p) >= THRESHOLD).astype(int) == np.asarray(y)))


def binary_metrics(y, p) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    two_classes = len(np.unique(y)) == 2
    pred = (p >= THRESHOLD).astype(int)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "accuracy": accuracy(y, p),
        # Balanced accuracy is undefined with one class; reported as unavailable.
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if two_classes else None,
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
    }


def reliability(y, p, bins: int = 5) -> tuple[list[dict], float]:
    """Equal-width probability bins on [0, 1]; returns (table, ECE)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        row = {"bin": f"[{b / bins:.1f}, {(b + 1) / bins:.1f}{']' if b == bins - 1 else ')'}", "n": n,
               "mean_p": float(p[m].mean()) if n else None,
               "frac_pos": float(y[m].mean()) if n else None}
        if n:
            ece += n / len(y) * abs(row["frac_pos"] - row["mean_p"])
        table.append(row)
    return table, float(ece)


def regime_metrics(y, p, groups: pd.DataFrame) -> dict:
    """Per-regime log loss / accuracy / Brier with counts, plus worst-regime values."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    out = {}
    for g in groups.columns:
        m = groups[g].to_numpy()
        n = int(m.sum())
        out[g] = {"n": n,
                  "log_loss": log_loss(y[m], p[m]) if n else None,
                  "accuracy": accuracy(y[m], p[m]) if n else None,
                  "brier": brier(y[m], p[m]) if n else None}
    present = [v for v in out.values() if v["n"]]
    worst = {
        "log_loss": max(v["log_loss"] for v in present) if present else None,
        "accuracy": min(v["accuracy"] for v in present) if present else None,
        "brier": max(v["brier"] for v in present) if present else None,
    }
    worst["log_loss_regime"] = (max((v["log_loss"], g) for g, v in out.items() if v["n"])[1]
                                if present else None)
    return {"regimes": out, "worst": worst}
