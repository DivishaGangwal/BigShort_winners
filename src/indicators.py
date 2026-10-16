"""Dataset indicator formulas, verified against the labelled rows of round two.

    sma_n         rolling mean of close over n rows (rounded to 2 dp)
    momentum_5    close[t] / close[t-5] - 1 (4 dp)
    volatility_20 rolling sample std (ddof=1) of close-to-close pct change, 20 rows (4 dp)
    rsi_14        100 - 100 / (1 + mean gain / mean loss), simple 14-row means (4 dp)
    volume_z_20   (volume - rolling mean) / rolling sample std, 20 rows (4 dp)

Each value at row t uses rows <= t only. Used by the crash simulator on
simulated price paths and by the audit/tests on real labelled rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(price: pd.Series, n: int) -> pd.Series:
    return price.rolling(n).mean().round(2)


def momentum(price: pd.Series, n: int = 5) -> pd.Series:
    return (price / price.shift(n) - 1).round(4)


def volatility(price: pd.Series, n: int = 20) -> pd.Series:
    return price.pct_change().rolling(n).std(ddof=1).round(4)


def rsi(price: pd.Series, n: int = 14) -> pd.Series:
    d = price.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d).clip(lower=0).rolling(n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + gain / loss)
    out = out.where(~((gain == 0) & (loss == 0)), 50.0)
    return out.round(4)


def volume_z(volume: pd.Series, n: int = 20) -> pd.Series:
    return ((volume - volume.rolling(n).mean()) / volume.rolling(n).std(ddof=1)).round(4)


def compute_all(price: pd.Series, volume: pd.Series | None = None) -> pd.DataFrame:
    out = pd.DataFrame({
        "sma_10": sma(price, 10), "sma_30": sma(price, 30), "momentum_5": momentum(price, 5),
        "volatility_20": volatility(price, 20), "rsi_14": rsi(price, 14),
    })
    if volume is not None:
        out["volume_z_20"] = volume_z(volume, 20)
    return out
