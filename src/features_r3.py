"""Round three: causal reconstruction of the current close and close-based features.

Hidden closes (851-900) are rebuilt row by row using rows <= t only:
    close[t] = close[t-5] * (1 + momentum_5[t])        (primary; momentum_5 has 4 dp)
    close[t] = close[t-10] + 10 * (sma_10[t] - sma_10[t-1])   (cross-check)
clipped to [low[t], high[t]]. Nothing from a later row is used, so close[t] is
treated like any other same-row input. The next close is never reconstructed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_r2 import ALLOWED_INPUTS_R2
from src.features_r2 import build_features_r2

CLOSE_FEATURES = ["ret_1d_c", "intraday", "gap", "close_sma10", "clv", "ret1_vol"]


def reconstruct_close(raw: pd.DataFrame) -> dict:
    c = raw["close"].to_numpy(float).copy()
    mom, s10 = raw["momentum_5"].to_numpy(float), raw["sma_10"].to_numpy(float)
    hi, lo = raw["high"].to_numpy(float), raw["low"].to_numpy(float)
    n = len(c)
    est_mom, est_s10 = np.full(n, np.nan), np.full(n, np.nan)
    rebuilt = np.zeros(n, dtype=bool)
    for i in range(n):
        if np.isnan(c[i]):
            if i < 10:
                raise ValueError("cannot rebuild a close without 10 earlier rows")
            est_mom[i] = c[i - 5] * (1 + mom[i])
            est_s10[i] = c[i - 10] + 10 * (s10[i] - s10[i - 1])
            c[i] = min(max(est_mom[i], lo[i]), hi[i])
            rebuilt[i] = True
    return {"close": c, "rebuilt": rebuilt, "est_mom": est_mom, "est_sma10": est_s10}


def close_features(frame: pd.DataFrame, price) -> pd.DataFrame:
    """Same-row close-based features; frame needs timestamp, open, high, low, sma_10, volatility_20."""
    c = pd.Series(np.asarray(price, dtype=float))
    o, h, l = (frame[k].reset_index(drop=True).astype(float) for k in ("open", "high", "low"))
    rng = (h - l).where(h > l)
    out = pd.DataFrame({
        "ret_1d_c": c / c.shift(1) - 1,
        "intraday": c / o - 1,
        "gap": o / c.shift(1) - 1,
        "close_sma10": c / frame["sma_10"].reset_index(drop=True).astype(float) - 1,
        "clv": ((c - l) / rng).fillna(0.5),
        "ret1_vol": (c / c.shift(1) - 1) / frame["volatility_20"].reset_index(drop=True).astype(float),
    })
    out.index = pd.Index(frame["timestamp"].to_numpy(), name="timestamp")
    return out[CLOSE_FEATURES]


def build_features_r3(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """raw: file rows (any cut), close NaN where hidden. Returns (features, reconstruction info)."""
    X = raw.loc[:, ["timestamp", *ALLOWED_INPUTS_R2]].astype({c: float for c in ALLOWED_INPUTS_R2})
    base, _ = build_features_r2(X)
    rec = reconstruct_close(raw)
    ex = close_features(X, rec["close"])
    feats = base.join(ex.loc[base.index])
    if feats.isna().any().any():
        raise ValueError("missing round-three features")
    return feats, rec
