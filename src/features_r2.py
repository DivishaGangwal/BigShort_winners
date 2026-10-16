"""Round-two causal, scale-free features, sequence windows and fold-local preprocessing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_r2 import ALLOWED_INPUTS_R2

FEATURE_VERSION_R2 = "causal_r2_v1"
LOOKBACK = 20

FEATURE_NAMES_R2 = [
    "hl_range",          # (high - low) / open
    "upper_wick",        # (high - open) / open
    "lower_wick",        # (open - low) / open
    "oo_ret_1", "oo_ret_3", "oo_ret_5",   # open[t] / open[t-k] - 1
    "open_sma10",        # open / sma_10 - 1
    "open_sma30",        # open / sma_30 - 1
    "sma10_sma30",       # sma_10 / sma_30 - 1
    "momentum_5", "volatility_20", "rsi_14", "volume_z_20", "order_flow",
    "order_flow_lag1", "order_flow_lag2", "volume_z_20_lag1", "volume_z_20_lag2",
]
REGIMES = ["high_vol", "low_vol", "high_rsi", "low_rsi"]
SIM_GROUP = "sim_crash"


def build_features_r2(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Features indexed by timestamp, plus the number of leading rows dropped for NaN.

    Reads only the allowed inputs; only current/earlier rows via positive shifts.
    """
    src = frame.loc[:, ALLOWED_INPUTS_R2].astype(float).reset_index(drop=True)
    o, h, l = src["open"], src["high"], src["low"]
    f = pd.DataFrame(index=pd.Index(frame["timestamp"].to_numpy(), name="timestamp"))
    f["hl_range"] = ((h - l) / o).to_numpy()
    f["upper_wick"] = ((h - o) / o).to_numpy()
    f["lower_wick"] = ((o - l) / o).to_numpy()
    for k in (1, 3, 5):
        f[f"oo_ret_{k}"] = (o / o.shift(k) - 1).to_numpy()
    f["open_sma10"] = (o / src["sma_10"] - 1).to_numpy()
    f["open_sma30"] = (o / src["sma_30"] - 1).to_numpy()
    f["sma10_sma30"] = (src["sma_10"] / src["sma_30"] - 1).to_numpy()
    for c in ("momentum_5", "volatility_20", "rsi_14", "volume_z_20", "order_flow"):
        f[c] = src[c].to_numpy()
    for k in (1, 2):
        f[f"order_flow_lag{k}"] = src["order_flow"].shift(k).to_numpy()
    for k in (1, 2):
        f[f"volume_z_20_lag{k}"] = src["volume_z_20"].shift(k).to_numpy()
    f = f[FEATURE_NAMES_R2]
    complete = f.notna().all(axis=1)
    first = int(np.argmax(complete.to_numpy()))
    if not complete.iloc[first:].all():
        raise ValueError("missing feature values after the warmup block")
    return f.iloc[first:], first


def sample_timestamps(feats: pd.DataFrame, lookback: int = LOOKBACK) -> np.ndarray:
    """Rows with a complete lookback window of feature rows (the common sample for LR and LSTM)."""
    return feats.index.to_numpy()[lookback - 1:]


def windows(arr: np.ndarray, positions: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
    """(n, lookback, F) windows ending at each position (rows <= position only)."""
    idx = np.asarray(positions)[:, None] + np.arange(-lookback + 1, 1)[None, :]
    if (idx < 0).any():
        raise ValueError("window would start before the first feature row")
    return arr[idx]


@dataclass
class RegimeCutoffs:
    vol_low: float
    vol_high: float
    rsi_low: float
    rsi_high: float

    def as_dict(self):
        return dict(vol_low=self.vol_low, vol_high=self.vol_high, rsi_low=self.rsi_low, rsi_high=self.rsi_high)


def fit_regime_cutoffs(train: pd.DataFrame, low_q=0.25, high_q=0.75) -> RegimeCutoffs:
    vol, rsi = train["volatility_20"].to_numpy(), train["rsi_14"].to_numpy()
    return RegimeCutoffs(float(np.quantile(vol, low_q)), float(np.quantile(vol, high_q)),
                         float(np.quantile(rsi, low_q)), float(np.quantile(rsi, high_q)))


def assign_regimes(frame: pd.DataFrame, cut: RegimeCutoffs) -> pd.DataFrame:
    vol, rsi = frame["volatility_20"], frame["rsi_14"]
    return pd.DataFrame({"high_vol": vol >= cut.vol_high, "low_vol": vol <= cut.vol_low,
                         "high_rsi": rsi >= cut.rsi_high, "low_rsi": rsi <= cut.rsi_low}, index=frame.index)[REGIMES]


@dataclass
class FoldData:
    ts_train: np.ndarray
    ts_val: np.ndarray
    X_train: np.ndarray        # (n, F) standardized, last row of each window
    X_val: np.ndarray
    W_train: np.ndarray        # (n, LOOKBACK, F) standardized windows
    W_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray | None
    g_train: pd.DataFrame
    g_val: pd.DataFrame
    scaler: StandardScaler
    cutoffs: RegimeCutoffs


def prepare_fold_r2(feats: pd.DataFrame, target: pd.Series, train_ts, val_ts, low_q=0.25, high_q=0.75) -> FoldData:
    """Scaler and regime cutoffs from training rows only; windows may reach back into earlier feature rows."""
    train_ts, val_ts = np.asarray(train_ts, dtype=int), np.asarray(val_ts, dtype=int)
    tr = feats.loc[train_ts]
    scaler = StandardScaler().fit(tr.to_numpy())
    cut = fit_regime_cutoffs(tr, low_q, high_q)
    arr = scaler.transform(feats.to_numpy())
    pos = pd.Series(np.arange(len(feats)), index=feats.index)
    va = feats.loc[val_ts]
    yv = target.loc[val_ts].to_numpy()
    return FoldData(
        ts_train=train_ts, ts_val=val_ts,
        X_train=arr[pos.loc[train_ts].to_numpy()], X_val=arr[pos.loc[val_ts].to_numpy()],
        W_train=windows(arr, pos.loc[train_ts].to_numpy()), W_val=windows(arr, pos.loc[val_ts].to_numpy()),
        y_train=target.loc[train_ts].to_numpy().astype(int),
        y_val=None if np.isnan(yv).any() else yv.astype(int),
        g_train=assign_regimes(tr, cut), g_val=assign_regimes(va, cut), scaler=scaler, cutoffs=cut,
    )
