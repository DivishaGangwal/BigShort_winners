"""Monte Carlo crash windows built from one fold's training rows only.

Each synthetic window:
1. Picks a random training row s. The burn-in path is the real open-price path
   of the BURN_IN + 1 training rows ending at s (a proxy price path; the
   dataset's close is never read). Volume, volume_z_20 and order_flow of the
   burn-in rows are the real training values.
2. Simulates WINDOW + 1 crash days of daily log returns by block-bootstrapping
   training open-to-open log returns (block length BLOCK), standardising them
   and rescaling so their realised volatility is m x the training median
   volatility_20, m drawn from MULTIPLIERS. One or two single-day drops drawn
   from DROP_RANGE are injected at random positions.
3. Builds open (previous price x small gap), high and low (calibrated on the
   training hl_range), and recomputes sma_10, sma_30, momentum_5, volatility_20
   and rsi_14 with src.indicators (the verified dataset formulas). volume,
   volume_z_20 and order_flow are resampled jointly from high-volatility
   training rows.
4. Features come from the same build_features_r2. Synthetic target =
   1 if the next simulated price > the current simulated price.

The simulated labels are synthetic: scores on them measure robustness to
crash-like inputs, not predictive skill on real data.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src import indicators
from src.features_r2 import LOOKBACK, build_features_r2


@dataclass(frozen=True)
class SimConfig:
    n_windows: int = 200
    window: int = 50
    burn_in: int = 60
    block: int = 5
    multipliers: tuple = (3.0, 5.0)
    n_drops: tuple = (1, 2)
    drop_range: tuple = (-0.15, -0.08)
    gap_frac: float = 0.2


@dataclass
class SimSet:
    W: np.ndarray            # (n, LOOKBACK, F) raw features
    y: np.ndarray            # (n,) synthetic labels
    window_id: np.ndarray
    multiplier: np.ndarray
    sha256: str
    settings: dict
    ret: np.ndarray | None = None   # simulated next-day simple return per row (for backtests)


def _block_bootstrap(pool: np.ndarray, n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, len(pool) - block + 1, size=-(-n // block))
    return np.concatenate([pool[s:s + block] for s in starts])[:n]


def simulate_crashes(train_raw: pd.DataFrame, med_vol: float, vol_high_cut: float, seed, cfg: SimConfig = SimConfig(),
                     extra=None) -> SimSet:
    """train_raw: timestamp + allowed inputs for the fold's training period only.

    extra: optional callable(frame, simulated_price) -> DataFrame of additional
    same-row features indexed by timestamp, appended before windowing.
    """
    rng = np.random.default_rng(seed)
    tr = train_raw.reset_index(drop=True)
    opens = tr["open"].to_numpy(float)
    pool = np.diff(np.log(opens))
    hl_mean = float(((tr["high"] - tr["low"]) / tr["open"]).mean())
    a = max(0.1, (hl_mean / med_vol - 0.8) / 1.6)
    hv = tr.loc[tr["volatility_20"] >= vol_high_cut, ["volume", "volume_z_20", "order_flow"]].to_numpy(float)
    if len(hv) == 0:
        raise ValueError("no high-volatility training rows to resample from")
    B, n_crash = cfg.burn_in, cfg.window + 1

    Ws, ys, wid, mult, rets = [], [], [], [], []
    for k in range(cfg.n_windows):
        s = int(rng.integers(B, len(tr)))
        burn = opens[s - B:s + 1]                       # B + 1 real prices ending at row s
        m = float(rng.choice(cfg.multipliers))
        target_sd = m * med_vol
        r = _block_bootstrap(pool, n_crash, cfg.block, rng)
        r = (r - r.mean()) / r.std(ddof=1) * target_sd + pool.mean()
        for p in rng.choice(cfg.window, size=int(rng.choice(cfg.n_drops)), replace=False):
            r[p] = np.log1p(rng.uniform(*cfg.drop_range))
        price = np.concatenate([burn, burn[-1] * np.exp(np.cumsum(r))])   # length B + 1 + n_crash
        n = len(price)
        sd = np.where(np.arange(n) > B, target_sd, med_vol)
        open_ = np.empty(n)
        open_[0] = price[0]
        open_[1:] = price[:-1] * np.exp(rng.normal(0, cfg.gap_frac * sd[1:]))
        hi = np.maximum(open_, price) * (1 + a * sd * np.abs(rng.normal(size=n)))
        lo = np.minimum(open_, price) * (1 - a * sd * np.abs(rng.normal(size=n)))
        side = np.vstack([tr.loc[s - B:s, ["volume", "volume_z_20", "order_flow"]].to_numpy(float),
                          hv[rng.integers(0, len(hv), size=n_crash)]])
        p = pd.Series(price)
        ind = indicators.compute_all(p)
        frame = pd.DataFrame({
            "timestamp": np.arange(1, n + 1), "open": open_.round(4), "high": hi.round(4), "low": lo.round(4),
            "volume": side[:, 0].round(), "sma_10": ind["sma_10"], "sma_30": ind["sma_30"],
            "momentum_5": ind["momentum_5"], "volatility_20": ind["volatility_20"], "rsi_14": ind["rsi_14"],
            "volume_z_20": side[:, 1], "order_flow": side[:, 2],
        })
        # Rows before the indicators are defined are burn-in context only.
        frame = frame.iloc[30:].reset_index(drop=True)
        frame["timestamp"] = np.arange(1, len(frame) + 1)
        feats, _ = build_features_r2(frame)
        if extra is not None:
            feats = feats.join(extra(frame, price[30:]).loc[feats.index])
        arr = feats.to_numpy()
        # crash rows in the trimmed frame: original positions B+1 .. B+window -> labels need the next price
        crash_orig = np.arange(B + 1, B + 1 + cfg.window)
        crash_ts = crash_orig - 30 + 1
        pos = pd.Series(np.arange(len(feats)), index=feats.index).loc[crash_ts].to_numpy()
        idx = pos[:, None] + np.arange(-LOOKBACK + 1, 1)[None, :]
        if (idx < 0).any():
            raise ValueError("burn-in too short for the lookback window")
        Ws.append(arr[idx])
        ys.append((price[crash_orig + 1] > price[crash_orig]).astype(int))
        rets.append(price[crash_orig + 1] / price[crash_orig] - 1)
        wid.append(np.full(cfg.window, k))
        mult.append(np.full(cfg.window, m))

    W, y = np.concatenate(Ws), np.concatenate(ys)
    if not np.isfinite(W).all():
        raise ValueError("non-finite simulated features")
    h = hashlib.sha256(np.ascontiguousarray(W.round(10)).tobytes() + y.astype(np.int8).tobytes()).hexdigest()
    settings = {**asdict(cfg), "hl_scale_a": a, "median_vol": med_vol, "vol_high_cut": vol_high_cut,
                "n_high_vol_rows": int(len(hv)), "return_pool": "training open-to-open log returns",
                "seed": [int(x) for x in np.atleast_1d(seed)]}
    return SimSet(W=W, y=y, window_id=np.concatenate(wid), multiplier=np.concatenate(mult), sha256=h, settings=settings,
                  ret=np.concatenate(rets))
