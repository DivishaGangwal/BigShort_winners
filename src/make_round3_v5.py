"""Round three v5: v3 raw signal, scaled to a 15% ex-ante annual risk target, |s| <= 0.05 set to 0.

    python -m src.make_round3_v5        (after `python -m src.run_round3 --lstm`)

Post-mortem lessons applied (see experiments/divisha/round3_postmortem.md): no smoothing;
the organizer treats |signal| <= 0.05 as flat; the leaderboard appears to reward money made.
Each signal_t uses rows <= t only, but these design choices were informed by the scored
period, so v5 is not an out-of-sample candidate. Its realized 851-900 result is not computed.
"""
from __future__ import annotations

import src  # noqa: F401

import hashlib

import numpy as np
import pandas as pd

from src.data_r2 import REPO_ROOT
from src.run_round3 import BLOCKS, backtest

OUT = REPO_ROOT / "outputs" / "local" / "divisha_round3"
TARGET_VOL = 0.15
FLAT = 0.05


def shape(raw, vol, k):
    s = np.clip(k * raw, -1, 1)
    return np.where(np.abs(s) <= FLAT, 0.0, s)


def main():
    d = pd.read_csv(OUT / "round3_prediction_detail_v3_lstm.csv")
    raw, vol = d["P5_regime_switch"].to_numpy(), d["volatility_20"].to_numpy()   # vol: same-row input
    exante = np.sqrt(np.mean((raw * vol) ** 2)) * np.sqrt(252)
    k = TARGET_VOL / exante
    sig = np.round(shape(raw, vol, k), 4)
    sub = OUT / "submission_round3_signal_v5.csv"
    pd.DataFrame({"timestamp": range(851, 901), "signal": sig}).to_csv(sub, index=False, lineterminator="\n")
    chk = pd.read_csv(sub)
    assert sub.read_text().splitlines()[0] == "timestamp,signal" and chk["timestamp"].tolist() == list(range(851, 901))
    assert chk["signal"].between(-1, 1).all() and np.isfinite(chk["signal"]).all()
    ds = np.abs(np.diff(np.concatenate([[0.0], sig])))
    print(f"scale k = {k:.2f} (v3 ex-ante vol {exante:.1%} -> target {TARGET_VOL:.0%})")
    print(f"net {sig.mean():+.3f}  gross {np.abs(sig).mean():.3f}  L/S/F {(sig > FLAT).sum()}/{(sig < -FLAT).sum()}/{(sig == 0).sum()}"
          f"  max |s| {np.abs(sig).max():.2f}  turnover {ds.sum():.2f}  ex-ante vol {np.sqrt(np.mean((sig * vol) ** 2)) * np.sqrt(252):.1%}")
    # Development check of the same shaping applied to v3's dev positions (Sharpe is scale-free; deadband is not).
    z = np.load(OUT / "dev_positions_v3_lstm.npz")
    for name, f in [("v3", lambda x: x), ("v5 shaping", lambda x: shape(x, None, k))]:
        rs = [backtest(f(z[f"{a}-{b}__real__P5_regime_switch"]), z[f"{a}-{b}__real__ret"], 0.001) for a, b in BLOCKS]
        print(f"dev {name:<11} real Sharpe {np.mean([q['sharpe'] for q in rs]):.3f}  mean return {np.mean([q['total_return'] for q in rs]):+.4f}"
              f"  mean MDD {np.mean([q['mdd'] for q in rs]):.3f}  max MDD {np.max([q['mdd'] for q in rs]):.3f}")
    print("wrote", sub, hashlib.sha256(sub.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
