"""Round three post-mortem (after final scoring): why v4 scored 36.4 vs v3 53.008.

    python -m src.postmortem_round3

Uses realized 851-900 returns rebuilt from the round-three file (each close[t]
from rows <= t). This is evaluation after the submissions were scored; none of
it was used to build v1-v4.
"""
from __future__ import annotations

import src  # noqa: F401

import numpy as np
import pandas as pd

from src.data_r2 import REPO_ROOT
from src.features_r3 import reconstruct_close

D = REPO_ROOT / "outputs" / "local" / "divisha_round3"
FILES = {"v1": ("submission_round3_signal.csv", 35.8), "v3": ("submission_round3_signal_v3_lstm.csv", 53.008),
         "v4": ("submission_round3_signal_v4.csv", 36.4), "v2 (not submitted)": ("submission_round3_signal_v2.csv", None),
         "teammate (not submitted by us)": ("submission_round3_teammate.csv", None)}
NOTE = REPO_ROOT / "experiments" / "divisha" / "round3_postmortem.md"


def bt(s, r, cost=0.001):
    ds = np.abs(np.diff(np.concatenate([[0.0], s])))
    pnl = s * r - cost * ds
    eq = np.cumprod(1 + pnl)
    pk = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    nz = s != 0
    return {"sharpe": pnl.mean() / pnl.std(ddof=1) * np.sqrt(252), "ret": (eq[-1] - 1) * 100,
            "mdd": np.max(1 - eq / pk) * 100, "hit": np.mean(np.sign(s[nz]) == np.sign(r[nz])),
            "mean_abs": np.abs(s).mean(), "flips": int((np.sign(s[1:]) * np.sign(s[:-1]) < 0).sum())}


def main():
    raw = pd.read_csv(REPO_ROOT / "KESTREL_EM_D01_round3.csv")
    c = pd.Series(reconstruct_close(raw)["close"], index=raw["timestamp"])
    o = raw.set_index("timestamp")["open"]
    T = list(range(851, 900))   # close[901] unknown, so day 900's next-day return is unavailable
    al = {"next close-to-close": np.array([c[t + 1] / c[t] - 1 for t in T]),
          "next open-to-open": np.array([o[t + 1] / o[t] - 1 for t in T]),
          "same-row return_1d": np.array([c[t] / c[t - 1] - 1 for t in T])}
    S = {k: pd.read_csv(D / f).set_index("timestamp")["signal"] for k, (f, _) in FILES.items()}
    L = ["# Round three post-mortem: why v4 (36.4) scored below v3 (53.008)\n",
         "Evaluation after final scoring, on 851–899 returns rebuilt from the round-three file (each close from rows ≤ t; "
         "rebuild error ≤ 0.013% when tested on 801–850). Nothing here was used to build any submission. Cost 10 bps per unit change.\n"]
    for name, r in al.items():
        L += [f"## Signal_t × {name} (buy & hold {((1 + r).prod() - 1) * 100:+.1f}%)\n",
              "| File | Leaderboard | Sharpe | Return % | Max DD % | Hit rate | Mean \\|s\\| | Sign flips |", "|---|---|---|---|---|---|---|---|"]
        for k, (f, lb) in FILES.items():
            m = bt(S[k].loc[T].to_numpy(), r)
            L.append(f"| {k} | {lb if lb is not None else '—'} | {m['sharpe']:.2f} | {m['ret']:.2f} | {m['mdd']:.2f} | {m['hit']:.2f} | {m['mean_abs']:.3f} | {m['flips']} |")
        L.append("")
    r = al["next close-to-close"]
    v3 = S["v3"].to_numpy()
    prev, nxt = v3[:48], r[1:49]
    lag_hit = np.mean(np.sign(prev[prev != 0]) == np.sign(nxt[prev != 0]))
    hist = c.loc[26:850].pct_change().dropna()
    rr = pd.Series(np.array([c[t] / c[t - 1] - 1 for t in range(851, 901)]))
    L += ["## Timing check\n",
          f"- Lag-1 autocorrelation of daily returns: 851–900 = {rr.autocorr(1):+.3f}; labelled history (26–850) = {hist.autocorr(1):+.3f}.",
          f"- v3's own call is right on {np.mean(np.sign(v3[:49][v3[:49] != 0]) == np.sign(r[v3[:49] != 0])):.0%} of days; "
          f"yesterday's v3 call applied to today is right on {lag_hit:.0%}. EMA smoothing mixes in yesterday's call.",
          f"- Daily return sd: 851–900 = {rr.std():.4f}; history = {hist.std():.4f}."]
    NOTE.write_text("\n".join(L) + "\n")
    print(NOTE.read_text())


if __name__ == "__main__":
    main()
