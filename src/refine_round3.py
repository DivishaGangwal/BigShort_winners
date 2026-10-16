"""Round three v4: prespecified turnover refinements of the v3 policy.

    python -m src.refine_round3        (after `python -m src.run_round3 --lstm`)

v3 = P5_regime_switch on the LR + HGB + LSTM hybrid; leaderboard 53.008
(v1, volatility-scaled long, scored 35.8). Candidates, fixed before evaluation:
  R0_v3              v3 unchanged
  R1_ema0.5          s_t = 0.5 * raw_t + 0.5 * s_{t-1}
  R2_hysteresis0.05  s_t = raw_t if |raw_t - s_{t-1}| >= 0.05 else s_{t-1}
Positions start flat (s_{-1} = 0) in every development block, every simulated
window and on day 851. Each refined s_t depends only on raw_{<=t}, so no future
rows enter. Same selection rule as run_round3: maximise 0.5 * real Sharpe +
0.5 * MC-crash Sharpe at 10 bps, subject to mean real MDD <= 15% and median MC
MDD <= 25%; candidates within 0.05 of the best go to the simpler one (R0 < R1 < R2).
"""
from __future__ import annotations

import src  # noqa: F401

import hashlib
import json

import numpy as np
import pandas as pd

from src.data_r2 import REPO_ROOT
from src.run_round3 import BLOCKS, COSTS, SELECT_COST, backtest

OUT = REPO_ROOT / "outputs" / "local" / "divisha_round3"
NOTE = REPO_ROOT / "experiments" / "divisha" / "round3_results_v4.md"
BASE = "P5_regime_switch"
ORDER = ["R0_v3", "R1_ema0.5", "R2_hysteresis0.05"]
LEADERBOARD = {"v1 (volatility-scaled long)": 35.8, "v3 (LSTM hybrid, R0)": 53.008}


def refine(raw: np.ndarray, name: str) -> np.ndarray:
    if name == "R0_v3":
        return raw.astype(float).copy()
    out, prev = np.empty(len(raw)), 0.0
    for i, x in enumerate(raw):
        if name == "R1_ema0.5":
            prev = 0.5 * x + 0.5 * prev
        elif name == "R2_hysteresis0.05":
            prev = x if abs(x - prev) >= 0.05 else prev
        else:
            raise ValueError(name)
        out[i] = prev
    return out


def main():
    z = np.load(OUT / "dev_positions_v3_lstm.npz")
    summary = {}
    for name in ORDER:
        real, mc = [], []
        for b0, b1 in BLOCKS:
            k = f"{b0}-{b1}"
            real.append(backtest(refine(z[f"{k}__real__{BASE}"], name), z[f"{k}__real__ret"], SELECT_COST))
            sraw, sret, wid = z[f"{k}__stress__{BASE}"], z[f"{k}__stress__ret"], z[f"{k}__stress__wid"]
            per = [backtest(refine(sraw[wid == w], name), sret[wid == w], SELECT_COST) for w in np.unique(wid)]
            mc.append({"sharpe": np.mean([q["sharpe"] for q in per]), "mdd": np.median([q["mdd"] for q in per]),
                       "p90_mdd": np.quantile([q["mdd"] for q in per], 0.9),
                       "ret": np.mean([q["total_return"] for q in per]), "turnover": np.mean([q["turnover"] for q in per])})
        s = {
            "real_mean_sharpe": float(np.mean([q["sharpe"] for q in real])),
            "real_blocks_positive": int(sum(q["sharpe"] > 0 for q in real)),
            "real_mean_return": float(np.mean([q["total_return"] for q in real])),
            "real_mean_mdd": float(np.mean([q["mdd"] for q in real])), "real_max_mdd": float(np.max([q["mdd"] for q in real])),
            "real_mean_turnover": float(np.mean([q["turnover"] for q in real])),
            "real_cost_paid": float(np.mean([q["cost_paid"] for q in real])),
            "mc_mean_sharpe": float(np.mean([q["sharpe"] for q in mc])), "mc_median_mdd": float(np.mean([q["mdd"] for q in mc])),
            "mc_p90_mdd": float(np.mean([q["p90_mdd"] for q in mc])), "mc_mean_return": float(np.mean([q["ret"] for q in mc])),
            "mc_mean_turnover": float(np.mean([q["turnover"] for q in mc])),
        }
        s["score"] = 0.5 * s["real_mean_sharpe"] + 0.5 * s["mc_mean_sharpe"]
        s["feasible"] = bool(s["real_mean_mdd"] <= 0.15 and s["mc_median_mdd"] <= 0.25)
        summary[name] = s
    feas = [n for n in ORDER if summary[n]["feasible"]]
    best = max(summary[n]["score"] for n in feas)
    chosen = next(n for n in ORDER if n in feas and summary[n]["score"] >= best - 0.05)

    detail = pd.read_csv(OUT / "round3_prediction_detail_v3_lstm.csv")
    raw = detail[BASE].to_numpy()
    submitted = pd.read_csv(OUT / "submission_round3_signal_v3_lstm.csv")["signal"].to_numpy()
    assert np.allclose(np.clip(np.round(raw, 4), -1, 1), submitted, atol=1.5e-4), "v3 raw signal does not match the submitted file"
    final = {}
    for name in ORDER:
        sig = np.clip(np.round(refine(raw, name), 4), -1, 1)
        ds = np.abs(np.diff(np.concatenate([[0.0], sig])))
        final[name] = {"signal": sig, "turnover": float(ds.sum()), "mean_abs": float(np.abs(sig).mean()),
                       "corr_with_v3": float(np.corrcoef(sig, submitted)[0, 1])}
    sig = submitted if chosen == "R0_v3" else final[chosen]["signal"]
    sub = OUT / "submission_round3_signal_v4.csv"
    pd.DataFrame({"timestamp": range(851, 901), "signal": sig}).to_csv(sub, index=False, lineterminator="\n")
    chk = pd.read_csv(sub)
    assert sub.read_text().splitlines()[0] == "timestamp,signal" and chk["timestamp"].tolist() == list(range(851, 901))
    assert chk["signal"].between(-1, 1).all() and np.isfinite(chk["signal"]).all()
    sha = hashlib.sha256(sub.read_bytes()).hexdigest()

    L = ["# Round three v4: turnover refinements of v3\n",
         "Leaderboard so far: " + "; ".join(f"{k}: {v}" for k, v in LEADERBOARD.items()) + ".\n",
         "Candidates and rule were fixed before evaluation (see `src/refine_round3.py`). Every refined position uses only "
         "the v3 positions up to that day; nothing from 851–900 outcomes is used.\n",
         "| Candidate | Score | Real Sharpe | Real blocks > 0 | Real mean return | Real mean / max MDD | Real turnover | MC Sharpe | MC median / p90 MDD | MC turnover | 851–900 turnover | 851–900 mean \\|s\\| | Corr. with v3 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ORDER:
        s, f = summary[n], final[n]
        L.append(f"| {n} | {s['score']:.3f} | {s['real_mean_sharpe']:.3f} | {s['real_blocks_positive']}/8 | {s['real_mean_return']:.4f} | "
                 f"{s['real_mean_mdd']:.3f} / {s['real_max_mdd']:.3f} | {s['real_mean_turnover']:.2f} | {s['mc_mean_sharpe']:.3f} | "
                 f"{s['mc_median_mdd']:.3f} / {s['mc_p90_mdd']:.3f} | {s['mc_mean_turnover']:.2f} | {f['turnover']:.2f} | "
                 f"{f['mean_abs']:.3f} | {f['corr_with_v3']:.3f} |")
    L += [f"\nChosen: **{chosen}**. File `{sub.relative_to(REPO_ROOT)}` (local), SHA-256 `{sha}`"
          + (" — identical to the submitted v3 file." if chosen == "R0_v3" else ".") + "\n"]
    NOTE.write_text("\n".join(L) + "\n")
    json.dump({"summary": summary, "chosen": chosen, "sha256": sha,
               "final": {n: {k: v for k, v in f.items() if k != "signal"} for n, f in final.items()}},
              open(NOTE.with_suffix(".json"), "w"), indent=1)
    print(open(NOTE).read())


if __name__ == "__main__":
    main()
