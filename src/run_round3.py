"""Round three: crash-aware hybrid signal in [-1, 1] for timestamps 851-900.

    python -m src.run_round3

Hybrid model (every component trained with simulated crashes, on clipped,
volatility-normalised features so nothing extrapolates wildly at 5-6x vol):
  * logistic regression (C=1) + histogram gradient boosting, both fitted on real
    training rows plus Monte Carlo crash rows (25% of the real weight);
  * p_up = mean of the two.
Signal policies turn p_up and volatility into a position. They are compared in
a walk-forward backtest on real blocks 451-849 and on Monte Carlo crash paths,
net of trading costs, and one policy is chosen by a rule fixed in advance.

Backtest convention (assumed; organizers' rule not given): signal at row t is
held from close t to close t+1; pnl_t = s_t * r_{t+1} - cost * |s_t - s_{t-1}|.
Realized returns come from the labelled close column (evaluation only; close
is never a model input).
"""
from __future__ import annotations

import src  # noqa: F401  (thread caps before numerical imports)

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.crash_sim import simulate_crashes
from src.data_r2 import REPO_ROOT, load_r2, verify_source_r2
from src.features_r2 import FEATURE_NAMES_R2, assign_regimes, fit_regime_cutoffs, sample_timestamps, windows
from src.features_r3 import CLOSE_FEATURES, build_features_r3, close_features
from src.models.logistic import sklearn_lr
from src.models.lstm_attn import configure_torch, fit_lstm, predict_lstm

CSV3 = REPO_ROOT / "KESTREL_EM_D01_round3.csv"
OUT = REPO_ROOT / "outputs" / "local" / "divisha_round3"
NOTE = REPO_ROOT / "experiments" / "divisha" / "round3_results.md"
JSON_PATH = REPO_ROOT / "experiments" / "divisha" / "round3_results.json"

SEED = 42
SIM_FRAC = 0.25
CLIP_Z = 5.0
GAIN = 5.0          # model signal = clip(GAIN * (2p - 1), -1, 1)
BAND = 0.02         # dead band on |2p - 1| for the banded policy
COSTS = [0.0, 0.001, 0.0025]   # per unit of |change in signal|
SELECT_COST = 0.001
BLOCKS = [(451, 500), (501, 550), (551, 600), (601, 650), (651, 700), (701, 750), (751, 800), (801, 849)]
POLICY_ORDER = ["P0_long_full", "P1_long_volscaled", "P3_blend", "P2_model_volscaled", "P4_model_band",
                "P5_regime_switch", "P6_trend_volscaled", "P7_flat"]
IDX = {n: i for i, n in enumerate(FEATURE_NAMES_R2)}
EXTRA = ["r1_vol", "r5_vol", "mom_vol", "hl_vol", "osma10_vol", "smagap_vol"]


def add_extra(base: np.ndarray) -> np.ndarray:
    """Volatility-normalised versions of the return/range features (same row only)."""
    v = base[:, IDX["volatility_20"]]
    s5 = v * np.sqrt(5)
    ex = np.column_stack([
        base[:, IDX["oo_ret_1"]] / v, base[:, IDX["oo_ret_5"]] / s5, base[:, IDX["momentum_5"]] / s5,
        base[:, IDX["hl_range"]] / v, base[:, IDX["open_sma10"]] / v, base[:, IDX["sma10_sma30"]] / v,
    ])
    return np.column_stack([base, ex])


LSTM_SEEDS = (42, 43, 44)


class Hybrid:
    """LR + HGB (+ optional attention LSTM) on clipped standardized, volatility-normalised
    features, all trained with MC crash rows (LSTM: worst-group objective incl. sim_crash)."""

    def __init__(self, X_raw, feats, y, train_ts, fold_index, use_lstm=False):
        self.feats, self.use_lstm = feats, use_lstm
        tr = feats.loc[train_ts]
        base = tr.to_numpy()
        Xtr = add_extra(base)
        self.scaler = StandardScaler().fit(Xtr)
        self.med_vol = float(tr["volatility_20"].median())
        self.max_vol = float(tr["volatility_20"].max())
        cut = fit_regime_cutoffs(tr)
        train_raw = X_raw.loc[X_raw["timestamp"] <= max(train_ts)]
        aug = simulate_crashes(train_raw, self.med_vol, cut.vol_high, [SEED, fold_index, 0], extra=close_features)
        self.stress = simulate_crashes(train_raw, self.med_vol, cut.vol_high, [SEED, fold_index, 1], extra=close_features)
        self.sim_hashes = {"aug": aug.sha256, "stress": self.stress.sha256}
        Z = self._z(np.vstack([Xtr, add_extra(aug.W[:, -1, :])]))
        yy = np.concatenate([y.loc[train_ts].to_numpy().astype(int), aug.y])
        n, m = len(train_ts), len(aug.y)
        sw = np.concatenate([np.ones(n), np.full(m, SIM_FRAC * n / m)])
        self.lr = sklearn_lr().fit(Z, yy, sample_weight=sw)
        self.hgb = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=False, random_state=SEED).fit(Z, yy, sample_weight=sw)
        if use_lstm:
            self.arr = self._z(add_extra(feats.to_numpy()))
            self.pos = pd.Series(np.arange(len(feats)), index=feats.index)
            W = np.concatenate([windows(self.arr, self.pos.loc[train_ts].to_numpy()), self._zw(aug.W)])
            G = np.zeros((len(yy), 5), dtype=bool)
            G[:n, :4] = assign_regimes(tr, cut).to_numpy()
            G[n:, 4] = True
            self.lstm = [fit_lstm(W, yy, seed, sample_weight=sw, groups=G, lam=0.5)[0] for seed in LSTM_SEEDS]

    def _z(self, X):
        return np.clip(self.scaler.transform(X), -CLIP_Z, CLIP_Z)

    def _zw(self, W):
        n, L, F = W.shape
        return self._z(add_extra(W.reshape(-1, F))).reshape(n, L, -1)

    def _combine(self, Zrows, Zwin):
        out = {"p_lr": self.lr.predict_proba(Zrows)[:, 1], "p_hgb": self.hgb.predict_proba(Zrows)[:, 1]}
        parts = [out["p_lr"], out["p_hgb"]]
        if self.use_lstm:
            out["p_lstm"] = np.mean([predict_lstm(m, Zwin) for m in self.lstm], axis=0)
            parts.append(out["p_lstm"])
        out["p"] = np.mean(parts, axis=0)
        return out

    def predict_ts(self, ts) -> dict:
        Zrows = self._z(add_extra(self.feats.loc[ts].to_numpy()))
        Zwin = windows(self.arr, self.pos.loc[ts].to_numpy()) if self.use_lstm else None
        return self._combine(Zrows, Zwin)

    def predict_sim(self, W) -> dict:
        Zw = self._zw(W)
        return self._combine(Zw[:, -1, :], Zw if self.use_lstm else None)


def policies(p, base, med_vol, max_vol) -> dict:
    vol = base[:, IDX["volatility_20"]]
    vs = np.clip(med_vol / vol, 0.0, 1.0)
    edge = 2 * p - 1
    m = np.clip(GAIN * edge, -1, 1)
    mom = np.clip(base[:, IDX["momentum_5"]] / (vol * np.sqrt(5)) / 2, -1, 1)
    crash = vol > max_vol
    return {
        "P0_long_full": np.ones_like(p),
        "P1_long_volscaled": vs,
        "P2_model_volscaled": m * vs,
        "P3_blend": np.clip(0.5 * vs + 0.5 * m * vs, -1, 1),
        "P4_model_band": np.where(np.abs(edge) < BAND, 0.0, m) * vs,
        "P5_regime_switch": np.where(crash, m * vs, np.clip(0.5 * vs + 0.5 * m * vs, -1, 1)),
        "P6_trend_volscaled": mom * vs,
        "P7_flat": np.zeros_like(p),
    }


def backtest(s, r, cost) -> dict:
    ds = np.abs(np.diff(np.concatenate([[0.0], s])))
    pnl = s * r - cost * ds
    eq = np.cumprod(1 + pnl)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    sd = pnl.std(ddof=1)
    return {"sharpe": float(pnl.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
            "total_return": float(eq[-1] - 1), "mdd": float(np.max(1 - eq / peak)),
            "turnover": float(ds.sum()), "cost_paid": float(cost * ds.sum()),
            "hit_rate": float(np.mean(np.sign(s[s != 0]) == np.sign(r[s != 0]))) if (s != 0).any() else None}


def main(argv=None):
    use_lstm = argv is not None and "--lstm" in argv
    tag = "v3_lstm" if use_lstm else "v2"
    if use_lstm:
        configure_torch(min(4, __import__("os").cpu_count() or 1))
    t0 = time.perf_counter()
    hashes = verify_source_r2(CSV3)          # the round-three file
    X, y = load_r2(900, CSV3)
    raw = pd.read_csv(CSV3)
    feats, rec = build_features_r3(raw)      # current close rebuilt from rows <= t only (851-900)
    n_warm = int(feats.index[0] - 1)
    sample = sample_timestamps(feats)
    price = raw.set_index("timestamp")["close"]   # realized returns for backtests (labelled rows only)
    r_next = (price.shift(-1) / price - 1)   # NaN for t >= 850

    real, mc, diag = {}, {}, {}
    for i, (b0, b1) in enumerate(BLOCKS, start=1):
        train_ts = [t for t in sample if t <= b0 - 2]
        ev_ts = [t for t in sample if b0 <= t <= b1]
        h = Hybrid(X, feats, y, train_ts, i, use_lstm)
        base_ev = feats.loc[ev_ts].to_numpy()
        pr = h.predict_ts(ev_ts)
        pol = policies(pr["p"], base_ev, h.med_vol, h.max_vol)
        r = r_next.loc[ev_ts].to_numpy()
        real[f"{b0}-{b1}"] = {k: {str(c): backtest(v, r, c) for c in COSTS} for k, v in pol.items()}
        # Monte Carlo crash paths (independent stress set, never trained on)
        st = h.stress
        base_st = st.W[:, -1, :]
        ps = h.predict_sim(st.W)
        pol_s = policies(ps["p"], base_st, h.med_vol, h.max_vol)
        res = {}
        for k, v in pol_s.items():
            per = [backtest(v[st.window_id == w], st.ret[st.window_id == w], SELECT_COST) for w in np.unique(st.window_id)]
            res[k] = {"mean_sharpe": float(np.mean([q["sharpe"] for q in per])),
                      "median_mdd": float(np.median([q["mdd"] for q in per])),
                      "p90_mdd": float(np.quantile([q["mdd"] for q in per], 0.9)),
                      "mean_return": float(np.mean([q["total_return"] for q in per])),
                      "p10_return": float(np.quantile([q["total_return"] for q in per], 0.1)),
                      "mean_turnover": float(np.mean([q["turnover"] for q in per]))}
        mc[f"{b0}-{b1}"] = res
        diag[f"{b0}-{b1}"] = {
            "n_train": len(train_ts), "sim": h.sim_hashes,
            "real_p": [float(pr["p"].min()), float(pr["p"].max())],
            "stress_p": [float(ps["p"].min()), float(ps["p"].max())],
            "stress_share_|2p-1|>0.3": float(np.mean(np.abs(2 * ps["p"] - 1) > 0.3)),
            "lr_hgb_corr_real": float(np.corrcoef(pr["p_lr"], pr["p_hgb"])[0, 1]),
        }
        print(f"block {b0}-{b1}: {time.perf_counter() - t0:.0f}s", flush=True)

    # Aggregate and select (rule fixed before running):
    #   score = 0.5 * mean real-block Sharpe + 0.5 * mean MC-crash Sharpe, net of 10 bps;
    #   feasible if mean real-block MDD <= 15% and median MC-crash MDD <= 25%;
    #   ties within 0.05 go to the simpler policy (POLICY_ORDER); P7_flat is the fallback.
    summary = {}
    for k in POLICY_ORDER:
        rs = [real[b][k][str(SELECT_COST)] for b in real]
        ms = [mc[b][k] for b in mc]
        summary[k] = {
            "real_mean_sharpe": float(np.mean([q["sharpe"] for q in rs])),
            "real_blocks_positive": int(sum(q["sharpe"] > 0 for q in rs)),
            "real_mean_return": float(np.mean([q["total_return"] for q in rs])),
            "real_mean_mdd": float(np.mean([q["mdd"] for q in rs])), "real_max_mdd": float(np.max([q["mdd"] for q in rs])),
            "real_mean_turnover": float(np.mean([q["turnover"] for q in rs])),
            "mc_mean_sharpe": float(np.mean([q["mean_sharpe"] for q in ms])),
            "mc_median_mdd": float(np.mean([q["median_mdd"] for q in ms])),
            "mc_p90_mdd": float(np.mean([q["p90_mdd"] for q in ms])),
            "mc_mean_return": float(np.mean([q["mean_return"] for q in ms])),
            "mc_p10_return": float(np.mean([q["p10_return"] for q in ms])),
        }
        s = summary[k]
        s["score"] = 0.5 * s["real_mean_sharpe"] + 0.5 * s["mc_mean_sharpe"]
        s["feasible"] = bool(s["real_mean_mdd"] <= 0.15 and s["mc_median_mdd"] <= 0.25)
    feas = [k for k in POLICY_ORDER if summary[k]["feasible"] and k != "P7_flat"]
    if feas:
        best = max(summary[k]["score"] for k in feas)
        chosen = next(k for k in POLICY_ORDER if k in feas and summary[k]["score"] >= best - 0.05)
    else:
        chosen = "P7_flat"
    print("chosen:", chosen, flush=True)

    # Final fit on all labelled samples (<= 850) and signal for 851-900.
    train_ts = [t for t in sample if t <= 850]
    pred_ts = list(range(851, 901))
    h = Hybrid(X, feats, y, train_ts, 9, use_lstm)
    base_pr = feats.loc[pred_ts].to_numpy()
    pr = h.predict_ts(pred_ts)
    pol = policies(pr["p"], base_pr, h.med_vol, h.max_vol)
    sig = np.clip(np.round(pol[chosen], 4), -1, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    sub = OUT / f"submission_round3_signal_{tag}.csv"
    pd.DataFrame({"timestamp": pred_ts, "signal": sig}).to_csv(sub, index=False, lineterminator="\n")
    chk = pd.read_csv(sub)
    assert sub.read_text().splitlines()[0] == "timestamp,signal"
    assert chk["timestamp"].tolist() == pred_ts and len(chk) == 50
    assert chk["signal"].between(-1, 1).all() and np.isfinite(chk["signal"]).all()
    detail = OUT / f"round3_prediction_detail_{tag}.csv"
    pd.DataFrame({"timestamp": pred_ts, **{k: v for k, v in pr.items()},
                  "volatility_20": base_pr[:, IDX["volatility_20"]], **{k: v for k, v in pol.items()}}).to_csv(
        detail, index=False, float_format="%.6g", lineterminator="\n")
    final = {"chosen": chosen, "train_last": 850, "n_train": len(train_ts), "sim": h.sim_hashes,
             "p_range": [float(pr["p"].min()), float(pr["p"].max())], "p_mean": float(pr["p"].mean()),
             "signal_mean": float(sig.mean()), "signal_abs_mean": float(np.abs(sig).mean()),
             "signal_min": float(sig.min()), "signal_max": float(sig.max()),
             "turnover": float(np.abs(np.diff(np.concatenate([[0], sig]))).sum()),
             "submission_sha256": hashlib.sha256(sub.read_bytes()).hexdigest(),
             "all_policy_signals_first5": {k: [float(x) for x in v[:5]] for k, v in pol.items()}}
    rb = rec["rebuilt"]
    recon = {"rows_rebuilt": int(rb.sum()), "max_mom_vs_sma10_diff": float(np.nanmax(np.abs(rec["est_mom"][rb] - rec["est_sma10"][rb])))}
    res = {"version": tag, "use_lstm": use_lstm, "close_reconstruction": recon, "features": list(feats.columns),
           "dataset": hashes, "warmup_rows": n_warm, "blocks": BLOCKS, "costs": COSTS, "select_cost": SELECT_COST,
           "summary": summary, "chosen": chosen, "real": real, "mc": mc, "diag": diag, "final": final,
           "runtime_s": time.perf_counter() - t0}
    JSON_PATH.with_name(f"round3_results_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")
    write_note(res, sub, NOTE.with_name(f"round3_results_{tag}.md"))
    print(json.dumps({k: final[k] for k in ("chosen", "p_range", "signal_mean", "signal_min", "signal_max", "turnover")}))
    print("wrote", sub, flush=True)


def write_note(res, sub, path):
    S, fn = res["summary"], res["final"]
    model = "LR + HGB + attention LSTM (worst-group with sim_crash, 3 seeds)" if res["use_lstm"] else "LR + HGB"
    L = [f"# Round three {res['version']}: crash-aware hybrid signal ({model})\n",
         f"Chosen policy: **{res['chosen']}**. Submission (local, not committed): `{sub.relative_to(REPO_ROOT)}`, "
         f"SHA-256 `{fn['submission_sha256']}`.\n",
         "Command: `python -m src.run_round3`. Data: round-three file, identical to the audited round-two data "
         f"(SHA-256 file `{res['dataset']['sha256_file']}`, CRLF form `{res['dataset']['sha256_crlf_normalised']}`).\n",
         f"Features ({len(res['features'])}): the 18 round-two causal features, plus close-based same-row features "
         f"({', '.join(CLOSE_FEATURES)}), plus 6 volatility-normalised versions inside the models. For 851–900 the current close is "
         f"rebuilt from rows ≤ t only (close[t−5]·(1+momentum_5[t])); {res['close_reconstruction']['rows_rebuilt']} rows rebuilt, "
         f"momentum vs sma_10 estimates agree within {res['close_reconstruction']['max_mom_vs_sma10_diff']:.3f}. "
         "Validation: hiding 801–850 and rebuilding gives max error 0.044 (0.013%). The next close is never reconstructed, "
         "and realized 851–900 returns are not computed or used for any choice.\n",
         "## Policy comparison (net of 10 bps per unit of signal change)\n",
         "Real = 8 walk-forward blocks 451–849 (models refitted per block on earlier rows, 1-row purge). "
         "MC = 200 independent simulated 50-day crash paths per block (3×/5× vol, injected −8% to −15% days), never used in training. "
         "MC returns are synthetic.\n",
         "| Policy | Score | Real Sharpe | Real blocks > 0 | Real mean return | Real mean / max MDD | Real turnover | MC Sharpe | MC median / p90 MDD | MC mean / p10 return | Feasible |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in POLICY_ORDER:
        s = S[k]
        L.append(f"| {k} | {s['score']:.3f} | {s['real_mean_sharpe']:.3f} | {s['real_blocks_positive']}/8 | {s['real_mean_return']:.4f} | "
                 f"{s['real_mean_mdd']:.3f} / {s['real_max_mdd']:.3f} | {s['real_mean_turnover']:.2f} | {s['mc_mean_sharpe']:.3f} | "
                 f"{s['mc_median_mdd']:.3f} / {s['mc_p90_mdd']:.3f} | {s['mc_mean_return']:.4f} / {s['mc_p10_return']:.4f} | {'yes' if s['feasible'] else 'no'} |")
    L += ["\nSelection rule (fixed before running): maximise 0.5·real Sharpe + 0.5·MC Sharpe at 10 bps, subject to mean real MDD ≤ 15% "
          "and median MC MDD ≤ 25%; ties within 0.05 go to the simpler policy; flat is the fallback.\n",
          "## Final signal for 851–900\n",
          f"Trained on samples 25–850 ({fn['n_train']}). p_up range {fn['p_range'][0]:.3f}–{fn['p_range'][1]:.3f} (mean {fn['p_mean']:.3f}); "
          f"signal range {fn['signal_min']:.3f} to {fn['signal_max']:.3f}, mean {fn['signal_mean']:.3f}, mean |signal| {fn['signal_abs_mean']:.3f}, "
          f"turnover {fn['turnover']:.2f}.\n",
          "## Crash-awareness checks per block\n",
          "| Block | n train | real p range | stress p range | stress share \\|2p−1\\| > 0.3 | corr(LR, HGB) |",
          "|---|---|---|---|---|---|"]
    for b, d in res["diag"].items():
        L.append(f"| {b} | {d['n_train']} | {d['real_p'][0]:.3f}–{d['real_p'][1]:.3f} | {d['stress_p'][0]:.3f}–{d['stress_p'][1]:.3f} | "
                 f"{d['stress_share_|2p-1|>0.3']:.3f} | {d['lr_hgb_corr_real']:.2f} |")
    L += ["\n## Assumptions and warnings\n",
          "- Scoring formula unknown; assumed daily pnl = signal_t × close-to-close return t→t+1 minus cost × |Δsignal|, Sharpe annualised with √252.",
          "- Same-row timing: features at row t (including sma_10/sma_30/momentum_5, which contain close[t]) are treated as known when setting signal_t.",
          "- No future rows are used: signal_t depends on rows ≤ t only. Hidden closes are not reconstructed from later rows.",
          "- Block 801–850 was the round-two reserved block; it is used here as development data. Return for t = 850 is unknown, so the last block ends at 849.",
          "- Simulator design (3×/5× vol, 1–2 drops of −8% to −15%, burn-in from real training prices) is a design choice; MC Sharpe measures robustness, not real skill.",
          "- Rounds 1–2 showed no directional edge on real data; the model component is expected to be close to noise.",
          "- Costs assumed 10 bps per unit change for selection; 0 and 25 bps are in the JSON.",
          "- Initial submission (v1, commit ea6c09d, no close features) scored 35.8 on the leaderboard."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
