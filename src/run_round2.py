"""Round two: LR and attention-LSTM arms under regimes and simulated crashes.

    python -m src.run_round2

1. Runs the test suite (abort on failure) and the round-two fact audit.
2. Development: seven expanding folds on rows <= 800 (baseline, 9 LR arms,
   9 LSTM arms x 3 seeds), real validation metrics and synthetic stress tests.
3. Applies the prespecified selection rule. If the training-prior baseline
   wins, stops before the reserved evaluation.
4. Otherwise evaluates the selected configuration once on 801-850, refits on
   1-850 and predicts 851-900, writing the two submission files locally.
Aggregate results go to experiments/divisha/; models, row-level predictions
and submission files stay in outputs/local/ (gitignored).
"""
from __future__ import annotations

import src  # noqa: F401  (thread caps before numerical imports)

import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_r2 import verify_round2_facts
from src.crash_sim import SimConfig, simulate_crashes
from src.data_r2 import REPO_ROOT, load_r2, verify_source_r2
from src.features_r2 import (
    FEATURE_NAMES_R2, FEATURE_VERSION_R2, REGIMES, SIM_GROUP, build_features_r2, prepare_fold_r2,
    sample_timestamps,
)
from src.metrics import binary_metrics, regime_metrics, reliability
from src.models.logistic import PriorModel, eta_for, fit_worst_regime, sklearn_lr
from src.models.lstm_attn import configure_torch, fit_lstm, predict_lstm
from src.splits_r2 import DEV_FOLDS, FINAL_FOLD, MANIFEST_VERSION_R2, RESERVED_FOLD, fold_ts, manifest_bytes, write_manifest

CONFIG_PATH = REPO_ROOT / "configs" / "models" / "divisha_round2.json"
MANIFEST_PATH = REPO_ROOT / "configs" / "splits" / f"{MANIFEST_VERSION_R2}.json"
RESULTS_DIR = REPO_ROOT / "experiments" / "divisha"
NOTE_PATH = RESULTS_DIR / "round2_results.md"
JSON_PATH = RESULTS_DIR / "round2_results.json"
LOCAL_DIR = REPO_ROOT / "outputs" / "local" / "divisha_round2"
COMMAND = "python -m src.run_round2"
MANUAL_START, MANUAL_END = "<!-- manual:start -->", "<!-- manual:end -->"

ARM_RANK = {"A": 0, "B": 1, "C": 1, "D": 1, "E": 2}


def config_list(cfg) -> list[dict]:
    a = cfg["arms"]
    out = [{"name": "baseline_prior", "model": "baseline", "arm": None, "param": None}]
    for model in ("LR", "LSTM"):
        out.append({"name": f"{model}_A", "model": model, "arm": "A", "param": None})
        out += [{"name": f"{model}_B_lam{l}", "model": model, "arm": "B", "param": l} for l in a["B_lambdas"]]
        out += [{"name": f"{model}_C_w{w}", "model": model, "arm": "C", "param": w} for w in a["C_weights"]]
        out.append({"name": f"{model}_D_mc", "model": model, "arm": "D", "param": a["D_sim_weight_fraction"]})
        out.append({"name": f"{model}_E_mc_lam{a['E_lambda']}", "model": model, "arm": "E", "param": a["E_lambda"]})
    return out


def complexity_key(c) -> tuple:
    if c["model"] == "baseline":
        return (-1, 0, 0.0)
    return (0 if c["model"] == "LR" else 1, ARM_RANK[c["arm"]], float(c["param"] or 0))


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


# --------------------------------------------------------------------------- fold data

def build_fold(X, y, feats, f, fold_index, cfg):
    """Fold-local preprocessing and both simulated sets, from training rows only."""
    q = cfg["regimes"]
    fd = prepare_fold_r2(feats, y, f["train_ts"], f["eval_ts"], q["low_quantile"], q["high_quantile"])
    train_end = max(f["train_ts"])
    train_raw = X.loc[X["timestamp"] <= train_end]
    med_vol = float(feats.loc[f["train_ts"], "volatility_20"].median())
    s = cfg["simulator"]
    sc = SimConfig(n_windows=s["n_windows"], window=s["window"], burn_in=s["burn_in"], block=s["block"],
                   multipliers=tuple(s["multipliers"]), n_drops=tuple(s["n_drops"]),
                   drop_range=tuple(s["drop_range"]), gap_frac=s["gap_frac"])
    aug = simulate_crashes(train_raw, med_vol, fd.cutoffs.vol_high, [cfg["seed"], fold_index, 0], sc)
    stress = simulate_crashes(train_raw, med_vol, fd.cutoffs.vol_high, [cfg["seed"], fold_index, 1], sc)

    def scale(W):
        return fd.scaler.transform(W.reshape(-1, W.shape[-1])).reshape(W.shape)

    sim = {"aug_W": scale(aug.W), "aug_y": aug.y, "stress_W": scale(stress.W), "stress_y": stress.y,
           "stress_mult": stress.multiplier, "aug_sha256": aug.sha256, "stress_sha256": stress.sha256,
           "settings": aug.settings}
    return fd, sim


def arm_data(fd, sim, arm, param, cfg):
    """Training rows, weights and groups for one arm (LR uses the last window row)."""
    G = fd.g_train.to_numpy()
    n_real = len(fd.y_train)
    if arm in ("A", "B"):
        return fd.W_train, fd.y_train, None, G
    if arm == "C":
        stressed = (fd.g_train["high_vol"] | fd.g_train["high_rsi"]).to_numpy()
        return fd.W_train, fd.y_train, np.where(stressed, float(param), 1.0), G
    n_sim = len(sim["aug_y"])
    frac = cfg["arms"]["D_sim_weight_fraction"]
    W = np.concatenate([fd.W_train, sim["aug_W"]])
    yy = np.concatenate([fd.y_train, sim["aug_y"]])
    sw = np.concatenate([np.ones(n_real), np.full(n_sim, frac * n_real / n_sim)])
    groups = np.zeros((n_real + n_sim, len(REGIMES) + 1), dtype=bool)
    groups[:n_real, :len(REGIMES)] = G
    groups[n_real:, len(REGIMES)] = True
    return W, yy, sw, groups


def fit_lr(fd, sim, c, cfg, cache):
    L = cfg["logistic"]
    W, yy, sw, groups = arm_data(fd, sim, c["arm"], c["param"], cfg)
    X = W[:, -1, :]
    diag = {}
    if c["arm"] in ("A", "C", "D"):
        model = sklearn_lr().fit(X, yy, sample_weight=sw)
        diag["n_iter"] = int(model.n_iter_[0])
        cache[c["arm"]] = model
        proba = lambda Z: model.predict_proba(Z)[:, 1]
        return model, proba, diag
    lam = c["param"]
    warm = cache["A"] if c["arm"] == "B" else cache["D"]
    eta = eta_for(len(yy) if sw is None else sw.sum(), L["C"])
    model, diag = fit_worst_regime(X, yy, groups, lam, eta, warm.coef_.ravel(), warm.intercept_[0],
                                   maxiter=L["slsqp_maxiter"], ftol=L["slsqp_ftol"], feas_tol=L["feasibility_tol"],
                                   sample_weight=sw)
    return model, model.predict_proba1, diag


def fit_lstm_config(fd, sim, c, cfg, seed):
    L = cfg["lstm"]
    W, yy, sw, groups = arm_data(fd, sim, c["arm"], c["param"], cfg)
    lam = c["param"] if c["arm"] in ("B", "E") else 0.0
    model, hist = fit_lstm(W, yy, seed, sample_weight=sw, groups=groups, lam=lam, epochs=L["epochs"], lr=L["lr"],
                           weight_decay=L["weight_decay"], hidden=L["hidden"], dropout=L["dropout"])
    return model, {"final_train_objective": hist[-1], "first_train_objective": hist[0]}


def run_config(fd, sim, c, cfg, lr_cache, eval_windows):
    """Fit one configuration; return predictions on eval windows and stress set, runtime, diagnostics, models."""
    out = {"runtime": {}, "diag": {}, "models": None}
    if c["model"] == "baseline":
        t0 = time.perf_counter()
        m = PriorModel(fd.y_train)
        out["runtime"]["fit"] = time.perf_counter() - t0
        out["p_eval"] = m.predict_proba1(eval_windows)
        out["p_stress"] = m.predict_proba1(sim["stress_W"])
        out["models"] = m
        return out
    if c["model"] == "LR":
        t0 = time.perf_counter()
        model, proba, diag = fit_lr(fd, sim, c, cfg, lr_cache)
        out["runtime"]["fit"] = time.perf_counter() - t0
        out["p_eval"], out["p_stress"] = proba(eval_windows[:, -1, :]), proba(sim["stress_W"][:, -1, :])
        out["diag"], out["models"] = diag, model
        return out
    per_seed_eval, per_seed_stress, states = {}, {}, {}
    total = 0.0
    for seed in cfg["lstm"]["seeds"]:
        t0 = time.perf_counter()
        model, diag = fit_lstm_config(fd, sim, c, cfg, seed)
        rt = time.perf_counter() - t0
        total += rt
        out["runtime"][f"seed{seed}"] = rt
        out["diag"][f"seed{seed}"] = diag
        per_seed_eval[seed] = predict_lstm(model, eval_windows)
        per_seed_stress[seed] = predict_lstm(model, sim["stress_W"])
        states[seed] = {k: v.clone() for k, v in model.state_dict().items()}
    out["runtime"]["fit"] = total
    out["p_eval_seeds"], out["p_stress_seeds"] = per_seed_eval, per_seed_stress
    out["p_eval"] = np.mean(list(per_seed_eval.values()), axis=0)
    out["p_stress"] = np.mean(list(per_seed_stress.values()), axis=0)
    out["models"] = states
    return out


# --------------------------------------------------------------------------- metrics

def evaluate_real(y, p, groups, bins):
    table, ece = reliability(y, p, bins)
    return {**binary_metrics(y, p), "ece": ece, "reliability": table, **regime_metrics(y, p, groups)}


def evaluate_stress(y, p, mult, bins):
    table, ece = reliability(y, p, bins)
    out = {**binary_metrics(y, p), "ece": ece, "reliability": table, "by_multiplier": {}}
    for m in np.unique(mult):
        k = mult == m
        out["by_multiplier"][f"{m:g}x"] = binary_metrics(y[k], p[k])
    return out


# --------------------------------------------------------------------------- main

def main(argv=None):
    skip_tests = argv is not None and "--skip-tests" in argv
    if not skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=REPO_ROOT, capture_output=True, text=True)
        tests_summary = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit("tests failed; experiment not run")
    else:
        tests_summary = "skipped"
    print("tests:", tests_summary, flush=True)

    cfg = json.loads(CONFIG_PATH.read_text())
    threads = min(cfg["lstm"]["max_threads"], os.cpu_count() or 1)
    configure_torch(threads)
    np.random.seed(cfg["seed"])
    hashes = verify_source_r2()
    audit = verify_round2_facts()
    if audit["problems"]:
        raise SystemExit(f"round-two facts failed: {audit['problems']}")
    configs = config_list(cfg)
    bins = cfg["evaluation"]["reliability_bins"]
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- development (rows <= 800 only)
    X, y = load_r2(800)
    feats, n_warm = build_features_r2(X)
    sample = sample_timestamps(feats, cfg["lookback"])
    folds = [fold_ts(sp, sample) for sp in DEV_FOLDS]
    warmup = {"feature_rows_dropped": n_warm, "first_feature_timestamp": int(feats.index[0]),
              "window_context_rows": cfg["lookback"] - 1, "first_sample_timestamp": int(sample[0]),
              "labelled_rows_unused_as_samples": int(sample[0] - 1)}
    manifest_hash = write_manifest(MANIFEST_PATH, manifest_bytes(folds, int(sample[0]), FEATURE_VERSION_R2, warmup))

    fold_info, per_fold, stress_fold, runtimes, diagnostics, seed_metrics = [], {}, {}, {}, {}, {}
    oof, stress_store, model_store = [], {}, {}
    for i, f in enumerate(folds, start=1):
        t_fold = time.perf_counter()
        fd, sim = build_fold(X, y, feats, f, i, cfg)
        name = f["name"]
        fold_info.append({
            "fold": name, "train_range": f["train_range"], "purged": f["purged"], "eval_range": f["eval_range"],
            "n_train": len(fd.y_train), "n_val": len(fd.y_val),
            "train_pos_rate": float(fd.y_train.mean()), "val_pos_rate": float(fd.y_val.mean()),
            "cutoffs": fd.cutoffs.as_dict(),
            "n_train_regime": {g: int(fd.g_train[g].sum()) for g in REGIMES},
            "n_val_regime": {g: int(fd.g_val[g].sum()) for g in REGIMES},
            "n_sim_aug": int(len(sim["aug_y"])), "n_sim_stress": int(len(sim["stress_y"])),
            "sim_aug_pos_rate": float(sim["aug_y"].mean()), "sim_stress_pos_rate": float(sim["stress_y"].mean()),
            "sim_aug_sha256": sim["aug_sha256"], "sim_stress_sha256": sim["stress_sha256"],
            "sim_settings": sim["settings"],
        })
        lr_cache = {}
        per_fold[name], stress_fold[name], runtimes[name], diagnostics[name], seed_metrics[name] = {}, {}, {}, {}, {}
        frame = pd.DataFrame({"timestamp": fd.ts_val, "fold": name, "y": fd.y_val})
        for g in REGIMES:
            frame[g] = fd.g_val[g].to_numpy()
        stress_store[name] = {"y": sim["stress_y"], "mult": sim["stress_mult"]}
        model_store[name] = {}
        for c in configs:
            res = run_config(fd, sim, c, cfg, lr_cache, fd.W_val)
            n = c["name"]
            per_fold[name][n] = evaluate_real(fd.y_val, res["p_eval"], fd.g_val, bins)
            stress_fold[name][n] = evaluate_stress(sim["stress_y"], res["p_stress"], sim["stress_mult"], bins)
            runtimes[name][n] = res["runtime"]
            diagnostics[name][n] = res["diag"]
            frame[n] = res["p_eval"]
            stress_store[name][n] = res["p_stress"].astype(np.float32)
            model_store[name][n] = res["models"]
            if c["model"] == "LSTM":
                seed_metrics[name][n] = {}
                for s, p in res["p_eval_seeds"].items():
                    frame[f"{n}_seed{s}"] = p
                    seed_metrics[name][n][str(s)] = {"log_loss": binary_metrics(fd.y_val, p)["log_loss"],
                                                     "accuracy": binary_metrics(fd.y_val, p)["accuracy"],
                                                     "stress_log_loss": binary_metrics(sim["stress_y"], res["p_stress_seeds"][s])["log_loss"]}
        oof.append(frame)
        print(f"{name}: {time.perf_counter() - t_fold:.1f}s", flush=True)

    oof = pd.concat(oof, ignore_index=True)
    names = [c["name"] for c in configs]
    pooled = {n: evaluate_real(oof["y"].to_numpy(), oof[n].to_numpy(), oof[REGIMES], bins) for n in names}
    st_y = np.concatenate([stress_store[f["name"]]["y"] for f in folds])
    st_m = np.concatenate([stress_store[f["name"]]["mult"] for f in folds])
    pooled_stress = {n: evaluate_stress(st_y, np.concatenate([stress_store[f["name"]][n] for f in folds]).astype(float), st_m, bins)
                     for n in names}
    mean_fold_ll = {n: float(np.mean([per_fold[f["name"]][n]["log_loss"] for f in folds])) for n in names}

    # ---------------- selection (development folds only)
    key = {c["name"]: (round(mean_fold_ll[c["name"]], 4), complexity_key(c)) for c in configs}
    ranking = sorted(names, key=lambda n: key[n])
    winner = ranking[0]
    selection = {"rule": cfg["selection"], "ranking": [{"config": n, "mean_fold_log_loss": mean_fold_ll[n]} for n in ranking],
                 "winner": winner, "stopped": winner == "baseline_prior"}
    print("selection:", winner, flush=True)

    # Local artifacts for development.
    oof_path = LOCAL_DIR / "dev_oof_predictions.csv"
    oof.to_csv(oof_path, index=False, float_format="%.10g", lineterminator="\n")
    stress_path = LOCAL_DIR / "dev_stress_predictions.npz"
    np.savez_compressed(stress_path, **{f"{fold}__{k}": v for fold, d in stress_store.items() for k, v in d.items()})
    model_path = LOCAL_DIR / "dev_models.pkl"
    model_path.write_bytes(pickle.dumps(model_store, protocol=5))
    local = {rel(p): sha256_file(p) for p in (oof_path, stress_path, model_path)}

    reserved = final = None
    if not selection["stopped"]:
        chosen = next(c for c in configs if c["name"] == winner)
        reserved, final, more = run_reserved_and_final(chosen, cfg, bins)
        local.update(more)

    import torch
    meta = {
        "task_id": cfg["task_id"], "command": COMMAND, "tests": tests_summary,
        "dataset_round2": {"path": "KESTREL_EM_D01_round2.csv", **hashes, "committed": False},
        "dataset_round1_sha256_crlf": "f554b3eb5f757c9eff67dcb9112ebf0018cf721ab6972989f529e74930f55531",
        "base_commit": git("merge-base", "HEAD", "origin/main"), "run_head": git("rev-parse", "HEAD"),
        "worktree_dirty_at_run": bool(git("status", "--porcelain", "--untracked-files=no")),
        "feature_version": FEATURE_VERSION_R2, "features": FEATURE_NAMES_R2, "warmup": warmup,
        "split_manifest": {"path": rel(MANIFEST_PATH), "sha256": manifest_hash},
        "config": cfg, "torch_threads": threads, "deterministic_algorithms": True, "device": "cpu",
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "pytest")},
        "torch_build": torch.__version__,
        "local_artifacts_sha256": local,
    }
    results = {"meta": meta, "facts": audit["facts"], "folds": fold_info, "configs": configs,
               "per_fold": per_fold, "pooled": pooled, "stress_per_fold": stress_fold, "stress_pooled": pooled_stress,
               "mean_fold_log_loss": mean_fold_ll, "lstm_seed_metrics": seed_metrics,
               "runtime_seconds": runtimes, "diagnostics": diagnostics, "selection": selection,
               "reserved": reserved, "final": final}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(results, indent=1, default=_json_default) + "\n", newline="\n")

    from src.report_r2 import render_note
    manual = ""
    if NOTE_PATH.exists():
        old = NOTE_PATH.read_text()
        if MANUAL_START in old and MANUAL_END in old:
            manual = old[old.index(MANUAL_START) + len(MANUAL_START):old.index(MANUAL_END)].strip("\n")
    NOTE_PATH.write_text(render_note(results, manual, MANUAL_START, MANUAL_END), newline="\n")
    print(f"wrote {rel(NOTE_PATH)} and {rel(JSON_PATH)}", flush=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run_reserved_and_final(chosen, cfg, bins):
    """Reserved check (train 1-799, purge 800, evaluate 801-850) once, then final fit 1-850 -> 851-900."""
    local = {}
    # Reserved: rows <= 850 loaded; scaler/cutoffs/simulator from rows <= 799.
    X, y = load_r2(850)
    feats, _ = build_features_r2(X)
    f = fold_ts(RESERVED_FOLD, sample_timestamps(feats, cfg["lookback"]))
    fd, sim = build_fold(X, y, feats, f, 8, cfg)
    cache = {}
    if chosen["model"] == "LR" and chosen["arm"] in ("B", "E"):
        warm = {"model": "LR", "arm": "A" if chosen["arm"] == "B" else "D", "param": None, "name": "warm"}
        run_config(fd, sim, warm, cfg, cache, fd.W_val)
    res = run_config(fd, sim, chosen, cfg, cache, fd.W_val)
    reserved = {"config": chosen["name"], "train_range": f["train_range"], "purged": f["purged"],
                "eval_range": f["eval_range"], "n_train": len(fd.y_train), "cutoffs": fd.cutoffs.as_dict(),
                "metrics": evaluate_real(fd.y_val, res["p_eval"], fd.g_val, bins),
                "baseline_metrics": evaluate_real(fd.y_val, np.full(len(fd.y_val), fd.y_train.mean()), fd.g_val, bins),
                "stress": evaluate_stress(sim["stress_y"], res["p_stress"], sim["stress_mult"], bins),
                "runtime": res["runtime"], "diag": res["diag"],
                "rsi_14_mean": float(feats.loc[f["eval_ts"], "rsi_14"].mean())}
    rp = LOCAL_DIR / "reserved_predictions.csv"
    pd.DataFrame({"timestamp": fd.ts_val, "y": fd.y_val, "p_up": res["p_eval"]}).to_csv(rp, index=False, float_format="%.10g", lineterminator="\n")
    local[rel(rp)] = sha256_file(rp)

    # Final: rows <= 900 loaded; everything fitted on rows <= 850; 851-900 features only in prediction.
    X, y = load_r2(900)
    feats, _ = build_features_r2(X)
    f = fold_ts(FINAL_FOLD, sample_timestamps(feats, cfg["lookback"]))
    fd, sim = build_fold(X, y, feats, f, 9, cfg)
    cache = {}
    if chosen["model"] == "LR" and chosen["arm"] in ("B", "E"):
        warm = {"model": "LR", "arm": "A" if chosen["arm"] == "B" else "D", "param": None, "name": "warm"}
        run_config(fd, sim, warm, cfg, cache, fd.W_val)
    res = run_config(fd, sim, chosen, cfg, cache, fd.W_val)
    p = np.clip(res["p_eval"], 0.0, 1.0)
    tr, pr = feats.loc[f["train_ts"]], feats.loc[f["eval_ts"]]
    ood = {}
    for c in FEATURE_NAMES_R2:
        lo, hi = tr[c].min(), tr[c].max()
        pct = [float((tr[c] <= v).mean() * 100) for v in pr[c]]
        ood[c] = {"train_min": float(lo), "train_max": float(hi), "pred_min": float(pr[c].min()), "pred_max": float(pr[c].max()),
                  "share_outside_train_range": float(((pr[c] < lo) | (pr[c] > hi)).mean()),
                  "pred_median_train_percentile": float(np.median(pct))}
    z = fd.X_val
    final = {"config": chosen["name"], "train_range": f["train_range"], "n_train": len(fd.y_train),
             "predict_range": f["eval_range"], "out_of_range": ood, "max_abs_standardized": float(np.abs(z).max()),
             "p_up_summary": {"min": float(p.min()), "max": float(p.max()), "mean": float(p.mean()),
                              "n_class1": int((p >= 0.5).sum())}}
    ts = fd.ts_val
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    cls = LOCAL_DIR / "submission_round2_class.csv"
    score = LOCAL_DIR / "submission_round2_score.csv"
    pd.DataFrame({"timestamp": ts, "prediction": pred, "confidence": np.round(conf, 6)}).to_csv(cls, index=False, lineterminator="\n")
    pd.DataFrame({"timestamp": ts, "prediction": np.round(2 * p - 1, 6)}).to_csv(score, index=False, lineterminator="\n")
    final["validation"] = validate_submissions(cls, score)
    for path in (cls, score):
        local[rel(path)] = sha256_file(path)
    final["submission_sha256"] = {cls.name: sha256_file(cls), score.name: sha256_file(score)}
    mp = LOCAL_DIR / "final_model.pkl"
    mp.write_bytes(pickle.dumps(res["models"], protocol=5))
    local[rel(mp)] = sha256_file(mp)
    return reserved, final, local


def validate_submissions(cls: Path, score: Path) -> dict:
    a, b = cls.read_text().splitlines(), score.read_text().splitlines()
    assert a[0] == "timestamp,prediction,confidence", a[0]
    assert b[0] == "timestamp,prediction", b[0]
    da, db = pd.read_csv(cls), pd.read_csv(score)
    for d in (da, db):
        assert len(d) == 50 and d["timestamp"].tolist() == list(range(851, 901))
        assert not d.isna().any().any()
    assert da["prediction"].dtype.kind == "i" and set(da["prediction"]) <= {0, 1}
    assert ((da["confidence"] >= 0.5) & (da["confidence"] <= 1)).all() and np.isfinite(da["confidence"]).all()
    assert ((db["prediction"] >= -1) & (db["prediction"] <= 1)).all() and np.isfinite(db["prediction"]).all()
    return {"class_file": "ok: header, 50 unique ascending timestamps 851-900, int 0/1, confidence in [0.5, 1], no NaN, no index",
            "score_file": "ok: header, 50 unique ascending timestamps 851-900, prediction in [-1, 1], no NaN, no index"}


if __name__ == "__main__":
    main(sys.argv[1:])
