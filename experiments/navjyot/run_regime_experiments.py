"""
experiments/navjyot/run_regime_experiments.py
Executes controlled comparison of:
1. Fold-local Training-Prior Baseline
2. Standard Extra Trees
3. Regime-Aware Extra Trees (Group DRO)
4. Regime-Aware Extra Trees with Monte Carlo Feedback

On the 5 chronological folds defined in PLAN.md.
"""

import sys
import os
import time
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.extra_trees import NavjyotExtraTreesModel

def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def build_causal_features(df):
    """
    Constructs features using ONLY the 8 continuously available inputs:
    open, high, low, volume, volatility_20, rsi_14, volume_z_20, order_flow.
    No future rows, no close, return_1d, sma_10, sma_30, momentum_5.
    """
    feats = pd.DataFrame(index=df.index)
    feats["timestamp"] = df["timestamp"]
    
    # Intraday normalized geometry
    feats["hl_spread"] = (df["high"] - df["low"]) / df["open"]
    feats["ho_spread"] = (df["high"] - df["open"]) / df["open"]
    feats["ol_spread"] = (df["open"] - df["low"]) / df["open"]
    
    # Trailing returns (open to open)
    feats["ret_open_1"] = df["open"].pct_change(1).fillna(0)
    feats["ret_open_2"] = df["open"].pct_change(2).fillna(0)
    feats["ret_open_5"] = df["open"].pct_change(5).fillna(0)
    
    # Changes in indicators
    feats["vol_pct_change"] = df["volume"].pct_change(1).fillna(0)
    feats["flow_diff_1"] = df["order_flow"].diff(1).fillna(0)
    feats["rsi_diff_1"] = df["rsi_14"].diff(1).fillna(0)
    feats["volat_diff_1"] = df["volatility_20"].diff(1).fillna(0)
    
    # Raw supplied indicators (backward looking)
    feats["volume_z_20"] = df["volume_z_20"]
    feats["volatility_20"] = df["volatility_20"]
    feats["rsi_14"] = df["rsi_14"]
    feats["order_flow"] = df["order_flow"]
    
    # Fill remaining NaNs from diffs
    feats = feats.fillna(0)
    return feats

def assign_regimes(df, vol_thresh, rsi_thresh, flow_thresh):
    """
    Assigns market regimes using fold-local thresholds:
    0: Normal
    1: High_Vol
    2: High_RSI
    3: Stressed_Flow
    """
    regimes = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        vol = df["volatility_20"].iloc[i]
        rsi = df["rsi_14"].iloc[i]
        flow = abs(df["order_flow"].iloc[i])
        
        if vol >= vol_thresh:
            regimes[i] = 1 # High Vol
        elif rsi >= rsi_thresh:
            regimes[i] = 2 # High RSI
        elif flow >= flow_thresh:
            regimes[i] = 3 # Stressed Flow
        else:
            regimes[i] = 0 # Normal
    return regimes

def evaluate_predictions(y_true, p_pred, regimes=None):
    eps = 1e-12
    p_clipped = np.clip(p_pred, eps, 1 - eps)
    y_pred = (p_pred >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, p_pred)
    except:
        auc = np.nan
    ll = log_loss(y_true, p_clipped)
    brier = brier_score_loss(y_true, p_pred)
    
    results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "roc_auc": float(auc) if not np.isnan(auc) else None,
        "log_loss": float(ll),
        "brier_score": float(brier)
    }
    
    # Worst regime metrics
    if regimes is not None:
        regime_accs = {}
        regime_lls = {}
        for r in np.unique(regimes):
            mask = (regimes == r)
            if np.sum(mask) >= 3:
                r_acc = accuracy_score(y_true[mask], y_pred[mask])
                r_ll = log_loss(y_true[mask], p_clipped[mask])
                regime_accs[int(r)] = float(r_acc)
                regime_lls[int(r)] = float(r_ll)
                
        if regime_accs:
            results["worst_regime_acc"] = float(min(regime_accs.values()))
            results["worst_regime_ll"] = float(max(regime_lls.values()))
            results["per_regime_acc"] = regime_accs
            results["per_regime_ll"] = regime_lls
        else:
            results["worst_regime_acc"] = None
            results["worst_regime_ll"] = None
            
    return results

def main():
    start_total_time = time.time()
    csv_path = "KESTREL_EM_D01_round1.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
        
    ds_hash = file_sha256(csv_path)
    print(f"Dataset SHA-256: {ds_hash}")
    
    df = pd.read_csv(csv_path)
    print(f"Data shape: {df.shape}")
    
    # Build causal features
    feature_df = build_causal_features(df)
    feature_cols = [c for c in feature_df.columns if c != "timestamp"]
    print(f"Engineered {len(feature_cols)} causal continuous features: {feature_cols}")
    
    # Define the 5 chronological folds from PLAN.md
    folds = [
        {"fold": 1, "train_end": 449, "purge": 450, "val_start": 451, "val_end": 500},
        {"fold": 2, "train_end": 499, "purge": 500, "val_start": 501, "val_end": 550},
        {"fold": 3, "train_end": 549, "purge": 550, "val_start": 551, "val_end": 600},
        {"fold": 4, "train_end": 599, "purge": 600, "val_start": 601, "val_end": 650},
        {"fold": 5, "train_end": 649, "purge": 650, "val_start": 651, "val_end": 700},
    ]
    
    models_to_test = [
        "Prior_Baseline",
        "Standard_ExtraTrees",
        "Regime_Aware_GroupDRO",
        "Regime_Aware_MC_Feedback"
    ]
    
    fold_metrics = {m: [] for m in models_to_test}
    oof_predictions = {m: [] for m in models_to_test}
    oof_timestamps = []
    oof_y_true = []
    oof_regimes = []
    
    model_runtimes = {m: 0.0 for m in models_to_test}
    
    for f_idx, fold_info in enumerate(folds):
        f_num = fold_info["fold"]
        t_end = fold_info["train_end"]
        v_start = fold_info["val_start"]
        v_end = fold_info["val_end"]
        
        # Subsets
        train_mask = (df["timestamp"] >= 1) & (df["timestamp"] <= t_end)
        val_mask = (df["timestamp"] >= v_start) & (df["timestamp"] <= v_end)
        
        train_df = df[train_mask]
        val_df = df[val_mask]
        
        X_train = feature_df.loc[train_mask, feature_cols].values
        y_train = train_df["target"].values.astype(int)
        
        X_val = feature_df.loc[val_mask, feature_cols].values
        y_val = val_df["target"].values.astype(int)
        
        # Compute fold-local regime thresholds from training history
        vol_75 = np.percentile(train_df["volatility_20"], 75)
        rsi_thresh = 70.0 # Standard overbought threshold
        flow_75 = np.percentile(np.abs(train_df["order_flow"]), 75)
        
        groups_train = assign_regimes(train_df, vol_75, rsi_thresh, flow_75)
        groups_val = assign_regimes(val_df, vol_75, rsi_thresh, flow_75)
        
        if f_idx == 0:
            oof_timestamps.extend(val_df["timestamp"].tolist())
            oof_y_true.extend(y_val.tolist())
            oof_regimes.extend(groups_val.tolist())
        else:
            oof_timestamps.extend(val_df["timestamp"].tolist())
            oof_y_true.extend(y_val.tolist())
            oof_regimes.extend(groups_val.tolist())
            
        print(f"\n--- Running Fold {f_num} (Train 1-{t_end}, Val {v_start}-{v_end}) ---")
        
        # 1. Prior baseline
        t0 = time.time()
        prior_p = np.mean(y_train)
        p_val_prior = np.full(len(y_val), prior_p)
        elapsed_prior = time.time() - t0
        model_runtimes["Prior_Baseline"] += elapsed_prior
        m_prior = evaluate_predictions(y_val, p_val_prior, groups_val)
        m_prior["elapsed"] = elapsed_prior
        fold_metrics["Prior_Baseline"].append(m_prior)
        oof_predictions["Prior_Baseline"].extend(p_val_prior.tolist())
        
        # 2. Standard Extra Trees
        t0 = time.time()
        et_std = NavjyotExtraTreesModel(
            mode="standard",
            n_estimators=300, max_depth=5, min_samples_leaf=20, max_features=1.0,
            bootstrap=False, n_jobs=1, random_state=42
        )
        et_std.fit(X_train, y_train)
        p_val_std = et_std.predict_proba(X_val)
        elapsed_std = time.time() - t0
        model_runtimes["Standard_ExtraTrees"] += elapsed_std
        m_std = evaluate_predictions(y_val, p_val_std, groups_val)
        m_std["elapsed"] = elapsed_std
        fold_metrics["Standard_ExtraTrees"].append(m_std)
        oof_predictions["Standard_ExtraTrees"].extend(p_val_std.tolist())
        
        # 3. Regime-Aware Extra Trees (Group DRO)
        t0 = time.time()
        et_dro = NavjyotExtraTreesModel(
            mode="regime_aware",
            n_estimators=300, max_depth=5, min_samples_leaf=20, max_features=1.0,
            bootstrap=False, n_jobs=1, random_state=42,
            lam=0.5, eta_step=0.1, n_dro_iters=25
        )
        et_dro.fit(X_train, y_train, groups=groups_train)
        p_val_dro = et_dro.predict_proba(X_val)
        elapsed_dro = time.time() - t0
        model_runtimes["Regime_Aware_GroupDRO"] += elapsed_dro
        m_dro = evaluate_predictions(y_val, p_val_dro, groups_val)
        m_dro["elapsed"] = elapsed_dro
        fold_metrics["Regime_Aware_GroupDRO"].append(m_dro)
        oof_predictions["Regime_Aware_GroupDRO"].extend(p_val_dro.tolist())
        
        # 4. Regime-Aware with Monte Carlo Feedback
        t0 = time.time()
        et_mc = NavjyotExtraTreesModel(
            mode="regime_aware_mc",
            n_estimators=300, max_depth=5, min_samples_leaf=20, max_features=1.0,
            bootstrap=False, n_jobs=1, random_state=42,
            lam=0.5, eta_step=0.1, n_dro_iters=25, mc_samples=50
        )
        et_mc.fit(X_train, y_train, groups=groups_train)
        p_val_mc = et_mc.predict_proba(X_val)
        elapsed_mc = time.time() - t0
        model_runtimes["Regime_Aware_MC_Feedback"] += elapsed_mc
        m_mc = evaluate_predictions(y_val, p_val_mc, groups_val)
        m_mc["elapsed"] = elapsed_mc
        fold_metrics["Regime_Aware_MC_Feedback"].append(m_mc)
        oof_predictions["Regime_Aware_MC_Feedback"].extend(p_val_mc.tolist())
        
        print(f"Fold {f_num} Accuracy: Standard={m_std['accuracy']:.4f}, DRO={m_dro['accuracy']:.4f}, MC={m_mc['accuracy']:.4f}, Prior={m_prior['accuracy']:.4f}")
        print(f"Fold {f_num} Log Loss: Standard={m_std['log_loss']:.4f}, DRO={m_dro['log_loss']:.4f}, MC={m_mc['log_loss']:.4f}, Prior={m_prior['log_loss']:.4f}")
        print(f"Fold {f_num} Worst-Regime Acc: Standard={m_std['worst_regime_acc']}, DRO={m_dro['worst_regime_acc']}, MC={m_mc['worst_regime_acc']}")
        
    # Pooled Out-of-Fold Evaluation
    y_true_all = np.array(oof_y_true)
    regimes_all = np.array(oof_regimes)
    
    print("\n================ POOLED OUT-OF-FOLD RESULTS (Timestamps 451-700, N=250) ================")
    summary_report = {}
    
    for m in models_to_test:
        preds = np.array(oof_predictions[m])
        pooled_eval = evaluate_predictions(y_true_all, preds, regimes_all)
        
        # Calibration check: Isotonic & Logistic on pooled OOF
        # Evaluated chronologically: fit on first 3 folds (150 samples), eval on last 2 folds (100 samples)
        cal_train_idx = slice(0, 150)
        cal_test_idx = slice(150, 250)
        
        lr_cal = LogisticRegression(C=1.0)
        lr_cal.fit(preds[cal_train_idx].reshape(-1, 1), y_true_all[cal_train_idx])
        p_cal_lr = lr_cal.predict_proba(preds[cal_test_idx].reshape(-1, 1))[:, 1]
        
        raw_brier_test = brier_score_loss(y_true_all[cal_test_idx], preds[cal_test_idx])
        cal_brier_test = brier_score_loss(y_true_all[cal_test_idx], p_cal_lr)
        
        raw_ll_test = log_loss(y_true_all[cal_test_idx], np.clip(preds[cal_test_idx], 1e-12, 1 - 1e-12))
        cal_ll_test = log_loss(y_true_all[cal_test_idx], np.clip(p_cal_lr, 1e-12, 1 - 1e-12))
        
        mean_fold_acc = np.mean([f["accuracy"] for f in fold_metrics[m]])
        mean_fold_ll = np.mean([f["log_loss"] for f in fold_metrics[m]])
        mean_worst_regime_acc = np.mean([f["worst_regime_acc"] for f in fold_metrics[m] if f["worst_regime_acc"] is not None])
        mean_worst_regime_ll = np.mean([f["worst_regime_ll"] for f in fold_metrics[m] if f["worst_regime_ll"] is not None])
        
        summary_report[m] = {
            "mean_fold_accuracy": float(mean_fold_acc),
            "mean_fold_log_loss": float(mean_fold_ll),
            "mean_worst_regime_accuracy": float(mean_worst_regime_acc),
            "mean_worst_regime_log_loss": float(mean_worst_regime_ll),
            "pooled_accuracy": pooled_eval["accuracy"],
            "pooled_balanced_acc": pooled_eval["balanced_accuracy"],
            "pooled_roc_auc": pooled_eval["roc_auc"],
            "pooled_log_loss": pooled_eval["log_loss"],
            "pooled_brier_score": pooled_eval["brier_score"],
            "pooled_worst_regime_acc": pooled_eval.get("worst_regime_acc"),
            "pooled_worst_regime_ll": pooled_eval.get("worst_regime_ll"),
            "per_regime_acc": pooled_eval.get("per_regime_acc"),
            "per_regime_ll": pooled_eval.get("per_regime_ll"),
            "raw_calibration_brier_test": float(raw_brier_test),
            "calibrated_brier_test": float(cal_brier_test),
            "raw_calibration_ll_test": float(raw_ll_test),
            "calibrated_ll_test": float(cal_ll_test),
            "total_runtime_seconds": float(model_runtimes[m])
        }
        
        print(f"\nModel: {m}")
        print(f"  Mean Fold Accuracy: {mean_fold_acc:.4f} | Pooled Acc: {pooled_eval['accuracy']:.4f}")
        print(f"  Mean Worst-Regime Acc: {mean_worst_regime_acc:.4f} | Pooled Worst-Regime Acc: {pooled_eval.get('worst_regime_acc')}")
        print(f"  Mean Fold Log Loss: {mean_fold_ll:.4f} | Pooled Log Loss: {pooled_eval['log_loss']:.4f}")
        print(f"  Brier Score: {pooled_eval['brier_score']:.4f}")
        print(f"  Calibration effect (Brier on later blocks): raw={raw_brier_test:.4f} -> calibrated={cal_brier_test:.4f}")
        print(f"  Total Runtime: {model_runtimes[m]:.2f}s")
        
    # Save OOF predictions
    oof_df = pd.DataFrame({
        "timestamp": oof_timestamps,
        "y_true": oof_y_true,
        "regime": oof_regimes,
        "p_prior": oof_predictions["Prior_Baseline"],
        "p_standard": oof_predictions["Standard_ExtraTrees"],
        "p_group_dro": oof_predictions["Regime_Aware_GroupDRO"],
        "p_mc_feedback": oof_predictions["Regime_Aware_MC_Feedback"]
    })
    oof_path = "experiments/navjyot/oof_predictions.csv"
    os.makedirs(os.path.dirname(oof_path), exist_ok=True)
    oof_df.to_csv(oof_path, index=False)
    oof_hash = file_sha256(oof_path)
    print(f"\nSaved OOF predictions to {oof_path} (SHA-256: {oof_hash})")
    
    # Save JSON summary
    summary_path = "experiments/navjyot/regime_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "dataset_sha256": ds_hash,
            "models": summary_report,
            "fold_details": fold_metrics
        }, f, indent=2)
        
    # Generate test submission on 801-850
    test_mask = (df["timestamp"] >= 801) & (df["timestamp"] <= 850)
    train_full_mask = (df["timestamp"] >= 1) & (df["timestamp"] <= 800)
    
    X_train_full = feature_df.loc[train_full_mask, feature_cols].values
    y_train_full = df.loc[train_full_mask, "target"].values.astype(int)
    X_test = feature_df.loc[test_mask, feature_cols].values
    
    # Refit Regime-Aware with MC Feedback on full history
    vol_75_full = np.percentile(df.loc[train_full_mask, "volatility_20"], 75)
    flow_75_full = np.percentile(np.abs(df.loc[train_full_mask, "order_flow"]), 75)
    groups_train_full = assign_regimes(df.loc[train_full_mask], vol_75_full, 70.0, flow_75_full)
    
    final_model = NavjyotExtraTreesModel(
        mode="regime_aware_mc",
        n_estimators=300, max_depth=5, min_samples_leaf=20, max_features=1.0,
        bootstrap=False, n_jobs=1, random_state=42,
        lam=0.5, eta_step=0.1, n_dro_iters=25, mc_samples=50
    )
    final_model.fit(X_train_full, y_train_full, groups=groups_train_full)
    p_test = final_model.predict_proba(X_test)
    preds_test = (p_test >= 0.5).astype(int)
    # Confidence: provisionally probability of predicted class
    conf_test = np.where(preds_test == 1, p_test, 1.0 - p_test)
    
    sub_df = pd.DataFrame({
        "timestamp": df.loc[test_mask, "timestamp"].values,
        "prediction": preds_test,
        "confidence": np.round(conf_test, 4)
    })
    sub_path = "experiments/navjyot/test_submission_801_850.csv"
    sub_df.to_csv(sub_path, index=False)
    sub_hash = file_sha256(sub_path)
    print(f"Generated test submission on 801-850: {sub_path} (SHA-256: {sub_hash})")
    
    print(f"\nAll experiments completed in {time.time() - start_total_time:.2f}s")

if __name__ == "__main__":
    main()
