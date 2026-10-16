"""
experiments/navjyot/run_regime_experiments.py
Executes controlled comparison of:
1. Fold-local Training-Prior Baseline
2. Standard LightGBM
3. Regime-Aware LightGBM (Group DRO)
4. Regime-Aware LightGBM with Monte Carlo Feedback
5. Standard Extra Trees
6. Regime-Aware Extra Trees with Monte Carlo Feedback

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
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.extra_trees import NavjyotExtraTreesModel
from src.models.lightgbm_model import NavjyotLightGBMModel

def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def build_causal_features(df):
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
    
    feats = feats.fillna(0)
    return feats

def assign_regimes(df, vol_thresh, rsi_thresh, flow_thresh):
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
    ds_hash = file_sha256(csv_path)
    df = pd.read_csv(csv_path)
    
    feature_df = build_causal_features(df)
    feature_cols = [c for c in feature_df.columns if c != "timestamp"]
    
    folds = [
        {"fold": 1, "train_end": 449, "purge": 450, "val_start": 451, "val_end": 500},
        {"fold": 2, "train_end": 499, "purge": 500, "val_start": 501, "val_end": 550},
        {"fold": 3, "train_end": 549, "purge": 550, "val_start": 551, "val_end": 600},
        {"fold": 4, "train_end": 599, "purge": 600, "val_start": 601, "val_end": 650},
        {"fold": 5, "train_end": 649, "purge": 650, "val_start": 651, "val_end": 700},
    ]
    
    models_to_test = [
        "Prior_Baseline",
        "LightGBM_Standard",
        "LightGBM_GroupDRO",
        "LightGBM_MC_Feedback",
        "ExtraTrees_Standard",
        "ExtraTrees_MC_Feedback"
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
        
        train_mask = (df["timestamp"] >= 1) & (df["timestamp"] <= t_end)
        val_mask = (df["timestamp"] >= v_start) & (df["timestamp"] <= v_end)
        
        train_df = df[train_mask]
        val_df = df[val_mask]
        
        X_train = feature_df.loc[train_mask, feature_cols].values
        y_train = train_df["target"].values.astype(int)
        
        X_val = feature_df.loc[val_mask, feature_cols].values
        y_val = val_df["target"].values.astype(int)
        
        vol_75 = np.percentile(train_df["volatility_20"], 75)
        flow_75 = np.percentile(np.abs(train_df["order_flow"]), 75)
        
        groups_train = assign_regimes(train_df, vol_75, 70.0, flow_75)
        groups_val = assign_regimes(val_df, vol_75, 70.0, flow_75)
        
        oof_timestamps.extend(val_df["timestamp"].tolist())
        oof_y_true.extend(y_val.tolist())
        oof_regimes.extend(groups_val.tolist())
        
        print(f"\n--- Running Fold {f_num} ---")
        
        # 1. Prior
        t0 = time.time()
        p_val_prior = np.full(len(y_val), np.mean(y_train))
        model_runtimes["Prior_Baseline"] += time.time() - t0
        m = evaluate_predictions(y_val, p_val_prior, groups_val)
        fold_metrics["Prior_Baseline"].append(m)
        oof_predictions["Prior_Baseline"].extend(p_val_prior.tolist())
        
        # 2. LightGBM Standard
        t0 = time.time()
        lgb_std = NavjyotLightGBMModel(mode="standard", random_state=42)
        lgb_std.fit(X_train, y_train)
        p_val_lgb_std = lgb_std.predict_proba(X_val)
        model_runtimes["LightGBM_Standard"] += time.time() - t0
        m = evaluate_predictions(y_val, p_val_lgb_std, groups_val)
        fold_metrics["LightGBM_Standard"].append(m)
        oof_predictions["LightGBM_Standard"].extend(p_val_lgb_std.tolist())
        
        # 3. LightGBM Group DRO
        t0 = time.time()
        lgb_dro = NavjyotLightGBMModel(mode="regime_aware", random_state=42)
        lgb_dro.fit(X_train, y_train, groups=groups_train)
        p_val_lgb_dro = lgb_dro.predict_proba(X_val)
        model_runtimes["LightGBM_GroupDRO"] += time.time() - t0
        m = evaluate_predictions(y_val, p_val_lgb_dro, groups_val)
        fold_metrics["LightGBM_GroupDRO"].append(m)
        oof_predictions["LightGBM_GroupDRO"].extend(p_val_lgb_dro.tolist())
        
        # 4. LightGBM MC Feedback
        t0 = time.time()
        lgb_mc = NavjyotLightGBMModel(mode="regime_aware_mc", random_state=42)
        lgb_mc.fit(X_train, y_train, groups=groups_train)
        p_val_lgb_mc = lgb_mc.predict_proba(X_val)
        model_runtimes["LightGBM_MC_Feedback"] += time.time() - t0
        m = evaluate_predictions(y_val, p_val_lgb_mc, groups_val)
        fold_metrics["LightGBM_MC_Feedback"].append(m)
        oof_predictions["LightGBM_MC_Feedback"].extend(p_val_lgb_mc.tolist())
        
        # 5. Extra Trees Standard
        t0 = time.time()
        et_std = NavjyotExtraTreesModel(mode="standard", random_state=42)
        et_std.fit(X_train, y_train)
        p_val_et_std = et_std.predict_proba(X_val)
        model_runtimes["ExtraTrees_Standard"] += time.time() - t0
        m = evaluate_predictions(y_val, p_val_et_std, groups_val)
        fold_metrics["ExtraTrees_Standard"].append(m)
        oof_predictions["ExtraTrees_Standard"].extend(p_val_et_std.tolist())
        
        # 6. Extra Trees MC Feedback
        t0 = time.time()
        et_mc = NavjyotExtraTreesModel(mode="regime_aware_mc", random_state=42)
        et_mc.fit(X_train, y_train, groups=groups_train)
        p_val_et_mc = et_mc.predict_proba(X_val)
        model_runtimes["ExtraTrees_MC_Feedback"] += time.time() - t0
        m = evaluate_predictions(y_val, p_val_et_mc, groups_val)
        fold_metrics["ExtraTrees_MC_Feedback"].append(m)
        oof_predictions["ExtraTrees_MC_Feedback"].extend(p_val_et_mc.tolist())
        
        print(f"Fold {f_num} Accuracy: LGB_Std={fold_metrics['LightGBM_Standard'][-1]['accuracy']:.4f}, LGB_DRO={fold_metrics['LightGBM_GroupDRO'][-1]['accuracy']:.4f}, LGB_MC={fold_metrics['LightGBM_MC_Feedback'][-1]['accuracy']:.4f}, ET_MC={fold_metrics['ExtraTrees_MC_Feedback'][-1]['accuracy']:.4f}")

    y_true_all = np.array(oof_y_true[:250])
    regimes_all = np.array(oof_regimes[:250])
    summary_report = {}
    
    print("\n================ POOLED OUT-OF-FOLD RESULTS (Timestamps 451-700) ================")
    for m in models_to_test:
        preds = np.array(oof_predictions[m][:250])
        pooled_eval = evaluate_predictions(y_true_all, preds, regimes_all)
        
        cal_train_idx = slice(0, 150)
        cal_test_idx = slice(150, 250)
        lr_cal = LogisticRegression(C=1.0)
        lr_cal.fit(preds[cal_train_idx].reshape(-1, 1), y_true_all[cal_train_idx])
        p_cal_lr = lr_cal.predict_proba(preds[cal_test_idx].reshape(-1, 1))[:, 1]
        
        raw_brier_test = brier_score_loss(y_true_all[cal_test_idx], preds[cal_test_idx])
        cal_brier_test = brier_score_loss(y_true_all[cal_test_idx], p_cal_lr)
        
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
            "raw_calibration_brier_test": float(raw_brier_test),
            "calibrated_brier_test": float(cal_brier_test),
            "total_runtime_seconds": float(model_runtimes[m])
        }
        print(f"Model: {m:25s} | Pooled Acc: {pooled_eval['accuracy']:.4f} | Worst-Regime LL: {pooled_eval.get('worst_regime_ll'):.4f} | Brier: {pooled_eval['brier_score']:.4f} | Time: {model_runtimes[m]:.2f}s")
        
    # Save OOF predictions
    oof_df = pd.DataFrame({
        "timestamp": oof_timestamps[:250],
        "y_true": oof_y_true[:250],
        "regime": oof_regimes[:250],
        "p_prior": oof_predictions["Prior_Baseline"][:250],
        "p_lgb_standard": oof_predictions["LightGBM_Standard"][:250],
        "p_lgb_group_dro": oof_predictions["LightGBM_GroupDRO"][:250],
        "p_lgb_mc_feedback": oof_predictions["LightGBM_MC_Feedback"][:250],
        "p_extra_trees_mc": oof_predictions["ExtraTrees_MC_Feedback"][:250]
    })
    oof_path = "experiments/navjyot/oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    oof_hash = file_sha256(oof_path)
    print(f"\nOOF predictions hash: {oof_hash}")
    
    # Save JSON summary
    summary_path = "experiments/navjyot/regime_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "dataset_sha256": ds_hash,
            "models": summary_report,
            "fold_details": fold_metrics
        }, f, indent=2)
        
    # Generate final test submission on 801-850 using LightGBM MC Feedback
    test_mask = (df["timestamp"] >= 801) & (df["timestamp"] <= 850)
    train_full_mask = (df["timestamp"] >= 1) & (df["timestamp"] <= 800)
    X_train_full = feature_df.loc[train_full_mask, feature_cols].values
    y_train_full = df.loc[train_full_mask, "target"].values.astype(int)
    X_test = feature_df.loc[test_mask, feature_cols].values
    
    vol_75_full = np.percentile(df.loc[train_full_mask, "volatility_20"], 75)
    flow_75_full = np.percentile(np.abs(df.loc[train_full_mask, "order_flow"]), 75)
    groups_train_full = assign_regimes(df.loc[train_full_mask], vol_75_full, 70.0, flow_75_full)
    
    final_model = NavjyotLightGBMModel(mode="regime_aware_mc", random_state=42)
    final_model.fit(X_train_full, y_train_full, groups=groups_train_full)
    p_test = final_model.predict_proba(X_test)
    preds_test = (p_test >= 0.5).astype(int)
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
    print(f"Total time: {time.time() - start_total_time:.2f}s")

if __name__ == "__main__":
    main()
