"""Verify the stated round-two facts. Audit only: this module reads close and
return_1d to check the target and indicator formulas; nothing here feeds a
model, a feature, a fitted statistic or a choice."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import indicators
from src.data_r2 import CSV_PATH_R2, ROUND1_PATH, check_schema_r2, verify_source_r2


def verify_round2_facts(path=CSV_PATH_R2, round1_path=ROUND1_PATH) -> dict:
    verify_source_r2(path)
    d = pd.read_csv(path)
    check_schema_r2(d)
    facts, problems = {}, []
    facts["shape"] = list(d.shape)
    facts["id_column"] = "stock_id"

    lab = d[d["timestamp"] <= 850]
    counts = lab["target"].value_counts().to_dict()
    facts["target_counts"] = {"ones": int(counts.get(1.0, 0)), "zeros": int(counts.get(0.0, 0))}
    nxt = (d["close"].shift(-1) > d["close"]).astype(float)
    chk = d["timestamp"] <= 849
    facts["target_rule_matches"] = int((nxt[chk] == d.loc[chk, "target"]).sum())
    facts["target_rule_checkable"] = int(chk.sum())
    if facts["target_rule_matches"] != 849 or facts["target_counts"] != {"ones": 440, "zeros": 410}:
        problems.append("target facts")

    if round1_path is not None and round1_path.exists():
        r1 = pd.read_csv(round1_path).rename(columns={"asset_id": "stock_id"})
        a, b = d[d["timestamp"] <= 800].reset_index(drop=True), r1[r1["timestamp"] <= 800].reset_index(drop=True)
        facts["rows_1_800_identical_to_round1"] = bool(a.equals(b))
        a, b = d[d["timestamp"].between(801, 850)].reset_index(drop=True), r1[r1["timestamp"].between(801, 850)].reset_index(drop=True)
        diff = {}
        for c in ["open", "high", "low", "volume", "volatility_20", "rsi_14", "volume_z_20", "order_flow"]:
            diff[c] = int((~np.isclose(a[c].astype(float), b[c].astype(float))).sum())
        facts["rows_801_850_cells_differing_from_round1"] = diff
        if not facts["rows_1_800_identical_to_round1"] or not all(diff[c] > 0 for c in ("volatility_20", "rsi_14", "order_flow")):
            problems.append("round-one comparison")

    v = d["volatility_20"]
    lv, pv = v[d["timestamp"] <= 850], v[d["timestamp"] > 850]
    facts["volatility_20"] = {
        "labelled_median": float(lv.median()), "labelled_max": float(lv.max()),
        "rows_801_850_mean": float(v[d["timestamp"].between(801, 850)].mean()),
        "pred_min": float(pv.min()), "pred_max": float(pv.max()),
        "pred_rows_853_900_min": float(v[d["timestamp"] >= 853].min()),
        "pred_max_over_labelled_median": float(pv.max() / lv.median()),
        "pred_max_over_labelled_max": float(pv.max() / lv.max()),
    }
    facts["rsi_14_rows_801_850_mean"] = float(d.loc[d["timestamp"].between(801, 850), "rsi_14"].mean())
    facts["sma_momentum_visible_in_851_900"] = bool(d.loc[d["timestamp"] > 850, ["sma_10", "sma_30", "momentum_5"]].notna().all().all())
    facts["formula_check"] = indicator_formula_check(lab)
    return {"facts": facts, "problems": problems}


def indicator_formula_check(lab: pd.DataFrame) -> dict:
    """Recompute indicators from labelled close/volume and compare to the file."""
    calc = indicators.compute_all(lab["close"].reset_index(drop=True), lab["volume"].reset_index(drop=True))
    out = {}
    for c in calc.columns:
        ok = calc[c].notna()
        err = (calc.loc[ok, c] - lab[c].reset_index(drop=True)[ok]).abs()
        tol = 0.0051 if c.startswith("sma") else 0.000101
        out[c] = {"rows": int(ok.sum()), "max_abs_err": float(err.max()), "rows_beyond_rounding": int((err > tol).sum()),
                  "rows_exact": int((err < 1e-9).sum())}
    return out
