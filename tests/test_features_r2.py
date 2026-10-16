import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.crash_sim import simulate_crashes
from src.data_r2 import ALLOWED_INPUTS_R2, BANNED_INPUTS_R2, CSV_PATH_R2, REPO_ROOT, load_r2
from src.features_r2 import (
    FEATURE_NAMES_R2, LOOKBACK, build_features_r2, prepare_fold_r2, sample_timestamps, windows,
)
from src.splits_r2 import DEV_FOLDS, FINAL_FOLD, RESERVED_FOLD, fold_ts

CUTS = [6, 24, 25, 449, 450, 700, 799, 800, 850, 870]


def _all_windows(feats):
    arr = feats.to_numpy()
    pos = np.arange(LOOKBACK - 1, len(feats))
    return pd.Series(list(windows(arr, pos)), index=feats.index[pos])


def _check_prefix(base, other, t):
    pd.testing.assert_frame_equal(base.loc[base.index <= t], other.loc[other.index <= t])
    wa, wb = _all_windows(base), _all_windows(other)
    for ts in wa.index[wa.index <= t]:
        np.testing.assert_array_equal(wa[ts], wb[ts])


@pytest.mark.parametrize("t", CUTS)
def test_perturbing_later_rows_leaves_earlier_features_and_windows(full, t):
    X, _ = full
    base = build_features_r2(X)[0]
    rng = np.random.default_rng(t)
    changed = X.copy()
    later = changed["timestamp"] > t
    for c in ALLOWED_INPUTS_R2:
        changed.loc[later, c] = rng.uniform(1, 500, later.sum())
    _check_prefix(base, build_features_r2(changed)[0], t)


@pytest.mark.parametrize("t", CUTS)
def test_deleting_later_rows_leaves_earlier_features_and_windows(full, t):
    X, _ = full
    base = build_features_r2(X)[0]
    _check_prefix(base, build_features_r2(X.loc[X["timestamp"] <= t])[0], t)


def test_feature_names_exclude_banned_and_raw_levels(dev_feats):
    assert list(dev_feats.columns) == FEATURE_NAMES_R2
    assert not set(dev_feats.columns) & set(BANNED_INPUTS_R2)
    assert not {"open", "high", "low", "volume", "sma_10", "sma_30"} & set(dev_feats.columns)


def test_close_and_return_1d_have_no_influence():
    raw = pd.read_csv(CSV_PATH_R2)
    rng = np.random.default_rng(0)
    scrambled = raw.copy()
    for c in ["close", "return_1d", "target"]:
        scrambled[c] = rng.normal(size=len(raw))
    scrambled["date"], scrambled["stock_id"] = "1999-01-01", "OTHER"
    pd.testing.assert_frame_equal(build_features_r2(raw)[0], build_features_r2(scrambled)[0])


MODELLING_MODULES = ["src/features_r2.py", "src/crash_sim.py", "src/splits_r2.py",
                     "src/models/logistic.py", "src/models/lstm_attn.py", "src/run_round2.py"]


@pytest.mark.parametrize("module", MODELLING_MODULES)
def test_modelling_code_never_names_close_or_return_1d(module):
    path = REPO_ROOT / module
    if not path.exists():
        pytest.skip(f"{module} not written yet")
    strings = {n.value for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "close" not in strings and "return_1d" not in strings


def test_warmup_and_sample_start(dev_feats):
    assert dev_feats.index[0] == 6
    assert sample_timestamps(dev_feats)[0] == 25


@pytest.mark.parametrize("spec", DEV_FOLDS + [RESERVED_FOLD])
def test_fold_boundaries(dev_feats, spec):
    f = fold_ts(spec, list(range(25, 851)))
    tr, ev = set(f["train_ts"]), set(f["eval_ts"])
    assert max(tr) < f["purged"] < min(ev)
    assert not tr & ev and f["purged"] not in tr | ev
    assert len(ev) == 50 and max(ev) - min(ev) == 49
    if spec[0].startswith("dev"):
        assert max(ev) <= 800 and max(tr) <= 749


def test_dev_folds_never_touch_801_onwards(dev_feats):
    s = sample_timestamps(dev_feats)
    assert s.max() <= 800
    for spec in DEV_FOLDS:
        f = fold_ts(spec, s)
        assert max(f["train_ts"] + f["eval_ts"]) <= 800


def test_final_fold_trains_through_850_and_predicts_851_900(full):
    feats = build_features_r2(full[0])[0]
    f = fold_ts(FINAL_FOLD, sample_timestamps(feats))
    assert max(f["train_ts"]) == 850 and f["eval_ts"] == list(range(851, 901))


def _fold_artifacts(X, y, spec):
    feats = build_features_r2(X)[0]
    f = fold_ts(spec, sample_timestamps(feats))
    fd = prepare_fold_r2(feats, y, f["train_ts"], f["eval_ts"])
    med = float(feats.loc[f["train_ts"], "volatility_20"].median())
    sim = simulate_crashes(X.loc[X["timestamp"] <= spec[1]], med, fd.cutoffs.vol_high, [42, 1, 0])
    return fd, sim


@pytest.mark.parametrize("spec", [DEV_FOLDS[0], DEV_FOLDS[6], RESERVED_FOLD])
def test_scaler_cutoffs_and_simulator_ignore_non_training_rows(full, spec):
    X, y = full
    base_fd, base_sim = _fold_artifacts(X, y, spec)
    # Perturb every row after the training block (validation, reserved, prediction rows).
    rng = np.random.default_rng(1)
    pert = X.copy()
    later = pert["timestamp"] > spec[1]
    for c in ALLOWED_INPUTS_R2:
        pert.loc[later, c] = pert.loc[later, c] * rng.uniform(0.5, 3, later.sum())
    fd, sim = _fold_artifacts(pert, y, spec)
    np.testing.assert_array_equal(base_fd.scaler.mean_, fd.scaler.mean_)
    np.testing.assert_array_equal(base_fd.scaler.scale_, fd.scaler.scale_)
    assert base_fd.cutoffs == fd.cutoffs
    assert base_sim.sha256 == sim.sha256
    # Training rows alone give the same objects.
    alone_fd, alone_sim = _fold_artifacts(X.loc[X["timestamp"] <= spec[1] + 1], y, (spec[0], spec[1], None, spec[1] + 1, spec[1] + 1))
    np.testing.assert_array_equal(base_fd.scaler.mean_, alone_fd.scaler.mean_)
    assert base_fd.cutoffs == alone_fd.cutoffs
    assert base_sim.sha256 == alone_sim.sha256


def test_simulator_is_seeded_and_shaped(dev):
    X, y = dev
    _, a = _fold_artifacts(X, y, DEV_FOLDS[0])
    _, b = _fold_artifacts(X, y, DEV_FOLDS[0])
    assert a.sha256 == b.sha256
    assert a.W.shape == (200 * 50, LOOKBACK, len(FEATURE_NAMES_R2))
    assert set(np.unique(a.y)) == {0, 1}
    assert np.isfinite(a.W).all()


def test_simulator_indicators_match_dataset_formulas_on_real_prices():
    # The simulator recomputes indicators with src.indicators; on real labelled
    # closes those functions reproduce the dataset's columns (audit-only read).
    from src import indicators
    raw = pd.read_csv(CSV_PATH_R2)
    lab = raw.loc[raw["timestamp"] <= 850].reset_index(drop=True)
    calc = indicators.compute_all(lab["close"], lab["volume"])
    for c, tol in [("sma_10", 0.0051), ("sma_30", 0.0051), ("momentum_5", 0.000101),
                   ("volatility_20", 0.000101), ("rsi_14", 0.000101), ("volume_z_20", 0.000101)]:
        ok = calc[c].notna()
        assert (calc.loc[ok, c] - lab.loc[ok, c]).abs().max() <= tol, c
