import numpy as np
import pytest
import torch
from scipy.optimize import check_grad
from sklearn.metrics import log_loss as sk_log_loss

from src.features_r2 import prepare_fold_r2, sample_timestamps
from src.metrics import binary_metrics, log_loss, reliability
from src.models.logistic import (
    eta_for, fit_lam0_direct, fit_worst_regime, mean_log_loss, objective, sklearn_lr,
)
from src.models.lstm_attn import AttnLSTM, configure_torch, fit_lstm, predict_lstm, training_objective
from src.splits_r2 import DEV_FOLDS, fold_ts


@pytest.fixture(scope="module", params=[0, 6])
def fold(request, dev, dev_feats):
    f = fold_ts(DEV_FOLDS[request.param], sample_timestamps(dev_feats))
    return prepare_fold_r2(dev_feats, dev[1], f["train_ts"], f["eval_ts"])


def test_lambda0_objective_matches_sklearn_c1(fold):
    X, y = fold.X_train, fold.y_train
    eta = eta_for(len(y))
    sk = sklearn_lr().set_params(tol=1e-10).fit(X, y)
    ours, res = fit_lam0_direct(X, y, eta)
    assert res.success
    # Coefficients agree to 1e-4 (two features are nearly collinear, so the optimum is
    # flat along one direction); the objective values agree to 1e-10.
    np.testing.assert_allclose(ours.w, sk.coef_.ravel(), atol=1e-4)
    assert abs(ours.b - sk.intercept_[0]) < 1e-4
    G = fold.g_train.to_numpy()
    assert abs(objective(sk.coef_.ravel(), sk.intercept_[0], X, y, G, 0.0, eta) - objective(ours.w, ours.b, X, y, G, 0.0, eta)) < 1e-10


def test_weighted_lambda0_epigraph_matches_sklearn_sample_weight(fold):
    X, y = fold.X_train, fold.y_train
    sw = np.where(fold.g_train["high_vol"].to_numpy(), 2.0, 1.0)
    sk = sklearn_lr().set_params(tol=1e-10).fit(X, y, sample_weight=sw)
    model, diag = fit_worst_regime(X, y, fold.g_train.to_numpy(), 0.0, eta_for(sw.sum()),
                                   np.zeros(X.shape[1]), 0.0, sample_weight=sw)
    assert diag["converged"]
    np.testing.assert_allclose(model.w, sk.coef_.ravel(), atol=1e-4)


def test_arm_a_default_tol_close_to_exact_optimum(fold):
    X, y = fold.X_train, fold.y_train
    eta = eta_for(len(y))
    sk = sklearn_lr().fit(X, y)
    ours, _ = fit_lam0_direct(X, y, eta)
    G = fold.g_train.to_numpy()
    gap = objective(sk.coef_.ravel(), sk.intercept_[0], X, y, G, 0.0, eta) - objective(ours.w, ours.b, X, y, G, 0.0, eta)
    assert 0 <= gap < 1e-6


def test_mean_log_loss_gradient_weighted(fold):
    X, y = fold.X_train, fold.y_train
    d = X.shape[1]
    sw = np.random.default_rng(0).uniform(0.5, 2, len(y))
    th = np.random.default_rng(1).normal(scale=0.3, size=d + 1)
    err = check_grad(lambda t: mean_log_loss(X, y, t[:d], t[d], sw)[0],
                     lambda t: np.append(*mean_log_loss(X, y, t[:d], t[d], sw)[1:]), th)
    assert err < 1e-6


@pytest.mark.parametrize("lam", [0.25, 0.5, 0.75])
def test_worst_regime_feasible_and_locally_optimal(fold, lam):
    X, y, G = fold.X_train, fold.y_train, fold.g_train.to_numpy()
    eta = eta_for(len(y))
    sk = sklearn_lr().fit(X, y)
    model, diag = fit_worst_regime(X, y, G, lam, eta, sk.coef_.ravel(), sk.intercept_[0])
    assert diag["converged"], diag
    j = objective(model.w, model.b, X, y, G, lam, eta)
    rng = np.random.default_rng(1)
    for _ in range(30):
        assert j <= objective(model.w + rng.normal(scale=1e-3, size=X.shape[1]), model.b + rng.normal(scale=1e-3), X, y, G, lam, eta) + 1e-9


def test_lstm_objective_uses_exact_group_max():
    losses = torch.tensor([0.1, 0.9, 0.5, 0.3])
    groups = [torch.tensor([True, True, False, False]), torch.tensor([False, False, True, True])]
    val = training_objective(losses, None, groups, lam=0.5)
    assert torch.isclose(val, 0.5 * losses.mean() + 0.5 * torch.tensor(0.5))
    sw = torch.tensor([1.0, 1.0, 3.0, 3.0])
    assert torch.isclose(training_objective(losses, sw), (sw * losses).sum() / sw.sum())


def test_lstm_deterministic_cpu_and_shapes(fold):
    configure_torch(2)
    W, y = fold.W_train[:120], fold.y_train[:120]
    a, _ = fit_lstm(W, y, seed=42, epochs=5)
    b, _ = fit_lstm(W, y, seed=42, epochs=5)
    pa, pb = predict_lstm(a, fold.W_val), predict_lstm(b, fold.W_val)
    np.testing.assert_array_equal(pa, pb)
    assert pa.shape == (len(fold.W_val),) and ((pa > 0) & (pa < 1)).all()
    assert all(p.device.type == "cpu" for p in a.parameters())
    logit, att = a(torch.as_tensor(fold.W_val[:3], dtype=torch.float32), return_attention=True)
    assert att.shape == (3, 20) and torch.allclose(att.sum(1), torch.ones(3))


def test_lstm_prediction_for_row_t_ignores_later_rows(fold):
    # A window is a self-contained (20, F) block of rows <= t: changing other
    # windows in the batch cannot change row t's prediction in eval mode.
    configure_torch(2)
    m, _ = fit_lstm(fold.W_train[:100], fold.y_train[:100], seed=43, epochs=3)
    base = predict_lstm(m, fold.W_val)
    W2 = fold.W_val.copy()
    W2[10:] = 99.0
    np.testing.assert_array_equal(base[:10], predict_lstm(m, W2)[:10])


def test_metrics_basics():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 200), rng.uniform(0.05, 0.95, 200)
    assert abs(log_loss(y, p) - sk_log_loss(y, p)) < 1e-12
    m = binary_metrics(np.ones(10, int), np.full(10, 0.6))
    assert m["roc_auc"] is None and m["balanced_accuracy"] is None
    table, ece = reliability(y, p, 5)
    assert sum(r["n"] for r in table) == 200 and 0 <= ece <= 1
