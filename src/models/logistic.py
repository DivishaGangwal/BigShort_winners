"""Regularized logistic regression arms and the training-prior baseline.

Objective (mean log loss per group, intercept unpenalized):
    J(theta) = (1 - lam) * L_overall + lam * max_g L_g + eta * 0.5 * ||w||^2
With eta = 1 / (n_train * C), lam = 0 is sklearn's L2 LogisticRegression
objective  0.5 * ||w||^2 + C * sum_i loss_i  divided by n_train * C.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

C = 1.0
MAX_ITER = 2000
SEED = 42


def sklearn_lr() -> LogisticRegression:
    return LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER, random_state=SEED)


def eta_for(n_train: float, c: float = C) -> float:
    """n_train is the row count, or the sum of sample weights for a weighted fit."""
    return 1.0 / (n_train * c)


class LinearProbModel:
    """p_up = sigmoid(X @ w + b)."""

    def __init__(self, w, b):
        self.w, self.b = np.asarray(w, dtype=float), float(b)

    def predict_proba1(self, X):
        return expit(X @ self.w + self.b)


class PriorModel:
    """Predicts the training class-1 frequency for every row."""

    def __init__(self, y_train):
        self.p = float(np.mean(y_train))

    def predict_proba1(self, X):
        return np.full(len(X), self.p)


def mean_log_loss(X, y, w, b, sample_weight=None):
    """(Weighted) mean log loss and its gradient in (w, b)."""
    z = X @ w + b
    sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    sw = sw / sw.sum()
    loss = float(np.sum(sw * (np.logaddexp(0.0, z) - y * z)))
    r = sw * (expit(z) - y)
    return loss, X.T @ r, float(r.sum())


def objective(w, b, X, y, groups, lam, eta, sample_weight=None) -> float:
    """Exact J(theta) with the true max over groups (groups: (n, G) bool array).

    The overall term is the (weighted) mean loss; group terms are unweighted means.
    """
    overall = mean_log_loss(X, y, w, b, sample_weight)[0]
    worst = max(mean_log_loss(X[m], y[m], w, b)[0] for m in groups.T) if lam else 0.0
    return (1 - lam) * overall + lam * worst + eta * 0.5 * float(w @ w)


def fit_lam0_direct(X, y, eta, tol=1e-12):
    """Minimise the lam = 0 objective with L-BFGS from zero (used to prove the sklearn match)."""
    d = X.shape[1]

    def f(th):
        loss, gw, gb = mean_log_loss(X, y, th[:d], th[d])
        return loss + eta * 0.5 * th[:d] @ th[:d], np.append(gw + eta * th[:d], gb)

    res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"gtol": tol, "ftol": tol, "maxiter": 10000})
    return LinearProbModel(res.x[:d], res.x[d]), res


def fit_worst_regime(X, y, groups, lam, eta, w0, b0, maxiter=1000, ftol=1e-12, feas_tol=1e-6, sample_weight=None):
    """Exact max via the epigraph form, solved with SLSQP and analytic gradients.

        min_{w,b,t}  (1 - lam) * L_overall + lam * t + eta * 0.5 * ||w||^2
        s.t.         L_g(w, b) <= t  for every group g with training rows.

    Warm start: (w0, b0) from the standard arm, t0 = max_g L_g(w0, b0) (feasible).
    With sample_weight, L_overall is the weighted mean (use eta = 1 / (C * sum(weights))
    so that lam = 0 is sklearn's weighted C objective); L_g stay unweighted means.
    """
    d = X.shape[1]
    masks = [m for m in np.asarray(groups, dtype=bool).T if m.any()]
    Xg, yg = [X[m] for m in masks], [y[m] for m in masks]

    def f(th):
        w, b, t = th[:d], th[d], th[d + 1]
        loss, gw, gb = mean_log_loss(X, y, w, b, sample_weight)
        val = (1 - lam) * loss + lam * t + eta * 0.5 * w @ w
        return val, np.concatenate([(1 - lam) * gw + eta * w, [(1 - lam) * gb, lam]])

    def cons(th):
        return np.array([th[d + 1] - mean_log_loss(a, c, th[:d], th[d])[0] for a, c in zip(Xg, yg)])

    def cons_jac(th):
        rows = []
        for a, c in zip(Xg, yg):
            _, gw, gb = mean_log_loss(a, c, th[:d], th[d])
            rows.append(np.concatenate([-gw, [-gb, 1.0]]))
        return np.vstack(rows)

    w0 = np.asarray(w0, dtype=float)
    t0 = max(mean_log_loss(a, c, w0, b0)[0] for a, c in zip(Xg, yg))
    x0 = np.concatenate([w0, [b0, t0]])
    res = minimize(f, x0, jac=True, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": cons, "jac": cons_jac}],
                   options={"maxiter": maxiter, "ftol": ftol})
    w, b, t = res.x[:d], res.x[d], res.x[d + 1]
    group_losses = -cons(res.x) + t
    violation = float(max(0.0, np.max(group_losses - t)))
    j_start = objective(w0, b0, X, y, groups, lam, eta, sample_weight)
    j_end = objective(w, b, X, y, groups, lam, eta, sample_weight)
    diag = {
        "success": bool(res.success), "status": int(res.status), "message": str(res.message),
        "nit": int(res.nit), "max_violation": violation,
        "objective_warm_start": float(j_start), "objective_final": float(j_end),
        "t": float(t), "max_group_loss": float(group_losses.max()),
        "active_groups": int(np.sum(group_losses >= t - 1e-6)),
        "converged": bool(res.success and violation <= feas_tol and j_end <= j_start + 1e-12),
    }
    return LinearProbModel(w, b), diag
