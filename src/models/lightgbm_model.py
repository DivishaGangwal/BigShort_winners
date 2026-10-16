"""
src/models/lightgbm_model.py
Navjyot's LightGBM implementation supporting:
1. Standard Training
2. Regime-Aware Training (Group Distributionally Robust Optimization - Group DRO)
3. Regime-Aware Training with Monte Carlo Feedback

Adheres to project constraints:
- Reproducible random_state
- Single-thread n_jobs=1
- Standard predict_proba returning P(class 1)
"""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss

class NavjyotLightGBMModel:
    def __init__(
        self,
        mode="standard",  # "standard", "regime_aware", "regime_aware_mc"
        n_estimators=150,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        learning_rate=0.05,
        n_jobs=1,
        random_state=42,
        lam=0.5,         # lambda weight on worst regime
        eta_step=0.1,    # step size for group weight updates
        n_dro_iters=20,  # iterations for DRO sample weight optimization
        mc_samples=50    # Monte Carlo simulation draws
    ):
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.lam = lam
        self.eta_step = eta_step
        self.n_dro_iters = n_dro_iters
        self.mc_samples = mc_samples
        
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=-1
        )
        self.group_weights_ = None
        self.sample_weights_ = None

    def _compute_group_dro_weights(self, X, y, groups):
        N = len(y)
        unique_groups = np.unique(groups)
        K = len(unique_groups)
        group_masks = {g: (groups == g) for g in unique_groups}
        
        q = np.ones(K) / K
        surrogate = lgb.LGBMClassifier(
            n_estimators=40,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=-1
        )
        w = np.ones(N) / N
        
        for iteration in range(self.n_dro_iters):
            surrogate.fit(X, y, sample_weight=w)
            probas = surrogate.predict_proba(X)
            eps = 1e-12
            p1 = np.clip(probas[:, 1] if probas.shape[1] == 2 else probas[:, 0], eps, 1 - eps)
            losses = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
            
            group_losses = np.zeros(K)
            for idx, g in enumerate(unique_groups):
                mask = group_masks[g]
                group_losses[idx] = np.mean(losses[mask]) if np.sum(mask) > 0 else 0.0
                
            q = q * np.exp(self.eta_step * (group_losses - np.mean(group_losses)))
            q = q / np.sum(q)
            
            w_robust = np.zeros(N)
            for idx, g in enumerate(unique_groups):
                mask = group_masks[g]
                n_g = np.sum(mask)
                if n_g > 0:
                    w_robust[mask] = q[idx] / n_g
            w_robust = w_robust / np.sum(w_robust)
            w = (1.0 - self.lam) * (np.ones(N) / N) + self.lam * w_robust
            w = w / np.sum(w)
            
        self.group_weights_ = dict(zip(unique_groups, q))
        return w

    def _compute_mc_feedback_weights(self, X, y, groups):
        base_w = self._compute_group_dro_weights(X, y, groups)
        N = len(y)
        unique_groups = np.unique(groups)
        K = len(unique_groups)
        group_masks = {g: (groups == g) for g in unique_groups}
        
        rng = np.random.RandomState(self.random_state + 200)
        mc_accum_w = np.zeros(N)
        for _ in range(self.mc_samples):
            base_q = np.array([self.group_weights_.get(g, 1.0/K) for g in unique_groups])
            dirichlet_alpha = np.maximum(base_q * 20.0, 1.0)
            stochastic_q = rng.dirichlet(dirichlet_alpha)
            
            draw_w = np.zeros(N)
            for idx, g in enumerate(unique_groups):
                mask = group_masks[g]
                n_g = np.sum(mask)
                if n_g > 0:
                    draw_w[mask] = stochastic_q[idx] / n_g
            draw_w = draw_w / np.sum(draw_w)
            mc_accum_w += draw_w
            
        mc_w = mc_accum_w / self.mc_samples
        final_w = (1.0 - self.lam) * base_w + self.lam * mc_w
        final_w = final_w / np.sum(final_w)
        return final_w

    def fit(self, X, y, groups=None):
        X = np.asarray(X)
        y = np.asarray(y)
        N = len(y)
        
        if self.mode == "standard" or groups is None:
            w = np.ones(N) / N
        elif self.mode == "regime_aware":
            w = self._compute_group_dro_weights(X, y, groups)
        elif self.mode == "regime_aware_mc":
            w = self._compute_mc_feedback_weights(X, y, groups)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
            
        self.sample_weights_ = w
        self.model.fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        probas = self.model.predict_proba(X)
        if probas.shape[1] == 2:
            return probas[:, 1]
        elif probas.shape[1] == 1:
            classes = list(self.model.classes_)
            return np.ones(len(X)) if classes[0] == 1 else np.zeros(len(X))
        return probas

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)
