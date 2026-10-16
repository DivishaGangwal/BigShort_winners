"""LSTM with additive temporal attention (PyTorch, CPU only).

Window (n, 20, F) -> 1-layer LSTM (hidden 16) -> dropout -> additive attention
over the 20 steps -> linear head -> logit for p_up. Full-batch AdamW for a
fixed number of epochs; no early stopping and no validation data.

Training objective per arm (per-row BCE losses l_i):
    standard / weighted:  sum_i s_i l_i / sum_i s_i    (s_i = 1 for the standard arm)
    worst group:          (1 - lam) * overall + lam * max_g mean_{i in g} l_i
The max is exact over the full batch at every step; Omega is AdamW weight decay.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cpu")


class AttnLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 16, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.att_w = nn.Linear(hidden, hidden)
        self.att_v = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, return_attention: bool = False):
        h, _ = self.lstm(x)
        h = self.drop(h)
        a = torch.softmax(self.att_v(torch.tanh(self.att_w(h))).squeeze(-1), dim=1)
        ctx = (a.unsqueeze(-1) * h).sum(dim=1)
        logit = self.head(ctx).squeeze(-1)
        return (logit, a) if return_attention else logit


def configure_torch(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def training_objective(losses, sample_weight=None, groups=None, lam: float = 0.0):
    if sample_weight is None:
        overall = losses.mean()
    else:
        overall = (sample_weight * losses).sum() / sample_weight.sum()
    if not lam:
        return overall
    group_losses = torch.stack([losses[m].mean() for m in groups])
    return (1 - lam) * overall + lam * group_losses.max()


def fit_lstm(W, y, seed: int, sample_weight=None, groups=None, lam: float = 0.0, epochs: int = 150,
             lr: float = 1e-3, weight_decay: float = 1e-3, hidden: int = 16, dropout: float = 0.2):
    """groups: (n, G) bool array; empty groups are dropped."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.as_tensor(W, dtype=torch.float32, device=DEVICE)
    t = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    sw = None if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32, device=DEVICE)
    masks = None
    if lam:
        masks = [torch.as_tensor(m.copy(), device=DEVICE) for m in np.asarray(groups, dtype=bool).T if m.any()]
    model = AttnLSTM(X.shape[-1], hidden, dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    history = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = training_objective(bce(model(X), t), sw, masks, lam)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    model.eval()
    return model, history


@torch.no_grad()
def predict_lstm(model, W) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.as_tensor(W, dtype=torch.float32, device=DEVICE))).numpy().astype(float)
