---
marp: true
title: KESTREL — Crash-Aware Regime Modelling and Trading Architecture
paginate: true
---

# KESTREL: Crash-Aware Regime Modelling
### From leak-free classification to a crash-robust trading book

Team BigShort Winners · Rounds 1–3

<!-- Speaker notes: One asset, three rounds. Round 1 predicts next-day direction, round 2 adds a crash we never saw in training, round 3 turns predictions into a scored trading book. The story: get the data contract right, design for the crash with Monte Carlo and robust losses, then convert to positions that survive it. -->

---

## The challenge in three rounds

| Round | Task | What made it hard |
|---|---|---|
| 1 | Next-day up/down (850 rows) | Five columns hidden in the prediction block; tiny signal |
| 2 | Same task, then predict a crash | Crash (vol up to **6.4×** the labelled median) appears **only** in rows 851–900; no labelled crash exists |
| 3 | Daily signal in [−1, 1] for 851–900 | Scored as a portfolio: Sharpe, money made, max drawdown, costs |

**Core problem:** train on calm history, perform in a crash the model has never seen.

<!-- Speaker notes: Labelled volatility_20 never exceeds 0.0204; the scored block runs 0.040–0.074 from row 853. Every architectural choice that follows answers "what happens to this model when volatility is 5–6x anything in training?" -->

---

## Principle 1: a leak-free data contract

- **Hash-verified data:** SHA-256 checked on every run; schema 900 × 17 enforced.
- **Causal features only:** row t uses rows ≤ t. No centred windows, no `shift(-k)`, no backfill.
- **Banned inputs:** target, timestamp, date and id are never features. close/return_1d were banned in rounds 1–2, where they were hidden.
- **Round 3:** today's close is rebuilt from **today's row only**: close[t] = close[t−5]·(1 + momentum_5[t]).
  - Validation: hide 801–850, rebuild them, max error 0.013%.
  - Tomorrow's close is **never** reconstructed, even though later rows would allow it.
- **Facts verified in code:** target = next-close direction on 849/849 rows; dataset indicator formulas reproduced exactly (SMA, RSI-14 with simple means, vol = 20-day sample std).

**Why:** a model that leaks looks brilliant locally and fails on the leaderboard. Every number we report is reproducible and honest.

<!-- Speaker notes: Leakage tests prove that perturbing or deleting any future row leaves every feature, sequence window, scaler, regime cutoff and simulated crash unchanged. 76 automated tests run before every experiment; if one fails, the experiment refuses to start. -->

---

## Principle 2: validation that respects time

```
Train 25 ─────────────── 449 │purge 450│ Validate 451–500
Train 25 ─────────────────────── 499 │500│ 501–550
   …  seven expanding folds  …
Train 25 ─────────────────────────────────── 749 │750│ 751–800
Reserved: train ≤799 │800│ 801–850 (evaluated once)
Final:    train ≤850 → predict 851–900
```

- **One-row purge** before each block: the label at t uses close[t+1].
- **Fold-local fitting:** scaler, regime cutoffs **and the crash simulator** are all fitted on training rows only.
- **Selection rules fixed before results:** e.g. "lowest mean-fold log loss; stop if the prior baseline wins".

**Why:** with 50-row blocks, choosing after looking is overfitting. Fixing rules first keeps our evidence honest.

<!-- Speaker notes: In round 2 the training-prior baseline won the prespecified rule, so we stopped before the reserved evaluation rather than force a model through. That discipline is what made the later portfolio results trustworthy. -->

---

## Features: causal, scale-free, crash-aware

| Group | Features | Why |
|---|---|---|
| Price geometry | (high−low)/open, upper and lower wick | Scale-free shape of the day |
| Trailing returns | open-to-open 1, 3, 5 days | Momentum without levels |
| Trend | open/SMA10, open/SMA30, SMA10/SMA30, momentum_5 | Regime and trend state |
| Supplied state | volatility_20, RSI-14, volume z-score, order flow + lags | Market microstructure |
| Close-based (round 3) | return, intraday, gap, close/SMA10, close location, return/vol | Same-row information |
| **Vol-normalised** | each return ÷ vol, momentum ÷ (vol·√5), range ÷ vol | **Comparable in calm and crash** |

- Standardise on training rows only, then **clip to ±5**: no input can push a model into wild extrapolation.

**Why:** in the crash, raw returns are 5× larger. Dividing by volatility keeps inputs in the range the models were trained on.

<!-- Speaker notes: Without normalisation the standardised features reached |z| = 24 in 851–900. Linear models extrapolate linearly, so a z of 24 produces garbage probabilities. Normalising and clipping is the first layer of crash awareness. -->

---

## Regimes: the groups our losses protect

Four overlapping market regimes, with cutoffs learned **per fold on training rows only**:

| Regime | Definition |
|---|---|
| high_vol | volatility_20 ≥ 75th percentile |
| low_vol | volatility_20 ≤ 25th percentile |
| high_rsi | RSI-14 ≥ 75th percentile |
| low_rsi | RSI-14 ≤ 25th percentile |
| **sim_crash** | every Monte Carlo crash row (rounds 2–3) |

**Why:** an average loss can hide a model that is terrible in stress. Regimes let us measure and **optimise** the worst case. RSI regimes matter because 801–850 was a high-RSI rally (mean RSI 78).

<!-- Speaker notes: Rows can belong to several regimes. Cutoffs are applied unchanged to validation rows, so validation regimes never influence training. -->

---

## Loss regimes: five training arms

| Arm | Objective | Idea |
|---|---|---|
| **A** Standard | mean log loss | Reference |
| **B** Worst-regime | (1−λ)·L_overall + λ·**max_g L_g** + η·‖w‖², λ ∈ {0.25, 0.5, 0.75} | Distributionally robust: never let one regime be terrible |
| **C** Regime-weighted | weight w ∈ {1.5, 2, 3} on high_vol ∪ high_rsi rows | Emphasise stressed rows |
| **D** Monte Carlo augmentation | real rows + crash rows at 25% of real weight | Teach the model what a crash looks like |
| **E** MC + worst-regime | arm D data, arm B objective at λ = 0.5, sim_crash as a group | Robust to crashes **and** to every real regime |

<!-- Speaker notes: B is a Group-DRO style objective. C is the cheap heuristic version. D is data-side robustness. E combines data and objective. All arms use identical folds, features and metrics, so differences are attributable to the loss alone. -->

---

## The worst-regime loss, solved exactly

**Logistic regression: the epigraph form**

min over (w, b, t) of (1−λ)·L_overall(w, b) + λ·t + η·½‖w‖²   subject to   L_g(w, b) ≤ t for every regime g

- Solved with SLSQP using **analytic gradients**, warm-started from arm A.
- The **exact max**, not a soft approximation: all 15 round-2 fits converged, constraint violation ≤ 1e-12.
- η = 1/(n·C), so λ = 0 equals sklearn `LogisticRegression(C=1)` exactly (unit-tested to 1e-10).

**LSTM:** the exact max over full-batch group mean losses at every step; weight decay plays the role of η.

**Why exact:** smooth approximations (softmax over regimes) blur which regime is actually worst. We wanted the true minimax.

<!-- Speaker notes: Converged solutions lowered the exact training objective below the warm start in every fold, with 2–4 regimes active at the optimum, so the robust losses were genuinely optimised, not approximations. -->

---

## Monte Carlo crash simulator

**Why:** the scored period is a crash with **no labelled precedent**. We manufacture realistic crashes from each fold's training data.

Per fold, from **training rows only** (seeded and hashed):

1. **Start state:** a random training day; a 60-day burn-in on its real price path.
2. **Returns:** block bootstrap (5-day blocks) of training returns, rescaled to **m × median vol, m ∈ {3, 5}**.
3. **Shocks:** 1–2 single-day drops of **−8% to −15%** injected at random positions.
4. **Bars:** open, high and low rebuilt from the path; range scale calibrated on training data.
5. **Indicators recomputed** with the verified dataset formulas (SMA, momentum, vol, RSI).
6. **Flow:** volume, volume z-score and order flow resampled jointly from high-vol training rows.

→ 200 windows × 50 days = **10,000 crash rows per fold**, labelled by the next simulated move.

<!-- Speaker notes: Because the simulator uses only training rows, a test proves it is byte-identical when validation, reserved and prediction rows are perturbed. The indicator formulas were first verified against the real dataset so simulated crashes look like the organizers' data. -->

---

## Two uses of Monte Carlo: train and stress-test

| Set | Seed | Used for |
|---|---|---|
| **Augmentation** | [42, fold, 0] | Training rows for arms D and E, and every round-3 model |
| **Stress test** | [42, fold, 1] | Independent, **never trained on**; reported separately |

- Real-data metrics and synthetic metrics are **never mixed**.
- Simulated labels are synthetic: stress scores measure **robustness to crash-like inputs**, not skill.
- In round 3, MC paths drive the **portfolio backtest**: the Sharpe and drawdown distribution over 1,600 crash paths.

**Why separate seeds:** testing on the same simulations we trained on would reward memorising the simulator.

<!-- Speaker notes: This separation is what let us say MC augmentation genuinely improved crash robustness rather than just fitting the generator. -->

---

## Models: three families, all crash-aware

| Model | Configuration | Crash-awareness |
|---|---|---|
| **Logistic regression** | C = 1, exact robust epigraph | Clipped vol-normalised inputs; MC rows; no runaway extrapolation |
| **Gradient boosting** (HGB) | 150 trees, 7 leaves, min leaf 30, L2 = 1 | Trees cannot extrapolate beyond leaf values; MC rows |
| **Attention LSTM** | 20-day lookback, hidden 16, additive attention, dropout 0.2, AdamW, 150 full-batch epochs, 3 seeds | Worst-group loss with **sim_crash** as a group; MC windows; seed-averaged |

- **Hybrid** = average of p_up across the models.
- **Checked in the crash:** predicted probabilities stay in 0.34–0.69 (LSTM 0.45–0.56); **no garbage outputs at 6× vol**.
- The models are only weakly correlated (LR–LSTM −0.11), so averaging reduces noise.

<!-- Speaker notes: CPU-only PyTorch with deterministic algorithms; reruns reproduce the submission byte for byte. -->

---

## Round 2 evidence: what MC and robust losses bought us

| Configuration | Real log loss (mean over folds) | Synthetic crash log loss |
|---|---|---|
| Prior baseline | **0.6943** | 0.694 |
| LSTM E (MC + worst-regime) | 0.6994 | **0.691** |
| LR E (MC + worst-regime) | 0.6998 | — |
| LR D (MC augmentation) | 0.7087 | 0.704 |
| LR A (standard) | 0.7136 | **1.221** |

- **Honest result:** no model beat the prior on real direction (accuracy 49–52%).
- **Key win:** MC augmentation cut crash-case loss for LR from **1.22 to 0.70**. Standard models blow up on crash inputs; MC-trained ones do not.
- The best real-data models were both MC + worst-regime (arm E).

**Takeaway:** robustness, not prediction, is the controllable edge. That shaped round 3.

<!-- Speaker notes: The prespecified rule picked the prior, so we stopped instead of forcing a submission. That integrity carried into round 3, where the same MC + worst-group LSTM became our winning component. -->

---

## Round 3: from probabilities to a portfolio

**Decision layer:** p_up plus volatility gives a position in [−1, 1].

| Policy | Rule |
|---|---|
| Vol-scaled long | position = min(1, median vol / current vol) |
| Model, vol-scaled | clip(5·(2p−1)) × vol scale |
| **Regime switch** (chosen) | Normal vol: blend of long and model. **Vol above anything in training: model only, vol-scaled** |
| Also tested | full long, banded model, trend, flat |

**Backtest engine:** pnl_t = s_t·r(t→t+1) − cost·|Δs|; Sharpe, return, max drawdown, turnover; 0 / 10 / 25 bps.
- 8 real walk-forward blocks (451–849) and 1,600 independent MC crash paths.
- **Fixed rule:** maximise ½·real Sharpe + ½·MC Sharpe, subject to drawdown caps (15% real, 25% MC).

<!-- Speaker notes: Vol scaling means each day carries comparable risk; in the crash, positions shrink automatically by about 3–6x. The regime switch trusts the crash-trained ensemble exactly when the market leaves the training distribution. -->

---

## Results

| Submission | Design | Leaderboard |
|---|---|---|
| v1 | LR + HGB, vol-scaled long | 35.8 |
| **v3** | **LR + HGB + MC/worst-group LSTM, regime switch** | **53.008** |
| v4 | v3 smoothed day to day (EMA 0.5) | 36.4 |

**Post-scoring analysis** of 851–899 (signal held to the next close, 10 bps):

| | Sharpe | Return | Max DD | Right days |
|---|---|---|---|---|
| v1 | 1.65 | +9.8% | 6.2% | 41% |
| **v3** | **4.75** | **+12.7%** | **1.8%** | **73%** |
| v4 | 3.45 | +5.2% | 1.7% | 55% |

In a −9% market, v3 made +12.7% with a 1.8% max drawdown.

<!-- Speaker notes: The post-scoring numbers use closes rebuilt from same-row data after submission; nothing from them was used to build any submission. -->

---

## What we learned: why v4 lost

1. **The crash reversed direction daily:** lag-1 autocorrelation −0.16 (history +0.13). v3's daily calls were right 73% of the time; yesterday's call applied to today was right only 29%. Smoothing mixed in yesterday's call and made us late.
2. **The organizers treat |signal| ≤ 0.05 as flat:** smoothing shrank v4 into 38/50 flat days (book preview 8/4/38).
3. **The score rewards money made, not only Sharpe:** v4 had double v1's Sharpe but half its return, and they scored about the same.
4. **Our backtest had the wrong market type:** history and simulator trended (5-day block bootstrap), so smoothing looked good there.

**Fixes:**
- Add mean-reverting crash scenarios to the simulator.
- No smoothing in high vol.
- Positions either decisive (above 0.05) or exactly 0.
- Size for return as well as Sharpe (v5).

<!-- Speaker notes: A clean post-mortem is part of the architecture: every decision is logged, versioned and explainable after the fact. -->

---

## Engineering quality

- **Reproducible:**
  - one command per stage (`python -m src.run_round2`, `src.run_round3 --lstm`, `src.refine_round3`, `src.make_round3_v5`)
  - pinned `requirements-lock.txt`; deterministic CPU PyTorch; the v3 rerun matched the submitted file byte for byte
- **76 automated tests** must pass before any experiment:
  - future-row invariance of features, windows and rebuilt closes
  - train-only scaler, cutoffs and simulator
  - fold boundaries and purges
  - λ = 0 equivalence to sklearn
  - exact group-max loss; LSTM determinism
- **Auditable:** dataset hashes, split-manifest hash, simulator hashes and submission hashes recorded in every results note.
- **Honest reporting:** prespecified rules, stop conditions, synthetic metrics kept separate, and every assumption written down.

<!-- Speaker notes: This is what lets the team iterate fast under deadline: v3 to v4 to the post-mortem took minutes because every stage is scripted and versioned. -->

---

## Summary

1. **Leak-free foundation:** causal features and time-respecting folds, verified by tests.
2. **Regime-aware learning:** exact worst-regime (Group-DRO) objectives protect stressed regimes.
3. **Monte Carlo crashes:** train on, and stress-test against, crashes history never showed us.
4. **Crash-aware hybrid:** LR + boosting + attention LSTM, vol-normalised and clipped; sane outputs at 6× vol.
5. **Portfolio decision layer:** vol-scaled, regime-switching positions chosen by a fixed rule on Sharpe, drawdown and cost.

**Result: 53.008. Positive return and a 1.8% max drawdown through a 6× volatility crash.**

<!-- Speaker notes: Close on the idea that robustness was designed in at every layer: data, loss, simulation, model and position sizing. -->
