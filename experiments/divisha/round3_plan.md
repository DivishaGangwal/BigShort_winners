# Round 3 plan: from probabilities to a scored trading policy

Status: plan only. No round-3 code until the round-3 dataset and rules arrive.
Owner: Divisha. Builds on V3 (ARCHITECTURE_V3_PLAN.md), round-two results and the Monte Carlo crash simulator.

## 0. What the evidence says so far

| Evidence | Value | Meaning for round 3 |
|---|---|---|
| Round 2, 7 dev folds (451–800), 19 configurations | Training-prior baseline has the lowest mean-fold log loss (0.6943). Best model: LSTM_E (MC + worst-regime) at 0.6994; LR_A at 0.7136 | No configuration shows a directional edge. Pooled accuracy is 0.49–0.52 for every model. |
| Round 2 synthetic crash stress test | LR_A stress log loss 1.221 → LR_D (MC-augmented) 0.704; LSTM_E 0.691 | Monte Carlo augmentation does not create an edge, but it **stops models from blowing up on crash-like inputs**. That matters more for drawdown than for accuracy. |
| Round 1 (LR, 5 folds) | Every arm worse than the prior | Same conclusion. |
| Prediction block 851–900 | volatility_20 up to 0.0744 (6.4× labelled median, 3.6× labelled max) | The next scored period may be a crash or its aftermath. Unscaled position sizes would carry 4–6× the usual risk. |
| Our round-2 score | 2.224 (from V3 plan), team currently 5th | The scoring formula is still unknown. The "correct-confidence" hypothesis already fails to reproduce it. |

**The main implication:** if round 3 scores Sharpe, PnL, drawdown and costs, the decisive component is **position sizing and turnover control**, not the classifier. With accuracy near 50%, a policy that trades on every small probability move pays costs on noise.

## 1. Issues in the V3 plan

1. **Wrong objective for round 3.** V3 selects by pooled log loss. Round 3 is scored on Sharpe, PnL, max drawdown and costs. A well-calibrated p near 0.5 can still produce a losing, high-turnover strategy. A **decision layer** (p, volatility, cost → position) must be designed and selected on backtest utility, with log loss kept as a diagnostic.
2. **The 0.5 logit shrinkage keeps every binary decision.** For trading, sign flips are what create turnover. Shrinkage that leaves `sign(2p−1)` unchanged does nothing about costs. A no-trade band and hysteresis are needed.
3. **The exp(−5·logloss) ensemble weights are estimated on three 40-row inner blocks.** Log-loss differences between experts are about 0.005–0.02, smaller than the sampling noise of a 40-row block (SE ≈ 0.05). The weights will be noise, and the factor 5 is itself an unvalidated hyperparameter. Equal weights (or the prior) will likely match it. Keep the adaptive ensemble only as a reported challenger.
4. **"No simulated future targets" discards our only robustness evidence.** Round 2 shows MC augmentation halving crash-case log loss for LR. For round 3, MC is most valuable for **simulating the trading policy's PnL and drawdown distribution**, not for labels. Keep it in the backtest even if the experts train only on real rows.
5. **The recent-300-row LightGBM expert** trains on 300 rows that contain no crash. It is the expert most likely to extrapolate badly when volatility is 5×. Its predictions need clipping or regime gating, or it should be excluded in high volatility.
6. **The commit reference does not match.** V3 cites "local merged 735b79d". Our local merge is `96b1619` (origin/main `7613cd7`). Whoever wrote V3 has a commit we don't. Resolve this before any audit claim.
7. **Hidden-close reconstruction (round2_v1)**, correctly flagged in V3. If the organizers evaluate round 3 on genuinely unseen data, any reconstruction-based edge disappears. Remove it and do not reuse its backtests.
8. **Execution timing is undefined.** Features at row t include sma_10, sma_30 and momentum_5, which contain close[t]. If round-3 positions execute at the open of t (or must be submitted before close[t]), those features are look-ahead and every backtest using them is inflated. The timing must come from the rules.
9. **Selecting on blocks 451–750** that we have already looked at many times (rounds 1–2 and V3) makes these blocks optimistic. Use them for diagnostics; the final policy choice should rest on prespecified rules plus MC robustness, not on small differences.
10. **The data can be revised between rounds.** Round 2 silently changed volatility_20, rsi_14 and order_flow in 801–850. The round-3 file must be re-audited row by row against round 2 before any reuse.

## 2. Why round 3 can fail (failure modes to design against)

| # | Failure mode | Mechanism | Guard |
|---|---|---|---|
| F1 | Costs eat everything | p hovers near 0.5 → daily sign flips → maximum turnover | No-trade band, hysteresis, minimum holding period, turnover budget |
| F2 | Crash drawdown | Fixed position size at 5–6× normal vol | Volatility targeting: position ∝ target_vol / σ_t, with a crash brake |
| F3 | Directional bias in the wrong regime | Long drift learned in 2018–2021 meets a crash; or short bias meets a V-shaped rebound (851–900 has +8% and −15% days) | Cap net exposure in high vol; test both crash and rebound paths in MC |
| F4 | Look-ahead in our backtest | Same-row close-inclusive features traded at the same close/open | Timing contract + prefix/mutation tests on the backtest engine |
| F5 | Score-formula mismatch | Optimising annualised Sharpe when they score per-period, % MDD vs absolute, or costs per side vs round trip | Score-hypothesis harness (section 4); choose policies robust across hypotheses |
| F6 | Sharpe noise | Over 50 days, annualised Sharpe SE ≈ √(252/50) ≈ 2.2 | Judge policies on MC/bootstrap distributions and a lower quantile, not point estimates |
| F7 | Overconfidence / extrapolation | Standardised features up to \|z\| ≈ 24 in 851–900 | Clip transformed features; damp positions when inputs are out of range |
| F8 | Path-dependent drawdown rules backfire | Stop-loss exits at the bottom, re-enters late, pays costs twice | Test drawdown rules in MC before adopting; keep them simple |
| F9 | Format/units error | Positions in [−1, 1] vs shares vs weights; sign conventions | Export validator and a toy example reproduced against their scorer if one is provided |
| F10 | Overfitting the policy | Many band/vol/leverage combinations searched on the same blocks | Small prespecified grid, selection rule fixed before looking |

## 3. Backtesting engine (to build once the data arrives)

Contract (to confirm from the rules): position `w_t ∈ [−1, 1]` decided with information available at the decision time, held over the next return interval.

```
pnl_t      = w_t · r_{t→t+1} − c · |w_t − w_{t−1}|
equity_t   = Π (1 + pnl)  (or Σ pnl if they score arithmetic PnL)
Sharpe     = mean(pnl) / std(pnl) · √252   (also per-period, unannualised)
Sortino, max drawdown (% and absolute), Calmar, turnover, exposure, hit rate, cost drag
```

- **Timing variants:** decide at close t / trade at close t; decide at close t / trade at open t+1. Features restricted accordingly; sma/momentum excluded when they are not available in time.
- **Cost grid:** 0, 5, 10, 20, 50 bps per unit turnover, plus whatever the rules state.
- **Tests:** mutating any data after t never changes w_≤t (prefix test); shifting returns by one row must destroy any apparent edge (alignment test); a known toy series reproduces hand-computed Sharpe/MDD.
- **Data sources:**
  1. Real walk-forward on 451–850 (folds frozen, policies frozen).
  2. 851–900 if round 3 reveals those closes: a genuine crash test. Evaluate each frozen policy **once**.
  3. Block bootstrap of real returns (normal regime).
  4. Monte Carlo crash paths from the round-two simulator (3×/5× vol, injected drops, rebounds), fitted on training rows only, giving the **distribution** of Sharpe and MDD per policy.

## 4. Score-formula hypotheses (to discriminate, not to assume)

We know: round 3 is analysed on Sharpe, money made, max drawdown and costs, and we need "≥ 50". The scale of 50 is unknown. Candidate formulations:

| ID | Hypothesis | Notes |
|---|---|---|
| H1 | Annualised net Sharpe | Simple; 50 impossible on this scale, so 50 is probably a scaled or composite score |
| H2 | Weighted composite: a·Sharpe + b·return − c·MDD − d·cost | Most consistent with "Sharpe, money, drawdown, cost" |
| H3 | Calmar-type: annual return / MDD | Rewards avoiding the crash |
| H4 | Percentile or rank among teams (0–100) | "50" = above median; relative to other teams |
| H5 | Normalised 0–100 per metric, then averaged | Each metric min-max scaled across teams |
| H6 | Final equity / PnL with drawdown or cost penalty | "how much money you made" |

Plan: compute all hypotheses for every candidate policy in the backtest. Prefer the policy that ranks well across all of them (minimax regret) over one that wins under a single guessed formula. Use rounds 1–2 leaderboard numbers (ours 2.224 and other teams' scores, if available) only to **rule hypotheses out**, never to fit weights from 2–3 data points.

## 5. Candidate policies (small, prespecified)

| Policy | Definition |
|---|---|
| P0 Flat | w = 0 (sanity: Sharpe undefined/0, no drawdown) |
| P1 Buy & hold | w = 1 |
| P2 Vol-targeted long | w = min(1, σ* / σ_t), σ* = training median volatility_20 |
| P3 Vol-targeted long + crash brake | P2, scaled to ≤ 0.25 when σ_t > training max |
| P4 Model signal, banded | s = 2p − 1; trade only if \|s\| > δ, else hold w_{t−1}; w = clip(k·s·σ*/σ_t) |
| P5 Blend | P3 base exposure + small P4 tilt |
| P6 Ensemble of V3 experts → P4/P5 | V3 experts used only through the decision layer |

Grid: δ ∈ {0.05, 0.10}, k ∈ {1, 2}, brake threshold ∈ {training q99, training max}. That is ≤ 20 policies in total. The selection rule is fixed before looking: **maximise the 25th percentile of the H2/H3/H5 scores across MC + bootstrap paths, subject to median MDD ≤ X; ties go to the simpler policy.** Report every policy.

## 6. Integration of previous architecture decisions

| Keep | Drop / change |
|---|---|
| Causal features, train-only preprocessing, purged chronological folds (rounds 1–2) | Hidden-close reconstruction (round2_v1) |
| Volatility-normalised features (V3) | Log loss as the final selection criterion |
| MC crash simulator for stress paths and optional augmentation (round 2, best stress results: LR_D, LSTM_E) | Unregularised adaptive ensemble weights from 40-row blocks as the default |
| Prior baseline always reported; stop/challenger logic | Selecting on the same 451–750 blocks repeatedly |
| Group-DRO / worst-regime as a challenger | Group-DRO with weights summing to 1 (Hessian scale issue, per V3) |

## 7. Execution order once the dataset arrives

1. Audit the round-3 file: hash, shape, revisions vs round 2, newly revealed closes (851–900?), cost and timing rules.
2. Lock the timing contract and the feature availability mask.
3. Build the backtest engine with tests (F4, F9).
4. Run the score-hypothesis harness on P0–P3 (no model needed) to see what the scale of "50" implies.
5. Add the V3 experts through the decision layer (P4–P6).
6. MC + bootstrap robustness, apply the prespecified selection rule, one frozen evaluation on revealed 851–900 data.
7. Export, validate format, record hashes. The user pushes and uploads.

## 8. Honest expectation

No configuration in rounds 1–2 showed a directional edge, so a score ≥ 50 cannot be guaranteed by modelling. The most controllable levers are **not losing**: low turnover, volatility-scaled exposure and drawdown control through the crash. If the score is relative to other teams (H4/H5), avoiding a blow-up while others take crash losses may itself be enough to move up the ranking.

## 9. Needed from the user before coding

1. The exact round-3 scoring text, and what scale "50" is on.
2. Leaderboard numbers for rounds 1–2 (ours 2.224 and others), to rule score hypotheses in or out.
3. Round-3 rules: position range, shorting/leverage, execution timing, cost per trade, scored horizon.
4. Where commit `735b79d` lives (V3 cites it; it isn't in our clone).
