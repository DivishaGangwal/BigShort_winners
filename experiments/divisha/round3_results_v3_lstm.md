# Round three v3_lstm: crash-aware hybrid signal (LR + HGB + attention LSTM (worst-group with sim_crash, 3 seeds))

Chosen policy: **P5_regime_switch**. Submission (local, not committed): `outputs/local/divisha_round3/submission_round3_signal_v3_lstm.csv`, SHA-256 `a7dea75fb05331f2ddd07af409ef1ba5f280e010a8ce26be1860724f1f64205f`.

Command: `python -m src.run_round3`. Data: round-three file, identical to the audited round-two data (SHA-256 file `462729cd85bff30c41bdd59a4ecd929ffa6c4050846b73c6cfa3811da052fb53`, CRLF form `462729cd85bff30c41bdd59a4ecd929ffa6c4050846b73c6cfa3811da052fb53`).

Features (24): the 18 round-two causal features, plus close-based same-row features (ret_1d_c, intraday, gap, close_sma10, clv, ret1_vol), plus 6 volatility-normalised versions inside the models. For 851–900 the current close is rebuilt from rows ≤ t only (close[t−5]·(1+momentum_5[t])); 50 rows rebuilt, momentum vs sma_10 estimates agree within 0.043. Validation: hiding 801–850 and rebuilding gives max error 0.044 (0.013%). The next close is never reconstructed, and realized 851–900 returns are not computed or used for any choice.

## Policy comparison (net of 10 bps per unit of signal change)

Real = 8 walk-forward blocks 451–849 (models refitted per block on earlier rows, 1-row purge). MC = 200 independent simulated 50-day crash paths per block (3×/5× vol, injected −8% to −15% days), never used in training. MC returns are synthetic.

| Policy | Score | Real Sharpe | Real blocks > 0 | Real mean return | Real mean / max MDD | Real turnover | MC Sharpe | MC median / p90 MDD | MC mean / p10 return | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| P0_long_full | 0.178 | 1.049 | 6/8 | 0.0339 | 0.075 / 0.142 | 1.00 | -0.692 | 0.339 / 0.459 | -0.1536 / -0.2644 | no |
| P1_long_volscaled | 0.151 | 0.938 | 6/8 | 0.0300 | 0.073 / 0.136 | 1.57 | -0.636 | 0.119 / 0.170 | -0.0384 / -0.1071 | yes |
| P3_blend | 0.056 | 0.266 | 5/8 | 0.0063 | 0.052 / 0.088 | 17.48 | -0.154 | 0.061 / 0.107 | -0.0079 / -0.0651 | yes |
| P2_model_volscaled | 0.011 | -0.701 | 3/8 | -0.0182 | 0.060 / 0.089 | 34.05 | 0.723 | 0.050 / 0.099 | 0.0191 / -0.0613 | yes |
| P4_model_band | 0.024 | -0.672 | 3/8 | -0.0176 | 0.060 / 0.089 | 33.96 | 0.720 | 0.050 / 0.098 | 0.0191 / -0.0610 | yes |
| P5_regime_switch | 0.497 | 0.266 | 5/8 | 0.0063 | 0.052 / 0.088 | 17.48 | 0.727 | 0.050 / 0.098 | 0.0181 / -0.0597 | yes |
| P6_trend_volscaled | -0.882 | -0.663 | 1/8 | -0.0050 | 0.047 / 0.069 | 11.64 | -1.100 | 0.070 / 0.115 | -0.0263 / -0.0984 | yes |
| P7_flat | 0.000 | 0.000 | 0/8 | 0.0000 | 0.000 / 0.000 | 0.00 | 0.000 | 0.000 / 0.000 | 0.0000 / 0.0000 | yes |

Selection rule (fixed before running): maximise 0.5·real Sharpe + 0.5·MC Sharpe at 10 bps, subject to mean real MDD ≤ 15% and median MC MDD ≤ 25%; ties within 0.05 go to the simpler policy; flat is the fallback.

## Final signal for 851–900

Trained on samples 25–850 (826). p_up range 0.420–0.578 (mean 0.504); signal range -0.137 to 0.359, mean 0.020, mean |signal| 0.058, turnover 4.44.

## Crash-awareness checks per block

| Block | n train | real p range | stress p range | stress share \|2p−1\| > 0.3 | corr(LR, HGB) |
|---|---|---|---|---|---|
| 451-500 | 425 | 0.346–0.683 | 0.288–0.718 | 0.018 | 0.65 |
| 501-550 | 475 | 0.353–0.670 | 0.286–0.704 | 0.009 | 0.60 |
| 551-600 | 525 | 0.354–0.629 | 0.288–0.704 | 0.011 | 0.64 |
| 601-650 | 575 | 0.326–0.659 | 0.307–0.704 | 0.009 | 0.65 |
| 651-700 | 625 | 0.323–0.661 | 0.306–0.695 | 0.006 | 0.40 |
| 701-750 | 675 | 0.358–0.651 | 0.318–0.696 | 0.005 | 0.72 |
| 751-800 | 725 | 0.444–0.633 | 0.262–0.686 | 0.005 | 0.15 |
| 801-849 | 775 | 0.347–0.634 | 0.329–0.685 | 0.002 | 0.51 |

## Assumptions and warnings

- Scoring formula unknown; assumed daily pnl = signal_t × close-to-close return t→t+1 minus cost × |Δsignal|, Sharpe annualised with √252.
- Same-row timing: features at row t (including sma_10/sma_30/momentum_5, which contain close[t]) are treated as known when setting signal_t.
- No future rows are used: signal_t depends on rows ≤ t only. Hidden closes are not reconstructed from later rows.
- Block 801–850 was the round-two reserved block; it is used here as development data. Return for t = 850 is unknown, so the last block ends at 849.
- Simulator design (3×/5× vol, 1–2 drops of −8% to −15%, burn-in from real training prices) is a design choice; MC Sharpe measures robustness, not real skill.
- Rounds 1–2 showed no directional edge on real data; the model component is expected to be close to noise.
- Costs assumed 10 bps per unit change for selection; 0 and 25 bps are in the JSON.
- Initial submission (v1, commit ea6c09d, no close features) scored 35.8 on the leaderboard.
