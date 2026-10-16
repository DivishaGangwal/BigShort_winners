# Round three v2: crash-aware hybrid signal (LR + HGB)

Chosen policy: **P5_regime_switch**. Submission (local, not committed): `outputs/local/divisha_round3/submission_round3_signal_v2.csv`, SHA-256 `52a6ccf564661a84a73664c511cbd093f8fb2f094b8fd1f8a2e3a4ef6f764cdd`.

Command: `python -m src.run_round3`. Data: round-three file, identical to the audited round-two data (SHA-256 file `462729cd85bff30c41bdd59a4ecd929ffa6c4050846b73c6cfa3811da052fb53`, CRLF form `462729cd85bff30c41bdd59a4ecd929ffa6c4050846b73c6cfa3811da052fb53`).

Features (24): the 18 round-two causal features, plus close-based same-row features (ret_1d_c, intraday, gap, close_sma10, clv, ret1_vol), plus 6 volatility-normalised versions inside the models. For 851–900 the current close is rebuilt from rows ≤ t only (close[t−5]·(1+momentum_5[t])); 50 rows rebuilt, momentum vs sma_10 estimates agree within 0.043. Validation: hiding 801–850 and rebuilding gives max error 0.044 (0.013%). The next close is never reconstructed, and realized 851–900 returns are not computed or used for any choice.

## Policy comparison (net of 10 bps per unit of signal change)

Real = 8 walk-forward blocks 451–849 (models refitted per block on earlier rows, 1-row purge). MC = 200 independent simulated 50-day crash paths per block (3×/5× vol, injected −8% to −15% days), never used in training. MC returns are synthetic.

| Policy | Score | Real Sharpe | Real blocks > 0 | Real mean return | Real mean / max MDD | Real turnover | MC Sharpe | MC median / p90 MDD | MC mean / p10 return | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| P0_long_full | 0.178 | 1.049 | 6/8 | 0.0339 | 0.075 / 0.142 | 1.00 | -0.692 | 0.339 / 0.459 | -0.1536 / -0.2644 | no |
| P1_long_volscaled | 0.151 | 0.938 | 6/8 | 0.0300 | 0.073 / 0.136 | 1.57 | -0.636 | 0.119 / 0.170 | -0.0384 / -0.1071 | yes |
| P3_blend | -0.124 | 0.019 | 4/8 | 0.0010 | 0.060 / 0.097 | 21.26 | -0.267 | 0.067 / 0.112 | -0.0113 / -0.0735 | yes |
| P2_model_volscaled | -0.327 | -1.043 | 2/8 | -0.0286 | 0.077 / 0.136 | 41.62 | 0.389 | 0.064 / 0.123 | 0.0122 / -0.0855 | yes |
| P4_model_band | -0.316 | -1.021 | 2/8 | -0.0281 | 0.077 / 0.135 | 41.50 | 0.389 | 0.064 / 0.123 | 0.0122 / -0.0857 | yes |
| P5_regime_switch | 0.216 | 0.019 | 4/8 | 0.0010 | 0.060 / 0.097 | 21.26 | 0.413 | 0.061 / 0.115 | 0.0115 / -0.0822 | yes |
| P6_trend_volscaled | -0.882 | -0.663 | 1/8 | -0.0050 | 0.047 / 0.069 | 11.64 | -1.100 | 0.070 / 0.115 | -0.0263 / -0.0984 | yes |
| P7_flat | 0.000 | 0.000 | 0/8 | 0.0000 | 0.000 / 0.000 | 0.00 | 0.000 | 0.000 / 0.000 | 0.0000 / 0.0000 | yes |

Selection rule (fixed before running): maximise 0.5·real Sharpe + 0.5·MC Sharpe at 10 bps, subject to mean real MDD ≤ 15% and median MC MDD ≤ 25%; ties within 0.05 go to the simpler policy; flat is the fallback.

## Final signal for 851–900

Trained on samples 25–850 (826). p_up range 0.362–0.598 (mean 0.507); signal range -0.177 to 0.450, mean 0.026, mean |signal| 0.078, turnover 6.37.

## Crash-awareness checks per block

| Block | n train | real p range | stress p range | stress share \|2p−1\| > 0.3 | corr(LR, HGB) |
|---|---|---|---|---|---|
| 451-500 | 425 | 0.281–0.807 | 0.214–0.794 | 0.083 | 0.65 |
| 501-550 | 475 | 0.301–0.772 | 0.217–0.778 | 0.062 | 0.60 |
| 551-600 | 525 | 0.295–0.743 | 0.250–0.773 | 0.047 | 0.64 |
| 601-650 | 575 | 0.267–0.749 | 0.243–0.765 | 0.061 | 0.65 |
| 651-700 | 625 | 0.239–0.693 | 0.236–0.768 | 0.046 | 0.40 |
| 701-750 | 675 | 0.299–0.707 | 0.240–0.750 | 0.040 | 0.72 |
| 751-800 | 725 | 0.408–0.653 | 0.168–0.756 | 0.037 | 0.15 |
| 801-849 | 775 | 0.278–0.639 | 0.261–0.739 | 0.022 | 0.51 |

## Assumptions and warnings

- Scoring formula unknown; assumed daily pnl = signal_t × close-to-close return t→t+1 minus cost × |Δsignal|, Sharpe annualised with √252.
- Same-row timing: features at row t (including sma_10/sma_30/momentum_5, which contain close[t]) are treated as known when setting signal_t.
- No future rows are used: signal_t depends on rows ≤ t only. Hidden closes are not reconstructed from later rows.
- Block 801–850 was the round-two reserved block; it is used here as development data. Return for t = 850 is unknown, so the last block ends at 849.
- Simulator design (3×/5× vol, 1–2 drops of −8% to −15%, burn-in from real training prices) is a design choice; MC Sharpe measures robustness, not real skill.
- Rounds 1–2 showed no directional edge on real data; the model component is expected to be close to noise.
- Costs assumed 10 bps per unit change for selection; 0 and 25 bps are in the JSON.
- Initial submission (v1, commit ea6c09d, no close features) scored 35.8 on the leaderboard.
