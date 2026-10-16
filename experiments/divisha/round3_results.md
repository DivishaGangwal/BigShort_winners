# Round three: crash-aware hybrid signal (initial submission)

Chosen policy: **P1_long_volscaled**. Submission (local, not committed): `outputs/local/divisha_round3/submission_round3_signal.csv`, SHA-256 `f15af26f1c16925b8a1208d78c3217088b9cb77b0ff42088b0ecd817d099c0d5`.

Command: `python -m src.run_round3`. Data: round-three file, identical to the audited round-two data (SHA-256 file `a96bd62a30eaf39f63e46f90d7da2876591ce810fcbdda4241b2df1a3a84d1f2`, CRLF form `462729cd85bff30c41bdd59a4ecd929ffa6c4050846b73c6cfa3811da052fb53`).

## Policy comparison (net of 10 bps per unit of signal change)

Real = 8 walk-forward blocks 451–849 (models refitted per block on earlier rows, 1-row purge). MC = 200 independent simulated 50-day crash paths per block (3×/5× vol, injected −8% to −15% days), never used in training. MC returns are synthetic.

| Policy | Score | Real Sharpe | Real blocks > 0 | Real mean return | Real mean / max MDD | Real turnover | MC Sharpe | MC median / p90 MDD | MC mean / p10 return | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| P0_long_full | 0.178 | 1.049 | 6/8 | 0.0339 | 0.075 / 0.142 | 1.00 | -0.692 | 0.339 / 0.459 | -0.1536 / -0.2644 | no |
| P1_long_volscaled | 0.151 | 0.938 | 6/8 | 0.0300 | 0.073 / 0.136 | 1.57 | -0.636 | 0.119 / 0.170 | -0.0384 / -0.1071 | yes |
| P3_blend | -0.160 | -0.025 | 3/8 | -0.0016 | 0.063 / 0.103 | 21.52 | -0.295 | 0.067 / 0.114 | -0.0120 / -0.0729 | yes |
| P2_model_volscaled | -0.402 | -1.154 | 1/8 | -0.0339 | 0.081 / 0.124 | 42.15 | 0.351 | 0.064 / 0.122 | 0.0108 / -0.0891 | yes |
| P4_model_band | -0.401 | -1.152 | 1/8 | -0.0339 | 0.081 / 0.123 | 42.24 | 0.350 | 0.064 / 0.121 | 0.0109 / -0.0889 | yes |
| P5_regime_switch | 0.166 | -0.025 | 3/8 | -0.0016 | 0.063 / 0.103 | 21.52 | 0.358 | 0.061 / 0.118 | 0.0098 / -0.0818 | yes |
| P6_trend_volscaled | -0.882 | -0.663 | 1/8 | -0.0050 | 0.047 / 0.069 | 11.64 | -1.100 | 0.070 / 0.115 | -0.0263 / -0.0984 | yes |
| P7_flat | 0.000 | 0.000 | 0/8 | 0.0000 | 0.000 / 0.000 | 0.00 | 0.000 | 0.000 / 0.000 | 0.0000 / 0.0000 | yes |

Selection rule (fixed before running): maximise 0.5·real Sharpe + 0.5·MC Sharpe at 10 bps, subject to mean real MDD ≤ 15% and median MC MDD ≤ 25%; ties within 0.05 go to the simpler policy; flat is the fallback.

## Final signal for 851–900

Trained on samples 25–850 (826). p_up range 0.364–0.616 (mean 0.508); signal range 0.157 to 0.929, mean 0.218, mean |signal| 0.218, turnover 1.90.

## Crash-awareness checks per block

| Block | n train | real p range | stress p range | stress share \|2p−1\| > 0.3 | corr(LR, HGB) |
|---|---|---|---|---|---|
| 451-500 | 425 | 0.256–0.808 | 0.218–0.822 | 0.095 | 0.64 |
| 501-550 | 475 | 0.312–0.817 | 0.231–0.789 | 0.066 | 0.60 |
| 551-600 | 525 | 0.297–0.760 | 0.233–0.757 | 0.054 | 0.62 |
| 601-650 | 575 | 0.297–0.751 | 0.237–0.781 | 0.051 | 0.54 |
| 651-700 | 625 | 0.232–0.685 | 0.233–0.756 | 0.049 | 0.43 |
| 701-750 | 675 | 0.331–0.731 | 0.239–0.750 | 0.036 | 0.74 |
| 751-800 | 725 | 0.403–0.651 | 0.185–0.751 | 0.037 | 0.08 |
| 801-849 | 775 | 0.280–0.659 | 0.246–0.758 | 0.024 | 0.60 |

## Assumptions and warnings

- Scoring formula unknown; assumed daily pnl = signal_t × close-to-close return t→t+1 minus cost × |Δsignal|, Sharpe annualised with √252.
- Same-row timing: features at row t (including sma_10/sma_30/momentum_5, which contain close[t]) are treated as known when setting signal_t.
- No future rows are used: signal_t depends on rows ≤ t only. Hidden closes are not reconstructed from later rows.
- Block 801–850 was the round-two reserved block; it is used here as development data. Return for t = 850 is unknown, so the last block ends at 849.
- Simulator design (3×/5× vol, 1–2 drops of −8% to −15%, burn-in from real training prices) is a design choice; MC Sharpe measures robustness, not real skill.
- Rounds 1–2 showed no directional edge on real data; the model component is expected to be close to noise.
- Costs assumed 10 bps per unit change for selection; 0 and 25 bps are in the JSON.
- The LSTM is not in this initial hybrid (runtime); planned for the improvement pass.
