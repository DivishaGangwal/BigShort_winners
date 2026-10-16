# Round three v4: turnover refinements of v3

Leaderboard so far: v1 (volatility-scaled long): 35.8; v3 (LSTM hybrid, R0): 53.008.

Candidates and rule were fixed before evaluation (see `src/refine_round3.py`). Every refined position uses only the v3 positions up to that day; nothing from 851–900 outcomes is used.

| Candidate | Score | Real Sharpe | Real blocks > 0 | Real mean return | Real mean / max MDD | Real turnover | MC Sharpe | MC median / p90 MDD | MC turnover | 851–900 turnover | 851–900 mean \|s\| | Corr. with v3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_v3 | 0.497 | 0.266 | 5/8 | 0.0063 | 0.052 / 0.088 | 17.48 | 0.727 | 0.050 / 0.098 | 7.17 | 4.44 | 0.058 | 1.000 |
| R1_ema0.5 | 0.887 | 0.754 | 4/8 | 0.0158 | 0.044 / 0.083 | 7.25 | 1.021 | 0.036 / 0.073 | 3.06 | 1.69 | 0.037 | 0.871 |
| R2_hysteresis0.05 | 0.504 | 0.280 | 5/8 | 0.0064 | 0.052 / 0.088 | 17.33 | 0.727 | 0.049 / 0.097 | 6.85 | 3.95 | 0.058 | 0.980 |

Chosen: **R1_ema0.5**. File `outputs/local/divisha_round3/submission_round3_signal_v4.csv` (local), SHA-256 `5ee9e33154b9f30fb8b737bfe924694f6275cbad5da31e352318915d731abff0`.

