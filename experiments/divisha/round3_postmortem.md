# Round three post-mortem: why v4 (36.4) scored below v3 (53.008)

Evaluation after final scoring, on 851–899 returns rebuilt from the round-three file (each close from rows ≤ t; rebuild error ≤ 0.013% when tested on 801–850). Nothing here was used to build any submission. Cost 10 bps per unit change.

## Signal_t × next close-to-close (buy & hold -9.2%)

| File | Leaderboard | Sharpe | Return % | Max DD % | Hit rate | Mean \|s\| | Sign flips |
|---|---|---|---|---|---|---|---|
| v1 | 35.8 | 1.65 | 9.83 | 6.16 | 0.41 | 0.219 | 0 |
| v3 | 53.008 | 4.75 | 12.68 | 1.84 | 0.73 | 0.057 | 33 |
| v4 | 36.4 | 3.45 | 5.15 | 1.72 | 0.55 | 0.037 | 13 |
| v2 (not submitted) | — | 4.94 | 16.90 | 1.89 | 0.71 | 0.077 | 29 |
| teammate (not submitted by us) | — | -1.17 | -1.44 | 1.92 | 0.45 | 0.049 | 5 |

## Signal_t × next open-to-open (buy & hold -9.0%)

| File | Leaderboard | Sharpe | Return % | Max DD % | Hit rate | Mean \|s\| | Sign flips |
|---|---|---|---|---|---|---|---|
| v1 | 35.8 | 1.16 | 3.86 | 6.26 | 0.49 | 0.219 | 0 |
| v3 | 53.008 | 4.10 | 6.84 | 0.71 | 0.69 | 0.057 | 33 |
| v4 | 36.4 | 3.87 | 4.34 | 1.05 | 0.59 | 0.037 | 13 |
| v2 (not submitted) | — | 4.19 | 8.79 | 1.16 | 0.67 | 0.077 | 29 |
| teammate (not submitted by us) | — | -0.10 | -0.11 | 1.21 | 0.49 | 0.049 | 5 |

## Signal_t × same-row return_1d (buy & hold -7.0%)

| File | Leaderboard | Sharpe | Return % | Max DD % | Hit rate | Mean \|s\| | Sign flips |
|---|---|---|---|---|---|---|---|
| v1 | 35.8 | 0.23 | 0.54 | 5.81 | 0.41 | 0.219 | 0 |
| v3 | 53.008 | 1.76 | 3.81 | 3.20 | 0.53 | 0.057 | 33 |
| v4 | 36.4 | 2.79 | 4.48 | 0.98 | 0.47 | 0.037 | 13 |
| v2 (not submitted) | — | 1.95 | 5.24 | 4.36 | 0.52 | 0.077 | 29 |
| teammate (not submitted by us) | — | 1.72 | 2.09 | 1.26 | 0.45 | 0.049 | 5 |

## Timing check

- Lag-1 autocorrelation of daily returns: 851–900 = -0.156; labelled history (26–850) = +0.128.
- v3's own call is right on 73% of days; yesterday's v3 call applied to today is right on 29%. EMA smoothing mixes in yesterday's call.
- Daily return sd: 851–900 = 0.0668; history = 0.0123.

## Book preview (organizer site) for v4

The preview reports net exposure +0.02, gross 0.04, long/short/flat 8/4/38, est. cost −0.013. v4 has no exact zeros; the 8/4/38 split is reproduced exactly by treating |signal| ≤ 0.05 as flat (0.04 gives 13/6/31, 0.06 gives 5/2/43). At that threshold v1 is 50/0/0, v3 13/8/29 and v4 8/4/38. If the scorer applies the same rule, smoothing pushed most of v4's positions into "flat". How the −0.013 cost is computed is not clear from v4's turnover (1.69).

## Conclusions

1. Smoothing lagged a reversing market: 851–900 returns reverse day to day (lag-1 autocorrelation −0.16 vs +0.13 in history). v3's own daily call was right 73% of days; yesterday's call applied to today was right 29%, and EMA 0.5 mixes yesterday in. v4's hit rate fell to 55%.
2. Smoothing shrank positions: averaging opposite calls cancels, so v4's gross exposure fell to 0.037 (v3 0.058) and most days fell under the apparent 0.05 flat threshold.
3. The leaderboard seems to reward money made, not only Sharpe: v4 (Sharpe 3.45, +5.2%) and v1 (Sharpe 1.65, +9.8%, larger drawdown) scored about the same (36.4 vs 35.8); v3 led on both (Sharpe 4.75, +12.7%).
4. The leaderboard order matches next-day returns (close-to-close or open-to-open), not same-row returns.
5. Our development evidence pointed the wrong way: the history trends (positive autocorrelation) and the simulator bootstraps 5-day blocks of that history, so both rewarded smoothing. No reversal-type crash was in the test set, and the 0.05 flat threshold was unknown.
