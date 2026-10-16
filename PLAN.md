# Round-one execution plan

Owners: **Friday, Divisha and Navjyot**. Repository: `DivishaGangwal/BigShort_winners`.

**Start gate:** documentation and local commits are authorized. The user will push. Model implementation, dependency installation, agent dispatch and training start only after the user explicitly approves this plan.

## 1. Freeze the common experiment contract — Friday

- Record the dataset SHA-256 and reproduce the 850-row, 800-labelled/50-unlabelled split.
- Use only the eight continuously available inputs: open, high, low, volume, volatility_20, rsi_14, volume_z_20 and order_flow.
- Exclude close, return_1d, sma_10, sma_30 and momentum_5 before feature construction, including in historical validation blocks.
- Record the inferred next-row close-direction target and current-row feature availability as explicit assumptions. The 799/799 visible label match supports the target inference; it cannot establish the scoring formula or prediction cutoff.
- Define a compact shared feature set: normalized OHLC geometry, trailing open-to-open returns, supplied volatility/RSI/volume/flow and a few trailing changes/lags. Drop initial warmup rows consistently. No future rows, centered windows or close reconstruction.
- Freeze feature names/order, missing-value handling, model factory interface, split manifest and metric definitions before model jobs begin.

All preprocessing learns from the relevant training block. A feature may use earlier observations in validation history when those observations are available under the declared submission mode. It may not use later rows or hidden labels. Report the supplied-row timing assumption until official timing is known.

## 2. Implement and verify the shared pipeline — Friday

Create data loading, causal features, chronological splits, fold-local preprocessing, common metrics, artifact naming and the single-machine launcher. Verify schema/mask enforcement, feature timing, disjoint split boundaries and probability/timestamp alignment with meaningful small fixtures before full model runs.

The model interface returns class-1 probabilities for a supplied validation block. Friday owns shared files; Divisha and Navjyot implement their model-specific modules against that interface.

No new dependency is installed during this planning stage. After approval, establish one reproducible Python environment with pandas, NumPy and scikit-learn and record exact resolved versions. CPU execution is sufficient for the first comparison; no GPU requirement is assumed.

## 3. Use the same chronological folds for all models

| Fold | Train timestamps | Purged timestamp | Validation timestamps |
| --- | --- | --- | --- |
| 1 | 1–449 | 450 | 451–500 |
| 2 | 1–499 | 500 | 501–550 |
| 3 | 1–549 | 550 | 551–600 |
| 4 | 1–599 | 600 | 601–650 |
| 5 | 1–649 | 650 | 651–700 |

Ranges precede any common warmup-row removal. The one-row purge is provisional for the inferred one-row label horizon. Use a larger gap if the actual horizon requires it.

Reserve timestamps 701–800 for one later evaluation with training timestamps 1–699 and purged timestamp 700. Do not fit, tune, choose thresholds, calibrate or blend using reserved labels. The descriptive full-file audit already inspected aggregate labels, so this is not a completely blind test set. Rows 801–850 remain genuinely unlabelled locally.

## 4. Run three distinct model families on one machine

First use a fixed configuration for each family. Do not open a large hyperparameter search before establishing comparable baseline results.

| Owner | First model | Initial settings |
| --- | --- | --- |
| Friday | Regularized logistic regression | C=1, solver=lbfgs, max_iter=2000, no class weighting, fold-local standardization |
| Divisha | Histogram gradient boosting | learning_rate=0.05, max_iter=150, max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=1, early_stopping=False |
| Navjyot | Extra Trees | n_estimators=300, max_depth=5, min_samples_leaf=20, max_features=1.0, bootstrap=False, n_jobs=1 |

Use seed 42 where supported. Disabling automatic boosting early stopping avoids introducing an internal random validation split. Friday also computes the training-prior probability/majority-class baseline independently within each fold.

The launcher runs at most three separate model-family processes. Each process handles its five folds sequentially and uses one compute thread. Set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and NUMEXPR_NUM_THREADS to 1 before numerical imports. Use Windows-compatible process spawning and isolated output folders. Cap concurrency to available CPU/memory; a constrained machine may queue jobs. Measure elapsed time without promising threefold speedup.

## 5. Compare development results before adding complexity

For each model and baseline, report per-fold and pooled out-of-fold accuracy, balanced accuracy, ROC AUC, log loss, Brier score, sample counts and elapsed time. Threshold class-1 probabilities at 0.5 for the first comparison. Single-class AUC remains unavailable rather than being replaced with a number.

Because official scoring is unavailable, present both classification and probability performance. A provisional development ranking can use mean-fold accuracy, with lower log loss as a tie-breaker. This is an internal convention, not an inferred organizer score. Disclose fold variation and the limited validation sample size.

Check high-RSI and high-volatility development windows with sample counts. Compare each family to the fold-specific prior baseline. Further tuning uses a small prespecified development-only search. Blending and sigmoid calibration are optional later experiments, justified by chronological evidence. Calibration fitting must be disjoint from base-model fitting and evaluated on later rows.

Friday presents a measured comparison and recommends whether any candidate merits further work. Do not generate a final submission merely because three training jobs completed.

## 6. Freeze, evaluate once and prepare the export

After development choices are fixed, evaluate the selected procedure once on timestamps 701–800. Record the outcome without retuning against that period. Refit the approved procedure on permitted labelled history for timestamps 801–850, keeping any required calibration data separate from base-model fitting.

The final export must have exactly the header `timestamp,prediction,confidence`, 50 unique ascending timestamps 801–850, integer 0/1 predictions, finite confidence values and no index column or missing values.

Confidence is provisionally the probability of the submitted class, pending official clarification. Keep p_up internally so P(class 1) can be exported instead if required. Do not use example rows as predictions. Record the source commit, dataset hash and configuration with the final artifact.

## Commit and handoff sequence

1. Commit the named role assignments in `ROLES.md`.
2. Commit this plan in `PLAN.md` and make the README point to both documents.
3. The user pushes the local documentation commits.
4. Friday asks for start approval. Until approval, no implementation, agent dispatch, dependency installation or training.
5. After approval, create the planned ownership branches and commit the shared experiment contract. Model owners then commit implementations/results against that base.
6. Friday reviews source SHAs and recorded metrics before integration. Never rewrite another owner's shared history.

Commits are the coordination record, not an automatic notification mechanism. How Divisha and Navjyot consume their assigned work must be established without claiming they have already received a message.

## Later rounds

Reuse source hashing, causal features, experiment records and temporal validation. Round two adds a separately defined bounded target and declared crash scenarios. Round three adds execution timing, position constraints, turnover-based costs and the organizer's risk-adjusted reward. Their targets and scoring are not implied by round one's labels. Hidden closes in 801–850 do not permit a local realized-return backtest for that block.
