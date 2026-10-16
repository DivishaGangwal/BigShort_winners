# Team roles

Repository: [DivishaGangwal/BigShort_winners](https://github.com/DivishaGangwal/BigShort_winners).

Status: roles assigned in this document. Implementation, agent dispatch and training await the user's explicit start approval. The user will push the prepared commits.

## Ownership

| Owner | Model responsibility | Shared responsibility | Planned branch |
| --- | --- | --- | --- |
| **Friday** | Regularized logistic regression and training-prior baseline | Dataset contract, causal features, split manifest, common metrics, training launcher, result review and integration | `agents/friday` |
| **Divisha** | Histogram gradient boosting classifier | Check nonlinear interactions and performance across chronological development folds | `agents/divisha` |
| **Navjyot** | Extra Trees classifier | Check an alternative tree ensemble and complementary errors across the same folds | `agents/navjyot` |

The three named owners define responsibility. This commit does not claim that another chat or process has been contacted. Branch names are agreed planning names; branches and implementation files will be created after start approval.

## File boundaries after approval

| Owner | Primary files / folders |
| --- | --- |
| Friday | `src/data.py`, `src/features.py`, `src/splits.py`, `src/metrics.py`, `src/train_parallel.py`, `src/export.py`, `src/models/logistic.py`, shared configuration and integration reports |
| Divisha | `src/models/hist_boost.py`, `configs/models/divisha_hist_boost.json`, `experiments/divisha/` |
| Navjyot | `src/models/extra_trees.py`, `configs/models/navjyot_extra_trees.json`, `experiments/navjyot/` |

Shared interfaces are frozen before model-specific implementation. A change to features, folds, scoring or shared files requires a coordination commit so every model remains comparable.

## One machine, three model jobs

Training can run on a single designated machine. After the shared pipeline is verified, launch independent model-family jobs for Friday, Divisha and Navjyot, each using the same source hash, feature version, split manifest and seed.

Use at most three worker processes and one compute thread per process initially. Run folds sequentially within each worker and set Extra Trees `n_jobs=1` to avoid nested parallelism. Cap BLAS/OpenMP threads in each worker before importing numerical libraries. Reduce concurrency if available CPU or memory cannot support three jobs. Agent collaboration through commits and parallel model execution are separate mechanisms.

## Required handoff

Each owner commits its implementation/configuration and a result note containing:

1. Task ID, base/source commit SHA, dataset SHA-256, feature version and split-manifest hash.
2. Model settings, dependency versions, seed and exact reproduction command.
3. Development-fold metrics against the same training-prior baseline, sample counts and elapsed time.
4. Out-of-fold probability artifact location and hash, warnings and unresolved assumptions.

Keep generated models and row-level prediction artifacts local unless sharing them is intentionally authorized. Commit small aggregate results and reproduction instructions. Friday reviews handoffs and records accepted source SHAs before integration. Do not rewrite another owner's shared branch history.

## Completion criteria for the first comparison

- All three model families use the same features, folds, missing-feature mask and metric implementation.
- No future-row predictors, random time splits, globally fitted preprocessing or training on reserved evaluation labels.
- Results are measured and reproducible. A successful training process alone is not evidence of predictive value.
- Calibration and blending are later development experiments, accepted only if chronological results justify them.
- Friday presents the comparison before generating a final competition submission.
