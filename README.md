# QRT Asset Allocation Performance Forecasting

Predict whether the next return of each of 278 systematic allocations will be
positive ([QRT × ENS Challenge Data](https://challengedata.ens.fr)). The metric is
accuracy. The whole project fits in one notebook: [`report.ipynb`](report.ipynb).

## Results

These numbers come from a holdout of 504 dates, set aside before any modelling:

| Model | Accuracy |
| --- | ---: |
| Always "up" (baseline) | 50.6% |
| Logistic regression | 52.3% |
| **equal3 ensemble** | **52.4%** |

- **Vs the baseline: +1.8 pt**, 95% CI [+0.9, +2.8] (bootstrap over dates).
- Vs the logistic regression: +0.12 pt, 95% CI [−0.10, +0.34]. This gain is
  **not** significant.

The notebook also runs a 5-fold cross-validation grouped by date, on all
527,073 rows:

| Model | Accuracy | Delta vs logistic | 95% CI |
| --- | ---: | ---: | ---: |
| Logistic | 52.49% | – | – |
| Ridge on `asinh(target)` | 52.49% | 0.00 pt | [−0.10, +0.10] |
| LightGBM | 52.53% | +0.04 pt | [−0.15, +0.23] |
| **equal3** | **52.65%** | **+0.16 pt** | [+0.06, +0.26] |

## Method

- **Validation grouped by date (`TS`).** The allocations of a given date share a
  common market factor. Splitting a date between train and validation would leak
  that factor.
- **Features fitted inside each fold.** Every statistic (medians, scaling,
  categories) is learned on the training fold only. The feature matrix contains:
  - raw returns and volumes, median-imputed and standardized, with
    missing-value indicators;
  - one-hot `GROUP` and `ALLOCATION`;
  - 18 cross-sectional features: rank, distance to the median and robust
    z-score within the current date.
- **Three models on the same features:**
  - a ridge logistic regression;
  - a ridge regression on `asinh(target)`, which uses return magnitudes while
    keeping the sign boundary;
  - a LightGBM with frozen hyperparameters, weakly correlated with the two
    linear models.
- **equal3:** the average of the three predicted probabilities, thresholded at
  0.5, with no tuned weights. Each model is trained on three random 80% subsets
  of the dates.

## Leakage avoided

- `ROW_ID` is never a feature. A 1-NN on `ROW_ID` scores 100% in-sample, which
  is pure memorization.
- `TS` is only used for grouping. The anonymized dates can be partly put back in
  order, because a date's target reappears as the next date's `RET_1` (88% sign
  agreement). This link is excluded: using it would be leakage, not forecasting.

## Run it

1. Download `X_train_9xQjqvZ.csv`, `y_train_Ppwhaz8.csv` and
   `X_test_1zTtEnD.csv` from the challenge page. The data is not included in this
   repository.
2. Put the three files in a `data/` folder next to `report.ipynb`.
3. Install the dependencies with `pip install -r requirements.txt`.
4. Run the notebook. It takes about 6 minutes and writes
   `submission_equal3.csv`.

## Limitations

- Differences between models (0.1–0.2 pt) are much smaller than the
  leaderboard noise (about ±1.4 pt).
- Most of the ensemble's gain comes from dates with few allocations.
- The cross-sectional features assume that all allocations of a date are known
  when predicting.
