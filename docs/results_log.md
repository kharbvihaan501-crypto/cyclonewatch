# Results Log



## Sat 14 Nov 2026


## Sat 14 Nov 2026 15:38: baselines on VALIDATION (2016-2019)

Command: `py -3.12 -m src.eval.baselines`

Over-water rows only (DIST2LAND > 0 at t). Persistence = logistic regression on DWIND_24, fitted on train (1980-2015). Climatology = train RI rate by BASIN. Thresholds chosen on validation, then scored on the same validation rows (optimistic). 95% CI: 1,000 bootstrap resamples of whole storms per subset, seed 0; resamples with no RI rows skipped. Episode = consecutive 6-hourly RI rows in one storm; lead time = hours the alert was already red, unbroken, before the episode's first row. Test split not loaded.

**Persistence: alert threshold 0.0816 (max F1 on validation over-water rows, all basins); mean predicted P 0.0794 vs actual RI rate 0.1011 (over water)**

```
                       subset  storms  rows  RI rows  RI episodes  episodes caught  false-alarm episodes               PR-AUC            precision               recall                   Brier recall (last-24h rise < 30 kt)  episodes caught %   median lead h false-alarm episodes / storm boot used / skipped
                Bay of Bengal      14   141       13            6                4                     9 0.129 [0.059, 0.263] 0.161 [0.059, 0.290] 0.692 [0.400, 1.000] 0.0863 [0.0199, 0.1637]           0.667 [0.400, 1.000] 66.7 [20.0, 100.0] 0.0 [0.0, 12.0]            0.64 [0.28, 1.00]             998 / 2
           North Indian Ocean      23   342       28           11                8                    18 0.154 [0.077, 0.258] 0.173 [0.083, 0.270] 0.821 [0.600, 1.000] 0.0752 [0.0366, 0.1176]           0.792 [0.565, 1.000] 72.7 [42.8, 100.0]  0.0 [0.0, 6.0]            0.78 [0.48, 1.13]            1000 / 0
All basins, <= 400 km of land     199  1741      120           57               36                   147 0.121 [0.092, 0.160] 0.132 [0.096, 0.171] 0.675 [0.562, 0.786] 0.0644 [0.0492, 0.0816]           0.594 [0.472, 0.720]  63.2 [51.7, 75.8]  0.0 [0.0, 6.0]            0.74 [0.63, 0.84]            1000 / 0
                   All basins     323  5333      539          170               99                   279 0.187 [0.164, 0.211] 0.203 [0.178, 0.226] 0.744 [0.690, 0.795] 0.0903 [0.0803, 0.1006]           0.664 [0.607, 0.729]  58.2 [50.8, 66.2]  0.0 [0.0, 6.0]            0.86 [0.78, 0.96]            1000 / 0
```

**Climatology: alert threshold 0.0614 (max F1 on validation over-water rows, all basins); mean predicted P 0.0676 vs actual RI rate 0.1011 (over water)**

```
                       subset  storms  rows  RI rows  RI episodes  episodes caught  false-alarm episodes               PR-AUC            precision               recall                   Brier recall (last-24h rise < 30 kt) episodes caught %    median lead h false-alarm episodes / storm boot used / skipped
                Bay of Bengal      14   141       13            6                0                     0 0.092 [0.019, 0.179]             - [-, -] 0.000 [0.000, 0.000] 0.0849 [0.0197, 0.1615]           0.000 [0.000, 0.000]    0.0 [0.0, 0.0]         - [-, -]            0.00 [0.00, 0.00]             998 / 2
           North Indian Ocean      23   342       28           11                0                     0 0.082 [0.038, 0.131]             - [-, -] 0.000 [0.000, 0.000] 0.0758 [0.0370, 0.1196]           0.000 [0.000, 0.000]    0.0 [0.0, 0.0]         - [-, -]            0.00 [0.00, 0.00]            1000 / 0
All basins, <= 400 km of land     199  1741      120           57               43                   182 0.064 [0.047, 0.086] 0.068 [0.048, 0.091] 0.725 [0.563, 0.860] 0.0645 [0.0494, 0.0812]           0.729 [0.566, 0.866] 75.4 [62.0, 86.4]   6.0 [0.0, 6.0]            0.91 [0.80, 1.04]            1000 / 0
                   All basins     323  5333      539          170              135                   190 0.115 [0.096, 0.134] 0.107 [0.093, 0.122] 0.809 [0.728, 0.876] 0.0918 [0.0815, 0.1024]           0.800 [0.722, 0.868] 79.4 [72.0, 85.9] 12.0 [6.0, 18.0]            0.59 [0.51, 0.67]            1000 / 0
```


## Sat 14 Nov 2026: persistence hard-case recall on VALIDATION (2016-2019)

Hard case = RI rows where the last-24h wind rose < 30 kt (DWIND_24 < 30). Persistence baseline, over-water rows only, alert threshold 0.0816 (max F1 on validation, same as the run above). 95% CI: 1,000 storm bootstrap resamples per subset, seed 0; resamples with no RI rows skipped. Test split not loaded.

Command:
```
py -3.12 -c "
import pandas as pd
from src.eval.baselines import TRAIN_PATH, VAL_PATH, fit_persistence, predict_persistence, choose_threshold
from src.eval.metrics import evaluate
tr = pd.read_parquet(TRAIN_PATH); va = pd.read_parquet(VAL_PATH)
va['P'] = predict_persistence(fit_persistence(tr), va)
w = va[va.DIST2LAND > 0]
thr = choose_threshold(w.RI.to_numpy(), w.P.to_numpy())
r = evaluate(va, 'P', thr)
m = 'recall (last-24h rise < 30 kt)'
t = pd.DataFrame({'subset': r.subset, 'storms': r.storms, 'RI rows': r['RI rows'],
  'RI rows rise < 30 kt': r['RI rows rise < 30 kt'],
  'recall rise < 30 kt [95% CI]': [f'{e:.3f} [{lo:.3f}, {hi:.3f}]' for e, lo, hi in zip(r[m], r[m+' lo'], r[m+' hi'])],
  'boot used / skipped': [f'{u} / {s}' for u, s in zip(r['boot used'], r['boot skipped (no RI)'])]})
print(f'threshold {thr:.4f}'); print(t.to_string(index=False))
"
```

```
threshold 0.0816
                       subset  storms  RI rows  RI rows rise < 30 kt recall rise < 30 kt [95% CI] boot used / skipped
                Bay of Bengal      14       13                    12         0.667 [0.400, 1.000]             998 / 2
           North Indian Ocean      23       28                    24         0.792 [0.565, 1.000]            1000 / 0
All basins, <= 400 km of land     199      120                    96         0.594 [0.472, 0.720]            1000 / 0
                   All basins     323      539                   411         0.664 [0.607, 0.729]            1000 / 0
```

Climatology is a floor: one constant per basin; report its PR-AUC only, not episodes caught.

## Sat 14 Nov 2026 15:55: Persistence at ~1 false-alarm episode per storm (VALIDATION 2016-2019)

Command: `py -3.12 -m src.eval.baselines --fa-target 1`

Over-water rows only. Threshold = the distinct validation probability whose false-alarm episodes per storm (all basins, over water) is closest to the target; chosen and scored on the same validation rows (optimistic). 95% CI: 1,000 storm bootstrap resamples per subset, seed 0; resamples with no RI rows skipped. Test split not loaded.

**Persistence at ~1 false-alarm episode per storm: threshold 0.0190 (chosen on validation, all basins over water; achieved 1.003 false-alarm episodes per storm)**

```
                       subset  storms  RI episodes  RI rows rise < 30 kt  episodes caught    episodes caught %    median lead h false-alarm episodes / storm recall (last-24h rise < 30 kt) boot used / skipped
                Bay of Bengal      14            6                    12                6 100.0 [100.0, 100.0] 42.0 [0.0, 72.0]            0.64 [0.36, 0.93]           1.000 [1.000, 1.000]             998 / 2
           North Indian Ocean      23           11                    24               11 100.0 [100.0, 100.0] 18.0 [0.0, 54.0]            0.96 [0.65, 1.35]           1.000 [1.000, 1.000]            1000 / 0
All basins, <= 400 km of land     199           57                    96               55   96.5 [91.2, 100.0]   0.0 [0.0, 6.0]            1.21 [1.10, 1.33]           0.969 [0.919, 1.000]            1000 / 0
                   All basins     323          170                   411              164    96.5 [93.3, 99.3]  9.0 [6.0, 18.0]            1.00 [0.92, 1.08]           0.978 [0.960, 0.993]            1000 / 0
```

Note: at threshold 0.0190 the persistence alert is red on 86% of validation over-water rows (any storm whose wind fell by less than 10 kt in the last 24 h). False alarms per storm stay near 1 only because red rows merge into long runs. The high catch rate here reflects near-constant alerting, not skill. Check: `py -3.12 -c "...(w.P>=0.0190).mean()..."` (same fit as above) gave 0.86; the lowest DWIND_24 among red rows was -10 kt.

## Sat 14 Nov 2026 15:58: Persistence at 20% alert rate (VALIDATION 2016-2019)

Command: `py -3.12 -m src.eval.baselines --alert-rate 0.2`

Over-water rows only. Threshold = the distinct validation probability whose share of over-water rows on red (all basins) is closest to the target; chosen and scored on the same validation rows (optimistic). Alert rate = share of over-water rows on red; red hours / storm = 6 h x red rows / storms. 95% CI: 1,000 storm bootstrap resamples per subset, seed 0; resamples with no RI rows skipped. Test split not loaded.

**Persistence at 20% alert rate: threshold 0.1285 (chosen on validation, all basins over water; achieved alert rate 18.2% of over-water rows)**

```
                       subset  storms  RI episodes  RI rows rise < 30 kt  episodes caught      alert rate % red hours / storm episodes caught %  median lead h false-alarm episodes / storm recall (last-24h rise < 30 kt)               PR-AUC boot used / skipped
                Bay of Bengal      14            6                    12                0  12.8 [2.8, 22.5]   7.7 [1.3, 15.5]    0.0 [0.0, 0.0]       - [-, -]            0.29 [0.07, 0.57]           0.083 [0.000, 0.150] 0.129 [0.059, 0.263]             998 / 2
           North Indian Ocean      23           11                    24                1  14.3 [7.8, 21.0]  12.8 [6.5, 19.6]   9.1 [0.0, 33.3] 0.0 [0.0, 0.0]            0.43 [0.26, 0.65]           0.208 [0.000, 0.444] 0.154 [0.077, 0.258]            1000 / 0
All basins, <= 400 km of land     199           57                    96               16 15.9 [12.8, 19.1]   8.4 [6.6, 10.2] 28.1 [17.1, 38.9] 0.0 [0.0, 0.0]            0.36 [0.28, 0.44]           0.177 [0.108, 0.258] 0.121 [0.092, 0.160]            1000 / 0
                   All basins     323          170                   411               35 18.2 [16.3, 20.1] 18.1 [15.8, 20.5] 20.6 [14.9, 26.2] 0.0 [0.0, 6.0]            0.46 [0.40, 0.54]           0.217 [0.176, 0.260] 0.187 [0.164, 0.211]            1000 / 0
```

Note: persistence probabilities are steps because DWIND_24 comes in 5 kt steps, so exactly 20% is not reachable. The nearest reachable alert rates on validation over-water rows are 26.8% (threshold 0.1027, DWIND_24 >= 20 kt), 18.2% (threshold 0.1285, DWIND_24 >= 25 kt, used above) and 12.4% (threshold 0.1598, DWIND_24 >= 30 kt).

Decision: models are compared at the same alert rate (20% of over-water rows red) plus PR-AUC. False-alarm episodes per storm alone is not used, because long red runs hide constant alerting.

## Sat 14 Nov 2026 17:15: LightGBM RI model (VALIDATION 2016-2019)

Command: `.venv\Scripts\python -m src.models.ri_model`

Fit on train (1980-2015, all 57,244 rows). Early stopping on validation over-water log loss (patience 100): best iteration 177, val log loss 0.24845. Params: {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0, "deterministic": true, "force_col_wise": true}. No class weights, no resampling; missing USA_PRES / DPRES_24 left missing. Saved models/ri_lgbm.txt and models/ri_lgbm_features.json. Test split not loaded.

Threshold: threshold_for_alert_rate(val, 'P_LGBM', 0.182) = 0.1535, achieved alert rate 18.2% of over-water validation rows. Persistence re-run with the same functions at threshold 0.1285 (18.2%) and matches the earlier logged numbers. Threshold, early stopping and scoring all use the same validation rows, so these numbers are optimistic.

Comparison (point estimates):

```
                       subset  storms  RI episodes LGBM caught persist caught  LGBM hard recall  persist hard recall  LGBM PR-AUC  persist PR-AUC  LGBM Brier  persist Brier  LGBM FA eps/storm  persist FA eps/storm  LGBM alert %  persist alert %
                Bay of Bengal      14            6      1 of 6         0 of 6             0.417                0.083        0.374           0.129      0.0724         0.0863               0.50                  0.29          17.0             12.8
           North Indian Ocean      23           11     2 of 11        1 of 11             0.417                0.208        0.294           0.154      0.0671         0.0752               0.78                  0.43          20.5             14.3
All basins, <= 400 km of land     199           57    22 of 57       16 of 57             0.406                0.177        0.262           0.121      0.0580         0.0644               0.43                  0.36          15.9             15.9
                   All basins     323          170   69 of 170      35 of 170             0.528                0.217        0.420           0.187      0.0733         0.0903               0.59                  0.46          18.2             18.2
```

LightGBM with 95% intervals (1,000 storm bootstrap resamples per subset, seed 0; resamples with no RI rows skipped):

```
                       subset  storms  rows  RI rows  RI rows rise < 30 kt  RI episodes  episodes caught  false-alarm episodes               PR-AUC            precision               recall                   Brier recall (last-24h rise < 30 kt) episodes caught %  median lead h false-alarm episodes / storm      alert rate % red hours / storm boot used / skipped
                Bay of Bengal      14   141       13                    12            6                1                     7 0.374 [0.073, 0.667] 0.250 [0.000, 0.545] 0.462 [0.000, 0.889] 0.0724 [0.0243, 0.1244]           0.417 [0.000, 0.857]  16.7 [0.0, 66.7] 0.0 [0.0, 0.0]            0.50 [0.07, 1.00]  17.0 [6.9, 27.6]  10.3 [3.4, 17.6]             998 / 2
           North Indian Ocean      23   342       28                    24           11                2                    18 0.294 [0.094, 0.507] 0.200 [0.034, 0.386] 0.500 [0.133, 0.741] 0.0671 [0.0389, 0.0950]           0.417 [0.133, 0.650]  18.2 [0.0, 40.0] 3.0 [0.0, 6.0]            0.78 [0.43, 1.17] 20.5 [13.9, 28.0] 18.3 [10.4, 26.9]            1000 / 0
All basins, <= 400 km of land     199  1741      120                    96           57               22                    85 0.262 [0.174, 0.385] 0.217 [0.152, 0.300] 0.500 [0.392, 0.610] 0.0580 [0.0448, 0.0717]           0.406 [0.301, 0.532] 38.6 [25.8, 52.6] 0.0 [0.0, 0.0]            0.43 [0.32, 0.53] 15.9 [12.7, 19.0]   8.4 [6.3, 10.2]            1000 / 0
                   All basins     323  5333      539                   411          170               69                   191 0.420 [0.349, 0.492] 0.341 [0.289, 0.389] 0.614 [0.554, 0.672] 0.0733 [0.0660, 0.0811]           0.528 [0.467, 0.593] 40.6 [33.5, 48.3] 0.0 [0.0, 6.0]            0.59 [0.50, 0.68] 18.2 [16.3, 20.1] 18.0 [16.0, 20.2]            1000 / 0
```

Calibration (validation, over water): mean predicted RI 0.0825 vs actual RI rate 0.1011 (ratio 0.82). Reliability plot: docs/figures/reliability_val.png (10 quantile bins):

```
  bin  1: mean predicted 0.003  observed 0.004  rows 534
  bin  2: mean predicted 0.004  observed 0.013  rows 533
  bin  3: mean predicted 0.007  observed 0.013  rows 533
  bin  4: mean predicted 0.013  observed 0.015  rows 533
  bin  5: mean predicted 0.022  observed 0.022  rows 533
  bin  6: mean predicted 0.037  observed 0.047  rows 534
  bin  7: mean predicted 0.063  observed 0.094  rows 533
  bin  8: mean predicted 0.108  observed 0.146  rows 533
  bin  9: mean predicted 0.189  observed 0.240  rows 533
  bin 10: mean predicted 0.378  observed 0.416  rows 534
```

Leakage check (features and the timestamps they use):

```
USA_WIND   uses t         (USA_WIND at t)
USA_PRES   uses t         (USA_PRES at t)
DWIND_6    uses t-6h, t   (USA_WIND(t) - USA_WIND(t-6h))
DWIND_12   uses t-12h, t  (USA_WIND(t) - USA_WIND(t-12h))
DWIND_24   uses t-24h, t  (USA_WIND(t) - USA_WIND(t-24h))
DPRES_24   uses t-24h, t  (USA_PRES(t) - USA_PRES(t-24h))
SPEED_12   uses t-12h, t  (great-circle km from position at t-12h to t, / 12 h)
DIR_12     uses t-12h, t  (bearing from position at t-12h to t)
SPEED_24   uses t-24h, t  (great-circle km from position at t-24h to t, / 24 h)
DIR_24     uses t-24h, t  (bearing from position at t-24h to t)
LAT        uses t         (latitude at t)
DIST2LAND  uses t         (IBTrACS distance to land from the position at t)
MONTH_SIN  uses t         (month of ISO_TIME at t)
MONTH_COS  uses t         (month of ISO_TIME at t)
BASIN      uses t         (BASIN code of the row at t)
truncation test: 200 random validation rows rebuilt from data up to t plus a random fake t+24h row -> all 15 features identical
```


## Sat 14 Nov 2026 17:24: Platt calibration of LightGBM (VALIDATION 2016-2019)

Command: `.venv\Scripts\python -m src.models.ri_model --calibrate`

Loads models/ri_lgbm.txt (not retrained). Platt scaling p_cal = sigmoid(a * logit(p) + b), fitted on validation over-water rows. Honest check: out-of-fold Platt with 5 folds grouped by storm (GroupKFold on SID). Saved models/ri_calibration.json; plot docs/figures/reliability_val_calibrated.png. Platt is monotonic, so at the same 18.2% alert rate the red rows are identical (asserted); episodes caught and PR-AUC do not change, only probabilities and Brier. Test split not loaded.

**Platt calibration: a = 0.9588, b = 0.2060; calibrated alert threshold 0.1929 (alert rate 18.2%), same red rows as raw threshold 0.1535**

```
                             probabilities  mean predicted  actual RI rate  ratio  Brier
                              raw LightGBM          0.0825          0.1011 0.8161 0.0733
Platt, out-of-fold (5 storm-grouped folds)          0.1011          0.1011 1.0008 0.0728
      Platt, fitted on all val (in-sample)          0.1011          0.1011 1.0000 0.0727
```

Out-of-fold reliability bins:

```
  bin  1: mean predicted 0.005  observed 0.006
  bin  2: mean predicted 0.007  observed 0.011
  bin  3: mean predicted 0.011  observed 0.013
  bin  4: mean predicted 0.019  observed 0.015
  bin  5: mean predicted 0.032  observed 0.030
  bin  6: mean predicted 0.052  observed 0.045
  bin  7: mean predicted 0.084  observed 0.088
  bin  8: mean predicted 0.140  observed 0.148
  bin  9: mean predicted 0.233  observed 0.240
  bin 10: mean predicted 0.429  observed 0.414
```

evaluate() on calibrated probabilities (in-sample calibrator; 95% storm-bootstrap CI, seed 0):

```
                       subset  storms  RI episodes  episodes caught      alert rate %               PR-AUC                   Brier recall (last-24h rise < 30 kt) false-alarm episodes / storm boot used / skipped
                Bay of Bengal      14            6                1  17.0 [6.9, 27.6] 0.374 [0.073, 0.667] 0.0735 [0.0286, 0.1215]           0.417 [0.000, 0.857]            0.50 [0.07, 1.00]             998 / 2
           North Indian Ocean      23           11                2 20.5 [13.9, 28.0] 0.294 [0.094, 0.507] 0.0686 [0.0434, 0.0945]           0.417 [0.133, 0.650]            0.78 [0.43, 1.17]            1000 / 0
All basins, <= 400 km of land     199           57               22 15.9 [12.7, 19.0] 0.262 [0.174, 0.385] 0.0593 [0.0467, 0.0721]           0.406 [0.301, 0.532]            0.43 [0.32, 0.53]            1000 / 0
                   All basins     323          170               69 18.2 [16.3, 20.1] 0.420 [0.349, 0.492] 0.0727 [0.0656, 0.0803]           0.528 [0.467, 0.593]            0.59 [0.50, 0.68]            1000 / 0
```


## Sat 14 Nov 2026 18:09: three LightGBM models, North Indian Ocean scope (VALIDATION 2016-2019)

Command: `.venv\Scripts\python -m src.models.ri_model`

Scope set by the user's instruction of 14 Nov 2026 (North Indian Ocean, three models). CLAUDE.md on disk still says Bay of Bengal / one model / train on all basins; it was not edited.

Identical settings for all three: {"objective": "binary", "metric": "binary_logloss", "num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.03, "feature_fraction": 0.8, "seed": 0, "deterministic": true, "force_col_wise": true}, max 5000 rounds, early stopping (patience 100) on log loss over NI over-water validation rows. No class weights, no resampling; missing USA_PRES / DPRES_24 left missing. LEAVE-NI-OUT drops BASIN. Thresholds: threshold_for_alert_rate(NI validation, prob, 0.182) per model. Early stopping, thresholds and scores all use the same NI validation rows, so these numbers are optimistic. Test split not loaded.

Counts:

```
NI train (1980-2015, all rows):            rows  1,596  storms  136  RI rows   91
Bay of Bengal train (1980-2015, all rows): rows    989  storms   91  RI rows   61
NI validation, over water:                 rows    342  storms   23  RI rows   28
Bay of Bengal validation, over water:      rows    141  storms   14  RI rows   13
```

Training:

```
             train rows train storms train RI rows best iteration val log loss                        file
NI-ONLY            1596          136            91             49     0.233407       models/ri_lgbm_ni.txt
ALL-BASINS        57244         3028          3949            288     0.239494      models/ri_lgbm_all.txt
LEAVE-NI-OUT      55648         2900          3858             53     0.245885  models/ri_lgbm_leaveni.txt
```

NI over-water validation (point estimates). 'persistence (logged)' = Shaurya's run at threshold 0.1285, set on ALL-basin validation rows (NI alert rate 14.3%, not 18.2%); ROC AUC and mean predicted were not logged. 'persistence (re-run, NI 18.2%)' = same persistence model, threshold set on NI rows like the models:

```
                NI-ONLY ALL-BASINS LEAVE-NI-OUT persistence (re-run, NI 18.2%) persistence (logged)
PR-AUC            0.316      0.256        0.253                          0.154                0.154
ROC AUC           0.846      0.794        0.782                          0.739                    -
Brier            0.0671     0.0701       0.0700                         0.0752               0.0752
mean predicted   0.0742     0.0954       0.0999                         0.0753                    -
actual RI rate   0.0819     0.0819       0.0819                         0.0819               0.0819
alert rate %       18.1       18.1         18.1                           14.3                 14.3
caught          2 of 11    1 of 11      2 of 11                        1 of 11              1 of 11
hard recall       0.375      0.292        0.292                          0.208                0.208
FA eps/storm       0.96       0.96         0.91                           0.43                 0.43
```

Bay of Bengal subset (same thresholds as NI):

```
               NI-ONLY ALL-BASINS LEAVE-NI-OUT persistence (re-run, NI 18.2%) persistence (logged)
PR-AUC           0.376      0.327        0.347                          0.129                0.129
ROC AUC          0.817      0.749        0.737                          0.623                    -
Brier           0.0725     0.0747       0.0768                         0.0863               0.0863
mean predicted  0.0774     0.1024       0.1070                         0.0784                    -
actual RI rate  0.0922     0.0922       0.0922                         0.0922               0.0922
alert rate %      21.3       18.4         16.3                           12.8                 12.8
caught          1 of 6     1 of 6       1 of 6                         0 of 6               0 of 6
hard recall      0.500      0.417        0.417                          0.083                0.083
FA eps/storm      0.71       0.57         0.64                           0.29                 0.29
```

With 95% storm-bootstrap intervals (1,000 resamples per subset, seed 0; resamples with no RI rows skipped):

NI-ONLY (threshold 0.0999):
```
            subset  storms  rows  RI rows  RI episodes  episodes caught               PR-AUC                   Brier recall (last-24h rise < 30 kt) false-alarm episodes / storm      alert rate % boot used / skipped
     Bay of Bengal      14   141       13            6                1 0.376 [0.074, 0.609] 0.0725 [0.0205, 0.1297]           0.500 [0.000, 0.714]            0.71 [0.21, 1.36] 21.3 [10.1, 33.6]             998 / 2
North Indian Ocean      23   342       28           11                2 0.316 [0.102, 0.549] 0.0671 [0.0370, 0.0983]           0.375 [0.125, 0.579]            0.96 [0.52, 1.48] 18.1 [11.6, 25.8]            1000 / 0
```

ALL-BASINS (threshold 0.1843):
```
            subset  storms  rows  RI rows  RI episodes  episodes caught               PR-AUC                   Brier recall (last-24h rise < 30 kt) false-alarm episodes / storm      alert rate % boot used / skipped
     Bay of Bengal      14   141       13            6                1 0.327 [0.069, 0.653] 0.0747 [0.0268, 0.1258]           0.417 [0.000, 0.857]            0.57 [0.14, 1.07]  18.4 [7.9, 29.9]             998 / 2
North Indian Ocean      23   342       28           11                1 0.256 [0.084, 0.462] 0.0701 [0.0428, 0.0976]           0.292 [0.077, 0.500]            0.96 [0.57, 1.43] 18.1 [11.3, 25.2]            1000 / 0
```

LEAVE-NI-OUT (threshold 0.1860):
```
            subset  storms  rows  RI rows  RI episodes  episodes caught               PR-AUC                   Brier recall (last-24h rise < 30 kt) false-alarm episodes / storm      alert rate % boot used / skipped
     Bay of Bengal      14   141       13            6                1 0.347 [0.069, 0.592] 0.0768 [0.0268, 0.1327]           0.417 [0.000, 0.857]            0.64 [0.29, 1.07]  16.3 [8.2, 24.5]             998 / 2
North Indian Ocean      23   342       28           11                2 0.253 [0.103, 0.410] 0.0700 [0.0398, 0.1011]           0.292 [0.071, 0.533]            0.91 [0.57, 1.30] 18.1 [11.8, 24.7]            1000 / 0
```

PERSISTENCE (threshold 0.1285):
```
            subset  storms  rows  RI rows  RI episodes  episodes caught               PR-AUC                   Brier recall (last-24h rise < 30 kt) false-alarm episodes / storm     alert rate % boot used / skipped
     Bay of Bengal      14   141       13            6                0 0.129 [0.059, 0.263] 0.0863 [0.0199, 0.1637]           0.083 [0.000, 0.150]            0.29 [0.07, 0.57] 12.8 [2.8, 22.5]             998 / 2
North Indian Ocean      23   342       28           11                1 0.154 [0.077, 0.258] 0.0752 [0.0366, 0.1176]           0.208 [0.000, 0.444]            0.43 [0.26, 0.65] 14.3 [7.8, 21.0]            1000 / 0
```

**PRE-REGISTERED MAIN MODEL: NI-ONLY** (decided Sat 14 Nov 2026 18:09; pre-registered rule, not to be changed). NI-ONLY PR-AUC 0.3158 vs ALL-BASINS PR-AUC 0.2556 (NI over-water validation rows; ties go to NI-ONLY) -> MAIN = NI-ONLY. Copied models/ri_lgbm_ni.txt to models/ri_lgbm.txt.

Calibration: MAIN (NI-ONLY) on NI over-water validation: mean predicted RI 0.0742 vs actual RI rate 0.0819 (ratio 0.91). Reliability plot: docs/figures/reliability_val.png (5 quantile bins):

```
bin 1: mean predicted 0.018  observed 0.000
bin 2: mean predicted 0.021  observed 0.000
bin 3: mean predicted 0.029  observed 0.030
bin 4: mean predicted 0.058  observed 0.176
bin 5: mean predicted 0.244  observed 0.203
```

Leakage check (MAIN features and the timestamps they use):

```
USA_WIND   uses t         (USA_WIND at t)
USA_PRES   uses t         (USA_PRES at t)
DWIND_6    uses t-6h, t   (USA_WIND(t) - USA_WIND(t-6h))
DWIND_12   uses t-12h, t  (USA_WIND(t) - USA_WIND(t-12h))
DWIND_24   uses t-24h, t  (USA_WIND(t) - USA_WIND(t-24h))
DPRES_24   uses t-24h, t  (USA_PRES(t) - USA_PRES(t-24h))
SPEED_12   uses t-12h, t  (great-circle km from position at t-12h to t, / 12 h)
DIR_12     uses t-12h, t  (bearing from position at t-12h to t)
SPEED_24   uses t-24h, t  (great-circle km from position at t-24h to t, / 24 h)
DIR_24     uses t-24h, t  (bearing from position at t-24h to t)
LAT        uses t         (latitude at t)
DIST2LAND  uses t         (IBTrACS distance to land from the position at t)
MONTH_SIN  uses t         (month of ISO_TIME at t)
MONTH_COS  uses t         (month of ISO_TIME at t)
BASIN      uses t         (BASIN code of the row at t)
truncation test: 200 random NI validation rows rebuilt from data up to t plus a random fake t+24h row -> all 15 features identical
```


## Sat 14 Nov 2026: decision, alert threshold rule (logged before FROZEN)

Decision: Alert threshold rule: target 20% of over-water VALIDATION rows red, computed on BASIN == NI only. Applies to every model in the app and in evaluation. Persistence uses its closest achievable step on the same rows; every results table reports each model's actual alert rate. PR-AUC is the main threshold-free comparison.

## Sat 14 Nov 2026 18:25: Persistence, NI 20% rule (VALIDATION 2016-2019)

Command: `py -3.12 -m src.eval.baselines --alert-rate 0.2`

Over-water rows only. Threshold = the distinct validation probability whose share of over-water BASIN == NI rows on red is closest to the target (logged alert threshold rule); chosen and scored on the same validation rows (optimistic). Alert rate = share of over-water rows on red; red hours / storm = 6 h x red rows / storms. 95% CI: 1,000 storm bootstrap resamples per subset, seed 0; resamples with no RI rows skipped. Test split not loaded.

**Persistence, NI 20% rule: threshold 0.1285 (chosen on validation, BASIN == NI over water; actual alert rate 14.3% of NI over-water validation rows, the closest step persistence can reach)**

```
                       subset  storms  RI episodes  RI rows rise < 30 kt  episodes caught      alert rate % red hours / storm episodes caught %  median lead h false-alarm episodes / storm recall (last-24h rise < 30 kt)               PR-AUC boot used / skipped
                Bay of Bengal      14            6                    12                0  12.8 [2.8, 22.5]   7.7 [1.3, 15.5]    0.0 [0.0, 0.0]       - [-, -]            0.29 [0.07, 0.57]           0.083 [0.000, 0.150] 0.129 [0.059, 0.263]             998 / 2
           North Indian Ocean      23           11                    24                1  14.3 [7.8, 21.0]  12.8 [6.5, 19.6]   9.1 [0.0, 33.3] 0.0 [0.0, 0.0]            0.43 [0.26, 0.65]           0.208 [0.000, 0.444] 0.154 [0.077, 0.258]            1000 / 0
All basins, <= 400 km of land     199           57                    96               16 15.9 [12.8, 19.1]   8.4 [6.6, 10.2] 28.1 [17.1, 38.9] 0.0 [0.0, 0.0]            0.36 [0.28, 0.44]           0.177 [0.108, 0.258] 0.121 [0.092, 0.160]            1000 / 0
                   All basins     323          170                   411               35 18.2 [16.3, 20.1] 18.1 [15.8, 20.5] 20.6 [14.9, 26.2] 0.0 [0.0, 6.0]            0.46 [0.40, 0.54]           0.217 [0.176, 0.260] 0.187 [0.164, 0.211]            1000 / 0
```

Note: persistence steps on the 342 over-water NI validation rows are 38.9% (threshold 0.0816, DWIND_24 >= 15 kt), 26.6% (0.1027, >= 20 kt), 14.3% (0.1285, >= 25 kt, used: 5.7 points from 20% vs 6.6 for 26.6%) and 8.5% (0.1598, >= 30 kt). The threshold is the same as in the earlier all-basins 20% run, so all subset numbers are unchanged; only the rule and the reported actual alert rate (14.3% on NI) differ. Steps listed with: `py -3.12 -c "...np.unique(P) over NI over-water val rows, share P >= t..."` (persistence fitted on train, as in baselines.py).
