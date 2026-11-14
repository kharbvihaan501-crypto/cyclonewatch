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
