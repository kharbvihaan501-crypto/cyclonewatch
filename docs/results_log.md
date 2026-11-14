\# Results Log



\## Sat 14 Nov 2026s


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

