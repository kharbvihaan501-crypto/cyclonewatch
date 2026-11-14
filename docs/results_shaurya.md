# Results (Shaurya)

Every number below is followed by the command that produced it. Validation split only
(2016-2019); the test split is not loaded.

## Sat 14 Nov 2026: decision, alert threshold rule (logged before FROZEN)

Decision: Alert threshold rule: 20% of over-water VALIDATION rows red, computed on BASIN == NI only, for every model in the app and in evaluation. Persistence uses its closest achievable step on the same rows; every table reports each model's actual alert rate. PR-AUC is the main threshold-free comparison.

## Sat 14 Nov 2026 18:46: Persistence, NI 20% rule (VALIDATION 2016-2019)

Command: `py -3.12 -m src.eval.baselines --alert-rate 0.2`

Over-water rows only. Threshold = the distinct validation probability whose share of over-water BASIN == NI rows on red is closest to the target (logged alert threshold rule); chosen and scored on the same validation rows (optimistic). Alert rate = share of over-water rows on red; red hours / storm = 6 h x red rows / storms. 95% CI: 1,000 storm bootstrap resamples per subset, seed 0; resamples with no RI rows skipped. Test split not loaded.

**Persistence, NI 20% rule: threshold 0.1285 (chosen on validation, BASIN == NI over water; actual alert rate 14.3% of NI over-water validation rows, the closest step persistence can reach)**

```
                       subset  storms  RI episodes  RI rows rise < 30 kt  episodes caught  missed episodes      alert rate % red hours / storm episodes caught %  median lead h false-alarm episodes / storm recall (last-24h rise < 30 kt)               PR-AUC boot used / skipped
                Bay of Bengal      14            6                    12                0                6  12.8 [2.8, 22.5]   7.7 [1.3, 15.5]    0.0 [0.0, 0.0]       - [-, -]            0.29 [0.07, 0.57]           0.083 [0.000, 0.150] 0.129 [0.059, 0.263]             998 / 2
           North Indian Ocean      23           11                    24                1               10  14.3 [7.8, 21.0]  12.8 [6.5, 19.6]   9.1 [0.0, 33.3] 0.0 [0.0, 0.0]            0.43 [0.26, 0.65]           0.208 [0.000, 0.444] 0.154 [0.077, 0.258]            1000 / 0
All basins, <= 400 km of land     199           57                    96               16               41 15.9 [12.8, 19.1]   8.4 [6.6, 10.2] 28.1 [17.1, 38.9] 0.0 [0.0, 0.0]            0.36 [0.28, 0.44]           0.177 [0.108, 0.258] 0.121 [0.092, 0.160]            1000 / 0
                   All basins     323          170                   411               35              135 18.2 [16.3, 20.1] 18.1 [15.8, 20.5] 20.6 [14.9, 26.2] 0.0 [0.0, 6.0]            0.46 [0.40, 0.54]           0.217 [0.176, 0.260] 0.187 [0.164, 0.211]            1000 / 0
```

Missed episodes = RI episodes with the alert not red at their first row (RI episodes minus episodes caught). Median lead time is over caught episodes only ("-" when none were caught).

Why 14.3% and not 20%: persistence probabilities move in steps because DWIND_24 comes in 5 kt steps. On the 342 over-water NI validation rows the reachable alert rates are 38.9% (threshold 0.0816, DWIND_24 >= 15 kt), 26.6% (0.1027, >= 20 kt), 14.3% (0.1285, >= 25 kt; used, 5.7 points from 20% vs 6.6 for 26.6%) and 8.5% (0.1598, >= 30 kt).

Command for the steps:
```
py -3.12 -c "
import numpy as np, pandas as pd
from src.eval.baselines import *
tr=pd.read_parquet(TRAIN_PATH); va=pd.read_parquet(VAL_PATH)
w=va[(va.DIST2LAND>0)&(va.BASIN=='NI')].copy(); w['P']=predict_persistence(fit_persistence(tr),w)
print('NI over-water val rows', len(w))
for t in np.unique(w.P):
    s=(w.P>=t).mean()
    if 0.08<s<0.4: print(round(t,4), round(100*s,1), 'DWIND_24 >=', w.loc[w.P>=t,'DWIND_24'].min())
"
```
