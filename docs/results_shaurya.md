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

## Sat 14 Nov 2026 18:58: Persistence, NI step at last-24h rise >= 20 kt (VALIDATION 2016-2019)

Command: `py -3.12 -m src.eval.baselines --min-rise 20`

Over-water rows only. Threshold = persistence probability at a last-24h wind rise of exactly 20 kt (red when DWIND_24 >= 20 kt); fixed by hand, not matched to a target rate; scored on validation rows (threshold choice also used validation: optimistic). Alert rate = share of over-water rows on red; red hours / storm = 6 h x red rows / storms. 95% CI: 1,000 storm bootstrap resamples per subset, seed 0; resamples with no RI rows skipped. Test split not loaded.

**Persistence, NI step at last-24h rise >= 20 kt: threshold 0.1027 (set by rule, BASIN == NI over water; actual alert rate 26.6% of NI over-water validation rows)**

```
            subset  storms  RI episodes  RI rows rise < 30 kt  episodes caught  missed episodes      alert rate % red hours / storm episodes caught %  median lead h false-alarm episodes / storm recall (last-24h rise < 30 kt)               PR-AUC boot used / skipped
     Bay of Bengal      14            6                    12                0                6  23.4 [9.9, 35.4]  14.1 [5.1, 24.9]    0.0 [0.0, 0.0]       - [-, -]            0.64 [0.21, 1.07]           0.167 [0.000, 0.286] 0.129 [0.059, 0.263]             998 / 2
North Indian Ocean      23           11                    24                2                9 26.6 [17.6, 36.1] 23.7 [14.1, 33.1]  18.2 [0.0, 40.0] 3.0 [0.0, 6.0]            0.83 [0.52, 1.17]           0.375 [0.143, 0.625] 0.154 [0.077, 0.258]            1000 / 0
```

### Persistence rows side by side (VALIDATION 2016-2019, over water, 95% storm-bootstrap intervals)

Row 1 command: `py -3.12 -m src.eval.baselines --alert-rate 0.2` (section "Persistence, NI 20% rule" above). Row 2 command: `py -3.12 -m src.eval.baselines --min-rise 20` (section just above). Alert rate in the threshold column is on NI over-water validation rows; the "alert rate %" column is per subset.

| Persistence row | Subset | Threshold (NI alert rate) | Alert rate % | Caught | Missed | Median lead h | Hard-case recall | False-alarm episodes / storm | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| rise >= 25 kt | Bay of Bengal | 0.1285 (14.3%) | 12.8 [2.8, 22.5] | 0 of 6 | 6 | - | 0.083 [0.000, 0.150] | 0.29 [0.07, 0.57] | 0.129 [0.059, 0.263] |
| rise >= 25 kt | North Indian Ocean | 0.1285 (14.3%) | 14.3 [7.8, 21.0] | 1 of 11 | 10 | 0.0 [0.0, 0.0] | 0.208 [0.000, 0.444] | 0.43 [0.26, 0.65] | 0.154 [0.077, 0.258] |
| rise >= 20 kt | Bay of Bengal | 0.1027 (26.6%) | 23.4 [9.9, 35.4] | 0 of 6 | 6 | - | 0.167 [0.000, 0.286] | 0.64 [0.21, 1.07] | 0.129 [0.059, 0.263] |
| rise >= 20 kt | North Indian Ocean | 0.1027 (26.6%) | 26.6 [17.6, 36.1] | 2 of 11 | 9 | 3.0 [0.0, 6.0] | 0.375 [0.143, 0.625] | 0.83 [0.52, 1.17] | 0.154 [0.077, 0.258] |

Hard-case recall = recall on RI rows where the last-24h wind rose < 30 kt. PR-AUC does not depend on the threshold, so it is the same in both rows. Bay of Bengal: 14 storms, 6 RI episodes, 12 hard-case RI rows; North Indian Ocean: 23 storms, 11 RI episodes, 24 hard-case RI rows. Bootstrap: 998 used / 2 skipped (Bay of Bengal), 1000 / 0 (North Indian Ocean).

Persistence rows at 14.3% and 26.6% bracket the model's 19.9% alert rate.

## Sat 14 Nov 2026: Independent leakage review (Shaurya)

Read src/data/features.py, src/data/split.py and src/data/clean.py (unchanged since commit dae144f), checked their outputs on data/processed, and recomputed 3 random validation rows by hand. test.parquet was not read.

**(a) Lags and target matched by timestamp, same SID, gaps asserted: yes.** `_merge_offset` (features.py:26-42) joins on `SID` plus `ISO_TIME` shifted by the offset (line 36-37, `validate="one_to_one"`) and asserts the matched gap (line 41). It is used for t-6h, t-12h and t-24h (lines 68-70) and for t+24h (line 71). No `shift()` or row positions are used. Minor point: because the join key is the exact shifted time, the gap assertion at line 41 cannot fail by construction. It documents the rule rather than catching errors. The join itself is the real protection.

**(b) No feature uses information after t: yes.** Features (FEATURES, features.py:14-18) use only the row at t and rows at t-6/-12/-24 h: wind and pressure at t, their changes (lines 76-78), motion from past positions (79-82), LAT and DIST2LAND at t, month of t (84-86), and BASIN at t (87). The future columns USA_WIND_P24 and DWIND_FUT_24 (line 23, "NEVER use as features") feed only the target (lines 90-91). Two limits, not leakage:
- Rule 7 (line 74) keeps only rows that have a t+24h row. That is selection by a future fact, which is needed for the label. So scored rows exclude each storm's last 24 h as a tropical storm, while the app forecasts on those rows too.
- IBTrACS best-track values are revised after the season. The value "at t" is not exactly what a forecaster had in real time.

**(c) No SID in two splits: yes.** split.py assigns by SEASON (line 19), asserts SEASON is constant per SID (line 18) and that every row gets a split (line 21), and asserts the splits are disjoint (lines 25-31, called at line 54). Checked on files: train has 3,028 SIDs and val has 326, with 0 shared. 0 val or train SIDs have SEASON >= 2020 rows in tracks.parquet. Minor point: Southern Hemisphere SEASON crosses calendar years. Val storms run from 2015-08-01 to 2019-12-28 in calendar time, and 5 test-season storms (3 SI, 2 SP) start before 2020-01-01. The SIDs are still disjoint; this does not affect North Indian Ocean storms, whose SEASON is the calendar year.

**(d) Only main track, 00/06/12/18 UTC, NATURE TS: yes.** clean.py filters TRACK_TYPE == "main" (line 68), hour in 00/06/12/18 with minute 0 (lines 72-73) and NATURE == "TS" (line 81), and asserts no duplicate SID/time (line 85). Checked on tracks.parquet: 103,228 rows, all TRACK_TYPE main, all NATURE TS, hours {0, 6, 12, 18}, minutes and seconds 0, no missing USA_WIND, no duplicate SID/time, SEASON constant within every SID.

**Hand recomputation: 54 of 54 values match.** 3 random validation rows (seed 2026) were recomputed from tracks.parquet by exact (SID, time) lookups, written independently of features.py: 15 features plus USA_WIND_P24, DWIND_FUT_24 and RI per row. The rows are HECTOR 2018213N12245 2018-08-05 06:00 (EP), CALEB 2017081S10096 2017-03-26 12:00 (SI) and ELLA 2017127S18184 2017-05-12 00:00 (SP). All 3 have RI = 0 and none is in NI, so this sample does not exercise an RI = 1 or NI row. Full table: `py -3.12 -m src.eval.leakage_check`.

**Problems found: none that leak.** The limits to state in the write-up are the rule-7 selection, best-track hindsight and the Southern Hemisphere season overlap.
