# Test results: TEST 2020+

Run: Sat 14 Nov 2026 19:30:50  
Command: `.venv\Scripts\python -m src.eval.test_run`  
Git commit: `896e88b0acf96d57bf9885e9be318f666fa06316`  
Freeze record: `docs/freeze.md`

## 1. Model files checked against docs/freeze.md (SHA-256)

```
OK       models/ri_lgbm.txt  6db069f53528a031713987a0c3e8c8db65f3578a2cf33a7357046ca2139bf1bf
OK       models/ri_lgbm_all.txt  924468978d5ea53ac579cd50d16d8aaae46f12037814c0fa1679899d683dbcdb
OK       models/ri_lgbm_features.json  bd8aa9f7c66ebb79bd68760bb3987547eebe939e605a758a270a7e33ce4045a7
OK       models/ri_lgbm_leaveni.txt  32a23e3f6755a0f5fe50932b9721d3028b85f9bdd5dbc5d51e8a217aca03ad8c
OK       models/ri_lgbm_ni.txt  6db069f53528a031713987a0c3e8c8db65f3578a2cf33a7357046ca2139bf1bf
```

## 2. Frozen thresholds (no threshold computed on test data)

Each red threshold was reproduced from validation data with the same call the freeze used and checked to round to the freeze.md value; the exact value is used. Amber = red / 2. Persistence = logistic regression on DWIND_24 fitted on train 1980-2015 (src/eval/baselines.py), at its two frozen thresholds.

```
NI-ONLY (MAIN)   red 0.088325 (freeze.md 0.088325)  amber from 0.044163
ALL-BASINS       red 0.177467 (freeze.md 0.177467)  amber from 0.088733
LEAVE-NI-OUT     red 0.175196 (freeze.md 0.175196)  amber from 0.087598
persistence @ 0.1285 (DWIND_24 >= 25 kt) red 0.128547 (freeze.md 0.1285)
persistence @ 0.1027 (DWIND_24 >= 20 kt) red 0.102692 (freeze.md 0.1027)
```

## 3. Metrics

Over-water rows only (DIST2LAND > 0 at t). evaluate() from src/eval/metrics.py unchanged: episodes, lead time, false-alarm episodes, PR-AUC, Brier, 95% intervals from 1,000 bootstrap resamples of whole storms per subset (seed 0; resamples with no RI rows skipped, count shown). ROC AUC, mean predicted and actual RI rate: own storm bootstrap with the same rule (1,000 resamples, seed 0). Caught = red at the episode's first row. Lead time = hours the alert was red, unbroken, before the first RI row (src/eval/episodes.py). Hard-case recall = recall on RI rows whose last-24h wind rise was < 30 kt.

### North Indian Ocean (BASIN NI)

```
                                           MAIN (NI-ONLY)               ALL-BASINS             LEAVE-NI-OUT persistence @ 0.1285 (DWIND_24 >= 25 kt) persistence @ 0.1027 (DWIND_24 >= 20 kt)
storms                                                 26                       26                       26                                       26                                       26
rows                                                  276                      276                      276                                      276                                      276
RI rows                                                35                       35                       35                                       35                                       35
RI episodes                                             9                        9                        9                                        9                                        9
episodes caught (X of Y)                           5 of 9                   4 of 9                   4 of 9                                   0 of 9                                   2 of 9
episodes caught %                       55.6 [25.0, 85.7]        44.4 [14.3, 71.4]        44.4 [14.3, 71.4]                           0.0 [0.0, 0.0]                         22.2 [0.0, 57.1]
median lead h                             0.0 [0.0, 12.0]          0.0 [0.0, 12.0]           0.0 [0.0, 6.0]                                 - [-, -]                           3.0 [0.0, 6.0]
recall (last-24h rise < 30 kt)       0.708 [0.474, 0.938]     0.583 [0.294, 0.852]     0.542 [0.286, 0.786]                     0.250 [0.059, 0.444]                     0.417 [0.083, 0.760]
false-alarm episodes / storm            0.46 [0.19, 0.77]        0.38 [0.15, 0.69]        0.54 [0.27, 0.85]                        0.54 [0.31, 0.81]                        0.54 [0.35, 0.77]
alert rate %                            21.4 [12.4, 31.7]         17.0 [9.8, 25.6]        16.7 [10.5, 23.3]                        23.2 [14.2, 33.8]                        30.1 [19.7, 42.1]
PR-AUC                               0.499 [0.208, 0.805]     0.487 [0.196, 0.779]     0.454 [0.151, 0.718]                     0.221 [0.085, 0.379]                     0.221 [0.085, 0.379]
ROC AUC                              0.886 [0.793, 0.960]     0.858 [0.739, 0.958]     0.815 [0.657, 0.934]                     0.706 [0.535, 0.842]                     0.706 [0.535, 0.842]
Brier                             0.0933 [0.0429, 0.1577]  0.0868 [0.0426, 0.1404]  0.0927 [0.0461, 0.1510]                  0.1144 [0.0549, 0.1872]                  0.1144 [0.0549, 0.1872]
mean predicted                    0.0673 [0.0476, 0.0886]  0.0871 [0.0658, 0.1122]  0.0948 [0.0796, 0.1123]                  0.0920 [0.0725, 0.1151]                  0.0920 [0.0725, 0.1151]
actual RI rate                    0.1268 [0.0505, 0.2163]  0.1268 [0.0505, 0.2163]  0.1268 [0.0505, 0.2163]                  0.1268 [0.0505, 0.2163]                  0.1268 [0.0505, 0.2163]
boot used / skipped (evaluate)                    999 / 1                  999 / 1                  999 / 1                                  999 / 1                                  999 / 1
boot used / skipped (ROC, means)                  999 / 1                  999 / 1                  999 / 1                                  999 / 1                                  999 / 1
```

### Bay of Bengal (BASIN NI, SUBBASIN BB)

```
                                           MAIN (NI-ONLY)               ALL-BASINS             LEAVE-NI-OUT persistence @ 0.1285 (DWIND_24 >= 25 kt) persistence @ 0.1027 (DWIND_24 >= 20 kt)
storms                                                 19                       19                       19                                       19                                       19
rows                                                  164                      164                      164                                      164                                      164
RI rows                                                18                       18                       18                                       18                                       18
RI episodes                                             3                        3                        3                                        3                                        3
episodes caught (X of Y)                           1 of 3                   1 of 3                   1 of 3                                   0 of 3                                   1 of 3
episodes caught %                       33.3 [0.0, 100.0]        33.3 [0.0, 100.0]        33.3 [0.0, 100.0]                           0.0 [0.0, 0.0]                        33.3 [0.0, 100.0]
median lead h                              0.0 [0.0, 0.0]           0.0 [0.0, 0.0]           0.0 [0.0, 0.0]                                 - [-, -]                           0.0 [0.0, 0.0]
recall (last-24h rise < 30 kt)       0.700 [0.333, 1.000]     0.600 [0.333, 1.000]     0.600 [0.333, 1.000]                     0.300 [0.000, 0.667]                     0.400 [0.000, 1.000]
false-alarm episodes / storm            0.53 [0.21, 0.90]        0.32 [0.05, 0.63]        0.53 [0.20, 0.95]                        0.42 [0.16, 0.74]                        0.42 [0.21, 0.63]
alert rate %                             18.3 [8.4, 30.1]         15.9 [6.8, 25.2]         15.9 [8.1, 23.5]                         20.1 [9.5, 32.7]                        26.8 [15.0, 40.1]
PR-AUC                               0.615 [0.138, 0.967]     0.497 [0.056, 0.843]     0.492 [0.054, 0.823]                     0.310 [0.042, 0.565]                     0.310 [0.042, 0.565]
ROC AUC                              0.886 [0.622, 0.998]     0.862 [0.621, 0.985]     0.847 [0.574, 0.983]                     0.794 [0.542, 0.932]                     0.794 [0.542, 0.932]
Brier                             0.0746 [0.0188, 0.1578]  0.0748 [0.0253, 0.1462]  0.0775 [0.0265, 0.1533]                  0.0948 [0.0253, 0.1886]                  0.0948 [0.0253, 0.1886]
mean predicted                    0.0632 [0.0395, 0.0910]  0.0910 [0.0656, 0.1168]  0.0979 [0.0789, 0.1179]                  0.0879 [0.0661, 0.1158]                  0.0879 [0.0661, 0.1158]
actual RI rate                    0.1098 [0.0222, 0.2334]  0.1098 [0.0222, 0.2334]  0.1098 [0.0222, 0.2334]                  0.1098 [0.0222, 0.2334]                  0.1098 [0.0222, 0.2334]
boot used / skipped (evaluate)                   955 / 45                 955 / 45                 955 / 45                                 955 / 45                                 955 / 45
boot used / skipped (ROC, means)                 955 / 45                 955 / 45                 955 / 45                                 955 / 45                                 955 / 45
```

### Secondary check: All basins, <= 400 km of land (MAIN, ALL-BASINS, persistence)

MAIN is an NI-only model applied here to every basin; this is a secondary check, not a headline.

```
                                           MAIN (NI-ONLY)               ALL-BASINS persistence @ 0.1285 (DWIND_24 >= 25 kt) persistence @ 0.1027 (DWIND_24 >= 20 kt)
storms                                                268                      268                                      268                                      268
rows                                                 2290                     2290                                     2290                                     2290
RI rows                                               213                      213                                      213                                      213
RI episodes                                            89                       89                                       89                                       89
episodes caught (X of Y)                         23 of 89                 35 of 89                                 23 of 89                                 38 of 89
episodes caught %                       25.8 [16.7, 35.3]        39.3 [29.4, 48.7]                        25.8 [16.3, 34.7]                        42.7 [33.3, 52.6]
median lead h                              0.0 [0.0, 0.0]           0.0 [0.0, 0.0]                           0.0 [0.0, 0.0]                           0.0 [0.0, 0.0]
recall (last-24h rise < 30 kt)       0.259 [0.161, 0.359]     0.377 [0.278, 0.469]                     0.198 [0.131, 0.267]                     0.432 [0.348, 0.516]
false-alarm episodes / storm            0.32 [0.24, 0.40]        0.32 [0.25, 0.40]                        0.46 [0.39, 0.54]                        0.61 [0.52, 0.71]
alert rate %                              9.9 [7.7, 12.3]        12.8 [10.6, 15.0]                        20.7 [18.0, 23.7]                        30.3 [27.2, 33.5]
PR-AUC                               0.260 [0.181, 0.348]     0.336 [0.258, 0.425]                     0.150 [0.119, 0.186]                     0.150 [0.119, 0.186]
ROC AUC                              0.726 [0.656, 0.786]     0.805 [0.762, 0.843]                     0.687 [0.634, 0.734]                     0.687 [0.634, 0.734]
Brier                             0.0809 [0.0661, 0.0981]  0.0722 [0.0597, 0.0861]                  0.0872 [0.0719, 0.1048]                  0.0872 [0.0719, 0.1048]
mean predicted                    0.0449 [0.0405, 0.0494]  0.0760 [0.0683, 0.0836]                  0.0849 [0.0788, 0.0907]                  0.0849 [0.0788, 0.0907]
actual RI rate                    0.0930 [0.0746, 0.1124]  0.0930 [0.0746, 0.1124]                  0.0930 [0.0746, 0.1124]                  0.0930 [0.0746, 0.1124]
boot used / skipped (evaluate)                   1000 / 0                 1000 / 0                                 1000 / 0                                 1000 / 0
boot used / skipped (ROC, means)                 1000 / 0                 1000 / 0                                 1000 / 0                                 1000 / 0
```

## 4. MAIN: every NI RI episode

Red threshold 0.088325. FIRST RED h = hours from the first RI row to the first red row: negative = red before the episode (equals minus the lead time; the run start is never earlier than the row after the storm's previous episode), 0 = red exactly at the first row, positive = first red row later inside the episode, blank = never red inside the episode.

```
    NAME  SEASON SUBBASIN            START  ROWS  RED ROWS ANY RED RED AT FIRST ROW FIRST RED h  MAX P IN EPISODE
  AMPHAN    2020       BB 2020-05-16 12:00     5         5     yes              yes          -0             0.441
 NISARGA    2020       AS 2020-06-02 06:00     1         1     yes              yes          -0             0.135
    GATI    2020       AS 2020-11-21 18:00     1         0      no               no                         0.041
 TAUKTAE    2021       AS 2021-05-15 00:00     6         6     yes              yes         -12             0.247
   MOCHA    2023       BB 2023-05-10 18:00    10         8     yes               no          +6             0.515
BIPARJOY    2023       AS 2023-06-06 00:00     1         1     yes              yes          -0             0.172
BIPARJOY    2023       AS 2023-06-09 00:00     4         1     yes               no         +18             0.263
     TEJ    2023       AS 2023-10-20 18:00     4         4     yes              yes          -0             0.209
  HAMOON    2023       BB 2023-10-23 00:00     3         1     yes               no          +6             0.221
```

```
episodes with an alert at the first row:          5 of 9
episodes with an alert at any row in the episode: 8 of 9
episodes never alerted:                           1 of 9
```

## 5. Casebook storms in the test rows (MAIN)

Every 6-hourly row of the storm in the test split. Rows over land are shown but not scored. Alert: red >= 0.088325, amber >= 0.044163, else green.

### AMPHAN 2020 (SID 2020136N10088)

16 test rows (16 over water); over-water RI episodes 1, caught at the first row 1; peak USA_WIND 145 kt.

```
        ISO_TIME SUBBASIN  USA_WIND  DWIND_24  DIST2LAND  MAIN P ALERT  RI
2020-05-16 12:00       BB        50        20        581   0.164   red   1
2020-05-16 18:00       BB        55        25        586   0.214   red   1
2020-05-17 00:00       BB        60        25        603   0.397   red   1
2020-05-17 06:00       BB        70        30        630   0.429   red   1
2020-05-17 12:00       BB        80        30        654   0.441   red   1
2020-05-17 18:00       BB       120        65        636   0.047 amber   0
2020-05-18 00:00       BB       130        70        582   0.047 amber   0
2020-05-18 06:00       BB       140        70        551   0.038 green   0
2020-05-18 12:00       BB       145        65        520   0.053 amber   0
2020-05-18 18:00       BB       140        20        464   0.040 green   0
2020-05-19 00:00       BB       125        -5        416   0.045 amber   0
2020-05-19 06:00       BB       110       -30        361   0.032 green   0
2020-05-19 12:00       BB       105       -40        298   0.033 green   0
2020-05-19 18:00       BB       105       -35        197   0.033 green   0
2020-05-20 00:00       BB       100       -25        145   0.031 green   0
2020-05-20 06:00       BB        95       -15        104   0.031 green   0
```

### YAAS 2021 (SID 2021143N15090)

7 test rows (7 over water); over-water RI episodes 0, caught at the first row 0; peak USA_WIND 65 kt.

```
        ISO_TIME SUBBASIN  USA_WIND  DWIND_24  DIST2LAND  MAIN P ALERT  RI
2021-05-24 06:00       BB        35        10        485   0.025 green   0
2021-05-24 12:00       BB        40        10        445   0.107   red   0
2021-05-25 00:00       BB        55        20        321   0.054 amber   0
2021-05-25 06:00       BB        55        20        229   0.028 green   0
2021-05-25 12:00       BB        65        25        180   0.065 amber   0
2021-05-25 18:00       BB        65        15         94   0.023 green   0
2021-05-26 00:00       BB        60         5         63   0.026 green   0
```

### GULAB:SHAHEEN-GU 2021 (SID 2021267N18094)

20 test rows (20 over water); over-water RI episodes 0, caught at the first row 0; peak USA_WIND 70 kt.

```
        ISO_TIME SUBBASIN  USA_WIND  DWIND_24  DIST2LAND  MAIN P ALERT  RI
2021-09-24 18:00       BB        30         5        321   0.022 green   0
2021-09-25 00:00       BB        35         5        376   0.027 green   0
2021-09-25 06:00       BB        35         5        321   0.021 green   0
2021-09-25 12:00       BB        40        10        291   0.024 green   0
2021-09-25 18:00       BB        40        10        268   0.025 green   0
2021-09-26 00:00       BB        35         0        212   0.021 green   0
2021-09-26 06:00       BB        35         0        138   0.022 green   0
2021-09-30 06:00       AS        30         5         69   0.018 green   0
2021-09-30 12:00       AS        30         5        108   0.019 green   0
2021-09-30 18:00       AS        35        10        150   0.021 green   0
2021-10-01 00:00       AS        40        10        227   0.032 green   0
2021-10-01 06:00       AS        50        20        223   0.081 amber   0
2021-10-01 12:00       AS        55        25        194   0.043 green   0
2021-10-01 18:00       AS        55        20        168   0.033 green   0
2021-10-02 00:00       AS        60        20        134   0.041 green   0
2021-10-02 06:00       AS        65        15        101   0.034 green   0
2021-10-02 12:00       AS        65        10         89   0.019 green   0
2021-10-02 18:00       AS        70        15        101   0.032 green   0
2021-10-03 00:00       AS        65         5        105   0.019 green   0
2021-10-03 06:00       AS        70         5         67   0.032 green   0
```

### JAWAD 2021 (SID 2021335N08096)

7 test rows (7 over water); over-water RI episodes 0, caught at the first row 0; peak USA_WIND 35 kt.

```
        ISO_TIME SUBBASIN  USA_WIND  DWIND_24  DIST2LAND  MAIN P ALERT  RI
2021-12-02 18:00       BB        30         5        729   0.048 amber   0
2021-12-03 00:00       BB        35        10        536   0.027 green   0
2021-12-03 06:00       BB        35        10        430   0.017 green   0
2021-12-03 12:00       BB        35        10        349   0.018 green   0
2021-12-03 18:00       BB        35         5        296   0.018 green   0
2021-12-04 00:00       BB        35         0        247   0.018 green   0
2021-12-04 06:00       BB        35         0        223   0.018 green   0
```

### DANA 2024 (SID 2024295N15092)

7 test rows (7 over water); over-water RI episodes 0, caught at the first row 0; peak USA_WIND 65 kt.

```
        ISO_TIME SUBBASIN  USA_WIND  DWIND_24  DIST2LAND  MAIN P ALERT  RI
2024-10-23 06:00       BB        40        15        479   0.196   red   0
2024-10-23 12:00       BB        50        25        407   0.549   red   0
2024-10-23 18:00       BB        55        25        329   0.318   red   0
2024-10-24 00:00       BB        60        25        238   0.076 amber   0
2024-10-24 06:00       BB        65        25        160   0.063 amber   0
2024-10-24 12:00       BB        65        15        100   0.018 green   0
2024-10-24 18:00       BB        60         5         23   0.019 green   0
```

MONTHA (2025) is not in the test set: its data is provisional (TRACK_TYPE US-PROVISIONAL), so cleaning rule 2 dropped all its rows (it is listed under rule 8). It is not scored.

## 6. Files

`data/processed/predictions.parquet`: 276 over-water NI rows (SID, NAME, SEASON, SUBBASIN, ISO_TIME, LAT, LON, USA_WIND, DIST2LAND, P_MAIN, ALERT, P_PERSISTENCE, RI). P_PERSISTENCE = persistence probability (thresholds above).

