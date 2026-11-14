# Results log

Every number in the write-up comes from here, with the command that produced it.
Run all commands from the project root with the project venv (`.venv\Scripts\python`).

## 2026-11-14 — Data cleaning counts (rules 2-5, 8)

Command:

```bash
python -m src.data.clean
```

Input: `data/raw/ibtracs.since1980.list.v04r01.csv` (units row skipped, `keep_default_na=False`).
Output: `data/processed/tracks.parquet` (103,228 rows, 4,256 storms).

| Step | Rows after | Storms after | Rows removed | Storms removed |
|---|---:|---:|---:|---:|
| raw (units row skipped) | 312,394 | 5,017 | | |
| rule 2: `TRACK_TYPE == "main"` | 300,439 | 4,790 | 11,955 | 227 |
| rule 3: 00/06/12/18 UTC only | 150,812 | 4,786 | 149,627 | 4 |
| rule 4: `USA_WIND` present (no WMO fallback) | 127,637 | 4,363 | 23,175 | 423 |
| rule 5: `NATURE == "TS"` | 103,228 | 4,256 | 24,409 | 107 |

Storms per BASIN, raw vs kept:

| BASIN | raw | kept |
|---|---:|---:|
| EP | 1162 | 929 |
| NA | 753 | 721 |
| NI | 474 | 210 |
| SA | 3 | 1 |
| SI | 918 | 750 |
| SP | 549 | 453 |
| WP | 1541 | 1301 |

Bay of Bengal (BASIN NI, SUBBASIN BB) storms: raw 373, kept 146.
Kept BB storms by split (counted by `SEASON`): 1980-2015 = 109, 2016-2019 = 16, 2020+ = 21.

Rule 8 (possibly provisional US data): 176 storms are flagged, by `SEASON` 2025 = 69, 2026 = 104, 2027 = 3.
The flag covers the last two seasons in the raw file plus any storm with `PROVISIONAL` or
`US-PROVISIONAL` track rows. All 176 are removed entirely by rule 2 (kept rows = 0), so none are in
`tracks.parquet`. The full list is printed by the command above.
Bay of Bengal storms in that list include MONTHA, SENYAR and DITWAH (late 2025), so the BB test
years effectively end in 2024. Retained 2024-2025 `main` storms may still be revised; the file does
not say.

Other checks on `tracks.parquet`: AMPHAN (2020) present, 24 rows, peak `USA_WIND` 145 kt;
`USA_PRES` is missing in 40.9% of rows; no missing `USA_WIND`, `LAT`, `LON` or `DIST2LAND`.

### Bay of Bengal storms among the 176 dropped for provisional data (added 2026-11-14)

Source: the rule 8 table printed by `python -m src.data.clean`, filtered to BASIN NI, SUBBASIN BB.
7 of the 176 storms are Bay of Bengal. "Dropped rows" is the number of rows removed by rule 2
(`PROV_ROWS`); each storm had all of its raw rows dropped, so none remain in `tracks.parquet`.

| SID | Name | SEASON | First row (UTC) | Last row (UTC) | Dropped rows |
|---|---|---:|---|---|---:|
| 2025274N16087 | UNNAMED | 2025 | 2025-10-01 06:00 | 2025-10-03 06:00 | 17 |
| 2025298N11089 | MONTHA | 2025 | 2025-10-25 00:00 | 2025-10-29 18:00 | 39 |
| 2025329N04100 | SENYAR | 2025 | 2025-11-25 00:00 | 2025-11-30 18:00 | 47 |
| 2025331N06083 | DITWAH | 2025 | 2025-11-26 18:00 | 2025-12-02 18:00 | 49 |
| 2026266N18085 | UNNAMED | 2026 | 2026-09-22 18:00 | 2026-09-24 00:00 | 11 |
| 2026299N16089 | ARNAB | 2026 | 2026-10-26 00:00 | 2026-10-29 06:00 | 27 |
| 2026305N12087 | MURASU | 2026 | 2026-11-01 00:00 | 2026-11-03 06:00 | 19 |

Total dropped rows for these 7 storms: 209.

## 2026-11-14 — Features, target and splits (rules 6-7, split rules)

Command:

```bash
python -m src.data.split
```

Input: `data/processed/tracks.parquet`. Output: `data/processed/train.parquet`, `val.parquet`, `test.parquet`.
Lags (t-6h, t-12h, t-24h) and target (t+24h) matched by timestamp within SID (merge on SID and
shifted ISO_TIME), gaps asserted exactly 6/12/24 h. RI = USA_WIND(t+24h) - USA_WIND(t) >= 30 kt.

| Step | Rows | Storms | Rows removed |
|---|---:|---:|---:|
| tracks.parquet | 103,228 | 4,256 | |
| rule 7: has t-24h and t+24h in same SID | 69,091 | 3,788 | 34,137 |

Missing values among the 69,091 kept rows (left missing, not filled): USA_PRES 29,596;
DPRES_24 29,631; DWIND_6 129; DWIND_12 198; SPEED_12 198; DIR_12 198.

Splits are by storm `SEASON` (one split per SID; asserted no SID in two splits).
Southern Hemisphere seasons start mid-year, so e.g. SEASON 2016 includes storms from late 2015.

| Split | Subset | Seasons present | Rows | Storms | RI rows | RI rate % | Storms with RI |
|---|---|---|---:|---:|---:|---:|---:|
| train | all basins | 1980-2015 | 57,244 | 3,028 | 3,949 | 6.90 | 1,032 |
| train | Bay of Bengal | 1980-2014 | 989 | 91 | 61 | 6.17 | 17 |
| val | all basins | 2016-2019 | 5,530 | 326 | 543 | 9.82 | 142 |
| val | Bay of Bengal | 2016-2019 | 149 | 14 | 13 | 8.72 | 5 |
| test | all basins | 2020-2025 | 6,317 | 434 | 586 | 9.28 | 152 |
| test | Bay of Bengal | 2020-2024 | 170 | 19 | 18 | 10.59 | 3 |

Bay of Bengal = BASIN NI and SUBBASIN BB at time t. Rates are over all rows (no over-water filter yet).
Hand check: 3 random rows (seed 0: VONGFONG 2020, WARREN 1995, EDME 1989) recomputed from the raw CSV
matched on all 17 fields.

## 2026-11-14 — Over-water RI rates, val and test (DIST2LAND > 0 at t)

CLAUDE.md testing rule: "Evaluate only rows over water (DIST2LAND > 0 at time t)". Split files are
unchanged; the filter is applied only when computing these numbers.

Command:

```bash
python -c "
import pandas as pd
rows=[]
for k in ('val','test'):
    d=pd.read_parquet(f'data/processed/{k}.parquet'); d=d[d.DIST2LAND>0]
    for name,s in [('all basins',d),('Bay of Bengal',d[(d.BASIN=='NI')&(d.SUBBASIN=='BB')])]:
        rows.append(dict(split=k,subset=name,rows=len(s),storms=s.SID.nunique(),ri_rows=int(s.RI.sum()),ri_rate=round(100*s.RI.mean(),2),storms_with_ri=s.loc[s.RI==1,'SID'].nunique()))
print(pd.DataFrame(rows).to_string(index=False))
"
```

| Split | Subset | Rows | Storms | RI rows | RI rate % | Storms with RI |
|---|---|---:|---:|---:|---:|---:|
| val | all basins | 5,333 | 323 | 539 | 10.11 | 141 |
| val | Bay of Bengal | 141 | 14 | 13 | 9.22 | 5 |
| test | all basins | 6,016 | 432 | 581 | 9.66 | 152 |
| test | Bay of Bengal | 164 | 19 | 18 | 10.98 | 3 |

Bay of Bengal = BASIN NI and SUBBASIN BB at time t.
