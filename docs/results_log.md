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
