# Freeze record

**Frozen:** Sat 14 Nov 2026 19:08 IST

**Git commit (`git rev-parse HEAD`):** `fe2574665d712df9496d76dc2088c1527a769d5e`

Working tree at freeze time differs from that commit only by the ALERT_RATE default change
(0.182 -> 0.20) in `src/eval/ni_report.py` line 34 and `src/models/ri_model.py` line 52, which was
not yet committed. No model was retrained for the freeze; the files below were not changed by it.

## Model files (SHA-256)

| File | SHA-256 |
|---|---|
| `models/ri_lgbm.txt` (MAIN, copy of `ri_lgbm_ni.txt`) | `6db069f53528a031713987a0c3e8c8db65f3578a2cf33a7357046ca2139bf1bf` |
| `models/ri_lgbm_ni.txt` | `6db069f53528a031713987a0c3e8c8db65f3578a2cf33a7357046ca2139bf1bf` |
| `models/ri_lgbm_all.txt` | `924468978d5ea53ac579cd50d16d8aaae46f12037814c0fa1679899d683dbcdb` |
| `models/ri_lgbm_leaveni.txt` | `32a23e3f6755a0f5fe50932b9721d3028b85f9bdd5dbc5d51e8a217aca03ad8c` |
| `models/ri_lgbm_features.json` | `bd8aa9f7c66ebb79bd68760bb3987547eebe939e605a758a270a7e33ce4045a7` |

## Pre-registered choice

MAIN = NI-ONLY, decided Sat 14 Nov 2026 18:09 (validation PR-AUC on NI over-water rows:
NI-ONLY 0.3158 vs ALL-BASINS 0.2556; ties would have gone to NI-ONLY).

## Alert-rate rule and thresholds

Rule: red threshold = `threshold_for_alert_rate(NI validation, prob, 0.20)`, i.e. the threshold that
puts 20% of North Indian Ocean (BASIN NI) over-water validation rows (2016-2019) on red.
Amber = half of each red threshold; green = below amber. Thresholds read from the saved models
above and `data/processed/val.parquet`:

| Model | File | Red threshold | Amber from (red / 2) | Achieved NI val alert rate |
|---|---|---:|---:|---:|
| NI-ONLY (MAIN) | `models/ri_lgbm_ni.txt` | 0.088325 | 0.044163 | 19.88% |
| ALL-BASINS | `models/ri_lgbm_all.txt` | 0.177467 | 0.088733 | 19.88% |
| LEAVE-NI-OUT | `models/ri_lgbm_leaveni.txt` | 0.175196 | 0.087598 | 19.88% |

Persistence baseline (logistic regression on DWIND_24, fitted on train 1980-2015) thresholds used:
0.1285 (DWIND_24 >= 25 kt, 14.3% of NI over-water validation rows red) and 0.1027
(DWIND_24 >= 20 kt, 26.6%). Persistence probabilities come in 5 kt steps, so 20% is not reachable.

## Test-period data seen before the freeze

- `src/eval/data_checks.py` touched 2020+ rows in `data/processed/tracks.parquet` in three places:
  missing-wind shares by decade, yearly RI counts for 2020-2025, and Amphan's 2020 track as CSV and map.
  Nothing was fitted, tuned or scored on them.
- The Amphan replay in the app was viewed but not used for any decision.

Also seen during data preparation and app checks (none of it used to fit, tune, choose or score
anything):

- `src/data/split.py` output: test-split (2020+) rows, storms, RI rows, RI rate and storms with RI,
  for all basins and the Bay of Bengal (logged in `docs/results_log.md`).
- Over-water RI rates for the test split (rows, storms, RI rows, RI rate, storms with RI; all basins
  and Bay of Bengal), computed by reading `test.parquet` before the hard rule against reading it
  was set (logged in `docs/results_log.md`).
- A feature hand-check of one random 2020 row (VONGFONG, WP) against the raw CSV.
- Cleaning checks: Amphan 2020 present (24 rows, peak USA_WIND 145 kt) and Bay of Bengal storm
  counts per season 2020-2024; the rule 8 list of provisional 2024-2026 storms.
- App checks printed Amphan replay probabilities and alert levels (earlier model versions and
  NI-ONLY) to confirm the app runs.

After this freeze the test set is run once with these files, and nothing above is changed.
