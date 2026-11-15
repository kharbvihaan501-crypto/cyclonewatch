# Failure cases on the test set (MAIN = NI-ONLY)

Sources: `data/processed/predictions.parquet` (276 over-water NI test rows) and `docs/results_test.md`
(frozen single test run). Alert levels for MAIN: red at P_MAIN >= 0.088325, amber from 0.044163,
green below. RI = 1 means the wind rose at least 30 kt in the next 24 hours. Times are UTC.
Only over-water rows with a reading 24 h before and 24 h after are in the test rows.

## 1. GATI 2020: the episode that was never alerted

| Time | USA_WIND (kt) | P_MAIN | ALERT | RI |
|---|---:|---:|---|---:|
| 2020-11-21 18:00 | 35 | 0.041 | green | 1 |
| 2020-11-22 00:00 | 60 | 0.085 | amber | 0 |
| 2020-11-22 06:00 | 100 | 0.051 | amber | 0 |
| 2020-11-22 12:00 | 90 | 0.039 | green | 0 |

Gati's RI episode is a single row at 35 kt, where MAIN gave 0.041 (just under amber) and the
persistence baseline, which sees only the last-24h wind change, also gave a low 0.064; the wind
then jumped to 60 kt and 100 kt within the next 12 hours, so the rise had not shown up in the
model's inputs at the one row where it counted.

## 2. DANA 2024: the near-miss false alarm

DANA had no RI row in the test set, but MAIN was red for three rows:

| Time | USA_WIND (kt) | P_MAIN | ALERT | Wind 24 h later (kt) | Change in 24 h |
|---|---:|---:|---|---:|---:|
| 2024-10-23 06:00 | 40 | 0.196 | red | 65 | +25 |
| 2024-10-23 12:00 | 50 | 0.549 | red | 65 | +15 |
| 2024-10-23 18:00 | 55 | 0.318 | red | 60 | +5 |

The wind 24 h later is read from DANA's later rows in the same file. After the red rows the wind rose to a
peak of 65 kt on 24 Oct 06:00-12:00 and was 60 kt at 18:00, 23 km from land. The largest 24 h rise
was +25 kt, below the 30 kt RI line. The red at 0.549 was the highest P_MAIN of any DANA row and
came 24 h before a rise of only +15 kt.

## 3. Late alerts: red only after the first RI row

| Storm | First RI row | P_MAIN at first RI row (alert) | First red | Hours late | Episode rows |
|---|---|---:|---|---:|---:|
| MOCHA 2023 | 2023-05-10 18:00 | 0.032 (green) | 2023-05-11 00:00, P 0.135 | +6 | 10 |
| BIPARJOY 2023 (second episode) | 2023-06-09 00:00 | 0.056 (amber) | 2023-06-09 18:00, P 0.263 | +18 | 4 |
| HAMOON 2023 | 2023-10-23 00:00 | 0.019 (green) | 2023-10-23 06:00, P 0.221 | +6 | 3 |

- MOCHA: one red row (0.121) at 2023-05-10 00:00, 18 h before the episode, then amber and green
  rows; the alert did not stay red into the episode. From +6 h it was red for 8 of the 10 RI rows.
- BIPARJOY: amber (0.056, 0.056, 0.076) for the first three RI rows; red only at the fourth
  and last RI row of the episode.
- HAMOON: red for one row at +6 h (0.221), then green again (0.031) at the third and last RI row.
