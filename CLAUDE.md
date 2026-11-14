# CycloneWatch

Veritas AI Hackathon 2026, AI + Climate. Deadline Mon 16 Nov 10:29 AM IST.

## What we are building (UPDATED Sat 14 Nov, 6 PM)
One model: probability that a tropical cyclone rapidly intensifies (RI) in the next 24 hours. RI = USA_WIND(t+24h) - USA_WIND(t) >= 30 kt. SCOPE: North Indian Ocean only (BASIN "NI": SUBBASIN BB = Bay of Bengal, AS = Arabian Sea). Showcase: Odisha. User: OSDMA, deciding to escalate now or hold. The alert is two-sided: it also says when a storm is NOT likely to blow up. We are a second, open signal beside IMD, not a replacement. IMD predicted Amphan's RI 24 h ahead: never claim IMD missed it.

## Models (three, same settings, same NI test rows)
MAIN = trained on NI storms only. COMPARISON = all basins. LEAVE-NI-OUT = all basins except NI, no BASIN feature. PRE-REGISTERED CHOICE: before the test run, pick MAIN vs all-basins by higher validation PR-AUC on NI rows (ties go to NI-only). Result: MAIN = NI-ONLY, decided Sat 14 Nov 2026 18:09 (NI-ONLY PR-AUC 0.3158 vs ALL-BASINS 0.2556). Not to be changed. Then run all three on the test set once. Alert levels: red = chosen threshold, amber = half of it, green = below. LEARNING CURVE: NI-only trained on 25/50/75/100% of training STORMS (5 random subsets each, fixed seeds), scored on validation. Nobody reads test.parquet before the freeze.

## Odisha casebook
Storms (check names in IBTrACS): 1999 super cyclone, PHAILIN 2013, HUDHUD 2014, TITLI 2018, FANI 2019, BULBUL 2019 (pre-2020: leave-one-storm-out retrain on the other NI storms from 1980-2019, label as illustration, NOT a headline metric); AMPHAN 2020, YAAS 2021, GULAB 2021, JAWAD 2021, DANA 2024, MONTHA 2025 (from the frozen test run). MONTHA is provisional data (TRACK_TYPE US-PROVISIONAL), is dropped by cleaning rule 2 and listed under rule 8, so it cannot be in the test run: show it only as a labelled, unscored demo storm, or leave it out.

## Demo
Gradio + Plotly viewer over precomputed data/processed/predictions.parquet (no live model): storm picker, time slider, alert tile, wind + model probability + persistence chart, top-3 reasons, casebook strip, threshold slider with real hits and false alarms, disclaimer. Fallback: static Plotly HTML + screen recording.

## Out of scope
Track forecasts, district maps, population data, "lives saved" claims. Climate drift only if time remains, as a labelled global side analysis. USA_WIND is JTWC 1-minute wind, not IMD's official wind: state this as a limit.

## Data rules
File: data/raw/ibtracs.since1980.list.v04r01.csv (NOAA IBTrACS v4,
Knapp et al. 2018, doi 10.25921/82ty-9e16).
0. Load with keep_default_na=False: the basin code NA (North Atlantic) is
   otherwise read as missing. Treat only blank or single-space cells as missing.
1. Second CSV row is units: skip it.
2. Keep TRACK_TYPE == "main" only. Print TRACK_TYPE counts by year first.
3. Keep only 00, 06, 12, 18 UTC (other rows are interpolated).
4. Use USA_WIND (1-min sustained, kt) only. Never mix with WMO_WIND.
5. Keep NATURE == "TS" rows only.
6. Compute every lag (t-6h, t-12h, t-24h) and the target (t+24h) BY TIMESTAMP
   within the same SID, never by row position. Assert the time gaps.
7. Drop rows without a matching t-24h or t+24h time in the same storm.
8. List storms whose US data is provisional (recent seasons) as a limit.

## Model rules
- LightGBM. NO scale_pos_weight or class weights. Choose the alert threshold
  on validation. Before freezing, check on validation that mean predicted RI
  probability is close to the actual RI rate; save a reliability plot.
- Persistence baseline = logistic regression on last-24h wind change only,
  fitted on training years. Climatology = basin RI rate from training years.

## Testing rules (most important)
- Split by year AND storm: train 1980-2015, validate 2016-2019, test 2020+.
  Assert no SID is in two splits.
- Thresholds and settings chosen on validation only. Test set run ONCE after "freeze".
- Evaluate only rows over water (DIST2LAND > 0 at time t).
- RI episode = consecutive RI=1 rows in one storm. Report per episode: caught
  (alert red at its first row), lead time in hours (how long before the first
  row the alert turned red), and false-alarm episodes per storm.
- Also report results on RI rows where the last 24h wind rose < 30 kt.
- Row metrics: PR-AUC, precision, recall, Brier score.
- Subsets: Bay of Bengal, North Indian Ocean, all basins within 400 km of land,
  all basins; always print storms and episodes.
- 95% bootstrap intervals resampling whole storms; skip resamples with no RI
  rows and report how many were skipped.
- Climate drift: models trained on 1980-1997 vs 1998-2015, same settings,
  same test rows. Leave-NI-out model: drop the BASIN feature.
- Fixed random seeds. Every number goes into docs/results_log.md with the command.

## Structure
src/data/    load.py, clean.py, features.py, split.py
src/models/  ri_model.py, experiments.py
src/eval/    baselines.py, metrics.py, episodes.py
app/         app.py
docs/        results_log.md, writeup_one_page.md, figures/

## Stack
Python 3.11 or newer, pandas, numpy, lightgbm, scikit-learn, plotly, gradio, matplotlib.
Ask before adding any other dependency or data source.
