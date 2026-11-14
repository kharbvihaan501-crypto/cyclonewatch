# CycloneWatch

Veritas AI Hackathon 2026, AI + Climate. Deadline Mon 16 Nov 10:29 AM IST.

## What we are building
One model: the probability that a tropical cyclone rapidly intensifies (RI) in the
next 24 hours. RI = USA_WIND at t+24h minus USA_WIND at t >= 30 kt.
Place: Bay of Bengal (BASIN NI, SUBBASIN BB). User: Odisha State Disaster
Management Authority, deciding whether to escalate a warning now or wait.
Train on ALL basins (Bay of Bengal has too few storms), report Bay of Bengal results.
IMD already runs an RI-Index (88 cyclones, 1981-2010, uses SST/shear/humidity);
we are an open second signal, not a replacement.
Demo: Gradio + Plotly replay of Cyclone Amphan (2020).
NOT in scope: track forecasting, cones, district maps, population data.

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
