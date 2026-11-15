# CycloneWatch

An open model that estimates the chance that a North Indian Ocean cyclone will rapidly strengthen in the next 24 hours, checked on storms it never saw.

**Try it:** Live demo: [LINK TO FILL AFTER DEPLOY]

Veritas AI Hackathon 2026, AI + Climate. Every number below is copied from the files named next to it.

---

## 1. Problem

**Rapid intensification (RI)** means a cyclone's wind rises by at least 30 kt in 24 hours: USA_WIND(t+24h) − USA_WIND(t) ≥ 30 kt (CLAUDE.md). It is common in our region. Ranalkar et al. 2024 (doi 10.1002/joc.8379) found that 59 of 153 North Indian Ocean cyclones from 1982-2020 underwent RI (docs/problem_facts.md, item 6, verified).

**Scope:** the North Indian Ocean only (BASIN NI: the Bay of Bengal and the Arabian Sea), with Odisha as the showcase (CLAUDE.md).

**User:** the Odisha State Disaster Management Authority (OSDMA), which has to decide whether to escalate a warning now or hold (CLAUDE.md). The alert works both ways: it also says when a storm is *not* likely to blow up.

**What CycloneWatch is:** an open *second* signal beside the India Meteorological Department (IMD), not a replacement (CLAUDE.md). CycloneWatch sits next to IMD's own rapid-intensification guidance.

**Amphan and IMD:** IMD states that its 16 May 2020 16:45 IST bulletin predicted rapid intensification about 24 hours ahead (IMD RSMC New Delhi, Amphan summary report, https://rsmcnewdelhi.imd.gov.in/uploads/archive/21/21_d7b2af_press%20release%20amphan.pdf). We never claim IMD missed it (CLAUDE.md).

## 2. Dataset

**Source:** NOAA IBTrACS v04r01 (Knapp et al.), NOAA National Centers for Environmental Information, https://doi.org/10.25921/82ty-9e16, file `ibtracs.since1980.list.v04r01.csv`: all basins since 1980 (CLAUDE.md).

**Cleaning rules** (CLAUDE.md):
1. Load with `keep_default_na=False`, so the basin code "NA" (North Atlantic) is not read as missing.
2. Skip the second CSV row (units).
3. Keep the main track only (TRACK_TYPE = main).
4. Keep only 00, 06, 12 and 18 UTC rows; the other rows are interpolated.
5. Use USA_WIND (1-minute sustained, kt) only, never mixed with WMO_WIND.
6. Keep NATURE = TS rows only.
7. Compute every lag (t−6 h, t−12 h, t−24 h) and the target (t+24 h) **by timestamp within the same storm**, never by row position, and assert the time gaps.
8. Drop rows without a matching t−24 h or t+24 h row.
9. Score only rows over water (DIST2LAND > 0 at time t).

**Provisional recent storms:** these are listed as a limit (cleaning rule 8). MONTHA 2025 is provisional (US-PROVISIONAL), so cleaning drops it and it is not in the test run (CLAUDE.md). For the full list, see rule 8 in `src/data/clean.py`.

**Size:** North Indian Ocean training data (1980-2015) has 1,596 rows, 136 storms and 91 RI rows (docs/results_log.md). The test set has 276 over-water North Indian Ocean rows from 26 storms (docs/results_test.md).

## 3. Method

**Model:** LightGBM, binary objective, with **no class weights and no resampling**. Missing pressure values are left missing (docs/results_log.md).

**Three models**, trained with identical settings (CLAUDE.md, docs/results_log.md):
- **MAIN, NI-ONLY:** trained on North Indian Ocean storms only.
- **ALL-BASINS:** a comparison model trained on every basin.
- **LEAVE-NI-OUT:** trained on all basins except the North Indian Ocean, without the BASIN feature.

MAIN was picked **before the test run**, by validation PR-AUC on North Indian Ocean rows: NI-ONLY 0.3158 vs ALL-BASINS 0.2556 (docs/freeze.md).

**Features (15)**, all using data at or before time t (docs/results_log.md):
- **Wind:** wind now, and the wind change over the last 6, 12 and 24 h.
- **Pressure:** pressure now, and the pressure change over 24 h.
- **Motion:** storm speed and direction over the last 12 and 24 h.
- **Position and time:** latitude, distance to land, month (as sine and cosine) and basin.

**Splits by year and by storm** (CLAUDE.md):
- train 1980-2015;
- validation 2016-2019;
- test 2020 onward.

No storm appears in two splits.

**Threshold, chosen on validation only:** red is set so that 20% of North Indian Ocean over-water validation rows are red (docs/freeze.md):
- **Red** from 0.088325. The rate achieved on validation was 19.88%.
- **Amber** from 0.044163 (half of red).
- **Green** below amber.

**Freeze:** the models were frozen on Sat 14 Nov 2026 at 19:08 IST. The SHA-256 of every model file is in docs/freeze.md.

**Single test run:** the test set was run once, on Sat 14 Nov 2026 at 19:30:50, after the model hashes were checked against the freeze record (docs/results_test.md). The git tag `freeze-v1` points to commit `896e88b`, the commit recorded for that test run.

## 4. Evaluation

**Scored by RI episode**, not just by row. An episode is a run of consecutive RI rows in one storm (CLAUDE.md). For each episode we report:
- **caught at the first row:** red at the episode's first row;
- **caught late:** red only later inside the episode;
- **missed:** never red during the episode.

We also report:
- **false-alarm episodes per storm:** red runs with no RI;
- **hard-case recall:** recall on RI rows whose last-24 h wind rise was under 30 kt, where rising wind alone gives no warning;
- **row metrics:** PR-AUC, ROC AUC and Brier score (docs/results_test.md).

**Baselines** (CLAUDE.md):
- **Persistence:** logistic regression on the last-24 h wind change only, fitted on 1980-2015.
- **Climatology:** the basin's RI rate in the training years.

**Why compare at a matched alert rate:** "models are compared at the same alert rate (20% of over-water rows red) plus PR-AUC. False-alarm episodes per storm alone is not used, because long red runs hide constant alerting" (docs/results_log.md).

Persistence moves in 5 kt steps, so it can't hit exactly 20%. Its two nearest steps put 14.3% and 26.6% of validation rows on red, and these "bracket the model's 19.9% alert rate" (docs/results_shaurya.md). Climatology gives one constant per basin, so it is "a floor … report its PR-AUC only" (docs/results_log.md).

**Uncertainty:** 95% intervals come from 1,000 bootstrap resamples of whole storms (seed 0). Resamples with no RI rows are skipped: 1 of 1,000 for the North Indian Ocean test set (docs/results_test.md).

**Leakage review:**
- Every feature uses only data at or before t.
- A truncation test rebuilt 200 random validation rows from data up to t, plus a fake t+24 h row. All 15 features came out identical (docs/results_log.md).
- An independent review found no leakage and recomputed 3 random validation rows by hand: 54 of 54 values matched (docs/results_shaurya.md).

## 5. Results

From the single test run, North Indian Ocean, over-water rows, MAIN model, with 95% storm-bootstrap intervals (docs/results_test.md, sections 3-4):

| | MAIN (NI-ONLY) | Persistence, rise ≥ 25 kt | Persistence, rise ≥ 20 kt |
|---|---|---|---|
| Storms / RI episodes | 26 / 9 | 26 / 9 | 26 / 9 |
| Caught at first RI row | 5 of 9 (55.6% [25.0, 85.7]) | 0 of 9 | 2 of 9 (22.2% [0.0, 57.1]) |
| Hard-case recall | 0.708 [0.474, 0.938] | 0.250 [0.059, 0.444] | 0.417 [0.083, 0.760] |
| False-alarm episodes per storm | 0.46 [0.19, 0.77] | 0.54 [0.31, 0.81] | 0.54 [0.35, 0.77] |
| Alert rate (% of rows red) | 21.4 [12.4, 31.7] | 23.2 [14.2, 33.8] | 30.1 [19.7, 42.1] |
| PR-AUC | 0.499 [0.208, 0.805] | 0.221 [0.085, 0.379] | 0.221 [0.085, 0.379] |
| ROC AUC | 0.886 [0.793, 0.960] | 0.706 [0.535, 0.842] | 0.706 [0.535, 0.842] |

Of the 9 RI episodes, the model was red at the first row in 5, red only later in 3 (8 of 9 had "an alert at any row in the episode") and never red in 1 (section 4).

- **Amphan 2020 (success):** red at the first RI row.
- **Dana 2024 (false alarm):** 3 red rows, but no RI episode (section 5).
- **Gati 2020 (missed):** never red; the highest probability during its RI episode was 0.041 (section 4).

The comparison models did slightly worse than MAIN. ALL-BASINS caught 4 of 9 (PR-AUC 0.487) and LEAVE-NI-OUT caught 4 of 9 (PR-AUC 0.454).

Climatology is reported on validation only: its North Indian Ocean PR-AUC was 0.082 [0.038, 0.131] (docs/results_log.md).

## 6. Limitations

- **Small test set:** 9 RI episodes, with wide intervals. The results are consistent with the model being useful, but they are **not proof**.
- **Low RI likelihood does not mean safe.** It only means sudden strengthening is less likely. The storm can still be dangerous.
- **North Indian Ocean only** (CLAUDE.md). The MAIN model was trained on North Indian Ocean storms only. ALL-BASINS (trained on every basin) and LEAVE-NI-OUT were comparisons and were not chosen (docs/freeze.md).
- **No storm-surge, wind-field or flood model.** The coastal exposure panel in the app is illustrative only: it is based on distance, not a forecast.
- **The model under-predicts on the test set.** Mean predicted RI probability was 0.0673 [0.0476, 0.0886], against an actual RI rate of 0.1268 [0.0505, 0.2163] (docs/results_test.md). On validation it was closer: 0.0742 vs 0.0819, a ratio of 0.91 (docs/results_log.md).
- **Climate drift:** climate-drift test not run (time).
- **Provisional data:** recent storms' data can be provisional. MONTHA 2025 is excluded for this reason (CLAUDE.md).
- **Wind source:** USA_WIND is the JTWC 1-minute wind, not IMD's official wind (CLAUDE.md).
- **Optimistic validation numbers:** threshold choice, early stopping and validation scoring used the same validation rows (docs/results_log.md). Only the test numbers above are a fair check.

## How to run

You need Python 3.11 or newer (tested with 3.12 and 3.13). Clone the repository, then from its root create a virtual environment and install the requirements: python -m venv .venv, then .venv\Scripts\python -m pip install -r requirements.txt (on macOS or Linux use .venv/bin/python in place of .venv\Scripts\python). Start the app with .venv\Scripts\python app\app.py and open http://127.0.0.1:7860 in a browser; the first start can take about a minute. The app uses the frozen model in models/ and the saved test-run predictions in data/processed/predictions.parquet. It checks the model's SHA-256 against docs/freeze.md at start-up and never reads the test split.

## AI tools used

- **Claude Code:** writing and checking code, running the evaluation scripts and reviews, and building the app.
- **Claude:** drafting and editing text.

## Credits

- **NOAA IBTrACS v04r01 (Knapp et al.), NOAA National Centers for Environmental Information, https://doi.org/10.25921/82ty-9e16**: cyclone track data.
- **LightGBM**, **scikit-learn**, **pandas**: modelling and evaluation.
- **Plotly**, **Gradio**: the app.
- **IMD**, the India Meteorological Department: the official forecasts CycloneWatch sits beside.

*Prototype for research and education. Not an official forecast. Always follow IMD and your state disaster authority.*
