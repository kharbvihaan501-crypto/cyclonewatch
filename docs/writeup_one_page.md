\# One-page write-up



\## The problem

Rapid intensification (RI) is when a cyclone's winds jump by at least 30 knots in 24 hours. Near coasts it is becoming more common: within 400 km of land, RI events rose from fewer than 5 a year in the 1980s to about 15 a year by 2020 (\[Li et al. 2023](https://doaj.org/article/f9786b7e58604de88fb3d76e187de121)). It leaves little time to act: Hurricane Otis (2023) was forecast as Category 1 sixteen hours before landfall and arrived as Category 5 (\[WTW](https://www.wtwco.com/en-ug/insights/2024/04/rapid-intensification-in-the-pacific-a-tale-of-two-tropical-cyclones)). India's weather service already runs an RI index, built from 88 cyclones between 1981 and 2010 (\[IMD](https://imdpune.gov.in/training/FTC1/NWP/Lecture2-sdk%20(1).pdf)). Early action saves lives: Odisha evacuated about 1 million people before Cyclone Phailin, and fewer than 40 died, versus about 10,000 in 1999 (\[World Bank](https://www.worldbank.org/en/results/2014/04/10/india-averts-cyclone-phailin-devastation)).



\## What we built



\## How it works



\## Results

From the single frozen test run, North Indian Ocean storms 2020 on, over water, 95% storm-bootstrap intervals (docs/results_test.md, sections 3-4). There were 26 storms and 9 RI episodes. The model was red at the first RI row in 5 (55.6% [25.0, 85.7]), only later in 3 (8 of 9 red at any row) and never in 1. Hard-case recall (RI rows where the last-24h wind rise was under 30 kt) was 0.708 [0.474, 0.938]. There were 0.46 [0.19, 0.77] false-alarm episodes per storm.

Alert rates were matched on validation, where persistence's nearest steps put 14.3% and 26.6% of rows on red (docs/results_shaurya.md). On test, the model was red on 21.4% of rows and persistence on 23.2% and 30.1%. Persistence caught 0 and 2 of 9 episodes, with hard-case recall 0.250 and 0.417. PR-AUC was 0.499 [0.208, 0.805] for the model and 0.221 [0.085, 0.379] for persistence. Climatology: [MISSING: no test-set climatology figures in either results file].

Amphan 2020 was caught at the first RI row, Dana 2024 was a false alarm (3 red rows, no RI) and Gati 2020 was missed. With only 9 episodes, this is consistent with usefulness, not proof. We did not compare against IMD's RI index. A low RI likelihood does not mean a cyclone is safe.



\## What we added



\## AI tools used



\## Sources

