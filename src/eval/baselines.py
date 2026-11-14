"""Baselines for 24 h rapid intensification (RI).

- Persistence: logistic regression on the last-24h wind change (DWIND_24) only,
  fitted on the training split (1980-2015, all rows).
- Climatology: RI rate per BASIN in the training split (all rows); a basin
  missing from training falls back to the overall training rate.

Run from the project root (VALIDATION only; the test split is never loaded):
    py -3.12 -m src.eval.baselines                  # both baselines, max-F1 threshold
    py -3.12 -m src.eval.baselines --fa-target 1.0  # persistence at ~1 false-alarm episode per storm
    py -3.12 -m src.eval.baselines --alert-rate 0.20  # persistence at ~20% of over-water rows red
"""
import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from src.eval.metrics import (evaluate, format_table, threshold_for_alert_rate,
                              threshold_for_false_alarms)

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
RESULTS_LOG = "docs/results_log.md"
SEED = 0
TARGET = "RI"


# ------------------------------------------------------------------ persistence
def fit_persistence(train: pd.DataFrame) -> LogisticRegression:
    assert train["DWIND_24"].notna().all(), "DWIND_24 missing in training rows"
    model = LogisticRegression(random_state=SEED)
    model.fit(train[["DWIND_24"]].to_numpy(), train[TARGET].to_numpy())
    return model


def predict_persistence(model: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[["DWIND_24"]].to_numpy())[:, 1]


# ------------------------------------------------------------------ climatology
def fit_climatology(train: pd.DataFrame) -> dict:
    by_basin = train.groupby("BASIN", observed=True)[TARGET].mean()
    return {"by_basin": by_basin.to_dict(), "overall": float(train[TARGET].mean())}


def predict_climatology(clim: dict, df: pd.DataFrame) -> np.ndarray:
    p = df["BASIN"].astype(str).map(clim["by_basin"]).astype(float)
    return p.fillna(clim["overall"]).to_numpy()


# ------------------------------------------------------------------ threshold
def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold with the highest F1 on the given rows (used on validation only)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def run_matched(mode: str, target: float) -> None:
    """Persistence on validation at a threshold matched on validation.

    mode "fa":   ~target false-alarm episodes per storm (all basins, over water).
    mode "rate": ~target share of over-water rows on red (all basins), target as a fraction.
    """
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    assert train["SEASON"].max() <= 2015 and val["SEASON"].between(2016, 2019).all()
    assert not set(train["SID"]) & set(val["SID"]), "SID in both train and val"
    val = val.copy()
    val["P_PERSISTENCE"] = predict_persistence(fit_persistence(train), val)

    if mode == "fa":
        thr, got = threshold_for_false_alarms(val, "P_PERSISTENCE", target)
        label = f"Persistence at ~{target:g} false-alarm episode per storm"
        achieved = f"achieved {got:.3f} false-alarm episodes per storm"
        rule = ("the distinct validation probability whose false-alarm episodes per storm "
                "(all basins, over water) is closest to the target")
        cmd = f"py -3.12 -m src.eval.baselines --fa-target {target:g}"
    else:
        thr, got = threshold_for_alert_rate(val, "P_PERSISTENCE", target)
        label = f"Persistence at {100 * target:g}% alert rate"
        achieved = f"achieved alert rate {100 * got:.1f}% of over-water rows"
        rule = ("the distinct validation probability whose share of over-water rows on red "
                "(all basins) is closest to the target")
        cmd = f"py -3.12 -m src.eval.baselines --alert-rate {target:g}"

    res = evaluate(val, "P_PERSISTENCE", thr)
    keep = ["subset", "storms", "RI episodes", "RI rows rise < 30 kt", "episodes caught",
            "alert rate %", "red hours / storm", "episodes caught %", "median lead h",
            "false-alarm episodes / storm", "recall (last-24h rise < 30 kt)", "PR-AUC",
            "boot used / skipped"]
    table = format_table(res)[keep]
    head = f"{label}: threshold {thr:.4f} (chosen on validation, all basins over water; {achieved})"
    print(head)
    print(table.to_string(index=False))

    stamp = dt.datetime.now().strftime("%a %d %b %Y %H:%M")
    block = "\n".join([
        f"\n## {stamp}: {label} (VALIDATION 2016-2019)", "",
        f"Command: `{cmd}`", "",
        f"Over-water rows only. Threshold = {rule}; chosen and scored on the same validation rows "
        "(optimistic). Alert rate = share of over-water rows on red; red hours / storm = 6 h x red "
        "rows / storms. 95% CI: 1,000 storm bootstrap resamples per subset, "
        f"seed {SEED}; resamples with no RI rows skipped. Test split not loaded.", "",
        f"**{head}**", "", "```", table.to_string(index=False), "```", "",
    ])
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(block + "\n")
    print(f"\n[appended to {RESULTS_LOG}]")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa-target", type=float, default=None,
                    help="run persistence only, at ~this many false-alarm episodes per storm")
    ap.add_argument("--alert-rate", type=float, default=None,
                    help="run persistence only, at ~this share (0-1) of over-water rows on red")
    args = ap.parse_args(argv)
    if args.fa_target is not None and args.alert_rate is not None:
        ap.error("use --fa-target or --alert-rate, not both")
    if args.fa_target is not None:
        return run_matched("fa", args.fa_target)
    if args.alert_rate is not None:
        return run_matched("rate", args.alert_rate)

    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    assert train["SEASON"].max() <= 2015 and val["SEASON"].between(2016, 2019).all()
    assert not set(train["SID"]) & set(val["SID"]), "SID in both train and val"
    print(f"train: {len(train):,} rows, {train['SID'].nunique():,} storms, "
          f"seasons {train['SEASON'].min()}-{train['SEASON'].max()}")
    print(f"val:   {len(val):,} rows, {val['SID'].nunique():,} storms, "
          f"seasons {val['SEASON'].min()}-{val['SEASON'].max()}")

    pers = fit_persistence(train)
    clim = fit_climatology(train)
    print(f"\npersistence: logit(P) = {pers.intercept_[0]:.4f} + {pers.coef_[0, 0]:.4f} * DWIND_24")
    print("climatology (train RI rate by BASIN): "
          + ", ".join(f"{b} {r:.4f}" for b, r in clim["by_basin"].items())
          + f"; overall {clim['overall']:.4f}")

    val = val.copy()
    val["P_PERSISTENCE"] = predict_persistence(pers, val)
    val["P_CLIMATOLOGY"] = predict_climatology(clim, val)

    water = val[val["DIST2LAND"] > 0]
    lines = []
    for name, col in [("Persistence", "P_PERSISTENCE"), ("Climatology", "P_CLIMATOLOGY")]:
        thr = choose_threshold(water[TARGET].to_numpy(), water[col].to_numpy())
        mean_p, rate = water[col].mean(), water[TARGET].mean()
        res = evaluate(val, col, thr)
        table = format_table(res)
        head = (f"{name}: alert threshold {thr:.4f} (max F1 on validation over-water rows, all basins); "
                f"mean predicted P {mean_p:.4f} vs actual RI rate {rate:.4f} (over water)")
        print("\n" + head)
        print(table.to_string(index=False))
        lines += [f"**{head}**", "", "```", table.to_string(index=False), "```", ""]

    cmd = "py -3.12 -m src.eval.baselines"
    stamp = dt.datetime.now().strftime("%a %d %b %Y %H:%M")
    block = "\n".join([
        f"\n## {stamp}: baselines on VALIDATION (2016-2019)", "",
        f"Command: `{cmd}`", "",
        "Over-water rows only (DIST2LAND > 0 at t). Persistence = logistic regression on DWIND_24, "
        "fitted on train (1980-2015). Climatology = train RI rate by BASIN. Thresholds chosen on "
        "validation, then scored on the same validation rows (optimistic). 95% CI: 1,000 bootstrap "
        f"resamples of whole storms per subset, seed {SEED}; resamples with no RI rows skipped. "
        "Episode = consecutive 6-hourly RI rows in one storm; lead time = hours the alert was "
        "already red, unbroken, before the episode's first row. Test split not loaded.", "",
        *lines,
    ])
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(block + "\n")
    print(f"\n[appended to {RESULTS_LOG}]")


if __name__ == "__main__":
    sys.exit(main())
