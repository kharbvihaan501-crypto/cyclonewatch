"""THE single test run (2020+). Frozen models and thresholds from docs/freeze.md; nothing is fitted
or chosen on test data.

Run from the project root (once):   .venv\\Scripts\\python -m src.eval.test_run
Dry run on validation (no test data, outputs to a scratch folder):
                                     .venv\\Scripts\\python -m src.eval.test_run --dry-run-val <out_dir>

Steps:
 1. SHA-256 of every file in models/ must equal docs/freeze.md, else stop before loading anything.
 2. Thresholds: red thresholds from docs/freeze.md. Each is reproduced exactly from VALIDATION data
    (the same call the freeze used) and asserted to round to the freeze.md value; the exact value is
    used so rounding cannot move a row across the threshold. Amber = red / 2.
    Persistence = logistic regression on DWIND_24 fitted on train (src/eval/baselines.py), at its two
    frozen thresholds 0.1285 (DWIND_24 >= 25 kt) and 0.1027 (DWIND_24 >= 20 kt).
 3. Test rows: over water only (DIST2LAND > 0). evaluate() from src/eval/metrics.py unchanged
    (episodes, lead time, false alarms, PR-AUC, Brier, 95% storm bootstrap, seed 0). ROC AUC and
    mean predicted vs actual get their own storm bootstrap (1,000 resamples, seed 0, same skip rule).
 4. Writes docs/results_test.md and data/processed/predictions.parquet. Refuses to run if
    docs/results_test.md already exists.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.features import BASINS, TARGET
from src.eval.baselines import fit_persistence, predict_persistence
from src.eval.episodes import episode_alerts
from src.eval.metrics import N_BOOT, SEED, evaluate, format_table, threshold_for_alert_rate

FREEZE = "docs/freeze.md"
MODELS_DIR = "models"
META_PATH = "models/ri_lgbm_features.json"
TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
TEST_PATH = "data/processed/test.parquet"
OUT_MD = "docs/results_test.md"
OUT_PRED = "data/processed/predictions.parquet"
CMD = r".venv\Scripts\python -m src.eval.test_run"
ALERT_RATE = 0.20
PERSIST_FROZEN = {"persistence @ 0.1285 (DWIND_24 >= 25 kt)": (0.1285, 25),
                  "persistence @ 0.1027 (DWIND_24 >= 20 kt)": (0.1027, 20)}
FREEZE_NAMES = {"NI-ONLY": "NI-ONLY (MAIN)", "ALL-BASINS": "ALL-BASINS", "LEAVE-NI-OUT": "LEAVE-NI-OUT"}
CASEBOOK = [("AMPHAN", 2020), ("YAAS", 2021), ("GULAB", 2021), ("JAWAD", 2021), ("DANA", 2024)]
SUBSET_NI = "North Indian Ocean"
SUBSET_BOB = "Bay of Bengal"
SUBSET_400 = "All basins, <= 400 km of land"


# ------------------------------------------------------------------ freeze checks
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hashes(freeze_txt: str) -> list[str]:
    frozen = dict(re.findall(r"\|\s*`(models/[^`]+)`[^|]*\|\s*`([0-9a-f]{64})`", freeze_txt))
    on_disk = {f"{MODELS_DIR}/{f}": sha256(os.path.join(MODELS_DIR, f)) for f in sorted(os.listdir(MODELS_DIR))}
    lines, bad = [], []
    for path in sorted(set(frozen) | set(on_disk)):
        a, b = frozen.get(path), on_disk.get(path)
        ok = a is not None and a == b
        lines.append(f"{'OK      ' if ok else 'MISMATCH'} {path}  {b or 'missing on disk'}")
        if not ok:
            bad.append(path)
    if bad:
        print("\n".join(lines))
        sys.exit(f"STOP: models/ differs from {FREEZE} for {bad}. Nothing was loaded.")
    return lines


def frozen_red(freeze_txt: str) -> dict[str, float]:
    out = {}
    for key, label in FREEZE_NAMES.items():
        m = re.search(r"\|\s*" + re.escape(label) + r"\s*\|[^|]*\|\s*([0-9.]+)\s*\|", freeze_txt)
        assert m, f"red threshold for {label} not found in {FREEZE}"
        out[key] = float(m.group(1))
    return out


# --------------------------------------------------------------------- predictions
def predict(model_file: str, features: list[str], df: pd.DataFrame) -> np.ndarray:
    booster = lgb.Booster(model_file=model_file)
    assert booster.feature_name() == features, f"{model_file}: feature order differs"
    X = df[features].copy()
    if "BASIN" in features:
        X["BASIN"] = pd.Categorical(X["BASIN"].astype(str), categories=BASINS)
    return booster.predict(X)


def colour(p: float, red: float) -> str:
    return "red" if p >= red else "amber" if p >= red / 2 else "green"


# -------------------------------------------------------------------- extra metrics
def roc_and_mean(d: pd.DataFrame, col: str) -> dict:
    """ROC AUC, mean predicted, actual RI rate with a storm bootstrap (N_BOOT, seed SEED, skip no-RI resamples)."""
    y, p = d[TARGET].to_numpy().astype(int), d[col].to_numpy().astype(float)
    sids = d["SID"].unique()
    code = pd.Series(np.arange(len(sids)), index=sids)
    rs = code.loc[d["SID"]].to_numpy()
    has_ri = np.bincount(rs, weights=y, minlength=len(sids)) > 0

    def stats(w):
        wr = w[rs]
        both = (wr[y == 1].sum() > 0) and (wr[y == 0].sum() > 0)
        return (roc_auc_score(y, p, sample_weight=wr) if both else np.nan,
                np.average(p, weights=wr), np.average(y, weights=wr))

    point = stats(np.ones(len(sids)))
    rng = np.random.default_rng(SEED)
    boots, skipped = [], 0
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(sids), len(sids)), minlength=len(sids))
        if not (w * has_ri).any():
            skipped += 1
            continue
        boots.append(stats(w))
    b = np.array(boots) if boots else np.full((0, 3), np.nan)

    def ci(i):
        v = b[:, i][~np.isnan(b[:, i])] if len(b) else np.array([])
        return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
    return {"ROC AUC": (point[0], *ci(0)), "mean predicted": (point[1], *ci(1)),
            "actual RI rate": (point[2], *ci(2)), "boot2 used / skipped": f"{len(boots)} / {skipped}"}


def f3(t, nd=3):
    e, lo, hi = t
    g = lambda v: "-" if pd.isna(v) else f"{v:.{nd}f}"  # noqa: E731
    return f"{g(e)} [{g(lo)}, {g(hi)}]"


ROWS = ["storms", "rows", "RI rows", "RI episodes", "episodes caught (X of Y)", "episodes caught %",
        "median lead h", "recall (last-24h rise < 30 kt)", "false-alarm episodes / storm", "alert rate %",
        "PR-AUC", "ROC AUC", "Brier", "mean predicted", "actual RI rate", "boot used / skipped (evaluate)",
        "boot used / skipped (ROC, means)"]


def column(res_row: pd.Series, fmt_row: pd.Series, extra: dict) -> list[str]:
    return [str(int(res_row["storms"])), str(int(res_row["rows"])), str(int(res_row["RI rows"])),
            str(int(res_row["RI episodes"])),
            f"{int(res_row['episodes caught'])} of {int(res_row['RI episodes'])}",
            fmt_row["episodes caught %"], fmt_row["median lead h"], fmt_row["recall (last-24h rise < 30 kt)"],
            fmt_row["false-alarm episodes / storm"], fmt_row["alert rate %"], fmt_row["PR-AUC"],
            f3(extra["ROC AUC"]), fmt_row["Brier"], f3(extra["mean predicted"], 4), f3(extra["actual RI rate"], 4),
            fmt_row["boot used / skipped"], extra["boot2 used / skipped"]]


def subset_tables(runs: dict[str, tuple[pd.DataFrame, float]], subsets: list[str],
                  masks: dict) -> dict[str, pd.DataFrame]:
    """runs: name -> (rows to evaluate, threshold). Returns subset -> table (metrics x models)."""
    tables = {s: {} for s in subsets}
    for name, (df, col, thr) in runs.items():
        res = evaluate(df, col, thr)
        fmt = format_table(res)
        water = df[df["DIST2LAND"] > 0]
        for s in subsets:
            i = res.index[res["subset"] == s]
            if not len(i) or res.loc[i[0], "storms"] == 0:
                continue
            extra = roc_and_mean(water[masks[s](water).to_numpy()], col)
            tables[s][name] = column(res.loc[i[0]], fmt.loc[i[0]], extra)
    return {s: pd.DataFrame(t, index=ROWS) for s, t in tables.items()}


# ----------------------------------------------------------------------- episodes
def episode_list(water: pd.DataFrame, col: str, red: float) -> tuple[pd.DataFrame, list[str]]:
    eps, _ = episode_alerts(water, col, red)
    info = water.groupby("SID").agg(NAME=("NAME", "first"), SEASON=("SEASON", "first"))
    eps = eps.join(info, on="SID").sort_values("START").reset_index(drop=True)
    rows = []
    for _, e in eps.iterrows():
        inside = water[(water["SID"] == e["SID"]) & water["ISO_TIME"].between(e["START"], e["END"])]
        assert len(inside) == e["N_ROWS"] and (inside[TARGET] == 1).all(), "episode rows mismatch"
        red_in = inside[inside[col] >= red]
        if e["CAUGHT"]:
            first = -float(e["LEAD_H"])
        elif len(red_in):
            first = (red_in["ISO_TIME"].min() - e["START"]).total_seconds() / 3600
        else:
            first = np.nan
        rows.append({"NAME": e["NAME"], "SEASON": int(e["SEASON"]),
                     "SUBBASIN": inside["SUBBASIN"].iloc[0], "START": e["START"].strftime("%Y-%m-%d %H:%M"),
                     "ROWS": int(e["N_ROWS"]), "RED ROWS": len(red_in),
                     "ANY RED": "yes" if len(red_in) else "no", "RED AT FIRST ROW": "yes" if e["CAUGHT"] else "no",
                     "FIRST RED h": "" if np.isnan(first) else f"{first:+.0f}",
                     "MAX P IN EPISODE": round(float(inside[col].max()), 3)})
    t = pd.DataFrame(rows)
    n = len(t)
    totals = [f"episodes with an alert at the first row:          {int((t['RED AT FIRST ROW'] == 'yes').sum()) if n else 0} of {n}",
              f"episodes with an alert at any row in the episode: {int((t['ANY RED'] == 'yes').sum()) if n else 0} of {n}",
              f"episodes never alerted:                           {int((t['ANY RED'] == 'no').sum()) if n else 0} of {n}"]
    return t, totals


def casebook(ni: pd.DataFrame, red: float) -> list[str]:
    out = []
    for name, season in CASEBOOK:
        m = ni["NAME"].str.upper().str.contains(name, regex=False) & (ni["SEASON"] == season)
        sids = ni.loc[m, "SID"].unique()
        if not len(sids):
            out += [f"### {name} {season}", "", f"Not found in the NI test rows (name contains '{name}', SEASON {season}).", ""]
            continue
        for sid in sids:
            s = ni[ni["SID"] == sid].sort_values("ISO_TIME")
            t = pd.DataFrame({"ISO_TIME": s["ISO_TIME"].dt.strftime("%Y-%m-%d %H:%M"),
                              "SUBBASIN": s["SUBBASIN"], "USA_WIND": s["USA_WIND"].astype(int),
                              "DWIND_24": s["DWIND_24"].astype(int), "DIST2LAND": s["DIST2LAND"].astype(int),
                              "MAIN P": s["P_MAIN"].round(3),
                              "ALERT": [colour(p, red) if d > 0 else f"({colour(p, red)}, over land: not scored)"
                                        for p, d in zip(s["P_MAIN"], s["DIST2LAND"])],
                              "RI": s[TARGET].astype(int)})
            water = s[s["DIST2LAND"] > 0]
            eps, _ = episode_alerts(water, "P_MAIN", red) if len(water) else (pd.DataFrame(), None)
            caught = int(eps["CAUGHT"].sum()) if len(eps) else 0
            out += [f"### {s['NAME'].iloc[0]} {season} (SID {sid})", "",
                    f"{len(s)} test rows ({len(water)} over water); over-water RI episodes {len(eps)}, caught at the "
                    f"first row {caught}; peak USA_WIND {int(s['USA_WIND'].max())} kt.", "",
                    "```", t.to_string(index=False), "```", ""]
    return out


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-val", metavar="OUT_DIR", default=None,
                    help="run everything on VALIDATION rows and write outputs to OUT_DIR (no test data)")
    args = ap.parse_args(argv)
    dry = args.dry_run_val is not None
    out_md = os.path.join(args.dry_run_val, "results_dry_run_val.md") if dry else OUT_MD
    out_pred = os.path.join(args.dry_run_val, "predictions_dry_run_val.parquet") if dry else OUT_PRED
    if not dry and os.path.exists(OUT_MD):
        sys.exit(f"STOP: {OUT_MD} already exists. The test set is run once; not rerunning.")

    started = dt.datetime.now()
    freeze_txt = open(FREEZE, encoding="utf-8").read()
    hash_lines = check_hashes(freeze_txt)          # 1. before loading any data
    red_frozen = frozen_red(freeze_txt)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["main_model"] == "NI-ONLY", "MAIN in models/ri_lgbm_features.json is not NI-ONLY"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    # 2. thresholds, reproduced from VALIDATION only and checked against freeze.md
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    val_ni = val[val["BASIN"] == "NI"].copy()
    red = {}
    for key, m in meta["models"].items():
        val_ni["P_" + key] = predict(m["file"], m["features"], val_ni)
        thr, _ = threshold_for_alert_rate(val_ni, "P_" + key, ALERT_RATE)
        assert round(thr, 6) == red_frozen[key], f"{key}: validation threshold {thr} != freeze.md {red_frozen[key]}"
        red[key] = thr
    assert np.allclose(predict(meta["main_file"], meta["features"], val_ni), val_ni["P_NI-ONLY"])
    pers = fit_persistence(train)
    pthr = {}
    for label, (frozen, kt) in PERSIST_FROZEN.items():
        exact = float(predict_persistence(pers, pd.DataFrame({"DWIND_24": [float(kt)]}))[0])
        assert round(exact, 4) == frozen, f"{label}: exact {exact} does not round to frozen {frozen}"
        pthr[label] = exact
    thr_lines = [f"{FREEZE_NAMES[k]:<16} red {red[k]:.6f} (freeze.md {red_frozen[k]:.6f})  amber from {red[k] / 2:.6f}"
                 for k in red]
    thr_lines += [f"{k:<40} red {v:.6f} (freeze.md {PERSIST_FROZEN[k][0]})" for k, v in pthr.items()]

    # 3. the evaluation rows
    if dry:
        data, split_name = val, "VALIDATION 2016-2019 (DRY RUN, not the test result)"
    else:
        data, split_name = pd.read_parquet(TEST_PATH), "TEST 2020+"
        assert (data["SEASON"] >= 2020).all(), "test split has seasons before 2020"
        assert not set(data["SID"]) & (set(train["SID"]) | set(val["SID"])), "test SID also in train/val"
    data = data[data["DIST2LAND"] > 0].copy()                    # over water only
    for key, m in meta["models"].items():
        data["P_" + key] = predict(m["file"], m["features"], data)
    data["P_MAIN"] = data["P_NI-ONLY"]
    data["P_PERSISTENCE"] = predict_persistence(pers, data)
    ni = data[data["BASIN"] == "NI"].copy()

    masks = {SUBSET_NI: lambda d: d["BASIN"] == "NI",
             SUBSET_BOB: lambda d: (d["BASIN"] == "NI") & (d["SUBBASIN"] == "BB"),
             SUBSET_400: lambda d: d["DIST2LAND"] <= 400}
    labels = {"NI-ONLY": "MAIN (NI-ONLY)", "ALL-BASINS": "ALL-BASINS", "LEAVE-NI-OUT": "LEAVE-NI-OUT"}
    runs_ni = {labels[k]: (ni, "P_" + k, red[k]) for k in red}
    runs_ni.update({k: (ni, "P_PERSISTENCE", v) for k, v in pthr.items()})
    tables = subset_tables(runs_ni, [SUBSET_NI, SUBSET_BOB], masks)
    runs_400 = {labels[k]: (data, "P_" + k, red[k]) for k in ("NI-ONLY", "ALL-BASINS")}
    runs_400.update({k: (data, "P_PERSISTENCE", v) for k, v in pthr.items()})
    tables.update(subset_tables(runs_400, [SUBSET_400], masks))

    ep_table, ep_totals = episode_list(ni, "P_MAIN", red["NI-ONLY"])
    case_lines = casebook(ni, red["NI-ONLY"])
    montha = data["NAME"].str.upper().str.contains("MONTHA", regex=False).any()
    montha_line = ("MONTHA (2025) is in these rows." if montha else
                   "MONTHA (2025) is not in the test set: its data is provisional (TRACK_TYPE US-PROVISIONAL), so "
                   "cleaning rule 2 dropped all its rows (it is listed under rule 8). It is not scored.")

    # 4. outputs
    pred = ni.assign(ALERT=[colour(p, red["NI-ONLY"]) for p in ni["P_MAIN"]])
    pred = pred[["SID", "NAME", "SEASON", "SUBBASIN", "ISO_TIME", "LAT", "LON", "USA_WIND", "DIST2LAND",
                 "P_MAIN", "ALERT", "P_PERSISTENCE", TARGET]].rename(columns={TARGET: "RI"})
    pred = pred.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_pred) or ".", exist_ok=True)
    pred.to_parquet(out_pred, index=False)

    def tbl(s):
        return "\n".join(["```", tables[s].to_string(), "```"])

    md = "\n".join([
        f"# Test results: {split_name}", "",
        f"Run: {started.strftime('%a %d %b %Y %H:%M:%S')}  ", f"Command: `{CMD}{' --dry-run-val ' + args.dry_run_val if dry else ''}`  ",
        f"Git commit: `{commit}`  ", f"Freeze record: `{FREEZE}`", "",
        "## 1. Model files checked against docs/freeze.md (SHA-256)", "", "```", *hash_lines, "```", "",
        "## 2. Frozen thresholds (no threshold computed on test data)", "",
        "Each red threshold was reproduced from validation data with the same call the freeze used and checked "
        "to round to the freeze.md value; the exact value is used. Amber = red / 2. Persistence = logistic "
        "regression on DWIND_24 fitted on train 1980-2015 (src/eval/baselines.py), at its two frozen thresholds.", "",
        "```", *thr_lines, "```", "",
        "## 3. Metrics", "",
        "Over-water rows only (DIST2LAND > 0 at t). evaluate() from src/eval/metrics.py unchanged: episodes, lead "
        "time, false-alarm episodes, PR-AUC, Brier, 95% intervals from 1,000 bootstrap resamples of whole storms "
        "per subset (seed 0; resamples with no RI rows skipped, count shown). ROC AUC, mean predicted and actual RI "
        "rate: own storm bootstrap with the same rule (1,000 resamples, seed 0). Caught = red at the episode's first "
        "row. Lead time = hours the alert was red, unbroken, before the first RI row (src/eval/episodes.py). "
        "Hard-case recall = recall on RI rows whose last-24h wind rise was < 30 kt.", "",
        f"### {SUBSET_NI} (BASIN NI)", "", tbl(SUBSET_NI), "",
        f"### {SUBSET_BOB} (BASIN NI, SUBBASIN BB)", "", tbl(SUBSET_BOB), "",
        f"### Secondary check: {SUBSET_400} (MAIN, ALL-BASINS, persistence)", "",
        "MAIN is an NI-only model applied here to every basin; this is a secondary check, not a headline.", "",
        tbl(SUBSET_400), "",
        "## 4. MAIN: every NI RI episode", "",
        f"Red threshold {red['NI-ONLY']:.6f}. FIRST RED h = hours from the first RI row to the first red row: negative "
        "= red before the episode (equals minus the lead time; the run start is never earlier than the row after "
        "the storm's previous episode), 0 = red exactly at the first row, positive = first red row later inside the "
        "episode, blank = never red inside the episode.", "",
        "```", ep_table.to_string(index=False) if len(ep_table) else "(no RI episodes)", "```", "",
        "```", *ep_totals, "```", "",
        "## 5. Casebook storms in the test rows (MAIN)", "",
        "Every 6-hourly row of the storm in the test split. Rows over land are shown but not scored. Alert: red >= "
        f"{red['NI-ONLY']:.6f}, amber >= {red['NI-ONLY'] / 2:.6f}, else green.", "",
        *case_lines,
        montha_line, "",
        "## 6. Files", "",
        f"`{out_pred}`: {len(pred):,} over-water NI rows (SID, NAME, SEASON, SUBBASIN, ISO_TIME, LAT, LON, USA_WIND, "
        "DIST2LAND, P_MAIN, ALERT, P_PERSISTENCE, RI). P_PERSISTENCE = persistence probability (thresholds above).", "",
    ])
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[wrote {out_md} and {out_pred}]")


if __name__ == "__main__":
    main()
