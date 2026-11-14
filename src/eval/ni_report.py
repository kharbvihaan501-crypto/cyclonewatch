"""North Indian Ocean validation report: persistence alert-rate steps and MAIN episode detail.

The test split is never loaded. Saved models are only read (no retraining).

Run from the project root:  .venv\\Scripts\\python -m src.eval.ni_report
  add --alert-rate 0.20 to score the three LightGBM models at a 20% NI alert rate (with CIs).

(1) Persistence's probabilities come in steps (DWIND_24 in 5 kt steps), so the 18.1% alert rate
    of the LightGBM models is not reachable. Print the reachable NI over-water alert rates just
    below and just above 18.1%, and add persistence at the step ABOVE to the NI and Bay of
    Bengal tables next to the three LightGBM models.
(2) MAIN at its 18.1% threshold: lead time per caught RI episode and the list of missed
    episodes (NI over-water validation rows). Lead time as in src/eval/episodes.py: hours from
    the start of the unbroken red run to the episode's first row; 0 h if red only at that row.
"""
import datetime as dt
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.features import BASINS, TARGET
from src.eval.baselines import fit_persistence, predict_persistence
from src.eval.episodes import episode_alerts
from src.eval.metrics import evaluate, threshold_for_alert_rate

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
META_PATH = "models/ri_lgbm_features.json"
RESULTS_LOG = "docs/results_log.md"
CMD = r".venv\Scripts\python -m src.eval.ni_report"
ALERT_RATE = 0.182  # same call as in src/models/ri_model.py; achieves 18.1% on NI rows
SUBSETS = ["North Indian Ocean", "Bay of Bengal"]
BOB = lambda d: (d["BASIN"] == "NI") & (d["SUBBASIN"] == "BB")  # noqa: E731


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train and validation only; the test split is never read."""
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    assert train["SEASON"].between(1980, 2015).all() and val["SEASON"].between(2016, 2019).all()
    assert not set(train["SID"]) & set(val["SID"]), "SID in both train and val"
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return train, val[val["BASIN"] == "NI"].copy(), meta


def predict(model_file: str, features: list[str], df: pd.DataFrame) -> np.ndarray:
    booster = lgb.Booster(model_file=model_file)
    assert booster.feature_name() == features, f"{model_file}: feature order differs from {META_PATH}"
    X = df[features].copy()
    if "BASIN" in features:
        X["BASIN"] = pd.Categorical(X["BASIN"].astype(str), categories=BASINS)
    return booster.predict(X)


def alert_steps(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Every distinct threshold and the share of NI over-water rows it puts on red (p >= threshold)."""
    p = df.loc[df["DIST2LAND"] > 0, col].to_numpy()
    cand = np.unique(p)
    return pd.DataFrame({"threshold": cand, "alert rate": [(p >= c).mean() for c in cand]})


def score(df: pd.DataFrame, col: str, thr: float) -> pd.DataFrame:
    res = evaluate(df, col, thr).set_index("subset")
    water = df[df["DIST2LAND"] > 0]
    out = {}
    for name, mask in [("North Indian Ocean", pd.Series(True, index=water.index)), ("Bay of Bengal", BOB(water))]:
        w, r = water[mask], res.loc[name]
        out[name] = {"threshold": thr, "PR-AUC": r["PR-AUC"], "ROC AUC": roc_auc_score(w[TARGET], w[col]),
                     "Brier": r["Brier"], "mean predicted": w[col].mean(), "actual RI rate": w[TARGET].mean(),
                     "alert rate %": r["alert rate %"],
                     "caught": f"{int(r['episodes caught'])} of {int(r['RI episodes'])}",
                     "hard recall": r["recall (last-24h rise < 30 kt)"],
                     "FA eps/storm": r["false-alarm episodes / storm"],
                     "storms": int(r["storms"]), "RI episodes": int(r["RI episodes"])}
    return pd.DataFrame(out)


def fmt_table(cols: dict[str, pd.Series]) -> pd.DataFrame:
    order = ["threshold", "PR-AUC", "ROC AUC", "Brier", "mean predicted", "actual RI rate", "alert rate %",
             "caught", "hard recall", "FA eps/storm", "storms", "RI episodes"]
    nd = {"threshold": 4, "Brier": 4, "mean predicted": 4, "actual RI rate": 4, "alert rate %": 1,
          "FA eps/storm": 2}

    def f(v, k):
        if isinstance(v, str):
            return v
        if isinstance(v, (int, np.integer)):
            return str(v)
        return "-" if pd.isna(v) else f"{v:.{nd.get(k, 3)}f}"
    return pd.DataFrame({name: [f(s[k], k) for k in order] for name, s in cols.items()}, index=order)


def episode_detail(df: pd.DataFrame, col: str, thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MAIN episodes on NI over-water rows: caught (with lead time) and missed."""
    water = df[df["DIST2LAND"] > 0]
    eps, _ = episode_alerts(water, col, thr)
    info = water.groupby("SID").agg(NAME=("NAME", "first"), SEASON=("SEASON", "first"))
    first = water.set_index(["SID", "ISO_TIME"])
    eps = eps.join(info, on="SID")
    eps["SUBBASIN"] = [first.loc[(s, t), "SUBBASIN"] for s, t in zip(eps["SID"], eps["START"])]
    eps["P_AT_START"] = [first.loc[(s, t), col] for s, t in zip(eps["SID"], eps["START"])]
    eps["MAX_P_BEFORE_END"] = [water.loc[(water["SID"] == s) & (water["ISO_TIME"] <= e), col].max()
                               for s, e in zip(eps["SID"], eps["END"])]
    cols = ["NAME", "SEASON", "SID", "SUBBASIN", "START", "END", "N_ROWS", "P_AT_START"]
    caught = eps[eps["CAUGHT"]][cols + ["LEAD_H"]].sort_values("START").reset_index(drop=True)
    missed = eps[~eps["CAUGHT"]][cols + ["MAX_P_BEFORE_END"]].sort_values("START").reset_index(drop=True)
    return caught, missed


def main() -> None:
    train, val_ni, meta = load()
    main_name, features = meta["main_model"], meta["features"]
    models = meta["models"]
    for name, m in models.items():
        val_ni["P_" + name] = predict(m["file"], m["features"], val_ni)
    assert np.allclose(val_ni["P_" + main_name], predict(meta["main_file"], features, val_ni)), \
        "models/ri_lgbm.txt does not match the MAIN model file"
    val_ni["P_PERSISTENCE"] = predict_persistence(fit_persistence(train), val_ni)

    # ---- (1) persistence steps around MAIN's achieved alert rate
    thr_main, rate_main = threshold_for_alert_rate(val_ni, "P_" + main_name, ALERT_RATE)
    steps = alert_steps(val_ni, "P_PERSISTENCE")
    below = steps[steps["alert rate"] < rate_main].sort_values("alert rate").iloc[-1]
    above = steps[steps["alert rate"] > rate_main].sort_values("alert rate").iloc[0]
    dw = lambda thr: int(round(val_ni.loc[val_ni["P_PERSISTENCE"] >= thr, "DWIND_24"].min()))  # noqa: E731
    step_lines = [
        f"MAIN ({main_name}) threshold {thr_main:.4f} -> alert rate {100 * rate_main:.2f}% of NI over-water validation rows",
        f"persistence step below: threshold {below['threshold']:.4f} (DWIND_24 >= {dw(below['threshold'])} kt) "
        f"-> alert rate {100 * below['alert rate']:.2f}%",
        f"persistence step above: threshold {above['threshold']:.4f} (DWIND_24 >= {dw(above['threshold'])} kt) "
        f"-> alert rate {100 * above['alert rate']:.2f}%   <- added to the tables",
    ]
    print("\n".join(step_lines))

    cols = {}
    for name in models:
        thr, _ = threshold_for_alert_rate(val_ni, "P_" + name, ALERT_RATE)
        cols[name + (" (MAIN)" if name == main_name else "")] = score(val_ni, "P_" + name, thr)
    pers_label = f"persistence @ {100 * above['alert rate']:.1f}%"
    cols[pers_label] = score(val_ni, "P_PERSISTENCE", float(above["threshold"]))
    tables = {s: fmt_table({k: v[s] for k, v in cols.items()}) for s in SUBSETS}
    for s in SUBSETS:
        print(f"\nVALIDATION 2016-2019, {s}, over water (thresholds set on NI over-water rows)")
        print(tables[s].to_string())

    # ---- (2) MAIN episode detail
    caught, missed = episode_detail(val_ni, "P_" + main_name, thr_main)
    show = lambda d: d.assign(START=d["START"].dt.strftime("%Y-%m-%d %H:%M"),  # noqa: E731
                              END=d["END"].dt.strftime("%Y-%m-%d %H:%M")).round(3).to_string(index=False)
    print(f"\nMAIN ({main_name}) at threshold {thr_main:.4f}: {len(caught)} caught of {len(caught) + len(missed)} "
          "RI episodes (NI over-water validation rows)")
    print("caught episodes, lead time in hours:")
    print(show(caught))
    print("\nmissed episodes:")
    print(show(missed))

    stamp = dt.datetime.now().strftime("%a %d %b %Y %H:%M")
    block = "\n".join([
        f"\n## {stamp}: persistence alert-rate steps and MAIN episode detail, NI (VALIDATION 2016-2019)", "",
        f"Command: `{CMD}`", "",
        "NI over-water validation rows only. Saved models read from models/ (no retraining); persistence = "
        "logistic regression on DWIND_24 fitted on train (src/eval/baselines.py). LightGBM thresholds from "
        "threshold_for_alert_rate(NI validation, prob, 0.182). Test split not loaded.", "",
        "Persistence alert-rate steps around MAIN's alert rate:", "", "```", *step_lines, "```", "",
        "North Indian Ocean (persistence added at the step above):", "", "```", tables["North Indian Ocean"].to_string(),
        "```", "",
        "Bay of Bengal subset (same thresholds):", "", "```", tables["Bay of Bengal"].to_string(), "```", "",
        f"MAIN ({main_name}) at threshold {thr_main:.4f}: {len(caught)} caught of {len(caught) + len(missed)} RI "
        "episodes. Lead time = hours from the start of the unbroken red run to the episode's first row "
        "(0 h = red only at the first RI row), as in src/eval/episodes.py. P_AT_START = MAIN probability at the "
        "episode's first row; MAX_P_BEFORE_END = highest MAIN probability in that storm up to the episode's end.", "",
        "Caught:", "", "```", show(caught), "```", "", "Missed:", "", "```", show(missed), "```", "",
    ])
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(block + "\n")
    print(f"\n[appended to {RESULTS_LOG}]")


# --------------------------------------------------------------- --alert-rate mode
def run_rate(rate: float) -> None:
    """Three LightGBM models at threshold_for_alert_rate(NI validation, prob, rate), with storm-bootstrap CIs,
    plus MAIN episode detail. No persistence rows; saved models only; test split never read."""
    from src.eval.metrics import format_table

    _, val_ni, meta = load()
    main_name, models = meta["main_model"], meta["models"]
    for name, m in models.items():
        val_ni["P_" + name] = predict(m["file"], m["features"], val_ni)
    assert np.allclose(val_ni["P_" + main_name], predict(meta["main_file"], meta["features"], val_ni)), \
        "models/ri_lgbm.txt does not match the MAIN model file"

    keep = ["subset", "storms", "RI episodes", "episodes caught", "alert rate %", "episodes caught %",
            "recall (last-24h rise < 30 kt)", "false-alarm episodes / storm", "PR-AUC", "boot used / skipped"]
    thr_lines, blocks, summary = [], [], []
    thr = {}
    for name in models:
        col = "P_" + name
        thr[name], got = threshold_for_alert_rate(val_ni, col, rate)
        label = name + (" (MAIN)" if name == main_name else "")
        thr_lines.append(f"{label:<16} threshold {thr[name]:.4f}  achieved alert rate {100 * got:.2f}% of NI over-water rows")
        res = evaluate(val_ni, col, thr[name])
        res = res[res["subset"].isin(SUBSETS)]
        table = format_table(res)[keep]
        blocks += [f"{label}:", table.to_string(index=False), ""]
        for _, r in res.iterrows():
            lo, hi = r["recall (last-24h rise < 30 kt) lo"], r["recall (last-24h rise < 30 kt) hi"]
            summary.append({"model": label, "subset": r["subset"],
                            "caught": f"{int(r['episodes caught'])} of {int(r['RI episodes'])}",
                            "caught % [95% CI]": f"{r['episodes caught %']:.1f} [{r['episodes caught % lo']:.1f}, {r['episodes caught % hi']:.1f}]",
                            "hard recall [95% CI]": f"{r['recall (last-24h rise < 30 kt)']:.3f} [{lo:.3f}, {hi:.3f}]",
                            "FA eps/storm [95% CI]": f"{r['false-alarm episodes / storm']:.2f} [{r['false-alarm episodes / storm lo']:.2f}, {r['false-alarm episodes / storm hi']:.2f}]",
                            "PR-AUC [95% CI]": f"{r['PR-AUC']:.3f} [{r['PR-AUC lo']:.3f}, {r['PR-AUC hi']:.3f}]",
                            "alert rate %": f"{r['alert rate %']:.1f}"})
    summary = pd.DataFrame(summary).sort_values(["subset", "model"], ascending=[False, True])
    print("\n".join(thr_lines))
    for s in SUBSETS:
        print(f"\nVALIDATION 2016-2019, {s}, over water, {100 * rate:g}% alert-rate thresholds (set on NI rows)")
        print(summary[summary["subset"] == s].drop(columns="subset").to_string(index=False))

    caught, missed = episode_detail(val_ni, "P_" + main_name, thr[main_name])
    show = lambda d: d.assign(START=d["START"].dt.strftime("%Y-%m-%d %H:%M"),  # noqa: E731
                              END=d["END"].dt.strftime("%Y-%m-%d %H:%M")).round(3).to_string(index=False)
    head = (f"MAIN ({main_name}) at threshold {thr[main_name]:.4f}: {len(caught)} caught of "
            f"{len(caught) + len(missed)} RI episodes (NI over-water validation rows)")
    print("\n" + head + "\ncaught episodes, lead time in hours:\n" + show(caught) + "\n\nmissed episodes:\n" + show(missed))

    now = dt.datetime.now()
    cmd = f"{CMD} --alert-rate {rate:g}"
    block = "\n".join([
        f"\n## Alert rate changed from 18.2% to {100 * rate:g}% on {now.strftime('%a %d %b %Y')}, "
        f"{now.strftime('%H:%M')}, after seeing validation numbers; validation-only decision, test not touched.", "",
        f"Command: `{cmd}`", "",
        "NI over-water validation rows (2016-2019) only. Saved models read from models/ (no retraining; MAIN = "
        f"{main_name}, pre-registered choice unchanged). Thresholds: threshold_for_alert_rate(NI validation, prob, "
        f"{rate:g}) per model. 95% CI: 1,000 bootstrap resamples of whole storms per subset (src/eval/metrics.py, "
        "seed 0); resamples with no RI rows skipped. Hard recall = recall on RI rows where the last-24h wind rose "
        "< 30 kt. Test split not loaded.", "",
        "```", *thr_lines, "```", "",
        "Summary:", "", "```", summary.to_string(index=False), "```", "",
        "Full evaluate() tables:", "", "```", *blocks, "```", "",
        f"{head}. Lead time = hours from the start of the unbroken red run to the episode's first row (0 h = red "
        "only at the first RI row), as in src/eval/episodes.py.", "",
        "Caught:", "", "```", show(caught), "```", "", "Missed:", "", "```", show(missed), "```", "",
    ])
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(block + "\n")
    print(f"\n[appended to {RESULTS_LOG}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alert-rate", type=float, default=None,
                    help="score the three LightGBM models at this NI alert rate (0-1), with CIs and MAIN episodes")
    args = ap.parse_args()
    run_rate(args.alert_rate) if args.alert_rate is not None else main()
