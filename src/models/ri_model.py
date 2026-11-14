"""Three LightGBM RI models for the North Indian Ocean (NI), scored on NI validation.

The test split is NEVER loaded here. Fit on train (1980-2015), evaluate on validation
(2016-2019), over-water rows only (DIST2LAND > 0 at t).

Run from the project root:  .venv\\Scripts\\python -m src.models.ri_model

Models (identical LightGBM settings, see PARAMS):
  NI-ONLY       trained on NI train rows only;            BASIN is a categorical feature
  ALL-BASINS    trained on all train rows;                BASIN is a categorical feature
  LEAVE-NI-OUT  trained on all train rows except NI;      BASIN feature dropped
Early stopping for all three on validation log loss over NI over-water validation rows.
No class weights, no resampling. Missing USA_PRES / DPRES_24 left missing.

Pre-registered choice (written to docs/results_log.md once, never changed):
  MAIN = the higher validation PR-AUC on NI over-water rows of NI-ONLY vs ALL-BASINS;
  ties go to NI-ONLY. If the log already holds the choice, it is reused, and the run
  stops if a rerun would pick differently.

Saves models/ri_lgbm_ni.txt, ri_lgbm_all.txt, ri_lgbm_leaveni.txt, copies MAIN to
models/ri_lgbm.txt, writes models/ri_lgbm_features.json and docs/figures/reliability_val.png.
"""
import datetime as dt
import json
import os
import re
import shutil

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.features import BASINS, FEATURES, FUTURE_COLS, TARGET, TRACKS_PATH, build_features  # noqa: E402
from src.eval.baselines import fit_persistence, predict_persistence  # noqa: E402
from src.eval.metrics import evaluate, format_table, threshold_for_alert_rate  # noqa: E402

TRAIN_PATH = "data/processed/train.parquet"
VAL_PATH = "data/processed/val.parquet"
MAIN_PATH = "models/ri_lgbm.txt"
META_PATH = "models/ri_lgbm_features.json"
FIG_PATH = "docs/figures/reliability_val.png"
RESULTS_LOG = "docs/results_log.md"
CMD = r".venv\Scripts\python -m src.models.ri_model"

SEED = 0
ALERT_RATE = 0.20
PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 15,
    "min_child_samples": 20,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}
MAX_ROUNDS = 5000
EARLY_STOP = 100

FEATURES_NO_BASIN = [f for f in FEATURES if f != "BASIN"]
MODELS = {
    "NI-ONLY": {"file": "models/ri_lgbm_ni.txt", "features": FEATURES,
                "rows": lambda d: d["BASIN"] == "NI"},
    "ALL-BASINS": {"file": "models/ri_lgbm_all.txt", "features": FEATURES,
                   "rows": lambda d: pd.Series(True, index=d.index)},
    "LEAVE-NI-OUT": {"file": "models/ri_lgbm_leaveni.txt", "features": FEATURES_NO_BASIN,
                     "rows": lambda d: d["BASIN"] != "NI"},
}
CANDIDATES = ("NI-ONLY", "ALL-BASINS")  # the pre-registered comparison
MAIN_TAG = "PRE-REGISTERED MAIN MODEL:"

# Persistence as logged by Shaurya (docs/results_log.md, "Persistence at 20% alert rate" run,
# threshold 0.1285 set on ALL-basin validation rows; Brier from the baselines run).
PERSISTENCE_LOGGED = {
    "North Indian Ocean": {"PR-AUC": 0.154, "ROC AUC": None, "Brier": 0.0752, "mean predicted": None,
                           "alert rate %": 14.3, "caught": "1 of 11", "hard recall": 0.208, "FA eps/storm": 0.43},
    "Bay of Bengal": {"PR-AUC": 0.129, "ROC AUC": None, "Brier": 0.0863, "mean predicted": None,
                      "alert rate %": 12.8, "caught": "0 of 6", "hard recall": 0.083, "FA eps/storm": 0.29},
}

# Timestamps each feature reads, relative to t. Checked below: nothing after t.
FEATURE_TIMES = {
    "USA_WIND": ("t", "USA_WIND at t"),
    "USA_PRES": ("t", "USA_PRES at t"),
    "DWIND_6": ("t-6h, t", "USA_WIND(t) - USA_WIND(t-6h)"),
    "DWIND_12": ("t-12h, t", "USA_WIND(t) - USA_WIND(t-12h)"),
    "DWIND_24": ("t-24h, t", "USA_WIND(t) - USA_WIND(t-24h)"),
    "DPRES_24": ("t-24h, t", "USA_PRES(t) - USA_PRES(t-24h)"),
    "SPEED_12": ("t-12h, t", "great-circle km from position at t-12h to t, / 12 h"),
    "DIR_12": ("t-12h, t", "bearing from position at t-12h to t"),
    "SPEED_24": ("t-24h, t", "great-circle km from position at t-24h to t, / 24 h"),
    "DIR_24": ("t-24h, t", "bearing from position at t-24h to t"),
    "LAT": ("t", "latitude at t"),
    "DIST2LAND": ("t", "IBTrACS distance to land from the position at t"),
    "MONTH_SIN": ("t", "month of ISO_TIME at t"),
    "MONTH_COS": ("t", "month of ISO_TIME at t"),
    "BASIN": ("t", "BASIN code of the row at t"),
}
ALLOWED_TIMES = {"t", "t-6h", "t-12h", "t-24h"}

BOB = lambda d: (d["BASIN"] == "NI") & (d["SUBBASIN"] == "BB")  # noqa: E731


# --------------------------------------------------------------------------- data
def load_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and validation splits only. The test split is never read."""
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    assert train["SEASON"].between(1980, 2015).all(), "train has seasons outside 1980-2015"
    assert val["SEASON"].between(2016, 2019).all(), "val has seasons outside 2016-2019"
    assert not set(train["SID"]) & set(val["SID"]), "SID in both train and val"
    return train, val


def to_X(df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    X = df[features].copy()
    if "BASIN" in features:
        X["BASIN"] = pd.Categorical(X["BASIN"].astype(str), categories=BASINS)
        assert X["BASIN"].notna().all(), "unknown BASIN code"
    return X


# -------------------------------------------------------------------------- model
def fit(train: pd.DataFrame, es_rows: pd.DataFrame, features: list[str]) -> lgb.Booster:
    cat = ["BASIN"] if "BASIN" in features else []
    dtrain = lgb.Dataset(to_X(train, features), label=train[TARGET].to_numpy(),
                         categorical_feature=cat, free_raw_data=False)
    dval = lgb.Dataset(to_X(es_rows, features), label=es_rows[TARGET].to_numpy(),
                       categorical_feature=cat, reference=dtrain, free_raw_data=False)
    return lgb.train(PARAMS, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dval], valid_names=["val"],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def apply_platt(p, a: float, b: float, eps: float = 1e-6) -> np.ndarray:
    """Platt scaling, kept because app/app.py imports it: sigmoid(a * logit(p) + b)."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))


# ------------------------------------------------------------------------ leakage
def leakage_check(features: list[str], val_ni: pd.DataFrame, n: int = 200) -> list[str]:
    """Static map of timestamps per feature, plus an empirical truncation test on NI validation rows.

    Truncation test: keep each sampled storm's track only up to t, append a fake t+24h row
    with random wind/pressure/position, rebuild with build_features(), assert every feature
    at t is unchanged. Only validation seasons are read from tracks.parquet.
    """
    assert not set(features) & set(FUTURE_COLS + [TARGET]), "future/target column used as a feature"
    lines = []
    for f in features:
        times, how = FEATURE_TIMES[f]
        assert {x.strip() for x in times.split(",")} <= ALLOWED_TIMES, f"{f} uses a time after t"
        lines.append(f"{f:<10} uses {times:<9} ({how})")

    tracks = pd.read_parquet(TRACKS_PATH, filters=[("SEASON", ">=", 2016), ("SEASON", "<=", 2019)])
    assert tracks["SEASON"].between(2016, 2019).all()
    rng = np.random.default_rng(SEED)
    sample = val_ni.iloc[rng.choice(len(val_ni), size=min(n, len(val_ni)), replace=False)]
    for _, r in sample.iterrows():
        past = tracks[(tracks["SID"] == r["SID"]) & (tracks["ISO_TIME"] <= r["ISO_TIME"])].copy()
        fake = past.iloc[[-1]].copy()
        fake["ISO_TIME"] = r["ISO_TIME"] + pd.Timedelta(hours=24)
        fake["USA_WIND"] = rng.uniform(10, 180)
        fake["USA_PRES"] = rng.uniform(880, 1010)
        fake["LAT"], fake["LON"] = rng.uniform(-40, 40), rng.uniform(0, 360)
        rebuilt = build_features(pd.concat([past, fake], ignore_index=True), verbose=False)
        row = rebuilt[rebuilt["ISO_TIME"] == r["ISO_TIME"]]
        assert len(row) == 1, f"row not rebuilt for {r['SID']} {r['ISO_TIME']}"
        for f in features:
            a, b = row.iloc[0][f], r[f]
            assert (pd.isna(a) and pd.isna(b)) or a == b or abs(float(a) - float(b)) < 1e-9, \
                f"{f} changed when data after t was replaced ({r['SID']} {r['ISO_TIME']})"
    lines.append(f"truncation test: {len(sample)} random NI validation rows rebuilt from data up to t "
                 f"plus a random fake t+24h row -> all {len(features)} features identical")
    return lines


# ---------------------------------------------------------------------- reporting
def counts(d: pd.DataFrame) -> str:
    return f"rows {len(d):>6,}  storms {d['SID'].nunique():>4,}  RI rows {int(d[TARGET].sum()):>4,}"


def score(val_ni: pd.DataFrame, col: str) -> tuple[pd.DataFrame, float, float, pd.DataFrame]:
    """Threshold at 18.2% of NI over-water rows, then evaluate(); plus ROC AUC and mean predicted."""
    thr, got = threshold_for_alert_rate(val_ni, col, ALERT_RATE)
    res = evaluate(val_ni, col, thr).set_index("subset")
    water = val_ni[val_ni["DIST2LAND"] > 0]
    rows = []
    for name, mask in [("North Indian Ocean", pd.Series(True, index=water.index)), ("Bay of Bengal", BOB(water))]:
        w, r = water[mask], res.loc[name]
        rows.append({"subset": name, "storms": int(r["storms"]), "rows": int(r["rows"]),
                     "RI rows": int(r["RI rows"]), "RI episodes": int(r["RI episodes"]),
                     "PR-AUC": r["PR-AUC"], "ROC AUC": roc_auc_score(w[TARGET], w[col]),
                     "Brier": r["Brier"], "mean predicted": w[col].mean(), "actual RI rate": w[TARGET].mean(),
                     "alert rate %": r["alert rate %"],
                     "caught": f"{int(r['episodes caught'])} of {int(r['RI episodes'])}",
                     "hard recall": r["recall (last-24h rise < 30 kt)"],
                     "FA eps/storm": r["false-alarm episodes / storm"]})
    return pd.DataFrame(rows), thr, got, res.reset_index()


def side_by_side(tables: dict[str, pd.DataFrame], subset: str) -> pd.DataFrame:
    cols = ["PR-AUC", "ROC AUC", "Brier", "mean predicted", "actual RI rate", "alert rate %", "caught",
            "hard recall", "FA eps/storm"]
    out = {}
    for name, t in tables.items():
        r = t.set_index("subset").loc[subset]
        out[name] = [r[c] for c in cols]
    lg = PERSISTENCE_LOGGED[subset]
    out["persistence (logged)"] = [lg["PR-AUC"], lg["ROC AUC"], lg["Brier"], lg["mean predicted"],
                                   tables["NI-ONLY"].set_index("subset").loc[subset, "actual RI rate"],
                                   lg["alert rate %"], lg["caught"], lg["hard recall"], lg["FA eps/storm"]]
    df = pd.DataFrame(out, index=cols)

    def fmt(v, c):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        if isinstance(v, str):
            return v
        nd = {"Brier": 4, "mean predicted": 4, "actual RI rate": 4, "alert rate %": 1, "FA eps/storm": 2}.get(c, 3)
        return f"{v:.{nd}f}"
    return df.apply(lambda col: [fmt(v, c) for v, c in zip(col, df.index)])


def logged_main() -> str | None:
    if not os.path.exists(RESULTS_LOG):
        return None
    found = re.findall(re.escape(MAIN_TAG) + r"\s*([A-Z-]+)", open(RESULTS_LOG, encoding="utf-8").read())
    return found[0] if found else None


def reliability_plot(val_ni: pd.DataFrame, col: str, main: str, n_bins: int = 5) -> list[str]:
    water = val_ni[val_ni["DIST2LAND"] > 0]
    y, p = water[TARGET].to_numpy(), water[col].to_numpy()
    frac, mean_p = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5.5, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    lim = max(0.05, float(max(mean_p.max(), frac.max())) * 1.1)
    ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1, label="perfect calibration")
    ax.plot(mean_p, frac, "o-", c="#1f6feb", label=f"MAIN = {main} ({n_bins} quantile bins)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_ylabel("observed RI rate")
    ax.set_title(f"Reliability, NI validation 2016-2019, over water\nmean predicted {p.mean():.3f} vs actual "
                 f"{y.mean():.3f} (n = {len(y)} rows, {int(y.sum())} RI)")
    ax.legend(loc="upper left")
    axh.hist(p, bins=30, range=(0, lim), color="#8b949e")
    axh.set_xlabel("predicted RI probability")
    axh.set_ylabel("rows")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    return [f"bin {i + 1}: mean predicted {m:.3f}  observed {o:.3f}" for i, (m, o) in enumerate(zip(mean_p, frac))]


# --------------------------------------------------------------------------- main
def main() -> None:
    train, val = load_train_val()
    val_ni = val[val["BASIN"] == "NI"].copy()
    es_rows = val_ni[val_ni["DIST2LAND"] > 0]

    tr_ni = train[train["BASIN"] == "NI"]
    head_counts = [
        f"NI train (1980-2015, all rows):            {counts(tr_ni)}",
        f"Bay of Bengal train (1980-2015, all rows): {counts(tr_ni[BOB(tr_ni)])}",
        f"NI validation, over water:                 {counts(es_rows)}",
        f"Bay of Bengal validation, over water:      {counts(es_rows[BOB(es_rows)])}",
    ]
    print("\n".join(head_counts))
    print("Test split not loaded.\n")

    fits = {}
    for name, m in MODELS.items():
        tr = train[m["rows"](train).to_numpy()]
        booster = fit(tr, es_rows, m["features"])
        booster.save_model(m["file"], num_iteration=booster.best_iteration)
        reloaded = lgb.Booster(model_file=m["file"])
        assert reloaded.feature_name() == m["features"], f"{name}: saved feature order differs"
        val_ni["P_" + name] = reloaded.predict(to_X(val_ni, m["features"]))
        fits[name] = {"train rows": len(tr), "train storms": tr["SID"].nunique(), "train RI rows": int(tr[TARGET].sum()),
                      "best iteration": booster.best_iteration,
                      "val log loss": booster.best_score["val"]["binary_logloss"], "file": m["file"]}
        print(f"{name:<13} trained on {len(tr):>6,} rows / {tr['SID'].nunique():>5,} storms "
              f"({int(tr[TARGET].sum()):,} RI rows); best iteration {booster.best_iteration}, "
              f"NI val log loss {fits[name]['val log loss']:.5f} -> {m['file']}")

    # Persistence re-run live at the same NI alert rate (Shaurya's functions), for a like-for-like line.
    val_ni["P_PERSISTENCE"] = predict_persistence(fit_persistence(train), val_ni)

    tables, thr, full = {}, {}, {}
    for name in [*MODELS, "PERSISTENCE"]:
        t, th, got, res = score(val_ni, "P_" + name)
        tables[name], thr[name], full[name] = t, (th, got), res
    print("\nthresholds at 18.2% of NI over-water validation rows (threshold_for_alert_rate):")
    for name, (th, got) in thr.items():
        print(f"  {name:<13} threshold {th:.4f}  achieved alert rate {100 * got:.1f}%")

    models_only = {k: tables[k] for k in MODELS}
    sbs = {}
    for subset in ["North Indian Ocean", "Bay of Bengal"]:
        sbs[subset] = side_by_side({**models_only, "persistence (re-run, NI 18.2%)": tables["PERSISTENCE"]}, subset)
        print(f"\nVALIDATION 2016-2019, {subset}, over water")
        print(sbs[subset].to_string())

    # ---- pre-registered choice
    pr = {k: float(tables[k].set_index("subset").loc["North Indian Ocean", "PR-AUC"]) for k in CANDIDATES}
    winner = "NI-ONLY" if pr["NI-ONLY"] >= pr["ALL-BASINS"] else "ALL-BASINS"
    prior = logged_main()
    if prior is not None:
        assert prior == winner, (f"results_log.md already pre-registers MAIN = {prior}, but this run would pick "
                                 f"{winner}. The choice is not changed; stopping.")
        print(f"\nMAIN already pre-registered in {RESULTS_LOG}: {prior} (this run agrees; not rewritten)")
    choice_line = (f"NI-ONLY PR-AUC {pr['NI-ONLY']:.4f} vs ALL-BASINS PR-AUC {pr['ALL-BASINS']:.4f} "
                   f"(NI over-water validation rows; ties go to NI-ONLY) -> MAIN = {winner}")
    print(f"\n{MAIN_TAG} {winner}\n  {choice_line}")

    shutil.copyfile(MODELS[winner]["file"], MAIN_PATH)
    meta = {
        "main_model": winner, "main_file": MAIN_PATH, "main_copied_from": MODELS[winner]["file"],
        "features": MODELS[winner]["features"], "categorical_features": ["BASIN"] if "BASIN" in MODELS[winner]["features"] else [],
        "basin_categories": BASINS,
        "models": {k: {"file": m["file"], "features": m["features"],
                       "categorical_features": ["BASIN"] if "BASIN" in m["features"] else [],
                       "best_iteration": fits[k]["best iteration"]} for k, m in MODELS.items()},
        "note": "Build inputs as DataFrame[features] with BASIN = pd.Categorical(BASIN, categories=basin_categories).",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"copied {MODELS[winner]['file']} -> {MAIN_PATH}; wrote {META_PATH}")

    bins = reliability_plot(val_ni, "P_" + winner, winner)
    mt = tables[winner].set_index("subset").loc["North Indian Ocean"]
    calib = (f"MAIN ({winner}) on NI over-water validation: mean predicted RI {mt['mean predicted']:.4f} vs "
             f"actual RI rate {mt['actual RI rate']:.4f} (ratio {mt['mean predicted'] / mt['actual RI rate']:.2f})")
    print(f"\n{calib}; saved {FIG_PATH}")
    print("\n".join("  " + b for b in bins))

    leak = leakage_check(MODELS[winner]["features"], val_ni)
    print(f"\nfinal feature list of MAIN ({winner}) and timestamps used (none later than t):")
    print("\n".join("  " + x for x in leak))
    print(f"  LEAVE-NI-OUT uses the same features minus BASIN: {', '.join(FEATURES_NO_BASIN)}")

    # ---- log
    stamp = dt.datetime.now().strftime("%a %d %b %Y %H:%M")
    fit_tbl = pd.DataFrame(fits).T[["train rows", "train storms", "train RI rows", "best iteration", "val log loss", "file"]]
    full_txt = []
    for name in [*MODELS, "PERSISTENCE"]:
        keep = ["subset", "storms", "rows", "RI rows", "RI episodes", "episodes caught", "PR-AUC", "Brier",
                "recall (last-24h rise < 30 kt)", "false-alarm episodes / storm", "alert rate %", "boot used / skipped"]
        ft = format_table(full[name])
        ft = ft[ft["subset"].isin(["Bay of Bengal", "North Indian Ocean"])][keep]
        full_txt += [f"{name} (threshold {thr[name][0]:.4f}):", "```", ft.to_string(index=False), "```", ""]
    block = "\n".join([
        f"\n## {stamp}: three LightGBM models, North Indian Ocean scope (VALIDATION 2016-2019)", "",
        f"Command: `{CMD}`", "",
        "Scope set by the user's instruction of 14 Nov 2026 (North Indian Ocean, three models). CLAUDE.md on disk "
        "still says Bay of Bengal / one model / train on all basins; it was not edited.", "",
        f"Identical settings for all three: {json.dumps({k: v for k, v in PARAMS.items() if k != 'verbosity'})}, "
        f"max {MAX_ROUNDS} rounds, early stopping (patience {EARLY_STOP}) on log loss over NI over-water validation "
        "rows. No class weights, no resampling; missing USA_PRES / DPRES_24 left missing. LEAVE-NI-OUT drops BASIN. "
        "Thresholds: threshold_for_alert_rate(NI validation, prob, 0.182) per model. Early stopping, thresholds and "
        "scores all use the same NI validation rows, so these numbers are optimistic. Test split not loaded.", "",
        "Counts:", "", "```", *head_counts, "```", "",
        "Training:", "", "```", fit_tbl.to_string(), "```", "",
        "NI over-water validation (point estimates). 'persistence (logged)' = Shaurya's run at threshold 0.1285, set "
        "on ALL-basin validation rows (NI alert rate 14.3%, not 18.2%); ROC AUC and mean predicted were not logged. "
        "'persistence (re-run, NI 18.2%)' = same persistence model, threshold set on NI rows like the models:", "",
        "```", sbs["North Indian Ocean"].to_string(), "```", "",
        "Bay of Bengal subset (same thresholds as NI):", "", "```", sbs["Bay of Bengal"].to_string(), "```", "",
        "With 95% storm-bootstrap intervals (1,000 resamples per subset, seed 0; resamples with no RI rows skipped):", "",
        *full_txt,
        f"**{MAIN_TAG} {winner}** (decided {stamp}; pre-registered rule, not to be changed). {choice_line}. "
        f"Copied {MODELS[winner]['file']} to {MAIN_PATH}.", "",
        f"Calibration: {calib}. Reliability plot: {FIG_PATH} (5 quantile bins):", "",
        "```", *bins, "```", "",
        f"Leakage check (MAIN features and the timestamps they use):", "", "```", *leak, "```", "",
    ])
    with open(RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(block + "\n")
    print(f"\n[appended to {RESULTS_LOG}]")


if __name__ == "__main__":
    main()
