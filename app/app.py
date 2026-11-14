"""CycloneWatch: will this cyclone suddenly strengthen in the next 24 hours?

Tab 1 forecasts from bulletin-style inputs; Tab 2 replays a past North Indian Ocean storm
to check the model against what actually happened.

Model: models/ri_lgbm.txt (LightGBM) if it exists, else the persistence baseline
(logistic regression on the last-24h wind change) fitted on the train split.
LightGBM probabilities are Platt-calibrated with models/ri_calibration.json when that
file exists and was fitted for the same model file; otherwise raw probabilities are shown.
The red-alert threshold is set on the VALIDATION split so that ALERT_RATE of
North Indian Ocean (BASIN NI) over-water validation rows are red; amber = half of red
(docs/freeze.md). Tab 2 uses data/processed/predictions.parquet (written by the frozen test
run) for the storms it contains, and computes other storms live.
test.parquet is NEVER read here.

Run from the project root:  py -3.12 app\\app.py
"""
import hashlib
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # src modules use paths relative to the project root
from src.data.features import (BASINS, FEATURES, RI_THRESHOLD_KT,  # noqa: E402
                               _bearing_deg, _haversine_km, _merge_offset)
from src.eval.baselines import TRAIN_PATH, VAL_PATH, fit_persistence  # noqa: E402
from src.eval.episodes import episode_alerts  # noqa: E402
from src.eval.metrics import threshold_for_alert_rate  # noqa: E402
from src.models.ri_model import apply_platt  # noqa: E402

ALERT_RATE = 0.20  # red threshold = the one that makes 20% of NI over-water validation rows red
THRESHOLD_BASIN = "NI"  # the main model is trained and checked on the North Indian Ocean only
AMBER_SHARE = 0.5  # amber from half of the red threshold (CLAUDE.md, docs/freeze.md)
MODEL_PATH = Path(os.environ.get("CYCLONEWATCH_MODEL", ROOT / "models" / "ri_lgbm.txt"))
CAL_PATH = MODEL_PATH.with_name("ri_calibration.json")
FEATURES_JSON = MODEL_PATH.with_name("ri_lgbm_features.json")  # exact feature order + BASIN categories
TRACKS_PATH = ROOT / "data" / "processed" / "tracks.parquet"
PRED_PATH = Path(os.environ.get("CYCLONEWATCH_PREDICTIONS",
                                ROOT / "data" / "processed" / "predictions.parquet"))
FREEZE_MD = ROOT / "docs" / "freeze.md"
PERS_RED = 0.1285  # frozen persistence threshold (docs/freeze.md): red when last-24h rise >= 25 kt
DEFAULT_STORM = ("AMPHAN", 2020)
# Odisha casebook (CLAUDE.md). SIDs checked in tracks.parquet; the 1999 storm is UNNAMED there.
CASEBOOK = [("1999 super cyclone", "1999298N12099"), ("Phailin 2013", "2013281N12098"),
            ("Hudhud 2014", "2014279N11096"), ("Titli 2018", "2018281N14088"),
            ("Fani 2019", "2019116N02090"), ("Bulbul 2019", "2019302N11118"),
            ("Amphan 2020", "2020136N10088"), ("Yaas 2021", "2021143N15090"),
            ("Gulab 2021", "2021267N18094"), ("Jawad 2021", "2021335N08096"),
            ("Dana 2024", "2024295N15092")]
CASEBOOK_NOTE = ("Montha 2025 is not shown: its data is provisional (US-PROVISIONAL), so cleaning "
                 "rule 2 dropped it.")
TITLE = "CycloneWatch: will this cyclone suddenly strengthen in the next 24 hours?"
DISCLAIMER = ("Prototype for research and education. Not an official forecast. "
              "Always follow IMD and your state disaster authority.")
COLOURS = {"green": "#2e9d4f", "amber": "#f0a202", "red": "#d62828", "none": "#9e9e9e"}
EXAMPLE = {"wind": 70, "pres": 978, "lat": 11.5, "lon": 86.2, "dist": 630, "month": 5,
           "w6": 60, "w12": 55, "w24": 40, "p24": 997}


# ===================================================================== features
def _finish_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns exactly as in src/data/features.py (from lag columns already attached)."""
    for h in (6, 12, 24):
        df[f"DWIND_{h}"] = df["USA_WIND"] - df[f"USA_WIND_M{h}"]
    df["DPRES_24"] = df["USA_PRES"] - df["USA_PRES_M24"]
    for h in (12, 24):
        df[f"SPEED_{h}"] = _haversine_km(df[f"LAT_M{h}"], df[f"LON_M{h}"], df["LAT"], df["LON"]) / h
        df[f"DIR_{h}"] = _bearing_deg(df[f"LAT_M{h}"], df[f"LON_M{h}"], df["LAT"], df["LON"])
    month = df["ISO_TIME"].dt.month if "ISO_TIME" in df else df["MONTH"]
    df["MONTH_SIN"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["BASIN"] = pd.Categorical(df["BASIN"], categories=BASINS)
    return df


def storm_features(storm: pd.DataFrame) -> pd.DataFrame:
    """Features for every row of one storm, using only rows at or before t.

    Lags matched by timestamp (features.py helper, gaps asserted). Rows without a
    t-24h row get HAS_HISTORY = False and no forecast.
    """
    df = storm.copy()
    for h in (-6, -12, -24):
        df = _merge_offset(df, h, ["USA_WIND", "USA_PRES", "LAT", "LON"])
    df["HAS_HISTORY"] = df["TIME_M24"].notna()
    return _finish_features(df)


def bulletin_features(v: dict) -> pd.DataFrame:
    """One feature row from bulletin inputs. Past positions are not given, so storm
    speed and direction are missing (LightGBM handles missing values). Basin = NI."""
    row = {"USA_WIND": v["wind"], "USA_PRES": v["pres"], "LAT": v["lat"], "LON": v["lon"],
           "DIST2LAND": v["dist"], "MONTH": v["month"], "BASIN": "NI",
           "USA_WIND_M6": v["w6"], "USA_WIND_M12": v["w12"], "USA_WIND_M24": v["w24"],
           "USA_PRES_M24": v["p24"]}
    for h in (12, 24):
        row[f"LAT_M{h}"] = np.nan
        row[f"LON_M{h}"] = np.nan
    return _finish_features(pd.DataFrame([row]).astype({"MONTH": int}))


# ======================================================================== model
GROUPS = {"MONTH_SIN": "MONTH", "MONTH_COS": "MONTH", "SPEED_12": "MOTION", "DIR_12": "MOTION",
          "SPEED_24": "MOTION", "DIR_24": "MOTION"}


def _describe(key: str, r: pd.Series) -> str:
    def f(x, fmt):
        return "unknown" if pd.isna(x) else format(x, fmt)
    return {
        "DWIND_24": f"Wind changed {f(r['DWIND_24'], '+.0f')} kt in the last 24 hours",
        "DWIND_12": f"Wind changed {f(r['DWIND_12'], '+.0f')} kt in the last 12 hours",
        "DWIND_6": f"Wind changed {f(r['DWIND_6'], '+.0f')} kt in the last 6 hours",
        "USA_WIND": f"Current wind is {f(r['USA_WIND'], '.0f')} kt",
        "USA_PRES": f"Central pressure is {f(r['USA_PRES'], '.0f')} hPa",
        "DPRES_24": f"Pressure changed {f(r['DPRES_24'], '+.0f')} hPa in the last 24 hours",
        "LAT": f"Storm is at latitude {f(r['LAT'], '.1f')}°N",
        "DIST2LAND": f"Storm is {f(r['DIST2LAND'], '.0f')} km from land",
        "MONTH": "Time of year",
        "MOTION": "How fast and which way the storm is moving",
        "BASIN": "Ocean basin",
    }.get(key, key)


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def _load_lgbm():
    """Load and check the LightGBM model + feature JSON. Raises with a clear message on any problem."""
    import json
    import lightgbm as lgb
    missing = [_rel(p) for p in (MODEL_PATH, FEATURES_JSON) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing file: " + ", ".join(missing))
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    spec = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
    if Path(spec["main_file"]).name != MODEL_PATH.name:
        raise ValueError(f"{FEATURES_JSON.name} describes {spec['main_file']}, not {MODEL_PATH.name}")
    if list(spec["features"]) != booster.feature_name():
        raise ValueError(f"feature order in {FEATURES_JSON.name} differs from {MODEL_PATH.name}")
    return booster, spec


class Model:
    """LightGBM from MODEL_PATH + FEATURES_JSON; if either is missing or fails to load,
    persistence fitted on train, with a console warning and a red banner (never silent)."""

    def __init__(self):
        self.fallback_reason = None
        try:
            self.booster, spec = _load_lgbm()
        except Exception as e:  # noqa: BLE001 - any load failure must fall back loudly
            self.booster = None
            self.fallback_reason = f"{type(e).__name__}: {e}"
            print("\n" + "!" * 78, file=sys.stderr)
            print(f"WARNING: LightGBM model NOT loaded ({self.fallback_reason}).", file=sys.stderr)
            print(f"WARNING: expected {_rel(MODEL_PATH)} and {_rel(FEATURES_JSON)}. "
                  "Using the PERSISTENCE BASELINE instead.", file=sys.stderr)
            print("!" * 78 + "\n", file=sys.stderr, flush=True)
        if self.booster is not None:
            self.kind = "lgbm"
            self.cols = list(spec["features"])
            self.cat_cols = list(spec.get("categorical_features", []))
            self.categories = {"BASIN": list(spec["basin_categories"])}
            self.short = f"LightGBM {spec.get('main_model', 'main model')}"
            self.file_time = pd.Timestamp.fromtimestamp(MODEL_PATH.stat().st_mtime)
            self.name = f"{self.short} ({MODEL_PATH.name})"
            self.cal = None
            if CAL_PATH.exists():
                import json
                cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
                if Path(cal["model_file"]).name == MODEL_PATH.name:
                    self.cal = cal
                    self.name += ", Platt-calibrated on 2016-2019 validation storms"
        else:
            self.kind = "persistence"
            self.cols, self.cat_cols, self.categories = ["DWIND_24"], [], {}
            self.pers = fit_persistence(pd.read_parquet(TRAIN_PATH))
            self.short = "persistence baseline (model file missing)"
            self.name = "Persistence baseline (last-24h wind change only, fitted on 1980-2015)"
        val = pd.read_parquet(VAL_PATH)  # validation only, for the threshold
        val = val[val["BASIN"] == THRESHOLD_BASIN].copy()
        val["P"] = self.predict(val)
        self.threshold, self.val_rate = threshold_for_alert_rate(val, "P", ALERT_RATE)
        self.amber_from = AMBER_SHARE * self.threshold

    def matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Model input: exactly self.cols in order. A column X lacks becomes NaN; categorical
        columns use the categories from the feature JSON."""
        M = pd.DataFrame(index=X.index)
        for c in self.cols:
            col = X[c] if c in X else pd.Series(np.nan, index=X.index)
            if c in self.cat_cols:
                M[c] = pd.Categorical(col.astype(object), categories=self.categories[c])
            else:
                M[c] = pd.to_numeric(col, errors="coerce").astype(float)
        return M

    def missing_inputs(self, X: pd.DataFrame) -> list[str]:
        """Features of this model that are missing (NaN) in the first row of X."""
        M = self.matrix(X.iloc[[0]])
        return [c for c in self.cols if pd.isna(M[c].iloc[0])]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is not None:
            p = self.booster.predict(self.matrix(X))
            return apply_platt(p, self.cal["a"], self.cal["b"]) if self.cal else p
        return self.pers.predict_proba(self.matrix(X).to_numpy())[:, 1]

    def reasons(self, X: pd.DataFrame) -> list[str]:
        """Top 3 drivers for ONE row, in plain words."""
        r = X.iloc[0]
        if self.booster is None:
            d = r["DWIND_24"]
            ref = self.pers.predict_proba([[0.0]])[0, 1]
            way = "raises" if d > 0 else "lowers" if d < 0 else "does not change"
            return [f"{_describe('DWIND_24', r)}: this {way} the chance "
                    f"(a storm holding steady would get {ref:.0%}).",
                    "This fallback model looks only at the last-24h wind change.",
                    "Pressure, position, distance to land and season are not used until the "
                    "trained model is added."]
        contrib = self.booster.predict(self.matrix(X.iloc[[0]]), pred_contrib=True)[0][:-1]
        by_group: dict[str, float] = {}
        for c, v in zip(self.cols, contrib):
            k = GROUPS.get(c, c)
            by_group[k] = by_group.get(k, 0.0) + float(v)
        top = sorted(by_group.items(), key=lambda kv: -abs(kv[1]))[:3]
        return [f"{_describe(k, r)}: {'raises' if v > 0 else 'lowers'} the chance" for k, v in top]

    def level(self, p: float) -> str:
        if pd.isna(p):
            return "none"
        if p >= self.threshold:
            return "red"
        return "amber" if p >= self.amber_from else "green"


MODEL = Model()


def banner_html() -> str:
    """'Model in use' banner shown at the top of every tab: blue for LightGBM, red for the fallback."""
    if MODEL.kind == "lgbm":
        bg, sub = "#1d3557", (f"{MODEL.name} · model file modified "
                              f"{MODEL.file_time:%d %b %Y %H:%M} (local time)")
    else:
        bg, sub = "#d62828", (f"{MODEL.name}. Reason: {MODEL.fallback_reason}. "
                              "Forecasts below are NOT from the trained model.")
    # !important: Gradio's own text colours otherwise override inline white text.
    return (f"<div style='background:{bg} !important;border-radius:10px;padding:12px 18px;margin-bottom:6px'>"
            f"<div style='color:#ffffff !important;font-size:22px;font-weight:800;line-height:1.3'>"
            f"Model in use: {MODEL.short}</div>"
            f"<div style='color:#ffffff !important;font-size:12px;opacity:0.92'>{sub}</div></div>")


def _freeze_record() -> tuple[str | None, float | None]:
    """(SHA-256 of models/ri_lgbm.txt, MAIN red threshold) as written in docs/freeze.md, or Nones."""
    if not FREEZE_MD.exists():
        return None, None
    txt = FREEZE_MD.read_text(encoding="utf-8")
    sha = re.search(r"`models/ri_lgbm\.txt`[^|]*\|\s*`([0-9a-f]{64})`", txt)
    thr = re.search(r"\|\s*NI-ONLY \(MAIN\)\s*\|[^|]*\|\s*([0-9.]+)\s*\|", txt)
    return (sha.group(1) if sha else None), (float(thr.group(1)) if thr else None)


FROZEN_SHA, FROZEN_THR = _freeze_record()
MODEL_SHA = (hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
             if MODEL.kind == "lgbm" else None)
IS_FROZEN = MODEL_SHA is not None and MODEL_SHA == FROZEN_SHA
if MODEL.kind == "lgbm" and not IS_FROZEN:
    print(f"WARNING: {_rel(MODEL_PATH)} does not match the SHA-256 in docs/freeze.md "
          "(or freeze.md is missing); it is NOT labelled as the frozen test model.", file=sys.stderr)


def model_line() -> str:
    """Threshold and the alert-rate rule, shown under the banner."""
    line = (f"**Alert threshold:** red at probability ≥ {MODEL.threshold:.3f}, amber from "
            f"{MODEL.amber_from:.3f} (half of red). **Rule:** {ALERT_RATE:.0%} of {THRESHOLD_BASIN} "
            f"over-water validation rows (2016-2019) red; this model reaches {MODEL.val_rate:.1%} on those rows.")
    if MODEL.kind == "lgbm":
        if IS_FROZEN and FROZEN_THR is not None and round(MODEL.threshold, 6) == FROZEN_THR:
            line += f" Model file and threshold match docs/freeze.md ({FROZEN_THR})."
        else:
            line += " ⚠ Model file or threshold does NOT match docs/freeze.md."
    return line


def alert_html(p: float, lvl: str) -> str:
    text = {"red": "RED: much more likely than usual. Consider escalating.",
            "amber": "AMBER: more likely than usual. Watch closely.",
            "green": "GREEN: no sign of sudden strengthening",
            "none": "No forecast"}[lvl]
    prob = "–" if pd.isna(p) else "<1%" if p < 0.005 else f"{p:.0%}"
    return (f"<div style='background:{COLOURS[lvl]};color:white;border-radius:12px;padding:18px 22px;"
            f"text-align:center'><div style='font-size:15px'>Chance of a rise of at least "
            f"{RI_THRESHOLD_KT} kt in the next 24 hours</div><div style='font-size:52px;font-weight:800;"
            f"line-height:1.1'>{prob}</div><div style='font-size:22px;font-weight:700'>{text}</div></div>")


# ======================================================================= tab 1
FIELDS = [("wind", "Current wind (kt)"), ("pres", "Current pressure (hPa)"), ("lat", "Latitude (°N)"),
          ("lon", "Longitude (°E)"), ("dist", "Distance to land (km)"), ("month", "Month (1-12)"),
          ("w6", "Wind 6 h ago (kt)"), ("w12", "Wind 12 h ago (kt)"), ("w24", "Wind 24 h ago (kt)"),
          ("p24", "Pressure 24 h ago (hPa)")]


FEATURE_WORDS = {"SPEED_12": "storm speed over the last 12 h", "DIR_12": "storm direction over the last 12 h",
                 "SPEED_24": "storm speed over the last 24 h", "DIR_24": "storm direction over the last 24 h",
                 "USA_PRES": "current pressure", "DPRES_24": "24 h pressure change"}


def forecast(*vals):
    v = dict(zip([k for k, _ in FIELDS], vals))
    missing = [lab for (k, lab) in FIELDS if v[k] is None]
    if missing:
        return alert_html(np.nan, "none"), "Please fill in: " + ", ".join(missing)
    if not 1 <= v["month"] <= 12:
        return alert_html(np.nan, "none"), "Month must be between 1 and 12."
    X = bulletin_features(v)
    p = float(MODEL.predict(X)[0])
    lines = ["**Top 3 reasons**", "\n".join(f"{i}. {t}" for i, t in enumerate(MODEL.reasons(X), 1))]
    gone = MODEL.missing_inputs(X)
    if gone:
        lines.append("**Not available from these inputs (left missing for the model):** "
                     + ", ".join(f"{c} ({FEATURE_WORDS.get(c, c)})" for c in gone))
    if v["dist"] <= 0:
        lines.append("_Note: the storm is over land. The model was only checked on storms over water._")
    return alert_html(p, MODEL.level(p)), "\n\n".join(lines)


# ======================================================================= tab 2
PERS = fit_persistence(pd.read_parquet(TRAIN_PATH))  # baseline shown next to the model (train only)


def load_tracks() -> pd.DataFrame:
    """All North Indian Ocean storms (any row with BASIN NI)."""
    df = pd.read_parquet(TRACKS_PATH)
    ni = df.loc[df["BASIN"] == "NI", "SID"].unique()
    return df[df["SID"].isin(ni)].sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def load_predictions() -> pd.DataFrame | None:
    """data/processed/predictions.parquet from the frozen test run, if present. Never test.parquet."""
    if not PRED_PATH.exists():
        print(f"[predictions] {_rel(PRED_PATH)} not found: every storm is computed live.", flush=True)
        return None
    p = pd.read_parquet(PRED_PATH)
    need = {"SID", "ISO_TIME", "P_MAIN"}
    if not need <= set(p.columns):
        print(f"WARNING: {_rel(PRED_PATH)} lacks {sorted(need - set(p.columns))}; ignoring it.",
              file=sys.stderr, flush=True)
        return None
    p["ISO_TIME"] = pd.to_datetime(p["ISO_TIME"])
    assert not p.duplicated(["SID", "ISO_TIME"]).any(), "duplicate SID/ISO_TIME in predictions file"
    print(f"[predictions] read {_rel(PRED_PATH)}: {len(p)} rows, {p['SID'].nunique()} storms.", flush=True)
    return p


TRACKS = load_tracks()
PRED = load_predictions()
PRED_SIDS = set(PRED["SID"]) if PRED is not None else set()

_names = TRACKS.groupby("SID").agg(NAME=("NAME", "first"), SEASON=("SEASON", "first"),
                                   SUB=("SUBBASIN", lambda s: "/".join(sorted(set(s) - {"MM"}) or ["NI"])),
                                   START=("ISO_TIME", "min")).reset_index()
_names["LABEL"] = (_names["NAME"].str.title() + " (" + _names["SEASON"].astype(str) + ", "
                   + _names["SUB"] + ")")
_default = (_names["NAME"] == DEFAULT_STORM[0]) & (_names["SEASON"] == DEFAULT_STORM[1])
_names = pd.concat([_names[_default], _names[~_default].sort_values("START", ascending=False)])
STORM_CHOICES = [(f"{lab}  [{sid}]", sid) for lab, sid in zip(_names["LABEL"], _names["SID"])]
SEASON_OF = dict(zip(_names["SID"], _names["SEASON"]))
CASEBOOK = [(lab, sid) for lab, sid in CASEBOOK if sid in SEASON_OF]


def source_label(sid: str) -> tuple[str, str]:
    """(short label, explanation) for how this storm's model predictions were made."""
    season = SEASON_OF[sid]
    if sid in PRED_SIDS:
        src = PRED.loc[PRED["SID"] == sid, "SOURCE"].iloc[0] if "SOURCE" in PRED else ""
        if re.search(r"loso|retrain|leave", str(src), re.I):
            return ("retrained without this storm (illustration)",
                    f"From {_rel(PRED_PATH)}: a model retrained without this storm. Illustration, not a headline metric.")
        return ("frozen test model",
                f"From {_rel(PRED_PATH)}, written by the single frozen test run (docs/results_test.md).")
    if MODEL.kind != "lgbm":
        return ("persistence fallback", "The trained model did not load; these are persistence probabilities.")
    if season >= 2020:
        if IS_FROZEN:
            return ("frozen test model",
                    "Computed live from models/ri_lgbm.txt, whose SHA-256 matches docs/freeze.md.")
        return ("model file not frozen", "Computed live from a model file that does NOT match docs/freeze.md.")
    if season >= 2016:
        return ("validation storm: illustration only",
                "2016-2019 validation storm: used to tune the model and set the threshold. Not a fair test, "
                "and not retrained without this storm.")
    return ("training storm: illustration only",
            "1980-2015 training storm: the model learned from it (in-sample). Not a fair test, and not "
            "retrained without this storm.")


def add_actual_ri(df: pd.DataFrame) -> pd.DataFrame:
    """Actual RI (for checking only, never a model input): wind at t+24h matched by timestamp.
    Rows with no t+24h row have unknown RI and are not scored."""
    df = _merge_offset(df, 24, ["USA_WIND"])
    df["RI"] = ((df["USA_WIND_P24"] - df["USA_WIND"]) >= RI_THRESHOLD_KT).astype(int)
    return df


@lru_cache(maxsize=64)
def storm_rows(sid: str) -> pd.DataFrame:
    """Features, live model + persistence probabilities, actual RI; predictions.parquet values
    replace the live model ones for storms it contains."""
    s = storm_features(TRACKS[TRACKS["SID"] == sid])
    s = add_actual_ri(s).sort_values("ISO_TIME").reset_index(drop=True)
    h = s["HAS_HISTORY"].to_numpy()
    p_live, p_pers = np.full(len(s), np.nan), np.full(len(s), np.nan)
    if h.any():
        p_live[h] = MODEL.predict(s[h])
        p_pers[h] = PERS.predict_proba(s.loc[h, ["DWIND_24"]].to_numpy())[:, 1]
    s["P_LIVE"], s["P_PERS"] = p_live, p_pers
    s["ri_prob"] = p_live
    if sid in PRED_SIDS:
        f = PRED[PRED["SID"] == sid].set_index("ISO_TIME")
        s["ri_prob"] = s["ISO_TIME"].map(f["P_MAIN"]).astype(float).to_numpy()
        if "P_PERSISTENCE" in f:
            s["P_PERS"] = s["ISO_TIME"].map(f["P_PERSISTENCE"]).astype(float).to_numpy()
        both = s["ri_prob"].notna() & s["P_LIVE"].notna()
        if both.any() and MODEL.kind == "lgbm":
            diff = float((s.loc[both, "ri_prob"] - s.loc[both, "P_LIVE"]).abs().max())
            if diff > 1e-6:
                print(f"WARNING: {sid}: predictions file differs from the live model by up to {diff:.4f}",
                      file=sys.stderr, flush=True)
    # Scored rows = what the evaluation scores: over water, forecast available, t+24h known.
    s["SCORED"] = (s["DIST2LAND"] > 0) & s["ri_prob"].notna() & s["USA_WIND_P24"].notna()
    return s


def level_at(p: float, thr: float) -> str:
    if pd.isna(p):
        return "none"
    return "red" if p >= thr else "amber" if p >= AMBER_SHARE * thr else "green"


def scored(sid: str, thr: float):
    """Alert levels, RI episodes and counts for one storm at red threshold thr (scored rows only)."""
    s = storm_rows(sid).copy()
    s["ALERT"] = [level_at(p, thr) for p in s["ri_prob"]]
    w = s[s["SCORED"]]
    if len(w):
        eps, storms = episode_alerts(w, "ri_prob", thr)
        red = w["ri_prob"] >= thr
        eps["RED_DURING"] = [bool(red[(w["ISO_TIME"] >= a) & (w["ISO_TIME"] <= b)].any())
                             for a, b in zip(eps["START"], eps["END"])]
        fa = int(storms["N_FALSE_ALARMS"].sum())
    else:
        eps = pd.DataFrame(columns=["START", "END", "CAUGHT", "LEAD_H", "RED_DURING"])
        fa = 0
    s["EP_START"] = s["ISO_TIME"].isin(eps["START"]) & s["SCORED"]
    counts = {"rows": len(w), "red rows": int((w["ri_prob"] >= thr).sum()) if len(w) else 0,
              "episodes": len(eps), "caught": int(eps["CAUGHT"].sum()) if len(eps) else 0,
              "false alarms": fa}
    counts["missed"] = counts["episodes"] - counts["caught"]
    return s, eps, counts


def _l(x) -> list:
    """Plain list (NaN -> None): Plotly 7 otherwise sends numpy arrays as base64, which Gradio's
    plot does not draw."""
    return [None if pd.isna(v) else float(v) for v in x]


def _t(x) -> list:
    return [t.strftime("%Y-%m-%d %H:%M") for t in x]


def make_map(s: pd.DataFrame, i: int) -> go.Figure:
    shown = s.iloc[: i + 1]
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=_l(s["LAT"]), lon=_l(s["LON"]), mode="lines", hoverinfo="skip",
                                line=dict(width=1, color="lightgrey", dash="dot"), name="full track"))
    fig.add_trace(go.Scattergeo(lat=_l(shown["LAT"]), lon=_l(shown["LON"]), mode="lines",
                                hoverinfo="skip", line=dict(width=2, color="grey"), showlegend=False))
    names = {"green": "green", "amber": "amber", "red": "red", "none": "no forecast"}
    for lvl, col in COLOURS.items():
        d = shown[shown["ALERT"] == lvl]
        if d.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lat=_l(d["LAT"]), lon=_l(d["LON"]), mode="markers", name=names[lvl],
            marker=dict(size=9, color=col, line=dict(width=0.5, color="black")),
            text=[f"{t:%d %b %H:%M} UTC<br>wind {w:.0f} kt<br>RI chance "
                  + ("–" if pd.isna(p) else f"{p:.0%}")
                  for t, w, p in zip(d["ISO_TIME"], d["USA_WIND"], d["ri_prob"])],
            hoverinfo="text"))
    st = shown[shown["EP_START"]]
    fig.add_trace(go.Scattergeo(
        lat=_l(st["LAT"]), lon=_l(st["LON"]), mode="markers", name="rapid strengthening began",
        marker=dict(symbol="star", size=18, color="rgba(0,0,0,0)", line=dict(width=2, color="black")),
        text=[f"Rapid strengthening began {t:%d %b %H:%M} UTC" for t in st["ISO_TIME"]],
        hoverinfo="text"))
    now = s.iloc[i]
    fig.add_trace(go.Scattergeo(lat=[float(now["LAT"])], lon=[float(now["LON"])], mode="markers",
                                name="now", hoverinfo="skip",
                                marker=dict(size=16, color="rgba(0,0,0,0)",
                                            line=dict(width=2.5, color="#1d3557"))))
    fig.update_geos(projection_type="mercator", showland=True, landcolor="#eeeeee",
                    showocean=True, oceancolor="#dfefff", showcountries=True,
                    lataxis_range=[float(s["LAT"].min()) - 4, float(s["LAT"].max()) + 4],
                    lonaxis_range=[float(s["LON"].min()) - 6, float(s["LON"].max()) + 6])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=430,
                      legend=dict(orientation="h", y=-0.02))
    return fig


def make_chart(s: pd.DataFrame, eps: pd.DataFrame, i: int, thr: float) -> go.Figure:
    """Wind (left axis) and model / persistence probability (right axis) over time, with the
    observed RI windows shaded and the red threshold drawn."""
    x = _t(s["ISO_TIME"])
    fig = go.Figure()
    for _, e in eps.iterrows():  # RI at t = rise over t..t+24h, so the window ends 24 h after the last RI row
        fig.add_vrect(x0=e["START"].strftime("%Y-%m-%d %H:%M"),
                      x1=(e["END"] + pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"),
                      fillcolor="#d62828", opacity=0.12, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=x, y=_l(s["USA_WIND"]), name="wind (kt)", mode="lines+markers",
                             line=dict(color="#1d3557", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x, y=_l(s["ri_prob"]), name="model RI probability", yaxis="y2",
                             mode="lines+markers", line=dict(color="#d62828", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x, y=_l(s["P_PERS"]), name="persistence probability", yaxis="y2",
                             mode="lines", line=dict(color="#6c757d", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[thr, thr], name=f"red threshold {thr:.3f}", yaxis="y2",
                             mode="lines", line=dict(color="#d62828", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=[None], y=[None], name="observed RI window (shaded)", mode="markers",
                             marker=dict(size=12, color="rgba(214,40,40,0.25)", symbol="square")))
    fig.add_vline(x=x[i], line=dict(color="#1d3557", width=1.5, dash="dash"))
    pmax = np.nanmax([s["ri_prob"].max(), s["P_PERS"].max(), thr, 0.1])
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        yaxis=dict(title="wind (kt)", rangemode="tozero"),
        yaxis2=dict(title="RI probability", overlaying="y", side="right", range=[0, min(1.0, 1.15 * pmax)],
                    tickformat=".0%", showgrid=False),
        legend=dict(orientation="h", y=-0.25))
    return fig


def pers_tile(p: float) -> str:
    lvl = level_at(p, PERS_RED)
    prob = "–" if pd.isna(p) else "<1%" if p < 0.005 else f"{p:.0%}"
    return (f"<div style='border:2px solid {COLOURS[lvl]};border-radius:10px;padding:10px 14px;"
            f"text-align:center'><div style='font-size:13px'>Persistence baseline (last-24h wind change only; "
            f"red at {PERS_RED}, i.e. rise ≥ 25 kt)</div><div style='font-size:30px;font-weight:800'>{prob}</div>"
            f"<div style='font-size:15px;font-weight:700;color:{COLOURS[lvl]}'>{lvl.upper()}</div></div>")


def lead_labels(eps: pd.DataFrame, now) -> str:
    """One line per scored RI episode that has started by the chosen time."""
    eps = eps[eps["START"] <= now]
    if eps.empty:
        return "_No rapid strengthening has begun yet at this time (over-water scored rows)._"
    out = []
    for _, e in eps.iterrows():
        when = f"{e['START']:%d %b %H:%M} UTC"
        if e["CAUGHT"]:
            msg = f"Alert first turned red {e['LEAD_H']:.0f} h before strengthening began"
            if e["LEAD_H"] == 0:
                msg += " (red at the 6-hourly point where it began)"
        elif e["RED_DURING"]:
            msg = "Alert turned red only after strengthening began"
        else:
            msg = "Alert did not turn red during this strengthening"
        out.append(f"⭐ **{when}:** {msg}.")
    return "\n\n".join(out)


def counts_md(c: dict, thr: float) -> str:
    frozen = abs(thr - MODEL.threshold) < 1e-9
    tag = "the frozen threshold" if frozen else f"an exploratory threshold (frozen is {MODEL.threshold:.3f})"
    return (f"**At red threshold {thr:.3f}** ({tag}), on this storm's {c['rows']} scored rows "
            f"(over water, forecast available, wind 24 h later known): "
            f"**{c['caught']} caught**, **{c['missed']} missed** of {c['episodes']} RI episodes; "
            f"**{c['false alarms']} false-alarm episodes**; {c['red rows']} of {c['rows']} rows red. "
            "Caught = red at the episode's first row.")


def side_panel(s: pd.DataFrame, eps: pd.DataFrame, i: int, thr: float) -> tuple[str, str, str]:
    r = s.iloc[i]
    p = r["ri_prob"]
    lines = [f"### {r['ISO_TIME']:%d %b %Y %H:%M} UTC",
             f"**Wind:** {r['USA_WIND']:.0f} kt · **Distance to land:** {r['DIST2LAND']:.0f} km"]
    if pd.isna(p):
        why = ("needs a reading from 24 h earlier" if not r["HAS_HISTORY"]
               else "this row is not in the predictions file (e.g. over land or no reading 24 h later)")
        lines.append(f"_No model forecast: {why}._")
    else:
        lines += ["**Top 3 reasons** (from the model file)",
                  "\n".join(f"{k}. {t}" for k, t in enumerate(MODEL.reasons(s.iloc[[i]]), 1))]
    actual = r["USA_WIND_P24"] - r["USA_WIND"]
    lines.append("**What actually happened next 24 h:** "
                 + ("not known (no reading 24 h later)" if pd.isna(actual)
                    else f"wind changed {actual:+.0f} kt"
                    + (" (rapid strengthening)" if actual >= RI_THRESHOLD_KT else "")))
    lines += ["---", "**Alert vs actual strengthening**", lead_labels(eps, r["ISO_TIME"])]
    return alert_html(p, level_at(p, thr)), pers_tile(r["P_PERS"]), "\n\n".join(lines)


def source_md(sid: str) -> str:
    short, long = source_label(sid)
    colour = "#1d3557" if short == "frozen test model" else "#8a5a00"
    return (f"<div style='border-left:5px solid {colour};padding:6px 12px;background:rgba(0,0,0,0.03)'>"
            f"<b>Predictions for this storm: {short}</b><br><span style='font-size:12px'>{long}</span></div>")


def time_label(s: pd.DataFrame) -> str:
    return (f"Time step (6-hourly, {s['ISO_TIME'].iloc[0]:%d %b} to {s['ISO_TIME'].iloc[-1]:%d %b %Y}; "
            "chosen time shown on the right)")


def view(sid: str, i: float, thr: float):
    s, eps, c = scored(sid, float(thr))
    i = int(min(max(i, 0), len(s) - 1))
    a, pt, panel = side_panel(s, eps, i, float(thr))
    return (make_map(s, i), make_chart(s, eps, i, float(thr)), a, pt, panel,
            counts_md(c, float(thr)), source_md(sid))


def on_storm(sid: str, thr: float):
    s = storm_rows(sid)
    i = len(s) - 1
    # max(i, 1): Gradio needs maximum > minimum; a one-row storm is clamped in view().
    return (gr.Slider(minimum=0, maximum=max(i, 1), value=i, step=1, label=time_label(s)),
            *view(sid, i, thr))


def casebook_md() -> str:
    rows = [f"**{lab}**: {source_label(sid)[0]}" for lab, sid in CASEBOOK]
    return "Casebook (click a name above): " + " · ".join(rows) + f". {CASEBOOK_NOTE}"


# ========================================================================== UI
def model_header() -> None:
    """Banner + threshold rule; called at the top of every tab."""
    gr.HTML(banner_html())
    gr.Markdown(model_line())


def build() -> gr.Blocks:
    first = STORM_CHOICES[0][1]
    s0 = storm_rows(first)
    i0 = len(s0) - 1
    v0 = view(first, i0, MODEL.threshold)
    with gr.Blocks(title="CycloneWatch") as demo:
        gr.Markdown(f"# {TITLE}")
        with gr.Tabs():
            with gr.Tab("Forecast a storm now"):
                model_header()
                gr.Markdown("Enter the latest values from the IMD bulletin.")
                boxes = {}
                with gr.Row():
                    with gr.Column(scale=3):
                        for row in (FIELDS[:3], FIELDS[3:6], FIELDS[6:]):
                            with gr.Row():
                                for k, lab in row:
                                    boxes[k] = gr.Number(label=lab, value=None)
                        with gr.Row():
                            example = gr.Button("Load example: Amphan, 17 May 2020 06:00 UTC")
                            go_btn = gr.Button("Forecast", variant="primary")
                    with gr.Column(scale=2):
                        out_alert = gr.HTML(alert_html(np.nan, "none"))
                        out_reasons = gr.Markdown()
                gr.Markdown(f"**{DISCLAIMER}**")
                inputs = [boxes[k] for k, _ in FIELDS]
                go_btn.click(forecast, inputs, [out_alert, out_reasons])
                example.click(lambda: [EXAMPLE[k] for k, _ in FIELDS], None, inputs).then(
                    forecast, inputs, [out_alert, out_reasons])

            with gr.Tab("Check it on a past storm"):
                model_header()
                gr.Markdown("This replays a real storm to check the model's predictions against "
                            "what actually happened.")
                with gr.Row():
                    cb_buttons = [(gr.Button(lab, size="sm"), sid) for lab, sid in CASEBOOK]
                gr.Markdown(casebook_md())
                with gr.Row():
                    storm = gr.Dropdown(choices=STORM_CHOICES, value=first, label="Storm (North Indian Ocean)")
                    t = gr.Slider(minimum=0, maximum=max(i0, 1), value=i0, step=1, label=time_label(s0))
                source = gr.HTML(v0[6])
                with gr.Row():
                    thr = gr.Slider(minimum=0.01, maximum=0.6, value=MODEL.threshold, step=0.0001,
                                    label=(f"Red alert threshold (frozen value {MODEL.threshold:.4f}; "
                                           "move it to explore, amber = half)"))
                    reset = gr.Button("Reset to frozen threshold", size="sm")
                counts = gr.Markdown(v0[5])
                with gr.Row():
                    with gr.Column(scale=3):
                        fmap = gr.Plot(value=v0[0], label="Track")
                        chart = gr.Plot(value=v0[1], label="Wind and RI probability over time")
                    with gr.Column(scale=2):
                        with gr.Row():
                            r_alert = gr.HTML(v0[2])
                        with gr.Row():
                            r_pers = gr.HTML(v0[3])
                        panel = gr.Markdown(v0[4])
                gr.Markdown(f"**{DISCLAIMER}**")
                outs = [fmap, chart, r_alert, r_pers, panel, counts, source]
                storm.change(on_storm, [storm, thr], [t, *outs])
                t.change(view, [storm, t, thr], outs)
                thr.change(view, [storm, t, thr], outs)
                reset.click(lambda: MODEL.threshold, None, thr)  # exact frozen value; thr.change redraws
                for b, sid in cb_buttons:
                    b.click(lambda sid=sid: sid, None, storm)
    return demo


if __name__ == "__main__":
    build().launch()
