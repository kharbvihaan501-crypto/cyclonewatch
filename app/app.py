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

# Eastern India towns for the illustrative exposure panel: approximate coordinates and the coastal
# yes/no flag from app/towns.csv (coordinates and states checked against Wikipedia, 15 Nov 2026).
TOWNS = pd.read_csv(Path(__file__).resolve().parent / "towns.csv")
assert set(TOWNS["coastal"]) <= {"yes", "no"}, "towns.csv: coastal must be yes or no"
_ROTATE = ["top right", "bottom right", "top left", "bottom left"]  # default when label_pos is blank
if "label_pos" not in TOWNS:
    TOWNS["label_pos"] = ""
TOWNS["label_pos"] = [p if isinstance(p, str) and p.strip() else _ROTATE[k % 4]
                      for k, p in enumerate(TOWNS["label_pos"])]
TOWN_NOTES = {"Kolkata": "Inland, but on the tidal Hooghly river: surge can travel upriver."}
N_NEAREST = 10  # towns in the table by default (the rest under "show all"); darker on the map


def exposure_rows(storm_lat: float, storm_lon: float) -> list[dict]:
    """Towns sorted by distance from the storm centre, with the exposure note."""
    d = _haversine_km(storm_lat, storm_lon, TOWNS["lat"].to_numpy(), TOWNS["lon"].to_numpy())
    rows = []
    for (_, t), km in zip(TOWNS.iterrows(), d):
        coastal = t["coastal"] == "yes"
        note = TOWN_NOTES.get(t["town"]) or (
            "Coastal: higher exposure to wind and surge" if coastal else
            "Inland: lower wind exposure, but heavy rain and river flooding can still affect them")
        rows.append({"town": t["town"], "state": t["state"], "lat": float(t["lat"]), "lon": float(t["lon"]),
                     "coastal": coastal, "storm_km": float(km), "note": note, "label_pos": t["label_pos"]})
    return sorted(rows, key=lambda r: r["storm_km"])

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


def _freeze_record() -> dict:
    """From docs/freeze.md: SHA-256 of models/ri_lgbm.txt, MAIN red threshold and its NI val alert rate."""
    out = {"sha": None, "thr": None, "rate": None}
    if not FREEZE_MD.exists():
        return out
    txt = FREEZE_MD.read_text(encoding="utf-8")
    sha = re.search(r"`models/ri_lgbm\.txt`[^|]*\|\s*`([0-9a-f]{64})`", txt)
    row = re.search(r"\|\s*NI-ONLY \(MAIN\)\s*\|[^|]*\|\s*([0-9.]+)\s*\|[^|]*\|\s*([0-9.]+)%", txt)
    if sha:
        out["sha"] = sha.group(1)
    if row:
        out["thr"], out["rate"] = float(row.group(1)), float(row.group(2)) / 100
    return out


FREEZE = _freeze_record()
PERS_VAL_RATE = 0.143  # docs/freeze.md: persistence at 0.1285 puts 14.3% of NI over-water validation rows red


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
        # Thresholds come from docs/freeze.md (no data read). Only a model file that does not match
        # the frozen SHA-256 gets its threshold recomputed from validation, with a warning.
        self.sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() if self.kind == "lgbm" else None
        self.is_frozen = self.sha is not None and self.sha == FREEZE["sha"] and FREEZE["thr"] is not None
        if self.kind == "persistence":
            self.threshold, self.val_rate, self.thr_source = PERS_RED, PERS_VAL_RATE, "docs/freeze.md (persistence)"
        elif self.is_frozen:
            self.threshold, self.val_rate, self.thr_source = FREEZE["thr"], FREEZE["rate"], "docs/freeze.md"
        else:
            print(f"WARNING: {_rel(MODEL_PATH)} does not match the SHA-256 in docs/freeze.md (or freeze.md "
                  "is missing): recomputing its threshold from val.parquet; it is NOT the frozen model.",
                  file=sys.stderr, flush=True)
            val = pd.read_parquet(VAL_PATH)
            val = val[val["BASIN"] == THRESHOLD_BASIN].copy()
            val["P"] = self.predict(val)
            self.threshold, self.val_rate = threshold_for_alert_rate(val, "P", ALERT_RATE)
            self.thr_source = "recomputed from val.parquet (model NOT frozen)"
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


IS_FROZEN = MODEL.is_frozen


def model_line() -> str:
    """Threshold and the alert-rate rule, shown under the banner."""
    line = (f"**Alert threshold:** red at probability ≥ {MODEL.threshold:.6f}, amber from "
            f"{MODEL.amber_from:.6f} (half of red), from {MODEL.thr_source}. **Rule:** {ALERT_RATE:.0%} of "
            f"{THRESHOLD_BASIN} over-water validation rows (2016-2019) red; this model reaches "
            f"{MODEL.val_rate:.1%} on those rows.")
    if MODEL.kind == "lgbm":
        line += (" Model file SHA-256 matches docs/freeze.md." if IS_FROZEN
                 else " ⚠ Model file does NOT match docs/freeze.md.")
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


# ======================================================================= replay tab
# Data: data/processed/predictions.parquet (saved by the frozen test run) + tracks.parquet.
# If predictions.parquet is missing, the same frozen-model predictions are computed live from
# tracks.parquet (clearly labelled). test.parquet is never read.
SAFETY = ("Low RI likelihood does NOT mean the cyclone is safe. "
          "It only means sudden strengthening is less likely.")
RETRO = ("Retrospective experiment: predictions made with the frozen model using only data "
         "available at each time.")
DEMO_FIRST = [("AMPHAN", 2020), ("DANA", 2024), ("YAAS", 2021), ("GULAB", 2021), ("JAWAD", 2021)]
SUCCESS_CASE = ("AMPHAN", 2020)
FALSE_ALARM_CASE = ("DANA", 2024)
BOB_VIEW = dict(lat=(6.0, 24.2), lon=(79.0, 97.0))  # Bay of Bengal + the Eastern India towns
MAP_ASPECT = 1.93  # map box width / height at 70vh on a 16:9 laptop window (measured in the browser)


def split_of(season: int) -> str:
    return "train (1980-2015)" if season <= 2015 else "validation (2016-2019)" if season <= 2019 else "test (2020+)"


def load_tracks() -> pd.DataFrame:
    """All North Indian Ocean storms (any row with BASIN NI), from tracks.parquet."""
    df = pd.read_parquet(TRACKS_PATH)
    ni = df.loc[df["BASIN"] == "NI", "SID"].unique()
    return df[df["SID"].isin(ni)].sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


TRACKS = load_tracks()


def live_predictions() -> pd.DataFrame:
    """Fallback when predictions.parquet is missing: the frozen model on the same rows the test run
    scored (NI storms, SEASON >= 2020, over water, with t-24h and t+24h), from tracks.parquet."""
    from src.eval.baselines import TRAIN_PATH as _TP  # persistence is fitted on train (fallback only)
    pers = fit_persistence(pd.read_parquet(_TP))
    out = []
    for sid, g in TRACKS[TRACKS["SEASON"] >= 2020].groupby("SID"):
        s = storm_features(g)
        s = _merge_offset(s, 24, ["USA_WIND"])
        s = s[s["HAS_HISTORY"] & s["TIME_P24"].notna() & (s["DIST2LAND"] > 0)
              & (s["BASIN"].astype(str) == "NI")].copy()
        if s.empty:
            continue
        s["P_MAIN"] = MODEL.predict(s)
        s["P_PERSISTENCE"] = pers.predict_proba(s[["DWIND_24"]].to_numpy())[:, 1]
        s["ALERT"] = [MODEL.level(p) for p in s["P_MAIN"]]
        s["RI"] = ((s["USA_WIND_P24"] - s["USA_WIND"]) >= RI_THRESHOLD_KT).astype(int)
        out.append(s[["SID", "ISO_TIME", "P_MAIN", "P_PERSISTENCE", "ALERT", "RI"]])
    return pd.concat(out, ignore_index=True)


def load_predictions() -> tuple[pd.DataFrame, str]:
    if PRED_PATH.exists():
        p = pd.read_parquet(PRED_PATH)
        need = {"SID", "ISO_TIME", "P_MAIN", "P_PERSISTENCE", "ALERT", "RI"}
        missing = need - set(p.columns)
        if missing:
            raise ValueError(f"{_rel(PRED_PATH)} lacks columns {sorted(missing)}")
        p["ISO_TIME"] = pd.to_datetime(p["ISO_TIME"])
        assert not p.duplicated(["SID", "ISO_TIME"]).any(), "duplicate SID/ISO_TIME in predictions file"
        print(f"[replay] read {_rel(PRED_PATH)}: {len(p)} rows, {p['SID'].nunique()} storms.", flush=True)
        return p, f"saved predictions from {_rel(PRED_PATH)} (frozen test run)"
    print(f"WARNING: {_rel(PRED_PATH)} not found: replay predictions are computed live from "
          "tracks.parquet with the frozen model (same rows as the test run).", file=sys.stderr, flush=True)
    p = live_predictions()
    print(f"[replay] computed live: {len(p)} rows, {p['SID'].nunique()} storms.", flush=True)
    return p, "computed live with the frozen model (predictions.parquet not found)"


PRED, PRED_SOURCE = load_predictions()
_meta = (TRACKS.groupby("SID").agg(NAME=("NAME", "first"), SEASON=("SEASON", "first"),
                                   START=("ISO_TIME", "min")).loc[lambda d: d.index.isin(PRED["SID"])]
         .reset_index())
_meta["SHORT"] = _meta["NAME"].str.split(":").str[0]  # e.g. "GULAB:SHAHEEN-GU" -> "GULAB"
_meta["LABEL"] = _meta["NAME"].str.title() + " " + _meta["SEASON"].astype(str)
_key = list(zip(_meta["SHORT"], _meta["SEASON"]))
_meta["ORDER"] = [DEMO_FIRST.index(k) if k in DEMO_FIRST else len(DEMO_FIRST) for k in _key]
_meta = _meta.sort_values(["ORDER", "START"], ascending=[True, False])
STORM_CHOICES = [(f"{lab}  [{sid}]", sid) for lab, sid in zip(_meta["LABEL"], _meta["SID"])]
SEASON_OF = dict(zip(_meta["SID"], _meta["SEASON"]))


def sid_of(name: str, season: int) -> str | None:
    m = _meta[(_meta["SHORT"] == name) & (_meta["SEASON"] == season)]
    return m["SID"].iloc[0] if len(m) else None


NOT_SCORED_NAMES = {"land": "over land (not scored)",
                    "outcome": "no outcome: no reading 24 h later (not scored)",
                    "history": "no forecast: needs a reading 24 h earlier",
                    "other": "not in the saved test rows (not scored)"}


def _not_scored_reason(r: pd.Series) -> str:
    """Why a replay row has no saved prediction. Over land is checked first."""
    if r["DIST2LAND"] <= 0:
        return "land"
    if not r["HAS24"]:
        return "history"
    if not r["HAS_OUTCOME"]:
        return "outcome"
    return "other"


@lru_cache(maxsize=64)
def storm_view(sid: str):
    """Track rows of one storm with saved predictions attached, plus RI episodes (scored rows only)."""
    s = TRACKS[TRACKS["SID"] == sid].copy()
    s = _merge_offset(s, 24, ["USA_WIND"])  # actual wind 24 h later, shown only after Reveal
    p = PRED[PRED["SID"] == sid].drop(columns=[c for c in ("SID",) if c in PRED])
    keep = ["ISO_TIME", "P_MAIN", "P_PERSISTENCE", "ALERT", "RI"]
    s = s.merge(p[keep], on="ISO_TIME", how="left").sort_values("ISO_TIME").reset_index(drop=True)
    s["ALERT"] = s["ALERT"].fillna("none")
    s["HAS24"] = (s["ISO_TIME"] - pd.Timedelta(hours=24)).isin(set(s["ISO_TIME"]))
    s["OVER_WATER"] = s["DIST2LAND"] > 0
    s["HAS_OUTCOME"] = s["TIME_P24"].notna()
    # Unscored rows (no saved prediction) say why; they are never part of hits, false alarms or lead times.
    s["NOT_SCORED"] = [None if pd.notna(p) else _not_scored_reason(r)
                       for p, (_, r) in zip(s["P_MAIN"], s.iterrows())]
    sc = s[s["P_MAIN"].notna()].copy()
    if len(sc):
        # Caught / lead time from the saved alert colours (red = 1), so the replay matches the test run.
        sc["RED"] = (sc["ALERT"] == "red").astype(float)
        eps, storms = episode_alerts(sc, "RED", 0.5)
        eps["RED_DURING"] = [bool(sc.loc[(sc["ISO_TIME"] >= a) & (sc["ISO_TIME"] <= b), "RED"].any())
                             for a, b in zip(eps["START"], eps["END"])]
        fa = int(storms["N_FALSE_ALARMS"].sum())
    else:
        eps, fa = pd.DataFrame(columns=["START", "END", "CAUGHT", "LEAD_H", "RED_DURING"]), 0
    s["EP_START"] = s["ISO_TIME"].isin(eps["START"])
    return s, eps, fa


def _l(x) -> list:
    """Plain list (NaN -> None): Plotly 7 otherwise sends numpy arrays as base64, which Gradio's
    plot does not draw."""
    return [None if pd.isna(v) else float(v) for v in x]


def _t(x) -> list:
    return [t.strftime("%Y-%m-%d %H:%M") for t in x]


def _merc(lat: float) -> float:
    return float(np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))))


def map_window(s: pd.DataFrame) -> tuple[tuple, tuple]:
    """(lat range, lon range) whose Mercator shape matches the map box (MAP_ASPECT wide : 1 high),
    so the map fills the frame with no empty bands. Bay of Bengal window when the storm fits in it,
    else a window around the storm."""
    lat, lon = BOB_VIEW["lat"], BOB_VIEW["lon"]
    if not (s["LAT"].between(*lat).all() and s["LON"].between(*lon).all()):
        lat = (float(s["LAT"].min()) - 3, float(s["LAT"].max()) + 3)
        lon = (float(s["LON"].min()) - 3, float(s["LON"].max()) + 3)
    need = max(_merc(lat[1]) - _merc(lat[0]), (lon[1] - lon[0]) / MAP_ASPECT)  # Mercator height to show
    mid_y = (_merc(lat[0]) + _merc(lat[1])) / 2
    inv = lambda y: float(np.degrees(2 * np.arctan(np.exp(np.radians(y))) - np.pi / 2))  # noqa: E731
    lat_r = (inv(mid_y - need / 2), inv(mid_y + need / 2))
    mid_x, half_w = sum(lon) / 2, need * MAP_ASPECT / 2
    return lat_r, (mid_x - half_w, mid_x + half_w)


# White halo around town names so they stay readable over coastlines and borders
# (Plotly has no text background box for map labels; textfont.shadow draws the halo).
HALO = "-1px -1px 1px #fff, 1px -1px 1px #fff, -1px 1px 1px #fff, 1px 1px 1px #fff, 0 0 3px #fff"


# ---- town label layout -------------------------------------------------------------------------
# Labels are placed in pixels for the map's own window, assuming the geo area is MAP_PX_W wide
# (measured: 1165 px in a 1400 x 900 window; smaller windows shrink the map, not the text).
MAP_PX_W = 1165.0
CHAR_PX, TEXT_H, GAP, MARK_PX = 6.2, 14.0, 5.0, 9.0
PAD = 2.5  # safety margin around each label: covers small offset differences in the browser
_POS = ["middle right", "middle left", "top center", "bottom center",
        "top right", "bottom right", "top left", "bottom left"]


def _label_box(x: float, y: float, w: float, pos: str) -> tuple:
    """Pixel box (x0, y0, x1, y1) of a label at marker (x, y) for a Plotly textposition."""
    v, h = pos.split()
    x0 = {"left": x - GAP - w, "center": x - w / 2, "right": x + GAP}[h]
    y0 = {"top": y - GAP - TEXT_H, "middle": y - TEXT_H / 2, "bottom": y + GAP}[v]
    return (x0, y0, x0 + w, y0 + TEXT_H)


def _hits(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def layout_labels(towns: list[dict], lat_r: tuple, lon_r: tuple) -> dict:
    """Greedy label placement, nearest towns first. Each town gets the free textposition closest
    to its preferred one (label_pos in towns.csv). If none of the 8 spots next to the marker is free
    of other labels and town markers, the label moves out along a short leader line.
    Returns {town: ("pos", textposition) or ("leader", label_lat, label_lon, textposition)}."""
    k = MAP_PX_W / (lon_r[1] - lon_r[0])
    top = _merc(lat_r[1])
    xy = {r["town"]: ((r["lon"] - lon_r[0]) * k, (top - _merc(r["lat"])) * k) for r in towns}
    H = (top - _merc(lat_r[0])) * k
    marks = {t: (x - MARK_PX / 2, y - MARK_PX / 2, x + MARK_PX / 2, y + MARK_PX / 2) for t, (x, y) in xy.items()}
    placed, out = [], {}

    def cost(box, own):
        """Overlaps with placed labels and OTHER towns' markers (with PAD margin), +10 if off the map."""
        off = box[0] < 0 or box[1] < 0 or box[2] > MAP_PX_W or box[3] > H
        big = (box[0] - PAD, box[1] - PAD, box[2] + PAD, box[3] + PAD)
        return (10 * off + sum(_hits(big, b) for b in placed)
                + sum(_hits(big, m) for t, m in marks.items() if t != own))

    for r in towns:  # already sorted nearest first
        x, y = xy[r["town"]]
        w = CHAR_PX * len(r["town"]) + 2
        pref = r["label_pos"] if r["label_pos"] in _POS else "middle right"
        order = [pref] + [p for p in _POS if p != pref]
        best = min(order, key=lambda p: (cost(_label_box(x, y, w, p), r["town"]), order.index(p)))
        box = _label_box(x, y, w, best)
        if cost(box, r["town"]) == 0:
            placed.append(box)
            out[r["town"]] = ("pos", best)
            continue
        choice = None
        for dist in (24, 36, 50, 66):  # leader line: move the label out, try 8 directions
            for ang in range(0, 360, 45):
                lx, ly = x + dist * np.cos(np.radians(ang)), y - dist * np.sin(np.radians(ang))
                p = "middle right" if np.cos(np.radians(ang)) > 0.1 else (
                    "middle left" if np.cos(np.radians(ang)) < -0.1 else
                    ("top center" if np.sin(np.radians(ang)) > 0 else "bottom center"))
                b = _label_box(lx, ly, w, p)
                if cost(b, r["town"]) == 0 and not _hits(b, marks[r["town"]]):
                    choice = (lx, ly, p, b)
                    break
            if choice:
                break
        if choice is None:  # nothing free: keep the least-bad spot next to the marker
            placed.append(box)
            out[r["town"]] = ("pos", best)
            continue
        lx, ly, p, b = choice
        placed.append(b)
        inv = lambda yy: float(np.degrees(2 * np.arctan(np.exp(np.radians(yy))) - np.pi / 2))  # noqa: E731
        out[r["town"]] = ("leader", inv(top - ly / k), lon_r[0] + lx / k, p)
    return out


def add_towns(fig: go.Figure, storm_lat: float, storm_lon: float, lat_r: tuple, lon_r: tuple) -> None:
    """Every town in towns.csv as a marker WITH its name. The N_NEAREST nearest to the storm centre
    are dark and size 11; the others are grey and size 10. Crowded labels sit on a short leader line."""
    towns = exposure_rows(storm_lat, storm_lon)
    nearest = {r["town"] for r in towns[:N_NEAREST]}
    lay = layout_labels(towns, lat_r, lon_r)
    font = lambda rows: dict(size=[11 if r["town"] in nearest else 10 for r in rows],  # noqa: E731
                             color=["#111" if r["town"] in nearest else "#555" for r in rows], shadow=HALO)
    lead = [r for r in towns if lay[r["town"]][0] == "leader"]
    if lead:  # thin leader lines, then the moved labels
        la, lo = [], []
        for r in lead:
            _, a, b, _ = lay[r["town"]]
            la += [r["lat"], a, None]
            lo += [r["lon"], b, None]
        fig.add_trace(go.Scattergeo(lat=la, lon=lo, mode="lines", line=dict(width=0.8, color="#777"),
                                    hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scattergeo(
            lat=[lay[r["town"]][1] for r in lead], lon=[lay[r["town"]][2] for r in lead], mode="text",
            text=[r["town"] for r in lead], textposition=[lay[r["town"]][3] for r in lead],
            textfont=font(lead), hoverinfo="skip", showlegend=False))
    for coastal, sym, nm in ((True, "square", "Town, coastal"), (False, "square-open", "Town, inland")):
        tw = [r for r in towns if r["coastal"] == coastal]
        near = [r["town"] in nearest for r in tw]
        fig.add_trace(go.Scattergeo(
            lat=[r["lat"] for r in tw], lon=[r["lon"] for r in tw], mode="markers+text", name=nm,
            text=[r["town"] if lay[r["town"]][0] == "pos" else "" for r in tw],
            textposition=[lay[r["town"]][1] if lay[r["town"]][0] == "pos" else "middle right" for r in tw],
            textfont=font(tw),
            marker=dict(symbol=sym, size=[8 if n else 7 for n in near],
                        color=["#111" if n else "#666" for n in near],
                        line=dict(width=1.5, color=["#111" if n else "#666" for n in near])),
            hovertext=[f"<b>{r['town']}</b> ({r['state']})<br>{'Coastal' if r['coastal'] else 'Inland'}"
                       f"<br>{r['storm_km']:.0f} km from storm centre"
                       + (f"<br>{TOWN_NOTES[r['town']]}" if r["town"] in TOWN_NOTES else "") for r in tw],
            hoverinfo="text"))


def make_map(s: pd.DataFrame, i: int, reveal: bool, extra=None) -> go.Figure:
    shown = s if reveal else s.iloc[: i + 1]
    fig = go.Figure()
    now = s.iloc[i]
    lat_r, lon_r = map_window(s)
    add_towns(fig, float(now["LAT"]), float(now["LON"]), lat_r, lon_r)  # first = under the storm track
    fig.add_trace(go.Scattergeo(lat=_l(shown["LAT"]), lon=_l(shown["LON"]), mode="lines",
                                hoverinfo="skip", line=dict(width=2.5, color="#555"), showlegend=False))
    groups = [(lvl, COLOURS[lvl], shown["ALERT"] == lvl, lvl, "circle") for lvl in ("green", "amber", "red")]
    groups += [(NOT_SCORED_NAMES[k], COLOURS["none"], shown["NOT_SCORED"] == k, "none",
                "x" if k == "land" else "circle-open") for k in ("land", "outcome", "history", "other")]
    for name, col, mask, lvl, sym in groups:
        d = shown[mask]
        if d.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lat=_l(d["LAT"]), lon=_l(d["LON"]), mode="markers", name=name,
            marker=dict(size=11, color=col, symbol=sym,
                        line=dict(width=2, color=col) if lvl == "none" else dict(width=0.6, color="black")),
            text=[f"{t:%d %b %H:%M} UTC<br>wind {w:.0f} kt<br>RI likelihood "
                  + ("–" if pd.isna(p) else f"{p:.0%}")
                  for t, w, p in zip(d["ISO_TIME"], d["USA_WIND"], d["P_MAIN"])],
            hoverinfo="text"))
    if reveal:
        st = s[s["EP_START"]]
        fig.add_trace(go.Scattergeo(
            lat=_l(st["LAT"]), lon=_l(st["LON"]), mode="markers", name="rapid strengthening began",
            marker=dict(symbol="star", size=22, color="rgba(0,0,0,0)", line=dict(width=2.5, color="black")),
            text=[f"Rapid strengthening began {t:%d %b %H:%M} UTC" for t in st["ISO_TIME"]],
            hoverinfo="text"))
    if extra is not None:
        extra(fig)
    fig.add_trace(go.Scattergeo(lat=[float(now["LAT"])], lon=[float(now["LON"])], mode="markers",
                                name="chosen time", hoverinfo="skip",
                                marker=dict(size=20, color="rgba(0,0,0,0)",
                                            line=dict(width=3, color="#1d3557"))))
    lat_r, lon_r = map_window(s)
    fig.update_geos(projection_type="mercator", showland=True, landcolor="#ececec",
                    showocean=True, oceancolor="#dceefc", showcountries=True, countrycolor="#999",
                    showcoastlines=True, coastlinecolor="#777", showframe=False, resolution=50,
                    lataxis_range=list(lat_r), lonaxis_range=list(lon_r),
                    domain=dict(x=[0, 1], y=[0, 1]))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), autosize=True,
                      paper_bgcolor="#dceefc",
                      legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,0.8)"))
    return fig


def make_chart(s: pd.DataFrame, eps: pd.DataFrame, i: int, reveal: bool) -> go.Figure:
    """Wind (left axis), model and persistence probability (right axis), dashed threshold,
    observed RI window shaded. Before Reveal: nothing after the chosen time."""
    now = s["ISO_TIME"].iloc[i]
    shown = s if reveal else s.iloc[: i + 1]
    x = _t(shown["ISO_TIME"])
    fig = go.Figure()
    for _, e in eps.iterrows():
        # RI at row r = wind rose >= 30 kt over r..r+24h. Before Reveal, a window is shown only once
        # it can be known at the chosen time (r + 24 h <= now), and it is cut at the chosen time.
        end = e["END"] + pd.Timedelta(hours=24)
        if not reveal:
            if e["START"] + pd.Timedelta(hours=24) > now:
                continue
            end = min(end, now)
        fig.add_vrect(x0=e["START"].strftime("%Y-%m-%d %H:%M"), x1=end.strftime("%Y-%m-%d %H:%M"),
                      fillcolor="#d62828", opacity=0.14, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=x, y=_l(shown["USA_WIND"]), name="wind (kt)", mode="lines+markers",
                             line=dict(color="#1d3557", width=2.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=x, y=_l(shown["P_MAIN"]), name="model RI probability", yaxis="y2",
                             mode="lines+markers", line=dict(color="#d62828", width=2.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=x, y=_l(shown["P_PERSISTENCE"]), name="persistence probability",
                             yaxis="y2", mode="lines", line=dict(color="#6c757d", width=2, dash="dot")))
    full = _t(s["ISO_TIME"])
    fig.add_trace(go.Scatter(x=[full[0], full[-1]], y=[MODEL.threshold] * 2, yaxis="y2", mode="lines",
                             name=f"alert threshold {MODEL.threshold:.3f}",
                             line=dict(color="#d62828", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="observed RI window",
                             marker=dict(size=12, symbol="square", color="rgba(214,40,40,0.3)")))
    fig.add_vline(x=_t([now])[0], line=dict(color="#1d3557", width=1.5, dash="dash"))
    pmax = np.nanmax([s["P_MAIN"].max(), s["P_PERSISTENCE"].max(), MODEL.threshold, 0.1])
    wmax = float(s["USA_WIND"].max())
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        xaxis=dict(range=[full[0], full[-1]]),
        yaxis=dict(title="wind (kt)", range=[0, 1.1 * wmax]),
        yaxis2=dict(title="RI probability", overlaying="y", side="right", tickformat=".0%",
                    range=[0, min(1.0, 1.15 * pmax)], showgrid=False),
        legend=dict(orientation="h", y=-0.22))
    return fig


def now_panel(s: pd.DataFrame, i: int) -> str:
    """Probability tile, alert colour, wind now, persistence, and the safety sentence."""
    r = s.iloc[i]
    p, lvl = r["P_MAIN"], r["ALERT"]
    words = {"red": "RED: much more likely than usual", "amber": "AMBER: more likely than usual",
             "green": "GREEN: less likely than usual", "none": "No saved prediction at this time"}
    if pd.isna(p):
        why = NOT_SCORED_NAMES[r["NOT_SCORED"]]
        prob = f"<div style='font-size:14px;color:#fff !important'>({why})</div>"
    else:
        prob = (f"<div style='font-size:48px;font-weight:800;line-height:1.05;color:#fff !important'>"
                f"{'<1%' if p < 0.005 else f'{p:.0%}'}</div>")
    pp = r["P_PERSISTENCE"]
    pers = "–" if pd.isna(pp) else ("<1%" if pp < 0.005 else f"{pp:.0%}")
    return (f"<div style='background:{COLOURS[lvl]};border-radius:12px;padding:14px 16px;text-align:center'>"
            f"<div style='font-size:14px;font-weight:700;color:#fff !important'>RI likelihood (next 24 h)</div>"
            f"{prob}<div style='font-size:18px;font-weight:800;color:#fff !important'>{words[lvl]}</div></div>"
            f"<div style='margin-top:8px;padding:8px 10px;border:2px solid #d62828;border-radius:8px;"
            f"font-weight:700'>{SAFETY}</div>"
            f"<div style='margin-top:10px;font-size:15px'><b>{r['ISO_TIME']:%d %b %Y %H:%M} UTC</b><br>"
            f"Current wind: <b>{r['USA_WIND']:.0f} kt</b> · distance to land {r['DIST2LAND']:.0f} km<br>"
            f"Persistence baseline: {pers}</div>")


def outcome_md(s: pd.DataFrame, eps: pd.DataFrame, fa: int, i: int, reveal: bool) -> str:
    if not reveal:
        return ("_What happened next is hidden. Everything above uses only data up to the chosen time. "
                "Press **Reveal what happened next** to see the rest of the track and the outcome._")
    r = s.iloc[i]
    nxt = r["USA_WIND_P24"] - r["USA_WIND"]
    lines = ["**What happened next**",
             "Wind 24 h after the chosen time: " + ("**no outcome** (no reading 24 h later; not scored)"
                                                   if pd.isna(nxt) else
                                                   f"{r['USA_WIND_P24']:.0f} kt ({nxt:+.0f} kt)")]
    if not r["OVER_WATER"]:
        lines.append("This time is **over land**: it is not scored and not counted in hits, "
                     "false alarms or lead times.")
    if eps.empty:
        lines.append("No rapid strengthening (rise of 30 kt or more in 24 h) on this storm's scored rows.")
    for _, e in eps.iterrows():
        when = f"{e['START']:%d %b %H:%M} UTC"
        if e["CAUGHT"]:
            msg = f"Alert first turned red {e['LEAD_H']:.0f} h before strengthening began"
            if e["LEAD_H"] == 0:
                msg += " (red at the 6-hourly point where it began)"
        elif e["RED_DURING"]:
            msg = "Alert turned red only after it began"
        else:
            msg = "Missed"
        lines.append(f"⭐ Strengthening began {when}: **{msg}**.")
    lines.append(f"False-alarm episodes on this storm (red runs with no rapid strengthening): **{fa}**.")
    return "\n\n".join(lines)


def exposure_html(s: pd.DataFrame, i: int) -> str:
    """Coastal exposure (illustrative): towns by distance from the storm centre at the chosen time.
    The N_NEAREST nearest are shown; the rest open with "show all"."""
    r = s.iloc[i]
    rows = exposure_rows(float(r["LAT"]), float(r["LON"]))

    def tr(x):
        return (f"<tr><td><b>{x['town']}</b></td><td>{x['state']}</td>"
                f"<td style='text-align:right'>{x['storm_km']:.0f}</td><td>{x['note']}</td></tr>")
    head = ("<tr style='text-align:left;border-bottom:1px solid #ccc'><th>Town</th><th>State</th>"
            "<th style='text-align:right'>From storm centre (km)</th><th>Exposure</th></tr>")
    table = "<table style='width:100%;border-collapse:collapse;font-size:14px'>{}</table>"
    rest = rows[N_NEAREST:]
    more = (f"<details style='margin-top:6px'><summary style='cursor:pointer;font-weight:700'>Show all "
            f"{len(rows)} towns ({len(rest)} more)</summary>{table.format(head + ''.join(map(tr, rest)))}"
            f"</details>") if rest else ""
    return (f"<div style='border:1px solid #ccc;border-radius:10px;padding:10px 14px'>"
            f"<div style='font-size:17px;font-weight:800'>Coastal exposure (illustrative)</div>"
            f"<div style='font-size:13px;font-weight:700;color:#8a5a00'>Illustrative only. Based on distance, "
            f"not a forecast. Does not model storm surge or flooding.</div>"
            f"<div style='font-size:12px;margin:2px 0 6px'>Storm centre at {r['ISO_TIME']:%d %b %Y %H:%M} UTC "
            f"({r['LAT']:.1f}°N, {r['LON']:.1f}°E). The {min(N_NEAREST, len(rows))} nearest of {len(rows)} "
            f"Eastern India towns. Town positions are approximate; coastal or inland is as listed in "
            f"app/towns.csv.</div>"
            f"{table.format(head + ''.join(map(tr, rows[:N_NEAREST])))}{more}</div>")


def label_html(sid: str) -> str:
    season = SEASON_OF[sid]
    return (f"<div style='border-left:5px solid #1d3557;padding:6px 12px;background:rgba(29,53,87,0.06)'>"
            f"<b>Split: {split_of(season)}</b> · predictions: {PRED_SOURCE}<br>"
            f"<span style='font-size:13px'>{RETRO}</span></div>")


def render(sid: str, i: float, reveal: bool):
    s, eps, fa = storm_view(sid)
    i = int(min(max(i, 0), len(s) - 1))
    return (make_map(s, i, reveal), make_chart(s, eps, i, reveal), now_panel(s, i),
            outcome_md(s, eps, fa, i, reveal), label_html(sid), exposure_html(s, i))


def time_label(s: pd.DataFrame) -> str:
    return (f"Time (6-hourly, {s['ISO_TIME'].iloc[0]:%d %b %H:%M} to "
            f"{s['ISO_TIME'].iloc[-1]:%d %b %Y %H:%M} UTC)")


def default_index(sid: str) -> int:
    """Start at the first time with a saved prediction, so the demo opens on a forecast."""
    s, _, _ = storm_view(sid)
    has = np.flatnonzero(s["P_MAIN"].notna().to_numpy())
    return int(has[0]) if len(has) else 0


def on_storm(sid: str):
    s, _, _ = storm_view(sid)
    i = default_index(sid)
    return (gr.Slider(minimum=0, maximum=max(len(s) - 1, 1), value=i, step=1, label=time_label(s)),
            False, *render(sid, i, False))


def on_time(sid: str, i: float):
    return (False, *render(sid, i, False))


def on_reveal(sid: str, i: float):
    return (True, *render(sid, i, True))


# ========================================================================== UI
CSS = """
#replay-map { height: 70vh !important; min-height: 420px; }
#replay-map > div, #replay-map .js-plotly-plot, #replay-map .plot-container,
#replay-map .svg-container { height: 70vh !important; min-height: 420px; width: 100% !important; }
"""


def model_header() -> None:
    """Banner + threshold rule; called at the top of every tab."""
    gr.HTML(banner_html())
    gr.Markdown(model_line())


def build() -> gr.Blocks:
    first = STORM_CHOICES[0][1]
    s0, _, _ = storm_view(first)
    i0 = default_index(first)
    v0 = render(first, i0, False)
    success_sid, fa_sid = sid_of(*SUCCESS_CASE), sid_of(*FALSE_ALARM_CASE)
    with gr.Blocks(title="CycloneWatch") as demo:
        gr.Markdown(f"# {TITLE}")
        with gr.Tabs():
            with gr.Tab("Check it on a past storm"):
                model_header()
                gr.Markdown("This replays a real storm to check the model's predictions against "
                            "what actually happened.")
                with gr.Row(equal_height=True):
                    b_success = gr.Button("Success case: Amphan 2020", variant="primary", scale=1)
                    b_fa = gr.Button("False alarm: Dana 2024", variant="secondary", scale=1)
                    storm = gr.Dropdown(choices=STORM_CHOICES, value=first, scale=2,
                                        label="Storm (casebook first: Amphan, Dana, Yaas, Gulab, Jawad)")
                t = gr.Slider(minimum=0, maximum=max(len(s0) - 1, 1), value=i0, step=1, label=time_label(s0))
                label = gr.HTML(v0[4])
                fmap = gr.Plot(value=v0[0], elem_id="replay-map", show_label=False)
                with gr.Row():
                    with gr.Column(scale=1, min_width=260):
                        panel = gr.HTML(v0[2])
                    with gr.Column(scale=2, min_width=420):
                        chart = gr.Plot(value=v0[1], label="Timeline: wind and RI probability")
                    with gr.Column(scale=1, min_width=260):
                        reveal_btn = gr.Button("Reveal what happened next", variant="primary")
                        outcome = gr.Markdown(v0[3])
                exposure = gr.HTML(v0[5])
                gr.Markdown(f"**{DISCLAIMER}**")
                revealed = gr.State(False)
                outs = [fmap, chart, panel, outcome, label, exposure]
                storm.change(on_storm, storm, [t, revealed, *outs])
                t.change(on_time, [storm, t], [revealed, *outs])
                reveal_btn.click(on_reveal, [storm, t], [revealed, *outs])
                if success_sid:
                    b_success.click(lambda: success_sid, None, storm)
                if fa_sid:
                    b_fa.click(lambda: fa_sid, None, storm)

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
    return demo


if __name__ == "__main__":
    build().launch(css=CSS)
