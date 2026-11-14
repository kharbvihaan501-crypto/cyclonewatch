"""CycloneWatch: will this cyclone suddenly strengthen in the next 24 hours?

Tab 1 forecasts from bulletin-style inputs; Tab 2 replays a past Bay of Bengal storm
to check the same model against what actually happened.

Model: models/ri_lgbm.txt (LightGBM) if it exists, else the persistence baseline
(logistic regression on the last-24h wind change) fitted on the train split.
LightGBM probabilities are Platt-calibrated with models/ri_calibration.json when that
file exists and was fitted for the same model file; otherwise raw probabilities are shown.
The red-alert threshold is set on the VALIDATION split so that ALERT_RATE of
North Indian Ocean (BASIN NI) over-water validation rows are red. The test split is never loaded here.

Run from the project root:  py -3.12 app\\app.py
"""
import os
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
AMBER_SHARE = 0.75  # amber from 75% of the red threshold
MODEL_PATH = Path(os.environ.get("CYCLONEWATCH_MODEL", ROOT / "models" / "ri_lgbm.txt"))
CAL_PATH = MODEL_PATH.with_name("ri_calibration.json")
FEATURES_JSON = MODEL_PATH.with_name("ri_lgbm_features.json")  # exact feature order + BASIN categories
TRACKS_PATH = ROOT / "data" / "processed" / "tracks.parquet"
DEFAULT_STORM = ("AMPHAN", 2020)
TITLE = "CycloneWatch: will this cyclone suddenly strengthen in the next 24 hours?"
DISCLAIMER = "Prototype. Always follow IMD and your state disaster authority."
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


class Model:
    """LightGBM from MODEL_PATH if present, else persistence fitted on train."""

    def __init__(self):
        if MODEL_PATH.exists():
            import lightgbm as lgb
            import json
            self.booster = lgb.Booster(model_file=str(MODEL_PATH))
            if not FEATURES_JSON.exists():
                raise FileNotFoundError(f"{FEATURES_JSON} is needed next to {MODEL_PATH.name} "
                                        "(feature order and BASIN categories); the app will not guess them.")
            spec = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
            assert Path(spec["main_file"]).name == MODEL_PATH.name, \
                f"{FEATURES_JSON.name} describes {spec['main_file']}, not {MODEL_PATH.name}"
            self.cols = list(spec["features"])
            self.cat_cols = list(spec.get("categorical_features", []))
            self.categories = {"BASIN": list(spec["basin_categories"])}
            assert self.cols == self.booster.feature_name(), \
                f"feature order in {FEATURES_JSON.name} differs from the model file"
            self.name = f"LightGBM ({MODEL_PATH.name}, {spec.get('main_model', 'main model')})"
            self.cal = None
            if CAL_PATH.exists():
                import json
                cal = json.loads(CAL_PATH.read_text(encoding="utf-8"))
                if Path(cal["model_file"]).name == MODEL_PATH.name:
                    self.cal = cal
                    self.name += ", Platt-calibrated on 2016-2019 validation storms"
        else:
            self.booster = None
            self.cols, self.cat_cols, self.categories = ["DWIND_24"], [], {}
            self.pers = fit_persistence(pd.read_parquet(TRAIN_PATH))
            self.name = ("Persistence baseline (last-24h wind change only, fitted on 1980-2015). "
                         f"Trained model {MODEL_PATH.as_posix().replace(ROOT.as_posix() + '/', '')} not found.")
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


def model_line() -> str:
    return (f"**Model in use:** {MODEL.name}  \nRed alert at probability ≥ {MODEL.threshold:.3f}, "
            f"amber from {MODEL.amber_from:.3f} (threshold set on 2016-2019 North Indian Ocean validation "
            f"storms so that {MODEL.val_rate:.0%} of their over-water forecasts are red).")


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
def load_tracks() -> pd.DataFrame:
    df = pd.read_parquet(TRACKS_PATH)
    bob = df.loc[(df["BASIN"] == "NI") & (df["SUBBASIN"] == "BB"), "SID"].unique()
    return df[df["SID"].isin(bob)].sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


TRACKS = load_tracks()
_names = TRACKS.groupby("SID").agg(NAME=("NAME", "first"), SEASON=("SEASON", "first"),
                                   START=("ISO_TIME", "min")).reset_index()
_names["LABEL"] = _names["NAME"].str.title() + " (" + _names["SEASON"].astype(str) + ")"
_default = (_names["NAME"] == DEFAULT_STORM[0]) & (_names["SEASON"] == DEFAULT_STORM[1])
_names = pd.concat([_names[_default], _names[~_default].sort_values("START", ascending=False)])
STORM_CHOICES = [(f"{lab}  [{sid}]", sid) for lab, sid in zip(_names["LABEL"], _names["SID"])]


def add_actual_ri(df: pd.DataFrame) -> pd.DataFrame:
    """Actual RI (for checking only, never a model input): wind at t+24h matched by timestamp.
    Rows with no t+24h row get RI = 0 (unknown)."""
    df = _merge_offset(df, 24, ["USA_WIND"])
    df["RI"] = ((df["USA_WIND_P24"] - df["USA_WIND"]) >= RI_THRESHOLD_KT).astype(int)
    return df


@lru_cache(maxsize=32)
def storm_rows(sid: str) -> pd.DataFrame:
    s = storm_features(TRACKS[TRACKS["SID"] == sid])
    s = add_actual_ri(s).sort_values("ISO_TIME").reset_index(drop=True)
    p = np.full(len(s), np.nan)
    h = s["HAS_HISTORY"].to_numpy()
    if h.any():
        p[h] = MODEL.predict(s[h])
    s["ri_prob"] = p
    s["ALERT"] = [MODEL.level(x) for x in p]
    eps, _ = episode_alerts(s.assign(P=s["ri_prob"].fillna(-1.0)), "P", MODEL.threshold)
    s["EP_START"] = s["ISO_TIME"].isin(eps["START"])
    red = s["ri_prob"] >= MODEL.threshold
    eps["RED_DURING"] = [bool(red[(s["ISO_TIME"] >= a) & (s["ISO_TIME"] <= b)].any())
                         for a, b in zip(eps["START"], eps["END"])]
    s.attrs["episodes"] = eps
    return s


def _l(x) -> list:
    """Plain list: Plotly 7 otherwise sends numpy arrays as base64, which Gradio's plot does not draw."""
    return [float(v) for v in x]


def make_map(s: pd.DataFrame, i: int) -> go.Figure:
    shown = s.iloc[: i + 1]
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=_l(s["LAT"]), lon=_l(s["LON"]), mode="lines", hoverinfo="skip",
                                line=dict(width=1, color="lightgrey", dash="dot"), name="full track"))
    fig.add_trace(go.Scattergeo(lat=_l(shown["LAT"]), lon=_l(shown["LON"]), mode="lines",
                                hoverinfo="skip", line=dict(width=2, color="grey"), showlegend=False))
    names = {"green": "green", "amber": "amber", "red": "red", "none": "no forecast (< 24 h of history)"}
    for lvl, col in COLOURS.items():
        d = shown[shown["ALERT"] == lvl]
        if lvl == "none" and d.empty:
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
        lat=_l(st["LAT"]), lon=_l(st["LON"]), mode="markers", name="rapid strengthening actually began",
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
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520,
                      legend=dict(orientation="h", y=-0.02))
    return fig


def lead_labels(s: pd.DataFrame, i: int) -> str:
    """One line per actual RI episode that has started by the chosen time."""
    eps = s.attrs["episodes"]
    eps = eps[eps["START"] <= s["ISO_TIME"].iloc[i]]
    if eps.empty:
        return "_No rapid strengthening has begun yet at this time._"
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


def side_panel(s: pd.DataFrame, i: int) -> tuple[str, str]:
    r = s.iloc[i]
    p, lvl = r["ri_prob"], r["ALERT"]
    lines = [f"### {r['ISO_TIME']:%d %b %Y %H:%M} UTC",
             f"**Wind:** {r['USA_WIND']:.0f} kt · **Distance to land:** {r['DIST2LAND']:.0f} km"]
    if lvl == "none":
        lines.append("_No forecast: needs a reading from 24 h earlier._")
    else:
        X = s.iloc[[i]]
        lines += ["**Top 3 reasons**",
                  "\n".join(f"{k}. {t}" for k, t in enumerate(MODEL.reasons(X), 1))]
    actual = r["USA_WIND_P24"] - r["USA_WIND"]
    lines.append("**What actually happened next 24 h:** "
                 + ("not known (no reading 24 h later)" if pd.isna(actual)
                    else f"wind changed {actual:+.0f} kt"
                    + (" (rapid strengthening)" if actual >= RI_THRESHOLD_KT else "")))
    lines += ["---", "**Alert vs actual strengthening**", lead_labels(s, i)]
    return alert_html(p, lvl), "\n\n".join(lines)


def time_label(s: pd.DataFrame, i: int) -> str:
    return (f"Time step (6-hourly, {s['ISO_TIME'].iloc[0]:%d %b} to {s['ISO_TIME'].iloc[-1]:%d %b %Y}; "
            "chosen time shown on the right)")


def on_storm(sid: str):
    s = storm_rows(sid)
    i = len(s) - 1
    # max(i, 1): Gradio needs maximum > minimum; a one-row storm is clamped in on_time.
    return (gr.Slider(minimum=0, maximum=max(i, 1), value=i, step=1, label=time_label(s, i)),
            make_map(s, i), *side_panel(s, i))


def on_time(sid: str, i: float):
    s = storm_rows(sid)
    i = int(min(max(i, 0), len(s) - 1))
    return make_map(s, i), *side_panel(s, i)


# ========================================================================== UI
def build() -> gr.Blocks:
    first = STORM_CHOICES[0][1]
    s0 = storm_rows(first)
    i0 = len(s0) - 1
    a0, p0 = side_panel(s0, i0)
    with gr.Blocks(title="CycloneWatch") as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(model_line())
        with gr.Tabs():
            with gr.Tab("Forecast a storm now"):
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
                gr.Markdown("This replays a real storm to check the model's predictions against "
                            "what actually happened.")
                with gr.Row():
                    storm = gr.Dropdown(choices=STORM_CHOICES, value=first, label="Storm")
                    t = gr.Slider(minimum=0, maximum=i0, value=i0, step=1, label=time_label(s0, i0))
                with gr.Row():
                    with gr.Column(scale=3):
                        fmap = gr.Plot(value=make_map(s0, i0), label="Track")
                    with gr.Column(scale=2):
                        r_alert = gr.HTML(a0)
                        panel = gr.Markdown(p0)
                gr.Markdown(f"**{DISCLAIMER}**")
                storm.change(on_storm, storm, [t, fmap, r_alert, panel])
                t.change(on_time, [storm, t], [fmap, r_alert, panel])
    return demo


if __name__ == "__main__":
    build().launch()
