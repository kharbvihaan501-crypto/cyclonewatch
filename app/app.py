"""CycloneWatch demo: replay a Bay of Bengal cyclone with a 24 h rapid-intensification alert.

PROTOTYPE. ri_prob currently comes from predict(), which returns RANDOM values.
Swap predict() for the trained model later; nothing else needs to change.

Run from the project root:  py -3.12 app\\app.py
"""
import sys
import zlib
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.eval.episodes import find_episodes  # noqa: E402

ALERT_THRESHOLD = 0.30           # red at or above this ri_prob
AMBER_FROM = 0.75 * ALERT_THRESHOLD  # amber = "near" the threshold (within 25% below it)
RI_THRESHOLD_KT = 30
TRACKS_PATH = ROOT / "data" / "processed" / "tracks.parquet"
DEFAULT_STORM = ("AMPHAN", 2020)
DISCLAIMER = "Prototype. Always follow IMD and your state disaster authority."
COLOURS = {"green": "#2e9d4f", "amber": "#f0a202", "red": "#d62828"}


# ----------------------------------------------------------------------- model
def predict(rows: pd.DataFrame) -> np.ndarray:
    """Probability of RI in the next 24 h, one per row.

    PLACEHOLDER: random values, seeded by storm so they are stable while replaying.
    Replace with the trained model (same input rows, same output shape).
    """
    seed = zlib.crc32(str(rows["SID"].iloc[0]).encode()) if len(rows) else 0
    return np.random.default_rng(seed).uniform(0, 0.6, len(rows))


def reasons(row: pd.Series) -> list[str]:
    """PLACEHOLDER reasons; to be replaced by model explanations."""
    return ["Placeholder reason 1 (e.g. recent intensification)",
            "Placeholder reason 2 (e.g. distance to land)",
            "Placeholder reason 3 (e.g. season)"]


# ------------------------------------------------------------------------ data
def load_tracks() -> pd.DataFrame:
    df = pd.read_parquet(TRACKS_PATH)
    bob_sids = df.loc[(df["BASIN"] == "NI") & (df["SUBBASIN"] == "BB"), "SID"].unique()
    return df[df["SID"].isin(bob_sids)].sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)


def add_ri(storm: pd.DataFrame) -> pd.DataFrame:
    """RI = USA_WIND(t+24h) - USA_WIND(t) >= 30 kt, t+24h matched by timestamp in the same storm.

    Rows with no t+24h row get RI = 0 (unknown) so they never start an episode.
    """
    fut = storm[["SID", "ISO_TIME", "USA_WIND"]].rename(
        columns={"ISO_TIME": "TIME_P24", "USA_WIND": "USA_WIND_P24"})
    fut["ISO_TIME"] = fut["TIME_P24"] - pd.Timedelta(hours=24)
    out = storm.merge(fut, on=["SID", "ISO_TIME"], how="left", validate="one_to_one")
    m = out["TIME_P24"].notna()
    assert ((out.loc[m, "TIME_P24"] - out.loc[m, "ISO_TIME"]) == pd.Timedelta(hours=24)).all()
    out["RI"] = ((out["USA_WIND_P24"] - out["USA_WIND"]) >= RI_THRESHOLD_KT).astype(int)
    return out


def alert_level(p: float) -> str:
    if p >= ALERT_THRESHOLD:
        return "red"
    return "amber" if p >= AMBER_FROM else "green"


TRACKS = load_tracks()
_names = (TRACKS.groupby("SID").agg(NAME=("NAME", "first"), SEASON=("SEASON", "first"),
                                     START=("ISO_TIME", "min")).reset_index())
_names["LABEL"] = _names["NAME"].str.title() + " (" + _names["SEASON"].astype(str) + ")"
_is_default = (_names["NAME"] == DEFAULT_STORM[0]) & (_names["SEASON"] == DEFAULT_STORM[1])
_names = pd.concat([_names[_is_default], _names[~_is_default].sort_values("START", ascending=False)])
STORM_CHOICES = [(f"{lab}  [{sid}]", sid) for lab, sid in zip(_names["LABEL"], _names["SID"])]


def storm_rows(sid: str) -> pd.DataFrame:
    s = add_ri(TRACKS[TRACKS["SID"] == sid])
    s["ri_prob"] = predict(s)
    s["ALERT"] = s["ri_prob"].map(alert_level)
    s["EP_START"] = False
    eps = find_episodes(s)
    s.loc[s["ISO_TIME"].isin(eps["START"]), "EP_START"] = True
    return s.reset_index(drop=True)


# ----------------------------------------------------------------------- views
def _l(x) -> list:
    """Plain Python list: Plotly 7 otherwise sends numpy arrays as base64, which Gradio's plot does not draw."""
    return [float(v) for v in x]


def make_map(s: pd.DataFrame, i: int) -> go.Figure:
    shown = s.iloc[: i + 1]
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lat=_l(s["LAT"]), lon=_l(s["LON"]), mode="lines", hoverinfo="skip",
                                line=dict(width=1, color="lightgrey", dash="dot"), name="full track"))
    fig.add_trace(go.Scattergeo(lat=_l(shown["LAT"]), lon=_l(shown["LON"]), mode="lines", hoverinfo="skip",
                                line=dict(width=2, color="grey"), showlegend=False))
    for lvl, col in COLOURS.items():
        d = shown[shown["ALERT"] == lvl]
        fig.add_trace(go.Scattergeo(
            lat=_l(d["LAT"]), lon=_l(d["LON"]), mode="markers", name=lvl,
            marker=dict(size=9, color=col, line=dict(width=0.5, color="black")),
            text=[f"{t:%d %b %H:%M} UTC<br>wind {w:.0f} kt<br>ri_prob {p:.2f}"
                  for t, w, p in zip(d["ISO_TIME"], d["USA_WIND"], d["ri_prob"])],
            hoverinfo="text"))
    st = shown[shown["EP_START"]]
    fig.add_trace(go.Scattergeo(
        lat=_l(st["LAT"]), lon=_l(st["LON"]), mode="markers", name="actual RI start",
        marker=dict(symbol="star", size=18, color="rgba(0,0,0,0)", line=dict(width=2, color="black")),
        text=[f"RI episode starts {t:%d %b %H:%M} UTC" for t in st["ISO_TIME"]], hoverinfo="text"))
    now = s.iloc[i]
    fig.add_trace(go.Scattergeo(lat=[float(now["LAT"])], lon=[float(now["LON"])], mode="markers", name="now",
                                marker=dict(size=16, color="rgba(0,0,0,0)",
                                            line=dict(width=2.5, color="#1d3557")), hoverinfo="skip"))
    fig.update_geos(projection_type="mercator", showland=True, landcolor="#eeeeee",
                    showocean=True, oceancolor="#dfefff", showcountries=True,
                    lataxis_range=[float(s["LAT"].min()) - 4, float(s["LAT"].max()) + 4],
                    lonaxis_range=[float(s["LON"].min()) - 6, float(s["LON"].max()) + 6])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520,
                      legend=dict(orientation="h", y=-0.02))
    return fig


def side_panel(s: pd.DataFrame, i: int) -> str:
    r = s.iloc[i]
    lvl = alert_level(r["ri_prob"])
    lines = [
        f"### {r['ISO_TIME']:%d %b %Y %H:%M} UTC",
        f"**RI probability (next 24 h):** {r['ri_prob']:.2f}",
        f"**Alert level:** <span style='color:{COLOURS[lvl]};font-weight:700'>{lvl.upper()}</span> "
        f"(red at {ALERT_THRESHOLD:.2f}, amber from {AMBER_FROM:.2f})",
        f"**Wind now:** {r['USA_WIND']:.0f} kt  ·  **Distance to land:** {r['DIST2LAND']:.0f} km",
        "**Reasons (placeholder):**",
        "\n".join(f"{k}. {txt}" for k, txt in enumerate(reasons(r), 1)),
        "_ri_prob values are random until the model is connected._",
    ]
    return "\n\n".join(lines)


def on_storm(sid: str):
    s = storm_rows(sid)
    i = len(s) - 1
    return (gr.Slider(minimum=0, maximum=len(s) - 1, value=i, step=1), make_map(s, i), side_panel(s, i))


def on_time(sid: str, i: float):
    s = storm_rows(sid)
    i = int(min(max(i, 0), len(s) - 1))
    return make_map(s, i), side_panel(s, i)


def build() -> gr.Blocks:
    first = STORM_CHOICES[0][1]
    s0 = storm_rows(first)
    with gr.Blocks(title="CycloneWatch") as demo:
        gr.Markdown("# CycloneWatch: rapid-intensification alert replay (Bay of Bengal)")
        with gr.Row():
            storm = gr.Dropdown(choices=STORM_CHOICES, value=first, label="Storm")
            t = gr.Slider(minimum=0, maximum=len(s0) - 1, value=len(s0) - 1, step=1,
                          label="Time step (6-hourly; time shown on the right)")
        with gr.Row():
            with gr.Column(scale=3):
                fmap = gr.Plot(value=make_map(s0, len(s0) - 1), label="Track")
                gr.Markdown(f"**{DISCLAIMER}**")
            with gr.Column(scale=2):
                panel = gr.Markdown(side_panel(s0, len(s0) - 1))
        storm.change(on_storm, storm, [t, fmap, panel])
        t.change(on_time, [storm, t], [fmap, panel])
    return demo


if __name__ == "__main__":
    build().launch()
