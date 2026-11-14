"""Sanity checks on data/processed/tracks.parquet.

Prints and saves (CSV tables in docs/data_checks/, map in docs/figures/):
  1) share of rows with missing USA_WIND by BASIN and by decade
  2) RI counts per year, all basins and Bay of Bengal, split by location at t
  3) Bay of Bengal storms and RI rows per split (train / val / test)
  4) Plotly map of Cyclone Amphan (2020) coloured by USA_WIND

RI at t = USA_WIND(t+24h) - USA_WIND(t) >= 30 kt. The t+24h row is found by
matching ISO_TIME + 24h within the same SID (never by row position).

Run from the project root:  python -m src.eval.data_checks
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TRACKS_PATH = "data/processed/tracks.parquet"
TABLE_DIR = "docs/data_checks"
AMPHAN_HTML = "docs/figures/amphan_track.html"
RI_THRESHOLD_KT = 30
NEAR_LAND_KM = 400
SPLITS = {"train": (1980, 2015), "val": (2016, 2019), "test": (2020, 9999)}


def show(title: str, table: pd.DataFrame, fname: str, index: bool = True) -> None:
    """Print a table under a clear header and save it as CSV."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(table.to_string(index=index))
    path = os.path.join(TABLE_DIR, fname)
    table.to_csv(path, index=index)
    print(f"[saved {path}]")


def is_bob(df: pd.DataFrame) -> pd.Series:
    return (df["BASIN"] == "NI") & (df["SUBBASIN"] == "BB")


# ---------------------------------------------------------------- 1) missing
def missing_wind(df: pd.DataFrame) -> None:
    miss = df["USA_WIND"].isna()
    decade = (df["SEASON"] // 10 * 10).astype(int).astype(str) + "s"
    for key, name, fname in [(df["BASIN"], "BASIN", "missing_usa_wind_by_basin.csv"),
                             (decade.rename("DECADE"), "DECADE", "missing_usa_wind_by_decade.csv")]:
        t = pd.DataFrame({"rows": miss.groupby(key).size(),
                          "missing USA_WIND": miss.groupby(key).sum().astype(int)})
        t["missing %"] = (100 * t["missing USA_WIND"] / t["rows"]).round(2)
        t.index.name = name
        show(f"1) Rows with missing USA_WIND by {name}", t, fname)
    print("Note: src/data/clean.py drops rows without USA_WIND (rule 4) before saving "
          "tracks.parquet, so 0 here is expected; this checks that the filter held.")


# --------------------------------------------------------------- 2) RI target
def add_ri(df: pd.DataFrame) -> pd.DataFrame:
    """Attach USA_WIND at t+24h by timestamp match within SID, and RI."""
    assert not df.duplicated(["SID", "ISO_TIME"]).any(), "duplicate SID/ISO_TIME rows"
    fut = df[["SID", "ISO_TIME", "USA_WIND"]].rename(
        columns={"ISO_TIME": "TIME_P24", "USA_WIND": "USA_WIND_P24"})
    fut["ISO_TIME"] = fut["TIME_P24"] - pd.Timedelta(hours=24)  # row at s is t+24h for t = s-24h
    out = df.merge(fut, on=["SID", "ISO_TIME"], how="left", validate="one_to_one")

    matched = out["TIME_P24"].notna()
    gap = out.loc[matched, "TIME_P24"] - out.loc[matched, "ISO_TIME"]
    assert (gap == pd.Timedelta(hours=24)).all(), "matched t+24h gap is not exactly 24 hours"
    print(f"\nt+24h matched by timestamp: {matched.sum():,} of {len(out):,} rows "
          f"(gap asserted == 24 h); unmatched rows are excluded from RI counts")

    out = out[matched].copy()
    out["DWIND_FUT_24"] = out["USA_WIND_P24"] - out["USA_WIND"]
    out["RI"] = (out["DWIND_FUT_24"] >= RI_THRESHOLD_KT).astype(int)
    d = out["DIST2LAND"]
    out["LOC"] = np.select([d > 0, d == 0], ["over water", "over land"], default="DIST2LAND missing")
    out["NEAR_LAND"] = (d > 0) & (d <= NEAR_LAND_KM)
    return out


def ri_by_year(ri: pd.DataFrame, label: str, fname: str) -> pd.DataFrame:
    g = ri.groupby("SEASON")
    t = pd.DataFrame({
        "storms": g["SID"].nunique(),
        "rows": g.size(),
        "RI rows": g["RI"].sum(),
        "RI over water": ri[ri["LOC"] == "over water"].groupby("SEASON")["RI"].sum(),
        "RI over land": ri[ri["LOC"] == "over land"].groupby("SEASON")["RI"].sum(),
        f"RI over water <= {NEAR_LAND_KM} km of land": ri[ri["NEAR_LAND"]].groupby("SEASON")["RI"].sum(),
        "RI DIST2LAND missing": ri[ri["LOC"] == "DIST2LAND missing"].groupby("SEASON")["RI"].sum(),
    }).fillna(0).astype(int)
    t.loc["TOTAL"] = [ri["SID"].nunique(), *t.drop(columns="storms").sum().tolist()]
    t.index.name = "SEASON"
    show(f"2) RI counts per year: {label}\n"
         f"   RI = USA_WIND(t+24h) - USA_WIND(t) >= {RI_THRESHOLD_KT} kt; location is DIST2LAND at t\n"
         f"   'over water' = DIST2LAND > 0; 'over land' = DIST2LAND == 0; "
         f"'<= {NEAR_LAND_KM} km' = 0 < DIST2LAND <= {NEAR_LAND_KM} (subset of over water)", t, fname)
    return t


# ------------------------------------------------------------------ 3) splits
def bob_splits(ri: pd.DataFrame) -> pd.DataFrame:
    assert (ri.groupby("SID")["SEASON"].nunique() == 1).all(), "SEASON varies within a SID"
    rows, sids = [], {}
    bob = ri[is_bob(ri)]
    for name, (lo, hi) in SPLITS.items():
        s = bob[bob["SEASON"].between(lo, hi)]
        sids[name] = set(s["SID"])
        rows.append({"split": name, "seasons": f"{lo}-{hi if hi < 9999 else 'on'}",
                     "storms": s["SID"].nunique(), "rows": len(s),
                     "RI rows": int(s["RI"].sum()),
                     "RI rows over water": int(s.loc[s["LOC"] == "over water", "RI"].sum()),
                     "storms with RI": s.loc[s["RI"] == 1, "SID"].nunique()})
    names = list(sids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sids[a] & sids[b]), f"SID in both {a} and {b}"
    assert sum(len(v) for v in sids.values()) == bob["SID"].nunique(), "Bay of Bengal storm outside every split"
    t = pd.DataFrame(rows)
    show("3) Bay of Bengal (BASIN NI, SUBBASIN BB at t) by split; no SID in two splits (asserted)",
         t, "bob_splits.csv", index=False)
    return t


# ----------------------------------------------------------------- 4) Amphan
def amphan_map(df: pd.DataFrame) -> None:
    a = df[(df["NAME"] == "AMPHAN") & (df["SEASON"] == 2020)].sort_values("ISO_TIME")
    assert a["SID"].nunique() == 1, f"expected one AMPHAN 2020 storm, got {a['SID'].nunique()}"
    t = a[["ISO_TIME", "LAT", "LON", "USA_WIND", "USA_PRES", "DIST2LAND", "BASIN", "SUBBASIN"]]
    show(f"4) Cyclone Amphan 2020 track (SID {a['SID'].iloc[0]}), rows in tracks.parquet",
         t.assign(ISO_TIME=t["ISO_TIME"].dt.strftime("%Y-%m-%d %H:%M")), "amphan_track.csv", index=False)

    hover = [f"{tm:%d %b %H:%M} UTC<br>USA_WIND {w:.0f} kt<br>DIST2LAND {d:.0f} km"
             for tm, w, d in zip(a["ISO_TIME"], a["USA_WIND"], a["DIST2LAND"])]
    fig = go.Figure([
        go.Scattergeo(lat=a["LAT"], lon=a["LON"], mode="lines",
                      line=dict(width=1.5, color="grey"), hoverinfo="skip", showlegend=False),
        go.Scattergeo(lat=a["LAT"], lon=a["LON"], mode="markers", text=hover, hoverinfo="text",
                      marker=dict(size=9, color=a["USA_WIND"], colorscale="YlOrRd",
                                  line=dict(width=0.5, color="black"),
                                  colorbar=dict(title="USA_WIND (kt)")),
                      showlegend=False),
    ])
    fig.update_geos(projection_type="mercator", showland=True, landcolor="#eeeeee",
                    showcountries=True, showocean=True, oceancolor="#dfefff",
                    lataxis_range=[a["LAT"].min() - 4, a["LAT"].max() + 4],
                    lonaxis_range=[a["LON"].min() - 6, a["LON"].max() + 6])
    fig.update_layout(title="Cyclone Amphan (2020), 6-hourly IBTrACS main track, coloured by USA_WIND",
                      margin=dict(l=10, r=10, t=50, b=10))
    os.makedirs(os.path.dirname(AMPHAN_HTML), exist_ok=True)
    fig.write_html(AMPHAN_HTML, include_plotlyjs="cdn")
    print(f"[saved {AMPHAN_HTML}]")


def main() -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    df = pd.read_parquet(TRACKS_PATH)
    print(f"read {TRACKS_PATH}: {len(df):,} rows, {df['SID'].nunique():,} storms, "
          f"seasons {df['SEASON'].min()}-{df['SEASON'].max()}")

    missing_wind(df)
    ri = add_ri(df)
    ri_by_year(ri, "ALL BASINS", "ri_per_year_all_basins.csv")
    ri_by_year(ri[is_bob(ri)], "BAY OF BENGAL (BASIN NI, SUBBASIN BB at t)", "ri_per_year_bay_of_bengal.csv")
    bob_splits(ri)
    amphan_map(df)


if __name__ == "__main__":
    main()
