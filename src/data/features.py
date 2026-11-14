"""Features at time t and the RI target (data rules 6-7).

Every lag (t-6h, t-12h, t-24h) and the target (t+24h) is found BY TIMESTAMP:
merge on SID and ISO_TIME shifted by the offset. Never shift() by row position.
"""
import numpy as np
import pandas as pd

TRACKS_PATH = "data/processed/tracks.parquet"
RI_THRESHOLD_KT = 30
BASINS = ["EP", "NA", "NI", "SA", "SI", "SP", "WP"]
EARTH_RADIUS_KM = 6371.0

FEATURES = [
    "USA_WIND", "USA_PRES",
    "DWIND_6", "DWIND_12", "DWIND_24", "DPRES_24",
    "SPEED_12", "DIR_12", "SPEED_24", "DIR_24",
    "LAT", "DIST2LAND", "MONTH_SIN", "MONTH_COS", "BASIN",
]
TARGET = "RI"
ID_COLS = ["SID", "NAME", "SEASON", "ISO_TIME", "SUBBASIN", "LON"]
# Future values kept for evaluation and the demo only. NEVER use as features.
FUTURE_COLS = ["USA_WIND_P24", "DWIND_FUT_24"]


def _merge_offset(df: pd.DataFrame, hours: int, cols: list[str]) -> pd.DataFrame:
    """Attach `cols` from the row exactly `hours` from t in the same SID.

    hours < 0 looks back, hours > 0 looks ahead. Adds TIME_<sfx> (the matched
    row's own time) and asserts it is exactly `hours` from t.
    """
    sfx = f"_{'M' if hours < 0 else 'P'}{abs(hours)}"
    other = df[["SID", "ISO_TIME", *cols]].rename(
        columns={c: c + sfx for c in cols} | {"ISO_TIME": "TIME" + sfx})
    # The row at time s is the (hours)-offset row for t = s - hours.
    other["ISO_TIME"] = other["TIME" + sfx] - pd.Timedelta(hours=hours)
    out = df.merge(other, on=["SID", "ISO_TIME"], how="left", validate="one_to_one")

    matched = out["TIME" + sfx].notna()
    gap = out.loc[matched, "TIME" + sfx] - out.loc[matched, "ISO_TIME"]
    assert (gap == pd.Timedelta(hours=hours)).all(), f"bad time gap for offset {hours}h"
    return out


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 (past) to point 2 (now), 0-360, 0 = north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(lon2 - lon1)
    x = np.sin(dlmb) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def build_features(tracks: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    if tracks is None:
        tracks = pd.read_parquet(TRACKS_PATH)
    assert not tracks.duplicated(["SID", "ISO_TIME"]).any(), "duplicate SID/ISO_TIME in tracks"
    df = tracks.copy()
    n0, s0 = len(df), df["SID"].nunique()

    past_cols = ["USA_WIND", "USA_PRES", "LAT", "LON"]
    for h in (-6, -12, -24):
        df = _merge_offset(df, h, past_cols)
    df = _merge_offset(df, 24, ["USA_WIND"])

    # Rule 7: need both t-24h and t+24h in the same storm.
    df = df[df["TIME_M24"].notna() & df["TIME_P24"].notna()].copy()

    for h in (6, 12, 24):
        df[f"DWIND_{h}"] = df["USA_WIND"] - df[f"USA_WIND_M{h}"]
    df["DPRES_24"] = df["USA_PRES"] - df["USA_PRES_M24"]  # NaN if either is missing
    for h in (12, 24):
        dist = _haversine_km(df[f"LAT_M{h}"], df[f"LON_M{h}"], df["LAT"], df["LON"])
        df[f"SPEED_{h}"] = dist / h  # km/h
        df[f"DIR_{h}"] = _bearing_deg(df[f"LAT_M{h}"], df[f"LON_M{h}"], df["LAT"], df["LON"])

    month = df["ISO_TIME"].dt.month
    df["MONTH_SIN"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * (month - 1) / 12)
    df["BASIN"] = pd.Categorical(df["BASIN"], categories=BASINS)
    assert df["BASIN"].notna().all(), "unknown BASIN code"

    df["DWIND_FUT_24"] = df["USA_WIND_P24"] - df["USA_WIND"]
    df[TARGET] = (df["DWIND_FUT_24"] >= RI_THRESHOLD_KT).astype("int8")

    out = df[ID_COLS + FEATURES + FUTURE_COLS + [TARGET]].sort_values(["SID", "ISO_TIME"])
    out = out.reset_index(drop=True)

    if verbose:
        print(f"tracks:                        rows {n0:>8,}  storms {s0:>6,}")
        print(f"with t-24h and t+24h (rule 7): rows {len(out):>8,}  storms {out['SID'].nunique():>6,}"
              f"   removed rows {n0 - len(out):,}")
        print("missing values among kept rows:")
        print(out[FEATURES].isna().sum()[lambda s: s > 0].to_string())
    return out


if __name__ == "__main__":
    build_features()
