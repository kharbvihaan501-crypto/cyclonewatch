"""Independent leakage check: recompute features + RI for 3 random VALIDATION rows from tracks.parquet.

Run from the project root:  py -3.12 -m src.eval.leakage_check   (never reads test.parquet)
Lookups are by exact (SID, time) key in a dict; no merges, no shifts, no row positions.
"""
import math

import numpy as np
import pandas as pd

SEED = 2026
val = pd.read_parquet("data/processed/val.parquet")
tracks = pd.read_parquet("data/processed/tracks.parquet")
key = {(r.SID, r.ISO_TIME): r for r in tracks.itertuples(index=False)}

def hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

rows = val.sample(3, random_state=SEED)
out, problems = [], 0
for _, v in rows.iterrows():
    sid, t = v["SID"], v["ISO_TIME"]
    now = key[(sid, t)]
    past = {h: key.get((sid, t - pd.Timedelta(hours=h))) for h in (6, 12, 24)}
    fut = key.get((sid, t + pd.Timedelta(hours=24)))
    assert past[24] is not None and fut is not None, "row should have t-24h and t+24h"
    for h, r in past.items():
        assert r is None or (r.SID == sid and t - r.ISO_TIME == pd.Timedelta(hours=h))
    m = t.month
    rec = {
        "USA_WIND": now.USA_WIND, "USA_PRES": now.USA_PRES,
        "DWIND_6": now.USA_WIND - past[6].USA_WIND if past[6] else np.nan,
        "DWIND_12": now.USA_WIND - past[12].USA_WIND if past[12] else np.nan,
        "DWIND_24": now.USA_WIND - past[24].USA_WIND,
        "DPRES_24": now.USA_PRES - past[24].USA_PRES,
        "SPEED_12": hav(past[12].LAT, past[12].LON, now.LAT, now.LON) / 12 if past[12] else np.nan,
        "DIR_12": bearing(past[12].LAT, past[12].LON, now.LAT, now.LON) if past[12] else np.nan,
        "SPEED_24": hav(past[24].LAT, past[24].LON, now.LAT, now.LON) / 24,
        "DIR_24": bearing(past[24].LAT, past[24].LON, now.LAT, now.LON),
        "LAT": now.LAT, "DIST2LAND": now.DIST2LAND,
        "MONTH_SIN": math.sin(2 * math.pi * (m - 1) / 12), "MONTH_COS": math.cos(2 * math.pi * (m - 1) / 12),
        "BASIN": now.BASIN,
        "USA_WIND_P24": fut.USA_WIND, "DWIND_FUT_24": fut.USA_WIND - now.USA_WIND,
        "RI": int(fut.USA_WIND - now.USA_WIND >= 30),
    }
    for f, r in rec.items():
        a = v[f]
        if isinstance(r, str) or isinstance(a, str):
            ok = str(a) == str(r)
        else:
            ok = (pd.isna(a) and pd.isna(r)) or (not pd.isna(a) and not pd.isna(r) and abs(float(a) - float(r)) < 1e-9)
        problems += not ok
        fmt = lambda x: x if isinstance(x, str) else ("NaN" if pd.isna(x) else f"{float(x):.4f}".rstrip("0").rstrip("."))
        out.append({"row": f"{v['NAME']} {sid} {t:%Y-%m-%d %H:%M}", "feature": f,
                    "val.parquet": fmt(a), "recomputed": fmt(r), "match": "yes" if ok else "NO"})

df = pd.DataFrame(out)
print(f"seed {SEED}; rows: {list(df['row'].unique())}")
print(df.to_string(index=False))
print(f"\nmismatches: {problems} of {len(df)}")
