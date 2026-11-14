"""Clean raw IBTrACS rows -> data/processed/tracks.parquet (data rules 2-5, 8).

Run from the project root:  python -m src.data.clean
"""
import os

import pandas as pd

from src.data.load import load_raw

OUT_PATH = "data/processed/tracks.parquet"
KEEP_COLS = ["SID", "NAME", "SEASON", "ISO_TIME", "TRACK_TYPE", "BASIN", "SUBBASIN",
             "LAT", "LON", "USA_WIND", "USA_PRES", "DIST2LAND", "NATURE"]
SYNOPTIC_HOURS = (0, 6, 12, 18)
NUMERIC = ["SEASON", "LAT", "LON", "USA_WIND", "USA_PRES", "DIST2LAND"]


def report(label: str, df: pd.DataFrame, prev: pd.DataFrame | None) -> None:
    """Print rows and storms after a step, plus what it removed."""
    msg = f"{label:<46} rows {len(df):>8,}  storms {df['SID'].nunique():>6,}"
    if prev is not None:
        msg += (f"   removed rows {len(prev) - len(df):>8,}"
                f"  storms {prev['SID'].nunique() - df['SID'].nunique():>5,}")
    print(msg)


def track_type_by_year(df: pd.DataFrame) -> None:
    """Rule 2 prerequisite: TRACK_TYPE counts by year (rows)."""
    year = pd.to_datetime(df["ISO_TIME"]).dt.year.rename("YEAR")
    table = pd.crosstab(year, df["TRACK_TYPE"])
    print("TRACK_TYPE row counts by year:")
    print(table.to_string())
    print()


def provisional_storms(raw: pd.DataFrame, kept: pd.DataFrame, n_seasons: int = 2) -> pd.DataFrame:
    """Rule 8: storms that may still be provisional.

    A storm qualifies if it is in the last n_seasons of the RAW file, or has any
    PROVISIONAL / US-PROVISIONAL track rows (rule 2 drops those rows). Computed
    from raw, so storms that rule 2 removed entirely are not hidden.
    """
    last = int(raw["SEASON"].max())
    is_prov_type = raw["TRACK_TYPE"].str.contains("PROVISIONAL")
    flagged = raw[(raw["SEASON"] > last - n_seasons) | is_prov_type]
    g = flagged.groupby("SID")
    out = g.agg(NAME=("NAME", "first"), SEASON=("SEASON", "first"),
                BASIN=("BASIN", "first"), SUBBASIN=("SUBBASIN", "first"),
                FIRST=("ISO_TIME", "min"), LAST=("ISO_TIME", "max"),
                RAW_ROWS=("SID", "size"))
    out["PROV_ROWS"] = flagged[flagged["TRACK_TYPE"].str.contains("PROVISIONAL")].groupby("SID").size()
    out["PROV_ROWS"] = out["PROV_ROWS"].fillna(0).astype(int)
    out["KEPT_ROWS"] = kept.groupby("SID").size().reindex(out.index).fillna(0).astype(int)
    return out.sort_values(["SEASON", "FIRST"]).reset_index()


def clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_raw()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], format="%Y-%m-%d %H:%M:%S")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    raw = df.copy()
    report("raw (units row skipped)", df, None)
    track_type_by_year(df)

    # Rule 2: main tracks only.
    prev, df = df, df[df["TRACK_TYPE"] == "main"]
    report("rule 2: TRACK_TYPE == 'main'", df, prev)

    # Rule 3: synoptic hours only (other rows are interpolated).
    prev, df = df, df[(df["ISO_TIME"].dt.hour.isin(SYNOPTIC_HOURS))
                      & (df["ISO_TIME"].dt.minute == 0)]
    report("rule 3: 00/06/12/18 UTC only", df, prev)

    # Rule 4: USA_WIND only, never WMO_WIND. Rows without it cannot be used.
    prev, df = df, df[df["USA_WIND"].notna()]
    report("rule 4: USA_WIND present (no WMO fallback)", df, prev)

    # Rule 5: NATURE == TS.
    prev, df = df, df[df["NATURE"] == "TS"]
    report("rule 5: NATURE == 'TS'", df, prev)

    out = (df[KEEP_COLS].sort_values(["SID", "ISO_TIME"]).reset_index(drop=True))
    assert not out.duplicated(["SID", "ISO_TIME"]).any(), "duplicate SID/time rows"
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved {OUT_PATH}: {len(out):,} rows, {out['SID'].nunique():,} storms")
    return out, raw


def basin_summary(raw: pd.DataFrame, kept: pd.DataFrame) -> None:
    b = pd.DataFrame({"raw": raw.groupby("BASIN")["SID"].nunique(),
                      "kept": kept.groupby("BASIN")["SID"].nunique()}).fillna(0).astype(int)
    print("\nstorms per BASIN, raw vs kept:")
    print(b.to_string())
    for name, mask_r, mask_k in [("NI/BB", (raw.BASIN == "NI") & (raw.SUBBASIN == "BB"),
                                  (kept.BASIN == "NI") & (kept.SUBBASIN == "BB"))]:
        print(f"{name} storms (any row): raw {raw[mask_r]['SID'].nunique()}, "
              f"kept {kept[mask_k]['SID'].nunique()}")


if __name__ == "__main__":
    tracks, raw = clean()
    basin_summary(raw, tracks)

    prov = provisional_storms(raw, tracks)
    last = int(raw["SEASON"].max())
    print(f"\nRule 8: storms that may still be provisional: {len(prov)} "
          f"(seasons {last - 1}-{last} in raw, or any PROVISIONAL track rows)")
    print("by SEASON / dropped-entirely vs partly kept:")
    prov["STATUS"] = prov["KEPT_ROWS"].map(lambda n: "all dropped" if n == 0 else "partly kept")
    print(pd.crosstab(prov["SEASON"], prov["STATUS"]).to_string())
    print()
    print(prov.to_string(index=False))
