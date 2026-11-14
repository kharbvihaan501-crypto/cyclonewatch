"""Load raw IBTrACS v4 (since 1980) with no filtering beyond the units row.

Run from the project root:  python -m src.data.load
"""
import pandas as pd

RAW_PATH = "data/raw/ibtracs.since1980.list.v04r01.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV.

    - skiprows=[1]: the second row is units.
    - keep_default_na=False: basin code "NA" (North Atlantic) must not become NaN.
      Only blank or single-space cells are missing.
    """
    df = pd.read_csv(
        path,
        skiprows=[1],
        low_memory=False,
        keep_default_na=False,
        na_values=[" ", ""],
    )
    df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")
    assert "NA" in set(df["BASIN"].dropna()), "BASIN 'NA' was lost while loading"
    return df


if __name__ == "__main__":
    df = load_raw()
    print(f"rows:          {len(df):,}")
    print(f"unique storms: {df['SID'].nunique():,}")
    print(f"columns:       {df.shape[1]}")
    print("\nstorms per BASIN:")
    print(df.groupby("BASIN")["SID"].nunique().sort_values(ascending=False).to_string())
    print("\nUSA_WIND missing share per BASIN (%):")
    print((df["USA_WIND"].isna().groupby(df["BASIN"]).mean() * 100).round(1).to_string())
