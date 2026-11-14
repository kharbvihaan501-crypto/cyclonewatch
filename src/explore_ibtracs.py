"""Quick look at the raw IBTrACS file (no modelling)."""
import pandas as pd

PATH = "data/raw/ibtracs.since1980.list.v04r01.csv"

# Row 1 after the header is the units row -> skip it.
# keep_default_na=False: BASIN "NA" (North Atlantic) must not be parsed as NaN.
# Blank / single-space cells are the file's real missing values.
df = pd.read_csv(PATH, skiprows=[1], low_memory=False,
                 keep_default_na=False, na_values=[" ", ""])
df["USA_WIND"] = pd.to_numeric(df["USA_WIND"], errors="coerce")

print(f"rows:           {len(df):,}")
print(f"unique storms:  {df['SID'].nunique():,}")
print(f"columns ({df.shape[1]}):")
print(list(df.columns))

print("\nstorms per BASIN:")
print(df.groupby("BASIN")["SID"].nunique().sort_values(ascending=False).to_string())

print("\nshare of rows with USA_WIND missing, per BASIN:")
share = df["USA_WIND"].isna().groupby(df["BASIN"]).mean().sort_values(ascending=False)
print((share * 100).round(1).astype(str).add("%").to_string())
print(f"\noverall: {df['USA_WIND'].isna().mean() * 100:.1f}%")
