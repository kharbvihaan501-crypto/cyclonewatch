"""Split by year AND storm: train 1980-2015, validate 2016-2019, test 2020+.

Each storm goes to one split by its SEASON (constant per SID), so no storm is
cut in two. Run from the project root:  python -m src.data.split
"""
import os

import numpy as np
import pandas as pd

from src.data.features import TARGET, build_features

OUT_DIR = "data/processed"
SPLITS = {"train": (1980, 2015), "val": (2016, 2019), "test": (2020, 9999)}


def assign_split(df: pd.DataFrame) -> pd.Series:
    assert (df.groupby("SID")["SEASON"].nunique() == 1).all(), "SEASON varies within a SID"
    conds = [df["SEASON"].between(lo, hi) for lo, hi in SPLITS.values()]
    split = pd.Series(np.select(conds, list(SPLITS), default=""), index=df.index)
    assert (split != "").all(), "row with SEASON outside every split"
    return split


def check_disjoint(parts: dict[str, pd.DataFrame]) -> None:
    sids = {k: set(v["SID"]) for k, v in parts.items()}
    names = list(sids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = sids[a] & sids[b]
            assert not both, f"SIDs in both {a} and {b}: {sorted(both)[:5]}"


def ri_table(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, d in parts.items():
        for subset, m in [("all basins", np.ones(len(d), bool)),
                          ("Bay of Bengal", (d["BASIN"] == "NI") & (d["SUBBASIN"] == "BB"))]:
            s = d[m]
            ri_sids = s.loc[s[TARGET] == 1, "SID"].nunique()
            rows.append({"split": name, "subset": subset,
                         "seasons": f"{s['SEASON'].min()}-{s['SEASON'].max()}" if len(s) else "-",
                         "rows": len(s), "storms": s["SID"].nunique(),
                         "RI rows": int(s[TARGET].sum()),
                         "RI rate %": round(100 * s[TARGET].mean(), 2) if len(s) else np.nan,
                         "storms with RI": ri_sids})
    return pd.DataFrame(rows)


def main() -> dict[str, pd.DataFrame]:
    df = build_features()
    split = assign_split(df)
    parts = {k: df[split == k].reset_index(drop=True) for k in SPLITS}
    check_disjoint(parts)
    assert sum(len(p) for p in parts.values()) == len(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    for k, p in parts.items():
        p.to_parquet(f"{OUT_DIR}/{k}.parquet", index=False)
    print(f"\nsaved {', '.join(f'{OUT_DIR}/{k}.parquet' for k in parts)}")
    print("no SID appears in two splits (asserted)\n")
    print("RI = USA_WIND(t+24h) - USA_WIND(t) >= 30 kt; Bay of Bengal = BASIN NI and SUBBASIN BB at t")
    print(ri_table(parts).to_string(index=False))
    return parts


if __name__ == "__main__":
    main()
