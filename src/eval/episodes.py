"""RI episodes, alert lead time and false-alarm episodes.

Definitions (rows are first sorted by SID, ISO_TIME):
- A "run" is a stretch of consecutive rows in one SID where each row is exactly
  6 h after the previous one. A time gap (a row dropped by cleaning or by the
  over-water filter) ends the run.
- RI episode: a run of RI == 1 rows.
- Red alert: prob >= threshold. Alert run: a run of red rows.
- Episode caught: the alert is red at the episode's first row.
- Lead time (hours): first row of the episode minus the start of the unbroken
  red run that reaches that row. The start is never earlier than the row after
  the previous RI episode in the same storm, so one long red run is not credited
  twice. 0 h means the alert turned red exactly at the first RI row (which is
  already a 24 h warning, since RI at t means intensification over t..t+24h).
  NaN when the episode is not caught.
- False-alarm episode: an alert run that contains no RI == 1 row.
"""
import numpy as np
import pandas as pd

STEP = pd.Timedelta(hours=6)


def _run_ids(df: pd.DataFrame, flag: pd.Series) -> pd.Series:
    """Label runs of consecutive (6 h apart, same SID) rows where flag is True.

    df must be sorted by SID, ISO_TIME. Returns an int id per row, -1 where flag is False.
    """
    flag = flag.astype(bool).to_numpy()
    sid = df["SID"].to_numpy()
    t = df["ISO_TIME"].to_numpy()
    prev_ok = np.zeros(len(df), bool)
    if len(df) > 1:
        prev_ok[1:] = (sid[1:] == sid[:-1]) & (t[1:] - t[:-1] == STEP.to_timedelta64()) & flag[:-1]
    starts = flag & ~prev_ok
    ids = np.cumsum(starts) - 1
    return pd.Series(np.where(flag, ids, -1), index=df.index)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    assert not df.duplicated(["SID", "ISO_TIME"]).any(), "duplicate SID/ISO_TIME rows"
    return df.sort_values(["SID", "ISO_TIME"])


def find_episodes(df: pd.DataFrame, target: str = "RI") -> pd.DataFrame:
    """One row per RI episode: SID, EP_ID, START, END, N_ROWS, START_IDX (index label of first row)."""
    d = _sorted(df)
    ep = _run_ids(d, d[target] == 1)
    d = d.assign(_EP=ep, _IDX=d.index)[ep >= 0]
    out = d.groupby("_EP").agg(SID=("SID", "first"), START=("ISO_TIME", "min"),
                               END=("ISO_TIME", "max"), N_ROWS=("SID", "size"),
                               START_IDX=("_IDX", "first"))
    return out.rename_axis("EP_ID").reset_index()


def episode_alerts(df: pd.DataFrame, prob_col: str, threshold: float,
                   target: str = "RI") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-episode alert outcome and per-storm false-alarm counts.

    Returns
    -------
    episodes : one row per RI episode with CAUGHT (bool) and LEAD_H (float, NaN if missed).
    storms   : one row per SID in df with N_EPISODES, N_CAUGHT, N_FALSE_ALARMS.
    """
    d = _sorted(df)
    red = d[prob_col] >= threshold
    d = d.assign(_RED=red, _ALERT=_run_ids(d, red))

    eps = find_episodes(d, target)
    first = d.loc[eps["START_IDX"]]
    eps["CAUGHT"] = first["_RED"].to_numpy()

    # Start time of each alert run.
    run_start = d[d["_ALERT"] >= 0].groupby("_ALERT")["ISO_TIME"].min()
    start_red = run_start.reindex(first["_ALERT"].to_numpy()).to_numpy()  # NaT where not red
    # Earliest allowed start: row after the previous episode of the same storm.
    prev_end = eps.groupby("SID")["END"].shift(1) + STEP
    start_red = pd.Series(start_red, index=eps.index)
    start_red = start_red.where(prev_end.isna() | (start_red >= prev_end), prev_end)
    eps["LEAD_H"] = np.where(eps["CAUGHT"], (eps["START"] - start_red).dt.total_seconds() / 3600, np.nan)

    # False-alarm episodes: alert runs with no RI row.
    runs = d[d["_ALERT"] >= 0].groupby("_ALERT").agg(SID=("SID", "first"), ANY_RI=(target, "max"))
    fa = runs[runs["ANY_RI"] == 0].groupby("SID").size()

    storms = pd.DataFrame(index=pd.Index(d["SID"].unique(), name="SID"))
    storms["N_EPISODES"] = eps.groupby("SID").size()
    storms["N_CAUGHT"] = eps.groupby("SID")["CAUGHT"].sum()
    storms["N_FALSE_ALARMS"] = fa
    storms = storms.fillna(0).astype(int)
    return eps.drop(columns="START_IDX"), storms.reset_index()
