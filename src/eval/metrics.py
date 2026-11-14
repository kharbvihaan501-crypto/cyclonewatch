"""Row and episode metrics with storm-level bootstrap intervals.

evaluate(df, prob_col, threshold) keeps only over-water rows (DIST2LAND > 0 at t)
and reports, per subset:
  storms, RI rows, RI episodes, episodes caught, median lead time (h),
  false-alarm episodes per storm, PR-AUC, precision, recall, Brier score,
  recall on RI rows where the last-24h wind rose < 30 kt (DWIND_24 < 30).

95% intervals: 1,000 bootstrap resamples of whole storms (fixed seed), done
separately inside each subset. A storm drawn k times counts k times (row and
episode weights = k). Resamples with no RI rows are skipped and counted.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.eval.episodes import episode_alerts

NEAR_LAND_KM = 400
SLOW_KT = 30
N_BOOT = 1000
SEED = 0

SUBSETS = {
    "Bay of Bengal": lambda d: (d["BASIN"] == "NI") & (d["SUBBASIN"] == "BB"),
    "North Indian Ocean": lambda d: d["BASIN"] == "NI",
    f"All basins, <= {NEAR_LAND_KM} km of land": lambda d: d["DIST2LAND"] <= NEAR_LAND_KM,
    "All basins": lambda d: pd.Series(True, index=d.index),
}

METRICS = ["PR-AUC", "precision", "recall", "Brier", "recall (last-24h rise < 30 kt)",
           "episodes caught %", "median lead h", "false-alarm episodes / storm",
           "alert rate %", "red hours / storm"]
ROW_HOURS = 6  # each row is one 6-hourly time step


def _stats(y, p, red, slow, w_rows, ep_caught, ep_lead, w_eps, fa_per_storm, w_storms) -> dict:
    """All metrics for one (possibly weighted) sample. Weights are bootstrap multiplicities."""
    pos = w_rows[y == 1].sum()
    out = {}
    out["PR-AUC"] = average_precision_score(y, p, sample_weight=w_rows) if pos > 0 else np.nan
    tp = w_rows[(y == 1) & red].sum()
    n_red = w_rows[red].sum()
    out["precision"] = tp / n_red if n_red > 0 else np.nan
    out["recall"] = tp / pos if pos > 0 else np.nan
    out["Brier"] = np.average((p - y) ** 2, weights=w_rows)
    sp = w_rows[(y == 1) & slow].sum()
    out["recall (last-24h rise < 30 kt)"] = w_rows[(y == 1) & slow & red].sum() / sp if sp > 0 else np.nan
    n_ep = w_eps.sum()
    out["episodes caught %"] = 100 * w_eps[ep_caught].sum() / n_ep if n_ep > 0 else np.nan
    lead_ok = ep_caught & (w_eps > 0)
    out["median lead h"] = (np.median(np.repeat(ep_lead[lead_ok], w_eps[lead_ok]))
                            if lead_ok.any() else np.nan)
    out["false-alarm episodes / storm"] = np.average(fa_per_storm, weights=w_storms)
    out["alert rate %"] = 100 * n_red / w_rows.sum()
    out["red hours / storm"] = ROW_HOURS * n_red / w_storms.sum()
    return out


def _subset_eval(d: pd.DataFrame, prob_col: str, threshold: float, rng) -> dict:
    eps, storms = episode_alerts(d, prob_col, threshold)
    sids = storms["SID"].to_numpy()
    code = {s: i for i, s in enumerate(sids)}
    row_storm = d["SID"].map(code).to_numpy()
    ep_storm = eps["SID"].map(code).to_numpy()

    y = d["RI"].to_numpy().astype(int)
    p = d[prob_col].to_numpy().astype(float)
    red = p >= threshold
    slow = (d["DWIND_24"] < SLOW_KT).to_numpy()
    ep_caught = eps["CAUGHT"].to_numpy().astype(bool)
    ep_lead = eps["LEAD_H"].to_numpy()
    fa = storms["N_FALSE_ALARMS"].to_numpy().astype(float)

    def run(w_storms):
        return _stats(y, p, red, slow, w_storms[row_storm], ep_caught, ep_lead,
                      w_storms[ep_storm], fa, w_storms)

    n = len(sids)
    point = run(np.ones(n, int))
    boots, skipped = [], 0
    storm_has_ri = np.bincount(row_storm, weights=y, minlength=n) > 0
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        if not (w * storm_has_ri).any():
            skipped += 1
            continue
        boots.append(run(w))
    boots = pd.DataFrame(boots, columns=METRICS)

    res = {"storms": n, "rows": len(d), "RI rows": int(y.sum()),
           "RI rows rise < 30 kt": int(((y == 1) & slow).sum()),
           "RI episodes": len(eps), "episodes caught": int(ep_caught.sum()),
           "missed episodes": int((~ep_caught).sum()),
           "false-alarm episodes": int(fa.sum())}
    for m in METRICS:
        res[m] = point[m]
        b = boots[m].dropna() if len(boots) else pd.Series(dtype=float)
        res[m + " lo"] = np.percentile(b, 2.5) if len(b) else np.nan
        res[m + " hi"] = np.percentile(b, 97.5) if len(b) else np.nan
    res["boot used"] = len(boots)
    res["boot skipped (no RI)"] = skipped
    return res


def evaluate(df: pd.DataFrame, prob_col: str, threshold: float, seed: int = SEED) -> pd.DataFrame:
    """Over-water metrics per subset with 95% storm-bootstrap intervals. One row per subset."""
    water = df[df["DIST2LAND"] > 0]
    rng = np.random.default_rng(seed)
    rows = []
    for name, mask in SUBSETS.items():
        d = water[mask(water).to_numpy()]
        if len(d) == 0:
            rows.append({"subset": name, "storms": 0})
            continue
        rows.append({"subset": name, **_subset_eval(d, prob_col, threshold, rng)})
    return pd.DataFrame(rows)


def threshold_for_false_alarms(df: pd.DataFrame, prob_col: str, target: float = 1.0) -> tuple[float, float]:
    """Threshold whose false-alarm episodes per storm (all basins, over water) is closest to target.

    Scans every distinct probability as a candidate (use on validation only).
    False alarms per storm need not be monotonic in the threshold, so this is a
    full scan, not a bisection. Ties go to the higher threshold (fewer alerts).
    Returns (threshold, achieved false-alarm episodes per storm).
    """
    water = df[df["DIST2LAND"] > 0]
    best = None
    for thr in np.unique(water[prob_col].to_numpy()):
        _, storms = episode_alerts(water, prob_col, thr)
        fa = storms["N_FALSE_ALARMS"].mean()
        key = (abs(fa - target), -thr)
        if best is None or key < best[0]:
            best = (key, float(thr), float(fa))
    return best[1], best[2]


ALERT_BASIN = "NI"  # logged rule: 20% of over-water VALIDATION rows red, BASIN == NI only


def threshold_for_alert_rate(df: pd.DataFrame, prob_col: str, target: float = 0.20,
                             basin: str | None = ALERT_BASIN) -> tuple[float, float]:
    """Threshold whose share of over-water rows on red (p >= threshold) is closest to target.

    Rows used: DIST2LAND > 0 and, unless basin is None, BASIN == basin (default NI,
    the logged rule). target is a fraction (0.20 = 20%). Candidates are the distinct
    probabilities, so with tied probabilities the exact target may not be reachable
    (persistence moves in steps). Ties go to the higher threshold. Use on validation
    only. Returns (threshold, achieved share on those rows).
    """
    rows = df["DIST2LAND"] > 0
    if basin is not None:
        rows &= (df["BASIN"].astype(str) == basin)
    assert rows.any(), f"no over-water rows for basin {basin}"
    p = np.sort(df.loc[rows, prob_col].to_numpy())
    cand = np.unique(p)
    share = 1 - np.searchsorted(p, cand, side="left") / len(p)  # share of rows with p >= cand
    i = min(range(len(cand)), key=lambda k: (abs(share[k] - target), -cand[k]))
    return float(cand[i]), float(share[i])


def _fmt(v, nd):
    return "-" if pd.isna(v) else f"{v:.{nd}f}"


def format_table(res: pd.DataFrame) -> pd.DataFrame:
    """Compact text table: 'estimate [lo, hi]' per metric."""
    nd = {"PR-AUC": 3, "precision": 3, "recall": 3, "Brier": 4,
          "recall (last-24h rise < 30 kt)": 3, "episodes caught %": 1,
          "median lead h": 1, "false-alarm episodes / storm": 2,
          "alert rate %": 1, "red hours / storm": 1}
    out = res[["subset", "storms", "rows", "RI rows", "RI rows rise < 30 kt", "RI episodes", "episodes caught",
               "missed episodes",
               "false-alarm episodes"]].copy()
    for m in METRICS:
        out[m] = [f"{_fmt(e, nd[m])} [{_fmt(lo, nd[m])}, {_fmt(hi, nd[m])}]"
                  for e, lo, hi in zip(res[m], res[m + " lo"], res[m + " hi"])]
    out["boot used / skipped"] = [f"{u} / {s}" for u, s in zip(res["boot used"], res["boot skipped (no RI)"])]
    return out
