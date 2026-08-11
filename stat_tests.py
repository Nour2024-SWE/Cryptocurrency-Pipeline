"""
stat_tests.py
==============
Framework points 45-46:
  45. Final MCS over full sample / crisis / calm / MS-high / MS-low, comparing
      Naive, standalone, hybrids, benchmark ensembles, and DMCE.
  46. Diebold-Mariano tests for specific pairwise comparisons.
"""
import numpy as np
import pandas as pd
from scipy import stats

import config
from mcs import model_confidence_set


def run_final_mcs(qlike_long_one_coin: pd.DataFrame, regime_flags: pd.DataFrame,
                   coin: str) -> pd.DataFrame:
    """
    qlike_long_one_coin: [date, model, QLIKE_daily] for one coin, full universe.
    Returns Table 13 rows: coin, period, mcs_selected_models, p_value.
    """
    merged = qlike_long_one_coin.merge(
        regime_flags[regime_flags.coin == coin], on="date", how="left")
    rows = []
    periods = {"Full_OOS": merged, "Calm": merged[merged.get("Calm", 0) == 1],
               "Crisis": merged[merged.get("Crisis", 0) == 1]}
    if "MSHigh" in merged.columns:
        periods["MS_High"] = merged[merged["MSHigh"] == 1]
        periods["MS_Low"] = merged[merged["MSHigh"] == 0]

    for label, sub in periods.items():
        if len(sub) == 0:
            continue
        res = model_confidence_set(sub[["date", "model", "QLIKE_daily"]])
        rows.append({"Coin": coin, "Period": label,
                     "MCS_selected_models": res["selected_models"],
                     "p_value": res["p_value"]})
    return pd.DataFrame(rows)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """
    DM test on the loss differential d_t = L_A,t - L_B,t.
    H0: E[d_t] = 0. Uses a Newey-West-style long-run variance with (h-1) lags.
    Returns statistic, p-value, and which model has the lower average loss.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"statistic": np.nan, "p_value": np.nan, "winner": "insufficient_data"}
    d_bar = d.mean()
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0
    for lag in range(1, h):
        w = 1 - lag / h
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        lrv += 2 * w * cov
    se = np.sqrt(max(lrv, 1e-12) / n)
    dm_stat = d_bar / se
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    if d_bar < 0:
        winner = "A"
    elif d_bar > 0:
        winner = "B"
    else:
        winner = "tie"
    return {"statistic": float(dm_stat), "p_value": float(p_value), "winner": winner}


def run_dm_comparisons(qlike_wide: pd.DataFrame, comparisons: list) -> pd.DataFrame:
    """
    qlike_wide: DataFrame indexed by date with one column per model (QLIKE_daily).
    comparisons: list of (label, model_A, model_B) tuples.
    Returns Table 14.
    """
    rows = []
    for label, a, b in comparisons:
        if a not in qlike_wide.columns or b not in qlike_wide.columns:
            continue
        res = diebold_mariano(qlike_wide[a].to_numpy(), qlike_wide[b].to_numpy())
        rows.append({"Comparison": label, "Model_A": a, "Model_B": b, **res})
    return pd.DataFrame(rows)


# Point 46's required comparison set (fill in the concrete winning model names
# at run time, e.g. best_standalone = regime_eval / dmce selection results).
def default_dm_comparisons(best_standalone, best_hybrid, best_ai) -> list:
    return [
        ("DMCE vs best standalone", "DMCE", best_standalone),
        ("DMCE vs best hybrid", "DMCE", best_hybrid),
        ("DMCE vs equal-weight ensemble", "DMCE", "EW_Ensemble"),
        ("DMCE vs static ensemble", "DMCE", "Static_Ensemble"),
        ("DMCE vs HAR-RV", "DMCE", "HAR_RV"),
        ("DMCE vs best AI model", "DMCE", best_ai),
        ("DMCE vs Realized GARCH", "DMCE", "Realized_GARCH"),
    ]
