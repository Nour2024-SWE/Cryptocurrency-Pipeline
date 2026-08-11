"""
jumps.py
========
Framework point 15: Bipower Variation, Jumps, and Continuous Components.

    BV_t = mu1^-2 * sum_{i=2..Mt} |r_i,t| |r_{i-1,t}|,   mu1 = sqrt(2/pi)
    J_t  = max(RV_t - BV_t, 0)
    C_t  = RV_t - J_t

*** DATA CAVEAT ***
A statistically correct BV_t requires the actual 5-minute log returns within
each day. The merged daily file the pipeline ships with (see config.DAILY_WIDE_CSV)
only contains the already-aggregated daily RV/LogRV -- the intraday returns are
not in it. Two paths are supported:

1. Preferred: point config.FIVE_MIN_LONG_CSV at a long-format 5-minute file with
   columns [timestamp, coin, close] (or open/high/low/close/volume). Then call
   `bv_jump_continuous_from_intraday()` which implements the formula exactly.

2. Fallback (used automatically if the intraday file is absent): a widely used
   daily-RV-only jump proxy based on Andersen-Bollerslev-Diebold-style truncation,
   `bv_jump_continuous_proxy()`. It is NOT a substitute for the true intraday
   bipower variation and is clearly flagged in all outputs (a `jump_is_proxy`
   column is attached to the daily dataset). Replace it as soon as intraday data
   is available.
"""
import warnings

import numpy as np
import pandas as pd

import config

MU1 = np.sqrt(2.0 / np.pi)


def bv_jump_continuous_from_intraday(returns_5min: pd.DataFrame) -> pd.DataFrame:
    """
    Exact implementation of Framework point 15 from 5-minute log returns.

    Parameters
    ----------
    returns_5min : DataFrame with columns [date, coin, ret] where `ret` is the
        5-minute log return (already computed per point 6), sorted within each
        (date, coin) group in chronological (intraday) order.

    Returns
    -------
    DataFrame [date, coin, BV] -- to be merged onto the daily RV panel, after
    which J = max(RV-BV,0), C = RV-J are computed by `attach_jump_components`.
    """
    out = []
    for (d, c), g in returns_5min.groupby(["date", "coin"], sort=False):
        r = g["ret"].to_numpy()
        if len(r) < 2:
            bv = np.nan
        else:
            bv = (MU1 ** -2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
        out.append({"date": d, "coin": c, "BV": bv})
    return pd.DataFrame(out)


def bv_jump_continuous_proxy(daily: pd.DataFrame, z_alpha: float = 3.0) -> pd.DataFrame:
    """
    Fallback proxy used ONLY when intraday 5-minute returns are unavailable.

    Method: treat abnormally large daily log-return moves relative to a rolling
    local volatility estimate as jump days (Lee-Mykland-style flag at the daily
    frequency), and allocate a shrinkage share of RV to the jump component on
    flagged days only. This is a coarse approximation intended to keep the
    pipeline runnable end-to-end; it is not equivalent to true bipower variation
    and should be replaced once 5-minute data is supplied.

    Sets BV_t = RV_t - J_t directly (by construction C_t + J_t = RV_t holds).
    """
    warnings.warn(
        "jumps.bv_jump_continuous_proxy: no intraday 5-minute data found at "
        f"{config.FIVE_MIN_LONG_CSV}. Falling back to a DAILY-RV-ONLY jump proxy. "
        "This is NOT the true bipower-variation jump test in Framework point 15. "
        "Supply the intraday file and re-run to get exact BV/J/C.",
        stacklevel=2,
    )
    daily = daily.sort_values(["coin", "date"]).copy()
    out_frames = []
    for c, g in daily.groupby("coin"):
        g = g.copy()
        # local volatility of daily returns (22-day rolling std, min_periods 5)
        local_sigma = g["daily_return"].rolling(22, min_periods=5).std()
        z = g["daily_return"] / local_sigma.replace(0, np.nan)
        is_jump_day = z.abs() > z_alpha
        is_jump_day = is_jump_day.fillna(False)
        # On flagged days, attribute a fraction of RV to the jump component,
        # proportional to how extreme the standardized move is (capped at 0.9).
        frac = np.clip((z.abs() - z_alpha) / z_alpha, 0, 0.9).fillna(0.0)
        J = np.where(is_jump_day, g["RV"] * frac, 0.0)
        C = g["RV"] - J
        BV = C  # by construction RV = C + J
        out_frames.append(pd.DataFrame({
            "date": g["date"], "coin": c, "BV": BV, "jump_is_proxy": True,
        }))
    return pd.concat(out_frames, ignore_index=True)


def attach_jump_components(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches BV, J, C to the daily panel. Uses intraday data if
    config.FIVE_MIN_LONG_CSV exists, else the daily-only proxy.

    Output columns added: BV, J, C, jump_is_proxy (bool)
    """
    daily = daily.sort_values(["coin", "date"]).reset_index(drop=True)

    if config.FIVE_MIN_LONG_CSV.exists():
        five = pd.read_csv(config.FIVE_MIN_LONG_CSV, parse_dates=["timestamp"])
        five = five.sort_values(["coin", "timestamp"])
        five["date"] = five["timestamp"].dt.floor("D")
        five["logclose"] = np.log(five["close"])
        five["ret"] = five.groupby(["coin", "date"])["logclose"].diff()
        five = five.dropna(subset=["ret"])
        bv_df = bv_jump_continuous_from_intraday(five[["date", "coin", "ret"]])
        daily = daily.merge(bv_df, on=["date", "coin"], how="left")
        daily["jump_is_proxy"] = False
    else:
        bv_df = bv_jump_continuous_proxy(daily)
        daily = daily.merge(bv_df, on=["date", "coin"], how="left")

    daily["BV"] = daily["BV"].clip(lower=0.0)
    daily["J"] = (daily["RV"] - daily["BV"]).clip(lower=0.0)
    daily["C"] = daily["RV"] - daily["J"]
    daily["jump_is_proxy"] = daily["jump_is_proxy"].fillna(True)
    return daily


if __name__ == "__main__":
    from data_prep import load_daily_long
    d = load_daily_long()
    d2 = attach_jump_components(d)
    print(d2[["date", "coin", "RV", "BV", "J", "C", "jump_is_proxy"]].head())
    print("Any RV != C+J beyond tol:",
          (~np.isclose(d2.RV, d2.C + d2.J)).sum())
