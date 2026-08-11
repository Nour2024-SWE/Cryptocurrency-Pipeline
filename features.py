"""
features.py
============
Framework points 13-16:
  13. Lagged RV and LogRV features
  14. HAR daily/weekly/monthly components (levels and logs)
  15. Weekly/monthly aggregates of continuous (C) and jump (J) components
  16. Forecasting target: RV_{t+1} (and LogRV_{t+1} for ML/DL)

All rolling means use only PAST information (rolling windows ending at t, i.e.
the value at t already realized, matching the framework's definitions of
RV_t^(w), RV_t^(m) which include RV_t itself).
"""
import numpy as np
import pandas as pd

import config


def _roll_mean_causal(s: pd.Series, window: int) -> pd.Series:
    """Mean of s_t, s_{t-1}, ..., s_{t-window+1} (inclusive of current obs)."""
    return s.rolling(window=window, min_periods=window).mean()


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Input: daily panel with columns [date, coin, RV, LogRV, C, J, ...]
           (i.e. after jumps.attach_jump_components).
    Output: same panel with lag/HAR/jump-aggregate features and the next-day
            target columns appended, sorted by coin, date.
    """
    daily = daily.sort_values(["coin", "date"]).reset_index(drop=True)
    out = []

    for c, g in daily.groupby("coin"):
        g = g.sort_values("date").copy()

        # --- point 13: lagged RV / LogRV ---
        for L in config.LAGS:
            g[f"RV_lag{L}"] = g["RV"].shift(L)
            g[f"LogRV_lag{L}"] = g["LogRV"].shift(L)

        # --- point 14: HAR components ---
        g["RV_d"] = g["RV"]
        g["RV_w"] = _roll_mean_causal(g["RV"], config.WEEKLY_WINDOW)
        g["RV_m"] = _roll_mean_causal(g["RV"], config.MONTHLY_WINDOW)
        g["LogRV_d"] = g["LogRV"]
        g["LogRV_w"] = _roll_mean_causal(g["LogRV"], config.WEEKLY_WINDOW)
        g["LogRV_m"] = _roll_mean_causal(g["LogRV"], config.MONTHLY_WINDOW)

        # --- point 15: continuous / jump weekly & monthly aggregates ---
        if "C" in g.columns and "J" in g.columns:
            g["C_w"] = _roll_mean_causal(g["C"], config.WEEKLY_WINDOW)
            g["C_m"] = _roll_mean_causal(g["C"], config.MONTHLY_WINDOW)
            g["J_w"] = _roll_mean_causal(g["J"], config.WEEKLY_WINDOW)
            g["J_m"] = _roll_mean_causal(g["J"], config.MONTHLY_WINDOW)

        # --- point 16: forecasting target RV_{t+1}, LogRV_{t+1} ---
        g["target_RV_next"] = g["RV"].shift(-1)
        g["target_LogRV_next"] = g["LogRV"].shift(-1)

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    return result.sort_values(["coin", "date"]).reset_index(drop=True)


ML_FEATURE_COLS = (
    ["RV"]
    + [f"RV_lag{L}" for L in config.LAGS]
    + ["RV_w", "RV_m", "LogRV", "LogRV_w", "LogRV_m"]
    + ["C", "J", "C_w", "C_m", "J_w", "J_m"]
)


if __name__ == "__main__":
    from data_prep import load_daily_long
    from jumps import attach_jump_components

    d = attach_jump_components(load_daily_long())
    f = build_features(d)
    print(f[f.coin == "BTCUSDT"][["date", "RV", "RV_w", "RV_m", "C_w", "J_w",
                                   "target_RV_next"]].iloc[20:26])
    print("NaN rows (burn-in) per coin:\n",
          f.groupby("coin").apply(lambda x: x[ML_FEATURE_COLS].isna().any(axis=1).sum()))
