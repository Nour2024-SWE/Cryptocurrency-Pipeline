"""
descriptives.py
================
Framework point 11: descriptive statistics for RV and LogRV, by coin, by
predefined regime, and by Markov state; ADF and KPSS stationarity tests.
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller, kpss

import config


def _desc_row(x: pd.Series) -> dict:
    x = x.dropna()
    return {
        "Mean": x.mean(), "Median": x.median(), "Std": x.std(),
        "Min": x.min(), "Max": x.max(),
        "Skewness": skew(x, bias=False), "Kurtosis": kurtosis(x, fisher=True, bias=False),
    }


def descriptive_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Framework Table 2: descriptive statistics of RV and LogRV by coin."""
    rows = []
    for c, g in daily.groupby("coin"):
        for var in ["RV", "LogRV"]:
            row = {"Coin": config.COIN_DISPLAY.get(c, c), "Variable": var}
            row.update(_desc_row(g[var]))
            rows.append(row)
    return pd.DataFrame(rows)


def descriptive_by_regime(daily: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """RV descriptives grouped by an arbitrary 0/1 (or categorical) regime column."""
    rows = []
    for c, g in daily.groupby("coin"):
        for r, gg in g.groupby(regime_col):
            x = gg["RV"].dropna()
            rows.append({
                "Coin": config.COIN_DISPLAY.get(c, c), "Regime": r,
                "Mean RV": x.mean(), "Median RV": x.median(),
                "Std RV": x.std(), "Max RV": x.max(),
            })
    return pd.DataFrame(rows)


def stationarity_tests(daily: pd.DataFrame) -> pd.DataFrame:
    """ADF and KPSS tests on LogRV per coin (descriptive use only, per point 11)."""
    rows = []
    for c, g in daily.groupby("coin"):
        y = g.sort_values("date")["LogRV"].dropna()
        adf_stat, adf_p, *_ = adfuller(y, autolag="AIC")
        try:
            kpss_stat, kpss_p, *_ = kpss(y, regression="c", nlags="auto")
        except Exception:
            kpss_stat, kpss_p = np.nan, np.nan
        rows.append({
            "Coin": config.COIN_DISPLAY.get(c, c),
            "ADF stat": adf_stat, "ADF p-value": adf_p,
            "KPSS stat": kpss_stat, "KPSS p-value": kpss_p,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_prep import load_daily_long
    from regimes import attach_crisis_dummies

    d = attach_crisis_dummies(load_daily_long())
    print(descriptive_table(d))
    print(descriptive_by_regime(d, "Crisis"))
    print(stationarity_tests(d))
