"""
data_prep.py
============
Loads the merged wide-format daily dataset (output of Framework points 1-8)
and reshapes it into a long panel: one row per (date, coin).

Expected wide columns (per coin suffix SYM in config.COINS):
    RV_5m_{SYM}, LogRV_5m_{SYM}, daily_return_{SYM}, daily_volume_{SYM},
    valid_5min_bars_{SYM}, valid_5min_returns_{SYM}
"""
import numpy as np
import pandas as pd

import config


def load_daily_long(path=None) -> pd.DataFrame:
    """Return a long panel: date, coin, RV, LogRV, daily_return, daily_volume, valid_5min_bars."""
    path = path or config.DAILY_WIDE_CSV
    wide = pd.read_csv(path, parse_dates=["date"])
    wide = wide.sort_values("date").reset_index(drop=True)

    frames = []
    for sym in config.COINS:
        cols = {
            f"RV_5m_{sym}": "RV",
            f"LogRV_5m_{sym}": "LogRV",
            f"daily_return_{sym}": "daily_return",
            f"daily_volume_{sym}": "daily_volume",
            f"valid_5min_bars_{sym}": "valid_5min_bars",
        }
        missing = [c for c in cols if c not in wide.columns]
        if missing:
            raise KeyError(f"Missing expected columns for {sym}: {missing}")
        sub = wide[["date"] + list(cols.keys())].rename(columns=cols)
        sub["coin"] = sym
        frames.append(sub)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.sort_values(["coin", "date"]).reset_index(drop=True)

    # Guard against non-positive RV before logging (framework point 7)
    bad = long_df["RV"] <= 0
    if bad.any():
        long_df.loc[bad, "LogRV"] = np.log(config.EPS_LOGRV)
    long_df["RV"] = long_df["RV"].clip(lower=0.0)

    return long_df


def coverage_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """Framework Table 1: data coverage by coin."""
    rows = []
    for sym in config.COINS:
        sub = long_df[long_df.coin == sym]
        rows.append({
            "Coin": config.COIN_DISPLAY[sym],
            "First available observation": sub.date.min().date(),
            "Last observation": sub.date.max().date(),
            "Number of daily RV observations": len(sub),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_daily_long()
    print(df.head())
    print(coverage_table(df))
