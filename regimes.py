"""
regimes.py
==========
Framework point 9: predefined calendar-based crisis regimes.
"""
import numpy as np
import pandas as pd

import config


def attach_crisis_dummies(daily: pd.DataFrame) -> pd.DataFrame:
    """Adds one 0/1 dummy per regime in config.CRISIS_REGIMES, plus Crisis and Calm."""
    daily = daily.copy()
    crisis_any = np.zeros(len(daily), dtype=bool)

    for name, (start, end) in config.CRISIS_REGIMES.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        flag = (daily["date"] >= start) & (daily["date"] <= end)
        daily[name] = flag.astype(int)
        crisis_any |= flag.to_numpy()

    daily["Crisis"] = crisis_any.astype(int)
    daily["Calm"] = 1 - daily["Crisis"]
    return daily


def crisis_definition_table() -> pd.DataFrame:
    """Framework Table 3: crisis and regime definitions."""
    rows = []
    for name, (start, end) in config.CRISIS_REGIMES.items():
        rows.append({"Regime": name, "Start": start, "End": end})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_prep import load_daily_long
    d = attach_crisis_dummies(load_daily_long())
    print(d[["date", "coin", "Crisis", "Calm"] + list(config.CRISIS_REGIMES)].sample(5))
    print(crisis_definition_table())
