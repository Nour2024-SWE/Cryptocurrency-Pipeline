"""
regime_eval.py
===============
Framework points 42-44:
  42. Regime-specific evaluation (full OOS, calm, each predefined crisis, MS states)
  43. Model-family authority analysis (Econ/AI/Hybrid weight shares by regime)
  44. Model-set composition analysis (|S_t|, inclusion frequency by regime)
"""
import numpy as np
import pandas as pd

import config
from losses import performance_table


def evaluate_all_regimes(forecast_df: pd.DataFrame, model_cols: list,
                          regime_flags: pd.DataFrame) -> pd.DataFrame:
    """
    forecast_df: [date, coin, actual_RV, <model columns>]
    regime_flags: [date, coin, Calm, Crisis, <crisis dummies>, MSHigh, MSLow]

    Returns File 11 / Tables 9-10: Coin Period Model QLIKE RMSE MAE OOS_R2 Rank
    for the full OOS sample, Calm, each predefined crisis, and each MS state.
    """
    merged = forecast_df.merge(regime_flags, on=["date", "coin"], how="left")
    frames = [performance_table(merged, model_cols, period_label="Full_OOS")]

    for flag_col, label in [("Calm", "Calm"), ("Crisis", "All_Crises")]:
        sub = merged[merged[flag_col] == 1]
        if len(sub):
            frames.append(performance_table(sub, model_cols, period_label=label))

    for name in config.OOS_CRISIS_REGIMES:
        if name in merged.columns:
            sub = merged[merged[name] == 1]
            if len(sub):
                frames.append(performance_table(sub, model_cols, period_label=name))

    for col, label in [("MSHigh", "MS_High"), ("MSLow", "MS_Low")]:
        if col in merged.columns:
            sub = merged[merged[col] == 1]
            if len(sub):
                frames.append(performance_table(sub, model_cols, period_label=label))

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(config.OUTPUT_DIR / "file11_performance_metrics.csv", index=False)
    return out


def family_weight_shares(dmce_df: pd.DataFrame, regime_flags: pd.DataFrame,
                          coin: str) -> pd.DataFrame:
    """
    Framework point 43 / Table 12: average EconWeight/AIWeight/HybridWeight by
    regime, plus the most frequent contributing family.

    dmce_df: output of dmce.build_dmce for a single coin, with EconWeight,
             AIWeight, HybridWeight, models_in_St columns.
    """
    d = dmce_df.merge(regime_flags[regime_flags.coin == coin], on="date", how="left")
    rows = []

    def _most_frequent_family(sub):
        shares = {"Econ": sub["EconWeight"].mean(), "AI": sub["AIWeight"].mean(),
                  "Hybrid": sub["HybridWeight"].mean()}
        return max(shares, key=shares.get)

    periods = {"Full_OOS": d, "Calm": d[d.get("Calm", 0) == 1], "Crisis": d[d.get("Crisis", 0) == 1]}
    if "MSHigh" in d.columns:
        periods["MS_High"] = d[d["MSHigh"] == 1]
        periods["MS_Low"] = d[d["MSHigh"] == 0]

    for label, sub in periods.items():
        if len(sub) == 0:
            continue
        rows.append({
            "Coin": coin, "Period": label,
            "Avg Econ Weight": sub["EconWeight"].mean(),
            "Avg AI Weight": sub["AIWeight"].mean(),
            "Avg Hybrid Weight": sub["HybridWeight"].mean(),
            "Most frequent model family": _most_frequent_family(sub),
        })
    return pd.DataFrame(rows)


def model_set_composition(dmce_df: pd.DataFrame, regime_flags: pd.DataFrame,
                           coin: str, all_models: list) -> pd.DataFrame:
    """
    Framework point 44 / Table 11: |S_t| over time and per-model inclusion
    frequency (full-sample, calm, crisis, MSHigh).
    """
    d = dmce_df.merge(regime_flags[regime_flags.coin == coin], on="date", how="left")
    d["set_size"] = d["models_in_St"].apply(len)

    def _freq(sub, model):
        if len(sub) == 0:
            return np.nan
        return sub["models_in_St"].apply(lambda s: model in s).mean()

    periods = {"full_sample": d, "calm": d[d.get("Calm", 0) == 1], "crisis": d[d.get("Crisis", 0) == 1]}
    if "MSHigh" in d.columns:
        periods["MSHigh"] = d[d["MSHigh"] == 1]

    rows = []
    for m in all_models:
        row = {"Coin": coin, "Model": m}
        for label, sub in periods.items():
            row[f"{label}_inclusion_frequency"] = _freq(sub, m)
        rows.append(row)
    return pd.DataFrame(rows)
