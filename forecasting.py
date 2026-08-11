"""
forecasting.py
================
Framework points 28-30: expanding-window one-step-ahead forecasting for every
standalone model, producing File 5:

    date coin actual_RV Naive HAR_RV HAR_RV_J HAR_RV_CJ Realized_GARCH RF XGBoost LightGBM LSTM TCN

Design (point 28):
  - Initial training: 2018-01-01 - 2019-06-30
  - Validation: 2019-07-01 - 2019-12-31 (used only for point 29 hyperparameter tuning)
  - OOS test: 2020-01-01 - 2026-06-20, expanding window, forecast one day ahead,
    then append the realized day and move forward.

Practical retraining cadence (point 28's "acceptable practical rule", extended
here to the tree models for tractability -- see README):
  - HAR-RV / HAR-RV-J / HAR-RV-CJ : refit every day (closed-form OLS, cheap)
  - Realized GARCH                : refit every RGARCH_RETRAIN_FREQ (default weekly)
  - RF / XGBoost / LightGBM       : refit every config.ML_RETRAIN_FREQ (default monthly)
  - LSTM / TCN                    : refit every config.DL_RETRAIN_FREQ (default monthly)
Forecasts are produced every day regardless of the refit cadence, using the
most recently trained model and the latest available features (no future
information is used at any point).
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from models_econ import (
    forecast_naive, fit_har_rv, forecast_har_rv, fit_har_rv_j, forecast_har_rv_j,
    fit_har_rv_cj, forecast_har_rv_cj, fit_realized_garch, forecast_realized_garch,
)
from models_ml import fit_rf, fit_xgb, fit_lgbm, forecast_ml
from models_dl import fit_lstm, fit_tcn, forecast_dl

RGARCH_RETRAIN_FREQ = "W"  # weekly refit for the MLE-based Realized GARCH (tractability)


def _period_key(date: pd.Timestamp, freq: str) -> str:
    if freq == "D":
        return date.strftime("%Y-%m-%d")
    if freq == "W":
        return date.strftime("%Y-%W")
    if freq == "M":
        return date.strftime("%Y-%m")
    raise ValueError(freq)


def run_expanding_window(coin_df: pd.DataFrame, coin: str,
                          oos_start=config.OOS_START, oos_end=config.OOS_END,
                          verbose=True) -> pd.DataFrame:
    """
    coin_df must be the single-coin output of features.build_features (sorted by date),
    already containing HAR/jump columns and target_RV_next / target_LogRV_next.
    """
    g = coin_df.sort_values("date").reset_index(drop=True)
    g = g.dropna(subset=["RV_m"]).reset_index(drop=True)  # drop HAR burn-in

    oos_mask = (g["date"] >= pd.Timestamp(oos_start)) & (g["date"] <= pd.Timestamp(oos_end))
    oos_idx = g.index[oos_mask].tolist()
    if not oos_idx:
        raise ValueError("No OOS observations found - check date ranges / data coverage.")

    records = []
    cached = {}  # model_name -> (fitted_object, period_key)

    iterator = tqdm(oos_idx, desc=f"{coin} expanding window") if verbose else oos_idx
    for i in iterator:
        train = g.iloc[:i + 1]  # information up to and including day t
        last = train.iloc[-1]
        today = last["date"]

        row = {"date": today, "coin": coin, "actual_RV": last["target_RV_next"]}

        # --- Naive ---
        row["Naive"] = forecast_naive(last)

        # --- HAR family (refit daily; closed-form, cheap) ---
        har = fit_har_rv(train)
        row["HAR_RV"] = forecast_har_rv(har, last)
        harj = fit_har_rv_j(train)
        row["HAR_RV_J"] = forecast_har_rv_j(harj, last)
        harcj = fit_har_rv_cj(train)
        row["HAR_RV_CJ"] = forecast_har_rv_cj(harcj, last)

        # --- Realized GARCH (weekly refit) ---
        key = _period_key(today, RGARCH_RETRAIN_FREQ)
        if cached.get("rgarch_key") != key:
            cached["rgarch"] = fit_realized_garch(train)
            cached["rgarch_key"] = key
        row["Realized_GARCH"] = forecast_realized_garch(cached["rgarch"], train)

        # --- ML models (monthly refit, per config.ML_RETRAIN_FREQ) ---
        key = _period_key(today, config.ML_RETRAIN_FREQ)
        if cached.get("rf_key") != key:
            cached["rf"] = fit_rf(train)
            cached["xgb"] = fit_xgb(train)
            cached["lgbm"] = fit_lgbm(train)
            cached["rf_key"] = cached["xgb_key"] = cached["lgbm_key"] = key
        row["RF"] = forecast_ml(cached["rf"], last)
        row["XGBoost"] = forecast_ml(cached["xgb"], last)
        row["LightGBM"] = forecast_ml(cached["lgbm"], last)

        # --- DL models (monthly refit, per config.DL_RETRAIN_FREQ) ---
        key = _period_key(today, config.DL_RETRAIN_FREQ)
        if cached.get("lstm_key") != key:
            cached["lstm"] = fit_lstm(train, L=22, epochs=40)
            cached["tcn"] = fit_tcn(train, L=22, epochs=40)
            cached["lstm_key"] = cached["tcn_key"] = key
        logrv_hist = train["LogRV"].to_numpy()
        row["LSTM"] = forecast_dl(cached["lstm"], logrv_hist)
        row["TCN"] = forecast_dl(cached["tcn"], logrv_hist)

        records.append(row)

    return pd.DataFrame(records)


def run_all_coins(feat_df: pd.DataFrame, coins=config.COINS, **kwargs) -> pd.DataFrame:
    out = []
    for c in coins:
        sub = feat_df[feat_df.coin == c]
        out.append(run_expanding_window(sub, c, **kwargs))
    result = pd.concat(out, ignore_index=True)
    result.to_csv(config.OUTPUT_DIR / "file5_standalone_forecasts.csv", index=False)
    return result


if __name__ == "__main__":
    # Smoke test on a short OOS window so this finishes quickly.
    import warnings
    warnings.filterwarnings("ignore")
    from data_prep import load_daily_long
    from jumps import attach_jump_components
    from features import build_features

    d = attach_jump_components(load_daily_long())
    f = build_features(d)
    sub = f[f.coin == "BTCUSDT"]
    out = run_expanding_window(sub, "BTCUSDT", oos_start="2020-01-01", oos_end="2020-01-10")
    print(out)
