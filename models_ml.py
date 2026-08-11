"""
models_ml.py
=============
Framework points 23-25: Random Forest, XGBoost, LightGBM.

Input feature vector X_t (framework point 23) and target Y_t = RV_{t+1} or
LogRV_{t+1} (point 16: ML/DL is allowed to train on LogRV and exponentiate back).
This implementation trains on LogRV_{t+1} and exponentiates, applying the
positivity floor, which is numerically more stable for tree models on
heavy-tailed RV.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

import config
from features import ML_FEATURE_COLS


def _xy(train: pd.DataFrame):
    d = train.dropna(subset=ML_FEATURE_COLS + ["target_LogRV_next"])
    X = d[ML_FEATURE_COLS].to_numpy()
    y = d["target_LogRV_next"].to_numpy()
    return X, y


def fit_rf(train: pd.DataFrame, n_estimators=500, max_depth=None):
    X, y = _xy(train)
    m = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    m.fit(X, y)
    return m


def fit_xgb(train: pd.DataFrame, n_estimators=500, max_depth=4, learning_rate=0.05):
    X, y = _xy(train)
    m = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, random_state=config.RANDOM_SEED,
        n_jobs=-1, objective="reg:squarederror",
    )
    m.fit(X, y)
    return m


def fit_lgbm(train: pd.DataFrame, n_estimators=500, num_leaves=31, learning_rate=0.05):
    X, y = _xy(train)
    m = lgb.LGBMRegressor(
        n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, random_state=config.RANDOM_SEED,
        n_jobs=-1, verbosity=-1,
    )
    m.fit(X, y)
    return m


def forecast_ml(model, last_row: pd.Series) -> float:
    x = last_row[ML_FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
    log_rv_hat = model.predict(x)[0]
    return max(float(np.exp(log_rv_hat)), config.RV_FLOOR)


# ---------------------------------------------------------------------------
# Point 29: hyperparameter tuning on the validation window only, selected by
# validation QLIKE. Small, practical grids (expand as needed).
# ---------------------------------------------------------------------------
RF_GRID = [{"n_estimators": ne, "max_depth": md}
           for ne in (300, 500) for md in (5, 10, None)]
XGB_GRID = [{"n_estimators": ne, "max_depth": md, "learning_rate": lr}
            for ne in (300, 500) for md in (3, 4, 6) for lr in (0.03, 0.05, 0.1)]
LGBM_GRID = [{"n_estimators": ne, "num_leaves": nl, "learning_rate": lr}
             for ne in (300, 500) for nl in (15, 31, 63) for lr in (0.03, 0.05, 0.1)]


def _qlike(actual, pred):
    pred = np.maximum(pred, config.RV_FLOOR)
    return np.mean(np.log(pred) + actual / pred)


def tune_by_validation_qlike(fit_fn, grid, train_df, valid_df):
    """Generic validation-QLIKE hyperparameter search (framework point 29)."""
    best_score, best_params, best_model = np.inf, None, None
    for params in grid:
        m = fit_fn(train_df, **params)
        preds = np.array([forecast_ml(m, valid_df.iloc[i]) for i in range(len(valid_df))])
        actual = valid_df["target_RV_next"].to_numpy()
        mask = ~np.isnan(actual)
        score = _qlike(actual[mask], preds[mask])
        if score < best_score:
            best_score, best_params, best_model = score, params, m
    return best_model, best_params, best_score
