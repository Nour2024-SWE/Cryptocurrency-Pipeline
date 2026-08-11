"""
losses.py
=========
Framework points 35-37: QLIKE (primary loss), RMSE, MAE, out-of-sample R^2,
and rolling-average QLIKE per model.
"""
import numpy as np
import pandas as pd

import config


def qlike(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pred = np.maximum(np.asarray(pred, dtype=float), config.RV_FLOOR)
    actual = np.asarray(actual, dtype=float)
    return np.log(pred) + actual / pred


def rmse(actual, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def mae(actual, pred) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def oos_r2(actual, pred, naive_pred) -> float:
    actual, pred, naive_pred = map(np.asarray, (actual, pred, naive_pred))
    num = np.sum((actual - pred) ** 2)
    den = np.sum((actual - naive_pred) ** 2)
    return float(1 - num / den) if den > 0 else np.nan


def build_qlike_panel(forecast_df: pd.DataFrame, model_cols: list, id_cols=("date", "coin"),
                       actual_col="actual_RV") -> pd.DataFrame:
    """
    Long panel: date, coin, model, QLIKE_daily (plus rolling QLIKE columns added
    by add_rolling_qlike). `forecast_df` is wide (one column per model), as in
    File 5 / File 6.
    """
    avail_cols = [m for m in model_cols if m in forecast_df.columns]
    keep = list(id_cols) + [actual_col] + avail_cols
    melted = forecast_df[keep].melt(id_vars=list(id_cols) + [actual_col],
                                     value_vars=avail_cols, var_name="model", value_name="pred")
    melted["QLIKE_daily"] = qlike(melted[actual_col].to_numpy(), melted["pred"].to_numpy())
    return melted.drop(columns=["pred"])


def add_rolling_qlike(qlike_panel: pd.DataFrame,
                       windows=config.ROLL_QLIKE_WINDOWS_ROBUST) -> pd.DataFrame:
    """Adds QLIKE_roll{W} columns using only past losses (s in [t-W, t-1])."""
    qlike_panel = qlike_panel.sort_values(["coin", "model", "date"]).reset_index(drop=True)
    for W in windows:
        col = f"QLIKE_roll{W}"
        qlike_panel[col] = (
            qlike_panel.groupby(["coin", "model"])["QLIKE_daily"]
            .transform(lambda s: s.shift(1).rolling(W, min_periods=max(5, W // 4)).mean())
        )
    return qlike_panel


def build_file8(forecast_df: pd.DataFrame, model_cols: list) -> pd.DataFrame:
    panel = build_qlike_panel(forecast_df, model_cols)
    panel = add_rolling_qlike(panel)
    panel.to_csv(config.OUTPUT_DIR / "file8_qlike_losses.csv", index=False)
    return panel


def performance_table(forecast_df: pd.DataFrame, model_cols: list,
                       naive_col="Naive", period_label="Full") -> pd.DataFrame:
    """Framework Table 5/6/7/8/9/10-style row set: QLIKE, RMSE, MAE, OOS R^2, rank."""
    rows = []
    for coin, g in forecast_df.groupby("coin"):
        actual = g["actual_RV"].to_numpy()
        naive_pred = g[naive_col].to_numpy() if naive_col in g.columns else None
        for m in model_cols:
            if m not in g.columns:
                continue
            pred = g[m].to_numpy()
            mask = np.isfinite(actual) & np.isfinite(pred)
            if mask.sum() == 0:
                continue
            row = {
                "Coin": coin, "Period": period_label, "Model": m,
                "QLIKE": float(np.mean(qlike(actual[mask], pred[mask]))),
                "RMSE": rmse(actual[mask], pred[mask]),
                "MAE": mae(actual[mask], pred[mask]),
            }
            if naive_pred is not None:
                row["OOS_R2"] = oos_r2(actual[mask], pred[mask], naive_pred[mask])
            rows.append(row)
    out = pd.DataFrame(rows)
    out["Rank"] = out.groupby(["Coin", "Period"])["QLIKE"].rank(method="min")
    return out.sort_values(["Coin", "Period", "Rank"])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n), "coin": "BTCUSDT",
        "actual_RV": np.abs(rng.normal(0.001, 0.0004, n)),
    })
    df["Naive"] = np.abs(df["actual_RV"] + rng.normal(0, 0.0002, n))
    df["HAR_RV"] = np.abs(df["actual_RV"] + rng.normal(0, 0.00015, n))
    panel = build_file8(df, ["Naive", "HAR_RV"])
    print(panel.tail())
    print(performance_table(df, ["Naive", "HAR_RV"]))
