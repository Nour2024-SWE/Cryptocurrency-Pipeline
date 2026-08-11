"""
hybrid.py
=========
Framework points 31-34: pairwise Econ x AI hybrids with a dynamic,
rolling-QLIKE-minimizing combination weight lambda_{e,a,t} in [0,1]:

    RV_hybrid(e,a),t+1 = lambda_{e,a,t} * RV_hat_e,t+1 + (1-lambda_{e,a,t}) * RV_hat_a,t+1

    lambda_{e,a,t} = argmin_{lambda in [0,1]} sum_{s=t-W}^{t-1}
                        QLIKE(RV_s, lambda*RVhat_e,s + (1-lambda)*RVhat_a,s)

Only information strictly before t (s < t) is used, per the framework.
Requires the standalone forecast file (File 5) as input.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import config


def _qlike_series(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    pred = np.maximum(pred, config.RV_FLOOR)
    return np.log(pred) + actual / pred


def _lambda_star(actual_win: np.ndarray, fe_win: np.ndarray, fa_win: np.ndarray) -> float:
    def obj(lam):
        blend = lam * fe_win + (1 - lam) * fa_win
        return np.sum(_qlike_series(actual_win, blend))
    res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def build_dynamic_hybrids(standalone: pd.DataFrame, window: int = config.HYBRID_LAMBDA_WINDOW,
                           pairs=config.HYBRID_PAIRS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    standalone: File-5-shaped DataFrame for ONE coin, sorted by date, with columns
        date, actual_RV, <model columns...>
    Returns (hybrid_forecasts_df, lambda_df).
    """
    standalone = standalone.sort_values("date").reset_index(drop=True)
    n = len(standalone)
    actual = standalone["actual_RV"].to_numpy()

    hyb_cols = {}
    lambda_records = []

    for e, a in pairs:
        col_name = f"hybrid_{e}_{a}"
        hyb_vals = np.full(n, np.nan)
        fe = standalone[e].to_numpy()
        fa = standalone[a].to_numpy()

        for t in range(n):
            if t < window:
                # not enough history for a stable rolling window -> equal weight fallback
                lam = 0.5
            else:
                a_win = actual[t - window:t]
                fe_win = fe[t - window:t]
                fa_win = fa[t - window:t]
                mask = np.isfinite(a_win) & np.isfinite(fe_win) & np.isfinite(fa_win)
                lam = _lambda_star(a_win[mask], fe_win[mask], fa_win[mask]) if mask.sum() > 5 else 0.5

            hyb_vals[t] = lam * fe[t] + (1 - lam) * fa[t]
            lambda_records.append({
                "date": standalone["date"].iloc[t], "hybrid_pair": f"{e}_{a}",
                "lambda_econ": lam, "weight_AI": 1 - lam,
            })
        hyb_cols[col_name] = hyb_vals

    hybrid_df = pd.DataFrame({"date": standalone["date"], "actual_RV": actual, **hyb_cols})
    lambda_df = pd.DataFrame(lambda_records)
    return hybrid_df, lambda_df


def build_all_coins(standalone_all: pd.DataFrame, window: int = config.HYBRID_LAMBDA_WINDOW):
    hyb_frames, lam_frames = [], []
    for c, g in standalone_all.groupby("coin"):
        h, l = build_dynamic_hybrids(g, window=window)
        h["coin"] = c
        l["coin"] = c
        hyb_frames.append(h)
        lam_frames.append(l)
    hyb_all = pd.concat(hyb_frames, ignore_index=True)
    lam_all = pd.concat(lam_frames, ignore_index=True)
    hyb_all.to_csv(config.OUTPUT_DIR / "file6_hybrid_forecasts.csv", index=False)
    lam_all.to_csv(config.OUTPUT_DIR / "file7_hybrid_lambdas.csv", index=False)
    return hyb_all, lam_all


if __name__ == "__main__":
    # Smoke test with synthetic-ish data derived from a short forecasting run.
    import warnings
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    n = 250
    dates = pd.date_range("2020-01-01", periods=n)
    actual = np.abs(rng.normal(0.001, 0.0005, n))
    df = pd.DataFrame({"date": dates, "actual_RV": actual})
    for m in config.ECON_MODELS + config.AI_MODELS:
        df[m] = np.abs(actual + rng.normal(0, 0.0003, n))
    h, l = build_dynamic_hybrids(df, window=60)
    print(h.head())
    print(l.head())
    print(h.shape, l.shape)
