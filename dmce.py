"""
dmce.py
=======
Framework points 39-41: Dynamic Model-Confidence Ensemble (DMCE) and the five
benchmark ensembles it is compared against.

DMCE forecast:
    RV_hat_DMCE,t+1 = sum_{m in S_t} w_{m,t} * RV_hat_{m,t+1}

Weighting schemes (point 40): inverse-QLIKE (default), equal, softmax(eta).
"""
import numpy as np
import pandas as pd

import config


def _inverse_qlike_weights(roll_qlike: dict) -> dict:
    # QLIKE can be negative (it's log(pred)+actual/pred, not bounded below by 0);
    # inverse weighting requires positive average loss, so shift by a constant
    # so all values are positive before inverting.
    if not roll_qlike:
        return {}
    shift = -min(roll_qlike.values()) + 1e-6 if min(roll_qlike.values()) <= 0 else 0.0
    inv = {m: 1.0 / (v + shift) for m, v in roll_qlike.items() if np.isfinite(v)}
    tot = sum(inv.values())
    return {m: v / tot for m, v in inv.items()} if tot > 0 else {m: 1 / len(roll_qlike) for m in roll_qlike}


def _equal_weights(roll_qlike: dict) -> dict:
    valid = [m for m, v in roll_qlike.items() if np.isfinite(v)]
    return {m: 1.0 / len(valid) for m in valid} if valid else {}


def _softmax_weights(roll_qlike: dict, eta: float) -> dict:
    items = [(m, v) for m, v in roll_qlike.items() if np.isfinite(v)]
    if not items:
        return {}
    vals = np.array([v for _, v in items])
    ex = np.exp(-eta * (vals - vals.min()))  # numerically stable
    w = ex / ex.sum()
    return {m: w[i] for i, (m, _) in enumerate(items)}


def compute_weights(roll_qlike: dict, scheme=config.DMCE_WEIGHT_SCHEME, eta=5) -> dict:
    if scheme == "inverse_qlike":
        return _inverse_qlike_weights(roll_qlike)
    if scheme == "equal":
        return _equal_weights(roll_qlike)
    if scheme == "softmax":
        return _softmax_weights(roll_qlike, eta)
    raise ValueError(scheme)


def build_dmce(all_forecasts_wide: pd.DataFrame, st_table: pd.DataFrame,
               roll_qlike_panel: pd.DataFrame, roll_window=config.ROLL_QLIKE_WINDOW,
               scheme=config.DMCE_WEIGHT_SCHEME, eta=5) -> pd.DataFrame:
    """
    all_forecasts_wide: [date, actual_RV, model_1, model_2, ...] for ONE coin
        (standalone + hybrid forecasts merged).
    st_table: output of mcs.rolling_mcs for the same coin: [date, models_in_St, ...]
    roll_qlike_panel: output of losses.add_rolling_qlike (long) for the same coin,
        used to fetch each model's rolling QLIKE at t (uses only past info, since
        add_rolling_qlike already shifts by 1 day).

    Returns File 10: [date coin actual_RV DMCE_forecast models_in_St weights EconWeight AIWeight HybridWeight]
    """
    roll_col = f"QLIKE_roll{roll_window}"
    roll_wide = roll_qlike_panel.pivot(index="date", columns="model", values=roll_col)

    fc = all_forecasts_wide.set_index("date")
    st = st_table.set_index("date")

    records = []
    for date in fc.index:
        if date not in st.index:
            continue
        s_t = st.loc[date, "models_in_St"]
        if not isinstance(s_t, list):
            s_t = list(s_t) if hasattr(s_t, "__iter__") else [s_t]

        if date in roll_wide.index:
            rq = {m: roll_wide.loc[date, m] for m in s_t if m in roll_wide.columns}
        else:
            rq = {}
        rq = {m: v for m, v in rq.items() if pd.notna(v)}
        if not rq:
            rq = {m: 1.0 for m in s_t if m in fc.columns}  # fallback: equal weight

        weights = compute_weights(rq, scheme=scheme, eta=eta)

        pred = 0.0
        for m, w in weights.items():
            if m in fc.columns and pd.notna(fc.loc[date, m]):
                pred += w * fc.loc[date, m]
        pred = max(pred, config.RV_FLOOR)

        econ_w = sum(w for m, w in weights.items() if m in config.ECON_MODELS)
        ai_w = sum(w for m, w in weights.items() if m in config.AI_MODELS)
        hybrid_w = sum(w for m, w in weights.items() if m.startswith("hybrid_"))

        records.append({
            "date": date, "actual_RV": fc.loc[date, "actual_RV"],
            "DMCE_forecast": pred, "models_in_St": list(weights.keys()),
            "weights": weights, "EconWeight": econ_w, "AIWeight": ai_w,
            "HybridWeight": hybrid_w,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Point 41: benchmark ensembles
# ---------------------------------------------------------------------------
def equal_weight_ensemble(df: pd.DataFrame, cols: list) -> pd.Series:
    avail = [c for c in cols if c in df.columns]
    return df[avail].mean(axis=1)


def static_inverse_qlike_ensemble(df: pd.DataFrame, cols: list,
                                   validation_qlike: dict) -> pd.Series:
    """point 41.4: weights fixed from the validation-period average QLIKE only."""
    inv = {m: 1.0 / v for m, v in validation_qlike.items() if m in cols and np.isfinite(v) and v > 0}
    tot = sum(inv.values())
    w = {m: v / tot for m, v in inv.items()}
    pred = sum(w.get(m, 0.0) * df[m] for m in cols if m in df.columns)
    return pred


def best_single_standalone(validation_qlike: dict, standalone_cols=config.STANDALONE_MODELS) -> str:
    cand = {m: v for m, v in validation_qlike.items() if m in standalone_cols}
    return min(cand, key=cand.get)


def best_dynamic_hybrid(validation_qlike: dict) -> str:
    cand = {m: v for m, v in validation_qlike.items() if m.startswith("hybrid_")}
    return min(cand, key=cand.get)
