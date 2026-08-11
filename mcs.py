"""
mcs.py
======
Implements Hansen, Lunde & Nason (2011) "The Model Confidence Set" via the
range statistic + stationary bootstrap, used for:
  - point 38: rolling MCS (loss matrix from the trailing W days)
  - point 45: final/static MCS over full-sample / regime subsets

This is a from-scratch implementation (no off-the-shelf MCS package is
installed in this environment); it follows the elimination algorithm:

  1. Start with the full set of models M0.
  2. Compute the range statistic T_R over the current set using a stationary
     bootstrap to obtain its null distribution.
  3. If p-value < alpha, eliminate the worst model (highest average relative
     loss) and repeat; else stop -- the current set is the (1-alpha) MCS.
"""
import numpy as np
import pandas as pd

import config


def _stationary_bootstrap_indices(n: int, n_boot: int, block_len: float, rng) -> np.ndarray:
    """Politis-Romano stationary bootstrap: returns (n_boot, n) index matrix."""
    p = 1.0 / block_len
    idx = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        i = rng.integers(0, n)
        for t in range(n):
            idx[b, t] = i
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i = (i + 1) % n
    return idx


def _mcs_once(loss_matrix: np.ndarray, alpha: float, n_boot: int, block_len: float, rng):
    """
    loss_matrix: (T, M) array of losses (e.g. QLIKE) for T days x M models
    (no NaNs). Returns the surviving column-index set (list) at confidence
    level 1-alpha, using the range statistic.
    """
    T, M = loss_matrix.shape
    active = list(range(M))
    boot_idx = _stationary_bootstrap_indices(T, n_boot, block_len, rng)

    while len(active) > 1:
        sub = loss_matrix[:, active]
        d_bar = sub.mean(axis=0)  # average loss per model in the active set
        T_R_obs = np.max(d_bar) - np.min(d_bar)

        # bootstrap distribution of the range statistic under H0 (equal predictive ability)
        boot_stats = np.empty(n_boot)
        for b in range(n_boot):
            sub_b = sub[boot_idx[b]]
            d_bar_b = sub_b.mean(axis=0) - d_bar  # recenter (studentized-free simplification)
            boot_stats[b] = np.max(d_bar_b) - np.min(d_bar_b)

        p_value = float(np.mean(boot_stats >= T_R_obs))

        if p_value >= alpha or len(active) <= 1:
            return active, p_value

        # eliminate model with the worst (highest) average loss
        worst_local = int(np.argmax(d_bar))
        active.pop(worst_local)

    return active, 1.0


def model_confidence_set(loss_df: pd.DataFrame, model_col="model", loss_col="QLIKE_daily",
                          alpha=config.MCS_ALPHA, n_boot=config.MCS_N_BOOT,
                          block_len=config.MCS_BLOCK_LEN, seed=config.RANDOM_SEED) -> dict:
    """
    loss_df: long DataFrame with one row per (date, model) for a SINGLE coin and
    a SINGLE evaluation window/period, columns [date, model, loss_col].
    Returns {"selected_models": [...], "p_value": float}.
    """
    wide = loss_df.pivot(index="date", columns=model_col, values=loss_col).dropna()
    if wide.shape[0] < 10 or wide.shape[1] < 2:
        return {"selected_models": list(wide.columns), "p_value": np.nan}
    rng = np.random.default_rng(seed)
    active_idx, p = _mcs_once(wide.to_numpy(), alpha, n_boot, block_len, rng)
    return {"selected_models": [wide.columns[i] for i in active_idx], "p_value": p}


# ---------------------------------------------------------------------------
# Point 38: rolling MCS. Updated daily or weekly (config.MCS_UPDATE_FREQ);
# S_t held fixed between updates if weekly.
# ---------------------------------------------------------------------------
def rolling_mcs(qlike_panel_one_coin: pd.DataFrame, window=config.MCS_WINDOW,
                 alpha=config.MCS_ALPHA, update_freq=config.MCS_UPDATE_FREQ,
                 n_boot=config.MCS_N_BOOT, block_len=config.MCS_BLOCK_LEN) -> pd.DataFrame:
    """
    qlike_panel_one_coin: long df [date, model, QLIKE_daily] for one coin, full
    forecast universe (standalone + hybrids), sorted by date.

    Returns one row per date: [date, models_in_St, number_models, mcs_p_value].
    Family weight columns (Econ/AI/Hybrid) are added later in dmce.py once the
    within-set weighting scheme is chosen.
    """
    wide = qlike_panel_one_coin.pivot(index="date", columns="model", values="QLIKE_daily")
    wide = wide.sort_index()
    dates = wide.index.to_list()

    rng = np.random.default_rng(config.RANDOM_SEED)
    records = []
    last_key, cached_set, last_p = None, list(wide.columns), np.nan

    for t_i, today in enumerate(dates):
        if t_i < window:
            records.append({"date": today, "models_in_St": list(wide.columns),
                             "number_models": wide.shape[1], "mcs_p_value": np.nan})
            continue

        key = today.strftime("%Y-%m-%d") if update_freq == "D" else today.strftime("%Y-%W")
        if key != last_key:
            win = wide.iloc[t_i - window:t_i].dropna(axis=0, how="any")
            if win.shape[0] >= 10 and win.shape[1] >= 2:
                active_idx, p = _mcs_once(win.to_numpy(), alpha, n_boot, block_len, rng)
                cached_set = [win.columns[i] for i in active_idx]
                last_p = p
            else:
                cached_set = list(wide.columns)
                last_p = np.nan
            last_key = key

        records.append({"date": today, "models_in_St": list(cached_set),
                         "number_models": len(cached_set), "mcs_p_value": last_p})

    return pd.DataFrame(records)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n, m = 260, 6
    dates = pd.date_range("2020-01-01", periods=n)
    models = [f"model_{i}" for i in range(m)]
    losses = rng.normal(0, 1, size=(n, m))
    losses[:, 0] -= 0.3  # model_0 is genuinely better
    df = pd.DataFrame(losses, columns=models)
    df["date"] = dates
    long = df.melt(id_vars="date", var_name="model", value_name="QLIKE_daily")

    res = model_confidence_set(long)
    print("Static MCS:", res)

    roll = rolling_mcs(long, window=120, n_boot=100)
    print(roll.tail())
