"""
markov_switching.py
====================
Framework point 10: Markov-Switching regime identification.

    LogRV_t = mu_St + phi_St * LogRV_{t-1} + eps_t,  eps_t ~ N(0, sigma_St^2),  St in {1,2}

Implemented with statsmodels' MarkovAutoregression (switching mean, AR(1),
switching variance). Fit separately per coin (the framework specifies the model
per-coin: "date coin LogRV P_high_filtered P_high_smoothed MSHigh").
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

import config


def fit_markov_switching(logrv: pd.Series) -> MarkovAutoregression:
    """Fit a 2-state Markov-switching AR(1) model with switching mean and variance."""
    y = logrv.dropna().to_numpy()
    mod = MarkovAutoregression(
        y, k_regimes=2, order=1,
        switching_ar=False,      # phi_St: set True to also switch the AR coefficient
        switching_variance=True,
    )
    res = mod.fit(search_reps=20)
    return mod, res


def _high_state_index(mod, res) -> int:
    """
    Identify which of the two regimes is the "high-volatility" state.
    Framework point 10 defines Regime 1/2 as low-/high-volatility states, so we
    rank by the switching variance sigma2_St (not the mean), since that is what
    separates calm vs turbulent LogRV dynamics.
    """
    names = list(mod.param_names)
    sigma2 = [res.params[names.index(f"sigma2[{i}]")] for i in range(2)]
    return int(np.argmax(sigma2))


def markov_regime_table(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Markov-switching per coin on LogRV and returns
    [date, coin, LogRV, P_high_filtered, P_high_smoothed, MSHigh, MSLow]
    (Framework point 10 required output).
    """
    out_frames = []
    for c, g in daily.groupby("coin"):
        g = g.sort_values("date").reset_index(drop=True)
        mod, res = fit_markov_switching(g["LogRV"])
        high_idx = _high_state_index(mod, res)

        # order=1 -> filtered/smoothed marginal probabilities are indexed from t=1
        filt_arr = np.asarray(res.filtered_marginal_probabilities)
        smooth_arr = np.asarray(res.smoothed_marginal_probabilities)
        filt = filt_arr[:, high_idx]
        smooth = smooth_arr[:, high_idx]

        n = len(g)
        pad = n - len(filt)  # first `pad` obs lost to the AR(1) lag
        p_high_filtered = np.concatenate([np.full(pad, np.nan), filt])
        p_high_smoothed = np.concatenate([np.full(pad, np.nan), smooth])

        sub = pd.DataFrame({
            "date": g["date"], "coin": c, "LogRV": g["LogRV"],
            "P_high_filtered": p_high_filtered,
            "P_high_smoothed": p_high_smoothed,
        })
        sub["MSHigh"] = (sub["P_high_filtered"] > 0.5).astype("Int64")
        sub.loc[sub["P_high_filtered"].isna(), "MSHigh"] = pd.NA
        sub["MSLow"] = 1 - sub["MSHigh"]
        out_frames.append(sub)

    return pd.concat(out_frames, ignore_index=True)


if __name__ == "__main__":
    from data_prep import load_daily_long
    d = load_daily_long()
    ms = markov_regime_table(d[d.coin == "BTCUSDT"])
    print(ms.head(10))
    print(ms["MSHigh"].value_counts(dropna=False))
