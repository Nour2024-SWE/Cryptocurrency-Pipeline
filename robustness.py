"""
robustness.py
==============
Framework point 47: robustness checks. This module doesn't re-implement the
whole pipeline; it exposes the specific knobs to vary and small standalone
utilities for the checks that don't fit naturally elsewhere, collecting
comparable summary rows (Table 15).

Robustness axes implemented here:
  1. Target: RV_{t+1} vs LogRV_{t+1} (see config + models_ml/models_dl, which
     already train on LogRV and floor-exponentiate; for a pure levels-target
     robustness run, refit HAR models with target_RV_next unchanged -- this is
     the default -- vs an all-Log target variant, toggle in main.py).
  2. Rolling window W in {90, 180, 365} for hybrid lambda estimation, rolling
     QLIKE, and MCS (config.HYBRID_LAMBDA_WINDOWS_ROBUST /
     ROLL_QLIKE_WINDOWS_ROBUST -- MCS window varied manually below).
  3. Daily vs weekly rolling MCS updates (config.MCS_UPDATE_FREQ).
  4. DMCE weighting scheme: inverse-QLIKE vs equal vs softmax(eta) (dmce.py).
  5. Predefined crises vs Markov-Switching regimes (already both computed by
     regimes.py / markov_switching.py; compare regime_eval.py outputs run with
     each regime definition).
  6. Exclude the 2026 candidate stress period if RV does not confirm elevated
     stress (see `confirm_2026_stress`).
  7. Appendix-only GARCH/EGARCH/GJR-GARCH benchmarks (see
     `fit_appendix_garch_variants`, requires the `arch` package).
"""
import numpy as np
import pandas as pd
from arch import arch_model

import config


def confirm_2026_stress(daily: pd.DataFrame, threshold_pctile: float = 75) -> dict:
    """
    Point 47 robustness 6: empirically check whether the 2026 candidate stress
    window actually shows elevated RV relative to the rest of the sample.
    """
    start, end = config.CRISIS_REGIMES["Correction2026"]
    out = {}
    for c, g in daily.groupby("coin"):
        window_rv = g.loc[(g.date >= start) & (g.date <= end), "RV"]
        other_rv = g.loc[(g.date < start) | (g.date > end), "RV"]
        if len(window_rv) == 0:
            continue
        pctile_cut = np.percentile(other_rv.dropna(), threshold_pctile)
        confirmed = window_rv.mean() > other_rv.mean()
        out[c] = {
            "window_mean_RV": window_rv.mean(), "rest_mean_RV": other_rv.mean(),
            f"rest_p{int(threshold_pctile)}": pctile_cut,
            "share_days_above_p75": float((window_rv > pctile_cut).mean()),
            "confirmed_elevated": bool(confirmed),
        }
    return out


def fit_appendix_garch_variants(returns_pct: np.ndarray) -> pd.DataFrame:
    """
    Point 47 robustness 7 (appendix only): standard GARCH(1,1), EGARCH(1,1),
    GJR-GARCH(1,1) on daily returns, using the `arch` package. Returns a
    one-row-per-model summary of fitted parameters and in-sample log-likelihood;
    these are NOT part of the main forecast universe (they forecast latent
    conditional variance, not realized variance -- framework point 22 note).
    """
    rows = []
    specs = {
        "GARCH": dict(vol="GARCH", p=1, q=1),
        "EGARCH": dict(vol="EGARCH", p=1, q=1),
        "GJR_GARCH": dict(vol="GARCH", p=1, o=1, q=1),  # o=1 -> GJR asymmetry term
    }
    for name, kwargs in specs.items():
        am = arch_model(returns_pct, mean="Zero", **kwargs, rescale=False)
        res = am.fit(disp="off")
        rows.append({"Model": name, "LogLik": res.loglikelihood, "AIC": res.aic,
                     "BIC": res.bic, "params": res.params.to_dict()})
    return pd.DataFrame(rows)


ROBUSTNESS_GRID = {
    "target": ["RV_level", "LogRV_then_exp"],
    "rolling_window": config.ROLL_QLIKE_WINDOWS_ROBUST,
    "mcs_update_freq": ["D", "W"],
    "dmce_weight_scheme": ["inverse_qlike", "equal", "softmax_eta1", "softmax_eta5", "softmax_eta10"],
    "regime_definition": ["predefined", "markov_switching"],
}


if __name__ == "__main__":
    from data_prep import load_daily_long
    d = load_daily_long()
    print(confirm_2026_stress(d))
