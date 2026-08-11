# RV-Crypto Forecasting Pipeline — Framework points 9–51

This package implements the RV-Crypto Framework from point **9 (predefined
crisis regimes)** through point **51 (final logic / write-up)**. Points 1–8
(data collection through daily RV/LogRV construction) are assumed already
done — this pipeline consumes the merged daily file you uploaded:

```
data/_Wide_format__MERGED_daily_RV_5m_BTC_ETH_XRP_LTC_common_sample.csv
```

## Install

```bash
pip install pandas numpy scipy statsmodels scikit-learn xgboost lightgbm arch torch tqdm matplotlib
```

(`torch` CPU-only is fine; it auto-detects CUDA if you have a GPU — much
faster for the LSTM/TCN steps.)

## Quick start

```bash
# 1. Sanity check on a 20-day window, one coin (~1-2 min)
python main.py --smoke --coin BTCUSDT

# 2. Full run for one coin over the full OOS window (slow, see warning below)
python main.py --coin BTCUSDT --oos-start 2020-01-01 --oos-end 2026-06-20

# 3. Full run, all four coins (run each coin on a separate machine/process if possible)
python main.py --full
```

Outputs land in `outputs/` (Files 5–11), `tables/` (Tables 1–15, as produced),
and `figures/` (Figures 1–15, via `tables_figures.py` — wire these calls into
`main.py` once you're happy with a full run; they're not auto-triggered by
`--smoke` to keep the sanity check fast).

## Module map (which file implements which framework point)

| Points | Module |
|---|---|
| 9  | `regimes.py` |
| 10 | `markov_switching.py` |
| 11 | `descriptives.py` |
| 13–16 | `features.py` |
| 15 (BV/J/C) | `jumps.py` |
| 17–22 (Naive, HAR family, Realized GARCH) | `models_econ.py` |
| 23–25 (RF, XGBoost, LightGBM) | `models_ml.py` |
| 26–27 (LSTM, TCN) | `models_dl.py` |
| 28–30 (expanding window → File 5) | `forecasting.py` |
| 31–34 (hybrids, dynamic lambda → Files 6, 7) | `hybrid.py` |
| 35–37 (QLIKE/RMSE/MAE/OOS R², rolling QLIKE → File 8) | `losses.py` |
| 38, 45 (rolling & final MCS) | `mcs.py` |
| 39–41 (DMCE, benchmark ensembles → File 10) | `dmce.py` |
| 42–44 (regime evaluation, family authority, set composition → File 11, Tables 9–12) | `regime_eval.py` |
| 45–46 (final MCS, DM tests → File 12, Tables 13–14) | `stat_tests.py` |
| 47 (robustness checks → Table 15) | `robustness.py` |
| 48–50 (output files/tables/figures) | `main.py`, `tables_figures.py` |
| Orchestration | `main.py` |

## Important caveats — please read before publishing results

1. **Jump/BV component (point 15) is a proxy, not the exact formula, unless
   you supply 5-minute data.** Your uploaded CSV only has daily RV/LogRV —
   the intraday 5-minute returns needed for true Bipower Variation
   (`BV_t = mu1^-2 * sum |r_i||r_{i-1}|`) aren't in it. `jumps.py` will:
   - use the exact formula automatically **if** you place a long-format
     5-minute file at `data/clean_5min_long.csv` with columns
     `[timestamp, coin, close]` (or full OHLCV) and point
     `config.FIVE_MIN_LONG_CSV` at it;
   - otherwise fall back to a **daily-RV-only jump proxy** (a Lee-Mykland-style
     standardized-return threshold rule), and every row is flagged
     `jump_is_proxy=True` so you can identify and, later, replace this in
     any output. **HAR-RV-J, HAR-RV-CJ, and all jump-related features are
     only as good as this proxy until you supply 5-minute data.**

2. **Retraining cadence deviates from "daily expanding window" for the
   computationally expensive models**, which the framework's point 28
   explicitly allows as a "practical rule" for DL and which this pipeline
   extends to the tree models and Realized GARCH for tractability:
   - HAR-RV / HAR-RV-J / HAR-RV-CJ: refit **daily** (closed-form OLS, cheap).
   - Realized GARCH: refit **weekly** (`forecasting.RGARCH_RETRAIN_FREQ`).
   - RF / XGBoost / LightGBM: refit **monthly** (`config.ML_RETRAIN_FREQ`).
   - LSTM / TCN: refit **monthly** (`config.DL_RETRAIN_FREQ`).
   Change these constants (e.g. to `"D"`) if you have the compute budget for
   true daily retraining — be aware this multiplies runtime by roughly the
   ratio of trading days to your current cadence.

3. **Full-scale runtime.** A full run (2020-01-01 → 2026-06-20, ≈2,300
   trading days, × 4 coins, × 30-model forecast universe, × rolling
   hybrid-lambda / rolling-QLIKE / rolling-MCS computations) is a genuinely
   large computation. Budget for **many hours** on a single CPU machine, less
   with a GPU for the DL steps and/or by parallelizing coins across
   processes/machines (they are fully independent — see `main.py`'s
   docstring).

4. **MCS implementation.** `mcs.py` implements the Hansen–Lunde–Nason (2011)
   range-statistic Model Confidence Set with a stationary bootstrap, written
   from scratch (no off-the-shelf MCS package was available in the build
   environment). It is a faithful implementation of the elimination
   algorithm but uses a simplified (non-studentized) range statistic rather
   than the full T_max/T_R battery in the original paper — adequate for
   model screening, but say so explicitly in your methodology section if you
   want to claim exact replication of Hansen et al.'s procedure.

5. **Realized GARCH** (`models_econ.RealizedGARCH`) is a custom maximum
   -likelihood implementation of the Hansen–Huang–Shek (2012) log-linear
   specification (no off-the-shelf Python package implements it). It uses
   Nelder–Mead optimization, which is robust but not the fastest — this is
   why it's refit weekly rather than daily by default.

6. **Figures are provided as callable functions**, not auto-generated in
   `--smoke` mode (to keep the sanity check fast). Call them from `main.py`
   (or interactively) once you have full-sample DataFrames — see
   `tables_figures.py` docstring for the function-to-figure mapping.

## Suggested workflow

1. Run `--smoke` to confirm the whole chain executes on your machine.
2. If you have 5-minute intraday data, drop it at
   `data/clean_5min_long.csv` and re-run `jumps.py` standalone to confirm
   `jump_is_proxy` is `False` everywhere before doing anything else.
3. Decide on retraining cadences given your compute budget (see caveat 2)
   and set them in `config.py`.
4. Run `--full` per coin (parallelized if possible), concatenate outputs.
5. Feed the concatenated File 5/6/8/10/11 outputs into `tables_figures.py`
   and `stat_tests.py` for the final Tables 1–15 / Figures 1–15 / DM tests.
6. Run `robustness.py`'s checks (rolling windows, MCS update frequency, DMCE
   weighting scheme, predefined-vs-Markov regimes, 2026-window confirmation,
   appendix GARCH variants) and assemble Table 15.
