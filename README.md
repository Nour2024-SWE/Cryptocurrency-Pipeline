# RV-Crypto: Regime-Aware Realized-Volatility Forecasting for Crypto Assets

A full research pipeline for forecasting daily realized variance (RV) of BTC,
ETH, LTC, and XRP using econometric (HAR, Realized GARCH), machine-learning
(RF, XGBoost, LightGBM), and deep-learning (LSTM, TCN) models — combined
through **dynamic pairwise hybrids** and a **Dynamic Model-Confidence
Ensemble (DMCE)** that screens the forecast universe with a rolling Model
Confidence Set at every point in time.

The pipeline evaluates every model across calendar-based crisis regimes
(COVID, China mining ban, Terra-Luna, FTX, ETF launch, October 2025
liquidation event, …) *and* data-driven Markov-switching volatility states,
and tracks whether forecasting authority shifts toward econometric, AI, or
hybrid models as market conditions change.

> Built to implement a specific empirical framework (crisis regimes →
> Markov-switching → standalone models → dynamic hybrids → rolling MCS →
> DMCE → regime evaluation → statistical testing → robustness checks). See
> [Methodology](#methodology--framework-map) for the full point-by-point map.

---

## Table of contents

- [Overview](#rv-crypto-regime-aware-realized-volatility-forecasting-for-crypto-assets)
- [Features](#features)
- [Installation](#installation)
- [Data](#data)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Methodology / framework map](#methodology--framework-map)
- [Configuration](#configuration)
- [Output files](#output-files)
- [Performance & runtime](#performance--runtime)
- [Known limitations](#known-limitations)
- [Suggested workflow](#suggested-workflow)
- [References](#references)
- [License](#license)

---

## Features

- **9 standalone forecasting models**: Naive persistence, HAR-RV, HAR-RV-J,
  HAR-RV-CJ, Realized GARCH (custom MLE implementation), Random Forest,
  XGBoost, LightGBM, LSTM, and TCN.
- **20 dynamic pairwise hybrids** (4 econometric × 5 AI models), each with a
  rolling QLIKE-minimizing combination weight re-estimated every day from a
  trailing window — no look-ahead.
- **Jump/continuous decomposition** (bipower variation) with automatic
  fallback to a documented proxy when only daily RV is available, and an
  exact intraday computation path once 5-minute data is supplied.
- **Two regime layers**: predefined calendar-based crisis windows and a
  2-state Markov-switching model fit on log realized variance (filtered +
  smoothed probabilities).
- **Rolling Model Confidence Set** (Hansen, Lunde & Nason, 2011 — implemented
  from scratch) screens the full 30-model forecast universe at every point in
  time; only the statistically competitive subset feeds the ensemble.
- **Dynamic Model-Confidence Ensemble (DMCE)** with a choice of inverse-QLIKE,
  equal, or softmax weighting inside the selected set, benchmarked against
  five alternative ensembles (best single model, equal-weight, static
  inverse-QLIKE, best hybrid).
- **Full evaluation suite**: QLIKE / RMSE / MAE / out-of-sample R², rolling
  losses, regime-conditional performance tables, model-family authority
  analysis (Econ vs AI vs Hybrid weight share over time), Diebold-Mariano
  tests, and a battery of robustness checks.
- Expanding-window, one-step-ahead forecasting with configurable retraining
  cadence per model family (daily for HAR, weekly for Realized GARCH,
  monthly for ML/DL — all adjustable).

## Installation

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd rv_pipeline
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
statsmodels>=0.14
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
arch>=6.0
torch>=2.0
tqdm>=4.65
matplotlib>=3.7
```

`torch` will automatically use a CUDA GPU if one is available (recommended
for the LSTM/TCN steps); a CPU-only install works fine, just slower.

## Data

Place your merged daily dataset at:

```
data/_Wide_format__MERGED_daily_RV_5m_BTC_ETH_XRP_LTC_common_sample.csv
```

Expected wide-format columns (per coin suffix, e.g. `BTCUSDT`):

```
date, RV_5m_{SYM}, LogRV_5m_{SYM}, daily_return_{SYM},
daily_volume_{SYM}, valid_5min_bars_{SYM}, valid_5min_returns_{SYM}
```

This is the output of the earlier data-collection / cleaning / daily-RV
construction stage (5-minute OHLCV → 5-minute log returns → daily RV/LogRV).
This repo picks up from there.

**Optional, for exact jump decomposition:** a long-format 5-minute file at
`data/clean_5min_long.csv` with columns `timestamp, coin, close` (or full
OHLCV). See [Known limitations](#known-limitations) below.

## Quick start

```bash
# 1. Sanity check: one coin, a 20-day window (~1-2 minutes)
python main.py --smoke --coin BTCUSDT

# 2. Full out-of-sample run for one coin (slow — see Performance & runtime)
python main.py --coin BTCUSDT --oos-start 2020-01-01 --oos-end 2026-06-20

# 3. Full run, all four coins
python main.py --full
```

Outputs are written to:

- `outputs/` — File 5–11 CSVs (standalone forecasts, hybrid forecasts,
  lambdas, QLIKE losses, DMCE forecasts, performance metrics)
- `tables/` — Tables 1–15 as CSV
- `figures/` — Figures 1–15 as PNG (call the `tables_figures.py` functions
  once you have full-sample DataFrames; not auto-triggered in `--smoke` mode)

## Project structure

```
rv_pipeline/
├── config.py             # all paths, dates, windows, hyperparameters
├── data_prep.py           # load merged wide CSV -> long panel
├── jumps.py                # bipower variation / jump / continuous components
├── regimes.py              # predefined crisis dummies
├── markov_switching.py     # 2-state Markov-switching on LogRV
├── descriptives.py         # summary stats, ADF/KPSS tests
├── features.py             # lagged RV/LogRV, HAR components, targets
├── models_econ.py          # Naive, HAR-RV, HAR-RV-J, HAR-RV-CJ, Realized GARCH
├── models_ml.py             # Random Forest, XGBoost, LightGBM
├── models_dl.py             # LSTM, TCN (PyTorch)
├── forecasting.py          # expanding-window orchestration -> File 5
├── hybrid.py                # dynamic pairwise hybrids -> Files 6, 7
├── losses.py                # QLIKE, RMSE, MAE, OOS R^2, rolling QLIKE -> File 8
├── mcs.py                    # rolling & final Model Confidence Set -> File 9
├── dmce.py                   # Dynamic Model-Confidence Ensemble -> File 10
├── regime_eval.py           # regime evaluation, family authority -> File 11
├── stat_tests.py             # final MCS, Diebold-Mariano tests -> File 12
├── robustness.py             # robustness checks
├── tables_figures.py         # Tables 1-15 / Figures 1-15
├── main.py                    # end-to-end orchestrator / CLI entry point
├── requirements.txt
├── data/                       # input data (not tracked — see .gitignore)
├── outputs/                    # generated forecast/loss CSVs
├── tables/                     # generated summary tables
└── figures/                    # generated figures
```

## Methodology / framework map

| Stage | Module(s) |
|---|---|
| Predefined crisis regimes | `regimes.py` |
| Markov-switching regime identification | `markov_switching.py` |
| Descriptive statistics & stationarity tests | `descriptives.py` |
| Lagged RV/LogRV, HAR components, forecasting target | `features.py` |
| Bipower variation / jump / continuous decomposition | `jumps.py` |
| Standalone models: Naive, HAR-RV, HAR-RV-J, HAR-RV-CJ, Realized GARCH | `models_econ.py` |
| Standalone models: Random Forest, XGBoost, LightGBM | `models_ml.py` |
| Standalone models: LSTM, TCN | `models_dl.py` |
| Expanding-window one-step-ahead forecasting | `forecasting.py` |
| Structured dynamic hybrids, rolling lambda estimation | `hybrid.py` |
| QLIKE / RMSE / MAE / OOS R², rolling loss averages | `losses.py` |
| Rolling & final Model Confidence Set | `mcs.py` |
| Dynamic Model-Confidence Ensemble, benchmark ensembles | `dmce.py` |
| Regime-conditional evaluation, family-authority analysis | `regime_eval.py` |
| Final MCS, Diebold-Mariano tests | `stat_tests.py` |
| Robustness checks | `robustness.py` |
| Tables & figures | `tables_figures.py` |
| Orchestration | `main.py` |

The forecasting design follows an expanding-window, one-step-ahead scheme:
initial training on 2018-01-01–2019-06-30, a validation window
(2019-07-01–2019-12-31) reserved for hyperparameter tuning, and an
out-of-sample test period from 2020-01-01 onward — so COVID and every later
crisis are evaluated purely out-of-sample.

## Configuration

All tunable parameters live in `config.py`, including:

- Sample / OOS date ranges
- HAR lag structure and rolling-window lengths
- ML/DL retraining cadence (`ML_RETRAIN_FREQ`, `DL_RETRAIN_FREQ`)
- Hybrid-lambda and rolling-QLIKE window lengths (with robustness alternates)
- MCS window, significance level, bootstrap replications, update frequency
- DMCE weighting scheme (`inverse_qlike`, `equal`, `softmax`)
- Predefined crisis regime date ranges

Edit these directly rather than passing long CLI argument lists.

## Output files

| File | Contents |
|---|---|
| `file5_standalone_forecasts.csv` | date, coin, actual_RV, and every standalone model's forecast |
| `file6_hybrid_forecasts.csv` | date, coin, actual_RV, and all 20 hybrid forecasts |
| `file7_hybrid_lambdas.csv` | date, coin, hybrid_pair, lambda_econ, weight_AI |
| `file8_qlike_losses.csv` | daily QLIKE per model plus rolling-window averages |
| `file9` (via `mcs.rolling_mcs`) | date, models_in_St, number_models, mcs_p_value |
| `file10_dmce_forecasts.csv` | date, coin, actual_RV, DMCE_forecast, models_in_St, weights, family weight shares |
| `file11_performance_metrics.csv` | QLIKE / RMSE / MAE / OOS R² / rank, by coin × period × model |
| `file12` (via `stat_tests.py`) | final MCS results and Diebold-Mariano test outcomes |

## Performance & runtime

A full run (≈2,300 out-of-sample trading days × 4 coins × a 30-model
forecast universe, with rolling hybrid-lambda, rolling-QLIKE, and rolling-MCS
computations at every step) is a genuinely large computation — budget for
**several hours on CPU**, less with a GPU for the LSTM/TCN steps.

To manage this:

- **Parallelize across coins.** Each coin is fully independent; run four
  separate processes/machines and concatenate the outputs.
- **Use a GPU** for `models_dl.py` (auto-detected via `torch.cuda`).
- **Relax retraining cadence** further in `config.py` if needed (e.g.
  quarterly instead of monthly for ML/DL) — this is an explicitly permitted
  practical compromise, just document it in your methodology write-up.

Default retraining cadence:

| Model family | Refit frequency |
|---|---|
| HAR-RV / HAR-RV-J / HAR-RV-CJ | Daily (closed-form OLS, cheap) |
| Realized GARCH | Weekly |
| Random Forest / XGBoost / LightGBM | Monthly |
| LSTM / TCN | Monthly |

Forecasts are produced **every day** regardless of retraining cadence, always
using only information available up to that day.

## Known limitations

1. **Jump/continuous decomposition is a proxy unless you supply 5-minute
   data.** True bipower variation requires intraday returns
   (`BV_t = μ₁⁻² Σ |r_i||r_{i-1}|`); the merged daily file only contains
   already-aggregated RV/LogRV. `jumps.py` automatically uses the exact
   formula if `data/clean_5min_long.csv` exists, otherwise falls back to a
   documented daily-RV-only jump proxy and flags every affected row with
   `jump_is_proxy=True`. HAR-RV-J, HAR-RV-CJ, and all jump-derived features
   inherit this limitation until intraday data is supplied.
2. **Custom MCS implementation.** `mcs.py` implements the Hansen, Lunde &
   Nason (2011) range-statistic elimination algorithm with a stationary
   bootstrap, written from scratch (no maintained Python package implements
   it). It uses a simplified, non-studentized range statistic — adequate for
   model screening, but note this if you want to claim exact replication of
   the original paper's test battery.
3. **Custom Realized GARCH.** `models_econ.RealizedGARCH` implements the
   Hansen, Huang & Shek (2012) log-linear specification via a from-scratch
   maximum-likelihood estimator (Nelder-Mead optimization), since no
   off-the-shelf package covers it.
4. **Figures are opt-in.** `tables_figures.py` exposes each figure as a
   callable function rather than auto-generating on every run, to keep quick
   sanity checks fast — wire these into `main.py` once you're ready for a
   full pass.

## Suggested workflow

1. Run `python main.py --smoke` to confirm the full chain executes.
2. If you have 5-minute intraday data, place it at
   `data/clean_5min_long.csv` and re-run `jumps.py` standalone to confirm
   `jump_is_proxy` is `False` everywhere.
3. Set retraining cadences in `config.py` to match your compute budget.
4. Run `--full` per coin (parallelized if possible) and concatenate outputs.
5. Feed the concatenated File 5/6/8/10/11 outputs into `tables_figures.py`
   and `stat_tests.py` to produce the final tables, figures, and DM tests.
6. Run the checks in `robustness.py` (rolling windows, MCS update frequency,
   DMCE weighting scheme, predefined vs. Markov-switching regimes, 2026
   stress-window confirmation, appendix GARCH variants) and assemble the
   robustness table.

## References

- Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003).
  Modeling and forecasting realized volatility. *Econometrica*, 71(2).
- Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. *Journal of Financial Econometrics*, 7(2).
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and bipower
  variation with stochastic volatility and jumps. *Journal of Financial
  Econometrics*, 2(1).
- Hansen, P. R., Huang, Z., & Shek, H. H. (2012). Realized GARCH: a joint
  model for returns and realized measures of volatility. *Journal of Applied
  Econometrics*, 27(6).
- Hamilton, J. D. (1989). A new approach to the economic analysis of
  nonstationary time series and the business cycle. *Econometrica*, 57(2).
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
  *Econometrica*, 79(2).
- Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies. *Journal of Econometrics*, 160(1).
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
  *Journal of Business & Economic Statistics*, 13(3).

## License

Add your preferred license here (e.g. MIT, Apache-2.0). No license is
currently specified.
