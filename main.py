"""
main.py
========
End-to-end orchestrator for Framework points 9-51.

*** PERFORMANCE WARNING ***
Running end-to-end over the full 2020-01-01..2026-06-20 OOS window x 4 coins x
9 standalone models (with LSTM/TCN retrained monthly, RF/XGB/LGBM retrained
monthly, HAR family refit daily, Realized GARCH refit weekly) will take a long
time on a single machine (many hours, and much longer on CPU-only for the DL
models). This is expected given the scope of the framework. Recommended ways
to run it in practice:

  1. Smoke-test first (default below): a short OOS window, to confirm the
     whole chain runs on your machine before committing to the full run.
  2. Parallelize across coins (they are fully independent) -- e.g. 4 separate
     processes/machines, one per coin, then concatenate File 5/6 outputs.
  3. Use a GPU for the LSTM/TCN steps (models_dl.py auto-detects CUDA).
  4. If a full daily-retrain schedule for every model is not tractable, relax
     ML_RETRAIN_FREQ / DL_RETRAIN_FREQ / RGARCH_RETRAIN_FREQ further (e.g.
     quarterly) -- document the deviation in your methodology section
     (framework point 28 explicitly allows this practical compromise for DL).

Usage:
    python main.py --smoke         # short window, all coins, sanity check
    python main.py --full          # full framework run (see warning above)
    python main.py --coin BTCUSDT --oos-start 2020-01-01 --oos-end 2020-06-30
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import config
from data_prep import load_daily_long, coverage_table
from jumps import attach_jump_components
from regimes import attach_crisis_dummies, crisis_definition_table
from markov_switching import markov_regime_table
from descriptives import descriptive_table, descriptive_by_regime, stationarity_tests
from features import build_features
from forecasting import run_expanding_window
from hybrid import build_dynamic_hybrids
from losses import build_qlike_panel, add_rolling_qlike, performance_table
from mcs import rolling_mcs
from dmce import build_dmce
from regime_eval import evaluate_all_regimes, family_weight_shares, model_set_composition
from stat_tests import run_final_mcs, run_dm_comparisons, diebold_mariano
from robustness import confirm_2026_stress

warnings.filterwarnings("ignore")


def build_daily_dataset() -> pd.DataFrame:
    """Points 9-16: full daily feature dataset with regimes and MS states merged in."""
    d = load_daily_long()
    d = attach_jump_components(d)
    d = attach_crisis_dummies(d)
    feat = build_features(d)

    ms = markov_regime_table(d)
    feat = feat.merge(ms[["date", "coin", "P_high_filtered", "P_high_smoothed", "MSHigh", "MSLow"]],
                       on=["date", "coin"], how="left")
    return feat


def run_descriptives(daily: pd.DataFrame):
    tbl2 = descriptive_table(daily)
    tbl2.to_csv(config.TABLE_DIR / "table2_descriptives.csv", index=False)
    crisis_definition_table().to_csv(config.TABLE_DIR / "table3_crisis_definitions.csv", index=False)
    descriptive_by_regime(daily, "Crisis").to_csv(config.TABLE_DIR / "table_descriptives_by_crisis.csv", index=False)
    if "MSHigh" in daily.columns:
        descriptive_by_regime(daily.dropna(subset=["MSHigh"]), "MSHigh").to_csv(
            config.TABLE_DIR / "table4_descriptives_by_ms.csv", index=False)
    stationarity_tests(daily).to_csv(config.TABLE_DIR / "table_stationarity.csv", index=False)
    coverage_table(daily).to_csv(config.TABLE_DIR / "table1_coverage.csv", index=False)


def run_pipeline_for_coin(feat_all: pd.DataFrame, coin: str, oos_start: str, oos_end: str,
                           verbose: bool = True) -> dict:
    sub = feat_all[feat_all.coin == coin]

    # Points 28-30 -> File 5
    standalone = run_expanding_window(sub, coin, oos_start=oos_start, oos_end=oos_end, verbose=verbose)

    # Points 31-34 -> Files 6, 7
    hyb_df, lam_df = build_dynamic_hybrids(standalone)

    all_models = config.STANDALONE_MODELS + [f"hybrid_{e}_{a}" for e, a in config.HYBRID_PAIRS]
    forecasts_wide = standalone.merge(
        hyb_df.drop(columns=["actual_RV"]), on="date", how="inner")
    forecasts_wide["coin"] = coin

    # Points 35-37 -> File 8
    qlike_panel = build_qlike_panel(forecasts_wide, all_models, id_cols=("date", "coin"))
    qlike_panel = add_rolling_qlike(qlike_panel)

    # Point 38 -> File 9 (S_t)
    st_table = rolling_mcs(qlike_panel[["date", "model", "QLIKE_daily"]])

    # Points 39-41 -> File 10
    dmce_df = build_dmce(forecasts_wide, st_table, qlike_panel, roll_window=config.ROLL_QLIKE_WINDOW)
    dmce_df["coin"] = coin

    return {
        "standalone": standalone, "hybrid": hyb_df, "lambdas": lam_df,
        "qlike_panel": qlike_panel, "st_table": st_table, "dmce": dmce_df,
        "forecasts_wide": forecasts_wide, "all_models": all_models,
    }


def run_full(coins, oos_start, oos_end, verbose=True):
    print("Building daily dataset (points 9-16)...")
    feat_all = build_daily_dataset()
    run_descriptives(feat_all)

    regime_flags_cols = ["date", "coin", "Crisis", "Calm", "MSHigh", "MSLow"] + list(config.CRISIS_REGIMES)
    regime_flags = feat_all[regime_flags_cols].drop_duplicates()

    all_standalone, all_hybrid, all_dmce, all_qlike = [], [], [], []

    for coin in coins:
        print(f"\n=== Running coin {coin} ===")
        res = run_pipeline_for_coin(feat_all, coin, oos_start, oos_end, verbose=verbose)
        all_standalone.append(res["standalone"])
        h = res["hybrid"].copy(); h["coin"] = coin
        all_hybrid.append(h)
        all_dmce.append(res["dmce"])
        q = res["qlike_panel"].copy(); q["coin"] = coin
        all_qlike.append(q)

        perf = evaluate_all_regimes(res["forecasts_wide"], res["all_models"], regime_flags)
        fam = family_weight_shares(res["dmce"], regime_flags, coin)
        fam.to_csv(config.TABLE_DIR / f"table12_family_weights_{coin}.csv", index=False)
        comp = model_set_composition(res["dmce"], regime_flags, coin, res["all_models"])
        comp.to_csv(config.TABLE_DIR / f"table11_model_inclusion_{coin}.csv", index=False)

    standalone_all = pd.concat(all_standalone, ignore_index=True)
    standalone_all.to_csv(config.OUTPUT_DIR / "file5_standalone_forecasts.csv", index=False)
    hybrid_all = pd.concat(all_hybrid, ignore_index=True)
    hybrid_all.to_csv(config.OUTPUT_DIR / "file6_hybrid_forecasts.csv", index=False)
    dmce_all = pd.concat(all_dmce, ignore_index=True)
    dmce_all.to_csv(config.OUTPUT_DIR / "file10_dmce_forecasts.csv", index=False)

    print("\nRobustness check 6 (2026 correction confirmation):")
    print(confirm_2026_stress(feat_all))

    print("\nDone. See:", config.OUTPUT_DIR, config.TABLE_DIR, config.FIGURE_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="short window, quick sanity check")
    ap.add_argument("--full", action="store_true", help="full framework run (slow, see warning)")
    ap.add_argument("--coin", default=None, help="restrict to a single coin")
    ap.add_argument("--oos-start", default=config.OOS_START)
    ap.add_argument("--oos-end", default=config.OOS_END)
    args = ap.parse_args()

    coins = [args.coin] if args.coin else config.COINS

    if args.smoke or not args.full:
        run_full(coins, oos_start="2020-01-01", oos_end="2020-01-20", verbose=True)
    else:
        run_full(coins, oos_start=args.oos_start, oos_end=args.oos_end, verbose=True)
