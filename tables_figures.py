"""
tables_figures.py
==================
Framework points 49-50: writes Tables 1-15 (as available from upstream outputs)
to config.TABLE_DIR as CSV, and produces Figures 1-15 as PNG in config.FIGURE_DIR.

This module is intentionally a thin "save what you have" layer: call the
individual `save_table` / `fig*` functions from main.py once the corresponding
upstream DataFrame exists.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def save_table(df: pd.DataFrame, name: str):
    path = config.TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


# --- Figures -----------------------------------------------------------
def fig1_price_series(daily_with_price: pd.DataFrame):
    """Requires a `price` or reconstructed-from-return series; if unavailable
    (this pipeline's daily file has returns, not levels), plots cumulative
    daily_return instead and labels it clearly."""
    fig, axes = plt.subplots(len(config.COINS), 1, figsize=(10, 2.2 * len(config.COINS)), sharex=True)
    for ax, c in zip(axes, config.COINS):
        g = daily_with_price[daily_with_price.coin == c].sort_values("date")
        cum_price = (1 + g["daily_return"]).cumprod()
        ax.plot(g["date"], cum_price)
        ax.set_title(f"{config.COIN_DISPLAY[c]} cumulative price index (from daily_return)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "figure1_price_series.png", dpi=150)
    plt.close(fig)


def fig2_3_rv_logrv_series(daily: pd.DataFrame):
    for var, fname, title in [("RV", "figure2_rv_series.png", "Realized Variance"),
                               ("LogRV", "figure3_logrv_series.png", "Log Realized Variance")]:
        fig, axes = plt.subplots(len(config.COINS), 1, figsize=(10, 2.2 * len(config.COINS)), sharex=True)
        for ax, c in zip(axes, config.COINS):
            g = daily[daily.coin == c].sort_values("date")
            ax.plot(g["date"], g[var])
            ax.set_title(f"{config.COIN_DISPLAY[c]} {title}")
        fig.tight_layout()
        fig.savefig(config.FIGURE_DIR / fname, dpi=150)
        plt.close(fig)


def fig4_rv_with_crises(daily_with_dummies: pd.DataFrame, coin: str = "BTCUSDT"):
    g = daily_with_dummies[daily_with_dummies.coin == coin].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(g["date"], g["RV"], lw=0.8)
    for name, (start, end) in config.CRISIS_REGIMES.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="red", alpha=0.15)
    ax.set_title(f"{config.COIN_DISPLAY[coin]} RV with predefined crisis windows shaded")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "figure4_rv_crises_shaded.png", dpi=150)
    plt.close(fig)


def fig5_6_markov_prob(ms_table: pd.DataFrame, coin: str = "BTCUSDT"):
    g = ms_table[ms_table.coin == coin].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(g["date"], g["P_high_filtered"])
    ax.set_title(f"{config.COIN_DISPLAY[coin]} filtered high-volatility probability")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "figure5_ms_high_prob.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(g["date"], g["LogRV"], lw=0.8)
    high = g[g["MSHigh"] == 1]
    ax.scatter(high["date"], high["LogRV"], color="red", s=4, alpha=0.5)
    ax.set_title(f"{config.COIN_DISPLAY[coin]} LogRV with Markov high-volatility periods marked")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "figure6_logrv_ms_high_shaded.png", dpi=150)
    plt.close(fig)


def fig7_8_qlike_comparison(perf_table: pd.DataFrame, hybrid: bool = False):
    models = config.HYBRID_PAIRS if hybrid else config.STANDALONE_MODELS
    model_names = [f"hybrid_{e}_{a}" for e, a in models] if hybrid else models
    sub = perf_table[(perf_table.Period == "Full_OOS") & (perf_table.Model.isin(model_names))]
    fig, ax = plt.subplots(figsize=(10, 4))
    piv = sub.pivot(index="Model", columns="Coin", values="QLIKE")
    piv.plot(kind="bar", ax=ax)
    ax.set_title("QLIKE comparison" + (" (hybrids)" if hybrid else " (standalone)"))
    fig.tight_layout()
    fname = "figure8_qlike_hybrid.png" if hybrid else "figure7_qlike_standalone.png"
    fig.savefig(config.FIGURE_DIR / fname, dpi=150)
    plt.close(fig)


def fig9_dmce_vs_actual(dmce_df: pd.DataFrame, coin: str):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(dmce_df["date"], dmce_df["actual_RV"], label="actual RV", lw=0.8)
    ax.plot(dmce_df["date"], dmce_df["DMCE_forecast"], label="DMCE forecast", lw=0.8)
    ax.legend()
    ax.set_title(f"{config.COIN_DISPLAY.get(coin, coin)} DMCE forecast vs actual RV")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "figure9_dmce_vs_actual.png", dpi=150)
    plt.close(fig)


def fig10_num_models_in_st(dmce_df: pd.DataFrame, coin: str):
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(dmce_df["date"], dmce_df["models_in_St"].apply(len))
    ax.set_title(f"{config.COIN_DISPLAY.get(coin, coin)} |S_t| over time")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "figure10_num_models_st.png", dpi=150)
    plt.close(fig)


def fig12_13_family_weights(dmce_df: pd.DataFrame, coin: str, crisis_flags: pd.DataFrame = None):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.stackplot(dmce_df["date"], dmce_df["EconWeight"], dmce_df["AIWeight"],
                 dmce_df["HybridWeight"], labels=["Econ", "AI", "Hybrid"])
    ax.legend(loc="upper left")
    ax.set_title(f"{config.COIN_DISPLAY.get(coin, coin)} Econ/AI/Hybrid weight shares over time")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "figure12_family_weights.png", dpi=150)
    plt.close(fig)

    if crisis_flags is not None:
        merged = dmce_df.merge(crisis_flags[crisis_flags.coin == coin], on="date", how="left")
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.stackplot(merged["date"], merged["EconWeight"], merged["AIWeight"],
                     merged["HybridWeight"], labels=["Econ", "AI", "Hybrid"])
        for name, (start, end) in config.CRISIS_REGIMES.items():
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="black", alpha=0.1)
        ax.legend(loc="upper left")
        ax.set_title(f"{config.COIN_DISPLAY.get(coin, coin)} family weights with crisis shading")
        fig.tight_layout()
        fig.savefig(config.FIGURE_DIR / "figure13_family_weights_crisis.png", dpi=150)
        plt.close(fig)
