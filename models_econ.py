"""
models_econ.py
===============
Framework points 18-22: standalone econometric / RV models.

Each `fit_*` function takes a training DataFrame (one coin, already containing
the HAR/jump feature columns from features.py) and returns a fitted object;
each `forecast_*` function takes that object plus the most recent row(s) and
returns a scalar RV forecast for t+1. All forecasts are floored at
config.RV_FLOOR to guarantee positivity (framework point 16).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

import config


def _floor(x):
    return max(float(x), config.RV_FLOOR)


# ---------------------------------------------------------------------------
# Point 18: Naive persistence benchmark
# ---------------------------------------------------------------------------
def forecast_naive(last_row: pd.Series) -> float:
    """RV_hat_{t+1} = RV_t"""
    return _floor(last_row["RV"])


# ---------------------------------------------------------------------------
# Point 19: HAR-RV
# ---------------------------------------------------------------------------
def fit_har_rv(train: pd.DataFrame):
    X = sm.add_constant(train[["RV_d", "RV_w", "RV_m"]])
    y = train["target_RV_next"]
    return sm.OLS(y, X, missing="drop").fit()


def forecast_har_rv(res, last_row: pd.Series) -> float:
    x = np.array([1.0, last_row["RV_d"], last_row["RV_w"], last_row["RV_m"]])
    return _floor(x @ res.params.to_numpy())


# ---------------------------------------------------------------------------
# Point 20: HAR-RV-J
# ---------------------------------------------------------------------------
def fit_har_rv_j(train: pd.DataFrame):
    X = sm.add_constant(train[["RV_d", "RV_w", "RV_m", "J"]])
    y = train["target_RV_next"]
    return sm.OLS(y, X, missing="drop").fit()


def forecast_har_rv_j(res, last_row: pd.Series) -> float:
    x = np.array([1.0, last_row["RV_d"], last_row["RV_w"], last_row["RV_m"], last_row["J"]])
    return _floor(x @ res.params.to_numpy())


# ---------------------------------------------------------------------------
# Point 21: HAR-RV-CJ
# ---------------------------------------------------------------------------
def fit_har_rv_cj(train: pd.DataFrame):
    cols = ["C", "C_w", "C_m", "J", "J_w", "J_m"]
    X = sm.add_constant(train[cols])
    y = train["target_RV_next"]
    return sm.OLS(y, X, missing="drop").fit()


def forecast_har_rv_cj(res, last_row: pd.Series) -> float:
    cols = ["C", "C_w", "C_m", "J", "J_w", "J_m"]
    x = np.array([1.0] + [last_row[c] for c in cols])
    return _floor(x @ res.params.to_numpy())


# ---------------------------------------------------------------------------
# Point 22: Realized GARCH (Hansen, Huang & Shek, 2012) - log-linear specification
#
#   r_t          = sqrt(h_t) z_t,           z_t ~ N(0,1)
#   log h_t      = omega + beta*log h_{t-1} + gamma*log x_{t-1}
#   log x_t      = xi + phi*log h_t + tau1*z_t + tau2*(z_t^2 - 1) + u_t,  u_t ~ N(0, sigma_u^2)
#
# with x_t = RV_t. Estimated by exact-form MLE (joint likelihood of returns and
# realized measures) via scipy.optimize, since this is not in statsmodels/arch
# off the shelf.
# ---------------------------------------------------------------------------
class RealizedGARCH:
    def __init__(self):
        self.params_ = None
        self._fitted_h = None
        self._fitted_x = None

    @staticmethod
    def _unpack(theta):
        omega, beta, gamma, xi, phi, tau1, tau2, log_sigma_u2 = theta
        sigma_u2 = np.exp(log_sigma_u2)
        return omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2

    def _negloglik(self, theta, r, x):
        omega, beta, gamma, xi, phi, tau1, tau2, sigma_u2 = self._unpack(theta)
        n = len(r)
        logx = np.log(np.maximum(x, config.RV_FLOOR))
        log_h = np.empty(n)
        log_h[0] = np.log(np.maximum(np.var(r) + 1e-12, config.RV_FLOOR))
        for t in range(1, n):
            log_h[t] = omega + beta * log_h[t - 1] + gamma * logx[t - 1]
        h = np.exp(log_h)
        z = r / np.sqrt(np.maximum(h, 1e-300))
        # return density: r_t = sqrt(h_t) z_t, z_t ~ N(0,1)
        ll_r = -0.5 * np.log(2 * np.pi * h) - 0.5 * (r ** 2) / h
        # measurement density: log x_t = xi + phi*log h_t + tau1 z_t + tau2(z_t^2-1) + u_t
        mean_logx = xi + phi * log_h + tau1 * z + tau2 * (z ** 2 - 1.0)
        resid = logx - mean_logx
        ll_x = -0.5 * np.log(2 * np.pi * sigma_u2) - 0.5 * (resid ** 2) / sigma_u2
        total = np.sum(ll_r) + np.sum(ll_x)
        if not np.isfinite(total):
            return 1e10
        return -total

    def fit(self, r: np.ndarray, x: np.ndarray):
        r = np.asarray(r, dtype=float)
        x = np.asarray(x, dtype=float)
        x0 = np.array([0.05, 0.85, 0.10, 0.0, 1.0, -0.05, 0.05, np.log(0.5)])
        res = minimize(self._negloglik, x0, args=(r, x), method="Nelder-Mead",
                        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
        self.params_ = res.x
        self._last_r, self._last_x = r, x
        return self

    def _log_h_path(self, r, x):
        omega, beta, gamma, *_ = self._unpack(self.params_)
        n = len(r)
        logx = np.log(np.maximum(x, config.RV_FLOOR))
        log_h = np.empty(n)
        log_h[0] = np.log(np.maximum(np.var(r) + 1e-12, config.RV_FLOOR))
        for t in range(1, n):
            log_h[t] = omega + beta * log_h[t - 1] + gamma * logx[t - 1]
        return log_h

    def forecast_next(self, r_hist: np.ndarray, x_hist: np.ndarray) -> float:
        """One-step-ahead forecast of h_{T+1} (used directly as RV forecast)."""
        omega, beta, gamma, *_ = self._unpack(self.params_)
        log_h = self._log_h_path(r_hist, x_hist)
        log_h_next = omega + beta * log_h[-1] + gamma * np.log(max(x_hist[-1], config.RV_FLOOR))
        return _floor(np.exp(log_h_next))


def fit_realized_garch(train: pd.DataFrame) -> RealizedGARCH:
    r = train["daily_return"].to_numpy()
    x = train["RV"].to_numpy()
    model = RealizedGARCH()
    model.fit(r, x)
    return model


def forecast_realized_garch(model: RealizedGARCH, train: pd.DataFrame) -> float:
    r = train["daily_return"].to_numpy()
    x = train["RV"].to_numpy()
    return model.forecast_next(r, x)
