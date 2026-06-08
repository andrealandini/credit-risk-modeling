"""SDE calibration for macro factors.

Ornstein-Uhlenbeck for GDP growth, unemployment gap, spread, and HICP.
CIR for non-negative rate processes (policy rate, euribor, AAA yield).

Discretised forms (Δt = 1 quarter):
  OU:  x_{t+1} = α + β x_t + ε,   κ = -ln(β),   θ = α/(1-β),   σ = sd(ε)/sqrt((1-β²)/(2κ)·...)
  CIR (pseudo-MLE via OLS on √-weighted form): same drift, σ √x diffusion.

Returns a dict per factor with κ, θ, σ and the residual series for the
correlation matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DT = 0.25  # one quarter expressed in years


@dataclass
class SDEParams:
    name: str
    process: str       # "OU" or "CIR"
    kappa: float
    theta: float
    sigma: float
    x0: float
    residuals: pd.Series


def calibrate_ou(series: pd.Series, name: str) -> SDEParams:
    s = series.dropna()
    x_t = s.shift(1).dropna()
    x_tp1 = s.loc[x_t.index]
    X = np.column_stack([np.ones(len(x_t)), x_t.values])
    y = x_tp1.values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    beta = min(max(beta, 1e-6), 0.9999)
    kappa = -np.log(beta) / DT
    theta = alpha / (1 - beta)
    resid = y - X @ coef
    var_eps = resid.var(ddof=2)
    sigma = float(np.sqrt(var_eps * 2 * kappa / (1 - beta ** 2)))
    return SDEParams(name=name, process="OU", kappa=float(kappa), theta=float(theta),
                     sigma=sigma, x0=float(s.iloc[-1]),
                     residuals=pd.Series(resid, index=x_tp1.index))


def calibrate_cir(series: pd.Series, name: str) -> SDEParams:
    s = series.dropna().clip(lower=1e-4)
    x_t = s.shift(1).dropna()
    x_tp1 = s.loc[x_t.index]
    w = 1.0 / np.sqrt(x_t.values)
    X = np.column_stack([np.ones(len(x_t)), x_t.values]) * w[:, None]
    y = x_tp1.values * w
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    beta = min(max(beta, 1e-6), 0.9999)
    kappa = -np.log(beta) / DT
    theta = alpha / (1 - beta)
    resid_full = x_tp1.values - (alpha + beta * x_t.values)
    sigma = float(np.sqrt(np.mean(resid_full ** 2 / x_t.values) / DT))
    return SDEParams(name=name, process="CIR", kappa=float(kappa), theta=float(theta),
                     sigma=sigma, x0=float(s.iloc[-1]),
                     residuals=pd.Series(resid_full / np.sqrt(x_t.values),
                                          index=x_tp1.index))


PROCESS_MAP = {
    "gdp_growth":   "OU",
    "unemployment": "OU",
    "spread":       "OU",
    "hicp":         "OU",
    "policy_rate":  "CIR",
    "euribor3m":    "CIR",
    "yield_aaa10y": "CIR",
}


def calibrate_all(panel: pd.DataFrame) -> tuple[dict[str, SDEParams], np.ndarray, list[str]]:
    params: dict[str, SDEParams] = {}
    for col, proc in PROCESS_MAP.items():
        if col not in panel.columns:
            continue
        s = panel[col]
        if proc == "OU":
            params[col] = calibrate_ou(s, col)
        else:
            shifted = s + max(0, -s.min() + 0.1)
            p = calibrate_cir(shifted, col)
            p.theta = p.theta - max(0, -s.min() + 0.1)
            p.x0 = float(s.iloc[-1])
            params[col] = p

    names = list(params.keys())
    res_df = pd.concat([params[n].residuals.rename(n) for n in names], axis=1).dropna()
    corr = res_df.corr().values
    eps = 1e-6
    corr = corr + eps * np.eye(len(corr))
    return params, corr, names
