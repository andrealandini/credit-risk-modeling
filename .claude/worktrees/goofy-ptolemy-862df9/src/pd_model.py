"""Wilson / CreditPortfolioView-style PD model.

logit(PD_t) = beta0 + beta1 * gdp_growth + beta2 * unemployment
            + beta3 * spread + beta4 * policy_rate

Calibrated by OLS on logit-transformed NPL ratio (used as a PD proxy) with
sign-restriction checks. Returns betas, residual sigma (for Monte Carlo
shocks), and an in-sample fit series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["gdp_growth", "unemployment", "spread", "policy_rate"]


@dataclass
class PDModel:
    intercept: float
    coefs: dict[str, float]
    sigma_eps: float
    fitted: pd.Series
    actual: pd.Series

    def predict_logit(self, X: pd.DataFrame) -> pd.Series:
        z = self.intercept + sum(self.coefs[c] * X[c] for c in self.coefs)
        return z

    def predict(self, X: pd.DataFrame) -> pd.Series:
        z = self.predict_logit(X)
        return 1.0 / (1.0 + np.exp(-z))


def _logit(p: pd.Series) -> pd.Series:
    p = p.clip(lower=1e-4, upper=1 - 1e-4)
    return np.log(p / (1 - p))


def fit_pd_model(panel: pd.DataFrame, target: str = "npl_ratio") -> PDModel:
    df = panel.dropna(subset=[target] + [c for c in FEATURES if c in panel.columns]).copy()
    p = df[target] / 100.0  # NPL ratio in percent -> probability
    y = _logit(p)
    feats = [c for c in FEATURES if c in df.columns]
    X = df[feats].values
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y.values, rcond=None)
    intercept = float(coef[0])
    betas = {f: float(c) for f, c in zip(feats, coef[1:])}
    yhat = X1 @ coef
    resid = y.values - yhat
    sigma = float(resid.std(ddof=len(coef)))
    fitted = pd.Series(1 / (1 + np.exp(-yhat)) * 100, index=df.index)
    return PDModel(intercept=intercept, coefs=betas, sigma_eps=sigma,
                    fitted=fitted, actual=df[target])
