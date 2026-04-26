"""Wilson / CreditPortfolioView logistic regression PD model.

logit(PD) = β₀ + β₁·GDP + β₂·UE + β₃·spread + β₄·policy_rate + β₅·credit_growth

Fitted via OLS on logit-transformed NPL ratio from the ECB panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseModel, ModelResult

_FEATURES = ["gdp_growth", "unemployment", "spread", "policy_rate", "credit_growth"]


def _fit_coefficients(panel: pd.DataFrame) -> tuple[float, dict[str, float], float]:
    df = panel.dropna(subset=["npl_ratio"] + [c for c in _FEATURES if c in panel.columns])
    p = (df["npl_ratio"] / 100.0).clip(1e-4, 1 - 1e-4)
    y = np.log(p / (1 - p))
    feats = [c for c in _FEATURES if c in df.columns]
    X = df[feats].values
    X1 = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X1, y.values, rcond=None)
    resid = y.values - X1 @ coef
    return float(coef[0]), {f: float(c) for f, c in zip(feats, coef[1:])}, float(resid.std(ddof=len(coef)))


class LogisticRegressionPD(BaseModel):
    name = "logistic_regression"
    label = "Logistic Regression"
    description = (
        "Wilson (1997) CreditPortfolioView model. Maps macro factors to PD "
        "via a logistic link: logit(PD) = β₀ + Σ βᵢ·xᵢ. Calibrated on ECB NPL data."
    )

    def __init__(self, panel: pd.DataFrame | None = None):
        if panel is not None:
            self._intercept, self._coefs, self._sigma = _fit_coefficients(panel)
        else:
            self._intercept = -3.5
            self._coefs = {
                "gdp_growth": -0.18,
                "unemployment": 0.22,
                "spread": 0.15,
                "policy_rate": -0.08,
                "credit_growth": -0.05,
            }
            self._sigma = 0.12

    def compute(self, **params) -> ModelResult:
        log: list[str] = []
        gdp = float(params.get("gdp_growth", 1.5))
        ue = float(params.get("unemployment", 8.0))
        spread = float(params.get("spread", 1.5))
        rate = float(params.get("policy_rate", 2.5))
        credit = float(params.get("credit_growth", 3.0))

        vals = {
            "gdp_growth": gdp,
            "unemployment": ue,
            "spread": spread,
            "policy_rate": rate,
            "credit_growth": credit,
        }
        z = self._intercept + sum(
            self._coefs.get(f, 0.0) * v for f, v in vals.items()
        )
        pd_hat = float(1.0 / (1.0 + np.exp(-z)))

        log.append("── Logistic Regression PD ──")
        log.append(f"  logit(PD) = {self._intercept:.4f}")
        for f, v in vals.items():
            b = self._coefs.get(f, 0.0)
            log.append(f"    + {b:+.4f} × {f} ({v:.2f}) = {b*v:+.4f}")
        log.append(f"  z = {z:.4f}")
        log.append(f"  PD = σ(z) = {pd_hat:.4%}")

        return ModelResult(
            value=pd_hat,
            log=log,
            metadata={"z": z, "intercept": self._intercept, "coefs": self._coefs},
        )

    @property
    def param_schema(self) -> list[dict]:
        return [
            {"name": "gdp_growth", "label": "GDP Growth (%)", "type": "range",
             "default": 1.5, "min": -8.0, "max": 6.0, "step": 0.1, "unit": "%"},
            {"name": "unemployment", "label": "Unemployment (%)", "type": "range",
             "default": 8.0, "min": 3.0, "max": 20.0, "step": 0.1, "unit": "%"},
            {"name": "spread", "label": "Credit Spread (%)", "type": "range",
             "default": 1.5, "min": 0.0, "max": 6.0, "step": 0.1, "unit": "%"},
            {"name": "policy_rate", "label": "ECB Policy Rate (%)", "type": "range",
             "default": 2.5, "min": 0.0, "max": 5.0, "step": 0.05, "unit": "%"},
            {"name": "credit_growth", "label": "Credit Growth (%)", "type": "range",
             "default": 3.0, "min": -10.0, "max": 15.0, "step": 0.5, "unit": "%"},
        ]
