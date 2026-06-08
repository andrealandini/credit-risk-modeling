"""Utilization regression EAD model.

Models credit-line utilization as a function of macro factors:
  logit(U) = α₀ + α₁·GDP + α₂·UE + α₃·policy_rate + α₄·credit_cycle

EAD = max(U × limit, drawn_balance)

High utilization = borrowers draw down credit lines when conditions deteriorate
(negative GDP, rising unemployment).
"""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelResult


class UtilizationRegressionEAD(BaseModel):
    name = "utilization"
    label = "Utilization Regression"
    description = (
        "Macro-driven utilization model: logit(U) = α·X. Utilization rises as "
        "conditions deteriorate (GDP ↓, UE ↑). EAD = U × committed limit."
    )

    # Default coefficients calibrated to match stylised facts
    _ALPHA = {
        "intercept": -0.50,
        "gdp_growth": -0.06,
        "unemployment": 0.08,
        "policy_rate": -0.04,
        "credit_growth": -0.03,
    }

    def compute(self, **params) -> ModelResult:
        log_: list[str] = []
        limit = float(params.get("committed_limit", 1_000_000.0))
        drawn = float(params.get("drawn_balance", 400_000.0))
        gdp = float(params.get("gdp_growth", 1.5))
        ue = float(params.get("unemployment", 8.0))
        rate = float(params.get("policy_rate", 2.5))
        credit = float(params.get("credit_growth", 3.0))

        a = self._ALPHA
        eta = (a["intercept"] + a["gdp_growth"] * gdp + a["unemployment"] * ue
               + a["policy_rate"] * rate + a["credit_growth"] * credit)
        util_hat = float(1.0 / (1.0 + np.exp(-eta)))
        util_hat = np.clip(util_hat, 0.01, 0.99)

        ead_pred = util_hat * limit
        ead = max(ead_pred, drawn)

        current_util = drawn / limit if limit > 0 else 0.0

        log_.append("── Utilization Regression EAD ──")
        log_.append(f"  Committed limit    : €{limit:,.2f}")
        log_.append(f"  Current drawn      : €{drawn:,.2f}  ({current_util:.1%} util.)")
        log_.append(f"  Macro inputs:")
        log_.append(f"    GDP growth       : {gdp:.1f}%  → {a['gdp_growth']*gdp:+.4f}")
        log_.append(f"    Unemployment     : {ue:.1f}%  → {a['unemployment']*ue:+.4f}")
        log_.append(f"    Policy rate      : {rate:.2f}%  → {a['policy_rate']*rate:+.4f}")
        log_.append(f"    Credit growth    : {credit:.1f}%  → {a['credit_growth']*credit:+.4f}")
        log_.append(f"  η (linear pred)    : {eta:.4f}")
        log_.append(f"  Predicted util     : {util_hat:.2%}")
        log_.append(f"  EAD = max({ead_pred:,.0f}, {drawn:,.0f})")
        log_.append(f"  EAD                : €{ead:,.2f}")

        return ModelResult(
            value=ead,
            log=log_,
            metadata={"utilization": util_hat, "ead_predicted": ead_pred,
                      "current_utilization": current_util, "eta": eta},
        )

    @property
    def param_schema(self) -> list[dict]:
        return [
            {"name": "committed_limit", "label": "Committed Limit (€)", "type": "number",
             "default": 1000000, "min": 10000, "max": 500000000, "step": 10000, "unit": "€"},
            {"name": "drawn_balance", "label": "Current Drawn (€)", "type": "number",
             "default": 400000, "min": 0, "max": 500000000, "step": 10000, "unit": "€"},
            {"name": "gdp_growth", "label": "GDP Growth (%)", "type": "range",
             "default": 1.5, "min": -8.0, "max": 6.0, "step": 0.1, "unit": "%"},
            {"name": "unemployment", "label": "Unemployment (%)", "type": "range",
             "default": 8.0, "min": 3.0, "max": 20.0, "step": 0.1, "unit": "%"},
            {"name": "policy_rate", "label": "Policy Rate (%)", "type": "range",
             "default": 2.5, "min": 0.0, "max": 5.0, "step": 0.05, "unit": "%"},
            {"name": "credit_growth", "label": "Credit Growth (%)", "type": "range",
             "default": 3.0, "min": -10.0, "max": 15.0, "step": 0.5, "unit": "%"},
        ]
