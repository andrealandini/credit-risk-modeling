"""Merton (1974) structural model.

Firm defaults when asset value at maturity V_T < D (face value of debt).
PD = N(-d₂)   where d₂ = [ln(V/D) + (r - σ²/2)T] / (σ√T)
"""
from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm

from ..base import BaseModel, ModelResult


class MertonPD(BaseModel):
    name = "merton"
    label = "Merton Structural"
    description = (
        "Merton (1974) distance-to-default model. The firm defaults when asset value "
        "falls below debt at maturity. PD = N(-d₂); Distance-to-Default = d₂."
    )

    def compute(self, **params) -> ModelResult:
        log_: list[str] = []
        V = float(params.get("asset_value", 100.0))     # €M
        D = float(params.get("debt", 70.0))              # €M
        sigma = float(params.get("asset_vol", 0.25))
        r = float(params.get("risk_free", 0.03))
        T = float(params.get("horizon", 1.0))

        if V <= 0 or D <= 0 or sigma <= 0 or T <= 0:
            return ModelResult(value=float("nan"), log=["Invalid parameters — all must be positive"])

        leverage = D / V
        d1 = (log(V / D) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        pd_val = float(norm.cdf(-d2))
        dd = float(d2)

        log_.append("── Merton Structural Model ──")
        log_.append(f"  V₀ (asset value)  = €{V:.2f}M")
        log_.append(f"  D  (debt)         = €{D:.2f}M")
        log_.append(f"  Leverage          = {leverage:.2%}")
        log_.append(f"  σ_V (asset vol)   = {sigma:.2%}")
        log_.append(f"  r  (risk-free)    = {r:.2%}")
        log_.append(f"  T  (horizon)      = {T:.2f}y")
        log_.append(f"  d₁                = {d1:.4f}")
        log_.append(f"  d₂                = {d2:.4f}")
        log_.append(f"  DD = d₂           = {dd:.4f}")
        log_.append(f"  PD = N(-d₂)       = {pd_val:.4%}")

        return ModelResult(
            value=pd_val,
            log=log_,
            metadata={"d1": d1, "d2": d2, "distance_to_default": dd, "leverage": leverage},
        )

    @property
    def param_schema(self) -> list[dict]:
        return [
            {"name": "asset_value", "label": "Asset Value (€M)", "type": "number",
             "default": 100.0, "min": 1.0, "max": 10000.0, "step": 1.0, "unit": "€M"},
            {"name": "debt", "label": "Debt Face Value (€M)", "type": "number",
             "default": 70.0, "min": 1.0, "max": 10000.0, "step": 1.0, "unit": "€M"},
            {"name": "asset_vol", "label": "Asset Volatility (σ)", "type": "range",
             "default": 0.25, "min": 0.05, "max": 0.80, "step": 0.01, "unit": "σ"},
            {"name": "risk_free", "label": "Risk-Free Rate (%)", "type": "range",
             "default": 0.03, "min": 0.00, "max": 0.10, "step": 0.005, "unit": "%"},
            {"name": "horizon", "label": "Horizon (years)", "type": "range",
             "default": 1.0, "min": 0.25, "max": 5.0, "step": 0.25, "unit": "y"},
        ]
