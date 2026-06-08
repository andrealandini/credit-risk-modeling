"""Beta regression LGD model.

LGD is modelled as Beta(α, β) where:
  μ = logistic(γ₀ + γ₁·seniority + γ₂·collateral + γ₃·ltv + γ₄·gdp)
  φ = concentration parameter (precision)
  α = μ · φ,  β = (1 - μ) · φ

Returns mean LGD, std, and regulatory quantiles (Basel downturn).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

from ..base import BaseModel, ModelResult

_SENIORITY_MAP = {"senior_secured": -0.50, "senior_unsecured": 0.0, "subordinated": 0.60}
_COLLATERAL_MAP = {"none": 0.40, "residential": -0.30, "commercial": -0.15, "financial": -0.40}


class BetaRegressionLGD(BaseModel):
    name = "beta_regression"
    label = "Beta / Regression LGD"
    description = (
        "Beta regression model: LGD ~ Beta(α, β). Parameters are driven by seniority, "
        "collateral type, LTV, and macro conditions. Returns mean, std, and quantiles."
    )

    def compute(self, **params) -> ModelResult:
        log_: list[str] = []
        seniority = str(params.get("seniority", "senior_unsecured"))
        collateral = str(params.get("collateral", "none"))
        ltv = float(params.get("ltv", 0.70))
        gdp = float(params.get("gdp_growth", 1.5))
        phi = float(params.get("concentration", 8.0))

        gamma0 = -0.20
        gamma_sr = _SENIORITY_MAP.get(seniority, 0.0)
        gamma_col = _COLLATERAL_MAP.get(collateral, 0.0)
        gamma_ltv = 0.80 * (ltv - 0.60)
        gamma_gdp = -0.03 * gdp

        eta = gamma0 + gamma_sr + gamma_col + gamma_ltv + gamma_gdp
        mu = float(1.0 / (1.0 + np.exp(-eta)))
        mu = np.clip(mu, 0.05, 0.95)

        a = mu * phi
        b = (1 - mu) * phi
        lgd_mean = float(beta_dist.mean(a, b))
        lgd_std = float(beta_dist.std(a, b))
        lgd_q75 = float(beta_dist.ppf(0.75, a, b))
        lgd_q90 = float(beta_dist.ppf(0.90, a, b))
        lgd_q99 = float(beta_dist.ppf(0.99, a, b))

        log_.append("── Beta Regression LGD ──")
        log_.append(f"  Seniority   : {seniority}  (γ = {gamma_sr:+.2f})")
        log_.append(f"  Collateral  : {collateral}  (γ = {gamma_col:+.2f})")
        log_.append(f"  LTV         : {ltv:.0%}  (γ = {gamma_ltv:+.4f})")
        log_.append(f"  GDP growth  : {gdp:.1f}%  (γ = {gamma_gdp:+.4f})")
        log_.append(f"  η (linear)  : {eta:.4f}")
        log_.append(f"  μ (mean LGD): {mu:.4f}")
        log_.append(f"  φ (precision): {phi:.1f}  →  α={a:.2f}, β={b:.2f}")
        log_.append(f"  LGD mean    : {lgd_mean:.2%}")
        log_.append(f"  LGD std     : {lgd_std:.2%}")
        log_.append(f"  LGD Q90     : {lgd_q90:.2%}")
        log_.append(f"  LGD Q99     : {lgd_q99:.2%}")

        return ModelResult(
            value=lgd_mean,
            log=log_,
            metadata={"std": lgd_std, "q75": lgd_q75, "q90": lgd_q90, "q99": lgd_q99,
                      "alpha": a, "beta": b, "mu": mu},
        )

    @property
    def param_schema(self) -> list[dict]:
        return [
            {"name": "seniority", "label": "Seniority", "type": "select",
             "default": "senior_unsecured", "options": [
                 {"value": "senior_secured", "label": "Senior Secured"},
                 {"value": "senior_unsecured", "label": "Senior Unsecured"},
                 {"value": "subordinated", "label": "Subordinated"},
             ]},
            {"name": "collateral", "label": "Collateral Type", "type": "select",
             "default": "none", "options": [
                 {"value": "none", "label": "None"},
                 {"value": "residential", "label": "Residential RE"},
                 {"value": "commercial", "label": "Commercial RE"},
                 {"value": "financial", "label": "Financial Collateral"},
             ]},
            {"name": "ltv", "label": "LTV Ratio", "type": "range",
             "default": 0.70, "min": 0.10, "max": 1.20, "step": 0.05, "unit": ""},
            {"name": "gdp_growth", "label": "GDP Growth (%)", "type": "range",
             "default": 1.5, "min": -8.0, "max": 6.0, "step": 0.1, "unit": "%"},
            {"name": "concentration", "label": "Concentration φ", "type": "range",
             "default": 8.0, "min": 2.0, "max": 50.0, "step": 0.5, "unit": ""},
        ]
