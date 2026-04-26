"""Survival / Hazard PD model.

Two sub-modes:
  1. CIR intensity (closed-form): default intensity λ follows CIR process.
     Survival probability S(T) = A(T) · exp(-B(T) · λ₀).
  2. Cox PH (simplified): h(t) = h₀(t) · exp(β·x) where x are macro covariates.
     PD ≈ 1 - exp(-h₀·T·exp(β·x)).
"""
from __future__ import annotations

from math import exp, log, sqrt

from scipy.stats import norm  # noqa: F401 (available for extensions)

from ..base import BaseModel, ModelResult


class SurvivalHazardPD(BaseModel):
    name = "survival"
    label = "Survival / Hazard"
    description = (
        "Reduced-form intensity model (CIR default intensity). Closed-form survival "
        "probability: S(T) = A(T)·exp(-B(T)·λ₀). Also supports simplified Cox PH."
    )

    def compute(self, **params) -> ModelResult:
        log_: list[str] = []
        mode = str(params.get("mode", "cir"))

        if mode == "cox":
            return self._cox_ph(params, log_)
        return self._cir_intensity(params, log_)

    # ------------------------------------------------------------------ CIR
    def _cir_intensity(self, params: dict, log_: list[str]) -> ModelResult:
        lambda0 = float(params.get("lambda0", 0.03))
        kappa = float(params.get("kappa", 0.40))
        theta = float(params.get("theta", 0.03))
        sigma = float(params.get("sigma", 0.08))
        T = float(params.get("horizon", 1.0))

        if any(x <= 0 for x in [kappa, theta, sigma, T]):
            return ModelResult(value=float("nan"), log=["κ, θ, σ, T must be > 0"])

        gamma = sqrt(kappa ** 2 + 2 * sigma ** 2)
        denom = (gamma + kappa) * (exp(gamma * T) - 1) + 2 * gamma
        B = 2 * (exp(gamma * T) - 1) / denom
        A = (2 * gamma * exp((kappa + gamma) * T / 2) / denom) ** (2 * kappa * theta / sigma ** 2)
        survival = A * exp(-B * lambda0)
        pd_val = float(1 - survival)

        log_.append("── CIR Intensity (Reduced-Form) ──")
        log_.append(f"  λ₀ (initial intensity) = {lambda0:.4f}")
        log_.append(f"  κ  (mean reversion)    = {kappa:.4f}")
        log_.append(f"  θ  (long-run mean)     = {theta:.4f}")
        log_.append(f"  σ  (vol of intensity)  = {sigma:.4f}")
        log_.append(f"  T  (horizon)           = {T:.2f}y")
        log_.append(f"  γ = √(κ² + 2σ²)       = {gamma:.4f}")
        log_.append(f"  A(T)                   = {A:.4f}")
        log_.append(f"  B(T)                   = {B:.4f}")
        log_.append(f"  S(T) = A·exp(-B·λ₀)   = {survival:.4%}")
        log_.append(f"  PD = 1 - S(T)          = {pd_val:.4%}")

        return ModelResult(
            value=pd_val,
            log=log_,
            metadata={"survival": float(survival), "gamma": gamma, "A": A, "B": B},
        )

    # ---------------------------------------------------------------- Cox PH
    def _cox_ph(self, params: dict, log_: list[str]) -> ModelResult:
        h0 = float(params.get("baseline_hazard", 0.02))
        beta_gdp = float(params.get("beta_gdp", -0.15))
        beta_ue = float(params.get("beta_ue", 0.12))
        gdp = float(params.get("gdp_growth", 1.5))
        ue = float(params.get("unemployment", 8.0))
        T = float(params.get("horizon", 1.0))

        eta = beta_gdp * gdp + beta_ue * ue
        hazard = h0 * exp(eta)
        pd_val = float(1 - exp(-hazard * T))

        log_.append("── Cox Proportional Hazards ──")
        log_.append(f"  Baseline hazard h₀     = {h0:.4f}")
        log_.append(f"  β_GDP × GDP            = {beta_gdp:.4f} × {gdp:.2f} = {beta_gdp*gdp:+.4f}")
        log_.append(f"  β_UE  × UE             = {beta_ue:.4f} × {ue:.2f}  = {beta_ue*ue:+.4f}")
        log_.append(f"  η (linear predictor)   = {eta:.4f}")
        log_.append(f"  h(t) = h₀·exp(η)       = {hazard:.4f}")
        log_.append(f"  PD = 1-exp(-h·T)       = {pd_val:.4%}")

        return ModelResult(
            value=pd_val,
            log=log_,
            metadata={"hazard": hazard, "eta": eta, "baseline_hazard": h0},
        )

    @property
    def param_schema(self) -> list[dict]:
        return [
            {"name": "mode", "label": "Sub-mode", "type": "select",
             "default": "cir", "options": [
                 {"value": "cir", "label": "CIR Intensity"},
                 {"value": "cox", "label": "Cox PH"},
             ]},
            {"name": "lambda0", "label": "Initial Intensity λ₀", "type": "range",
             "default": 0.03, "min": 0.001, "max": 0.20, "step": 0.001, "unit": ""},
            {"name": "kappa", "label": "Mean Reversion κ", "type": "range",
             "default": 0.40, "min": 0.01, "max": 2.0, "step": 0.01, "unit": ""},
            {"name": "theta", "label": "Long-Run Mean θ", "type": "range",
             "default": 0.03, "min": 0.001, "max": 0.20, "step": 0.001, "unit": ""},
            {"name": "sigma", "label": "Intensity Volatility σ", "type": "range",
             "default": 0.08, "min": 0.01, "max": 0.50, "step": 0.01, "unit": ""},
            {"name": "horizon", "label": "Horizon (years)", "type": "range",
             "default": 1.0, "min": 0.25, "max": 5.0, "step": 0.25, "unit": "y"},
            {"name": "baseline_hazard", "label": "Baseline Hazard h₀ (Cox only)", "type": "range",
             "default": 0.02, "min": 0.001, "max": 0.15, "step": 0.001, "unit": ""},
            {"name": "beta_gdp", "label": "β GDP (Cox only)", "type": "range",
             "default": -0.15, "min": -0.50, "max": 0.0, "step": 0.01, "unit": ""},
            {"name": "beta_ue", "label": "β Unemployment (Cox only)", "type": "range",
             "default": 0.12, "min": 0.0, "max": 0.50, "step": 0.01, "unit": ""},
        ]
