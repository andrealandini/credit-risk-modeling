"""Workout LGD model — discounted recovery cash-flow approach.

LGD = 1 - (PV of recoveries - costs) / EAD

Recoveries are modelled as staged cash flows:
  t=0:  immediate collateral realisation (if secured)
  t=1:  litigation / workout cash flows
  t=2+: residual recovery and estate distribution

Discounted at a workout discount rate r_w = risk-free + risk premium.
"""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelResult


class WorkoutLGD(BaseModel):
    name = "workout"
    label = "Workout LGD"
    description = (
        "Discounted cash-flow approach. Recovery staged over a multi-year workout "
        "process; discounted at risk-free + risk premium. Includes admin/legal costs."
    )

    def compute(self, **params) -> ModelResult:
        log_: list[str] = []
        ead = float(params.get("ead", 1_000_000.0))
        collateral_cover = float(params.get("collateral_cover", 0.40))
        legal_cost_pct = float(params.get("legal_cost_pct", 0.08))
        admin_cost_pct = float(params.get("admin_cost_pct", 0.04))
        workout_years = max(1, min(7, int(round(float(params.get("workout_years", 3))))))
        risk_free = float(params.get("risk_free", 0.03))
        risk_premium = float(params.get("risk_premium", 0.05))
        time_value_discount = risk_free + risk_premium

        # Stage 0: immediate collateral recovery (haircut 20%)
        collateral_recovery = ead * collateral_cover * 0.80

        # Staged cash flows: exponentially front-loaded
        raw_w = np.exp(-0.5 * np.arange(workout_years))
        weights = raw_w / raw_w.sum()
        remaining = ead - collateral_recovery
        # Unsecured recovery rate ~ 30–40% of remaining
        unsecured_rcr = float(params.get("unsecured_recovery", 0.35))
        unsecured_recovery = remaining * unsecured_rcr

        cash_flows = weights * unsecured_recovery
        years = np.arange(1, workout_years + 1)
        discount_factors = 1.0 / (1 + time_value_discount) ** years
        pv_unsecured = float(np.dot(cash_flows, discount_factors))

        # PV of collateral realised at t=0.5 (quick sale)
        pv_collateral = collateral_recovery / (1 + time_value_discount) ** 0.5

        total_recovery = pv_collateral + pv_unsecured
        costs = ead * (legal_cost_pct + admin_cost_pct)
        net_recovery = max(total_recovery - costs, 0.0)
        lgd = float(1.0 - net_recovery / ead)
        lgd = np.clip(lgd, 0.0, 1.0)

        log_.append("── Workout LGD ──")
        log_.append(f"  EAD                = €{ead:,.2f}")
        log_.append(f"  Collateral cover   = {collateral_cover:.0%}")
        log_.append(f"  Collateral recovery= €{collateral_recovery:,.2f}")
        log_.append(f"  PV collateral      = €{pv_collateral:,.2f}")
        log_.append(f"  Unsecured RCR      = {unsecured_rcr:.0%}")
        log_.append(f"  PV unsecured rec.  = €{pv_unsecured:,.2f}")
        log_.append(f"  Costs (legal+admin)= €{costs:,.2f}  ({legal_cost_pct+admin_cost_pct:.0%})")
        log_.append(f"  Net recovery       = €{net_recovery:,.2f}")
        log_.append(f"  Workout discount r = {time_value_discount:.2%}")
        log_.append(f"  LGD                = {lgd:.4%}")

        return ModelResult(
            value=lgd,
            log=log_,
            metadata={"net_recovery": net_recovery, "costs": costs,
                      "pv_collateral": pv_collateral, "pv_unsecured": pv_unsecured},
        )

    @property
    def param_schema(self) -> list[dict]:
        return [
            {"name": "ead", "label": "EAD (€)", "type": "number",
             "default": 1000000, "min": 10000, "max": 100000000, "step": 10000, "unit": "€"},
            {"name": "collateral_cover", "label": "Collateral Cover", "type": "range",
             "default": 0.40, "min": 0.0, "max": 1.20, "step": 0.05, "unit": ""},
            {"name": "unsecured_recovery", "label": "Unsecured Recovery Rate", "type": "range",
             "default": 0.35, "min": 0.0, "max": 0.80, "step": 0.05, "unit": ""},
            {"name": "legal_cost_pct", "label": "Legal Costs (%EAD)", "type": "range",
             "default": 0.08, "min": 0.01, "max": 0.25, "step": 0.01, "unit": ""},
            {"name": "admin_cost_pct", "label": "Admin Costs (%EAD)", "type": "range",
             "default": 0.04, "min": 0.01, "max": 0.15, "step": 0.01, "unit": ""},
            {"name": "workout_years", "label": "Workout Duration (years)", "type": "range",
             "default": 3, "min": 1, "max": 7, "step": 1, "unit": "y"},
            {"name": "risk_free", "label": "Risk-Free Rate", "type": "range",
             "default": 0.03, "min": 0.00, "max": 0.08, "step": 0.005, "unit": ""},
            {"name": "risk_premium", "label": "Workout Risk Premium", "type": "range",
             "default": 0.05, "min": 0.01, "max": 0.15, "step": 0.005, "unit": ""},
        ]
