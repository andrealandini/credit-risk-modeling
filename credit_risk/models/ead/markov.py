"""Markov transition EAD model.

Models credit-line utilization as a 3-state Markov chain:
  State 0: Low  utilization (< 33%)
  State 1: Med  utilization (33 – 66%)
  State 2: High utilization (> 66%)

Transition matrix P is macro-conditioned:
  In stress, P shifts mass toward high-utilization state.

EAD(T) = E[utilization at horizon] × limit
       = π(T) · [mid-state EAD levels]

π(T) = π(0) · P^T   (matrix exponentiation)
"""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelResult

_STATE_MID_UTIL = np.array([0.165, 0.495, 0.830])  # mid-point of each state
_STATE_LABELS = ["Low (<33%)", "Med (33–66%)", "High (>66%)"]


def _macro_adjusted_matrix(gdp: float, ue: float) -> np.ndarray:
    """Base transition matrix softly conditioned on macro state."""
    # Stress factor: negative GDP + high unemployment → more mass toward High state
    stress = np.clip((-gdp / 4.0 + ue / 12.0) / 2, 0.0, 1.0)

    # Base matrix (calibrated to average cycle)
    P_base = np.array([
        [0.70, 0.22, 0.08],
        [0.12, 0.68, 0.20],
        [0.06, 0.20, 0.74],
    ])
    # Stressed matrix (crisis regime)
    P_stress = np.array([
        [0.55, 0.28, 0.17],
        [0.08, 0.60, 0.32],
        [0.03, 0.15, 0.82],
    ])
    P = (1 - stress) * P_base + stress * P_stress
    # Row-normalise to ensure valid stochastic matrix
    return P / P.sum(axis=1, keepdims=True)


def _matrix_power(P: np.ndarray, n: int) -> np.ndarray:
    result = np.eye(len(P))
    for _ in range(n):
        result = result @ P
    return result


class MarkovTransitionEAD(BaseModel):
    name = "markov"
    label = "Markov Transition EAD"
    description = (
        "3-state Markov chain on credit-line utilization (low / med / high). "
        "Macro-conditioned transition matrix. EAD = E[utilization at horizon] × limit."
    )

    def compute(self, **params) -> ModelResult:
        log_: list[str] = []
        limit = float(params.get("committed_limit", 1_000_000.0))
        drawn = float(params.get("drawn_balance", 400_000.0))
        gdp = float(params.get("gdp_growth", 1.5))
        ue = float(params.get("unemployment", 8.0))
        horizon_q = int(params.get("horizon_quarters", 4))

        current_util = drawn / limit if limit > 0 else 0.40
        if current_util < 0.33:
            init_state = 0
        elif current_util < 0.66:
            init_state = 1
        else:
            init_state = 2

        pi0 = np.zeros(3)
        pi0[init_state] = 1.0

        P = _macro_adjusted_matrix(gdp, ue)
        P_T = _matrix_power(P, horizon_q)
        pi_T = pi0 @ P_T

        expected_util = float(np.dot(pi_T, _STATE_MID_UTIL))
        ead = max(expected_util * limit, drawn)

        log_.append("── Markov Transition EAD ──")
        log_.append(f"  Committed limit     : €{limit:,.2f}")
        log_.append(f"  Current drawn       : €{drawn:,.2f}  ({current_util:.1%} util.)")
        log_.append(f"  Initial state       : {_STATE_LABELS[init_state]}")
        log_.append(f"  Horizon             : {horizon_q}Q")
        log_.append(f"  Macro (GDP={gdp:.1f}%, UE={ue:.1f}%)")
        log_.append("")
        log_.append("  Transition matrix P (1Q):")
        for i, row in enumerate(P):
            log_.append(f"    {_STATE_LABELS[i]}: {' | '.join(f'{v:.3f}' for v in row)}")
        log_.append("")
        log_.append(f"  State distribution π(T={horizon_q}Q):")
        for i, (label, prob) in enumerate(zip(_STATE_LABELS, pi_T)):
            log_.append(f"    {label}: {prob:.3f}")
        log_.append(f"  E[utilization]      : {expected_util:.2%}")
        log_.append(f"  EAD                 : €{ead:,.2f}")

        return ModelResult(
            value=ead,
            log=log_,
            metadata={
                "state_probs": pi_T.tolist(),
                "expected_utilization": expected_util,
                "init_state": init_state,
                "transition_matrix": P.tolist(),
            },
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
            {"name": "horizon_quarters", "label": "Horizon (quarters)", "type": "range",
             "default": 4, "min": 1, "max": 12, "step": 1, "unit": "Q"},
        ]
