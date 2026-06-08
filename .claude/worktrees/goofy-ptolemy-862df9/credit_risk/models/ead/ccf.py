"""Credit Conversion Factor (CCF) EAD model.

EAD = drawn_balance + CCF × (committed_limit - drawn_balance)
    = drawn + CCF × undrawn

CCF varies by product type and, for IRB institutions, is estimated
from historical drawdown patterns at/near default.

Regulatory SA CCFs:
  - Committed undrawn (revolver) : 75%
  - Term loan undrawn            : 100%
  - Trade finance                : 20%
"""
from __future__ import annotations

from ..base import BaseModel, ModelResult

_CCF_TABLE = {
    "revolver":     {"regulatory": 0.75, "internal": 0.60, "description": "Revolving credit line"},
    "term_loan":    {"regulatory": 1.00, "internal": 0.95, "description": "Committed term loan"},
    "trade_finance":{"regulatory": 0.20, "internal": 0.15, "description": "Trade finance facility"},
    "overdraft":    {"regulatory": 0.75, "internal": 0.55, "description": "Overdraft facility"},
    "custom":       {"regulatory": None, "internal": None,  "description": "User-defined CCF"},
}


class CCFEAD(BaseModel):
    name = "ccf"
    label = "Credit Conversion Factor (CCF)"
    description = (
        "EAD = Drawn + CCF × Undrawn. CCF converts off-balance-sheet commitments "
        "into on-balance-sheet equivalents for capital/ECL purposes."
    )

    def compute(self, **params) -> ModelResult:
        log_: list[str] = []
        product = str(params.get("product_type", "revolver"))
        limit = float(params.get("committed_limit", 1_000_000.0))
        drawn = float(params.get("drawn_balance", 400_000.0))
        ccf_source = str(params.get("ccf_source", "internal"))
        custom_ccf = float(params.get("custom_ccf", 0.60))
        stress_adj = float(params.get("stress_adjustment", 0.0))

        undrawn = max(limit - drawn, 0.0)

        info = _CCF_TABLE.get(product, _CCF_TABLE["revolver"])
        if product == "custom":
            ccf = custom_ccf
        else:
            ccf = info.get(ccf_source) or info["regulatory"] or custom_ccf
        ccf = min(max(float(ccf) + stress_adj, 0.0), 1.0)

        ead = drawn + ccf * undrawn
        utilization = drawn / limit if limit > 0 else 0.0

        log_.append("── CCF EAD Model ──")
        log_.append(f"  Product         : {info.get('description', product)}")
        log_.append(f"  Committed limit : €{limit:,.2f}")
        log_.append(f"  Drawn balance   : €{drawn:,.2f}  ({utilization:.1%} utilisation)")
        log_.append(f"  Undrawn         : €{undrawn:,.2f}")
        log_.append(f"  CCF ({ccf_source})   : {ccf:.2%}")
        if stress_adj != 0:
            log_.append(f"  Stress adj      : {stress_adj:+.2%}")
        log_.append(f"  EAD = {drawn:,.0f} + {ccf:.2f} × {undrawn:,.0f}")
        log_.append(f"  EAD             = €{ead:,.2f}")

        return ModelResult(
            value=ead,
            log=log_,
            metadata={"drawn": drawn, "undrawn": undrawn, "ccf": ccf,
                      "utilization": utilization, "limit": limit},
        )

    @property
    def param_schema(self) -> list[dict]:
        return [
            {"name": "product_type", "label": "Product Type", "type": "select",
             "default": "revolver", "options": [
                 {"value": "revolver", "label": "Revolving Credit"},
                 {"value": "term_loan", "label": "Term Loan"},
                 {"value": "trade_finance", "label": "Trade Finance"},
                 {"value": "overdraft", "label": "Overdraft"},
                 {"value": "custom", "label": "Custom"},
             ]},
            {"name": "committed_limit", "label": "Committed Limit (€)", "type": "number",
             "default": 1000000, "min": 10000, "max": 500000000, "step": 10000, "unit": "€"},
            {"name": "drawn_balance", "label": "Drawn Balance (€)", "type": "number",
             "default": 400000, "min": 0, "max": 500000000, "step": 10000, "unit": "€"},
            {"name": "ccf_source", "label": "CCF Source", "type": "select",
             "default": "internal", "options": [
                 {"value": "internal", "label": "Internal estimate"},
                 {"value": "regulatory", "label": "Regulatory SA"},
             ]},
            {"name": "custom_ccf", "label": "Custom CCF (if product=custom)", "type": "range",
             "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01, "unit": ""},
            {"name": "stress_adjustment", "label": "Stress CCF add-on", "type": "range",
             "default": 0.0, "min": -0.20, "max": 0.30, "step": 0.01, "unit": ""},
        ]
