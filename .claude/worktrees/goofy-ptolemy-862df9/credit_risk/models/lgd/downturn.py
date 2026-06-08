"""Downturn / Stressed LGD model.

Implements the BCBS/EBA downturn LGD concept:
  LGD_DT = LGD_TTC + add-on(cycle position)

The add-on is calibrated to the worst observed LGD period (peak of credit
losses historically). The stress multiplier scales with the severity of the
macro downturn (GDP decline, unemployment surge, lending standards tightening).

This model is used for IFRS 9 Stage 3 ECL and regulatory IRB capital floors.
"""
from __future__ import annotations

import numpy as np

from ..base import BaseModel, ModelResult


class DownturnLGD(BaseModel):
    name = "downturn"
    label = "Downturn / Stressed LGD"
    description = (
        "BCBS/EBA downturn LGD. Through-the-cycle base LGD with a macro-driven stress "
        "add-on that peaks in recession scenarios. Used for IFRS 9 and IRB capital."
    )

    def compute(self, **params) -> ModelResult:
        log_: list[str] = []
        lgd_ttc = float(params.get("lgd_ttc", 0.40))
        max_addon = float(params.get("max_addon", 0.20))
        gdp_growth = float(params.get("gdp_growth", 1.5))
        unemployment = float(params.get("unemployment", 8.0))
        lending_standards = float(params.get("lending_standards", 0.0))
        downturn_gdp = float(params.get("downturn_gdp", -5.0))
        peak_unemployment = float(params.get("peak_unemployment", 12.0))

        # Composite stress index normalised 0..1
        gdp_stress = np.clip((downturn_gdp - gdp_growth) / (downturn_gdp - 3.0), 0.0, 1.0)
        ue_stress = np.clip((unemployment - 7.0) / (peak_unemployment - 7.0), 0.0, 1.0)
        ls_stress = np.clip(lending_standards / 50.0, 0.0, 1.0)
        composite = float(0.50 * gdp_stress + 0.35 * ue_stress + 0.15 * ls_stress)

        add_on = max_addon * composite
        lgd_dt = float(np.clip(lgd_ttc + add_on, 0.0, 1.0))

        # Sensitivity: Δ LGD per 1% Δ GDP
        delta_lgd_per_gdp = -max_addon * 0.50 / (downturn_gdp - 3.0)

        log_.append("── Downturn / Stressed LGD ──")
        log_.append(f"  LGD TTC (through-the-cycle) = {lgd_ttc:.2%}")
        log_.append(f"  Max downturn add-on          = {max_addon:.2%}")
        log_.append("")
        log_.append("  Stress drivers:")
        log_.append(f"    GDP growth    {gdp_growth:.1f}%  →  stress index {gdp_stress:.3f}")
        log_.append(f"    Unemployment  {unemployment:.1f}%  →  stress index {ue_stress:.3f}")
        log_.append(f"    Lend. stds    {lending_standards:.0f}   →  stress index {ls_stress:.3f}")
        log_.append(f"  Composite stress = {composite:.3f}")
        log_.append(f"  Add-on = {max_addon:.2%} × {composite:.3f} = {add_on:.2%}")
        log_.append(f"  LGD_DT = {lgd_ttc:.2%} + {add_on:.2%} = {lgd_dt:.2%}")
        log_.append(f"  Sensitivity: Δ LGD / Δ GDP = {delta_lgd_per_gdp:.4f}")

        return ModelResult(
            value=lgd_dt,
            log=log_,
            metadata={
                "lgd_ttc": lgd_ttc, "add_on": add_on, "composite_stress": composite,
                "gdp_stress": float(gdp_stress), "ue_stress": float(ue_stress),
                "ls_stress": float(ls_stress), "delta_lgd_per_gdp": delta_lgd_per_gdp,
            },
        )

    @property
    def param_schema(self) -> list[dict]:
        return [
            {"name": "lgd_ttc", "label": "LGD TTC (base)", "type": "range",
             "default": 0.40, "min": 0.05, "max": 0.90, "step": 0.01, "unit": ""},
            {"name": "max_addon", "label": "Max Downturn Add-on", "type": "range",
             "default": 0.20, "min": 0.0, "max": 0.40, "step": 0.01, "unit": ""},
            {"name": "gdp_growth", "label": "Scenario GDP Growth (%)", "type": "range",
             "default": 1.5, "min": -8.0, "max": 6.0, "step": 0.1, "unit": "%"},
            {"name": "unemployment", "label": "Scenario Unemployment (%)", "type": "range",
             "default": 8.0, "min": 3.0, "max": 20.0, "step": 0.1, "unit": "%"},
            {"name": "lending_standards", "label": "Lending Standards Index", "type": "range",
             "default": 0.0, "min": -40.0, "max": 60.0, "step": 1.0, "unit": ""},
            {"name": "downturn_gdp", "label": "Worst GDP (calibration)", "type": "range",
             "default": -5.0, "min": -12.0, "max": -1.0, "step": 0.5, "unit": "%"},
            {"name": "peak_unemployment", "label": "Peak Unemployment (calib.)", "type": "range",
             "default": 12.0, "min": 8.0, "max": 25.0, "step": 0.5, "unit": "%"},
        ]
