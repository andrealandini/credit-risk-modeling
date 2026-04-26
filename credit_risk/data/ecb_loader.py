"""Synthetic ECB macroeconomic dataset with realistic economic cycles.

Generates 100 quarters (2000 Q1 – 2024 Q4) encoding:
  - 2008/09 GFC, 2011/12 Euro debt crisis, 2020 COVID, 2022 inflation shock.

All figures are in standard economic units (%, pp, index points).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MacroScenario:
    name: str
    gdp_growth: float = 1.5
    unemployment: float = 8.0
    inflation: float = 2.0
    policy_rate: float = 1.5
    credit_growth: float = 3.0
    lending_standards: float = 0.0

    def as_dict(self) -> dict:
        return {
            "gdp_growth": self.gdp_growth,
            "unemployment": self.unemployment,
            "inflation": self.inflation,
            "policy_rate": self.policy_rate,
            "credit_growth": self.credit_growth,
            "lending_standards": self.lending_standards,
        }


BUILTIN_SCENARIOS: dict[str, MacroScenario] = {
    "baseline": MacroScenario("ECB Baseline", 1.5, 7.5, 2.0, 2.5, 4.0, 5.0),
    "recession": MacroScenario("Recession Stress", -3.0, 11.0, 1.0, 0.5, -2.0, -35.0),
    "stagflation": MacroScenario("Stagflation", 0.0, 9.5, 7.5, 4.0, 1.5, -20.0),
    "recovery": MacroScenario("Strong Recovery", 3.5, 6.5, 2.5, 3.0, 7.0, 20.0),
}


def _ar1(n: int, phi: float, mu: float, sigma: float, x0: float, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = mu * (1 - phi) + phi * x[t - 1] + sigma * rng.standard_normal()
    return x


def load_ecb_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 100
    idx = pd.date_range("2000-01-01", periods=n, freq="QE")
    t = np.arange(n)

    # GDP growth (%) — mean-reverting with crisis shocks
    gdp = _ar1(n, 0.6, 1.8, 0.8, 2.0, rng)
    # GFC 2008 Q3-Q4 (t=34-35), 2009 Q1-Q3 (t=36-38)
    gdp[34:40] -= np.array([2.0, 5.0, 6.5, 5.5, 3.0, 1.5])
    # Euro debt crisis 2012 Q1-Q4
    gdp[48:52] -= np.array([1.5, 2.0, 2.0, 1.2])
    # COVID 2020 Q1-Q2 (t=80-81)
    gdp[80:83] -= np.array([3.0, 8.0, 2.0])

    # Unemployment (%)
    ue = _ar1(n, 0.92, 8.0, 0.25, 7.5, rng)
    ue[35:45] += np.linspace(0, 2.5, 10)
    ue[45:60] += np.linspace(2.5, 1.5, 15)
    ue[60:75] -= np.linspace(0, 1.5, 15)
    ue[80:85] += np.array([0.5, 1.5, 2.0, 1.8, 1.0])

    # HICP inflation (%)
    hicp = _ar1(n, 0.8, 2.0, 0.4, 2.2, rng)
    hicp[88:96] += np.array([1.0, 2.5, 4.0, 6.5, 8.5, 9.5, 8.0, 5.5])  # 2022 inflation

    # ECB policy rate (%) — non-negative, follows hiking/cutting cycles
    policy_rate = np.zeros(n)
    policy_rate[:32] = np.interp(t[:32], [0, 28, 32], [3.5, 4.25, 1.5])
    policy_rate[32:40] = np.interp(t[32:40], [32, 40], [1.5, 0.25])
    policy_rate[40:68] = np.interp(t[40:68], [40, 68], [0.25, 0.0])
    policy_rate[68:76] = 0.0
    policy_rate[76:88] = np.interp(t[76:88], [76, 88], [0.0, 0.25])
    policy_rate[88:] = np.interp(t[88:], [88, 98], [0.25, 4.5])
    policy_rate += rng.normal(0, 0.05, n)
    policy_rate = np.clip(policy_rate, 0.0, None)

    # Credit growth (% YoY)
    credit_growth = _ar1(n, 0.75, 3.5, 1.2, 4.0, rng)
    credit_growth[30:36] += np.linspace(0, 3.0, 6)   # pre-crisis boom
    credit_growth[36:46] -= np.linspace(0, 7.0, 10)   # deleveraging
    credit_growth[80:84] -= np.array([2.0, 5.0, 4.0, 2.0])  # COVID

    # Lending standards (diffusion index, positive = tighter)
    lending_std = _ar1(n, 0.7, 0.0, 8.0, 5.0, rng)
    lending_std[34:44] += np.linspace(0, 40, 10)
    lending_std[44:60] -= np.linspace(0, 35, 16)
    lending_std[80:84] += np.array([15, 35, 30, 15])

    # NPL ratio (%) — derived from macro with lag
    npl = 3.5 - 0.25 * gdp + 0.15 * ue + 0.05 * np.abs(credit_growth - 3)
    npl += 0.05 * lending_std
    npl = _ar1(n, 0.85, 3.5, 0.3, 3.5, rng) + np.clip(npl - npl.mean(), -3, 5)
    npl = np.clip(npl, 1.0, 20.0)

    # 10Y–2Y spread (proxy for lending spread, %)
    spread = _ar1(n, 0.8, 1.5, 0.4, 1.0, rng)
    spread[34:42] += np.array([0.5, 1.0, 2.0, 2.5, 2.0, 1.5, 1.0, 0.5])
    spread[48:56] += np.array([0.5, 1.2, 2.0, 2.8, 2.5, 2.0, 1.5, 0.8])

    df = pd.DataFrame({
        "gdp_growth": gdp,
        "unemployment": ue,
        "inflation": hicp,
        "policy_rate": policy_rate,
        "credit_growth": credit_growth,
        "lending_standards": lending_std,
        "npl_ratio": npl,
        "spread": spread,
    }, index=idx)

    df.index.name = "date"
    return df.round(4)
