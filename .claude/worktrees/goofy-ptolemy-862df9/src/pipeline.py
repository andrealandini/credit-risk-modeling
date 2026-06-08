"""End-to-end orchestrator. Caches the model so the dashboard is snappy."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import SDEParams, calibrate_all
from .data_fetcher import DataBundle, fetch_ecb
from .monte_carlo import MCResult, PortfolioSpec, run_monte_carlo, stress_scenario
from .pd_model import PDModel, fit_pd_model


@dataclass
class CreditRiskModel:
    bundle: DataBundle
    sde_params: dict[str, SDEParams]
    corr: np.ndarray
    factor_names: list[str]
    pd_model: PDModel
    portfolio: PortfolioSpec = field(default_factory=PortfolioSpec)
    last_mc: MCResult | None = None

    @property
    def panel(self) -> pd.DataFrame:
        return self.bundle.panel

    def run(self, horizon_q: int = 4, n_paths: int = 10_000,
            portfolio: PortfolioSpec | None = None) -> MCResult:
        port = portfolio or self.portfolio
        self.last_mc = run_monte_carlo(self.panel, self.sde_params, self.corr,
                                        self.factor_names, self.pd_model, port,
                                        horizon_q=horizon_q, n_paths=n_paths)
        return self.last_mc

    def stress(self, shocks: dict[str, float], horizon_q: int = 4) -> dict:
        return stress_scenario(self.sde_params, self.factor_names, self.pd_model,
                                self.portfolio, shocks, horizon_q=horizon_q,
                                corr=self.corr)


def build_model() -> CreditRiskModel:
    bundle = fetch_ecb()
    sde_params, corr, names = calibrate_all(bundle.panel)
    pd_model = fit_pd_model(bundle.panel)
    return CreditRiskModel(bundle=bundle, sde_params=sde_params, corr=corr,
                            factor_names=names, pd_model=pd_model)
