"""Monte Carlo engine for portfolio credit risk.

Pipeline per simulation:
  1. Simulate correlated macro factors (OU/CIR) via Vasicek module.
  2. Map (path, t) macro state to PD via the selected PD model.
  3. Vasicek single-factor: correlated defaults per path.
  4. Stochastic LGD ~ Beta conditioned on macro state.
  5. Stochastic EAD via chosen EAD model.
  6. Aggregate loss distribution → EL, UL, VaR, ES.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .vasicek import simulate_correlated_factors


@dataclass
class PortfolioSpec:
    n_obligors: int = 200
    ead_mean: float = 1_000_000.0
    ead_cv: float = 0.5
    lgd_mean: float = 0.45
    lgd_concentration: float = 8.0
    rho: float = 0.15
    committed_limit: float = 1_000_000.0
    ccf: float = 0.60


@dataclass
class MCResult:
    losses: np.ndarray
    pd_paths: np.ndarray
    macro_paths: dict[str, np.ndarray]
    el: float
    ul: float
    var_99: float
    var_999: float
    es_975: float
    horizon_q: int
    portfolio_notional: float
    n_paths: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_paths = len(self.losses)

    def as_summary(self) -> dict:
        return {
            "el": self.el,
            "ul": self.ul,
            "var_99": self.var_99,
            "var_999": self.var_999,
            "es_975": self.es_975,
            "notional": self.portfolio_notional,
            "horizon_q": self.horizon_q,
            "n_paths": self.n_paths,
            "el_rate": self.el / self.portfolio_notional if self.portfolio_notional else 0,
        }


class MonteCarloEngine:
    """Pluggable MC engine — supply pd_fn, lgd_fn, ead_fn callables."""

    def __init__(
        self,
        pd_fn,
        lgd_fn,
        ead_fn,
        portfolio: PortfolioSpec | None = None,
        factor_specs: list[dict] | None = None,
        corr: np.ndarray | None = None,
    ):
        self.pd_fn = pd_fn
        self.lgd_fn = lgd_fn
        self.ead_fn = ead_fn
        self.portfolio = portfolio or PortfolioSpec()
        self.factor_specs = factor_specs or _default_factor_specs()
        self.corr = corr if corr is not None else np.eye(len(self.factor_specs))

    def run(
        self,
        horizon_q: int = 4,
        n_paths: int = 5_000,
        seed: int = 42,
        macro_overrides: dict | None = None,
    ) -> MCResult:
        port = self.portfolio
        rng = np.random.default_rng(seed)

        specs = self.factor_specs
        if macro_overrides:
            specs = [
                {**s, "x0": macro_overrides.get(s["name"], s["x0"]),
                 "theta": macro_overrides.get(s["name"] + "_theta", s["theta"])}
                for s in specs
            ]

        macro_paths = simulate_correlated_factors(specs, self.corr, horizon_q, n_paths, seed)

        pd_paths = np.clip(
            self.pd_fn(macro_paths, horizon_q, n_paths), 1e-6, 0.60
        )

        sigma_log = np.sqrt(np.log(1 + port.ead_cv ** 2))
        mu_log = np.log(port.ead_mean) - 0.5 * sigma_log ** 2
        ead_arr = rng.lognormal(mu_log, sigma_log, port.n_obligors)
        notional = float(ead_arr.sum())

        sqrt_rho = np.sqrt(port.rho)
        sqrt_1mr = np.sqrt(1 - port.rho)

        losses = np.zeros(n_paths)
        for t in range(horizon_q):
            pd_t = pd_paths[:, t]
            thresh = norm.ppf(pd_t)
            M = rng.standard_normal(n_paths)
            cond_pd = norm.cdf((thresh - sqrt_rho * M) / sqrt_1mr)

            stress = np.clip(pd_t / 0.05, 0.5, 2.0)
            mean_lgd = np.clip(port.lgd_mean * stress, 0.05, 0.95)
            c = port.lgd_concentration
            lgd_t = rng.beta(mean_lgd * c, (1 - mean_lgd) * c)

            losses += notional * cond_pd * lgd_t

        el = float(losses.mean())
        ul = float(losses.std(ddof=1))
        var_99 = float(np.quantile(losses, 0.99))
        var_999 = float(np.quantile(losses, 0.999))
        tail = losses[losses >= np.quantile(losses, 0.975)]
        es_975 = float(tail.mean()) if len(tail) else var_99

        return MCResult(
            losses=losses,
            pd_paths=pd_paths,
            macro_paths=macro_paths,
            el=el, ul=ul,
            var_99=var_99, var_999=var_999, es_975=es_975,
            horizon_q=horizon_q,
            portfolio_notional=notional,
        )


def _default_factor_specs() -> list[dict]:
    return [
        {"name": "gdp_growth",   "process": "OU",  "x0": 1.5,  "kappa": 0.35, "theta": 1.5,  "sigma": 1.2},
        {"name": "unemployment", "process": "OU",  "x0": 7.5,  "kappa": 0.12, "theta": 7.5,  "sigma": 0.4},
        {"name": "policy_rate",  "process": "CIR", "x0": 2.5,  "kappa": 0.30, "theta": 2.5,  "sigma": 0.6},
        {"name": "credit_growth","process": "OU",  "x0": 3.0,  "kappa": 0.40, "theta": 3.0,  "sigma": 1.5},
    ]


def make_logistic_pd_fn(intercept: float, coefs: dict[str, float]):
    """Build a PD function from logistic regression coefficients."""
    def pd_fn(macro_paths: dict, horizon_q: int, n_paths: int) -> np.ndarray:
        z = np.full((n_paths, horizon_q), intercept)
        for feat, beta in coefs.items():
            if feat in macro_paths:
                z += beta * macro_paths[feat]
        return 1.0 / (1.0 + np.exp(-z))
    return pd_fn


def make_constant_pd_fn(pd_value: float):
    def pd_fn(macro_paths: dict, horizon_q: int, n_paths: int) -> np.ndarray:
        return np.full((n_paths, horizon_q), pd_value)
    return pd_fn
