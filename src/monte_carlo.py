"""Monte Carlo engine: macro SDE simulation + Vasicek single-factor losses.

Pipeline per simulation path:
  1. Simulate correlated macro factors (OU/CIR) over H quarters.
  2. Map each (path, t) state to a scenario PD via the fitted PD model.
  3. Vasicek single-factor: D_i = 1{ sqrt(rho)*M + sqrt(1-rho)*Z_i < Phi^-1(PD) }
  4. Aggregate L = sum_i EAD_i * LGD_i * D_i with stochastic LGD ~ Beta.

Returns the per-path total loss (over the horizon) and the per-quarter
average PD path for diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .calibration import SDEParams
from .pd_model import PDModel

DT = 0.25


@dataclass
class PortfolioSpec:
    n_obligors: int = 500
    ead_mean: float = 1_000_000.0   # euro per obligor
    ead_cv: float = 0.5
    lgd_mean: float = 0.45
    lgd_concentration: float = 8.0  # Beta concentration parameter
    rho: float = 0.15               # asset correlation


@dataclass
class MCResult:
    losses: np.ndarray            # shape (n_paths,)
    pd_paths: np.ndarray          # shape (n_paths, horizon)
    macro_paths: dict[str, np.ndarray]  # name -> (n_paths, horizon)
    el: float
    ul: float
    var_99: float
    var_999: float
    es_975: float
    horizon_q: int
    portfolio_notional: float


def simulate_macro(params: dict[str, SDEParams], corr: np.ndarray,
                   names: list[str], n_paths: int, horizon_q: int,
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    L = np.linalg.cholesky(corr)
    n = len(names)
    paths = {name: np.zeros((n_paths, horizon_q + 1)) for name in names}
    for name in names:
        paths[name][:, 0] = params[name].x0

    for t in range(1, horizon_q + 1):
        Z = rng.standard_normal(size=(n_paths, n))
        E = Z @ L.T
        for j, name in enumerate(names):
            p = params[name]
            x_prev = paths[name][:, t - 1]
            kdt = p.kappa * DT
            mean = x_prev + kdt * (p.theta - x_prev)
            if p.process == "OU":
                vol = p.sigma * np.sqrt(DT)
            else:  # CIR
                vol = p.sigma * np.sqrt(np.maximum(x_prev, 1e-6) * DT)
            x_next = mean + vol * E[:, j]
            if p.process == "CIR":
                x_next = np.maximum(x_next, 1e-4)
            paths[name][:, t] = x_next
    return {k: v[:, 1:] for k, v in paths.items()}


def _scenario_pd(macro_paths: dict[str, np.ndarray], pd_model: PDModel) -> np.ndarray:
    z = np.full_like(next(iter(macro_paths.values())), pd_model.intercept, dtype=float)
    for feat, beta in pd_model.coefs.items():
        if feat in macro_paths:
            z = z + beta * macro_paths[feat]
    return 1.0 / (1.0 + np.exp(-z))


def run_monte_carlo(panel: pd.DataFrame,
                    sde_params: dict[str, SDEParams],
                    corr: np.ndarray,
                    names: list[str],
                    pd_model: PDModel,
                    portfolio: PortfolioSpec,
                    horizon_q: int = 4,
                    n_paths: int = 10_000,
                    seed: int = 7) -> MCResult:
    rng = np.random.default_rng(seed)
    macro = simulate_macro(sde_params, corr, names, n_paths, horizon_q, rng)
    pd_paths = _scenario_pd(macro, pd_model)            # (n_paths, H)
    pd_paths = np.clip(pd_paths, 1e-6, 0.5)

    # EAD across obligors (lognormal) — fixed across paths for variance reduction
    sigma_log = np.sqrt(np.log(1 + portfolio.ead_cv ** 2))
    mu_log = np.log(portfolio.ead_mean) - 0.5 * sigma_log ** 2
    ead = rng.lognormal(mean=mu_log, sigma=sigma_log, size=portfolio.n_obligors)
    notional = float(ead.sum())

    # Vasicek single-factor: per-path systematic factor M and idiosyncratic Z
    sqrt_rho = np.sqrt(portfolio.rho)
    sqrt_1mr = np.sqrt(1 - portfolio.rho)

    losses = np.zeros(n_paths)
    for t in range(horizon_q):
        pd_t = pd_paths[:, t]                     # (n_paths,)
        thresh = norm.ppf(pd_t)                   # (n_paths,)
        M = rng.standard_normal(n_paths)
        # default rate per path under the systematic factor (closed form)
        cond_pd = norm.cdf((thresh - sqrt_rho * M) / sqrt_1mr)  # (n_paths,)
        # draw stochastic LGD per path (Beta with concentration around lgd_mean,
        # bumped up in stressed states where pd_t is high)
        stress = np.clip(pd_t / 0.05, 0.5, 2.0)
        mean_lgd = np.clip(portfolio.lgd_mean * stress, 0.1, 0.95)
        c = portfolio.lgd_concentration
        a = mean_lgd * c
        b = (1 - mean_lgd) * c
        lgd_t = rng.beta(a, b)
        # expected loss for the period over the portfolio under cond_pd
        loss_t = notional * cond_pd * lgd_t
        losses += loss_t

    el = float(losses.mean())
    ul = float(losses.std(ddof=1))
    var_99 = float(np.quantile(losses, 0.99))
    var_999 = float(np.quantile(losses, 0.999))
    tail = losses[losses >= np.quantile(losses, 0.975)]
    es_975 = float(tail.mean()) if len(tail) else var_99

    return MCResult(losses=losses, pd_paths=pd_paths, macro_paths=macro,
                    el=el, ul=ul, var_99=var_99, var_999=var_999, es_975=es_975,
                    horizon_q=horizon_q, portfolio_notional=notional)


def stress_scenario(sde_params: dict[str, SDEParams], names: list[str],
                    pd_model: PDModel, portfolio: PortfolioSpec,
                    shocks: dict[str, float], horizon_q: int = 4,
                    n_paths: int = 5_000, corr: np.ndarray | None = None,
                    seed: int = 11) -> dict:
    """Apply additive shocks to the long-run mean theta, simulate, and report
    the conditional expected loss."""
    rng = np.random.default_rng(seed)
    shocked = {}
    for n, p in sde_params.items():
        new_theta = p.theta + shocks.get(n, 0.0)
        shocked[n] = SDEParams(name=p.name, process=p.process, kappa=p.kappa,
                                theta=new_theta, sigma=p.sigma, x0=p.x0,
                                residuals=p.residuals)
    if corr is None:
        corr = np.eye(len(names))
    macro = simulate_macro(shocked, corr, names, n_paths, horizon_q, rng)
    pd_paths = np.clip(_scenario_pd(macro, pd_model), 1e-6, 0.5)
    avg_pd = float(pd_paths.mean())
    el = portfolio.ead_mean * portfolio.n_obligors * portfolio.lgd_mean * avg_pd * horizon_q
    return {"avg_pd": avg_pd, "expected_loss": el}
