"""Ornstein-Uhlenbeck / Vasicek and CIR path simulators.

OU/Vasicek:  dx = κ(θ - x) dt + σ dW
CIR:         dx = κ(θ - x) dt + σ√x dW

Correlated multi-factor simulation uses Cholesky decomposition.
"""
from __future__ import annotations

import numpy as np

DT = 0.25  # quarterly step


def simulate_ou_paths(
    x0: float, kappa: float, theta: float, sigma: float,
    horizon_q: int, n_paths: int, rng: np.random.Generator,
) -> np.ndarray:
    """OU/Vasicek paths, shape (n_paths, horizon_q)."""
    x = np.empty((n_paths, horizon_q))
    x_prev = np.full(n_paths, x0)
    for t in range(horizon_q):
        drift = x_prev + kappa * DT * (theta - x_prev)
        vol = sigma * np.sqrt(DT)
        x_prev = drift + vol * rng.standard_normal(n_paths)
        x[:, t] = x_prev
    return x


def simulate_cir_paths(
    x0: float, kappa: float, theta: float, sigma: float,
    horizon_q: int, n_paths: int, rng: np.random.Generator,
) -> np.ndarray:
    """CIR paths, shape (n_paths, horizon_q). Reflects at 0."""
    x = np.empty((n_paths, horizon_q))
    x_prev = np.full(n_paths, max(x0, 1e-4))
    for t in range(horizon_q):
        drift = x_prev + kappa * DT * (theta - x_prev)
        vol = sigma * np.sqrt(np.maximum(x_prev, 1e-6) * DT)
        x_prev = np.maximum(drift + vol * rng.standard_normal(n_paths), 1e-4)
        x[:, t] = x_prev
    return x


def simulate_correlated_factors(
    factor_specs: list[dict],
    corr: np.ndarray,
    horizon_q: int,
    n_paths: int,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate multiple correlated macro factors.

    Each spec dict: {name, process('OU'|'CIR'), x0, kappa, theta, sigma}.
    Returns dict name → (n_paths, horizon_q).
    """
    rng = np.random.default_rng(seed)
    n = len(factor_specs)
    L = np.linalg.cholesky(np.clip(corr, -0.999, 0.999) + 1e-8 * np.eye(n))

    paths = {s["name"]: np.empty((n_paths, horizon_q)) for s in factor_specs}
    x_prev = {s["name"]: np.full(n_paths, s["x0"]) for s in factor_specs}

    for t in range(horizon_q):
        Z = rng.standard_normal((n_paths, n)) @ L.T
        for j, s in enumerate(factor_specs):
            x = x_prev[s["name"]]
            kdt = s["kappa"] * DT
            drift = x + kdt * (s["theta"] - x)
            if s["process"] == "CIR":
                vol = s["sigma"] * np.sqrt(np.maximum(x, 1e-6) * DT)
                x_new = np.maximum(drift + vol * Z[:, j], 1e-4)
            else:
                vol = s["sigma"] * np.sqrt(DT)
                x_new = drift + vol * Z[:, j]
            paths[s["name"]][:, t] = x_new
            x_prev[s["name"]] = x_new

    return paths
