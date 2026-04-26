"""Geometric Brownian Motion simulator.

dV = μ V dt + σ V dW

Used for Merton model asset-value simulation and stress testing of
firm asset value processes.
"""
from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    V0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Return shape (n_paths, n_steps+1) array of asset value paths."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * dW
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    return V0 * np.exp(log_paths)
