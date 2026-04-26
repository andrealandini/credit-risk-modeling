"""Extended continuous-time processes for the SDE layer.

Each process implements `calibrate(series, dt)` and `simulate(x0, n_paths,
horizon, dt, rng)`. All return a uniform dict with parameters + log lines
so the dashboard can display the calibration trace.

Variants
--------
* OU                — standard Ornstein-Uhlenbeck
* OU + jumps        — OU with compound Poisson Gaussian jumps (Merton-style)
* CIR               — standard Cox-Ingersoll-Ross
* CIR + jumps       — CIR with compound Poisson Gaussian jumps
* Hull-White        — OU with deterministic time-varying θ(t) (piecewise mean)
* Two-factor Vasicek — stochastic mean-reversion level (G2++ skeleton)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProcessFit:
    name: str
    params: dict
    log: list[str]
    sample_paths: dict        # {"qtrs": [...], "mean": [...], "p05": [...], "p95": [...]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ar1(series: pd.Series):
    s = series.dropna()
    x_t = s.shift(1).dropna()
    x_tp1 = s.loc[x_t.index]
    X = np.column_stack([np.ones(len(x_t)), x_t.values])
    coef, *_ = np.linalg.lstsq(X, x_tp1.values, rcond=None)
    alpha, beta = coef
    resid = x_tp1.values - X @ coef
    return float(alpha), float(beta), resid, x_t.values, x_tp1.values


def _ou_from_ar1(alpha, beta, resid, dt):
    beta = min(max(beta, 1e-6), 0.9999)
    kappa = -np.log(beta) / dt
    theta = alpha / (1 - beta)
    sigma = float(np.sqrt(resid.var(ddof=2) * 2 * kappa / (1 - beta ** 2)))
    return float(kappa), float(theta), sigma


def _summarize_paths(paths: np.ndarray) -> dict:
    H = paths.shape[1]
    return {
        "qtrs": list(range(1, H + 1)),
        "mean": paths.mean(axis=0).tolist(),
        "p05": np.quantile(paths, 0.05, axis=0).tolist(),
        "p95": np.quantile(paths, 0.95, axis=0).tolist(),
    }


def _simulate_ou(x0, kappa, theta, sigma, n_paths, horizon, dt, rng):
    paths = np.zeros((n_paths, horizon))
    x = np.full(n_paths, x0)
    for t in range(horizon):
        x = x + kappa * (theta - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        paths[:, t] = x
    return paths


def _simulate_cir(x0, kappa, theta, sigma, n_paths, horizon, dt, rng):
    paths = np.zeros((n_paths, horizon))
    x = np.full(n_paths, max(x0, 1e-4))
    for t in range(horizon):
        x = np.maximum(x + kappa * (theta - x) * dt
                        + sigma * np.sqrt(np.maximum(x, 1e-6) * dt) * rng.standard_normal(n_paths),
                        1e-4)
        paths[:, t] = x
    return paths


def _add_jumps(paths, lam, mu_j, sigma_j, dt, rng):
    """Compound Poisson Gaussian jumps applied path-by-path."""
    n_paths, H = paths.shape
    p_jump = lam * dt
    n_jumps = rng.poisson(lam=p_jump, size=(n_paths, H))
    jump_size = rng.normal(mu_j, sigma_j, size=(n_paths, H)) * (n_jumps > 0)
    return paths + np.cumsum(jump_size, axis=1)


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------

def calibrate_ou(series: pd.Series, dt: float = 0.25,
                  horizon: int = 8, n_paths: int = 2000) -> ProcessFit:
    alpha, beta, resid, *_ = _ar1(series)
    kappa, theta, sigma = _ou_from_ar1(alpha, beta, resid, dt)
    rng = np.random.default_rng(7)
    paths = _simulate_ou(series.dropna().iloc[-1], kappa, theta, sigma,
                          n_paths, horizon, dt, rng)
    log = ["── OU calibration ──",
            f"  AR(1)     : x_t+1 = {alpha:+.4f} + {beta:+.4f}·x_t + ε",
            f"  κ         = {kappa:.4f}",
            f"  θ         = {theta:.4f}",
            f"  σ         = {sigma:.4f}",
            f"  half-life = {np.log(2)/kappa:.2f}y" if kappa > 0 else "  half-life = ∞"]
    return ProcessFit(name="OU",
                       params={"kappa": kappa, "theta": theta, "sigma": sigma},
                       log=log, sample_paths=_summarize_paths(paths))


def calibrate_ou_jump(series: pd.Series, dt: float = 0.25, jump_threshold: float = 3.0,
                       horizon: int = 8, n_paths: int = 2000) -> ProcessFit:
    """OU with compound Poisson Gaussian jumps. Jumps detected as residuals
    beyond `jump_threshold` × MAD."""
    alpha, beta, resid, *_ = _ar1(series)
    mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
    is_jump = np.abs(resid - np.median(resid)) > jump_threshold * mad
    diff_resid = resid[~is_jump]
    kappa, theta, sigma = _ou_from_ar1(
        alpha, beta, diff_resid if len(diff_resid) > 5 else resid, dt)
    n_jumps = int(is_jump.sum())
    lam = n_jumps / (len(resid) * dt) if len(resid) else 0.0
    if n_jumps:
        mu_j, sigma_j = float(resid[is_jump].mean()), float(resid[is_jump].std(ddof=1) or 0.0)
    else:
        mu_j, sigma_j = 0.0, 0.0
    rng = np.random.default_rng(11)
    base = _simulate_ou(series.dropna().iloc[-1], kappa, theta, sigma, n_paths, horizon, dt, rng)
    paths = _add_jumps(base, lam, mu_j, sigma_j, dt, rng)
    log = ["── OU + jumps calibration ──",
            f"  diffusion κ/θ/σ = {kappa:.4f} / {theta:.4f} / {sigma:.4f}",
            f"  jump count      = {n_jumps} (threshold {jump_threshold}·MAD)",
            f"  jump intensity λ = {lam:.4f}/y",
            f"  jump μ          = {mu_j:+.4f}",
            f"  jump σ          = {sigma_j:.4f}"]
    return ProcessFit(name="OU+Jumps",
                       params={"kappa": kappa, "theta": theta, "sigma": sigma,
                                "lambda": lam, "mu_j": mu_j, "sigma_j": sigma_j},
                       log=log, sample_paths=_summarize_paths(paths))


def calibrate_cir(series: pd.Series, dt: float = 0.25,
                   horizon: int = 8, n_paths: int = 2000) -> ProcessFit:
    s = series.dropna().clip(lower=1e-4)
    x_t = s.shift(1).dropna()
    x_tp1 = s.loc[x_t.index]
    w = 1.0 / np.sqrt(x_t.values)
    X = np.column_stack([np.ones(len(x_t)), x_t.values]) * w[:, None]
    y = x_tp1.values * w
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    beta = min(max(beta, 1e-6), 0.9999)
    kappa = -np.log(beta) / dt
    theta = alpha / (1 - beta)
    resid = x_tp1.values - (alpha + beta * x_t.values)
    sigma = float(np.sqrt(np.mean(resid ** 2 / x_t.values) / dt))
    feller = 2 * kappa * theta - sigma ** 2
    rng = np.random.default_rng(13)
    paths = _simulate_cir(s.iloc[-1], kappa, theta, sigma, n_paths, horizon, dt, rng)
    log = ["── CIR calibration ──",
            f"  κ          = {kappa:.4f}",
            f"  θ          = {theta:.4f}",
            f"  σ          = {sigma:.4f}",
            f"  Feller 2κθ-σ² = {feller:+.4f}  ({'ok' if feller > 0 else 'violated'})"]
    return ProcessFit(name="CIR",
                       params={"kappa": float(kappa), "theta": float(theta),
                                "sigma": sigma, "feller": float(feller)},
                       log=log, sample_paths=_summarize_paths(paths))


def calibrate_cir_jump(series: pd.Series, dt: float = 0.25, jump_threshold: float = 3.0,
                        horizon: int = 8, n_paths: int = 2000) -> ProcessFit:
    base = calibrate_cir(series, dt=dt, horizon=horizon, n_paths=n_paths)
    s = series.dropna().clip(lower=1e-4)
    x_t = s.shift(1).dropna().values
    x_tp1 = s.loc[s.shift(1).dropna().index].values
    pred = base.params["theta"] + (x_t - base.params["theta"]) * np.exp(-base.params["kappa"] * dt)
    std_resid = (x_tp1 - pred) / (np.sqrt(x_t) * base.params["sigma"] * np.sqrt(dt) + 1e-9)
    is_jump = np.abs(std_resid) > jump_threshold
    n_jumps = int(is_jump.sum())
    lam = n_jumps / (len(x_t) * dt) if len(x_t) else 0.0
    raw_jumps = (x_tp1 - pred)[is_jump]
    mu_j = float(raw_jumps.mean()) if n_jumps else 0.0
    sigma_j = float(raw_jumps.std(ddof=1)) if n_jumps > 1 else 0.0
    rng = np.random.default_rng(17)
    diff = _simulate_cir(s.iloc[-1], base.params["kappa"], base.params["theta"],
                          base.params["sigma"], n_paths, horizon, dt, rng)
    paths = _add_jumps(diff, lam, mu_j, sigma_j, dt, rng)
    paths = np.maximum(paths, 1e-4)
    log = ["── CIR + jumps calibration ──", *base.log[1:],
            f"  jump count      = {n_jumps}",
            f"  jump intensity λ = {lam:.4f}/y",
            f"  jump μ / σ      = {mu_j:+.4f} / {sigma_j:.4f}"]
    return ProcessFit(name="CIR+Jumps",
                       params={**base.params, "lambda": lam, "mu_j": mu_j, "sigma_j": sigma_j},
                       log=log, sample_paths=_summarize_paths(paths))


def calibrate_hull_white(series: pd.Series, dt: float = 0.25, window: int = 8,
                          horizon: int = 8, n_paths: int = 2000) -> ProcessFit:
    """Hull-White extended Vasicek with rolling-window θ(t).
    θ_t estimated as a rolling-window mean; κ, σ from OU on de-trended series."""
    s = series.dropna()
    theta_t = s.rolling(window=window, min_periods=1).mean()
    detrended = s - theta_t
    alpha, beta, resid, *_ = _ar1(detrended)
    beta = min(max(beta, 1e-6), 0.9999)
    kappa = float(-np.log(beta) / dt)
    sigma = float(np.sqrt(resid.var(ddof=2) * 2 * kappa / (1 - beta ** 2)))
    theta_path = theta_t.iloc[-horizon:].tolist() if len(theta_t) >= horizon else \
                  [float(theta_t.iloc[-1])] * horizon
    if len(theta_path) < horizon:
        theta_path = theta_path + [theta_path[-1]] * (horizon - len(theta_path))
    rng = np.random.default_rng(19)
    paths = np.zeros((n_paths, horizon))
    x = np.full(n_paths, s.iloc[-1])
    for t in range(horizon):
        x = x + kappa * (theta_path[t] - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
        paths[:, t] = x
    log = ["── Hull-White (extended Vasicek) calibration ──",
            f"  κ           = {kappa:.4f}",
            f"  σ           = {sigma:.4f}",
            f"  θ(t) window = {window}q rolling mean",
            f"  θ̄ recent    = {np.mean(theta_path):+.4f}"]
    return ProcessFit(name="Hull-White",
                       params={"kappa": kappa, "sigma": sigma, "theta_path": theta_path,
                                "theta_window": window},
                       log=log, sample_paths=_summarize_paths(paths))


def calibrate_two_factor_vasicek(series: pd.Series, dt: float = 0.25,
                                   horizon: int = 8, n_paths: int = 2000) -> ProcessFit:
    """Two-factor Gaussian (G2++ skeleton): r_t = x_t + y_t with two independent
    OU components calibrated on level (slow) and first-difference (fast) parts."""
    s = series.dropna()
    slow = s.rolling(8, min_periods=1).mean()
    fast = s - slow
    a_s, b_s, r_s, *_ = _ar1(slow)
    a_f, b_f, r_f, *_ = _ar1(fast)
    k1, t1, sg1 = _ou_from_ar1(a_s, b_s, r_s, dt)
    k2, t2, sg2 = _ou_from_ar1(a_f, b_f, r_f, dt)
    rng = np.random.default_rng(23)
    p1 = _simulate_ou(slow.iloc[-1], k1, t1, sg1, n_paths, horizon, dt, rng)
    p2 = _simulate_ou(fast.iloc[-1], k2, t2, sg2, n_paths, horizon, dt, rng)
    paths = p1 + p2
    log = ["── Two-factor Vasicek calibration ──",
            f"  slow factor  : κ={k1:.4f}  θ={t1:.4f}  σ={sg1:.4f}",
            f"  fast factor  : κ={k2:.4f}  θ={t2:.4f}  σ={sg2:.4f}",
            f"  half-lives   : {np.log(2)/k1:.2f}y  /  {np.log(2)/k2:.2f}y" if k1 > 0 and k2 > 0 else "  half-lives undefined"]
    return ProcessFit(name="Two-factor Vasicek",
                       params={"slow": {"kappa": k1, "theta": t1, "sigma": sg1},
                                "fast": {"kappa": k2, "theta": t2, "sigma": sg2}},
                       log=log, sample_paths=_summarize_paths(paths))


PROCESSES = {
    "OU":               calibrate_ou,
    "OU+Jumps":         calibrate_ou_jump,
    "CIR":              calibrate_cir,
    "CIR+Jumps":        calibrate_cir_jump,
    "Hull-White":       calibrate_hull_white,
    "Two-factor Vasicek": calibrate_two_factor_vasicek,
}


def calibrate(series: pd.Series, process: str, **kwargs) -> ProcessFit:
    if process not in PROCESSES:
        raise ValueError(f"unknown process {process}; choose from {list(PROCESSES)}")
    return PROCESSES[process](series, **kwargs)
