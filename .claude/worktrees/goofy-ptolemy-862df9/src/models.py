"""Closed-form / analytical credit-risk models.

Each function returns a dict with the headline number plus a list of
human-readable log lines so the dashboard's terminal can display
parameter-by-parameter trace of the calculation.
"""
from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm


def _log(lines: list[str], msg: str) -> None:
    lines.append(msg)


def simple_el(pd: float, lgd: float, ead: float) -> dict:
    """Single-obligor EL = PD * LGD * EAD."""
    log_: list[str] = []
    _log(log_, "── Simple Expected Loss ──")
    _log(log_, f"  PD  = {pd:.4%}")
    _log(log_, f"  LGD = {lgd:.2%}")
    _log(log_, f"  EAD = €{ead:,.2f}")
    el = pd * lgd * ead
    _log(log_, f"  EL  = PD × LGD × EAD = €{el:,.2f}")
    return {"el": el, "log": log_}


def portfolio_el(pd_arr: list[float], lgd_arr: list[float],
                  ead_arr: list[float]) -> dict:
    """Sum of per-obligor EL with optional unexpected-loss approximation."""
    log_: list[str] = []
    _log(log_, "── Portfolio Expected Loss ──")
    pd = np.array(pd_arr, dtype=float)
    lgd = np.array(lgd_arr, dtype=float)
    ead = np.array(ead_arr, dtype=float)
    n = len(pd)
    el_i = pd * lgd * ead
    el = float(el_i.sum())
    notional = float(ead.sum())
    weighted_pd = float((pd * ead).sum() / notional) if notional else 0.0
    weighted_lgd = float((lgd * ead).sum() / notional) if notional else 0.0
    ul_i = ead * lgd * np.sqrt(pd * (1 - pd))
    ul_ind = float(np.sqrt((ul_i ** 2).sum()))  # independence assumption
    _log(log_, f"  Obligors      : {n}")
    _log(log_, f"  Notional      : €{notional:,.2f}")
    _log(log_, f"  Weighted PD   : {weighted_pd:.4%}")
    _log(log_, f"  Weighted LGD  : {weighted_lgd:.2%}")
    _log(log_, f"  Σ EL_i        : €{el:,.2f}")
    _log(log_, f"  Σ √Σ UL_i²    : €{ul_ind:,.2f}  (independent)")
    return {"el": el, "ul_independent": ul_ind, "notional": notional,
            "weighted_pd": weighted_pd, "weighted_lgd": weighted_lgd, "log": log_}


def basel_irb(pd: float, lgd: float, ead: float, maturity: float = 2.5,
               asset_class: str = "corporate") -> dict:
    """Basel II/III IRB risk-weight formula. Returns regulatory capital K,
    risk-weighted assets RWA, and the implied capital requirement at 99.9%."""
    log_: list[str] = []
    pd = max(pd, 3e-4)  # PD floor
    if asset_class == "corporate":
        r_lo, r_hi, k_decay = 0.12, 0.24, 50
    elif asset_class == "retail_mortgage":
        r_lo, r_hi, k_decay = 0.15, 0.15, 0  # fixed 15%
    else:  # retail_other
        r_lo, r_hi, k_decay = 0.03, 0.16, 35

    if k_decay > 0:
        w = (1 - exp(-k_decay * pd)) / (1 - exp(-k_decay))
        R = r_lo * (1 - w) + r_hi * w
    else:
        R = r_lo

    b = (0.11852 - 0.05478 * log(pd)) ** 2
    ma = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

    inner = norm.ppf(pd) / sqrt(1 - R) + sqrt(R / (1 - R)) * norm.ppf(0.999)
    cond_pd = norm.cdf(inner)
    k_unscaled = lgd * (cond_pd - pd)
    k = k_unscaled * ma
    rwa = k * 12.5 * ead
    capital = k * ead
    el_reg = pd * lgd * ead

    _log(log_, f"── Basel IRB ({asset_class}) ──")
    _log(log_, f"  PD            = {pd:.4%}")
    _log(log_, f"  LGD           = {lgd:.2%}")
    _log(log_, f"  EAD           = €{ead:,.2f}")
    _log(log_, f"  M (years)     = {maturity:.2f}")
    _log(log_, f"  Asset corr R  = {R:.4f}")
    _log(log_, f"  b(PD)         = {b:.4f}")
    _log(log_, f"  Maturity adj  = {ma:.4f}")
    _log(log_, f"  Cond. PD@99.9 = {cond_pd:.4%}")
    _log(log_, f"  K (capital %) = {k:.4%}")
    _log(log_, f"  RWA           = €{rwa:,.2f}")
    _log(log_, f"  Capital       = €{capital:,.2f}")
    _log(log_, f"  Reg. EL       = €{el_reg:,.2f}")
    return {"k": k, "rwa": rwa, "capital": capital, "el": el_reg,
            "asset_corr": R, "maturity_adj": ma, "log": log_}


def merton_structural(asset_value: float, debt: float, asset_vol: float,
                       risk_free: float, T: float, lgd: float = 0.6) -> dict:
    """Merton 1974 structural model. Default if V_T < D."""
    log_: list[str] = []
    if asset_value <= 0 or debt <= 0 or asset_vol <= 0 or T <= 0:
        return {"error": "all parameters must be positive", "log": ["invalid input"]}
    d1 = (log(asset_value / debt) + (risk_free + 0.5 * asset_vol ** 2) * T) \
         / (asset_vol * sqrt(T))
    d2 = d1 - asset_vol * sqrt(T)
    pd = float(norm.cdf(-d2))
    dd = float(d2)
    el = pd * lgd * debt
    _log(log_, "── Merton structural model ──")
    _log(log_, f"  V₀ (asset value)     = €{asset_value:,.2f}")
    _log(log_, f"  D  (debt face value) = €{debt:,.2f}")
    _log(log_, f"  σ_V (asset vol)      = {asset_vol:.2%}")
    _log(log_, f"  r  (risk-free)       = {risk_free:.2%}")
    _log(log_, f"  T                    = {T:.2f}y")
    _log(log_, f"  d1                   = {d1:.4f}")
    _log(log_, f"  d2 (= -DD)           = {d2:.4f}")
    _log(log_, f"  Distance to default  = {dd:.4f}")
    _log(log_, f"  PD = N(-d₂)          = {pd:.4%}")
    _log(log_, f"  EL (assuming LGD={lgd:.0%}) = €{el:,.2f}")
    return {"pd": pd, "dd": dd, "d1": d1, "d2": d2, "el": el, "log": log_}


def cir_intensity_pd(lambda0: float, kappa: float, theta: float,
                      sigma: float, T: float, lgd: float = 0.6,
                      ead: float = 1_000_000.0) -> dict:
    """Reduced-form intensity model with CIR default intensity.
    Closed-form survival probability from CIR bond pricing analogy."""
    log_: list[str] = []
    if any(x <= 0 for x in [theta, sigma, T]) or kappa <= 0:
        return {"error": "kappa, theta, sigma, T must be positive", "log": ["invalid input"]}
    gamma = sqrt(kappa ** 2 + 2 * sigma ** 2)
    denom = (gamma + kappa) * (exp(gamma * T) - 1) + 2 * gamma
    B = 2 * (exp(gamma * T) - 1) / denom
    A = (2 * gamma * exp((kappa + gamma) * T / 2) / denom) ** (2 * kappa * theta / sigma ** 2)
    survival = A * exp(-B * lambda0)
    pd = float(1 - survival)
    el = pd * lgd * ead
    _log(log_, "── CIR intensity (reduced-form) ──")
    _log(log_, f"  λ₀     = {lambda0:.4f}")
    _log(log_, f"  κ      = {kappa:.4f}")
    _log(log_, f"  θ      = {theta:.4f}")
    _log(log_, f"  σ      = {sigma:.4f}")
    _log(log_, f"  T      = {T:.2f}y")
    _log(log_, f"  γ      = √(κ² + 2σ²) = {gamma:.4f}")
    _log(log_, f"  A(T)   = {A:.4f}")
    _log(log_, f"  B(T)   = {B:.4f}")
    _log(log_, f"  S(T)   = A·exp(-B·λ₀) = {survival:.4%}")
    _log(log_, f"  PD     = 1 - S(T) = {pd:.4%}")
    _log(log_, f"  EL     = PD × LGD × EAD = €{el:,.2f}")
    return {"pd": pd, "survival": float(survival), "el": el, "log": log_}


def vasicek_portfolio_loss(pd: float, lgd: float, ead: float,
                             rho: float, alpha: float = 0.999,
                             n_obligors: int = 1) -> dict:
    """Vasicek large-portfolio loss at confidence alpha — closed form.
    For an infinitely fine-grained portfolio this is exact; for finite n
    it is the asymptotic single-factor approximation."""
    log_: list[str] = []
    if not (0 < pd < 1) or not (0 < rho < 1):
        return {"error": "PD and rho must be in (0,1)", "log": ["invalid input"]}
    portfolio_ead = ead * n_obligors
    cond_pd = norm.cdf((norm.ppf(pd) + sqrt(rho) * norm.ppf(alpha)) / sqrt(1 - rho))
    var = lgd * portfolio_ead * cond_pd
    el = pd * lgd * portfolio_ead
    ul = lgd * portfolio_ead * sqrt(pd * (1 - pd))  # standalone UL
    _log(log_, "── Vasicek portfolio loss (closed form) ──")
    _log(log_, f"  PD          = {pd:.4%}")
    _log(log_, f"  LGD         = {lgd:.2%}")
    _log(log_, f"  EAD/obligor = €{ead:,.2f}")
    _log(log_, f"  Obligors    = {n_obligors}")
    _log(log_, f"  ρ           = {rho:.3f}")
    _log(log_, f"  α (conf.)   = {alpha:.4f}")
    _log(log_, f"  Cond. PD(α) = {cond_pd:.4%}")
    _log(log_, f"  EL          = €{el:,.2f}")
    _log(log_, f"  UL (stand-alone) = €{ul:,.2f}")
    _log(log_, f"  VaR(α)      = €{var:,.2f}")
    return {"el": el, "var": float(var), "ul": float(ul),
            "cond_pd": float(cond_pd), "log": log_}
