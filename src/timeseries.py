"""Time-series models for the macro layer.

Wraps statsmodels AR/ARMA and arch GARCH so each fit returns a uniform
dict with parameters, fit statistics, residual diagnostics, and a small
forecast summary. These models describe the *discrete* dynamics; the
continuous-time SDE in src/processes.py is a separate, downstream choice.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class TSFit:
    model: str
    params: dict
    aic: float
    bic: float
    loglik: float
    ljung_box_p: float       # residual autocorrelation (>0.05 = ok)
    arch_lm_p: float | None  # residual heteroskedasticity (>0.05 = ok)
    n_obs: int
    forecast_mean: list[float]
    forecast_std: list[float]
    log: list[str]


def _ljung_box(resid: np.ndarray, lags: int = 8) -> float:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    out = acorr_ljungbox(resid, lags=[lags], return_df=True)
    return float(out["lb_pvalue"].iloc[0])


def _arch_lm(resid: np.ndarray, lags: int = 8) -> float:
    from statsmodels.stats.diagnostic import het_arch
    return float(het_arch(resid, nlags=lags)[1])


def fit_ar(series: pd.Series, p: int = 1, horizon: int = 8) -> TSFit:
    from statsmodels.tsa.ar_model import AutoReg
    s = series.dropna()
    res = AutoReg(s, lags=p, old_names=False).fit()
    fc = res.get_prediction(start=len(s), end=len(s) + horizon - 1)
    fmean = fc.predicted_mean.tolist()
    fstd = np.sqrt(fc.var_pred_mean).tolist() if hasattr(fc, "var_pred_mean") else [float("nan")] * horizon
    log = [f"── AR({p}) fit ──",
            f"  n            = {len(s)}",
            f"  log-lik      = {res.llf:.3f}",
            f"  AIC / BIC    = {res.aic:.2f} / {res.bic:.2f}"]
    for k, v in res.params.items():
        log.append(f"  {k:<12} = {v:+.4f}")
    lb = _ljung_box(res.resid.values)
    log.append(f"  Ljung-Box p  = {lb:.3f}  ({'ok' if lb > 0.05 else 'autocorr left'})")
    return TSFit(model=f"AR({p})", params={k: float(v) for k, v in res.params.items()},
                  aic=float(res.aic), bic=float(res.bic), loglik=float(res.llf),
                  ljung_box_p=lb, arch_lm_p=None, n_obs=len(s),
                  forecast_mean=fmean, forecast_std=fstd, log=log)


def fit_arma(series: pd.Series, p: int = 1, q: int = 1, horizon: int = 8) -> TSFit:
    from statsmodels.tsa.arima.model import ARIMA
    s = series.dropna()
    res = ARIMA(s, order=(p, 0, q)).fit()
    fc = res.get_forecast(steps=horizon)
    fmean = fc.predicted_mean.tolist()
    fstd = np.sqrt(fc.var_pred_mean).tolist()
    log = [f"── ARMA({p},{q}) fit ──",
            f"  n            = {len(s)}",
            f"  log-lik      = {res.llf:.3f}",
            f"  AIC / BIC    = {res.aic:.2f} / {res.bic:.2f}"]
    for k, v in res.params.items():
        log.append(f"  {k:<14} = {v:+.4f}")
    lb = _ljung_box(res.resid.values)
    log.append(f"  Ljung-Box p  = {lb:.3f}  ({'ok' if lb > 0.05 else 'autocorr left'})")
    return TSFit(model=f"ARMA({p},{q})", params={k: float(v) for k, v in res.params.items()},
                  aic=float(res.aic), bic=float(res.bic), loglik=float(res.llf),
                  ljung_box_p=lb, arch_lm_p=None, n_obs=len(s),
                  forecast_mean=fmean, forecast_std=fstd, log=log)


def fit_garch(series: pd.Series, p: int = 1, q: int = 1, horizon: int = 8,
               mean: str = "Constant") -> TSFit:
    """GARCH(p,q) on the de-meaned series. mean ∈ {"Constant", "AR"}."""
    from arch import arch_model
    s = series.dropna()
    am = arch_model(s, mean=mean, lags=1 if mean == "AR" else 0,
                     vol="GARCH", p=p, q=q, dist="normal", rescale=False)
    res = am.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    fmean = fc.mean.values[-1].tolist()
    fstd = np.sqrt(fc.variance.values[-1]).tolist()
    resid = res.resid.dropna().values
    log = [f"── GARCH({p},{q}) [{mean} mean] fit ──",
            f"  n            = {len(s)}",
            f"  log-lik      = {res.loglikelihood:.3f}",
            f"  AIC / BIC    = {res.aic:.2f} / {res.bic:.2f}"]
    for k in res.params.index:
        log.append(f"  {k:<14} = {float(res.params[k]):+.4f}")
    lb = _ljung_box(resid)
    arch_p = _arch_lm(resid)
    log.append(f"  Ljung-Box p  = {lb:.3f}  ({'ok' if lb > 0.05 else 'autocorr left'})")
    log.append(f"  ARCH-LM p    = {arch_p:.3f}  ({'ok' if arch_p > 0.05 else 'het. left'})")
    return TSFit(model=f"GARCH({p},{q})", params={k: float(res.params[k]) for k in res.params.index},
                  aic=float(res.aic), bic=float(res.bic), loglik=float(res.loglikelihood),
                  ljung_box_p=lb, arch_lm_p=arch_p, n_obs=len(s),
                  forecast_mean=fmean, forecast_std=fstd, log=log)


def fit_arma_garch(series: pd.Series, ar_p: int = 1, ma_q: int = 0,
                    gp: int = 1, gq: int = 1, horizon: int = 8) -> TSFit:
    """AR(p) mean + GARCH(p,q) variance. (arch package supports AR-X mean.)"""
    from arch import arch_model
    s = series.dropna()
    am = arch_model(s, mean="ARX", lags=ar_p, vol="GARCH",
                     p=gp, q=gq, dist="normal", rescale=False)
    res = am.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    fmean = fc.mean.values[-1].tolist()
    fstd = np.sqrt(fc.variance.values[-1]).tolist()
    resid = res.resid.dropna().values
    log = [f"── AR({ar_p})-GARCH({gp},{gq}) fit ──",
            f"  n            = {len(s)}",
            f"  log-lik      = {res.loglikelihood:.3f}",
            f"  AIC / BIC    = {res.aic:.2f} / {res.bic:.2f}"]
    for k in res.params.index:
        log.append(f"  {k:<14} = {float(res.params[k]):+.4f}")
    lb = _ljung_box(resid)
    arch_p = _arch_lm(resid)
    log.append(f"  Ljung-Box p  = {lb:.3f}")
    log.append(f"  ARCH-LM p    = {arch_p:.3f}")
    return TSFit(model=f"AR({ar_p})-GARCH({gp},{gq})",
                  params={k: float(res.params[k]) for k in res.params.index},
                  aic=float(res.aic), bic=float(res.bic), loglik=float(res.loglikelihood),
                  ljung_box_p=lb, arch_lm_p=arch_p, n_obs=len(s),
                  forecast_mean=fmean, forecast_std=fstd, log=log)


MODELS = {
    "AR": fit_ar,
    "ARMA": fit_arma,
    "GARCH": fit_garch,
    "ARMA-GARCH": fit_arma_garch,
}


def fit(series: pd.Series, model: str, **kwargs) -> TSFit:
    if model not in MODELS:
        raise ValueError(f"unknown model {model}; choose from {list(MODELS)}")
    return MODELS[model](series, **kwargs)
