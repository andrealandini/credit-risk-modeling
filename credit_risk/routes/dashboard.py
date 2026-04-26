"""Dashboard blueprint — portfolio overview and macro panel."""
from __future__ import annotations

import json

import numpy as np
from flask import Blueprint, jsonify, render_template

from ..data import load_ecb_data, MacroScenario, BUILTIN_SCENARIOS
from ..models import PD_REGISTRY, LGD_REGISTRY, EAD_REGISTRY
from ..simulation.monte_carlo import (
    MonteCarloEngine, PortfolioSpec, make_logistic_pd_fn, make_constant_pd_fn
)
from ..utils import (
    macro_series_fig, loss_distribution_fig, pd_term_structure_fig,
    correlation_heatmap_fig,
)

dashboard_bp = Blueprint("dashboard", __name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_PANEL = load_ecb_data()

_PD_MODEL = PD_REGISTRY["logistic_regression"](_PANEL)
_LGD_MODEL = LGD_REGISTRY["beta_regression"]()
_EAD_MODEL = EAD_REGISTRY["ccf"]()

_PD_COEFS = _PD_MODEL._coefs
_PD_INTERCEPT = _PD_MODEL._intercept

_DEFAULT_PORTFOLIO = PortfolioSpec(n_obligors=200, ead_mean=1_000_000, lgd_mean=0.45, rho=0.15)
_ENGINE = MonteCarloEngine(
    pd_fn=make_logistic_pd_fn(_PD_INTERCEPT, _PD_COEFS),
    lgd_fn=None,
    ead_fn=None,
    portfolio=_DEFAULT_PORTFOLIO,
)
_LAST_MC = _ENGINE.run(horizon_q=4, n_paths=5000, seed=42)


def _kpis(mc) -> dict:
    return {
        "el":       f"€{mc.el / 1e6:,.2f}M",
        "ul":       f"€{mc.ul / 1e6:,.2f}M",
        "var_99":   f"€{mc.var_99 / 1e6:,.2f}M",
        "var_999":  f"€{mc.var_999 / 1e6:,.2f}M",
        "es_975":   f"€{mc.es_975 / 1e6:,.2f}M",
        "el_rate":  f"{mc.el / mc.portfolio_notional:.3%}",
        "notional": f"€{mc.portfolio_notional / 1e6:,.1f}M",
        "n_paths":  f"{mc.n_paths:,}",
    }


@dashboard_bp.route("/")
def index():
    mc = _LAST_MC
    factor_names = ["gdp_growth", "unemployment", "policy_rate", "credit_growth"]
    macro_fig = macro_series_fig(_PANEL, list(_PANEL.columns[:6]))
    loss_fig = loss_distribution_fig(mc.losses, mc.el, mc.var_99, mc.var_999, mc.es_975)
    pd_ts_fig = pd_term_structure_fig(mc.pd_paths, mc.horizon_q)

    corr_data = _PANEL[factor_names].dropna().corr().values
    corr_fig = correlation_heatmap_fig(corr_data, factor_names)

    scenarios = {k: v.as_dict() for k, v in BUILTIN_SCENARIOS.items()}

    return render_template(
        "dashboard.html",
        kpis=_kpis(mc),
        macro_fig=macro_fig,
        loss_fig=loss_fig,
        pd_ts_fig=pd_ts_fig,
        corr_fig=corr_fig,
        scenarios=json.dumps(scenarios),
        pd_model_name=_PD_MODEL.label,
        lgd_model_name=_LGD_MODEL.label,
        ead_model_name=_EAD_MODEL.label,
        n_obs=len(_PANEL),
        data_range=f"{_PANEL.index[0].strftime('%Y Q%q' if hasattr(_PANEL.index[0], 'quarter') else '%Y-%m')} – {_PANEL.index[-1].strftime('%Y-%m')}",
    )


@dashboard_bp.route("/api/summary")
def api_summary():
    mc = _LAST_MC
    return jsonify({
        "kpis": _kpis(mc),
        "el": mc.el,
        "ul": mc.ul,
        "var_99": mc.var_99,
        "var_999": mc.var_999,
        "es_975": mc.es_975,
        "notional": mc.portfolio_notional,
    })
