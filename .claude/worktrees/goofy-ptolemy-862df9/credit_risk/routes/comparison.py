"""Hypothesis / Results Comparison blueprint.

Accepts up to 4 user-defined scenarios. Each scenario specifies:
  - Macro inputs (GDP, UE, inflation, policy rate, credit growth, lending standards)
  - PD model + params
  - LGD model + params
  - EAD model + params
  - Monte Carlo settings

Returns EL, UL, VaR, ES per scenario for comparison charts and tables.
"""
from __future__ import annotations

import csv
import io
import json

import numpy as np
from flask import Blueprint, Response, jsonify, render_template, request

from ..data import load_ecb_data, BUILTIN_SCENARIOS
from ..models import PD_REGISTRY, LGD_REGISTRY, EAD_REGISTRY
from ..simulation.monte_carlo import (
    MonteCarloEngine, PortfolioSpec, make_logistic_pd_fn, make_constant_pd_fn
)
from ..utils import scenario_comparison_fig, loss_distribution_fig, pd_term_structure_fig

comparison_bp = Blueprint("comparison", __name__)
_PANEL = load_ecb_data()


def _run_scenario(scenario: dict) -> dict:
    name = scenario.get("name", "Scenario")
    pd_model_name = scenario.get("pd_model", "logistic_regression")
    lgd_model_name = scenario.get("lgd_model", "beta_regression")
    ead_model_name = scenario.get("ead_model", "ccf")
    pd_params = scenario.get("pd_params", {})
    lgd_params = scenario.get("lgd_params", {})
    ead_params = scenario.get("ead_params", {})
    mc_params = scenario.get("mc_params", {})

    # Instantiate models
    pd_model = PD_REGISTRY[pd_model_name](_PANEL) if pd_model_name == "logistic_regression" else PD_REGISTRY[pd_model_name]()
    lgd_model = LGD_REGISTRY[lgd_model_name]()
    ead_model = EAD_REGISTRY[ead_model_name]()

    # Point estimates for PD / LGD / EAD
    pd_result = pd_model.compute(**pd_params)
    lgd_result = lgd_model.compute(**lgd_params)
    ead_result = ead_model.compute(**ead_params)

    pd_val = pd_result.value
    lgd_val = lgd_result.value
    ead_val = ead_result.value
    el_point = pd_val * lgd_val * ead_val

    # Monte Carlo
    n_obligors = int(mc_params.get("n_obligors", 200))
    n_paths = int(mc_params.get("n_paths", 3000))
    horizon_q = int(mc_params.get("horizon_q", 4))
    rho = float(mc_params.get("rho", 0.15))
    seed = int(mc_params.get("seed", 42))

    if pd_model_name == "logistic_regression":
        pd_fn = make_logistic_pd_fn(pd_model._intercept, pd_model._coefs)
    else:
        pd_fn = make_constant_pd_fn(pd_val)

    portfolio = PortfolioSpec(
        n_obligors=n_obligors,
        ead_mean=ead_val,
        ead_cv=float(mc_params.get("ead_cv", 0.5)),
        lgd_mean=lgd_val,
        lgd_concentration=float(mc_params.get("lgd_concentration", 8.0)),
        rho=rho,
    )

    engine = MonteCarloEngine(pd_fn=pd_fn, lgd_fn=None, ead_fn=None, portfolio=portfolio)

    # Build macro overrides from scenario macro inputs
    macro = scenario.get("macro", {})
    macro_overrides = {
        "gdp_growth": float(macro.get("gdp_growth", 1.5)),
        "unemployment": float(macro.get("unemployment", 8.0)),
        "policy_rate": float(macro.get("policy_rate", 2.5)),
        "credit_growth": float(macro.get("credit_growth", 3.0)),
    } if macro else None

    mc = engine.run(horizon_q=horizon_q, n_paths=n_paths, seed=seed,
                    macro_overrides=macro_overrides)

    loss_fig = loss_distribution_fig(
        mc.losses, mc.el, mc.var_99, mc.var_999, mc.es_975,
        title=f"{name} — Loss Distribution",
    )
    pd_ts = pd_term_structure_fig(mc.pd_paths, horizon_q)

    return {
        "name": name,
        "pd_model": pd_model.label,
        "lgd_model": lgd_model.label,
        "ead_model": ead_model.label,
        "pd": pd_val,
        "lgd": lgd_val,
        "ead": ead_val,
        "el_point": el_point,
        "el": mc.el,
        "ul": mc.ul,
        "var_99": mc.var_99,
        "var_999": mc.var_999,
        "es_975": mc.es_975,
        "notional": mc.portfolio_notional,
        "el_rate": mc.el / mc.portfolio_notional if mc.portfolio_notional else 0,
        "loss_fig": json.loads(loss_fig),
        "pd_ts_fig": json.loads(pd_ts),
        "pd_pct": f"{pd_val:.4%}",
        "lgd_pct": f"{lgd_val:.2%}",
        "ead_fmt": f"€{ead_val:,.2f}",
        "el_fmt": f"€{mc.el / 1e6:,.2f}M",
        "ul_fmt": f"€{mc.ul / 1e6:,.2f}M",
        "var_99_fmt": f"€{mc.var_99 / 1e6:,.2f}M",
        "var_999_fmt": f"€{mc.var_999 / 1e6:,.2f}M",
    }


@comparison_bp.route("/")
def index():
    pd_models = {k: {"label": v.label, "description": v.description} for k, v in PD_REGISTRY.items()}
    lgd_models = {k: {"label": v.label, "description": v.description} for k, v in LGD_REGISTRY.items()}
    ead_models = {k: {"label": v.label, "description": v.description} for k, v in EAD_REGISTRY.items()}

    pd_schemas = {}
    for k, cls in PD_REGISTRY.items():
        inst = cls(_PANEL) if k == "logistic_regression" else cls()
        pd_schemas[k] = inst.param_schema
    lgd_schemas = {k: cls().param_schema for k, cls in LGD_REGISTRY.items()}
    ead_schemas = {k: cls().param_schema for k, cls in EAD_REGISTRY.items()}

    builtin = {k: v.as_dict() for k, v in BUILTIN_SCENARIOS.items()}

    return render_template(
        "comparison.html",
        pd_models=pd_models,
        lgd_models=lgd_models,
        ead_models=ead_models,
        pd_schemas=json.dumps(pd_schemas),
        lgd_schemas=json.dumps(lgd_schemas),
        ead_schemas=json.dumps(ead_schemas),
        builtin_scenarios=json.dumps(builtin),
    )


@comparison_bp.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True) or {}
    scenarios = payload.get("scenarios", [])
    if not scenarios:
        return jsonify({"error": "No scenarios provided"}), 400

    results = []
    for sc in scenarios[:4]:
        try:
            results.append(_run_scenario(sc))
        except Exception as e:
            results.append({"name": sc.get("name", "?"), "error": str(e)})

    valid = [r for r in results if "error" not in r]
    comparison_fig_json = json.loads(scenario_comparison_fig(valid)) if valid else None

    return jsonify({"results": results, "comparison_fig": comparison_fig_json})


@comparison_bp.route("/api/export", methods=["POST"])
def api_export():
    payload = request.get_json(force=True) or {}
    results = payload.get("results", [])

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "name", "pd_model", "lgd_model", "ead_model",
        "pd_pct", "lgd_pct", "ead_fmt",
        "el_fmt", "ul_fmt", "var_99_fmt", "var_999_fmt",
    ])
    writer.writeheader()
    for r in results:
        writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=credit_risk_scenarios.csv"},
    )
