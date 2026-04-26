"""PD models blueprint."""
from __future__ import annotations

import json

import numpy as np
from flask import Blueprint, jsonify, render_template, request

from ..data import load_ecb_data
from ..models import PD_REGISTRY
from ..simulation.gbm import simulate_gbm_paths
from ..utils import tornado_fig, gbm_paths_fig

pd_bp = Blueprint("pd", __name__)
_PANEL = load_ecb_data()


def _build_tornado(model_name: str, params: dict) -> str:
    model = PD_REGISTRY[model_name](_PANEL) if model_name == "logistic_regression" else PD_REGISTRY[model_name]()
    base = model.compute(**params)
    base_pd = base.value

    schema = model.param_schema
    rows = []
    for p in schema:
        if p["type"] not in ("range", "number"):
            continue
        name = p["name"]
        val = float(params.get(name, p["default"]))
        step = float(p.get("step", 0.1))
        shock = step * 5

        up_params = {**params, name: val + shock}
        dn_params = {**params, name: val - shock}
        pd_up = model.compute(**up_params).value
        pd_dn = model.compute(**dn_params).value
        rows.append({"factor": p["label"], "up": pd_up - base_pd, "down": pd_dn - base_pd})

    return tornado_fig(base_pd, rows)


@pd_bp.route("/")
def index():
    models_info = {
        k: {"label": cls.label if hasattr(cls, "label") else k,
            "description": cls.description if hasattr(cls, "description") else ""}
        for k, cls in PD_REGISTRY.items()
    }
    schemas = {}
    for k, cls in PD_REGISTRY.items():
        inst = cls(_PANEL) if k == "logistic_regression" else cls()
        schemas[k] = inst.param_schema

    return render_template(
        "pd_models.html",
        models=models_info,
        schemas=json.dumps(schemas),
        panel_cols=list(_PANEL.columns),
    )


@pd_bp.route("/api/compute", methods=["POST"])
def api_compute():
    payload = request.get_json(force=True) or {}
    model_name = payload.pop("model", "logistic_regression")
    if model_name not in PD_REGISTRY:
        return jsonify({"error": f"Unknown PD model: {model_name}"}), 400

    model = PD_REGISTRY[model_name](_PANEL) if model_name == "logistic_regression" else PD_REGISTRY[model_name]()
    result = model.compute(**payload)
    tornado = _build_tornado(model_name, payload)

    merton_fig_json = None
    if model_name == "merton":
        V = float(payload.get("asset_value", 100.0)) * 1e6
        D = float(payload.get("debt", 70.0)) * 1e6
        sig = float(payload.get("asset_vol", 0.25))
        r = float(payload.get("risk_free", 0.03))
        T = float(payload.get("horizon", 1.0))
        paths = simulate_gbm_paths(V, r, sig, T, n_steps=252, n_paths=500)
        merton_fig_json = gbm_paths_fig(paths / 1e6, D / 1e6, "Merton Asset Paths (€M)")

    return jsonify({
        "pd": result.value,
        "pd_pct": f"{result.value:.4%}",
        "log": result.log,
        "metadata": result.metadata,
        "tornado_fig": json.loads(tornado),
        "merton_fig": json.loads(merton_fig_json) if merton_fig_json else None,
    })
