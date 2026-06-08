"""LGD models blueprint."""
from __future__ import annotations

import json

import numpy as np
from flask import Blueprint, jsonify, render_template, request

from ..models import LGD_REGISTRY

lgd_bp = Blueprint("lgd", __name__)


def _build_sensitivity(model_name: str, params: dict) -> list[dict]:
    model = LGD_REGISTRY[model_name]()
    base = model.compute(**params).value
    rows = []
    for p in model.param_schema:
        if p["type"] not in ("range", "number"):
            continue
        name = p["name"]
        val = float(params.get(name, p["default"]))
        shock = float(p.get("step", 0.01)) * 5
        up = model.compute(**{**params, name: val + shock}).value
        dn = model.compute(**{**params, name: val - shock}).value
        rows.append({"param": p["label"], "base": base, "up": up, "down": dn,
                     "delta_up": up - base, "delta_dn": dn - base})
    return rows


@lgd_bp.route("/")
def index():
    models_info = {
        k: {"label": cls.label, "description": cls.description}
        for k, cls in LGD_REGISTRY.items()
    }
    schemas = {k: cls().param_schema for k, cls in LGD_REGISTRY.items()}
    return render_template("lgd_models.html", models=models_info, schemas=json.dumps(schemas))


@lgd_bp.route("/api/compute", methods=["POST"])
def api_compute():
    payload = request.get_json(force=True) or {}
    model_name = payload.pop("model", "beta_regression")
    if model_name not in LGD_REGISTRY:
        return jsonify({"error": f"Unknown LGD model: {model_name}"}), 400

    model = LGD_REGISTRY[model_name]()
    result = model.compute(**payload)
    sensitivity = _build_sensitivity(model_name, payload)

    return jsonify({
        "lgd": result.value,
        "lgd_pct": f"{result.value:.4%}",
        "log": result.log,
        "metadata": result.metadata,
        "sensitivity": sensitivity,
    })
