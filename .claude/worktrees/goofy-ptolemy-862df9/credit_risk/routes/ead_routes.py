"""EAD models blueprint."""
from __future__ import annotations

import json

from flask import Blueprint, jsonify, render_template, request

from ..models import EAD_REGISTRY

ead_bp = Blueprint("ead", __name__)


@ead_bp.route("/")
def index():
    models_info = {
        k: {"label": cls.label, "description": cls.description}
        for k, cls in EAD_REGISTRY.items()
    }
    schemas = {k: cls().param_schema for k, cls in EAD_REGISTRY.items()}
    return render_template("ead_models.html", models=models_info, schemas=json.dumps(schemas))


@ead_bp.route("/api/compute", methods=["POST"])
def api_compute():
    payload = request.get_json(force=True) or {}
    model_name = payload.pop("model", "ccf")
    if model_name not in EAD_REGISTRY:
        return jsonify({"error": f"Unknown EAD model: {model_name}"}), 400

    model = EAD_REGISTRY[model_name]()
    result = model.compute(**payload)

    return jsonify({
        "ead": result.value,
        "ead_fmt": f"€{result.value:,.2f}",
        "log": result.log,
        "metadata": result.metadata,
    })
