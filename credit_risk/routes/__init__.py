from flask import Flask

from .crypto_routes import crypto_bp
from .dashboard import dashboard_bp
from .pd_routes import pd_bp
from .lgd_routes import lgd_bp
from .ead_routes import ead_bp
from .comparison import comparison_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(crypto_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(pd_bp, url_prefix="/pd")
    app.register_blueprint(lgd_bp, url_prefix="/lgd")
    app.register_blueprint(ead_bp, url_prefix="/ead")
    app.register_blueprint(comparison_bp, url_prefix="/comparison")
