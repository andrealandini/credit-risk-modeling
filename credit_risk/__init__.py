from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["SECRET_KEY"] = "credit-risk-dev-key"
    app.config["JSON_SORT_KEYS"] = False

    from .routes import register_blueprints
    register_blueprints(app)

    return app
