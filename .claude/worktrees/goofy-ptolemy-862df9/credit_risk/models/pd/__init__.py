from .logistic_regression import LogisticRegressionPD
from .merton import MertonPD
from .survival import SurvivalHazardPD

REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegressionPD,
    "merton": MertonPD,
    "survival": SurvivalHazardPD,
}

__all__ = ["REGISTRY", "LogisticRegressionPD", "MertonPD", "SurvivalHazardPD"]
