from .beta_regression import BetaRegressionLGD
from .workout import WorkoutLGD
from .downturn import DownturnLGD

REGISTRY: dict[str, type] = {
    "beta_regression": BetaRegressionLGD,
    "workout": WorkoutLGD,
    "downturn": DownturnLGD,
}

__all__ = ["REGISTRY", "BetaRegressionLGD", "WorkoutLGD", "DownturnLGD"]
