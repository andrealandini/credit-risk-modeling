from .ccf import CCFEAD
from .utilization import UtilizationRegressionEAD
from .markov import MarkovTransitionEAD

REGISTRY: dict[str, type] = {
    "ccf": CCFEAD,
    "utilization": UtilizationRegressionEAD,
    "markov": MarkovTransitionEAD,
}

__all__ = ["REGISTRY", "CCFEAD", "UtilizationRegressionEAD", "MarkovTransitionEAD"]
