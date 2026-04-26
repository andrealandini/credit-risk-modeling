"""Shared base classes for all credit risk model components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResult:
    value: float
    log: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"value": self.value, "log": self.log, "metadata": self.metadata}


class BaseModel(ABC):
    name: str = ""
    label: str = ""
    description: str = ""

    @abstractmethod
    def compute(self, **params) -> ModelResult:
        pass

    @property
    @abstractmethod
    def param_schema(self) -> list[dict]:
        """Return list of param dicts: {name, label, type, default, min, max, step, unit}."""
        pass
