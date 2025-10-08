from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from numpy.typing import DTypeLike

from ..component import Component
from ..observation import Observation

from .errors import PersistenceError

__all__ = ["ScarletComponentBaseData"]


@dataclass(kw_only=True)
class ScarletComponentBaseData(ABC):
    """Base data for a scarlet component"""

    component_registry: ClassVar[dict[str, type[ScarletComponentBaseData]]] = {}
    component_type: str
    version: str

    @classmethod
    def register(cls) -> None:
        """Register a new component type"""
        ScarletComponentBaseData.component_registry[cls.component_type] = cls

    @abstractmethod
    def to_component(self, observation: Observation) -> Component:
        """Convert the storage data model into a scarlet Component

        Parameters
        ----------
        observation :
            The observation that the component is associated with

        Returns
        -------
        component :
            A scarlet component extracted from persisted data.
        """

    @abstractmethod
    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """

    @staticmethod
    def from_dict(data: dict, dtype: DTypeLike | None = None) -> ScarletComponentBaseData:
        """Reconstruct `ScarletComponentBaseData` from JSON compatible
        dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object
        dtype :
            Datatype of the resulting model.

        Returns
        -------
        result :
            The reconstructed object
        """
        component_type = data["component_type"]
        # Fix legacy naming
        if component_type == "component":
            component_type = "cube"
        if component_type not in ScarletComponentBaseData.component_registry:
            raise PersistenceError(f"Unknown component type: {component_type}")
        cls = ScarletComponentBaseData.component_registry[component_type]
        return cls.from_dict(data, dtype=dtype)
