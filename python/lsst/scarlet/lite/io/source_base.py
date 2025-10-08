from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from numpy.typing import DTypeLike
import numpy as np

from .errors import PersistenceError

__all__ = ["ScarletSourceBaseData"]


@dataclass(kw_only=True)
class ScarletSourceBaseData(ABC):
    """Data for a scarlet source

    Attributes
    ----------
    components :
        The components contained in the source that are not factorized.
    factorized_components :
        The components contained in the source that are factorized.
    peak_id :
        The peak ID of the source in it's parent's footprint peak catalog.
    """
    source_registry: ClassVar[dict[str, type[ScarletSourceBaseData]]] = {}
    peak_id: int
    version: str

    @classmethod
    def register(cls) -> None:
        """Register a new source type"""
        ScarletSourceBaseData.source_registry[cls.source_type] = cls

    @abstractmethod
    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """

    @staticmethod
    def from_dict(data: dict, dtype: DTypeLike = np.float32) -> ScarletSourceBaseData:
        """Reconstruct `ScarletSourceData` from JSON compatible
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
        source_type = data.get("source_type", None)

        # Fix legacy data that did not have a source_type
        if source_type is None:
            source_type = "source"

        cls = ScarletSourceBaseData.source_registry.get(source_type, None)
        if cls is None:
            raise PersistenceError(f"Unknown source type: {source_type}")

        return cls.from_dict(data, dtype=dtype)
