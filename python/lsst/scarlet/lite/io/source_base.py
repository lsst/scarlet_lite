from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import DTypeLike

from ..observation import Observation
from ..source import Source
from .utils import PersistenceError

__all__ = ["ScarletSourceBaseData"]


@dataclass(kw_only=True)
class ScarletSourceBaseData(ABC):
    """Data for a scarlet source

    Attributes
    ----------
    source_type :
        The type of source being stored.
    source_registry :
        A registry of all known source types used for deserialization.
    metadata :
        Metadata associated with the source.
        If `metadata` contains the `id` key, it is used as the
        key for the source in a `Blend`'s dictionary of sources.
    version :
        The schema version of the exact data class.
    """

    source_type: str
    source_registry: ClassVar[dict[str, type[ScarletSourceBaseData]]] = {}
    metadata: dict[str, Any] | None = None
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

    @abstractmethod
    def to_source(self, observation: Observation) -> Source:
        """Convert to a `Source` for use in scarlet

        Parameters
        ----------
        observation:
            The observation used to render the source.

        Returns
        -------
        source:
            The `Source` representation of this data.
        """
