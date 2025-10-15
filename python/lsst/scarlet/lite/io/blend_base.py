from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from numpy.typing import DTypeLike

from .utils import PersistenceError

__all__ = ["ScarletBlendBaseData"]


@dataclass(kw_only=True)
class ScarletBlendBaseData(ABC):
    """Base data for a scarlet Blend data.

    Attributes
    ----------
    blend_registry :
        A registry of all known blend types used for deserialization.
    blend_type :
        The type of blend being stored.
    metadata :
        Metadata associated with the blend.
    version :
        The schema version of the exact data class.
    """

    blend_registry: ClassVar[dict[str, type[ScarletBlendBaseData]]] = {}
    blend_type: str
    metadata: dict[str, Any] | None = None
    version: str

    @classmethod
    def register(cls) -> None:
        """Register a new Blend type"""
        ScarletBlendBaseData.blend_registry[cls.blend_type] = cls

    @abstractmethod
    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """

    @staticmethod
    def from_dict(data: dict, dtype: DTypeLike | None = None) -> ScarletBlendBaseData:
        """Reconstruct `ScarletBlendBaseData` from JSON compatible dict.

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
        # Default to "blend" for backwards compatibility
        blend_type = data.get("blend_type", "blend")

        if blend_type not in ScarletBlendBaseData.blend_registry:
            raise PersistenceError(f"Unknown blend type: {blend_type}")

        cls = ScarletBlendBaseData.blend_registry[blend_type]
        return cls.from_dict(data, dtype=dtype)
