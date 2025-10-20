from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from .blend import ScarletBlendBaseData
from .migration import PRE_SCHEMA, MigrationRegistry, migration
from .utils import PersistenceError, decode_metadata, encode_metadata

__all__ = ["ScarletModelData"]

CURRENT_SCHEMA = "1.0.0"
MODEL_TYPE = "scarlet_model"
MigrationRegistry.set_current(MODEL_TYPE, CURRENT_SCHEMA)


class ScarletModelData:
    """A container that propagates scarlet models for an entire catalog.

    Attributes
    ----------
    blends :
        Map from parent IDs in the source catalog
        to scarlet model data for each parent ID (blend).
    metadata :
        Metadata associated with the model,
        for example the order of bands.
    model_type :
        The type of model being stored.
    version :
        The schema version of the ScarletModelData.
    """

    model_type: str = MODEL_TYPE
    blends: dict[int, ScarletBlendBaseData]
    metadata: dict[str, Any] | None
    version: str = CURRENT_SCHEMA

    def __init__(
        self,
        blends: dict[int, ScarletBlendBaseData] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize an instance"""
        self.metadata = metadata
        if blends is None:
            blends = {}
        self.blends = blends

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        result = {
            "model_type": self.model_type,
            "blends": {bid: blend_data.as_dict() for bid, blend_data in self.blends.items()},
            "metadata": encode_metadata(self.metadata),
            "version": self.version,
        }
        return result

    def json(self) -> str:
        """Serialize the data model to a JSON formatted string

        Returns
        -------
        result : `str`
            The result of the object converted into a JSON format
        """
        result = self.as_dict()
        return json.dumps(result)

    @classmethod
    def from_dict(
        cls, data: dict, dtype: DTypeLike = np.float32, **kwargs: dict[str, Any]
    ) -> ScarletModelData:
        """Reconstruct `ScarletModelData` from JSON compatible dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object
        dtype :
            Datatype of the resulting model.
        kwargs :
            Additional keyword arguments.

        Returns
        -------
        result :
            The reconstructed object
        """
        data = MigrationRegistry.migrate(cls.model_type, data)
        blends: dict[int, ScarletBlendBaseData] | None = {}
        for bid, blend in data.get("blends", {}).items():
            if "blend_type" not in blend:
                # Assume that this is a legacy model
                blend["blend_type"] = "blend"
            try:
                blend_data = ScarletBlendBaseData.from_dict(blend, dtype=dtype)
            except KeyError:
                raise PersistenceError(f"Unknown blend type: {blend['blend_type']} for blend ID: {bid}")
            blends[int(bid)] = blend_data  # type: ignore

        return cls(
            blends=blends,
            metadata=decode_metadata(data["metadata"]),
            **kwargs,
        )

    @classmethod
    def parse_obj(cls, data: dict) -> ScarletModelData:
        """Construct a ScarletModelData from python decoded JSON object.

        Parameters
        ----------
        data :
            The result of json.load(s) on a JSON persisted ScarletModelData

        Returns
        -------
        result :
            The `ScarletModelData` that was loaded the from the input object
        """
        return cls.from_dict(data, dtype=np.float32)


@migration(MODEL_TYPE, PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema model to schema version 1.0.0

    Parameters
    ----------
    data :
        The data to migrate.
    Returns
    -------
    result :
        The migrated data.
    """
    if "psfShape" in data:
        # Support legacy models before metadata was used
        data["metadata"] = {
            "model_psf": data["psf"],
            "model_psf_shape": data["psfShape"],
            "array_keys": ["model_psf"],
        }
    data["version"] = "1.0.0"
    return data
