from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from .blend_base import ScarletBlendBaseData
from .migration import PRE_SCHEMA, MigrationRegistry, migration
from .utils import PersistenceError, decode_metadata, encode_metadata

__all__ = ["HierarchicalBlendData"]

CURRENT_SCHEMA = "1.0.0"
BLEND_TYPE = "hierarchical"
MigrationRegistry.set_current(BLEND_TYPE, CURRENT_SCHEMA)


@dataclass(kw_only=True)
class HierarchicalBlendData(ScarletBlendBaseData):
    """Data for a hierarchical blend.

    Attributes
    ----------
    blend_type :
        The type of blend being stored
    children :
        Map from blend IDs to child blends.
    version :
        The schema version of the HierarchicalBlendData.
    """

    blend_type: str = BLEND_TYPE
    children: dict[int, ScarletBlendBaseData]
    version: str = CURRENT_SCHEMA

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        result: dict[str, Any] = {
            "blend_type": self.blend_type,
            "children": {bid: child.as_dict() for bid, child in self.children.items()},
            "version": self.version,
        }
        if self.metadata is not None:
            result["metadata"] = encode_metadata(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> HierarchicalBlendData:
        """Reconstruct `HierarchicalBlendData` from JSON compatible dict.

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
        data = MigrationRegistry.migrate(BLEND_TYPE, data)
        children: dict[int, ScarletBlendBaseData] = {}
        for blend_id, child in data["children"].items():
            try:
                children[int(blend_id)] = ScarletBlendBaseData.from_dict(child, dtype=dtype)
            except KeyError:
                raise PersistenceError(f"Unknown blend type: {child['blend_type']} for blend ID: {blend_id}")

        metadata = decode_metadata(data.get("metadata", None))
        return cls(children=children, metadata=metadata)


HierarchicalBlendData.register()


@migration(BLEND_TYPE, PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema hierarchical blend to schema version 1.0.0

    There were no changes to this data model in v1.0.0 but we need
    to provide a way to migrate pre-schema data.

    Parameters
    ----------
    data :
        The data to migrate.

    Returns
    -------
    result :
        The migrated data.
    """
    data["version"] = "1.0.0"
    return data
