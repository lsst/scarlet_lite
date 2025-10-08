from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from ..component import Component
from ..observation import Observation
from ..source import Source
from .component import ScarletComponentBaseData
from .migration import PRE_SCHEMA, MigrationRegistry, migration
from .source_base import ScarletSourceBaseData
from .utils import decode_metadata, encode_metadata

__all__ = ["ScarletSourceData"]

CURRENT_SCHEMA = "1.0.0"
SOURCE_TYPE = "source"
MigrationRegistry.set_current(SOURCE_TYPE, CURRENT_SCHEMA)


@dataclass(kw_only=True)
class ScarletSourceData(ScarletSourceBaseData):
    """Data for a scarlet source

    Attributes
    ----------
    components :
        The components contained in the source that are not factorized.
    factorized_components :
        The components contained in the source that are factorized.
    metadata :
        Metadata associated with the source.
        If `metadata` contains the `id` key, it is used as the
        key for the source in a `Blend`'s dictionary of sources.
    """

    source_type: str = SOURCE_TYPE
    components: list[ScarletComponentBaseData]
    metadata: dict[str, Any] | None = None
    version: str = CURRENT_SCHEMA

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        result: dict[str, Any] = {
            "source_type": self.source_type,
            "components": [component.as_dict() for component in self.components],
            "version": self.version,
        }
        if self.metadata is not None:
            result["metadata"] = encode_metadata(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletSourceData:
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
        data = MigrationRegistry.migrate(SOURCE_TYPE, data)
        metadata = data.get("metadata", None)
        components = [
            ScarletComponentBaseData.from_dict(component, dtype=dtype) for component in data["components"]
        ]
        return cls(components=components, metadata=decode_metadata(metadata))

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
        components: list[Component] = [component.to_component(observation) for component in self.components]
        return Source(components=components, metadata=self.metadata)


ScarletSourceData.register()


@migration(SOURCE_TYPE, PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema source to schema version 1.0.0

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
    # Check for legacy models
    if "factorized" in data:
        data["data_type"] = "factorized"
        data["components"] = data["factorized"]
        del data["factorized"]
    data["version"] = "1.0.0"
    return data
