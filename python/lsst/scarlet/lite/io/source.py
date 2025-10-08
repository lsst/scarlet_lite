from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import DTypeLike
import numpy as np

from .component import ScarletComponentBaseData
from .source_base import ScarletSourceBaseData
from .migration import migration, PRE_SCHEMA, MigrationRegistry

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
    peak_id :
        The peak ID of the source in it's parent's footprint peak catalog.
    """
    source_type: str = SOURCE_TYPE
    components: list[ScarletComponentBaseData]
    peak_id: int
    version: str = CURRENT_SCHEMA

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        result = {
            "source_type": self.source_type,
            "components": [component.as_dict() for component in self.components],
            "peak_id": self.peak_id,
            "version": self.version,
        }
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
        components = [
            ScarletComponentBaseData.from_dict(component, dtype=dtype) for component in data["components"]
        ]
        return cls(components=components, peak_id=int(data["peak_id"]))


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
