from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike

from .migration import migration, PRE_SCHEMA, MigrationRegistry

from ..bbox import Box
from .component import ScarletComponentBaseData
from ..observation import Observation
from ..parameters import FixedParameter

if TYPE_CHECKING:
    from ..component import FactorizedComponent

__all__ = ["ScarletFactorizedComponentData"]

CURRENT_SCHEMA = "1.0.0"
COMPONENT_TYPE = "factorized"
MigrationRegistry.set_current(COMPONENT_TYPE, CURRENT_SCHEMA)


@dataclass(kw_only=True)
class ScarletFactorizedComponentData(ScarletComponentBaseData):
    """Data for a factorized component

    Attributes
    ----------
    origin :
        The lower bound of the component's bounding box.
    peak :
        The ``(y, x)`` peak of the component.
    spectrum :
        The SED of the component.
    morph :
        The 2D morphology of the component.
    version :
        The schema version of the stored data.
    """

    component_type: str = COMPONENT_TYPE
    origin: tuple[int, int]
    peak: tuple[float, float]
    spectrum: np.ndarray
    morph: np.ndarray
    version: str = CURRENT_SCHEMA

    @property
    def shape(self):
        return self.morph.shape

    def to_component(self, observation: Observation) -> FactorizedComponent:
        """Convert the storage data model into a scarlet FactorizedComponent

        Parameters
        ----------
        observation :
            The observation that the component is associated with

        Returns
        -------
        factorized_component :
            A scarlet factorized component extracted from persisted data.
        """
        from ..component import FactorizedComponent

        bbox = Box(self.shape, origin=self.origin)
        spectrum = self.spectrum
        morph = self.morph
        if self.peak is None:
            peak = None
        else:
            peak = (int(np.round(self.peak[0])), int(np.round(self.peak[1])))
        assert peak is not None
        # Note: since we aren't fitting a model, we don't need to
        # set the RMS of the background.
        # We set it to NaN just to be safe.
        component = FactorizedComponent(
            bands=observation.bands,
            spectrum=FixedParameter(spectrum),
            morph=FixedParameter(morph),
            peak=peak,
            bbox=bbox,
            bg_rms=np.full((len(observation.bands),), np.nan),
        )
        return component

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        return {
            "origin": tuple(int(o) for o in self.origin),
            "shape": tuple(int(s) for s in self.morph.shape),
            "peak": tuple(int(p) for p in self.peak),
            "spectrum": tuple(self.spectrum.astype(float)),
            "morph": tuple(self.morph.flatten().astype(float)),
            "component_type": self.component_type,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletFactorizedComponentData:
        """Reconstruct `ScarletFactorizedComponentData` from JSON compatible
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
        data = MigrationRegistry.migrate(COMPONENT_TYPE, data)
        shape = tuple(data["shape"])
        return cls(
            origin=tuple(data["origin"]),  # type: ignore
            peak=data["peak"],
            spectrum=np.array(data["spectrum"]).astype(dtype),
            morph=np.array(data["morph"]).reshape(shape).astype(dtype),
        )


ScarletFactorizedComponentData.register()


@migration(COMPONENT_TYPE, PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema factorized component to schema version 1.0.0

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
