from __future__ import annotations

from dataclasses import dataclass

from typing import Callable

import numpy as np
from numpy.typing import DTypeLike

from .component import ScarletComponentBaseData
from .migration import migration, PRE_SCHEMA, MigrationRegistry

from ..bbox import Box
from ..component import Component
from ..image import Image
from ..observation import Observation

__all__ = ["ScarletCubeComponentData", "ComponentCube"]

CURRENT_SCHEMA = "1.0.0"
COMPONENT_TYPE = "cube"
MigrationRegistry.set_current(COMPONENT_TYPE, CURRENT_SCHEMA)


class ComponentCube(Component):
    """Dummy component for a component cube.

    This is duck-typed to a `lsst.scarlet.lite.Component` in order to
    generate a model from the component but it is currently not functional
    in that it cannot be optimized, only persisted and loaded.

    If scarlet lite ever implements a component as a data cube,
    this class can be removed.
    """

    def __init__(self, model: Image, peak: tuple[int, int]):
        """Initialization

        Parameters
        ----------
        bands :
        model :
            The 3D (bands, y, x) model of the component.
        peak :
            The `(y, x)` peak of the component.
        bbox :
            The bounding box of the component.
        """
        super().__init__(model.bands, model.bbox)
        self._model = model
        self.peak = peak

    def get_model(self) -> Image:
        """Generate the model for the source

        Returns
        -------
        model :
            The model as a 3D `(band, y, x)` array.
        """
        return self._model

    def resize(self, model_box: Box) -> bool:
        """Test whether or not the component needs to be resized"""
        return False

    def update(self, it: int, input_grad: np.ndarray) -> None:
        """Implementation of unused abstract method"""

    def parameterize(self, parameterization: Callable) -> None:
        """Implementation of unused abstract method"""

    def to_component_data(self) -> ScarletCubeComponentData:
        """Convert the component to persistable ScarletComponentData

        Returns
        -------
        component_data: ScarletComponentData
            The data object containing the component information
        """
        return ScarletCubeComponentData(
            origin=self.bbox.origin,  # type: ignore
            peak=self.peak,  # type: ignore
            model=self.get_model().data,
        )


@dataclass(kw_only=True)
class ScarletCubeComponentData(ScarletComponentBaseData):
    """Data for a component expressed as a 3D data cube

    This is used for scarlet component models that are not factorized,
    storing their entire model as a 3D data cube (bands, y, x).

    Attributes
    ----------
    origin :
        The lower bound of the components bounding box.
    peak :
        The peak of the component.
    model :
        The model for the component.
    """

    origin: tuple[int, int]
    peak: tuple[float, float]
    model: np.ndarray
    component_type: str = COMPONENT_TYPE
    version: str = CURRENT_SCHEMA

    @property
    def shape(self):
        return self.model.shape[-2:]

    def to_component(self, observation: Observation) -> ComponentCube:
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
        bbox = Box(self.shape, origin=self.origin)
        model = self.model
        if self.peak is None:
            peak = None
        else:
            peak = (int(np.round(self.peak[0])), int(np.round(self.peak[0])))
        assert peak is not None
        component = ComponentCube(
            model=Image(model, yx0=bbox.origin, bands=observation.bands),  # type: ignore
            peak=peak,
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
            "origin": self.origin,
            "shape": self.model.shape,
            "peak": self.peak,
            "model": tuple(self.model.flatten().astype(float)),
            "component_type": "component",
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike | None = None) -> ScarletCubeComponentData:
        """Reconstruct `ScarletComponentData` from JSON compatible dict

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
            model=np.array(data["model"]).reshape(shape).astype(dtype),
        )


ScarletCubeComponentData.register()


@migration(COMPONENT_TYPE, PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema CubeComponent to schema version 1.0.0

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
