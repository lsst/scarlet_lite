# This file is part of scarlet_lite.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = ["Source"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

from .bbox import Box
from .component import Component
from .image import Image

if TYPE_CHECKING:
    from .io import ScarletSourceBaseData, ScarletSourceData


class SourceBase(ABC):
    """Base class for a source

    This is primarily to allow `isinstance` checks
    without importing the full `Source` class.
    """

    metadata: dict[str, Any] | None = None

    @abstractmethod
    def to_source_data(self) -> ScarletSourceBaseData:
        """Convert to a `ScarletSourceBaseData` for serialization

        Returns
        -------
        source_data:
            The `ScarletSourceData` representation of this source.
        """


class Source(SourceBase):
    """A container for components associated with the same astrophysical object

    A source can have a single component, or multiple components,
    and each can be contained in different bounding boxes.

    Parameters
    ----------
    components:
        The components contained in the source.
    """

    def __init__(self, components: list[Component], metadata: dict | None = None):
        self.components = components
        self.flux_weighted_image: Image | None = None
        self.metadata = metadata

    @property
    def n_components(self) -> int:
        """The number of components in this source"""
        return len(self.components)

    @property
    def center(self) -> tuple[int, int] | None:
        """The center of the source in the full Blend."""
        if not self.is_null and hasattr(self.components[0], "peak"):
            return self.components[0].peak  # type: ignore
        return None

    @property
    def source_center(self) -> tuple[int, int] | None:
        """The center of the source in its local bounding box."""
        _center = self.center
        _origin = self.bbox.origin
        if _center is not None:
            center = (
                _center[0] - _origin[0],
                _center[1] - _origin[1],
            )
            return center
        return None

    @property
    def is_null(self) -> bool:
        """True if the source does not have any components"""
        return self.n_components == 0

    @property
    def bbox(self) -> Box:
        """The minimal bounding box to contain all of this sources components

        Null sources have a bounding box with shape `(0,0,0)`
        """
        if self.n_components == 0:
            return Box((0, 0))
        bbox = self.components[0].bbox
        for component in self.components[1:]:
            bbox = bbox | component.bbox
        return bbox

    @property
    def bands(self) -> tuple:
        """The bands in the full source model."""
        if self.is_null:
            return ()
        return self.components[0].bands

    def get_model(self, use_flux: bool = False) -> Image:
        """Build the model for the source

        This is never called during optimization and is only used
        to generate a model of the source for investigative purposes.

        Parameters
        ----------
        use_flux:
            Whether to use the re-distributed flux associated with the source
            instead of the component models.

        Returns
        -------
        model:
            The full-color model.
        """
        if self.n_components == 0:
            return 0  # type: ignore

        if use_flux:
            # Return the redistributed flux
            # (calculated by scarlet.lite.measure.weight_sources)
            return self.flux_weighted_image  # type: ignore

        model = self.components[0].get_model()
        for component in self.components[1:]:
            model = model + component.get_model()
        return model

    def parameterize(self, parameterization: Callable):
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization:
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        for component in self.components:
            component.parameterize(parameterization)

    def to_source_data(self) -> ScarletSourceData:
        """Convert to a `ScarletSourceData` for serialization

        Returns
        -------
        source_data:
            The `ScarletSourceData` representation of this source.
        """
        from .io import ScarletSourceData

        component_data = [c.to_component_data() for c in self.components]
        return ScarletSourceData(components=component_data, metadata=self.metadata)

    def __str__(self):
        return f"Source<{len(self.components)}>"

    def __repr__(self):
        return f"Source(components={repr(self.components)})>"
