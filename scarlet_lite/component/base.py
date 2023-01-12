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

__all__ = ["Component"]

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..bbox import Box
from ..image import Image


class Component(ABC):
    """A base component in scarlet lite"""

    def __init__(
        self,
        bands: tuple,
        bbox: Box,
        model_bbox: Box,
    ):
        """Initialize a LiteComponent instance

        Parameters
        ----------
        bands:
            The bands used when the component model is created.
        bbox: Box
            The bounding box for this component.
        model_bbox: Box
            The bounding box for the full blend model.
        """
        self._bands = bands
        self._bbox = bbox
        self.spectral_box = Box((len(bands),))
        self.model_bbox = model_bbox
        self.overlap = model_bbox & bbox
        self.grad_box = self.spectral_box @ bbox
        self.grad_slices = (self.spectral_box @ self.model_bbox).overlapped_slices(
            self.grad_box
        )

    @property
    def bbox(self):
        """The bounding box that contains the component in the full image"""
        return self._bbox

    @property
    def bands(self):
        """The bands in the component model"""
        return self._bands

    @abstractmethod
    def resize(self) -> bool:
        """Test whether or not the component needs to be resized

        This should be overriden in inherited classes and return `True`
        if the component needs to be resized.
        """
        pass

    @abstractmethod
    def update(self, it: int, input_grad: np.ndarray):
        """Update the component parameters from an input gradient

        Parameters
        ----------
        it:
            The current iteration of the optimizer.
        input_grad:
            Gradient of the likelihood wrt the component model
        """
        pass

    @abstractmethod
    def get_model(self) -> Image:
        """Generate a model for the component

        This must be implemented in inherited classes.

        Returns
        -------
        model: Image
            The image of the component model.
        """
        pass

    @abstractmethod
    def parameterize(self, parameterization: Callable) -> None:
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization: Callable
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        pass

    def __str__(self):
        return "Component"

    def __repr__(self):
        return "Component"
