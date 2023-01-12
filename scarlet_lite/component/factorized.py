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

__all__ = ["FactorizedComponent"]

from typing import Callable

import numpy as np

from ..bbox import Box
from .base import Component
from ..image import Image
from ..operators import Monotonicity
from ..parameters import parameter, Parameter


class FactorizedComponent(Component):
    """A component that can be factorized into SED and morphology parameters"""

    def __init__(
        self,
        bands: tuple,
        sed: Parameter | np.ndarray,
        morph: Parameter | np.ndarray,
        bbox: Box,
        model_bbox: Box,
        center: tuple[int, int] | None = None,
        bg_rms: np.ndarray | None = None,
        bg_thresh: float | None = 0.25,
        floor: float = 1e-20,
        monotonicity: Monotonicity | None = None,
        padding: int = 5,
    ):
        """Initialize the component.

        Parameters
        ----------
        bands:
            The bands of the spectral dimension, in order.
        sed:
            The parameter to store and update the SED.
        morph:
            The parameter to store and update the morphology.
        center:
            Center of the source.
        bbox:
            The `Box` in the `model_bbox` that contains the source.
        model_bbox:
            The `Box` that contains the model.
            This is simplified from the main scarlet, where the model exists
            in a `frame`, which primarily exists because not all
            observations in main scarlet will use the same set of bands.
        bg_rms:
            The RMS of the background used to threshold, grow,
            and shrink the component.
        floor:
            Minimum value of the SED or center morphology pixel.
        monotonicity:
            The monotonicity operator to use for making the source monotonic.
            If this parameter is `None`, the source will not be made monotonic.
        """
        # Initialize all of the base attributes
        super().__init__(
            bands=bands,
            bbox=bbox,
            model_bbox=model_bbox,
        )
        self._sed = parameter(sed)
        self._morph = parameter(morph)
        self._center = center
        self.bg_rms = bg_rms
        self.bg_thresh = bg_thresh

        self.floor = floor
        self.monotonicity = monotonicity
        self.padding = padding

    @property
    def center(self) -> tuple[int, int] | None:
        """The center of the component

        Returns
        -------
        center:
            The center of the component
        """
        return self._center

    @property
    def component_center(self) -> tuple[int, int] | None:
        """The center of the component in its bounding box

        This is likely to be different than `Component.center`,
        since `Component.center` is the center of the component in the
        full model, whereas `component_center` is the center of the component
        inside its bounding box.

        Returns
        -------
        center:
            The center of the component in its bounding box
        """
        _center = self.center
        if _center is None:
            return None
        center = (
            self._center[0] - self.bbox.origin[-2],
            self._center[1] - self.bbox.origin[-1],
        )
        return center

    @property
    def sed(self) -> np.ndarray:
        """The array of SED values"""
        return self._sed.x

    @property
    def morph(self) -> np.ndarray:
        """The array of morphology values"""
        return self._morph.x

    @property
    def shape(self) -> tuple:
        """Shape of the resulting model image"""
        return self.sed.shape + self.morph.shape

    def get_model(self) -> Image:
        """Build the model from the SED and morphology"""
        # The sed and morph might be Parameters,
        # so cast them as arrays in the model.
        sed = self.sed
        morph = self.morph
        model = sed[:, None, None] * morph[None, :, :]
        return Image(model, bands=self.bands, yx0=self.bbox.origin)

    def grad_sed(self, input_grad, sed, morph):
        """Gradient of the SED wrt. the component model"""
        _grad = np.zeros(self.grad_box.shape, dtype=self.morph.dtype)
        _grad[self.grad_slices[1]] = input_grad[self.grad_slices[0]]
        return np.einsum("...jk,jk", _grad, morph)

    def grad_morph(self, input_grad, morph, sed):
        """Gradient of the morph wrt. the component model"""
        _grad = np.zeros(self.grad_box.shape, dtype=self.morph.dtype)
        _grad[self.grad_slices[1]] = input_grad[self.grad_slices[0]]
        return np.einsum("i,i...", sed, _grad)

    def prox_sed(self, sed: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the SED"""
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        return sed

    def prox_morph(self, morph: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the morphology"""
        # monotonicity
        if self.monotonicity is not None:
            morph = self.monotonicity(morph, self.component_center)

        if self.bg_thresh is not None and self.bg_rms is not None:
            bg_thresh = self.bg_rms * self.bg_thresh
            # Enforce background thresholding
            model = self.sed[:, None, None] * morph[None, :, :]
            morph[np.all(model < bg_thresh[:, None, None], axis=0)] = 0
        else:
            # enforce positivity
            morph[morph < 0] = 0

        # prevent divergent morphology
        shape = morph.shape
        if self.center is None:
            center = (shape[0] // 2, shape[1] // 2)
            print("center:", center)
            print(morph)
            print(np.max([morph[center], self.floor]))
        else:
            center = (
                self.center[0] - self.bbox.origin[-2],
                self.center[1] - self.bbox.origin[-1],
            )
        morph[center] = np.max([morph[center], self.floor])
        # Normalize the morphology
        morph[:] = morph / morph[center]
        return morph

    def resize(self) -> bool:
        """Test whether or not the component needs to be resized"""
        # No need to resize if there is no size threshold.
        # To allow box sizing but no thresholding use `bg_thresh=0`.
        if self.bg_thresh is None or self.bg_rms is None:
            return False

        model = self.sed[:, None, None] * self.morph[None, :, :]
        bg_thresh = self.bg_rms * self.bg_thresh
        significant = np.any(model >= bg_thresh[:, None, None], axis=0)
        if np.sum(significant) == 0:
            # There are no significant pixels,
            # so make a small box around the center
            new_box = Box((1, 1), self.center).grow(self.padding)
        else:
            new_box = (
                Box.from_data(significant, min_value=0).grow(self.padding)
                + self.bbox.origin
            ) & self.model_bbox
        if new_box == self.bbox:
            return False

        old_box = self.bbox
        self._bbox = new_box
        self._morph.grow(old_box, new_box)

        self.overlap = self.model_bbox & new_box
        self.grad_box = self.spectral_box @ new_box
        self.grad_slices = (self.spectral_box @ self.model_bbox).overlapped_slices(
            self.grad_box
        )
        return True

    def update(self, it: int, input_grad: np.ndarray):
        """Update the SED and morphology parameters"""
        # Store the input SED so that the morphology can
        # have a consistent update
        sed = self.sed.copy()
        self._sed.update(it, input_grad, self.morph)
        self._morph.update(it, input_grad, sed)

    def parameterize(self, parameterization: Callable) -> None:
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization: Callable
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        # Update the SED and morph in place
        parameterization(self)
        # update the parameters
        self._sed.grad = self.grad_sed
        self._sed.prox = self.prox_sed
        self._morph.grad = self.grad_morph
        self._morph.prox = self.prox_morph

    def __str__(self):
        return "FactorizedComponent"

    def __repr__(self):
        return "FactorizedComponent"
