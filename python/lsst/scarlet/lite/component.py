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

__all__ = [
    "Component",
    "FactorizedComponent",
    "default_fista_parameterization",
    "default_adaprox_parameterization",
]

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, cast

import numpy as np

from .bbox import Box
from .image import Image
from .operators import Monotonicity, prox_uncentered_symmetry
from .parameters import AdaproxParameter, FistaParameter, Parameter, parameter, relative_step


class Component(ABC):
    """A base component in scarlet lite.

    Parameters
    ----------
    bands:
        The bands used when the component model is created.
    bbox: Box
        The bounding box for this component.
    """

    def __init__(
        self,
        bands: tuple,
        bbox: Box,
    ):
        self._bands = bands
        self._bbox = bbox

    @property
    def bbox(self) -> Box:
        """The bounding box that contains the component in the full image"""
        return self._bbox

    @property
    def bands(self) -> tuple:
        """The bands in the component model"""
        return self._bands

    @abstractmethod
    def resize(self, model_box: Box) -> bool:
        """Test whether or not the component needs to be resized

        This should be overriden in inherited classes and return `True`
        if the component needs to be resized.
        """

    @abstractmethod
    def update(self, it: int, input_grad: np.ndarray) -> None:
        """Update the component parameters from an input gradient

        Parameters
        ----------
        it:
            The current iteration of the optimizer.
        input_grad:
            Gradient of the likelihood wrt the component model
        """

    @abstractmethod
    def get_model(self) -> Image:
        """Generate a model for the component

        This must be implemented in inherited classes.

        Returns
        -------
        model: Image
            The image of the component model.
        """

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


class FactorizedComponent(Component):
    """A component that can be factorized into spectrum and morphology
    parameters.

    Parameters
    ----------
    bands:
        The bands of the spectral dimension, in order.
    spectrum:
        The parameter to store and update the spectrum.
    morph:
        The parameter to store and update the morphology.
    peak:
        Location of the peak for the source.
    bbox:
        The `Box` in the `model_bbox` that contains the source.
    bg_rms:
        The RMS of the background used to threshold, grow,
        and shrink the component.
    bg_thresh:
        The threshold to use for the background RMS.
        If `None`, no background thresholding is applied, otherwise
        a sparsity constraint is applied to the morpholigy that
        requires flux in at least one band to be bg_thresh multiplied by
        `bg_rms` in that band.
    floor:
        Minimum value of the spectrum or center morphology pixel
        (depending on which is normalized).
    monotonicity:
        The monotonicity operator to use for making the source monotonic.
        If this parameter is `None`, the source will not be made monotonic.
    padding:
        The amount of padding to add to the component bounding box
        when resizing the component.
    is_symmetric:
        Whether the component is symmetric or not.
        If `True`, the morphology will be symmetrized using
        `prox_uncentered_symmetry`.
        If `False`, the morphology will not be symmetrized.
    """

    def __init__(
        self,
        bands: tuple,
        spectrum: Parameter | np.ndarray,
        morph: Parameter | np.ndarray,
        bbox: Box,
        peak: tuple[int, int] | None = None,
        bg_rms: np.ndarray | None = None,
        bg_thresh: float | None = 0.25,
        floor: float = 1e-20,
        monotonicity: Monotonicity | None = None,
        padding: int = 5,
        is_symmetric: bool = False,
    ):
        # Initialize all of the base attributes
        super().__init__(
            bands=bands,
            bbox=bbox,
        )
        self._spectrum = parameter(spectrum)
        self._morph = parameter(morph)
        self._peak = peak
        self.bg_rms = bg_rms
        self.bg_thresh = bg_thresh

        self.floor = floor
        self.monotonicity = monotonicity
        self.padding = padding
        self.is_symmetric = is_symmetric

    @property
    def peak(self) -> tuple[int, int] | None:
        """The peak of the component

        Returns
        -------
        peak:
            The peak of the component
        """
        return self._peak

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
        _center = self.peak
        if _center is None:
            return None
        center = (
            _center[0] - self.bbox.origin[-2],
            _center[1] - self.bbox.origin[-1],
        )
        return center

    @property
    def spectrum(self) -> np.ndarray:
        """The array of spectrum values"""
        return self._spectrum.x

    @property
    def morph(self) -> np.ndarray:
        """The array of morphology values"""
        return self._morph.x

    @property
    def shape(self) -> tuple:
        """Shape of the resulting model image"""
        return self.spectrum.shape + self.morph.shape

    def get_model(self) -> Image:
        """Build the model from the spectrum and morphology"""
        # The spectrum and morph might be Parameters,
        # so cast them as arrays in the model.
        spectrum = self.spectrum
        morph = self.morph
        model = spectrum[:, None, None] * morph[None, :, :]
        return Image(model, bands=self.bands, yx0=cast(tuple[int, int], self.bbox.origin))

    def grad_spectrum(self, input_grad: np.ndarray, spectrum: np.ndarray, morph: np.ndarray):
        """Gradient of the spectrum wrt. the component model"""
        return np.einsum("...jk,jk", input_grad, morph)

    def grad_morph(self, input_grad: np.ndarray, morph: np.ndarray, spectrum: np.ndarray):
        """Gradient of the morph wrt. the component model"""
        return np.einsum("i,i...", spectrum, input_grad)

    def prox_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the spectrum"""
        # prevent divergent spectrum
        spectrum[spectrum < self.floor] = self.floor
        spectrum[~np.isfinite(spectrum)] = self.floor
        return spectrum

    def prox_morph(self, morph: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the morphology"""
        # Get the peak position in the current bbox
        shape = morph.shape
        if self.peak is None:
            peak = (shape[0] // 2, shape[1] // 2)
        else:
            peak = (
                self.peak[0] - self.bbox.origin[-2],
                self.peak[1] - self.bbox.origin[-1],
            )

        # monotonicity
        if self.monotonicity is not None:
            morph = self.monotonicity(morph, cast(tuple[int, int], self.component_center))

        # symmetry
        if self.is_symmetric:
            # Apply the symmetry operator
            morph = prox_uncentered_symmetry(morph, peak, fill=0.0)

        if self.bg_thresh is not None and self.bg_rms is not None:
            bg_thresh = self.bg_rms * self.bg_thresh
            # Enforce background thresholding
            model = self.spectrum[:, None, None] * morph[None, :, :]
            morph[np.all(model < bg_thresh[:, None, None], axis=0)] = 0
        else:
            # enforce positivity
            morph[morph < 0] = 0

        # prevent divergent morphology
        morph[peak] = np.max([morph[peak], self.floor])

        # Ensure that the morphology is finite
        morph[~np.isfinite(morph)] = 0

        # Normalize the morphology
        max_value = np.max(morph)
        if max_value > 0:
            morph[:] = morph / max_value
        return morph

    def resize(self, model_box: Box) -> bool:
        """Test whether or not the component needs to be resized"""
        # No need to resize if there is no size threshold.
        # To allow box sizing but no thresholding use `bg_thresh=0`.
        if self.bg_thresh is None or self.bg_rms is None:
            return False

        model = self.spectrum[:, None, None] * self.morph[None, :, :]
        bg_thresh = self.bg_rms * self.bg_thresh
        significant = np.any(model >= bg_thresh[:, None, None], axis=0)
        if np.sum(significant) == 0:
            # There are no significant pixels,
            # so make a small box around the center
            center = self.peak
            if center is None:
                center = (0, 0)
            new_box = Box((1, 1), center).grow(self.padding) & model_box
        else:
            new_box = (
                Box.from_data(significant, threshold=0).grow(self.padding) + self.bbox.origin  # type: ignore
            ) & model_box
        if new_box == self.bbox:
            return False

        old_box = self.bbox
        self._bbox = new_box
        self._morph.resize(old_box, new_box)
        return True

    def update(self, it: int, input_grad: np.ndarray):
        """Update the spectrum and morphology parameters"""
        # Store the input spectrum so that the morphology can
        # have a consistent update
        spectrum = self.spectrum.copy()
        self._spectrum.update(it, input_grad, self.morph)
        self._morph.update(it, input_grad, spectrum)

    def parameterize(self, parameterization: Callable) -> None:
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization: Callable
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        # Update the spectrum and morph in place
        parameterization(self)
        # update the parameters
        self._spectrum.grad = self.grad_spectrum
        self._spectrum.prox = self.prox_spectrum
        self._morph.grad = self.grad_morph
        self._morph.prox = self.prox_morph

    def __str__(self):
        result = (
            f"FactorizedComponent<\n    bands={self.bands},\n    center={self.peak},\n    "
            f"spectrum={self.spectrum},\n    morph_shape={self.morph.shape}\n>"
        )
        return result

    def __repr__(self):
        return self.__str__()


def default_fista_parameterization(component: Component):
    """Initialize a factorized component to use FISTA PGM for optimization"""
    if isinstance(component, FactorizedComponent):
        component._spectrum = FistaParameter(component.spectrum, step=0.5)
        component._morph = FistaParameter(component.morph, step=0.5)
    else:
        raise NotImplementedError(f"Unrecognized component type {component}")


def default_adaprox_parameterization(component: Component, noise_rms: float | None = None):
    """Initialize a factorized component to use Proximal ADAM
    for optimization
    """
    if noise_rms is None:
        noise_rms = 1e-16
    if isinstance(component, FactorizedComponent):
        component._spectrum = AdaproxParameter(
            component.spectrum,
            step=partial(relative_step, factor=1e-2, minimum=noise_rms),
        )
        component._morph = AdaproxParameter(
            component.morph,
            step=1e-2,
        )
    else:
        raise NotImplementedError(f"Unrecognized component type {component}")
