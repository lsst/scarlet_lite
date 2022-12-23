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
    "SedComponent",
    "ParametricComponent",
    "EllipticalParametricComponent",
]

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Sequence

import numpy as np
from scipy.special import erf
from scipy.stats import gamma

from .bbox import Box, get_minimal_boxsize, overlapped_slices
from .frame import CartesianFrame, EllipseFrame
from .image import Image
from .operators import MonotonicityConstraint
from .parameters import (
    parameter,
    Parameter,
    FistaParameter,
    AdaproxParameter,
    relative_step,
)
from .detect import scarlet_footprints_to_image


# Some operations fail at the origin in radial coordinates,
# so we make use of a very small offset.
MIN_RADIUS = 1e-20

# Useful constants
SQRT_PI_2 = np.sqrt(np.pi / 2)

# Stored sersic constants
SERSIC_B1 = gamma.ppf(0.5, 2)


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
        spectral_box = Box((len(bands),))
        self.result_box = spectral_box @ bbox
        self.slices = overlapped_slices(spectral_box @ model_bbox, spectral_box @ bbox)

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
        it: int
            The current iteration of the optimizer.
        input_grad: np.ndarray
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

    @abstractmethod
    def clear_parameters(self):
        """Convert all of the parameters back into numpy arrays"""
        pass

    def __str__(self):
        return "Component"

    def __repr__(self):
        return "Component"


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
        fit_center_radius: int = 1,
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
        fit_center_radius:
            The number of pixels around the `center` to search
            for a higher flux value when applying monotonicity.
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

        # Initialize the monotonicity constraint
        self.monotonicity = MonotonicityConstraint(
            neighbor_weight="angle", min_gradient=0, fit_center_radius=fit_center_radius
        )
        self.floor = floor
        self.model_bbox = model_bbox

    @property
    def center(self) -> tuple[int, int] | None:
        """The center of the component

        Returns
        -------
        center: Sequence[int, int]
            The center of the component
        """
        return self._center

    @property
    def sed(self) -> np.ndarray:
        """The array of SED values"""
        return self._sed.view(np.ndarray)

    @property
    def morph(self) -> np.ndarray:
        """The array of morphology values"""
        return self._morph.view(np.ndarray)

    @property
    def shape(self) -> tuple:
        """Shape of the resulting model image"""
        return self.sed.shape + self.morph.shape

    def get_model(self) -> Image:
        """Build the model from the SED and morphology"""
        # The sed and morph might be Parameters,
        # so cast them as arrays in the model.
        sed = self.sed.view(np.ndarray)
        morph = self.morph.view(np.ndarray)
        model = sed[:, None, None] * morph[None, :, :]
        return Image(model, bands=self.bands, yx0=self.bbox.origin[-2:])

    def grad_sed(self, input_grad, sed, morph):
        """Gradient of the SED wrt. the component model"""
        _grad = np.zeros(self.result_box.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("...jk,jk", _grad, morph)

    def grad_morph(self, input_grad, morph, sed):
        """Gradient of the morph wrt. the component model"""
        _grad = np.zeros(self.result_box.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("i,i...", sed, _grad)

    def prox_sed(self, sed: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the SED"""
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        return sed

    def prox_morph(self, morph: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the morphology"""
        # monotonicity
        morph = self.monotonicity(morph)

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
        else:
            center = (
                self.center[0] - self.bbox.origin[-2],
                self.center[1] - self.bbox.origin[-1],
            )
        morph[center] = np.max([morph[center], self.floor])
        # Normalize the morphology
        morph[:] = morph / morph.max()
        return morph

    def resize(self) -> bool:
        """Test whether or not the component needs to be resized"""
        # No need to resize if there is no size threshold.
        # To allow box sizing but no thresholding use `bg_thresh=0`.
        if self.bg_thresh is None or self.bg_rms is None:
            return False

        morph = self.morph
        size = max(morph.shape)

        # shrink the box? peel the onion
        dist = 0
        while (
            np.all(morph[dist, :] == 0)
            and np.all(morph[-dist, :] == 0)
            and np.all(morph[:, dist] == 0)
            and np.all(morph[:, -dist] == 0)
        ):
            dist += 1

        new_size = get_minimal_boxsize(size - 2 * dist)
        if new_size < size:
            dist = (size - new_size) // 2
            self.bbox.origin = (
                self.bbox.origin[0],
                self.bbox.origin[1] + dist,
                self.bbox.origin[2] + dist,
            )
            self.bbox.shape = (self.bbox.shape[0], new_size, new_size)
            self._morph.shrink(dist)
            self.slices = overlapped_slices(self.model_bbox, self.bbox)
            return True

        # grow the box?
        model = self.get_model()
        edge_flux = np.array(
            [
                np.sum(model[:, 0]),
                np.sum(model[:, -1]),
                np.sum(model[0, :]),
                np.sum(model[-1, :]),
            ]
        )

        edge_mask = np.array(
            [
                np.sum(model[:, 0] > 0),
                np.sum(model[:, -1] > 0),
                np.sum(model[0, :] > 0),
                np.sum(model[-1, :] > 0),
            ]
        )

        if np.any(edge_flux / edge_mask > self.bg_thresh * self.bg_rms[:, None, None]):
            new_size = get_minimal_boxsize(size + 1)
            dist = (new_size - size) // 2
            self.bbox.origin = (
                self.bbox.origin[0],
                self.bbox.origin[1] - dist,
                self.bbox.origin[2] - dist,
            )
            self.bbox.shape = (self.bbox.shape[0], new_size, new_size)
            self._morph.grow(self.bbox.shape[1:], dist)
            self.slices = overlapped_slices(self.model_bbox, self.bbox)
            return True
        return False

    def update(self, it: int, input_grad: np.ndarray):
        """Update the SED and morphology parameters"""
        # Store the input SED so that the morphology can
        # have a consistent update
        sed = self.sed.copy()
        self._sed.update(it, input_grad, self.morph)
        self._morph.update(it, input_grad, sed)

    def parameterize(self, parameterization: Callable) -> None:
        """Convert the component parameter arrays into Parameter instances"""
        # Update the SED and morph in place
        parameterization(self)
        # update the parameters
        self._sed.grad = self.grad_sed
        self._sed.prox = self.prox_sed
        self._morph.grad = self.grad_morph
        self._morph.prox = self.prox_morph

    def clear_parameters(self):
        """Convert all of the parameters back into numpy arrays"""
        self._sed = self.sed
        self._morph = self.morph

    def __str__(self):
        return "FactorizedComponent"

    def __repr__(self):
        return "FactorizedComponent"


def default_fista_parameterization(component: Component):
    """Initialize a factorized component to use FISTA PGM for optimization"""
    if isinstance(component, FactorizedComponent):
        component._sed = FistaParameter(component.sed, step=0.5)
        component._morph = FistaParameter(component.morph, step=0.5)
    else:
        raise NotImplementedError(f"Unrecognized component type {component}")


def default_adaprox_parameterization(
    component: Component, noise_rms: float = None, max_prox_iter: int = 1
):
    """Initialize a factorized component to use Proximal ADAM for optimization"""
    if noise_rms is None:
        noise_rms = 1e-16
    if isinstance(component, FactorizedComponent):
        component._sed = AdaproxParameter(
            component.sed,
            step=partial(relative_step, factor=1e-2, minimum=noise_rms),
            max_prox_iter=max_prox_iter,
        )
        component._morph = AdaproxParameter(
            component.morph,
            step=1e-2,
            max_prox_iter=max_prox_iter,
        )
    else:
        raise NotImplementedError(f"Unrecognized component type {component}")


class SedComponent(FactorizedComponent):
    """Implements a free-form component

    With no constraints this component is typically either a garbage collector,
    or part of a set of components to deconvolve an image by separating out
    the different spectral components.
    """

    def __init__(
        self,
        bands: tuple,
        sed: np.ndarray | Parameter,
        morph: np.ndarray | Parameter,
        model_bbox: Box,
        bg_thresh: float = None,
        bg_rms: np.ndarray = None,
        floor: float = 1e-20,
        peaks: list[tuple[int, int]] = None,
        min_area: float = 0,
    ):
        """Initialize the component.

        See `FactorizedComponent` for a list of parameters not shown here.

        Parameters
        ----------
        peaks: `list` of `tuple`
            A set of ``(cy, cx)`` peaks for detected sources.
            If peak is not ``None`` then only pixels in the same "footprint"
            as one of the peaks are included in the morphology.
            If `peaks` is ``None`` then there is no constraint applied.
        min_area: float
            The minimum area for a peak.
            If `min_area` is not `None` then all regions of the morphology
            with fewer than `min_area` connected pixels are removed.
        """
        super().__init__(
            bands=bands,
            sed=sed,
            morph=morph,
            bbox=model_bbox,
            model_bbox=model_bbox,
            center=None,
            bg_rms=bg_rms,
            bg_thresh=bg_thresh,
            floor=floor,
        )

        self.peaks = peaks
        self.min_area = min_area
        self.slices = [slice(None), slice(None)]

    def prox_sed(self, sed: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the SED

        This differs from `FactorizedComponent` because an
        `SedComponent` has the SED normalized to unity.
        """
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        # Normalize the SED
        sed = sed / np.sum(sed)
        return sed

    def prox_morph(self, morph: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the morphology

        This is the main difference between an `SedComponent` and a
        `FactorizedComponent`, since this component has fewer constraints.
        """
        from .detect_pybind11 import get_connected_multipeak, get_footprints

        if self.bg_thresh is not None:
            bg_thresh = self.bg_rms * self.bg_thresh
            # Enforce background thresholding
            model = self.sed[:, None, None] * morph[None, :, :]
            morph[np.all(model < bg_thresh[:, None, None], axis=0)] = 0
        else:
            # enforce positivity
            morph[morph < 0] = 0

        if self.peaks is not None:
            morph = morph * get_connected_multipeak(morph > 0, self.peaks, 0)

        if self.min_area > 0:
            footprints = get_footprints(morph > 0, 4.0, self.min_area, 0, False)
            morph = morph * (scarlet_footprints_to_image(footprints, morph.shape) > 0)

        if np.all(morph == 0):
            morph[0, 0] = self.floor

        return morph

    def resize(self) -> bool:
        return False

    def __str__(self):
        return "SedComponent"

    def __repr__(self):
        return "SedComponent"


def gaussian2d(params: np.ndarray, ellipse: EllipseFrame) -> np.ndarray:
    """Model of a 2D elliptical gaussian

    Parameters
    ----------
    params: np.ndarray
        The parameters of the function.
        In this case there are none outside of the ellipticity
    ellipse: EllipseFrame
        The ellipse parameters to scale the radius in all directions.

    Returns
    -------
    result: np.ndarray
        The 2D guassian for the given ellipse parameters
    """
    return np.exp(-ellipse.r2_grid)


def grad_gaussian(
    input_grad: np.ndarray,
    params: np.ndarray,
    cls: Component,
    morph: np.ndarray,
    sed: np.ndarray,
    ellipse: EllipseFrame,
) -> np.ndarray:
    """Gradient of the the component model wrt the Gaussian
    morphology parameters

    Parameters
    ----------
    input_grad: np.ndarray
        Gradient of the likelihood wrt the component model
    params: np.ndarray
        The parameters of the morphology.
    cls: Component
        The component of the model that contains the morphology.
    morph: np.ndarray
        The model of the morphology.
    sed: np.ndarray
        The model of the SED.
    ellipse: EllipseFrame
        The ellipse parameters to scale the radius in all directions.
    """
    # Calculate the gradient of the likelihod
    # wrt the Gaussian e^-r**2
    _grad = np.zeros(cls.bbox.shape, dtype=morph.dtype)
    _grad[cls.slices[1]] = input_grad[cls.slices[0]]
    _grad = -morph * np.einsum("i,i...", sed, _grad)
    d_y0 = ellipse.grad_y0(_grad, True)
    d_x0 = ellipse.grad_x0(_grad, True)
    d_sigma_y = ellipse.grad_major(_grad, True)
    d_sigma_x = ellipse.grad_minor(_grad, True)
    d_theta = ellipse.grad_theta(_grad, True)
    return np.array([d_y0, d_x0, d_sigma_y, d_sigma_x, d_theta], dtype=params.dtype)


def circular_gaussian(center: Sequence[int], frame: CartesianFrame, sigma: float):
    """Model of a circularly symmetric Gaussian

    Parameters
    ----------
    center: np.ndarray
        The center of the Gaussian.
    frame: CartesianFrame
        The frame in which to generate the image of the circular Gaussian
    sigma: float
        The standard deviation.

    Returns
    -------
    result: np.ndarray
        The image of the circular Gaussian.
    """
    y0, x0 = center[:2]
    two_sigma = 2 * sigma
    r2 = ((frame.x_grid - x0) / two_sigma) ** 2 + ((frame.y_grid - y0) / two_sigma) ** 2
    return np.exp(-r2)


def grad_circular_gaussian(
    input_grad: np.ndarray,
    params: np.ndarray,
    cls: Component,
    morph: np.ndarray,
    sed: np.ndarray,
    frame: CartesianFrame,
    sigma: float,
) -> np.ndarray:
    """Gradient of the the component model wrt the Gaussian
    morphology parameters

    Parameters
    ----------
    input_grad: np.ndarray
        Gradient of the likelihood wrt the component model
    params: np.ndarray
        The parameters of the morphology.
    cls: Component
        The component of the model that contains the morphology.
    morph: np.ndarray
        The model of the morphology.
    sed: np.ndarray
        The model of the SED.
    frame: CartesianFrame
        The frame in which to generate the image of the circular Gaussian.
    sigma: float
        The standard deviation.
    """
    # Calculate the gradient of the likelihod
    # wrt the Gaussian e^-r**2
    _grad = np.zeros(cls.bbox.shape, dtype=morph.dtype)
    _grad[cls.slices[1]] = input_grad[cls.slices[0]]
    _grad = -morph * np.einsum("i,i...", sed, _grad)

    y0, x0 = params[:2]
    d_y0 = -2 * np.sum((frame.y_grid - y0) * _grad)
    d_x0 = -2 * np.sum((frame.x_grid - x0) * _grad)
    return np.array([d_y0, d_x0], dtype=params.dtype)


def integrated_gaussian(params: np.ndarray, frame: CartesianFrame):
    """Model of a circularly symmetric Gaussian integrated over pixels

    This differs from `circularGaussian` because the gaussian function
    is integrated over each pixel to replicate the pixelated image
    version of a Gaussian function.

    Parameters
    ----------
    params: np.ndarray
        The center of the Gaussian.
    frame: CartesianFrame
        The frame in which to generate the image of the circular Gaussian

    Returns
    -------
    result: np.ndarray
        The image of the circular Gaussian.
    """
    # Unpack the parameters and define constants
    y0, x0, sigma = params
    r = np.sqrt((frame.x_grid - x0) ** 2 + (frame.y_grid - y0) ** 2)
    sqrt_c = 1 / np.sqrt(2) / sigma
    # Integrate from half a pixel left and right
    lhs = erf((r - 0.5) * sqrt_c)
    rhs = erf((r + 0.5) * sqrt_c)
    z = 0.5 * np.sqrt(np.pi) / sqrt_c * (rhs - lhs)
    return z


def grad_integrated_gaussian(
    input_grad: np.ndarray,
    params: np.ndarray,
    cls: Component,
    morph: np.ndarray,
    sed: np.ndarray,
    frame: CartesianFrame,
) -> np.ndarray:
    """Gradient of the the component model wrt the Gaussian
    morphology parameters

    Parameters
    ----------
    input_grad: np.ndarray
        Gradient of the likelihood wrt the component model
    params: np.ndarray
        The parameters of the morphology.
    cls: Component
        The component of the model that contains the morphology.
    morph: np.ndarray
        The model of the morphology.
    sed: np.ndarray
        The model of the SED.
    frame: CartesianFrame
        The frame in which to generate the image of the circular Gaussian.
    """
    # Calculate the gradient of the likelihood
    # wrt the Gaussian e^-r**2
    _grad = np.zeros(cls.bbox.shape, dtype=morph.dtype)
    _grad[cls.slices[1]] = input_grad[cls.slices[0]]
    _grad = np.einsum("i,i...", sed, _grad)

    # Extract the parameters
    y0, x0, sigma = params
    # define useful constants
    x = frame.x_grid - x0
    y = frame.y_grid - y0
    c = 0.5 / sigma**2
    sqrt_c = np.sqrt(c)
    # Add a small constant to the radius to prevent a divergence at r==0
    r = np.sqrt(x**2 + y**2 + MIN_RADIUS)
    # Shift half a pixel in each direction for the integration
    r1 = r - 0.5
    r2 = r + 0.5
    # Calculate the gradient of the ERF wrt. each shifted radius
    d_model1 = np.exp(-c * r1**2)
    d_model2 = np.exp(-c * r2**2)
    # Calculate the gradients of the parameters
    d_x0 = np.sum(-x / r * (d_model2 - d_model1) * _grad)
    d_y0 = np.sum(-y / r * (d_model2 - d_model1) * _grad)
    d_sigma1 = -(r1 * d_model1 / sigma - SQRT_PI_2 * erf(r1 * sqrt_c))
    d_sigma2 = -(r2 * d_model2 / sigma - SQRT_PI_2 * erf(r2 * sqrt_c))
    d_sigma = np.sum((d_sigma2 - d_sigma1) * _grad)

    return np.array([d_y0, d_x0, d_sigma])


def bounded_prox(
    params: np.ndarray, proxmin: np.ndarray, proxmax: np.ndarray
) -> np.ndarray:
    """A bounded proximal operator

    This function updates `params` in place.

    Parameters
    ----------
    params: np.ndarray
        The array of parameters to constrain.
    proxmin: np.ndarray
        The array of minimum values for each parameter.
    proxmax: np.ndarray
        The array of maximum values for each parameter.

    Returns
    -------
    result: np.ndarray
        The updated parameters.
    """
    cuts = params < proxmin
    params[cuts] = proxmin[cuts]
    cuts = params > proxmax
    params[cuts] = proxmax[cuts]
    return params


def sersic(params: np.ndarray, ellipse: EllipseFrame):
    """Generate a Sersic Model.

    Parameters
    ----------
    params: np.ndarray
        The parameters of the function.
        In this case the only parameter is the sersic index ``n``.
    ellipse: EllipseFrame
        The ellipse parameters to scale the radius in all directions.

    Returns
    -------
    result: np.ndarray
        The 2D guassian for the given ellipse parameters
    """
    (n,) = params

    r = ellipse.r_grid

    if n == 1:
        result = np.exp(-SERSIC_B1 * r)
    else:
        bn = gamma.ppf(0.5, 2 * n)
        result = np.exp(-bn * (r ** (1 / n) - 1))
    return result


def grad_sersic(
    input_grad: np.ndarray,
    params: np.ndarray,
    cls: Component,
    morph: np.ndarray,
    sed: np.ndarray,
    ellipse: EllipseFrame,
):
    """Gradient of the component model wrt the Gaussian morphology parameters

    Parameters
    ----------
    input_grad: np.ndarray
        Gradient of the likelihood wrt the component model
    params: np.ndarray
        The parameters of the morphology.
    cls: Component
        The component of the model that contains the morphology.
    morph: np.ndarray
        The model of the morphology.
    sed: np.ndarray
        The model of the SED.
    ellipse: EllipseFrame
        The ellipse parameters to scale the radius in all directions.
    """
    n = params[5]
    bn = gamma.ppf(0.5, 2 * n)
    if n == 1:
        # Use a simplified model for faster calculation
        d_exp = -SERSIC_B1 * morph
    else:
        r = ellipse.r_grid
        d_exp = -bn / n * morph * r ** (1 / n - 1)

    _grad = np.zeros(cls.bbox.shape, dtype=morph.dtype)
    _grad[cls.slices[1]] = input_grad[cls.slices[0]]
    _grad = np.einsum("i,i...", sed, _grad)
    d_n = np.sum(
        _grad
        * bn
        * morph
        * ellipse.r_grid ** (1 / n)
        * np.log10(ellipse.r_grid)
        / n**2
    )
    _grad = _grad * d_exp
    d_y0 = ellipse.grad_y0(_grad, False)
    d_x0 = ellipse.grad_x0(_grad, False)
    d_sigma_y = ellipse.grad_major(_grad, False)
    d_sigma_x = ellipse.grad_minor(_grad, False)
    d_theta = ellipse.grad_theta(_grad, False)
    return np.array(
        [d_y0, d_x0, d_sigma_y, d_sigma_x, d_theta, d_n], dtype=params.dtype
    )


class ParametricComponent(Component):
    """A parametric model of an astrophysical source"""

    def __init__(
        self,
        bands: tuple,
        bbox: Box,
        model_bbox: Box,
        sed: Parameter | np.ndarray,
        morph_params: Parameter | np.ndarray,
        morph_func: Callable,
        morph_grad: Callable,
        morph_prox: Callable,
        morph_step: Callable,
        model_frame: CartesianFrame,
        prox_sed: Callable = None,
        floor: float = 1e-20,
    ):
        """Initialize the component

        Parameters
        ----------
        bands:
            The bands used in the model.
        bbox:
            The bounding box that holds the model.
        model_bbox:
            The bounding box for the full blend model.
        sed:
            The SED of the component.
        morph_params:
            The parameters of the morphology.
        morph_func:
            The function to generate the 2D morphology image
            based on `morphParams`.
        morph_grad:
            The function to calculate the gradient of the
            likelihood wrt the morphological parameters.
        morph_prox:
            The proximal operator for the morphology parameters.
        morph_step:
            The function that calculates the gradient of the
            morphological model.
        prox_sed:
            Proximal operator for the SED.
            If `prox_sed` is `None` then the default proximal
            operator `self.prox_sed` is used.
        floor:
            The minimum value of the SED, used to prevent
            divergences in the gradients.
        """
        super().__init__(bands=bands, bbox=bbox, model_bbox=model_bbox)

        self._sed = parameter(sed)
        self._params = parameter(morph_params)
        self._func = morph_func
        self._morph_grad = morph_grad
        self._morph_prox = morph_prox
        self._morph_step = morph_step
        self._sed = sed
        self._bbox = bbox
        if prox_sed is None:
            self._prox_sed = self.prox_sed
        else:
            self._prox_sed = prox_sed
        self.slices = overlapped_slices(model_frame.bbox, bbox)
        self.floor = floor

    @property
    def center(self) -> tuple[float, float]:
        """The center of the component"""
        return self.y0, self.x0

    @property
    def y0(self) -> float:
        """The y-center of the component"""
        return self._params[0]

    @property
    def x0(self) -> float:
        """The x-center of the component"""
        return self._params[1]

    @property
    def sed(self) -> np.ndarray:
        """The array of SED values"""
        return self._sed.view(np.ndarray)

    @property
    def bbox(self) -> Box:
        """The bounding box that contains the component"""
        return self._bbox

    @property
    def frame(self) -> CartesianFrame:
        """The coordinate system that contains the model"""
        return CartesianFrame(self._bbox)

    @property
    def radial_params(self) -> np.ndarray:
        """The parameters used to model the radial function"""
        return self._params.view(np.ndarray)

    def _get_morph(self, frame: CartesianFrame = None) -> np.ndarray:
        """The 2D image of the morphology

        This callable generates an image of the morphology
        in the given frame.

        Parameters
        ----------
        frame: CartesianFrame
            The frame (bounding box, pixel grid) that the image is
            placed in.

        Returns
        -------
        result: np.ndarray
            The image of the morphology in the `frame`.
        """
        if frame is None:
            frame = self.frame
        return self._func(self.radial_params, frame)

    @property
    def morph(self, frame: CartesianFrame = None) -> np.ndarray:
        """The morphological model"""
        return self._get_morph(frame)

    @property
    def prox_morph(self) -> Callable:
        """The function used to constrain the morphological model"""
        return self._morph_prox

    @property
    def grad_morph(self) -> Callable:
        """The function that calculates the gradient of the
        morphological model
        """
        return self._morph_grad

    @property
    def morph_step(self) -> Callable:
        """The function that calculates the gradient of the
        morphological model
        """
        return self._morph_step

    def get_model(self, frame: CartesianFrame = None) -> Image:
        """Generate the full model for this component"""
        model = self.sed[:, None, None] * self._get_morph(frame)[None, :, :]
        return Image(model, bands=self.bands, yx0=self.bbox.origin[-2:])

    def prox_sed(self, sed: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the SED

        Parameters
        ----------
        sed: np.ndarray
            The SED of the model.
        """
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        return sed

    def grad_sed(
        self, input_grad: np.ndarray, sed: np.ndarray, morph: np.ndarray
    ) -> np.ndarray:
        """Gradient of the SED wrt. the component model

        Parameters
        ----------
        input_grad: np.ndarray
            Gradient of the likelihood wrt the component model
        sed: np.ndarray
            The model of the SED.
        morph: np.ndarray
            The model of the morphology.

        Returns
        -------
        result: np.ndarray
            The gradient of the likelihood wrt. the SED.
        """
        _grad = np.zeros(self.bbox.shape, dtype=self.sed.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("...jk,jk", _grad, morph)

    def update(self, it: int, input_grad: np.ndarray):
        """Update the component parameters from an input gradient

        Parameters
        ----------
        it: int
            The current iteration of the optimizer.
        input_grad: np.ndarray
            Gradient of the likelihood wrt the component model
        """
        sed = self.sed.copy()
        morph = self.morph
        self._sed.update(it, input_grad, morph)
        self._params.update(it, input_grad, self, morph, sed, self.frame)

    def resize(self) -> bool:
        """Resize the box that contains the model

        Not yet implemented, so for now the model box
        does not grow. If this is ever implemented in production,
        in the long run this will be based on a cutoff value for the model.
        """
        return False

    def parameterize(self, parameterization: Callable) -> None:
        """Convert the component parameter arrays into Parameter instances"""
        # Update the SED and morph in place
        parameterization(self)
        # update the parameters
        self._sed.grad = self.grad_sed
        self._sed.prox = self.prox_sed
        self._params.grad = self.grad_morph
        self._params.prox = self.prox_morph

    def clear_parameters(self):
        """Convert all of the parameters back into numpy arrays"""
        self._sed = self.sed
        self._params = self.radial_params


class EllipticalParametricComponent(ParametricComponent):
    """A radial density/surface brightness profile with elliptical symmetry"""

    def __init__(
        self,
        bands: tuple,
        bbox: Box,
        model_bbox: Box,
        sed: np.ndarray | Parameter,
        morph_params: np.ndarray | Parameter,
        morph_func: Callable,
        morph_grad: Callable,
        morph_prox: Callable,
        morph_step: Callable,
        model_frame: CartesianFrame,
        prox_sed: Callable = None,
        floor: float = 1e-20,
    ):
        """Initialize the component

        Parameters
        ----------
        bands:
            The bands used in the model.
        bbox: Box
            The bounding box that holds this component model.
        model_bbox: Box
            The bounding box that holds the entire blend.
        sed: np.ndarray
            The SED of the component.
        morph_params: np.ndarray
            The parameters passed to `morph_func` to
            generate the morphology in image space.
        morph_func: Callable
            The function to generate the morphology
            based on `morphParams`.
        morph_grad: Callable
            The function to calculate the gradient of the
            likelihood wrt the morphological parameters.
        morph_prox: Callable
            The proximal operator for the morphology parameters.
        model_frame: CartesianFrame
            The coordinates of the model frame,
            used to speed up the creation of the
            polar grid for each source.
        prox_sed: Callable
            Proximal operator for the SED.
            If `prox_sed` is `None` then the default proximal
            operator `self.prox_sed` is used.
        floor: float
            The minimum value of the SED, used to prevent
            divergences in the gradients.
        """
        super().__init__(
            bands=bands,
            bbox=bbox,
            model_bbox=model_bbox,
            sed=sed,
            morph_params=morph_params,
            morph_func=morph_func,
            morph_grad=morph_grad,
            morph_prox=morph_prox,
            morph_step=morph_step,
            model_frame=model_frame,
            prox_sed=prox_sed,
            floor=floor,
        )

    @property
    def semi_major(self) -> float:
        """The length of the semi-major axis of the model"""
        return self._params[2]

    @property
    def semi_minor(self) -> float:
        """The length of the semi-minor axis of the model"""
        return self._params[3]

    @property
    def theta(self) -> float:
        """The counter-clockwise rotation angle of the model from the x-axis."""
        return self._params[4]

    @property
    def ellipse_params(self) -> np.ndarray:
        """The parameters used to generate the scaled radius"""
        return self._params[:5].view(np.ndarray)

    @property
    def radial_params(self) -> np.ndarray:
        """The parameters used to model the radial function"""
        return self._params[5:].view(np.ndarray)

    @property
    def frame(self) -> EllipseFrame:
        """The `EllipseFrame` that parameterizes the model"""
        return EllipseFrame(*self.ellipse_params, self._bbox)  # type: ignore

    @property
    def morph_prox(self) -> Callable:
        """The function used to constrain the morphological model"""
        return self._morph_prox

    @property
    def morph_grad(self) -> Callable:
        """The function that calculates the gradient of the morphological model"""
        return self._morph_grad

    def update(self, it: int, input_grad: np.ndarray):
        """Update the component

        Parameters
        ----------
        it: int
            The current iteration of the optimizer.
        input_grad: np.ndarray
            Gradient of the likelihood wrt the component model
        """
        ellipse = self.frame
        sed = self.sed.copy()
        morph = self._func(self.radial_params, ellipse)
        self._sed.update(it, input_grad, morph)
        self._params.update(it, input_grad, self, morph, sed, ellipse)
