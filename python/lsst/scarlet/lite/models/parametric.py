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

__all__ = [
    "bounded_prox",
    "gaussian2d",
    "grad_gaussian2",
    "circular_gaussian",
    "grad_circular_gaussian",
    "integrated_gaussian",
    "grad_integrated_gaussian",
    "sersic",
    "grad_sersic",
    "CartesianFrame",
    "EllipseFrame",
    "ParametricComponent",
    "EllipticalParametricComponent",
]

from typing import TYPE_CHECKING, Callable, Sequence, cast

import numpy as np
from scipy.special import erf
from scipy.stats import gamma

from ..bbox import Box
from ..component import Component
from ..image import Image
from ..parameters import Parameter, parameter

if TYPE_CHECKING:
    from ..io import ScarletComponentBaseData

# Some operations fail at the origin in radial coordinates,
# so we make use of a very small offset.
MIN_RADIUS = 1e-20

# Useful constants
SQRT_PI_2 = np.sqrt(np.pi / 2)

# Stored sersic constants
SERSIC_B1 = gamma.ppf(0.5, 2)


class CartesianFrame:
    """A grid of X and Y values contained in a bbox"""

    def __init__(self, bbox: Box):
        """
        Parameters
        ----------
        bbox: Box
            The bounding box that contains this frame.
        """
        # Store the new bounding box
        self._bbox = bbox
        # Get the range of x and y
        yi, xi = bbox.start[-2:]
        yf, xf = bbox.stop[-2:]
        height, width = bbox.shape[-2:]
        y = np.linspace(yi, yf - 1, height)
        x = np.linspace(xi, xf - 1, width)
        # Create the grid used to create the image of the frame
        self._x, self._y = np.meshgrid(x, y)
        self._r = None
        self._r2 = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the frame."""
        return self._bbox.shape

    @property
    def bbox(self) -> Box:
        """Bounding box containing the frame"""
        return self._bbox

    @property
    def x_grid(self) -> np.ndarray:
        """The grid of x-values for the entire frame"""
        return self._x

    @property
    def y_grid(self) -> np.ndarray:
        """The grid of y-values for the entire frame"""
        return self._y


class EllipseFrame(CartesianFrame):
    """Frame to scale the radius based on the parameters of an ellipse

    This frame is used to calculate the coordinates of the
    radius and radius**2 from a given center location,
    based on the semi-major axis, semi-minor axis, and rotation angle.
    It is also used to calculate the gradient wrt either the
    radius**2 or radius for all of the model parameters.

    Parameters
    ----------
    y0: float
        The y-center of the ellipse.
    x0: float
        The x-center of the ellipse.
    major: float
        The length of the semi-major axis.
    minor: float
        The length of the semi-minor axis.
    theta: float
        The counter-clockwise rotation angle
        from the semi-major axis.
    bbox: Box
        The bounding box that contains the entire frame.
    r_min: float
        The minimum value of the radius.
        This is used to prevent divergences that occur
        when calculating the gradient at radius == 0.
    """

    def __init__(
        self,
        y0: float,
        x0: float,
        major: float,
        minor: float,
        theta: float,
        bbox: Box,
        r_min: float = 1e-20,
    ):
        super().__init__(bbox)
        # Set some useful parameters for derivations
        sin = np.sin(theta)
        cos = np.cos(theta)

        # Rotate into the frame with xMajor as the x-axis
        # and xMinor as the y-axis
        self._xMajor = (self._x - x0) * cos + (self._y - y0) * sin
        self._xMinor = -(self._x - x0) * sin + (self._y - y0) * cos
        # The scaled major and minor axes
        self._xa = self._xMajor / major
        self._yb = self._xMinor / minor

        # Store parameters useful for gradient calculation
        self._y0, self._x0 = y0, x0
        self._theta = theta
        self._major = major
        self._minor = minor
        self._sin, self._cos = sin, cos
        self._bbox = bbox
        self._rMin = r_min
        # Store the scaled radius**2 and radius
        self._radius2 = self._xa**2 + self._yb**2
        self._radius: np.ndarray | None = None

    def grad_x0(self, input_grad: np.ndarray, use_r2: bool) -> float:
        """The gradient of either the radius or radius**2 wrt. the x-center

        Parameters
        ----------
        input_grad:
            Gradient of the likelihood wrt the component model
        use_r2:
            Whether to calculate the gradient of the radius**2
            (``useR2==True``) or the radius (``useR2==False``).

        Returns
        -------
        result:
            The gradient of the likelihood wrt x0.
        """
        grad = -self._xa * self._cos / self._major + self._yb * self._sin / self._minor
        if use_r2:
            grad *= 2
        else:
            r = self.r_grid
            grad *= 1 / r
        return np.sum(grad * input_grad)  # type: ignore

    def grad_y0(self, input_grad: np.ndarray, use_r2: bool) -> float:
        """The gradient of either the radius or radius**2 wrt. the y-center

        Parameters
        ----------
        input_grad:
            Gradient of the likelihood wrt the component model
        use_r2:
            Whether to calculate the gradient of the radius**2
            (``useR2==True``) or the radius (``useR2==False``).

        Returns
        -------
        result:
            The gradient of the likelihood wrt y0.
        """
        grad = -(self._xa * self._sin / self._major + self._yb * self._cos / self._minor)
        if use_r2:
            grad *= 2
        else:
            r = self.r_grid
            grad *= 1 / r
        return np.sum(grad * input_grad)  # type: ignore

    def grad_major(self, input_grad: np.ndarray, use_r2: bool) -> float:
        """The gradient of either the radius or radius**2 wrt.
        the semi-major axis

        Parameters
        ----------
        input_grad:
            Gradient of the likelihood wrt the component model
        use_r2:
            Whether to calculate the gradient of the radius**2
            (``use_r2==True``) or the radius (``use_r2==False``).

        Returns
        -------
        result:
            The gradient of the likelihood wrt the semi-major axis.
        """
        grad = -2 / self._major * self._xa**2
        if use_r2:
            grad *= 2
        else:
            r = self.r_grid
            grad *= 1 / r
        return np.sum(grad * input_grad)  # type: ignore

    def grad_minor(self, input_grad: np.ndarray, use_r2: bool) -> float:
        """The gradient of either the radius or radius**2 wrt.
        the semi-minor axis

        Parameters
        ----------
        input_grad:
            Gradient of the likelihood wrt the component model
        use_r2:
            Whether to calculate the gradient of the radius**2
            (``useR2==True``) or the radius (``useR2==False``).

        Returns
        -------
        result:
            The gradient of the likelihood wrt the semi-minor axis.
        """
        grad = -2 / self._minor * self._yb**2
        if use_r2:
            grad *= 2
        else:
            r = self.r_grid
            grad *= 1 / r
        return np.sum(grad * input_grad)  # type: ignore

    def grad_theta(self, input_grad: np.ndarray, use_r2: bool) -> float:
        """The gradient of either the radius or radius**2 wrt.
        the rotation angle

        Parameters
        ----------
        input_grad:
            Gradient of the likelihood wrt the component model
        use_r2:
            Whether to calculate the gradient of the radius**2
            (``useR2==True``) or the radius (``useR2==False``).

        Returns
        -------
        result:
            The gradient of the likelihood wrt the rotation angle.
        """
        grad = self._xa * self._xMinor / self._major - self._yb * self._xMajor / self._minor
        if use_r2:
            grad *= 2
        else:
            r = self.r_grid
            grad *= 1 / r
        return np.sum(grad * input_grad)  # type: ignore

    @property
    def x0(self) -> float:
        """The x-center"""
        return self._x0

    @property
    def y0(self) -> float:
        """The y-center"""
        return self._y0

    @property
    def major(self) -> float:
        """The semi-major axis"""
        return self._major

    @property
    def minor(self) -> float:
        """The semi-minor axis"""
        return self._minor

    @property
    def theta(self) -> float:
        """The rotation angle"""
        return self._theta

    @property
    def bbox(self) -> Box:
        """The bounding box to hold the model"""
        return self._bbox

    @property
    def r_grid(self) -> np.ndarray:
        """The radial coordinates of each pixel"""
        if self._radius is None:
            self._radius = np.sqrt(self.r2_grid)
        return self._radius

    @property
    def r2_grid(self) -> np.ndarray:
        """The radius squared located at each pixel"""
        return self._radius2 + self._rMin**2


def gaussian2d(params: np.ndarray, ellipse: EllipseFrame) -> np.ndarray:
    """Model of a 2D elliptical gaussian

    Parameters
    ----------
    params:
        The parameters of the function.
        In this case there are none outside of the ellipticity
    ellipse:
        The ellipse parameters to scale the radius in all directions.

    Returns
    -------
    result:
        The 2D guassian for the given ellipse parameters
    """
    return np.exp(-ellipse.r2_grid)


def grad_gaussian2(
    input_grad: np.ndarray,
    params: np.ndarray,
    morph: np.ndarray,
    spectrum: np.ndarray,
    ellipse: EllipseFrame,
) -> np.ndarray:
    """Gradient of the the component model wrt the Gaussian
    morphology parameters

    Parameters
    ----------
    input_grad:
        Gradient of the likelihood wrt the component model
    params:
        The parameters of the morphology.
    morph:
        The model of the morphology.
    spectrum:
        The model of the spectrum.
    ellipse:
        The ellipse parameters to scale the radius in all directions.
    """
    # Calculate the gradient of the likelihod
    # wrt the Gaussian e^-r**2
    _grad = -morph * np.einsum("i,i...", spectrum, input_grad)
    d_y0 = ellipse.grad_y0(_grad, True)
    d_x0 = ellipse.grad_x0(_grad, True)
    d_sigma_y = ellipse.grad_major(_grad, True)
    d_sigma_x = ellipse.grad_minor(_grad, True)
    d_theta = ellipse.grad_theta(_grad, True)
    return np.array([d_y0, d_x0, d_sigma_y, d_sigma_x, d_theta], dtype=params.dtype)


def circular_gaussian(center: Sequence[int], frame: CartesianFrame, sigma: float) -> np.ndarray:
    """Model of a circularly symmetric Gaussian

    Parameters
    ----------
    center:
        The center of the Gaussian.
    frame:
        The frame in which to generate the image of the circular Gaussian
    sigma:
        The standard deviation.

    Returns
    -------
    result:
        The image of the circular Gaussian.
    """
    y0, x0 = center[:2]
    two_sigma = 2 * sigma
    r2 = ((frame.x_grid - x0) / two_sigma) ** 2 + ((frame.y_grid - y0) / two_sigma) ** 2
    return np.exp(-r2)


def grad_circular_gaussian(
    input_grad: np.ndarray,
    params: np.ndarray,
    morph: np.ndarray,
    spectrum: np.ndarray,
    frame: CartesianFrame,
    sigma: float,
) -> np.ndarray:
    """Gradient of the the component model wrt the Gaussian
    morphology parameters

    Parameters
    ----------
    input_grad:
        Gradient of the likelihood wrt the component model
    params:
        The parameters of the morphology.
    morph:
        The model of the morphology.
    spectrum:
        The model of the spectrum.
    frame:
        The frame in which to generate the image of the circular Gaussian.
    sigma:
        The standard deviation.
    """
    # Calculate the gradient of the likelihod
    # wrt the Gaussian e^-r**2

    _grad = -morph * np.einsum("i,i...", spectrum, input_grad)

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
    params:
        The center of the Gaussian.
    frame:
        The frame in which to generate the image of the circular Gaussian

    Returns
    -------
    result:
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
    morph: np.ndarray,
    spectrum: np.ndarray,
    frame: CartesianFrame,
) -> np.ndarray:
    """Gradient of the component model wrt the Gaussian
    morphology parameters

    Parameters
    ----------
    input_grad:
        Gradient of the likelihood wrt the component model parameters.
    params:
        The parameters of the morphology.
    morph:
        The model of the morphology.
    spectrum:
        The model of the spectrum.
    frame:
        The frame in which to generate the image of the circular Gaussian.

    Returns
    -------
    result:
        The gradient of the component morphology.
    """
    # Calculate the gradient of the likelihood
    # wrt the Gaussian e^-r**2
    _grad = np.einsum("i,i...", spectrum, input_grad)

    # Extract the parameters
    y0, x0, sigma = params
    # define useful constants
    x = frame.x_grid - x0
    y = frame.y_grid - y0
    c = 0.5 / sigma**2
    sqrt_c = np.sqrt(c)
    # Add a small constant to the radius to prevent a divergence at r==0
    r = np.sqrt(x**2 + y**2) + MIN_RADIUS
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


def bounded_prox(params: np.ndarray, proxmin: np.ndarray, proxmax: np.ndarray) -> np.ndarray:
    """A bounded proximal operator

    This function updates `params` in place.

    Parameters
    ----------
    params:
        The array of parameters to constrain.
    proxmin:
        The array of minimum values for each parameter.
    proxmax:
        The array of maximum values for each parameter.

    Returns
    -------
    result:
        The updated parameters.
    """
    cuts = params < proxmin
    params[cuts] = proxmin[cuts]
    cuts = params > proxmax
    params[cuts] = proxmax[cuts]
    return params


def sersic(params: np.ndarray, ellipse: EllipseFrame) -> np.ndarray:
    """Generate a Sersic Model.

    Parameters
    ----------
    params:
        The parameters of the function.
        In this case the only parameter is the sersic index ``n``.
    ellipse:
        The ellipse parameters to scale the radius in all directions.

    Returns
    -------
    result:
        The model for the given ellipse parameters
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
    morph: np.ndarray,
    spectrum: np.ndarray,
    ellipse: EllipseFrame,
) -> np.ndarray:
    """Gradient of the component model wrt the morphology parameters

    Parameters
    ----------
    input_grad:
        Gradient of the likelihood wrt the component model
    params:
        The parameters of the morphology.
    morph:
        The model of the morphology.
    spectrum:
        The model of the spectrum.
    ellipse:
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

    _grad = np.einsum("i,i...", spectrum, input_grad)
    d_n = np.sum(_grad * bn * morph * ellipse.r_grid ** (1 / n) * np.log10(ellipse.r_grid) / n**2)
    _grad = _grad * d_exp
    d_y0 = ellipse.grad_y0(_grad, False)
    d_x0 = ellipse.grad_x0(_grad, False)
    d_sigma_y = ellipse.grad_major(_grad, False)
    d_sigma_x = ellipse.grad_minor(_grad, False)
    d_theta = ellipse.grad_theta(_grad, False)
    return np.array([d_y0, d_x0, d_sigma_y, d_sigma_x, d_theta, d_n], dtype=params.dtype)


class ParametricComponent(Component):
    """A parametric model of an astrophysical source

    Parameters
    ----------
    bands:
        The bands used in the model.
    bbox:
        The bounding box that holds the model.
    spectrum:
        The spectrum of the component.
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
    prox_spectrum:
        Proximal operator for the spectrum.
        If `prox_spectrum` is `None` then the default proximal
        operator `self.prox_spectrum` is used.
    floor:
        The minimum value of the spectrum, used to prevent
        divergences in the gradients.
    """

    def __init__(
        self,
        bands: tuple,
        bbox: Box,
        spectrum: Parameter | np.ndarray,
        morph_params: Parameter | np.ndarray,
        morph_func: Callable,
        morph_grad: Callable,
        morph_prox: Callable,
        morph_step: Callable | np.ndarray,
        prox_spectrum: Callable | None = None,
        floor: float = 1e-20,
    ):
        super().__init__(bands=bands, bbox=bbox)

        self._spectrum = parameter(spectrum)
        self._params = parameter(morph_params)
        self._func = morph_func
        self._morph_grad = morph_grad
        self._morph_prox = morph_prox
        self._morph_step = morph_step
        self._bbox = bbox
        if prox_spectrum is None:
            self._prox_spectrum: Callable = self.prox_spectrum
        else:
            self._prox_spectrum = prox_spectrum
        self.floor = floor

    @property
    def peak(self) -> tuple[float, float]:
        """The center of the component"""
        return self.y0, self.x0

    @property
    def y0(self) -> float:
        """The y-center of the component"""
        return self._params.x[0]

    @property
    def x0(self) -> float:
        """The x-center of the component"""
        return self._params.x[1]

    @property
    def spectrum(self) -> np.ndarray:
        """The array of spectrum values"""
        return self._spectrum.x

    @property
    def frame(self) -> CartesianFrame:
        """The coordinate system that contains the model"""
        return CartesianFrame(self._bbox)

    @property
    def radial_params(self) -> np.ndarray:
        """The parameters used to model the radial function"""
        return self._params.x

    def _get_morph(self, frame: CartesianFrame | None = None) -> np.ndarray:
        """The 2D image of the morphology

        This callable generates an image of the morphology
        in the given frame.

        Parameters
        ----------
        frame:
            The frame (bounding box, pixel grid) that the image is
            placed in.

        Returns
        -------
        result:
            The image of the morphology in the `frame`.
        """
        if frame is None:
            frame = self.frame
        return self._func(self.radial_params, frame)

    @property
    def morph(self, frame: CartesianFrame | None = None) -> np.ndarray:
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
        return cast(Callable, self._morph_step)

    def get_model(self, frame: CartesianFrame | None = None) -> Image:
        """Generate the full model for this component"""
        model = self.spectrum[:, None, None] * self._get_morph(frame)[None, :, :]
        return Image(model, bands=self.bands, yx0=cast(tuple[int, int], self.bbox.origin[-2:]))

    def prox_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the spectrum

        Parameters
        ----------
        spectrum:
            The spectrum of the model.
        """
        # prevent divergent spectrum
        spectrum[spectrum < self.floor] = self.floor
        return spectrum

    def grad_spectrum(self, input_grad: np.ndarray, spectrum: np.ndarray, morph: np.ndarray) -> np.ndarray:
        """Gradient of the spectrum wrt. the component model

        Parameters
        ----------
        input_grad:
            Gradient of the likelihood wrt the component model
        spectrum:
            The model of the spectrum.
        morph:
            The model of the morphology.

        Returns
        -------
        result:
            The gradient of the likelihood wrt. the spectrum.
        """
        return np.einsum("...jk,jk", input_grad, morph)

    def update(self, it: int, input_grad: np.ndarray):
        """Update the component parameters from an input gradient

        Parameters
        ----------
        it:
            The current iteration of the optimizer.
        input_grad:
            Gradient of the likelihood wrt the component model
        """
        spectrum = self.spectrum.copy()
        morph = self.morph
        self._spectrum.update(it, input_grad, morph)
        self._params.update(it, input_grad, morph, spectrum, self.frame)

    def resize(self, model_box: Box) -> bool:
        """Resize the box that contains the model

        Not yet implemented, so for now the model box
        does not grow. If this is ever implemented in production,
        in the long run this will be based on a cutoff value for the model.
        """
        return False

    def parameterize(self, parameterization: Callable) -> None:
        """Convert the component parameter arrays into Parameter instances"""
        # Update the spectrum and morph in place
        parameterization(self)
        # update the parameters
        self._spectrum.grad = self.grad_spectrum
        self._spectrum.prox = self.prox_spectrum
        self._params.grad = self.grad_morph
        self._params.prox = self.prox_morph

    def to_component_data(self) -> ScarletComponentBaseData:
        raise NotImplementedError("Saving elliptical parametric components is not yet implemented")


class EllipticalParametricComponent(ParametricComponent):
    """A radial density/surface brightness profile with elliptical symmetry

    Parameters
    ----------
    bands:
        The bands used in the model.
    bbox:
        The bounding box that holds this component model.
    spectrum:
        The spectrum of the component.
    morph_params:
        The parameters passed to `morph_func` to
        generate the morphology in image space.
    morph_func:
        The function to generate the morphology
        based on `morphParams`.
    morph_grad:
        The function to calculate the gradient of the
        likelihood wrt the morphological parameters.
    morph_prox:
        The proximal operator for the morphology parameters.
    prox_spectrum:
        Proximal operator for the spectrum.
        If `prox_spectrum` is `None` then the default proximal
        operator `self.prox_spectrum` is used.
    floor:
        The minimum value of the spectrum, used to prevent
        divergences in the gradients.
    """

    def __init__(
        self,
        bands: tuple,
        bbox: Box,
        spectrum: Parameter | np.ndarray,
        morph_params: Parameter | np.ndarray,
        morph_func: Callable,
        morph_grad: Callable,
        morph_prox: Callable,
        morph_step: Callable | np.ndarray,
        prox_spectrum: Callable | None = None,
        floor: float = 1e-20,
    ):
        super().__init__(
            bands=bands,
            bbox=bbox,
            spectrum=spectrum,
            morph_params=morph_params,
            morph_func=morph_func,
            morph_grad=morph_grad,
            morph_prox=morph_prox,
            morph_step=morph_step,
            prox_spectrum=prox_spectrum,
            floor=floor,
        )

    @property
    def semi_major(self) -> float:
        """The length of the semi-major axis of the model"""
        return self._params.x[2]

    @property
    def semi_minor(self) -> float:
        """The length of the semi-minor axis of the model"""
        return self._params.x[3]

    @property
    def theta(self) -> float:
        """The counter-clockwise rotation angle of the model from the
        x-axis.
        """
        return self._params.x[4]

    @property
    def ellipse_params(self) -> np.ndarray:
        """The parameters used to generate the scaled radius"""
        return self._params.x[:5]

    @property
    def radial_params(self) -> np.ndarray:
        """The parameters used to model the radial function"""
        return self._params.x[5:]

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
        """The function that calculates the gradient of the
        morphological model
        """
        return self._morph_grad

    def update(self, it: int, input_grad: np.ndarray):
        """Update the component

        Parameters
        ----------
        it:
            The current iteration of the optimizer.
        input_grad:
            Gradient of the likelihood wrt the component model
        """
        ellipse = self.frame
        spectrum = self.spectrum.copy()
        morph = self._func(self.radial_params, ellipse)
        self._spectrum.update(it, input_grad, morph)
        self._params.update(it, input_grad, morph, spectrum, ellipse)
