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


__all__ = ["LiteComponent", "LiteFactorizedComponent", "SedComponent",
           "ParametricComponent", "EllipticalParametricComponent"]

from functools import partial

import numpy as np
from scipy.special import erf
from scipy.stats import gamma

from .parameters import FixedParameter, AdaproxParameter, DEFAULT_FACTOR
from .frame import CartesianFrame, EllipseFrame
from .bbox import overlapped_slices
from .parameters import relative_step
from . import initialization
from .constraint import MonotonicityConstraint
from .detect import scarletFootprintsToImage


# Some operations fail at the origin in radial coordinates,
# so we make use of a very small offset.
MIN_RADIUS = 1e-20

# Useful constants
SQRT_PI_2 = np.sqrt(np.pi/2)

# Stored sersic constants
SERSIC_B1 = gamma.ppf(0.5, 2)


class LiteComponent:
    """A base component in scarlet lite

    If `sed` and `morph` are arrays and not `LiteParameter`s then the
    component is not `initialized` and must still be initialized by
    another function.

    Parameters
    ----------
    center: `tuple` of `int`
        Location of the center pixel of the component in the full blend.
    bbox: `scarlet.bbox.Box`
        The bounding box for this component
    sed: `numpy.darray`
        The array of values for the SED `(bands,)`
    morph: `numpy.darray`
        The `(height, wdidth)` array of values for the morphology.
    initialized: `bool`
        Whether or not the component has been initialized.
    bg_thresh: `float`
        Level of the background thresh, required by some parameterizations.
    bg_rms: `float`
        The RMS of the background, required by some parameterizations.
    """
    def __init__(self, center, bbox, sed=None, morph=None, initialized=False,
                 bg_thresh=0.25, bg_rms=0):
        self._center = center
        self._bbox = bbox
        self._sed = sed
        self._morph = morph
        self.initialized = initialized
        self.bg_thresh = bg_thresh
        self.bg_rms = bg_rms

    @property
    def center(self):
        """The central locaation of the peak"""
        return self._center

    @property
    def bbox(self):
        """The bounding box that contains the component in the full image"""
        return self._bbox

    @property
    def sed(self):
        """The array of SED values"""
        return self._sed

    @property
    def morph(self):
        """The array of morphology values"""
        return self._morph

    def resize(self):
        """Test whether or not the component needs to be resized
        """
        # No need to resize if there is no size threshold.
        # To allow box sizing but no thresholding use `bg_thresh=0`.
        if self.bg_thresh is None:
            return False

        morph = self.morph
        size = max(morph.shape)

        # shrink the box? peel the onion
        dist = 0
        while (
            np.all(morph[dist, :] == 0) and
            np.all(morph[-dist, :] == 0) and
            np.all(morph[:, dist] == 0) and
            np.all(morph[:, -dist] == 0)
        ):
            dist += 1

        new_size = initialization.get_minimal_boxsize(size - 2 * dist)
        if new_size < size:
            dist = (size - new_size) // 2
            self.bbox.origin = (self.bbox.origin[0], self.bbox.origin[1]+dist, self.bbox.origin[2]+dist)
            self.bbox.shape = (self.bbox.shape[0], new_size, new_size)
            self._morph.shrink(dist)
            self.slices = overlapped_slices(self.model_bbox, self.bbox)
            return True

        # grow the box?
        model = self.get_model()
        edge_flux = np.array([
            np.sum(model[:, 0]),
            np.sum(model[:, -1]),
            np.sum(model[0, :]),
            np.sum(model[-1, :]),
        ])

        edge_mask = np.array([
            np.sum(model[:, 0] > 0),
            np.sum(model[:, -1] > 0),
            np.sum(model[0, :] > 0),
            np.sum(model[-1, :] > 0),
        ])

        if np.any(edge_flux/edge_mask > self.bg_thresh*self.bg_rms[:, None, None]):
            new_size = initialization.get_minimal_boxsize(size + 1)
            dist = (new_size - size) // 2
            self.bbox.origin = (self.bbox.origin[0], self.bbox.origin[1]-dist, self.bbox.origin[2]-dist)
            self.bbox.shape = (self.bbox.shape[0], new_size, new_size)
            self._morph.grow(self.bbox.shape[1:], dist)
            self.slices = overlapped_slices(self.model_bbox, self.bbox)
            return True
        return False

    def __str__(self):
        return "LiteComponent"

    def __repr__(self):
        return "LiteComponent"


class LiteFactorizedComponent(LiteComponent):
    """Implementation of a `FactorizedComponent` for simplified observations.
    """
    def __init__(self, sed, morph, center, bbox, model_bbox, bg_rms, bg_thresh=0.25, floor=1e-20,
                 fit_center_radius=1):
        """Initialize the component.

        Parameters
        ----------
        sed: `LiteParameter`
            The parameter to store and update the SED.
        morph: `LiteParameter`
            The parameter to store and update the morphology.
        center: `array-like`
            The center `(y,x)` of the source in the full model.
        bbox: `~scarlet.bbox.Box`
            The `Box` in the `model_bbox` that contains the source.
        model_bbox: `~scarlet.bbox.Box`
            The `Box` that contains the model.
            This is simplified from the main scarlet, where the model exists
            in a `frame`, which primarily exists because not all
            observations in main scarlet will use the same set of bands.
        bg_rms: `numpy.array`
            The RMS of the background used to threshold, grow,
            and shrink the component.
        floor: `float`
            Minimum value of the SED or center morphology pixel.
        """
        # Initialize all of the base attributes
        super().__init__(center, bbox, sed, morph, initialized=True, bg_thresh=bg_thresh, bg_rms=bg_rms)
        # Initialize the monotonicity constraint
        self.monotonicity = MonotonicityConstraint(
            neighbor_weight="angle",
            min_gradient=0,
            fit_center_radius=fit_center_radius
        )
        self.floor = floor
        self.model_bbox = model_bbox

        # update the parameters
        self._sed.grad = self.grad_sed
        self._sed.prox = self.prox_sed
        self._morph.grad = self.grad_morph
        self._morph.prox = self.prox_morph
        self.slices = overlapped_slices(model_bbox, bbox)

    @property
    def sed(self):
        """The array of SED values"""
        return self._sed.x

    @property
    def morph(self):
        """The array of morphology values"""
        return self._morph.x

    def get_model(self, bbox=None):
        """Build the model from the SED and morphology"""
        model = self.sed[:, None, None] * self.morph[None, :, :]

        if bbox is not None:
            slices = overlapped_slices(bbox, self.bbox)
            _model = np.zeros(bbox.shape, self.morph.dtype)
            _model[slices[0]] = model[slices[1]]
            model = _model
        return model

    def grad_sed(self, input_grad, sed, morph):
        """Gradient of the SED wrt. the component model"""
        _grad = np.zeros(self.bbox.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("...jk,jk", _grad, morph)

    def grad_morph(self, input_grad, morph, sed):
        """Gradient of the morph wrt. the component model"""
        _grad = np.zeros(self.bbox.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("i,i...", sed, _grad)

    def prox_sed(self, sed, prox_step=0):
        """Apply a prox-like update to the SED"""
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        return sed

    def prox_morph(self, morph, prox_step=0):
        """Apply a prox-like update to the morphology"""
        # monotonicity
        morph = self.monotonicity(morph, 0)

        if self.bg_thresh is not None:
            bg_thresh = self.bg_rms * self.bg_thresh
            # Enforce background thresholding
            model = self.sed[:, None, None] * morph[None, :, :]
            morph[np.all(model < bg_thresh[:, None, None], axis=0)] = 0
        else:
            # enforce positivity
            morph[morph < 0] = 0

        # prevent divergent morphology
        shape = morph.shape
        center = (shape[0] // 2, shape[1] // 2)
        morph[center] = np.max([morph[center], self.floor])
        # Normalize the morphology
        morph[:] = morph / morph.max()
        return morph

    def update(self, it, input_grad):
        """Update the SED and morphology parameters"""
        # Store the input SED so that the morphology can
        # have a consistent update
        sed = self.sed.copy()
        self._sed.update(it, input_grad, self.morph)
        self._morph.update(it, input_grad, sed)

    def __str__(self):
        return "LiteFactorizedComponent"

    def __repr__(self):
        return "LiteFactorizedComponent"


class SedComponent(LiteComponent):
    """Implements a free-form component component

    With no constraints this component is typically either a garbage collector,
    or part of a set of components to deconvolve an image by separating out
    the different spectral components.
    """
    def __init__(self, sed, morph, model_bbox, bg_thresh=None, bg_rms=None, floor=1e-20, peaks=None, min_area=0):
        """Initialize the component.

        See `~LiteComponent` for the rest of the parameters.

        Parameters
        ----------
        model_bbox: `~scarlet.bbox.Box`
            The `Box` that contains the model.
            This is simplified from the main scarlet, where the model exists
            in a `frame`, which primarily exists because not all
            observations in main scarlet will use the same set of bands.
        peaks: `list` of `tuple`
            A set of ``(cy, cx)`` peaks for detected sources.
            If peak is not ``None`` then only pixels in the same "footprint"
            as one of the peaks are included in the morphology.
            If `peaks` is ``None`` then there is no constraint applied.
        floor: `float`
            Minimum value of the SED or center morphology pixel.
        """
        cy = (model_bbox.shape[0]-1)//2
        cx = (model_bbox.shape[1]-1)//2
        self.floor = floor
        self.model_bbox = model_bbox
        self.peaks = peaks
        self.min_area = min_area

        super().__init__(
            (cy, cx),
            model_bbox,
            sed,
            morph,
            initialized=True,
            bg_thresh=bg_thresh,
            bg_rms=bg_rms
        )

        # update the parameters
        self._sed.grad = self.grad_sed
        self._sed.prox = self.prox_sed
        self._morph.grad = self.grad_morph
        self._morph.prox = self.prox_morph
        self.slices = [slice(None), slice(None)]

    @property
    def sed(self):
        """The array of SED values"""
        return self._sed.x

    @property
    def morph(self):
        """The array of morphology values"""
        return self._morph.x

    def get_model(self, bbox=None):
        """Build the model from the SED and morphology"""
        model = self.sed[:, None, None] * self.morph[None, :, :]

        if bbox is not None:
            slices = overlapped_slices(bbox, self.bbox)
            _model = np.zeros(bbox.shape, self.morph.dtype)
            _model[slices[0]] = model[slices[1]]
            model = _model
        return model

    def grad_sed(self, input_grad, sed, morph):
        """Gradient of the SED wrt. the component model"""
        _grad = np.zeros(self.bbox.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("...jk,jk", _grad, morph)

    def grad_morph(self, input_grad, morph, sed):
        """Gradient of the morph wrt. the component model"""
        _grad = np.zeros(self.bbox.shape, dtype=self.morph.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("i,i...", sed, _grad)

    def prox_sed(self, sed, prox_step=0):
        """Apply a prox-like update to the SED"""
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        # Normalize the SED
        sed = sed/np.sum(sed)
        return sed

    def prox_morph(self, morph, prox_step=0):
        """Apply a prox-like update to the morphology"""
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
            morph = morph * get_connected_multipeak(morph>0, self.peaks, 0)

        if self.min_area > 0:
            footprints = get_footprints(morph>0, 4.0, self.min_area, 0, False)
            morph = morph * (scarletFootprintsToImage(footprints, morph.shape) > 0)

        if np.all(morph==0):
            morph[0,0] = self.floor

        return morph

    def resize(self):
        return False

    def update(self, it, input_grad):
        """Update the SED and morphology parameters"""
        # Store the input SED so that the morphology can
        # have a consistent update
        sed = self.sed.copy()
        self._sed.update(it, input_grad, self.morph)
        self._morph.update(it, input_grad, sed)

    def __str__(self):
        return "SedComponent"

    def __repr__(self):
        return "SedComponent"


def gaussian2d(params, ellipse):
    """Model of a 2D elliptical gaussian

    Parameters
    ----------
    params: `numpy.ndarray`
        The parameters of the function.
        In this case there are none outside of the ellipticity
    ellipse: `EllipseFrame`
        The ellipse parameters to scale the radius in all directions.

    Returns
    -------
    result: `numpy.ndarray`
        The 2D guassian for the given ellipse parameters
    """
    return np.exp(-ellipse.R2)


def grad_gaussian(input_grad, params, cls, morph, sed, ellipse):
    """Gradient of the the component model wrt the Gaussian morphology parameters

    Parameters
    ----------
    input_grad: `numpy.ndarray`
        Gradient of the likelihood wrt the component model
    params: `numpy.ndarray`
        The parameters of the morphology.
    cls: `LiteComponent`
        The component of the model that contains the morphology.
    morph: `numpy.ndarray`
        The model of the morphology.
    sed: `numpy.ndarray`
        The model of the SED.
    ellipse: `EllipseFrame`
        The ellipse parameters to scale the radius in all directions.
    """
    # Calculate the gradient of the likelihod
    # wrt the Gaussian e^-r**2
    _grad = np.zeros(cls.bbox.shape, dtype=morph.dtype)
    _grad[cls.slices[1]] = input_grad[cls.slices[0]]
    _grad = -morph*np.einsum("i,i...", sed, _grad)
    dY0 = ellipse.grad_y0(_grad, True)
    dX0 = ellipse.grad_x0(_grad, True)
    dSigmaY = ellipse.grad_major(_grad, True)
    dSigmaX = ellipse.grad_minor(_grad, True)
    dTheta = ellipse.grad_theta(_grad, True)
    return np.array([dY0, dX0, dSigmaY, dSigmaX, dTheta], dtype=params.dtype)


def circular_gaussian(center, frame, sigma):
    """Model of a circularly symmetric Gaussian

    Parameters
    ----------
    center: `numpy.ndarray`
        The center of the Gaussian.
    frame: `CartesianFrame`
        The frame in which to generate the image of the circular Gaussian
    sigma: `float`
        The standard deviation.

    Returns
    -------
    result: `numpy.ndarray`
        The image of the circular Gaussian.
    """
    y0, x0 = center[:2]
    two_sigma = 2*sigma
    r2 = ((frame.X - x0)/two_sigma)**2 + ((frame.Y - y0)/two_sigma)**2
    return np.exp(-r2)


def grad_circular_gaussian(input_grad, params, cls, morph, sed, frame, sigma):
    """Gradient of the the component model wrt the Gaussian morphology parameters

    Parameters
    ----------
    input_grad: `numpy.ndarray`
        Gradient of the likelihood wrt the component model
    params: `numpy.ndarray`
        The parameters of the morphology.
    cls: `LiteComponent`
        The component of the model that contains the morphology.
    morph: `numpy.ndarray`
        The model of the morphology.
    sed: `numpy.ndarray`
        The model of the SED.
    frame: `CartesianFrame`
        The frame in which to generate the image of the circular Gaussian.
    """
    # Calculate the gradient of the likelihod
    # wrt the Gaussian e^-r**2
    _grad = np.zeros(cls.bbox.shape, dtype=morph.dtype)
    _grad[cls.slices[1]] = input_grad[cls.slices[0]]
    _grad = -morph*np.einsum("i,i...", sed, _grad)

    y0, x0 = params[:2]
    dY0 = -2*np.sum((frame.Y - y0)*_grad)
    dX0 = -2*np.sum((frame.X - x0)*_grad)
    return np.array([dY0, dX0], dtype=params.dtype)


def integrated_gaussian(params, frame):
    """Model of a circularly symmetric Gaussian integrated over pixels

    This differs from `circularGaussian` because the gaussian function
    is integrated over each pixel to replicate the pixelated image
    version of a Gaussian function.

    Parameters
    ----------
    params: `numpy.ndarray`
        The center of the Gaussian.
    frame: `CartesianFrame`
        The frame in which to generate the image of the circular Gaussian

    Returns
    -------
    result: `numpy.ndarray`
        The image of the circular Gaussian.
    """
    # Unpack the parameters and define constants
    y0, x0, sigma = params
    r = np.sqrt((frame.X - x0)**2 + (frame.Y - y0)**2)
    sqrt_c = 1/np.sqrt(2)/sigma
    # Integrate from half a pixel left and right
    lhs = erf((r - 0.5)*sqrt_c)
    rhs = erf((r + 0.5)*sqrt_c)
    z = 0.5*np.sqrt(np.pi)/sqrt_c*(rhs - lhs)
    return z


def grad_integrated_gaussian(input_grad, params, cls, morph, sed, frame):
    """Gradient of the the component model wrt the Gaussian morphology parameters

    Parameters
    ----------
    input_grad: `numpy.ndarray`
        Gradient of the likelihood wrt the component model
    params: `numpy.ndarray`
        The parameters of the morphology.
    cls: `LiteComponent`
        The component of the model that contains the morphology.
    morph: `numpy.ndarray`
        The model of the morphology.
    sed: `numpy.ndarray`
        The model of the SED.
    frame: `CartesianFrame`
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
    x = frame.X - x0
    y = frame.Y - y0
    c = 0.5/sigma**2
    sqrt_c = np.sqrt(c)
    # Add a small constant to the radius to prevent a divergence at r==0
    r = np.sqrt(x**2 + y**2 + MIN_RADIUS)
    # Shift half a pixel in each direction for the integration
    r1 = r - 0.5
    r2 = r + 0.5
    # Calculate the gradient of the ERF wrt. each shifted radius
    dModel1 = np.exp(-c*r1**2)
    dModel2 = np.exp(-c*r2**2)
    # Calculate the gradients of the parameters
    dX0 = np.sum(-x/r*(dModel2 - dModel1)*_grad)
    dY0 = np.sum(-y/r*(dModel2 - dModel1)*_grad)
    dSigma1 = -(r1*dModel1/sigma - SQRT_PI_2*erf(r1*sqrt_c))
    dSigma2 = -(r2*dModel2/sigma - SQRT_PI_2*erf(r2*sqrt_c))
    dSigma = np.sum((dSigma2 - dSigma1)*_grad)

    return np.array([dY0, dX0, dSigma])


def bounded_prox(params, prox_step, proxmin, proxmax):
    """A bounded proximal operator

    This function updates `params` in place.

    Parameters
    ----------
    params: `numpy.ndarray`
        The array of parameters to constrain.
    prox_step: `float`
        A scaling parameter used in some proximal operators
        in proxmin, but ignored here.
    proxmin: `numpy.ndarray`
        The array of minimum values for each parameter.
    proxmax: `numpy.ndarray`
        The array of maximum values for each parameter.

    Returns
    -------
    result: `numpy.ndarray`
        The updated parameters.
    """
    cuts = params < proxmin
    params[cuts] = proxmin[cuts]
    cuts = params > proxmax
    params[cuts] = proxmax[cuts]
    return params


def sersic(params, ellipse):
    """Generate a Sersic Model

    Parameters
    ----------
    params: `numpy.ndarray`
        The parameters of the function.
        In this case the only parameter is the sersic index ``n``.
    n: `float`
        The seric index. To avoid having too many
        degrees of freedom, we do not attempt to fit n,
        and typically use either `n=0` (exponential/disk profile) or
        `n=4` (de Vaucouleurs profile).
    ellipse: `EllipseFrame`
        The ellipse parameters to scale the radius in all directions.

    Returns
    -------
    result: `numpy.ndarray`
        The 2D guassian for the given ellipse parameters
    """
    n, = params

    r = ellipse.R

    if n == 1:
        result = np.exp(-SERSIC_B1*r)
    else:
        bn = gamma.ppf(0.5, 2*n)
        result = np.exp(-bn*(r**(1/n) - 1))
    return result


def grad_sersic(input_grad, params, cls, morph, sed, ellipse):
    """Gradient of the component model wrt the Gaussian morphology parameters

    Parameters
    ----------
    input_grad: `numpy.ndarray`
        Gradient of the likelihood wrt the component model
    params: `numpy.ndarray`
        The parameters of the morphology.
    cls: `LiteComponent`
        The component of the model that contains the morphology.
    morph: `numpy.ndarray`
        The model of the morphology.
    sed: `numpy.ndarray`
        The model of the SED.
    ellipse: `EllipseFrame`
        The ellipse parameters to scale the radius in all directions.
    """
    n = params[5]
    bn = gamma.ppf(0.5, 2*n)
    if n == 1:
        # Use a simplified model for faster calculation
        dExp = -SERSIC_B1*morph
    else:
        r = ellipse.R
        dExp = -bn/n*morph*r**(1 / n - 1)

    _grad = np.zeros(cls.bbox.shape, dtype=morph.dtype)
    _grad[cls.slices[1]] = input_grad[cls.slices[0]]
    _grad = np.einsum("i,i...", sed, _grad)
    dN = np.sum(_grad*bn*morph*ellipse.R**(1/n)*np.log10(ellipse.R)/n**2)
    _grad = _grad*dExp
    dY0 = ellipse.grad_y0(_grad, False)
    dX0 = ellipse.grad_x0(_grad, False)
    dSigmaY = ellipse.grad_major(_grad, False)
    dSigmaX = ellipse.grad_minor(_grad, False)
    dTheta = ellipse.grad_theta(_grad, False)
    return np.array([dY0, dX0, dSigmaY, dSigmaX, dTheta, dN], dtype=params.dtype)


class ParametricComponent:
    """A parametric model of an astrophysical source
    """

    def __init__(self, sed, morph_params, morph_func, morph_grad, morph_prox,
                 morph_step, model_frame, bbox, prox_sed=None, floor=1e-20):
        """Initialize the component

        Parameters
        ----------
        sed: `numpy.ndarray`
            The SED of the component.
        morph_params: `numpy.ndarray`
            The parameters of the morphology.
        morph_func: `Callable`
            The function to generate the 2D morphology image
            based on `morphParams`.
        morph_grad: `Callable`
            The function to calculate the gradient of the
            likelihood wrt the morphological parameters.
        morph_prox: `Callable`
            The proximal operator for the morphology parameters.
        bbox: `scarlet.bbox.Box`
            The bounding box that holds the model.
        prox_sed: `Function`
            Proximal operator for the SED.
            If `prox_sed` is `None` then the default proximal
            operator `self.prox_sed` is used.
        floor: `float`
            The minimum value of the SED, used to prevent
            divergences in the gradients.
        """
        params = FixedParameter(morph_params)
        sed = FixedParameter(sed)

        self._params = params
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
    def center(self):
        """The center of the component"""
        return self.y0, self.x0

    @property
    def y0(self):
        """The y-center of the component"""
        return self._params.x[0]

    @property
    def x0(self):
        """The x-center of the component"""
        return self._params.x[1]

    @property
    def sed(self):
        """The SED of the component"""
        return self._sed.x

    @property
    def bbox(self):
        """The bounding box that contains the component"""
        return self._bbox

    @property
    def frame(self):
        """The coordinate system that contains the model"""
        return CartesianFrame(self._bbox)

    @property
    def radial_params(self):
        """The parameters used to model the radial function"""
        return self._params.x

    def _morph(self, frame=None):
        """The 2D image of the morphology

        This callable generates an image of the morphology
        in the given frame.


        Parameters
        ----------
        frame: `CartesianFrame`
            The frame (bounding box, pixel grid) that the image is
            placed in.

        Returns
        -------
        result: `numpy.ndarray`
            The image of the morphology in the `frame`.
        """
        if frame is None:
            frame = self.frame
        return self._func(self.radial_params, frame)

    @property
    def morph(self, frame=None):
        """The morphological model"""
        return self._morph()

    @property
    def morph_prox(self):
        """The function used to constrain the morphological model"""
        return self._morph_prox

    @property
    def morph_grad(self):
        """The function that calculates the gradient of the morphological model"""
        return self._morph_grad

    @property
    def morph_step(self):
        """The function that calculates the gradient of the morphological model"""
        return self._morph_step

    def get_model(self, bbox=None, frame=None):
        """Generate the full model for this component"""
        model = self.sed[:, None, None] * self._morph(frame)[None, :, :]

        if bbox is not None:
            slices = overlapped_slices(bbox, self.bbox)
            _model = np.zeros(bbox.shape, self.morph.dtype)
            _model[slices[0]] = model[slices[1]]
            model = _model
        return model

    def prox_sed(self, sed, prox_step=0):
        """Apply a prox-like update to the SED

        Parameters
        ----------
        sed: `numpy.ndarray`
            The SED of the model.
        prox_step: `float`
            A scaling parameter used in some proximal operators,
            but ignored here.
        """
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        return sed

    def grad_sed(self, input_grad, sed, morph):
        """Gradient of the SED wrt. the component model

        Parameters
        ----------
        input_grad: `numpy.ndarray`
            Gradient of the likelihood wrt the component model
        sed: `numpy.ndarray`
            The model of the SED.
        morph: `numpy.ndarray`
            The model of the morphology.

        Returns
        -------
        result: `float`
            The gradient of the likelihood wrt. the SED.
        """
        _grad = np.zeros(self.bbox.shape, dtype=self.sed.dtype)
        _grad[self.slices[1]] = input_grad[self.slices[0]]
        return np.einsum("...jk,jk", _grad, morph)

    def update(self, it, input_grad):
        """Update the component parameters from an input gradient

        Parameters
        ----------
        it: `int`
            The current iteration of the optimizer.
        input_grad: `numpy.ndarray`
            Gradient of the likelihood wrt the component model
        """
        sed = self.sed.copy()
        morph = self.morph
        self._sed.update(it, input_grad, morph)
        self._params.update(it, input_grad, self, morph, sed, self.frame)

    def resize(self):
        """Resize the box that contains the model

        Not yet implemented, so for now the model box
        does not grow. In the long run this will be
        based on a cutoff value for the model.
        """
        return False

    def init_adaprox(self, noise_rms, max_prox_iter=1, factor=10):
        """Convert all of the parameters into adaprox parameters

        Parameters
        ----------
        noise_rms: `numpy.ndarray`
            The RMS noise in each band.
        max_prox_iter: `int`
            Maximum number of proximal iterations.
        factor: `int`
            The factor to scale the noise to set the
            SED step.
        """
        self._sed = AdaproxParameter(
            self._sed.x,
            step=partial(relative_step, factor=DEFAULT_FACTOR, minimum=noise_rms/factor),
            max_prox_iter=max_prox_iter,
            prox=self._prox_sed,
            grad=self.grad_sed,
        )
        self._params = AdaproxParameter(
            self._params.x,
            step=self.morph_step,
            max_prox_iter=max_prox_iter,
            prox=self.morph_prox,
            grad=self.morph_grad,
        )


class EllipticalParametricComponent(ParametricComponent):
    """A radial density/surface brightness profile with elliptical symmetry
    """

    def __init__(self, sed, morph_params, morph_func, morph_grad, morph_prox, morph_step,
                 bbox, model_frame, prox_sed=None, floor=1e-20):
        """Initialize the component

        Parameters
        ----------
        sed: `numpy.ndarray`
            The SED of the component.
        morph_params: `numpy.ndarray`
            The parameters passed to `morph_func` to
            generate the morphology in image space.
        morph_func: `Function`
            The function to generate the morphology
            based on `morphParams`.
        morph_grad: `Function`
            The function to calculate the gradient of the
            likelihood wrt the morphological parameters.
        morph_prox: `Function`
            The proximal operator for the morphology parameters.
        bbox: `scarlet.bbox.Box`
            The bounding box that holds the model.
        frame: `CartesianGrid`
            The coordinates of the model frame,
            used to speed up the creation of the
            polar grid for each source.
        prox_sed: `Function`
            Proximal operator for the SED.
            If `prox_sed` is `None` then the default proximal
            operator `self.prox_sed` is used.
        floor: `float`
            The minimum value of the SED, used to prevent
            divergences in the gradients.
        """
        super().__init__(
            sed=sed,
            morph_params=morph_params,
            morph_func=morph_func,
            morph_grad=morph_grad,
            morph_prox=morph_prox,
            morph_step=morph_step,
            model_frame=model_frame,
            bbox=bbox,
            prox_sed=prox_sed,
            floor=floor
        )

    @property
    def semi_major(self):
        """The length of the semi-major axis of the model"""
        return self._params.x[2]

    @property
    def semi_minor(self):
        """The length of the semi-minor axis of the model"""
        return self._params.x[3]

    @property
    def theta(self):
        """The counter-clockwise rotation angle of the model from the x-axis."""
        return self._params.x[4]

    @property
    def ellipse_params(self):
        """The parameters used to generate the scaled radius"""
        return self._params.x[:5]

    @property
    def radial_params(self):
        """The parameters used to model the radial function"""
        return self._params.x[5:]

    @property
    def frame(self):
        """The `EllipseFrame` that parameterizes the model"""
        return EllipseFrame(*self.ellipse_params, self._bbox)

    @property
    def morph_prox(self):
        """The function used to constrain the morphological model"""
        return self._morph_prox

    @property
    def morph_grad(self):
        """The function that calculates the gradient of the morphological model"""
        return self._morph_grad

    def update(self, it, input_grad):
        """Update the component

        Parameters
        ----------
        it: `int`
            The current iteration of the optimizer.
        input_grad: `numpy.ndarray`
            Gradient of the likelihood wrt the component model
        """
        ellipse = self.frame
        sed = self.sed.copy()
        morph = self._func(self.radial_params, ellipse)
        self._sed.update(it, input_grad, morph)
        self._params.update(it, input_grad, self, morph, sed, ellipse)

    def resize(self):
        """Resize the box that contains the model

        Not yet implemented, so for now the model box
        does not grow. In the long run this will be
        based on a cutoff value for the model.
        """
        return False
