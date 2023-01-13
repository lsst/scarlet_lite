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

__all__ = ["Observation", "convolve", "FitPsfObservation"]

import numpy as np
import numpy.typing as npt
from typing import Sequence, TypeVar

from .bbox import Box
from .fft import Fourier, match_psf, convolve as fft_convolve
from .image import Image
from .parameters import FistaParameter, Parameter


TObservation = TypeVar("TObservation", bound="Observation")


def get_filter_coords(
    filter_values: np.ndarray, center: tuple[int, int] = None
) -> np.ndarray:
    """Create filter coordinate grid needed for the apply filter function

    Parameters
    ----------
    filter_values: np.ndarray
        The 2D array of the filter to apply.
    center: Sequence[int, int]
        The center (y,x) of the filter. If `center` is `None` then
        `filter_values` must have an odd number of rows and columns
        and the center will be set to the center of `filter_values`.

    Returns
    -------
    coords: np.ndarray
        The coordinates of the pixels in `filter_values`,
        where the coordinates of the `center` pixel are `(0,0)`.
    """
    if len(filter_values.shape) != 2:
        raise ValueError("`filter_values` must be 2D")
    if center is None:
        if filter_values.shape[0] % 2 == 0 or filter_values.shape[1] % 2 == 0:
            msg = """Ambiguous center of the `filter_values` array,
                     you must use a `filter_values` array
                     with an odd number of rows and columns or
                     calculate `coords` on your own."""
            raise ValueError(msg)
        center = [filter_values.shape[0] // 2, filter_values.shape[1] // 2]
    x = np.arange(filter_values.shape[1])
    y = np.arange(filter_values.shape[0])
    x, y = np.meshgrid(x, y)
    x -= center[1]
    y -= center[0]
    coords = np.dstack([y, x])
    return coords


def get_filter_bounds(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Get the slices in x and y to apply a filter

    Parameters
    ----------
    coords: np.ndarray
        The coordinates of the filter,
        defined by `get_filter_coords`.

    Returns
    -------
    y_start, y_end, x_start, x_end: Sequence[int, int, int, int]
        The start and end of each slice that is passed to `apply_filter`.
    """
    z = np.zeros((len(coords),), dtype=int)
    # Set the y slices
    y_start = np.max([z, coords[:, 0]], axis=0)
    y_end = -np.min([z, coords[:, 0]], axis=0)
    # Set the x slices
    x_start = np.max([z, coords[:, 1]], axis=0)
    x_end = -np.min([z, coords[:, 1]], axis=0)
    return y_start, y_end, x_start, x_end


def convolve(image, psf, bounds):
    """Convolve an image with a PSF in real space"""
    from scarlet_lite.operators_pybind11 import apply_filter

    result = np.empty(image.shape, dtype=image.dtype)
    for band in range(len(image)):
        img = image[band]

        apply_filter(
            img,
            psf[band].reshape(-1),
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            result[band],
        )
    return result


def _grad_convolve(convolved, image, psf, slices):
    """Gradient of a real space convolution"""
    return lambda input_grad: convolve(input_grad, psf[:, ::-1, ::-1], slices)


def _set_image_like(
    images: np.ndarray | Image, bands: tuple | None = None, bbox: Box | None = None
) -> Image:
    """Ensure that an image-like array is cast appropriately as an image

     Parameters
     ----------
     images:
        The multiband image-like array to cast as an Image.
        If it already has `bands` and `bbox` properties then it is returned
        with no modifications.
    bands:
        The bands for the multiband-image.
        If `images` is a numpy array, this parameter is mandatory.
        If `images` is an `Image` and `bands` is not `None`,
        then `bands` must match `images.bands`.
    bbox:
        Bounding box containing the image.
        If `images` is a numpy array, this parameter is mandatory.
        If `images` is an `Image` and `bbox` is not `None`,
        then `bbox` must match `images.bbox`.

    Returns
    -------
    images: Image
        The input images converted into an image.
    """
    if hasattr(images, "bbox") and hasattr(images, "bands"):
        # This is already an image
        return images

    if bbox is None:
        bbox = Box(images.shape)
    elif hasattr(images, "bbox") and images.bbox != bbox:
        raise ValueError(
            f"Mismatched bounding boxes, images.bbox is {images.bbox} while bbox is {bbox}"
        )
    if not hasattr(images, "bands"):
        if bands is None:
            msg = f"""The `images` must be either an `Image` instance or a numpy `ndarray` with
            `bands` specified. Got {type(images)} and `bands = None`.
            """
            raise ValueError(msg)
        images = Image(images, bands=bands, yx0=bbox.origin[-2:])
    elif hasattr(images, "abnds") and images.bands != bands:
        raise ValueError(
            f"Mismatched bands, images.bands is {images.bands} while bands is {bands}"
        )
    return images


class Observation:
    """A single observation

    This is effectively a combination of the `Observation` and
    `Renderer` class from base scarlet, greatly simplified due
    to the assumptions that the observations are all resampled
    onto the same pixel grid and that the `images` contain all
    of the information for all of the model bands.
    """

    def __init__(
        self,
        images: np.ndarray | Image,
        variance: np.ndarray | Image,
        weights: np.ndarray | Image,
        psfs: np.ndarray,
        model_psf: np.ndarray | None = None,
        noise_rms: np.ndarray | None = None,
        bbox: Box = None,
        bands: Sequence[object] = None,
        padding: int = 3,
        convolution_mode: str = "fft",
    ):
        """Initialize an Observation.

        Parameters
        ----------
        bands:
            The filters (in order) used for all of the observation's
            properties.
        images:
            (bands, y, x) array of observed images.
        variance:
            (bands, y, x) array of variance for each image pixel.
        weights:
            (bands, y, x) array of weights to use when calculatint the
            likelihood of each pixel.
        psfs:
            (bands, y, x) array of the PSF image in each band.
        model_psf:
            (bands, y, x) array of the model PSF image in each band.
            If `model_psf` is `None` then convolution has no
            affect on the model, as it is generated in the same seeing as
            the observation(s).
        noise_rms:
            Per-band average noise RMS. If `noise_rms` is `None` then the mean
            of the sqrt of the variance is used.
        bbox:
            The bounding box containing the model. If `bbox` is `None` then
            a `Box` is created that is the shape of `images` with an origin
            at `(0, 0)`.
        padding:
            Padding to use when performing an FFT convolution.
        convolution_mode:
            The method of convolution. This should be either "fft" or "real".
        """
        # Convert the images to a multi-band `Image` and use the resulting
        # bbox and bands.
        images = _set_image_like(images, bands, bbox)
        bands = images.bands
        bbox = images.bbox
        self.images = images
        self.variance = _set_image_like(variance, bands, bbox)
        self.weights = _set_image_like(weights, bands, bbox)
        # make sure that the images and psfs have the same dtype
        if psfs.dtype != images.dtype:
            psfs = psfs.astype(images.dtype)
        self.psfs = psfs

        assert convolution_mode in [
            "fft",
            "real",
        ], "convolution_mode must be either 'fft' or 'real'"
        self.mode = convolution_mode
        if noise_rms is None:
            noise_rms = np.array(np.mean(np.sqrt(variance.data), axis=(1, 2)))
        self.noise_rms = noise_rms

        # Create a difference kernel to convolve the model to the PSF
        # in each band
        self.model_psf = model_psf
        self.padding = padding
        if model_psf is not None:
            self.diff_kernel = match_psf(psfs, model_psf, padding=padding)
            # The gradient of a convolution is another convolution,
            # but with the flipped and transposed kernel.
            diff_img = self.diff_kernel.image
            self.grad_kernel = Fourier(diff_img[:, ::-1, ::-1])
        else:
            self.diff_kernel = self.grad_kernel = None

        self._convolution_bounds = None

    @property
    def bands(self) -> tuple:
        return self.images.bands

    @property
    def bbox(self) -> Box:
        return self.images.bbox

    def convolve(self, image: Image, mode: str = None, grad: bool = False) -> Image:
        """Convolve the model into the observed seeing in each band.

        Parameters
        ----------
        image:
            The 3D image to convolve.
        mode:
            The convolution mode to use.
            This should be "real" or "fft" or `None`,
            where `None` will use the default `convolution_mode`
            specified during init.
        grad:
            Whether this is a backward gradient convolution
            (`grad==True`) or a pure convolution with the PSF.

        Returns
        -------
        result: Image
            The convolved image.
        """
        if grad:
            kernel = self.grad_kernel
        else:
            kernel = self.diff_kernel

        if kernel is None:
            return image

        if mode is None:
            mode = self.mode
        if mode == "fft":
            result = fft_convolve(
                Fourier(image.data),
                kernel,
                axes=(1, 2),
            ).image
        elif mode == "real":
            result = convolve(image.data, kernel.image, self.convolution_bounds)
        else:
            raise ValueError(f"mode must be either 'fft' or 'real', got {mode}")
        return Image(result, bands=image.bands, yx0=image.yx0)

    def log_likelihood(self, model: Image) -> float:
        """Calculate the log likelihood of the given model

        Parameters
        ----------
        model:
            Model to compare with the observed images.

        Returns
        -------
        result: float
            The log-likelihood of the given model.
        """
        result = 0.5 * -np.sum((self.weights * (self.images - model) ** 2).data)
        return result

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the images, variance, etc."""
        return self.images.shape

    @property
    def n_bands(self) -> int:
        """The number of bands in the observation"""
        return self.images.shape[0]

    @property
    def dtype(self) -> npt.DTypeLike:
        """The dtype of the observation is the dtype of the images"""
        return self.images.dtype

    @property
    def convolution_bounds(self) -> tuple[int, int, int, int]:
        """Build the slices needed for convolution in real space"""
        if self._convolution_bounds is None:
            coords = get_filter_coords(self.diff_kernel[0])
            self._convolution_bounds = get_filter_bounds(
                coords.reshape(-1, 2)
            )
        return self._convolution_bounds


class FitPsfObservation(Observation):
    """An observation that fits the PSF used to convolve the model."""

    def __init__(
        self,
        diff_kernel: np.ndarray | Parameter,
        fft_shape: tuple[int, int],
        images: np.ndarray | Image,
        variance: np.ndarray | Image,
        weights: np.ndarray | Image,
        psfs: np.ndarray,
        model_psf: np.ndarray = None,
        noise_rms: np.ndarray = None,
        bbox: Box = None,
        bands: Sequence[object] = None,
        padding: int = 3,
        convolution_mode: str = "fft",
    ):
        """Initialize a `FitPsfObservation`

        See `Observation` for a description of the parameters.
        """
        super().__init__(
            images,
            variance,
            weights,
            psfs,
            model_psf,
            noise_rms,
            bbox,
            bands,
            padding,
            convolution_mode,
        )

        self.mode = "fft"
        self.axes = (-2, -1)
        self.fft_shape = fft_shape

        # Make the DFT of the psf a fittable parameter
        self._fitKernel = FistaParameter(diff_kernel.fft(fft_shape, self.axes), 1e-2)
        self._fitKernel.grad = self.grad_fit_kernel
        self._fitKernel.prox = self.prox_kernel

    def grad_fit_kernel(
        self, input_grad: np.ndarray, kernel: np.ndarray, model_fft: np.ndarray
    ) -> np.ndarray:
        # Transform the upstream gradient into k-space
        grad_fft = Fourier(input_grad)
        grad_fft = grad_fft.fft(self.fft_shape, self.axes)
        return grad_fft * model_fft

    def prox_kernel(self, kernel: np.ndarray) -> np.ndarray:
        # No prox for now
        return kernel

    @property
    def fit_kernel(self) -> np.ndarray:
        return self._fitKernel

    @property
    def chached_kernel(self):
        return self.fit_kernel.real - self.fit_kernel.imag * 1j

    def convolve(self, image, mode=None, grad=False):
        """Convolve the model into the observed seeing in each band.

        Parameters
        ----------
        image: `~numpy.array`
            The image to convolve
        mode: `str`
            The convolution mode to use.
            This should be "real" or "fft" or `None`,
            where `None` will use the default `convolution_mode`
            specified during init.
        grad: `bool`
            Whether this is a backward gradient convolution
            (`grad==True`) or a pure convolution with the PSF.
        """
        if grad:
            kernel = self.chached_kernel
        else:
            kernel = self.fit_kernel

        if kernel is None:
            return image

        assert mode is None or mode == "fft"

        image = Fourier(image)
        fft = image.fft(self.fft_shape, self.axes)

        result = Fourier.from_fft(fft * kernel, self.fft_shape, image.shape, self.axes)
        return result.image

    def update(self, it, input_grad, model):
        model = Fourier(model[:, ::-1, ::-1])
        model_fft = model.fft(self.fft_shape, self.axes)
        self._fitKernel.update(it, input_grad, model_fft)
