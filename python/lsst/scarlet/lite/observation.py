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

__all__ = ["Observation", "convolve"]

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .bbox import Box
from .fft import Fourier, _pad, centered
from .fft import convolve as fft_convolve
from .fft import match_kernel
from .image import Image


def get_filter_coords(filter_values: np.ndarray, center: tuple[int, int] | None = None) -> np.ndarray:
    """Create filter coordinate grid needed for the apply filter function

    Parameters
    ----------
    filter_values:
        The 2D array of the filter to apply.
    center:
        The center (y,x) of the filter. If `center` is `None` then
        `filter_values` must have an odd number of rows and columns
        and the center will be set to the center of `filter_values`.

    Returns
    -------
    coords:
        The coordinates of the pixels in `filter_values`,
        where the coordinates of the `center` pixel are `(0,0)`.
    """
    if filter_values.ndim != 2:
        raise ValueError("`filter_values` must be 2D")
    if center is None:
        if filter_values.shape[0] % 2 == 0 or filter_values.shape[1] % 2 == 0:
            msg = """Ambiguous center of the `filter_values` array,
                     you must use a `filter_values` array
                     with an odd number of rows and columns or
                     calculate `coords` on your own."""
            raise ValueError(msg)
        center = tuple([filter_values.shape[0] // 2, filter_values.shape[1] // 2])  # type: ignore
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
    coords:
        The coordinates of the filter,
        defined by `get_filter_coords`.

    Returns
    -------
    y_start, y_end, x_start, x_end:
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


def convolve(image: np.ndarray, psf: np.ndarray, bounds: tuple[int, int, int, int]):
    """Convolve an image with a PSF in real space

    Parameters
    ----------
    image:
        The multi-band image to convolve.
    psf:
        The psf to convolve the image with.
    bounds:
        The filter bounds required by the ``apply_filter`` C++ method,
        usually obtained by calling `get_filter_bounds`.
    """
    from lsst.scarlet.lite.operators_pybind11 import apply_filter  # type: ignore

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


def _set_image_like(images: np.ndarray | Image, bands: tuple | None = None, bbox: Box | None = None) -> Image:
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
        then `bands` is ignored.
    bbox:
        Bounding box containing the image.
        If `images` is a numpy array, this parameter is mandatory.
        If `images` is an `Image` and `bbox` is not `None`,
        then `bbox` is ignored.

    Returns
    -------
    images: Image
        The input images converted into an image.
    """
    if isinstance(images, Image):
        # This is already an image
        if bbox is not None and images.bbox != bbox:
            raise ValueError(f"Bounding boxes {images.bbox} and {bbox} do not agree")
        return images

    if bbox is None:
        bbox = Box(images.shape[-2:])
    return Image(images, bands=bands, yx0=cast(tuple[int, int], bbox.origin))


class Observation:
    """A single observation

    This class contains all of the observed images and derived
    properties, like PSFs, variance map, and weight maps,
    required for most optimizers.
    This includes methods to match a scarlet model PSF to the oberved PSF
    in each band.

    Notes
    -----
    This is effectively a combination of the `Observation` and
    `Renderer` class from scarlet main, greatly simplified due
    to the assumptions that the observations are all resampled
    onto the same pixel grid and that the `images` contain all
    of the information for all of the model bands.

    Parameters
    ----------
    images:
        (bands, y, x) array of observed images.
    variance:
        (bands, y, x) array of variance for each image pixel.
    weights:
        (bands, y, x) array of weights to use when calculate the
        likelihood of each pixel.
    psfs:
        (bands, y, x) array of the PSF image in each band.
    model_psf:
        (bands, y, x) array of the model PSF image in each band.
        If `model_psf` is `None` then convolution is performed,
        which should only be done when the observation is a
        PSF matched coadd, and the scarlet model has the same PSF.
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

    def __init__(
        self,
        images: np.ndarray | Image,
        variance: np.ndarray | Image,
        weights: np.ndarray | Image,
        psfs: np.ndarray,
        model_psf: np.ndarray | None = None,
        noise_rms: np.ndarray | None = None,
        bbox: Box | None = None,
        bands: tuple | None = None,
        padding: int = 3,
        convolution_mode: str = "fft",
    ):
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

        if convolution_mode not in [
            "fft",
            "real",
        ]:
            raise ValueError("convolution_mode must be either 'fft' or 'real'")
        self.mode = convolution_mode
        if noise_rms is None:
            noise_rms = np.array([np.mean(np.sqrt(v[np.isfinite(v)])) for v in self.variance.data])
        self.noise_rms = noise_rms

        # Create a difference kernel to convolve the model to the PSF
        # in each band
        self.model_psf = model_psf
        self.padding = padding
        if model_psf is not None:
            if model_psf.dtype != images.dtype:
                self.model_psf = model_psf.astype(images.dtype)
            self.diff_kernel: Fourier | None = cast(Fourier, match_kernel(psfs, model_psf, padding=padding))
            # The gradient of a convolution is another convolution,
            # but with the flipped and transposed kernel.
            diff_img = self.diff_kernel.image
            self.grad_kernel: Fourier | None = Fourier(diff_img[:, ::-1, ::-1])
        else:
            self.diff_kernel = None
            self.grad_kernel = None

        self._convolution_bounds: tuple[int, int, int, int] | None = None

    @property
    def bands(self) -> tuple:
        """The bands in the observations."""
        return self.images.bands

    @property
    def bbox(self) -> Box:
        """The bounding box for the full observation."""
        return self.images.bbox

    def convolve(self, image: Image, mode: str | None = None, grad: bool = False) -> Image:
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
        result:
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
                return_fourier=False,
            )
        elif mode == "real":
            dy = image.shape[1] - kernel.image.shape[1]
            dx = image.shape[2] - kernel.image.shape[2]
            if dy < 0 or dx < 0:
                # The image needs to be padded because it is smaller than
                # the psf kernel
                _image = image.data
                newshape = list(_image.shape)
                if dy < 0:
                    newshape[1] += kernel.image.shape[1] - image.shape[1]
                if dx < 0:
                    newshape[2] += kernel.image.shape[2] - image.shape[2]
                _image = _pad(_image, newshape)
                result = convolve(_image, kernel.image, self.convolution_bounds)
                result = centered(result, image.data.shape)  # type: ignore
            else:
                result = convolve(image.data, kernel.image, self.convolution_bounds)
        else:
            raise ValueError(f"mode must be either 'fft' or 'real', got {mode}")
        return Image(cast(np.ndarray, result), bands=image.bands, yx0=image.yx0)

    def log_likelihood(self, model: Image) -> float:
        """Calculate the log likelihood of the given model

        Parameters
        ----------
        model:
            Model to compare with the observed images.

        Returns
        -------
        result:
            The log-likelihood of the given model.
        """
        result = 0.5 * -np.sum((self.weights * (self.images - model) ** 2).data)
        return result

    def __getitem__(self, indices: Any) -> Observation:
        """Get the subset of an image

        Parameters
        ----------
        indices:
            The indices to select a subsection of the image.

        Returns
        -------
        result:
            The resulting image obtained by selecting subsets of the iamge
            based on the `indices`.
        """
        new_image = self.images[indices]
        new_variance = self.variance[indices]
        new_weights = self.weights[indices]

        return Observation(
            images=new_image,
            variance=new_variance,
            weights=new_weights,
            psfs=self.psfs,
            model_psf=self.model_psf,
            noise_rms=self.noise_rms,
            bbox=new_image.bbox,
            bands=self.bands,
            padding=self.padding,
            convolution_mode=self.mode,
        )

    def __copy__(self, deep: bool = False) -> Observation:
        """Create a copy of the observation

        Parameters
        ----------
        deep:
            Whether to perform a deep copy or not.

        Returns
        -------
        result:
            The copy of the observation.
        """
        if deep:
            if self.model_psf is None:
                model_psf = None
            else:
                model_psf = self.model_psf.copy()

            if self.noise_rms is None:
                noise_rms = None
            else:
                noise_rms = self.noise_rms.copy()

            if self.bands is None:
                bands = None
            else:
                bands = tuple([b for b in self.bands])
        else:
            model_psf = self.model_psf
            noise_rms = self.noise_rms
            bands = self.bands

        return Observation(
            images=self.images.copy(),
            variance=self.variance.copy(),
            weights=self.weights.copy(),
            psfs=self.psfs.copy(),
            model_psf=model_psf,
            noise_rms=noise_rms,
            bands=bands,
            padding=self.padding,
            convolution_mode=self.mode,
        )

    def copy(self, deep: bool = False) -> Observation:
        """Create a copy of the observation

        Parameters
        ----------
        deep:
            Whether to perform a deep copy or not.

        Returns
        -------
        result:
            The copy of the observation.
        """
        return self.__copy__(deep)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of the images, variance, etc."""
        return cast(tuple[int, int, int], self.images.shape)

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
            coords = get_filter_coords(cast(Fourier, self.diff_kernel).image[0])
            self._convolution_bounds = get_filter_bounds(coords.reshape(-1, 2))
        return self._convolution_bounds

    @staticmethod
    def empty(
        bands: tuple[Any], psfs: np.ndarray, model_psf: np.ndarray, bbox: Box, dtype: npt.DTypeLike
    ) -> Observation:
        dummy_image = np.zeros((len(bands),) + bbox.shape, dtype=dtype)

        return Observation(
            images=dummy_image,
            variance=dummy_image,
            weights=dummy_image,
            psfs=psfs,
            model_psf=model_psf,
            noise_rms=np.zeros((len(bands),), dtype=dtype),
            bbox=bbox,
            bands=bands,
            convolution_mode="real",
        )
