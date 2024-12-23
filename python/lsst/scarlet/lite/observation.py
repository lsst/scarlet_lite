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

import logging
from typing import Any, cast

from deprecated.sphinx import deprecated

import numpy as np
import numpy.typing as npt

from .bbox import Box
from .image import Image
from . import fft
from . import rendering


@deprecated(
    version="v28.0",
    reason="Use scarlet_lite.renderer.convolve instead."
           "This function will be removed in v29.0.",
    category=FutureWarning
)
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
    return rendering.convolve(image, psf, bounds)


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
        psfs: np.ndarray | None = None,
        model_psf: np.ndarray | None = None,
        renderer: rendering.Renderer | None = None,
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

        if noise_rms is None:
            noise_rms = np.array([np.mean(np.sqrt(v[np.isfinite(v)])) for v in self.variance.data])
        self.noise_rms = noise_rms

        # make sure that the images and psfs have the same dtype
        if renderer is None:
            if psfs is None:
                raise ValueError("psfs must be provided if renderer is not provided")
            if psfs.dtype != images.dtype:
                psfs = psfs.astype(images.dtype)

            renderer = rendering.Renderer(bands, psfs, model_psf, padding, convolution_mode)

        self.renderer = renderer

    @property
    def bands(self) -> tuple:
        """The bands in the observations."""
        return self.images.bands

    @property
    def bbox(self) -> Box:
        """The bounding box for the full observation."""
        return self.images.bbox

    @property
    def psfs(self) -> np.ndarray:
        """The PSF in each band."""
        return self.renderer.psfs

    @property
    def model_psf(self) -> np.ndarray | None:
        """The model PSF in each band."""
        return self.renderer.model_psf

    @property
    def diff_kernel(self) -> fft.Fourier | None:
        """The difference kernel for the renderer."""
        return self.renderer.diff_kernel

    @property
    def grad_kernel(self) -> fft.Fourier | None:
        """The gradient kernel for the renderer."""
        return self.renderer.grad_kernel

    def convolve(self, image: Image, mode: str | None = None, grad: bool = False) -> Image:
        return self.renderer.convolve(image, mode, grad)

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
        """Get the subset of an Observation

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

        # Extract the appropriate bands from the PSF
        bands = self.images.bands
        new_bands = new_image.bands
        if bands != new_bands:
            band_indices = self.images.spectral_indices(new_bands)
            psfs = self.renderer.psfs[band_indices,]
            renderer = rendering.Renderer(
                bands=new_bands,
                psfs=psfs,
                model_psf=self.renderer.model_psf,
                padding=self.renderer.padding,
                mode=self.renderer.mode
            )
            noise_rms = self.noise_rms[band_indices,]
        else:
            renderer = self.renderer
            noise_rms = self.noise_rms

        return Observation(
            images=new_image,
            variance=new_variance,
            weights=new_weights,
            renderer=renderer,
            noise_rms=noise_rms,
            bbox=new_image.bbox,
            bands=self.bands,
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
            renderer = self.renderer.copy(deep=deep)

            if self.noise_rms is None:
                noise_rms = None
            else:
                noise_rms = self.noise_rms.copy()

            if self.bands is None:
                bands = None
            else:
                bands = tuple([b for b in self.bands])
        else:
            renderer = self.renderer
            noise_rms = self.noise_rms
            bands = self.bands

        return Observation(
            images=self.images.copy(),
            variance=self.variance.copy(),
            weights=self.weights.copy(),
            renderer=renderer,
            noise_rms=noise_rms,
            bands=bands,
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


class EmptyObservation(Observation):
    """An empty observation

    This class is used to represent an observation with no image data
    but includes the PSF and model PSF information for rendering.

    Parameters
    ----------
    bands:
        The bands in the observation.
    psfs:
        The PSF in each band.
    model_psf:
        The model PSF in each band.
    """

    def __init__(
        self,
        bands: tuple,
        psfs: np.ndarray | None = None,
        model_psf: np.ndarray | None = None,
        renderer: rendering.Renderer | None = None,
        dtype: npt.DTypeLike | None = None,
        padding: int = 3,
        convolution_mode: str = "fft",
    ):
        if renderer is None:
            if psfs is None:
                raise ValueError("psfs must be provided if renderer is not provided")
            if dtype is not None:
                if psfs.dtype != dtype:
                    psfs = psfs.astype(dtype)
                if model_psf is not None and model_psf.dtype != dtype:
                    model_psf = model_psf.astype(dtype)
            renderer = rendering.Renderer(bands, psfs, model_psf, padding=padding, mode=convolution_mode)
        self.renderer = renderer
        self._bands = bands

    @property
    def bbox(self) -> Box:
        raise AttributeError("bbox is not available for EmptyObservation")

    @property
    def shape(self) -> tuple[int, int, int]:
        raise AttributeError("shape is not available for EmptyObservation")

    @property
    def bands(self) -> tuple:
        return self._bands

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def images(self) -> Image:
        raise AttributeError("images is not available for EmptyObservation")

    @property
    def variance(self) -> Image:
        raise AttributeError("variance is not available for EmptyObservation")

    @property
    def weights(self) -> Image:
        raise AttributeError("weights is not available for EmptyObservation")

    def log_likelihood(self, model: Image) -> float:
        raise AttributeError("log_likelihood is not available for EmptyObservation")

    def __getitem__(self, indices: Any) -> Observation:
        """Get the subset of an Observation

        This overrides the `__getitem__` method of the `Observation` class
        to return an `EmptyObservation` with only the spectral dimension
        sliced.

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
        bands = self.bands
        test_image = Image(np.zeros((len(bands), 1, 1), dtype=self.renderer.psfs.dtype), bands=bands)
        test_image = test_image[indices]

        if test_image.bands == bands:
            # The slicing was in the spatial dimensions,
            # so the rendering information is unchanged.
            return self

        new_bands = test_image.bands
        band_indices = self.images.spectral_indices(new_bands)
        psfs = self.renderer.psfs[band_indices,]
        renderer = rendering.Renderer(
            bands=new_bands,
            psfs=psfs,
            model_psf=self.renderer.model_psf,
            padding=self.renderer.padding,
            mode=self.renderer.mode
        )

        return EmptyObservation(
            bands=new_bands,
            renderer=renderer,
        )
