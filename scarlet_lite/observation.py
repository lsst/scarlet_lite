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
from . import interpolation
from .parameters import FistaParameter, Parameter


TObservation = TypeVar("TObservation", bound="Observation")


def convolve(image, psf, bounds):
    """Convolve an image with a PSF in real space"""
    from .operators_pybind11 import apply_filter

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
        images: np.ndarray,
        variance: np.ndarray,
        weights: np.ndarray,
        psfs: np.ndarray,
        model_psf: np.ndarray | None = None,
        noise_rms: np.ndarray | None = None,
        bbox: Box = None,
        padding: int = 3,
        convolution_mode: str = "fft",
    ):
        """Initialize an Observation.

         Parameters
         ----------
         images: np.ndarray
            (bands, y, x) array of observed images.
        variance: np.ndarray
            (bands, y, x) array of variance for each image pixel.
        weights: np.ndarray
            (bands, y, x) array of weights to use when calculatint the
            likelihood of each pixel.
        psfs: np.ndarray
            (bands, y, x) array of the PSF image in each band.
        model_psf: np.ndarray
            (bands, y, x) array of the model PSF image in each band.
            If `model_psf` is `None` then convolution has no
            affect on the model, as it is generated in the same seeing as
            the observation(s).
        noise_rms: np.ndarray
            Per-band average noise RMS. If `noise_rms` is `None` then the mean
            of the sqrt of the variance is used.
        bbox: Box
            The bounding box containing the model. If `bbox` is `None` then
            a `Box` is created that is the shape of `images` with an origin
            at `(0, 0, 0)`.
        padding: int
            Padding to use when performing an FFT convolution.
        convolution_model: str
            The method of convolution. This should be either "fft" or "real".
        """
        self.images = images
        self.variance = variance
        self.weights = weights
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
            noise_rms = np.array(np.mean(np.sqrt(variance), axis=(1, 2)))
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

        if bbox is None:
            self.bbox = Box(images.shape)
        else:
            self.bbox = bbox

    def convolve(
        self, image: np.ndarray, mode: str = None, grad: bool = False
    ) -> np.ndarray:
        """Convolve the model into the observed seeing in each band.

        Parameters
        ----------
        image: np.ndarray
            The 2D image to convolve.
        mode: str
            The convolution mode to use.
            This should be "real" or "fft" or `None`,
            where `None` will use the default `convolution_mode`
            specified during init.
        grad: bool
            Whether this is a backward gradient convolution
            (`grad==True`) or a pure convolution with the PSF.

        Returns
        -------
        result: npndarray
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
                Fourier(image),
                kernel,
                axes=(1, 2),
            ).image
        elif mode == "real":
            result = convolve(image, kernel.image, self.convolution_bounds)
        else:
            raise ValueError(f"mode must be either 'fft' or 'real', got {mode}")
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
        if not hasattr(self, "_convolution_bounds"):
            coords = interpolation.get_filter_coords(self.diff_kernel[0])
            self._convolution_bounds = interpolation.get_filter_bounds(
                coords.reshape(-1, 2)
            )
        return self._convolution_bounds

    def __getitem__(self, i: int | Sequence[int] | slice) -> TObservation:
        """Allow the user to slice the observations with python indexing"""
        images = self.images[i]
        variance = self.variance[i]
        weights = self.weights[i]
        psfs = self.psfs[i]
        noise_rms = self.noise_rms[i]

        if len(images.shape) == 2:
            images = images[None]
            variance = variance[None]
            weights = weights[None]
            psfs = psfs[None]
            noise_rms = np.array([noise_rms])

        return Observation(
            images,
            variance,
            weights,
            psfs,
            model_psf=self.model_psf,
            noise_rms=noise_rms,
            bbox=self.bbox,
            padding=self.padding,
            convolution_mode=self.mode,
        )


class FitPsfObservation(Observation):
    """An observation that fits the PSF used to convolve the model."""

    def __init__(
        self,
        diff_kernel: np.ndarray | Parameter,
        fft_shape: tuple[int, int],
        images: np.ndarray,
        variance: np.ndarray,
        weights: np.ndarray,
        psfs: np.ndarray,
        model_psf: np.ndarray = None,
        noise_rms: np.ndarray = None,
        bbox: Box = None,
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
