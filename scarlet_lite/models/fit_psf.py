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

__all__ = ["FitPsfObservation", "FitPsfBlend"]

from typing import Sequence, cast, Callable

import numpy as np

from ..bbox import Box
from ..blend import Blend
from ..fft import Fourier, get_fft_shape
from ..image import Image
from ..observation import Observation
from ..parameters import parameter


class FitPsfObservation(Observation):
    """An observation that fits the PSF used to convolve the model."""

    def __init__(
        self,
        images: np.ndarray | Image,
        variance: np.ndarray | Image,
        weights: np.ndarray | Image,
        psfs: np.ndarray,
        model_psf: np.ndarray = None,
        noise_rms: np.ndarray = None,
        bbox: Box = None,
        bands: Sequence[object] = None,
        padding: int = 3,
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
            "fft",
        )

        self.axes = (-2, -1)

        self.fft_shape = get_fft_shape(images[0], psfs[0], padding, self.axes)

        # Make the DFT of the psf a fittable parameter
        self._fit_kernel = parameter(self.diff_kernel.fft(self.fft_shape, self.axes))

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
        return self._fit_kernel.x

    @property
    def cached_kernel(self):
        return self.fit_kernel.real - self.fit_kernel.imag * 1j

    def convolve(self, image: Image, mode: str | None = None, grad: bool = False) -> Image:
        """Convolve the model into the observed seeing in each band.

        Parameters
        ----------
        image:
            The image to convolve
        mode:
            The convolution mode to use.
            This should be "real" or "fft" or `None`,
            where `None` will use the default `convolution_mode`
            specified during init.
        grad:
            Whether this is a backward gradient convolution
            (`grad==True`) or a pure convolution with the PSF.
        """
        if grad:
            kernel = self.cached_kernel
        else:
            kernel = self.fit_kernel

        if mode != "fft" and mode is not None:
            return super().convolve(image, mode, grad)

        fft_image = Fourier(image.data)
        fft = fft_image.fft(self.fft_shape, self.axes)

        result = Fourier.from_fft(fft * kernel, self.fft_shape, image.shape, self.axes)
        return Image(result.image, bands=image.bands, yx0=image.yx0)

    def update(self, it: int, input_grad: np.ndarray, model: Image):
        model = Fourier(model[:, ::-1, ::-1].data)
        model_fft = model.fft(self.fft_shape, self.axes)
        self._fit_kernel.update(it, input_grad, model_fft)

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
        self._fit_kernel.grad = self.grad_fit_kernel
        self._fit_kernel.prox = self.prox_kernel


class FitPsfBlend(Blend):
    """A blend that attempts to fit the PSF along with the source models."""

    def _grad_log_likelihood(self) -> Image:
        """Gradient of the likelihood wrt the unconvolved model"""
        model = self.get_model(convolve=True)
        # Update the loss
        self.loss.append(self.observation.log_likelihood(model))
        # Calculate the gradient wrt the model d(logL)/d(model)
        result = self.observation.weights * (model - self.observation.images)
        return result

    def fit(
        self,
        max_iter: int,
        e_rel: float = 1e-4,
        min_iter: int = 1,
        resize: int = 10,
    ) -> tuple[int, float]:
        """Fit all of the parameters

        Parameters
        ----------
        max_iter: int
            The maximum number of iterations
        e_rel: float
            The relative error to use for determining convergence.
        min_iter: int
            The minimum number of iterations.
        resize: int
            Number of iterations before attempting to resize the
            resizable components. If `resize` is `None` then
            no resizing is ever attempted.
        """
        it = self.it
        while it < max_iter:
            # Calculate the gradient wrt the on-convolved model
            grad_log_likelihood = self._grad_log_likelihood()
            _grad_log_likelihood = self.observation.convolve(
                grad_log_likelihood, grad=True
            )
            # Update each component given the current gradient
            for component in self.components:
                component.update(it, _grad_log_likelihood.data)
            # Check to see if any components need to be resized
            if resize is not None and it > 0 and it % resize == 0:
                for component in self.components:
                    component.resize()

            # Update the PSF
            cast(FitPsfObservation, self.observation).update(
                it, grad_log_likelihood.data, self.get_model()
            )
            # Stopping criteria
            if it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(
                self.loss[-1]
            ):
                break
            it += 1
        self.it = it
        return it, self.loss[-1]
