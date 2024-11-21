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

__all__ = ["FittedPsfObservation", "FittedPsfBlend"]

from typing import Callable, cast

import numpy as np

from ..bbox import Box
from ..blend import Blend
from ..fft import Fourier, centered
from ..fft import convolve as fft_convolve
from ..image import Image
from ..observation import Observation
from ..parameters import parameter


class FittedPsfObservation(Observation):
    """An observation that fits the PSF used to convolve the model."""

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
        shape: tuple[int, int] | None = None,
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

        self.axes = (-2, -1)

        if shape is None:
            shape = (41, 41)

        # Make the DFT of the psf a fittable parameter
        self._fitted_kernel = parameter(cast(Fourier, self.diff_kernel).image)

    def grad_fit_kernel(self, input_grad: np.ndarray, psf: np.ndarray, model: np.ndarray) -> np.ndarray:
        """Gradient of the loss wrt the PSF

        This is just the cross correlation of the input gradient
        with the model.

        Parameters
        ----------
        input_grad:
            The gradient of the loss wrt the model
        psf:
            The PSF of the model.
        model:
            The deconvolved model.
        """
        grad = cast(
            np.ndarray,
            fft_convolve(
                Fourier(model),
                Fourier(input_grad[:, ::-1, ::-1]),
                axes=(1, 2),
                return_fourier=False,
            ),
        )

        return centered(grad, psf.shape)

    def prox_kernel(self, kernel: np.ndarray) -> np.ndarray:
        # No prox for now
        return kernel

    @property
    def fitted_kernel(self) -> np.ndarray:
        return self._fitted_kernel.x

    @property
    def cached_kernel(self):
        return self.fitted_kernel[:, ::-1, ::-1]

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
            kernel = self.fitted_kernel

        if mode != "fft" and mode is not None:
            return super().convolve(image, mode, grad)

        result = fft_convolve(
            Fourier(image.data),
            Fourier(kernel),
            axes=(1, 2),
            return_fourier=False,
        )
        return Image(cast(np.ndarray, result), bands=image.bands, yx0=image.yx0)

    def update(self, it: int, input_grad: np.ndarray, model: np.ndarray):
        """Update the PSF given the gradient of the loss

        Parameters
        ----------
        it: int
            The current iteration
        input_grad: np.ndarray
            The gradient of the loss wrt the model
        model: np.ndarray
            The deconvolved model.
        """
        self._fitted_kernel.update(it, input_grad, model)

    def parameterize(self, parameterization: Callable) -> None:
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization: Callable
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        # Update the fitted kernel in place
        parameterization(self)
        # update the parameters
        self._fitted_kernel.grad = self.grad_fit_kernel
        self._fitted_kernel.prox = self.prox_kernel


class FittedPsfBlend(Blend):
    """A blend that attempts to fit the PSF along with the source models."""

    def _grad_log_likelihood(self) -> tuple[Image, np.ndarray]:
        """Gradient of the likelihood wrt the unconvolved model"""
        model = self.get_model(convolve=True)
        # Update the loss
        self.loss.append(self.observation.log_likelihood(model))
        # Calculate the gradient wrt the model d(logL)/d(model)
        residual = self.observation.weights * (model - self.observation.images)

        return residual, model.data

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
            grad_log_likelihood, model = self._grad_log_likelihood()
            _grad_log_likelihood = self.observation.convolve(grad_log_likelihood, grad=True)
            # Check if resizing needs to be performed in this iteration
            if resize is not None and self.it > 0 and self.it % resize == 0:
                do_resize = True
            else:
                do_resize = False
            # Update each component given the current gradient
            for component in self.components:
                overlap = component.bbox & self.bbox
                component.update(it, _grad_log_likelihood[overlap].data)
                # Check to see if any components need to be resized
                if do_resize:
                    component.resize(self.bbox)

            # Update the PSF
            cast(FittedPsfObservation, self.observation).update(
                self.it,
                grad_log_likelihood.data,
                model,
            )
            # Stopping criteria
            it += 1
            if it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(self.loss[-1]):
                break
        self.it = it
        return it, self.loss[-1]

    def parameterize(self, parameterization: Callable):
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization:
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        for source in self.sources:
            source.parameterize(parameterization)
        cast(FittedPsfObservation, self.observation).parameterize(parameterization)
