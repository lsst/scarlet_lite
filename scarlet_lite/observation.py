__all__ = ["LiteObservation", "convolve", "FitPsfObservation"]


import numpy as np

from .bbox import Box
from .fft import Fourier, match_psf, convolve as fft_convolve
import interpolation
from .parameters import FistaParameter


def convolve(image, psf, bounds):
    """Convolve an image with a PSF in real space
    """
    from .operators_pybind11 import apply_filter

    result = np.empty(image.shape, dtype=image.dtype)
    for band in range(len(image)):
        if hasattr(image[band], "_value"):
            # This is an ArrayBox
            img = image[band]._value
        else:
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
    """Gradient of a real space convolution
    """
    return lambda input_grad: convolve(input_grad, psf[:, ::-1, ::-1], slices)


class LiteObservation:
    """A single observation

    This is effectively a combination of the `Observation` and
    `Renderer` class from base scarlet, greatly simplified due
    to the assumptions that the observations are all resampled
    onto the same pixel grid and that the `images` contain all
    of the information for all of the model bands.
    """
    def __init__(self, images, variance, weights, psfs, model_psf=None, noise_rms=None,
                 bbox=None, padding=3, convolution_mode="fft"):
        self.images = images
        self.variance = variance
        self.weights = weights
        # make sure that the images and psfs have the same dtype
        if psfs.dtype != images.dtype:
            psfs = psfs.astype(images.dtype)
        self.psfs = psfs

        assert convolution_mode in ["fft", "real"], "convolution_mode must be either 'fft' or 'real'"
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
            kernel = self.grad_kernel
        else:
            kernel = self.diff_kernel

        if kernel is None:
            return image

        if mode is None:
            mode = self.mode
        if mode == "fft":
            result = fft_convolve(
                Fourier(image), kernel, axes=(1, 2),
            ).image
        elif mode == "real":
            result = convolve(image, kernel.image, self.convolution_bounds)
        else:
            raise ValueError(f"mode must be either 'fft' or 'real', got {mode}")
        return result

    def render(self, model):
        """Mirror of `Observation.render to make APIs match
        """
        return self.convolve(model)

    @property
    def data(self):
        """Mirror of `Observation.data` to make APIs match
        """
        return self.images

    @property
    def shape(self):
        """The shape of the iamges, variance, etc."""
        return self.images.shape

    @property
    def n_bands(self):
        """The number of bands in the observation"""
        return self.images.shape[0]

    @property
    def dtype(self):
        """The dtype of the observation is the dtype of the images
        """
        return self.images.dtype

    @property
    def convolution_bounds(self):
        """Build the slices needed for convolution in real space
        """
        if not hasattr(self, "_convolution_bounds"):
            coords = interpolation.get_filter_coords(self.diff_kernel[0])
            self._convolution_bounds = interpolation.get_filter_bounds(
                coords.reshape(-1, 2)
            )
        return self._convolution_bounds

    def __getitem__(self, i):
        """Allow the user to slice the observations with python indexing
        """
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

        return LiteObservation(
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


class FitPsfObservation(LiteObservation):
    """An observation that fits the PSF used to convolve the model.
    """
    def __init__(self, diff_kernel, fft_shape, images, variance, weights, psfs, model_psf=None, noise_rms=None,
                 bbox=None, padding=3, convolution_mode="fft"):
        super().__init__(images, variance, weights, psfs, model_psf, noise_rms,
                 bbox, padding, convolution_mode)

        self.mode = "fft"
        self.axes = (-2, -1)
        self.fft_shape = fft_shape

        # Make the DFT of the psf a fittable parameter
        self._fitKernel = FistaParameter(diff_kernel.fft(fft_shape, self.axes), 1e-2)
        self._fitKernel.grad = self.grad_fit_kernel
        self._fitKernel.prox = self.prox_kernel

    def grad_fit_kernel(self, input_grad, kernel, model_fft):
        # Transform the upstream gradient into k-space
        grad_fft = Fourier(input_grad)
        grad_fft = grad_fft.fft(self.fft_shape, self.axes)
        return grad_fft * model_fft

    def prox_kernel(self, kernel, prox_step=0):
        # No prox for now
        return kernel

    @property
    def fitKernel(self):
        return self._fitKernel.x

    @property
    def gradFitKernel(self):
        return self.fitKernel.real - self.fitKernel.imag*1j

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
            kernel = self.gradFitKernel
        else:
            kernel = self.fitKernel

        if kernel is None:
            return image

        assert(mode is None or mode == "fft")

        image = Fourier(image)
        fft = image.fft(self.fft_shape, self.axes)

        result = fft.Fourier.from_fft(fft*kernel, self.fft_shape, image.shape, self.axes)
        return result.image

    def update(self, it, input_grad, model):
        model = Fourier(model[:, ::-1, ::-1])
        model_fft = model.fft(self.fft_shape, self.axes)
        self._fitKernel.update(it, input_grad, model_fft)
