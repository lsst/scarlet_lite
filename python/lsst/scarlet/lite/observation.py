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

import warnings
from copy import deepcopy
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt

from .bbox import Box
from .image import Image
from .psf import ImagePsf, Psf


class _Required:
    """Sentinel for a logically-required argument that must carry a default.

    Used by `Observation.empty`, where the new ``psf`` argument needs a
    default so the deprecated ``psfs`` alias can substitute for it; that
    forces the arguments after ``psf`` to also have defaults. Marking them
    ``_REQUIRED`` lets the body re-impose the "required" contract while the
    deprecation period lasts.

    This can be removed after v31.0, at which point the arguments it marks
    can be made required in the signature and the checks in `empty`
    removed.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<required>"


# Annotate as Any so `= _REQUIRED` typechecks against any parameter type
# without per-line ignores, and the public annotations don't gain `| None`.
_REQUIRED: Final[Any] = _Required()


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
    _x = np.arange(filter_values.shape[1])
    _y = np.arange(filter_values.shape[0])
    x, y = np.meshgrid(_x, _y)
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
    psf:
        The observed PSF as a `Psf`. A bare ``(bands, y, x)`` array is also
        accepted for backwards compatibility (it is wrapped in an `ImagePsf`),
        but doing so is deprecated.
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
    psfs:
        Deprecated alias for `psf`. Will be removed after v31.0.
    """

    def __init__(
        self,
        images: np.ndarray | Image,
        variance: np.ndarray | Image,
        weights: np.ndarray | Image,
        psf: np.ndarray | Psf | None = None,
        model_psf: np.ndarray | Psf | None = None,
        noise_rms: np.ndarray | None = None,
        bbox: Box | None = None,
        bands: tuple | None = None,
        padding: int = 3,
        convolution_mode: str = "fft",
        *,
        psfs: np.ndarray | None = None,
    ):
        # Convert the images to a multi-band `Image` and use the resulting
        # bbox and bands.
        images = _set_image_like(images, bands, bbox)
        bands = images.bands
        bbox = images.bbox
        self.images = images
        self.variance = _set_image_like(variance, bands, bbox)
        self.weights = _set_image_like(weights, bands, bbox)
        self.padding = padding

        # Resolve the observed PSF from the new ``psf`` argument or the
        # deprecated ``psfs`` alias.
        if psfs is not None:
            if psf is not None:
                raise ValueError("Provide only one of `psf` or the deprecated `psfs`.")
            warnings.warn(
                "The `psfs` argument is deprecated in favor of `psf` and will "
                "be removed after v31.0. Pass a `Psf` as `psf` instead.",
                FutureWarning,
                stacklevel=2,
            )
            psf = ImagePsf(psfs, bands=bands if bands is not None else (), padding=padding)
        elif psf is not None and not isinstance(psf, Psf):
            warnings.warn(
                "Passing an ndarray as `psf` is deprecated and will be "
                "unsupported after v31.0. Wrap it in an `ImagePsf` instead.",
                FutureWarning,
                stacklevel=2,
            )
            psf = ImagePsf(psf, bands=bands if bands is not None else (), padding=padding)
        if psf is None:
            raise ValueError("`psf` is required.")

        # Make sure the PSF and images share a dtype.
        self.psf = psf
        if self.psf.dtype != images.dtype:
            self.psf = self.psf.astype(images.dtype)

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
        # in each band. The kernel owns both the forward convolution and its
        # adjoint (the gradient pass). The model PSF and observed PSF are both
        # cast to the image dtype so the difference kernel is built from
        # consistent precision in every band.
        self.model_psf: Psf | None
        self.diff_kernel: Psf | None
        if model_psf is not None:
            if not isinstance(model_psf, Psf):
                warnings.warn(
                    "Passing an ndarray as `model_psf` is deprecated and will "
                    "be unsupported after v31.0. Wrap it in an `ImagePsf` "
                    "instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                model_psf = ImagePsf(model_psf, bands=(), padding=padding)
            self.model_psf = model_psf
            if self.model_psf.dtype != images.dtype:
                self.model_psf = self.model_psf.astype(images.dtype)
            self.diff_kernel = self.psf.match(self.model_psf, padding=padding)
        else:
            self.model_psf = None
            self.diff_kernel = None

    @property
    def bands(self) -> tuple:
        """The bands in the observations."""
        return self.images.bands

    @property
    def bbox(self) -> Box:
        """The bounding box for the full observation."""
        return self.images.bbox

    def convolve(
        self,
        image: Image,
        mode: str | None = None,
        grad: bool = False,
        cache: bool = False,
    ) -> Image:
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
        cache:
            Whether to cache the FFT of the kernel at this image's shape.
            Defaults to ``False`` because most call sites convolve
            many different shapes (per-source / per-component) and would
            grow the kernel's FFT cache unboundedly. Pass ``cache=True``
            for repeated full-blend convolutions (e.g. inside the fit
            loop), where the same shape recurs every iteration.
            Ignored for ``mode == "real"``.

        Returns
        -------
        result:
            The convolved image.
        """
        if self.diff_kernel is None:
            return image

        if mode is None:
            mode = self.mode
        if grad:
            return self.diff_kernel.grad(image, mode=mode, cache=cache)
        return self.diff_kernel.convolve(image, mode=mode, cache=cache)

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
        """Get a view for the subset of an image

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

        # If the indices is a single band, make sure to keep the band axis
        if new_image.ndim == 2:
            if indices in self.bands:
                new_bands = (indices,)
            else:
                # The indices contain spatial and band indices
                new_bands = (indices[0],)
            new_image = Image(
                new_image.data[None, :, :],
                yx0=new_image.yx0,
                bands=new_bands,
            )
            new_variance = Image(
                new_variance.data[None, :, :],
                yx0=new_variance.yx0,
                bands=new_bands,
            )
            new_weights = Image(
                new_weights.data[None, :, :],
                yx0=new_weights.yx0,
                bands=new_bands,
            )

        # Extract the appropriate bands from the PSF
        bands = self.images.bands
        new_bands = new_image.bands
        if bands != new_bands:
            band_indices = self.images.spectral_indices(new_bands)
            psf = self.psf[new_bands]
            noise_rms = self.noise_rms[band_indices,]
        else:
            psf = self.psf
            noise_rms = self.noise_rms

        return Observation(
            images=new_image,
            variance=new_variance,
            weights=new_weights,
            psf=psf,
            model_psf=self.model_psf,
            noise_rms=noise_rms,
            bbox=new_image.bbox,
            bands=new_bands,
            padding=self.padding,
            convolution_mode=self.mode,
        )

    def __copy__(self) -> Observation:
        """Create a copy of the observation

        Returns
        -------
        result:
            The copy of the observation.
        """
        return Observation(
            images=self.images,
            variance=self.variance,
            weights=self.weights,
            psf=self.psf,
            model_psf=self.model_psf,
            noise_rms=self.noise_rms,
            bands=self.bands,
            padding=self.padding,
            convolution_mode=self.mode,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> Observation:
        """Create a deep copy of the observation

        Parameters
        ----------
        memo: dict[int, Any]
            The memoization dictionary used by `copy.deepcopy`.

        Returns
        -------
        result:
            The deep copy of the observation.
        """
        # Check if already copied
        if id(self) in memo:
            return memo[id(self)]

        # Create placeholder and add to memo FIRST
        result = Observation.__new__(Observation)
        memo[id(self)] = result

        # Now safely initialize the placeholder with deepcopied arguments
        result.__init__(  # type: ignore[misc]
            images=deepcopy(self.images, memo),
            variance=deepcopy(self.variance, memo),
            weights=deepcopy(self.weights, memo),
            psf=deepcopy(self.psf, memo),
            model_psf=deepcopy(self.model_psf, memo),
            noise_rms=deepcopy(self.noise_rms, memo),
            bands=deepcopy(self.bands, memo),
            padding=deepcopy(self.padding, memo),
            convolution_mode=self.mode,
        )

        return result

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
        if deep:
            return self.__deepcopy__({})
        return self.__copy__()

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

    @staticmethod
    def empty(
        bands: tuple[Any],
        psf: np.ndarray | Psf = _REQUIRED,
        model_psf: np.ndarray | Psf = _REQUIRED,
        bbox: Box = _REQUIRED,
        dtype: npt.DTypeLike = _REQUIRED,
        *,
        psfs: np.ndarray | None = None,
    ) -> Observation:
        """Create an observation with no image data.

        Parameters
        ----------
        bands:
            The bands of the observation.
        psf:
            The observed PSF.
        model_psf:
            The model-space PSF, as a `Psf` or a ``(y, x)`` array.
        bbox:
            The bounding box of the (empty) model.
        dtype:
            The dtype of the dummy image data.
        psfs:
            Deprecated alias for `psf` as an array with shape (bands, y, x).
            Will be removed after v31.0.

        Returns
        -------
        result:
            An `Observation` whose images, variance and weights are all zero.
        """
        # The three steps below exist only to support the deprecated `psfs`
        # alias and the `_REQUIRED` defaults it forces (see `_Required`). Once
        # both are removed after v31.0 this method collapses to just building
        # `dummy_image` and the final `return`.

        # 1. Resolve the deprecated `psfs` alias onto `psf`.
        if psfs is not None:
            if psf is not _REQUIRED:
                raise ValueError("Provide only one of `psf` or the deprecated `psfs`.")
            warnings.warn(
                "The `psfs` argument is deprecated in favor of `psf` and will be "
                "removed after v31.0. Pass a `Psf` as `psf` instead.",
                FutureWarning,
                stacklevel=2,
            )
            psf = psfs

        # 2. Re-impose the "required" contract on the temporarily-defaulted
        #    arguments. This must run before anything reads them (e.g. the
        #    `bbox.shape`/`dtype` use below) so a missing argument raises a
        #    clear ``TypeError`` rather than an obscure ``AttributeError``.
        missing = [
            name
            for name, value in (("psf", psf), ("model_psf", model_psf), ("bbox", bbox), ("dtype", dtype))
            if value is _REQUIRED
        ]
        if missing:
            raise TypeError(f"empty() missing required argument(s): {', '.join(map(repr, missing))}")

        # 3. Coerce raw ndarrays to `Psf`, independent of how `psf` arrived.
        if not isinstance(psf, Psf):
            warnings.warn(
                "Passing an ndarray as `psf` is deprecated and will be unsupported "
                "after v31.0. Wrap it in an `ImagePsf` instead.",
                FutureWarning,
                stacklevel=2,
            )
            psf = ImagePsf(psf, bands=bands, padding=3)

        if not isinstance(model_psf, Psf):
            warnings.warn(
                "Passing an ndarray as `model_psf` is deprecated and will be "
                "unsupported after v31.0. Wrap it in an `ImagePsf` instead.",
                FutureWarning,
                stacklevel=2,
            )
            model_psf = ImagePsf(model_psf, bands=bands, padding=3)

        dummy_image = np.zeros((len(bands),) + bbox.shape, dtype=dtype)
        return Observation(
            images=dummy_image,
            variance=dummy_image,
            weights=dummy_image,
            psf=psf,
            model_psf=model_psf,
            noise_rms=np.zeros((len(bands),), dtype=dtype),
            bbox=bbox,
            bands=bands,
            convolution_mode="real",
        )
