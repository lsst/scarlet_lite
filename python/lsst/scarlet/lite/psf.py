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

__all__ = ["Psf", "ImagePsf"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import DTypeLike

from .fft import Fourier, _pad, centered
from .fft import convolve as fft_convolve
from .fft import match_kernel
from .image import Image

if TYPE_CHECKING:
    from .io.psf import ImagePsfData, PsfBaseData


class Psf(ABC):
    """A point spread function used to convolve a model into observed seeing.

    A `Psf` owns both directions of its convolution operator: the forward
    convolution (`convolve`) and its adjoint (`grad`), used for the gradient
    pass in the optimizer. The kernel that is actually applied by an
    `Observation` is the *difference* kernel built by `match`, not the PSF
    itself, but both are represented as `Psf` instances so the optimizer,
    measurement and IO code never need to special-case the concrete type.
    """

    @abstractmethod
    def to_data(self) -> PsfBaseData:
        """Convert this PSF into a persistable data object.

        Returns
        -------
        result:
            The `~lsst.scarlet.lite.io.PsfBaseData` that serializes this
            PSF, mirroring `Blend.to_data`, `Component.to_data`, etc.
        """

    @property
    @abstractmethod
    def bands(self) -> tuple:
        """The bands of the PSF."""

    @property
    @abstractmethod
    def dtype(self) -> DTypeLike:
        """The numpy dtype of the PSF."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """The spatial ``(height, width)`` of the PSF kernel."""

    @abstractmethod
    def astype(self, dtype: DTypeLike) -> Psf:
        """Return a copy of this PSF cast to a new dtype.

        Parameters
        ----------
        dtype:
            The numpy dtype of the returned PSF.

        Returns
        -------
        result:
            A copy of this PSF with its data cast to ``dtype``.
        """

    @abstractmethod
    def __getitem__(self, bands: object) -> Psf:
        """Select a subset (or reordering) of bands as a new PSF.

        Parameters
        ----------
        bands:
            A band, or tuple of bands, to select from this PSF.

        Returns
        -------
        result:
            A new PSF containing only the requested bands, in order.
        """

    @abstractmethod
    def get_image(self, center: tuple[int, int] | None = None) -> Image:
        """Return the PSF image at a model-coordinate location.

        Parameters
        ----------
        center:
            The ``(y, x)`` location in the model frame at which the PSF is
            requested. Spatially-constant PSFs ignore this argument.

        Returns
        -------
        result:
            The PSF at ``center`` as an `Image`.
        """

    @abstractmethod
    def match(self, other: Psf, padding: int | None = None) -> Psf:
        """Build the difference kernel that matches ``other`` to ``self``.

        The result is a new `Psf` ``K`` such that convolving an image with
        ``other`` seeing by ``K`` yields an image with ``self`` seeing
        (in k-space, ``self / other``).

        Parameters
        ----------
        other:
            The PSF to match from (typically the model PSF).
        padding:
            Padding to use when generating the FFT. If `None`, the PSF's own
            ``padding`` is used.

        Returns
        -------
        result:
            The difference kernel as a new `Psf` of the same concrete type.
        """

    @abstractmethod
    def convolve(self, image: Image, mode: str | None = None, cache: bool = False) -> Image:
        """Apply this PSF's forward convolution operator to ``image``.

        Parameters
        ----------
        image:
            The multi-band image to convolve.
        mode:
            The convolution mode, ``"fft"`` or ``"real"``. If `None`, the
            descendant of this PSF's concrete type must choose a sensible
            default.
        cache:
            Whether to cache the FFT of the kernel at this image's shape.

        Returns
        -------
        result:
            The convolved image.
        """

    @abstractmethod
    def grad(self, image: Image, mode: str | None = None, cache: bool = False) -> Image:
        """Apply the adjoint of `convolve` to ``image`` (the gradient pass).

        Parameters
        ----------
        image:
            The multi-band image (gradient) to convolve.
        mode:
            The convolution mode, ``"fft"`` or ``"real"``. If `None`, the
            descendant of this PSF's concrete type must choose a sensible
            default.
        cache:
            Whether to cache the FFT of the adjoint kernel at this image's
            shape.

        Returns
        -------
        result:
            The result of applying the adjoint convolution to ``image``.
        """


class ImagePsf(Psf):
    """A spatially-constant PSF backed by a ``(bands, height, width)`` array.

    This is the class that the legacy ``ndarray`` PSF maps onto. The same
    class is used both for the observed/model PSFs and for the difference
    kernel produced by `match` (the object an `Observation` actually convolves
    with).

    Parameters
    ----------
    data:
        The ``(bands, height, width)`` PSF image.
    bands:
        The bands of the PSF. May be empty for internal kernels whose bands
        are never inspected (e.g. a broadcast model PSF).
    padding:
        Padding to use when generating the FFT for convolution.
    """

    def __init__(
        self,
        data: np.ndarray,
        bands: tuple = (),
        padding: int = 3,
    ):
        self._data = data
        self._bands = tuple(bands) if bands is not None else ()
        self._padding = padding
        # Lazily-built helpers (see the corresponding properties/methods).
        self._fourier: Fourier | None = None
        self._adjoint: ImagePsf | None = None
        self._convolution_bounds: tuple[int, int, int, int] | None = None

    @classmethod
    def _from_fourier(cls, fourier: Fourier, bands: tuple, padding: int) -> ImagePsf:
        """Build an `ImagePsf` from a precomputed `Fourier`.

        Used by `match` so the difference kernel keeps the `Fourier` (and any
        FFTs already cached on it) rather than rebuilding it on first use.

        Parameters
        ----------
        fourier:
            The `Fourier` whose real-space image becomes the PSF data.
        bands:
            The bands of the PSF.
        padding:
            Padding to use when generating the FFT for convolution.

        Returns
        -------
        result:
            An `ImagePsf` wrapping ``fourier``.
        """
        result = cls(fourier.image, bands=bands, padding=padding)
        result._fourier = fourier
        return result

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """The real-space ``(bands, height, width)`` PSF image."""
        return self._data

    @property
    def bands(self) -> tuple:
        """Bands of the PSF. May be empty for internal kernels whose bands
        are never inspected.
        """
        return self._bands

    @property
    def dtype(self) -> DTypeLike:
        """numpy dtype of the PSF."""
        return self._data.dtype

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial ``(height, width)`` of the PSF kernel."""
        return cast(tuple[int, int], self._data.shape[-2:])

    @property
    def padding(self) -> int:
        """Padding used when generating the FFT for convolution."""
        return self._padding

    def astype(self, dtype: DTypeLike) -> ImagePsf:
        """Return a copy of this PSF cast to a new dtype.

        Parameters
        ----------
        dtype:
            The numpy dtype of the returned PSF.

        Returns
        -------
        result:
            A copy of this PSF with its data cast to ``dtype``.
        """
        return ImagePsf(self._data.astype(dtype), bands=self._bands, padding=self._padding)

    def __getitem__(self, bands: object) -> ImagePsf:
        """Return a new `ImagePsf` containing only the specified bands.

        Parameters
        ----------
        bands:
            A band, or tuple of bands, to select from this PSF.

        Returns
        -------
        result:
            A new `ImagePsf` containing only the requested bands, in order.
        """
        if not isinstance(bands, tuple):
            bands = (bands,)
        indices = tuple(self._bands.index(b) for b in bands)
        return ImagePsf(self._data[indices, ...], bands=bands, padding=self._padding)

    def get_image(self, center: tuple[int, int] | None = None) -> Image:
        """Return the PSF image at a model-coordinate location.

        Parameters
        ----------
        center:
            The ``(y, x)`` location in the model frame. Ignored, since an
            `ImagePsf` is spatially constant.

        Returns
        -------
        result:
            The PSF as an `Image`. A band-less PSF (e.g. the model PSF, which
            is stored as a ``(1, y, x)`` broadcast cube) is returned as a 2D
            image; a multi-band PSF is returned with its band dimension.
        """
        # An ImagePsf is spatially constant, so ``center`` is ignored.
        if len(self._bands) == 0:
            data = self._data
            if data.ndim == 3:
                # A band-less PSF is logically 2D; drop a singleton broadcast
                # axis. Refuse to silently discard a genuine multi-band cube.
                if data.shape[0] != 1:
                    raise ValueError(
                        "Cannot return a 2D image for a band-less PSF backed by "
                        f"a multi-band array (shape {data.shape}); the band to "
                        "use is ambiguous."
                    )
                data = data[0]
            return Image(data)
        return Image(self._data, bands=self._bands)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_data(self) -> ImagePsfData:
        """Convert this PSF into a persistable data object.

        Returns
        -------
        result:
            A `~lsst.scarlet.lite.io.ImagePsfData` carrying this PSF's
            array, bands and padding.
        """
        from .io.psf import ImagePsfData

        return ImagePsfData(
            data=self._data,
            bands=self._bands,
            padding=self._padding,
        )

    # ------------------------------------------------------------------
    # Convolution helpers
    # ------------------------------------------------------------------
    @property
    def fourier(self) -> Fourier:
        """The `Fourier` representation of the kernel (lazily built)."""
        if self._fourier is None:
            self._fourier = Fourier(self._data)
        return self._fourier

    @property
    def adjoint(self) -> ImagePsf:
        """The adjoint kernel (spatial flip), lazily built and cached.

        Convolving with the adjoint is the transpose of `convolve`, which is
        exactly the gradient pass. It is a separate `ImagePsf`, so its FFT
        cache is independent of the forward kernel's.
        """
        if self._adjoint is None:
            self._adjoint = ImagePsf(
                self._data[:, ::-1, ::-1],
                bands=self._bands,
                padding=self._padding,
            )
        return self._adjoint

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """The real-space convolution filter bounds (lazily built)."""
        if self._convolution_bounds is None:
            from .observation import get_filter_bounds, get_filter_coords

            coords = get_filter_coords(self._data[0])
            self._convolution_bounds = get_filter_bounds(coords.reshape(-1, 2))
        return self._convolution_bounds

    def match(self, other: Psf, padding: int | None = None) -> ImagePsf:
        """Build the difference kernel that matches ``other`` to ``self``.

        The result is a new `ImagePsf` ``K`` such that convolving an image
        with ``other`` seeing by ``K`` yields an image with ``self`` seeing
        (in k-space, ``self / other``).

        Parameters
        ----------
        other:
            The PSF to match from (typically the model PSF). Must be an
            `ImagePsf`.
        padding:
            Padding to use when generating the FFT. If `None`, this PSF's own
            ``padding`` is used.

        Returns
        -------
        result:
            The difference kernel as a new `ImagePsf`.

        Raises
        ------
        NotImplementedError
            If ``other`` is not an `ImagePsf`.
        """
        if padding is None:
            padding = self._padding
        if not isinstance(other, ImagePsf):
            raise NotImplementedError(f"ImagePsf can only be matched to another ImagePsf, got {type(other)}")
        fourier = cast(
            Fourier,
            match_kernel(self._data, other.data, padding=padding, return_fourier=True),
        )
        return ImagePsf._from_fourier(fourier, bands=self._bands, padding=padding)

    def convolve(self, image: Image, mode: str | None = None, cache: bool = False) -> Image:
        """Apply this PSF's forward convolution operator to ``image``.

        Parameters
        ----------
        image:
            The multi-band image to convolve.
        mode:
            The convolution mode, ``"fft"`` or ``"real"``. If `None`,
            ``"fft"`` is used.
        cache:
            Whether to cache the FFT of the kernel at this image's shape.
            Ignored for ``mode == "real"``.

        Returns
        -------
        result:
            The convolved image, with the bands and origin of ``image``.

        Raises
        ------
        ValueError
            If ``mode`` is neither ``"fft"`` nor ``"real"``.
        """
        if mode is None:
            mode = "fft"
        if mode == "fft":
            result = fft_convolve(
                Fourier(image.data),
                self.fourier,
                axes=(1, 2),
                return_fourier=False,
                cache=cache,
            )
        elif mode == "real":
            result = self._convolve_real(image)
        else:
            raise ValueError(f"mode must be either 'fft' or 'real', got {mode}")
        return Image(cast(np.ndarray, result), bands=image.bands, yx0=image.yx0)

    def grad(self, image: Image, mode: str | None = None, cache: bool = False) -> Image:
        """Apply the adjoint of `convolve` to ``image`` (the gradient pass).

        The adjoint of a convolution is a convolution with the spatially
        flipped kernel, so the gradient pass is just the forward convolution
        of the `adjoint` PSF.

        Parameters
        ----------
        image:
            The multi-band image (gradient) to convolve.
        mode:
            The convolution mode, ``"fft"`` or ``"real"``. If `None`,
            ``"fft"`` is used.
        cache:
            Whether to cache the FFT of the adjoint kernel at this image's
            shape. Ignored for ``mode == "real"``.

        Returns
        -------
        result:
            The result of applying the adjoint convolution to ``image``.
        """
        return self.adjoint.convolve(image, mode=mode, cache=cache)

    def _convolve_real(self, image: Image) -> np.ndarray:
        """Real-space convolution with the kernel, handling small images.

        Parameters
        ----------
        image:
            The multi-band image to convolve. If it is smaller than the
            kernel it is zero-padded up to the kernel size and the result is
            trimmed back to the image shape.

        Returns
        -------
        result:
            The convolved data as a ``(bands, height, width)`` array matching
            ``image``.
        """
        from .observation import convolve as real_convolve

        kernel = self._data
        dy = image.shape[1] - kernel.shape[1]
        dx = image.shape[2] - kernel.shape[2]
        if dy < 0 or dx < 0:
            # The image needs to be padded because it is smaller than the
            # PSF kernel.
            _image = image.data
            newshape = list(_image.shape)
            if dy < 0:
                newshape[1] += kernel.shape[1] - image.shape[1]
            if dx < 0:
                newshape[2] += kernel.shape[2] - image.shape[2]
            _image = _pad(_image, newshape)
            result = real_convolve(_image, kernel, self.bounds)
            result = centered(result, image.data.shape)
        else:
            result = real_convolve(image.data, kernel, self.bounds)
        return cast(np.ndarray, result)
