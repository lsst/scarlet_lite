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

__all__ = ["Renderer", "convolve"]

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
    assert center is not None
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


class Renderer:
    def __init__(
        self,
        bands: tuple,
        psfs: np.ndarray,
        model_psf: np.ndarray | None,
        padding: int = 3,
        mode: str = "fft",
    ):
        self._bands = bands
        self.mode = mode
        if len(psfs) != len(bands):
            raise ValueError(f"Number of PSFs must match number of bands, got {len(psfs)} and{len(bands)}")
        self.update_difference_kernel(psfs, model_psf, padding)

        if mode not in ["fft", "real"]:
            raise ValueError("convolution_mode must be either 'fft' or 'real'")

    def update_difference_kernel(self, psfs: np.ndarray, model_psf: np.ndarray | None, padding: int):
        if model_psf is not None:
            if model_psf.dtype != psfs.dtype:
                model_psf = model_psf.astype(psfs.dtype)
            self._diff_kernel: Fourier | None = cast(Fourier, match_kernel(psfs, model_psf, padding=padding))
            # The gradient of a convolution is another convolution,
            # but with the flipped and transposed kernel.
            diff_img = self._diff_kernel.image
            self._grad_kernel: Fourier | None = Fourier(diff_img[:, ::-1, ::-1])
        else:
            self._diff_kernel = None
            self._grad_kernel = None

        self._convolution_bounds: tuple[int, int, int, int] | None = None
        self._psfs = psfs
        self._model_psf = model_psf
        self.padding = padding

    @property
    def bands(self) -> tuple:
        return self._bands

    @property
    def psfs(self) -> np.ndarray:
        return self._psfs

    @psfs.setter
    def psfs(self, psfs: np.ndarray):
        self.update_difference_kernel(psfs, self._model_psf, self.padding)

    @property
    def model_psf(self) -> np.ndarray | None:
        return self._model_psf

    @model_psf.setter
    def model_psf(self, model_psf: np.ndarray | None):
        self.update_difference_kernel(self._psfs, model_psf, self.padding)

    @property
    def diff_kernel(self) -> Fourier | None:
        return self._diff_kernel

    @property
    def grad_kernel(self) -> Fourier | None:
        return self._grad_kernel

    @property
    def convolution_bounds(self) -> tuple[int, int, int, int]:
        """Build the slices needed for convolution in real space"""
        if self._convolution_bounds is None:
            coords = get_filter_coords(cast(Fourier, self.diff_kernel).image[0])
            self._convolution_bounds = get_filter_bounds(coords.reshape(-1, 2))
        return self._convolution_bounds

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

    def __copy__(self, deep: bool = True) -> Renderer:
        if deep:
            if self.model_psf is None:
                model_psf = None
            else:
                model_psf = self.model_psf.copy()
            return Renderer(
                tuple([b for b in self.bands]),
                self.psfs.copy(),
                model_psf,
                self.padding,
                self.mode
            )
        else:
            return Renderer(
                self.bands,
                self.psfs,
                self.model_psf,
                self.padding,
                self.mode,
            )

    def copy(self, deep: bool = True) -> Renderer:
        return self.__copy__(deep)