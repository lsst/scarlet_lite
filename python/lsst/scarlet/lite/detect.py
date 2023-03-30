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

import logging
from typing import Sequence, cast

import numpy as np
from lsst.scarlet.lite.detect_pybind11 import Footprint  # type: ignore

from .bbox import Box
from .image import Image
from .utils import continue_class
from .wavelet import get_multiresolution_support, starlet_transform

logger = logging.getLogger("scarlet.detect")


def bounds_to_bbox(bounds: tuple[int, int, int, int]) -> Box:
    """Convert the bounds of a Footprint into a Box

    Notes
    -----
    Unlike slices, the bounds are _inclusive_ of the end points.

    Parameters
    ----------
    bounds:
        The bounds of the `Footprint` as a `tuple` of
        ``(bottom, top, left, right)``.
    Returns
    -------
    result:
        The `Box` created from the bounds
    """
    return Box(
        (bounds[1] + 1 - bounds[0], bounds[3] + 1 - bounds[2]),
        origin=(bounds[0], bounds[2]),
    )


@continue_class
class Footprint:  # type: ignore # noqa
    @property
    def bbox(self) -> Box:
        """Bounding box for the Footprint

        Returns
        -------
        bbox:
            The minimal `Box` that contains the entire `Footprint`.
        """
        return bounds_to_bbox(self.bounds)  # type: ignore

    @property
    def yx0(self) -> tuple[int, int]:
        """Origin in y, x of the lower left corner of the footprint"""
        return self.bounds[0], self.bounds[2]  # type: ignore

    def intersection(self, other: Footprint) -> Image | None:
        """The intersection of two footprints

        Parameters
        ----------
        other:
            The other footprint to compare.

        Returns
        -------
        intersection:
            The intersection of two footprints.
        """
        footprint1 = Image(self.data, yx0=self.yx0)  # type: ignore
        footprint2 = Image(other.data, yx0=other.yx0)  # type: ignore # noqa
        return footprint1 & footprint2

    def union(self, other: Footprint) -> Image | None:
        """The intersection of two footprints

        Parameters
        ----------
        other:
            The other footprint to compare.

        Returns
        -------
        union:
            The union of two footprints.
        """
        footprint1 = Image(self.data, yx0=self.yx0)  # type: ignore
        footprint2 = Image(other.data, yx0=other.yx0)
        return footprint1 | footprint2


def footprints_to_image(footprints: Sequence[Footprint], shape: tuple[int, int]) -> Image:
    """Convert a set of scarlet footprints to a pixelized image.

    Parameters
    ----------
    footprints:
        The footprints to convert into an image.
    shape:
        The shape of the image that is created from the footprints.

    Returns
    -------
    result:
        The image created from the footprints.
    """
    result = Image.from_box(Box(shape), dtype=int)
    for k, footprint in enumerate(footprints):
        bbox = bounds_to_bbox(footprint.bounds)
        fp_image = Image(footprint.data, yx0=cast(tuple[int, int], bbox.origin))
        result = result + fp_image * (k + 1)
    return result


def get_wavelets(images: np.ndarray, variance: np.ndarray, scales: int | None = None) -> np.ndarray:
    """Calculate wavelet coefficents given a set of images and their variances

    Parameters
    ----------
    images:
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance:
        An array of variances with the same shape as `images`.
    scales:
        The maximum number of wavelet scales to use.

    Returns
    -------
    coeffs:
        The array of coefficents with shape `(scales+1, bands, Ny, Nx)`.
        Note that the result has `scales+1` total arrays,
        since the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.
    """
    sigma = np.median(np.sqrt(variance), axis=(1, 2))
    # Create the wavelet coefficients for the significant pixels
    coeffs = []
    for b, image in enumerate(images):
        _coeffs = starlet_transform(image, scales=scales)
        support = get_multiresolution_support(
            image=image,
            starlets=_coeffs,
            sigma=sigma[b],
            sigma_scaling=3,
            epsilon=1e-1,
            max_iter=20,
        )
        coeffs.append(support * _coeffs)
    return np.array(coeffs)


def get_detect_wavelets(images: np.ndarray, variance: np.ndarray, scales: int = 3) -> np.ndarray:
    """Get an array of wavelet coefficents to use for detection

    Parameters
    ----------
    images:
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance:
        An array of variances with the same shape as `images`.
    scales:
        The maximum number of wavelet scales to use.
        Note that the result will have `scales+1` total arrays,
        where the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.

    Returns
    -------
    starlets:
        The array of wavelet coefficients for pixels with siignificant
        amplitude in each scale.
    """
    sigma = np.median(np.sqrt(variance))
    # Create the wavelet coefficients for the significant pixels
    detect = np.sum(images, axis=0)
    _coeffs = starlet_transform(detect, scales=scales)
    support = get_multiresolution_support(
        image=detect,
        starlets=_coeffs,
        sigma=sigma,  # type: ignore
        sigma_scaling=3,
        epsilon=1e-1,
        max_iter=20,
    )
    return support * _coeffs
