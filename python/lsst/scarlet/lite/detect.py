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
from typing import Sequence

import numpy as np
from lsst.scarlet.lite.detect_pybind11 import Footprint, get_footprints  # type: ignore

from .bbox import Box, overlapped_slices
from .image import Image
from .utils import continue_class
from .wavelet import (
    get_multiresolution_support,
    get_starlet_scales,
    multiband_starlet_reconstruction,
    starlet_transform,
)

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


def bbox_to_bounds(bbox: Box) -> tuple[int, int, int, int]:
    """Convert a Box into the bounds of a Footprint

    Notes
    -----
    Unlike slices, the bounds are _inclusive_ of the end points.

    Parameters
    ----------
    bbox:
        The `Box` to convert into bounds.

    Returns
    -------
    result:
        The bounds of the `Footprint` as a `tuple` of
        ``(bottom, top, left, right)``.
    """
    bounds = (
        bbox.origin[0],
        bbox.origin[0] + bbox.shape[0] - 1,
        bbox.origin[1],
        bbox.origin[1] + bbox.shape[1] - 1,
    )
    return bounds


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


def footprints_to_image(footprints: Sequence[Footprint], bbox: Box) -> Image:
    """Convert a set of scarlet footprints to a pixelized image.

    Parameters
    ----------
    footprints:
        The footprints to convert into an image.
    box:
        The full box of the image that will contain the footprints.

    Returns
    -------
    result:
        The image created from the footprints.
    """
    result = Image.from_box(bbox, dtype=int)
    for k, footprint in enumerate(footprints):
        slices = overlapped_slices(result.bbox, footprint.bbox)
        result.data[slices[0]] += footprint.data[slices[1]] * (k + 1)
    return result


def get_wavelets(
    images: np.ndarray,
    variance: np.ndarray,
    scales: int | None = None,
    generation: int = 2,
) -> np.ndarray:
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
    scales = get_starlet_scales(images[0].shape, scales)
    coeffs = np.empty((scales + 1,) + images.shape, dtype=images.dtype)
    for b, image in enumerate(images):
        _coeffs = starlet_transform(image, scales=scales, generation=generation)
        support = get_multiresolution_support(
            image=image,
            starlets=_coeffs,
            sigma=sigma[b],
            sigma_scaling=3,
            epsilon=1e-1,
            max_iter=20,
        )
        coeffs[:, b] = (support.support * _coeffs).astype(images.dtype)
    return coeffs


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
    return (support.support * _coeffs).astype(images.dtype)


def detect_footprints(
    images: np.ndarray,
    variance: np.ndarray,
    scales: int = 1,
    generation: int = 2,
    origin: tuple[int, int] | None = None,
    min_separation: float = 4,
    min_area: int = 4,
    peak_thresh: float = 5,
    footprint_thresh: float = 5,
    find_peaks: bool = True,
    remove_high_freq: bool = True,
    min_pixel_detect: int = 1,
) -> list[Footprint]:
    """Detect footprints in an image

    Parameters
    ----------
    images:
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance:
        An array of variances with the same shape as `images`.
    scales:
        The maximum number of wavelet scales to use.
        If `remove_high_freq` is `False`, then this argument is ignored.
    generation:
        The generation of the starlet transform to use.
        If `remove_high_freq` is `False`, then this argument is ignored.
    origin:
        The location (y, x) of the lower corner of the image.
    min_separation:
        The minimum separation between peaks in pixels.
    min_area:
        The minimum area of a footprint in pixels.
    peak_thresh:
        The threshold for peak detection.
    footprint_thresh:
        The threshold for footprint detection.
    find_peaks:
        If `True`, then detect peaks in the detection image,
        otherwise only the footprints are returned.
    remove_high_freq:
        If `True`, then remove high frequency wavelet coefficients
        before detecting peaks.
    min_pixel_detect:
        The minimum number of bands that must be above the
        detection threshold for a pixel to be included in a footprint.
    """

    if origin is None:
        origin = (0, 0)
    if remove_high_freq:
        # Build the wavelet coefficients
        wavelets = get_wavelets(
            images,
            variance,
            scales=scales,
            generation=generation,
        )
        # Remove the high frequency wavelets.
        # This has the effect of preventing high frequency noise
        # from interfering with the detection of peak positions.
        wavelets[0] = 0
        # Reconstruct the image from the remaining wavelet coefficients
        _images = multiband_starlet_reconstruction(
            wavelets,
            generation=generation,
        )
    else:
        _images = images
    # Build a SNR weighted detection image
    sigma = np.median(np.sqrt(variance), axis=(1, 2)) / 2
    detection = np.sum(_images / sigma[:, None, None], axis=0)
    if min_pixel_detect > 1:
        mask = np.sum(images > 0, axis=0) >= min_pixel_detect
        detection[~mask] = 0
    # Detect peaks on the detection image
    footprints = get_footprints(
        detection,
        min_separation,
        min_area,
        peak_thresh,
        footprint_thresh,
        find_peaks,
        origin[0],
        origin[1],
    )

    return footprints
