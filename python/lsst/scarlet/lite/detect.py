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
from dataclasses import dataclass
from lsst.scarlet.lite.detect_pybind11 import Footprint, get_footprints, Peak  # type: ignore

from .bbox import Box, overlapped_slices
from .image import Image
from .utils import continue_class
from .wavelet import (
    get_multiresolution_support,
    get_starlet_scales,
    starlet_reconstruction,
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

    Parameters
    ----------
    bbox:
        The `Box` to convert into bounds.

    Returns
    -------
    result:
        The bounds of the `Footprint` as a `tuple` of
        ``(bottom, top, left, right)``.

    Notes
    -----
    Unlike slices, the bounds are _inclusive_ of the end points.
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


def get_support(
    image: np.ndarray,
    sigma: float | None = None,
    epsilon: float = 1e-1
):
    if sigma is None:
        sigma = np.median(np.absolute(image - np.median(image)))
    last_sigma = sigma
    for _ in range(20):
        m = np.abs(image) > 5 * sigma
        s = ~m
        sigma = np.std(image*s.astype(int))
        if np.abs(sigma - last_sigma)/sigma < epsilon:
            break
    return m, sigma


@dataclass
class DetectionResult:
    image_sigmas: list[float]
    detection: np.ndarray
    starlets: np.ndarray
    peak_footprints: list[Footprint]
    starlet_sigma: float
    starlet_support: np.ndarray
    detection_sigma: float
    detection_support: np.ndarray
    footprints: list[Footprint]
    peaks: list[Peak]
    dropped_peaks: list[Peak]


def detect_footprints(
    images: np.ndarray,
    detection: np.ndarray | None = None,
    starlet_scale: int = 1,
    generation: int = 2,
    origin: tuple[int, int] | None = None,
    min_separation: float = 4,
    min_starlet_area: int = 4,
    peak_thresh: float = 3,
    footprint_thresh: float = 2,
    min_footprint_area: int = 8,
) -> DetectionResult:
    """Detect footprints in an image

    Parameters
    ----------
    images:
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    detection:
        An optional detection image. If `None` then one is created.
    starlet_scale:
        The scale of the starlet transform to use for detection.
        All of the default configs are tuned for `starlet_scale=1`.
        If using `starlet_scale=2` then it is recommended to use:
          - `min_starlet_area=1`
          - `peak_thresh=2`
    generation:
        The generation of the starlet transform to use.
    origin:
        The location (y, x) of the lower corner of the image.
    min_separation:
        The minimum separation between peaks in pixels.
    min_starlet_area:
        The minimum area of a footprint in starlet space in pixels.
    peak_thresh:
        The threshold for peak detection.
    footprint_thresh:
        The threshold for footprint detection.
    min_footprint_area:
        The minimum area of a footprint in the image in pixels.
    """

    if origin is None:
        origin = (0, 0)
    y0, x0 = origin

    if detection is None:
        # Find the standard deviation of the noise in each band
        sigmas = []
        for image in images:
            support, sigma = get_support(image)
            sigmas.append(sigma)

        # Create the variance weighted detection image
        detection = np.sum([image/sigma for image, sigma in zip(images, sigmas)], axis=0)

    # Use the chosen scale of starlets to act as a compensated filter
    # for detection
    starlets = starlet_transform(detection, scales=starlet_scale+1, generation=generation)

    # Estimate the noise in the detection image
    starlet_sigma = np.median(np.absolute(detection - np.median(detection)))
    starlet_support = get_multiresolution_support(
        image=detection,
        starlets=starlets,
        sigma=starlet_sigma,
        sigma_scaling=3,
    )

    # Detect peaks on the detection image
    peak_footprints = get_footprints(
        starlets[starlet_scale],
        min_separation,
        min_starlet_area,
        peak_thresh*starlet_support.sigma[starlet_scale],
        footprint_thresh*starlet_support.sigma[starlet_scale],
        True,
        y0,
        x0,
    )

    # Extract the peaks from the detection footprints
    peaks = [peak for fp in peak_footprints for peak in fp.peaks]

    # Create a new set of footprints on the images.
    # Detection is made on the boolean image, so the parameters
    # that we pass to it are simplified.
    support, sigma = get_support(detection)
    footprints = get_footprints(
        detection > sigma,
        0,
        min_footprint_area,
        0,
        0,
        False,
        y0,
        x0,
    )

    # Create an image of all of the footprints
    footprint_image = footprints_to_image(footprints, Box(detection.shape, origin=origin))
    dropped_peaks = []
    for peak in peaks:
        footprint_index = footprint_image.at(peak.y, peak.x) - 1
        if footprint_index >= 0:
            footprints[footprint_index].add_peak(peak)
        else:
            dropped_peaks.append(peak)
            logger.warning(f"Peak at ({peak.y}, {peak.x}) not in footprint")

    return DetectionResult(
        image_sigmas=sigmas,
        detection=detection,
        starlets=starlets,
        peak_footprints=peak_footprints,
        starlet_sigma=starlet_sigma,
        starlet_support=starlet_support,
        detection_sigma=sigma,
        detection_support=support,
        footprints=footprints,
        peaks=peaks,
        dropped_peaks=dropped_peaks,
    )


def detect_footprints_new(
    images: np.ndarray,
    detection: np.ndarray | None = None,
    skip_scales: list[int] | None = None,
    generation: int = 2,
    origin: tuple[int, int] | None = None,
    min_separation: float = 4,
    min_starlet_area: int = 4,
    peak_thresh: float = 3,
    footprint_thresh: float = 2,
    min_footprint_area: int = 8,
) -> DetectionResult:
    """Detect footprints in an image

    Parameters
    ----------
    images:
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    detection:
        An optional detection image. If `None` then one is created.
    starlet_scale:
        The scale of the starlet transform to use for detection.
        All of the default configs are tuned for `starlet_scale=1`.
        If using `starlet_scale=2` then it is recommended to use:
          - `min_starlet_area=1`
          - `peak_thresh=2`
    generation:
        The generation of the starlet transform to use.
    origin:
        The location (y, x) of the lower corner of the image.
    min_separation:
        The minimum separation between peaks in pixels.
    min_starlet_area:
        The minimum area of a footprint in starlet space in pixels.
    peak_thresh:
        The threshold for peak detection.
    footprint_thresh:
        The threshold for footprint detection.
    min_footprint_area:
        The minimum area of a footprint in the image in pixels.
    """
    if skip_scales is None:
        skip_scales = [0]
    starlet_scale = np.max(skip_scales) + 1

    if origin is None:
        origin = (0, 0)
    y0, x0 = origin

    if detection is None:
        # Find the standard deviation of the noise in each band
        sigmas = []
        for image in images:
            support, sigma = get_support(image)
            sigmas.append(sigma)

        # Create the variance weighted detection image
        detection = np.sum([image/sigma for image, sigma in zip(images, sigmas)], axis=0)

    # Use the chosen scale of starlets to act as a compensated filter
    # for detection
    starlets = starlet_transform(detection, scales=starlet_scale+1, generation=generation)
    starlet_sigma = np.median(np.absolute(detection - np.median(detection)))
    starlet_support = get_multiresolution_support(
        image=detection,
        starlets=starlets,
        sigma=starlet_sigma,
        sigma_scaling=3,
        image_type='space',
    )

    coeffs = starlet_support.support * starlets
    clean_detection = starlet_reconstruction(
        coeffs,
        skip_scales=skip_scales,
        generation=generation,
    )

    # Estimate the noise in the detection image
    starlet_sigma = np.median(np.absolute(clean_detection - np.median(clean_detection)))
    starlet_support, support_sigma = get_support(
        image=clean_detection,
        sigma=starlet_sigma,
    )

    # Detect peaks on the detection image
    peak_footprints = get_footprints(
        clean_detection,
        min_separation,
        min_starlet_area,
        peak_thresh*support_sigma,
        footprint_thresh*support_sigma,
        True,
        y0,
        x0,
    )

    # Remove this if we use this function
    peaks = [peak for fp in peak_footprints for peak in fp.peaks]
    footprints = peak_footprints
    dropped_peaks = []

    return DetectionResult(
        image_sigmas=sigmas,
        detection=clean_detection,
        starlets=starlets,
        peak_footprints=peak_footprints,
        starlet_sigma=starlet_sigma,
        starlet_support=starlet_support,
        detection_sigma=sigma,
        detection_support=support,
        footprints=footprints,
        peaks=peaks,
        dropped_peaks=dropped_peaks,
    )
