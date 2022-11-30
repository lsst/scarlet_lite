import logging
from typing import Sequence

import numpy as np

from .bbox import Box, overlapped_slices
from .detect_pybind11 import Footprint
from .wavelet import starlet_transform, get_multiresolution_support


logger = logging.getLogger("scarlet.detect")


def bounds_to_bbox(bounds: tuple[int, int, int, int]) -> Box:
    """Convert the bounds of a Footprint into a Box

    Parameters
    ----------
    bounds: Sequence[int, int, int, int]
        The bounds of the `Footprint`
    """
    return Box(
        (bounds[1] + 1 - bounds[0], bounds[3] + 1 - bounds[2]),
        origin=(bounds[0], bounds[2]),
    )


def footprint_intersect(
    footprint1: Footprint, box1: Box, footprint2: Footprint, box2: Box
) -> bool:
    """Check if two footprints overlap

    Parameters
    ----------
    box1, box2: Box
        The boxes of the footprints to check for overlap.
    footprint1, footprint2: Footprint
        The boolean mask for the two footprints.

    Returns
    -------
    overlap: `bool`
        True when the two footprints overlap.
    """
    if not box1.intersects(box2):
        return False
    slices1, slices2 = overlapped_slices(box1, box2)
    overlap = footprint1[slices1] * footprint2[slices2]
    return np.sum(overlap) > 0


def scarlet_footprints_to_image(
    footprints: Sequence[Footprint], shape: tuple[int, int]
) -> np.ndarray:
    """Convert a set of scarlet footprints to a pixelized image.

    Parameters
    ----------
    footprints:
        The footprints to convert into an iamge.
    shape:
        The shape of the image that is created from the footprints.

    Returns
    -------
    result: np.ndarray
        The image created from the footprints.
    """
    result = np.zeros(shape, dtype=int)
    for k, fp in enumerate(footprints):
        bbox = bounds_to_bbox(fp.bounds)
        result[bbox.slices] += fp.footprint * (k + 1)
    return result


def get_wavelets(
    images: np.ndarray, variance: np.ndarray, scales: int = 3
) -> np.ndarray:
    """Calculate wavelet coefficents given a set of images and their variances

    Parameters
    ----------
    images: np.ndarray
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance: np.ndarray
        An array of variances with the same shape as `images`.
    scales: int
        The maximum number of wavelet scales to use.
        Note that the result will have `scales+1` total arrays,
        where the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.

    Returns
    -------
    coeffs: np.ndarray
        The array of coefficents with shape `(scales+1, bands, Ny, Nx)`.
    """
    sigma = np.median(np.sqrt(variance), axis=(1, 2))
    # Create the wavelet coefficients for the significant pixels
    coeffs = []
    for b, image in enumerate(images):
        logger.debug(f"generating wavelets for band {b}")
        _coeffs = starlet_transform(image, scales=scales)
        support = get_multiresolution_support(
            image, _coeffs, sigma[b], sigma_scaling=3, epsilon=1e-1, max_iter=20
        )
        coeffs.append(support * _coeffs)
    return np.array(coeffs)


def get_detect_wavelets(
    images: np.ndarray, variance: np.ndarray, scales: int = 3
) -> np.ndarray:
    """Get an array of wavelet coefficents to use for detection

    Parameters
    images: np.ndarray
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance: np.ndarray
        An array of variances with the same shape as `images`.
    scales: int
        The maximum number of wavelet scales to use.
        Note that the result will have `scales+1` total arrays,
        where the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.
    """
    sigma = np.median(np.sqrt(variance))
    # Create the wavelet coefficients for the significant pixels
    detect = np.sum(images, axis=0)
    _coeffs = starlet_transform(detect, scales=scales)
    support = get_multiresolution_support(
        detect, _coeffs, sigma, sigma_scaling=3, epsilon=1e-1, max_iter=20  # type: ignore
    )
    return support * _coeffs
