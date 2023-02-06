import logging
from typing import Sequence, TypeVar

import numpy as np

from .bbox import Box
from lsst.scarlet.lite.detect_pybind11 import Footprint
from .image import Image
from .utils import continue_class
from .wavelet import starlet_transform, get_multiresolution_support


logger = logging.getLogger("scarlet.detect")
TFootprint = TypeVar("TFootprint", bound="Footprint")


def bounds_to_bbox(bounds: tuple[int, int, int, int]) -> Box:
    """Convert the bounds of a Footprint into a Box
    Parameters
    ----------
    bounds:
        The bounds of the `Footprint` as a `tuple` of
        ``(bottom, top, left, right)``.
    Returns
    -------
    result:
        The `Box` created fro the bounds
    """
    return Box(
        (bounds[1] + 1 - bounds[0], bounds[3] + 1 - bounds[2]),
        origin=(bounds[0], bounds[2]),
    )


@continue_class
class Footprint:  # noqa: F811
    @property
    def bbox(self) -> Box:
        """Bounding box for the Footprint

        Returns
        -------
        bbox:
            The minimal `Box` that contains the entire `Footprint`.
        """
        return bounds_to_bbox(self.bounds)

    @property
    def yx0(self) -> tuple[int, int]:
        """Origin in y, x of the lower left corner of the footprint"""
        return self.bounds[0], self.bounds[2]

    def intersection(self, other: TFootprint) -> Image | None:
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
        footprint1 = Image(self.footprint, yx0=self.yx0)
        footprint2 = Image(other.footprint, yx0=other.yx0)
        return footprint1 & footprint2

    def union(self, other: TFootprint) -> Image | None:
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
        footprint1 = Image(self.footprint, yx0=self.yx0)
        footprint2 = Image(other.footprint, yx0=other.yx0)
        return footprint1 | footprint2


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
    result:
        The image created from the footprints.
    """
    result = Image.from_box(Box(shape), dtype=int)
    for k, footprint in enumerate(footprints):
        bbox = bounds_to_bbox(footprint.bounds)
        fp_image = Image(footprint.footprint, yx0=bbox.origin)
        result = result + fp_image * (k + 1)
    return result


def get_wavelets(
    images: np.ndarray, variance: np.ndarray, scales: int = None
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
        Note that the result will have `scales+1` total arrays,
        where the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.

    Returns
    -------
    coeffs:
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
    """
    sigma = np.median(np.sqrt(variance))
    # Create the wavelet coefficients for the significant pixels
    detect = np.sum(images, axis=0)
    _coeffs = starlet_transform(detect, scales=scales)
    support = get_multiresolution_support(
        detect, _coeffs, sigma, sigma_scaling=3, epsilon=1e-1, max_iter=20  # type: ignore
    )
    return support * _coeffs
