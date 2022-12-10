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

import sys
from typing import Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import erfc

from .bbox import overlapped_slices, get_minimal_boxsize, Box


ScalarLike = bool | int | float | complex
ScalarTypes = (bool, int, float, complex)


def insert_image(
    image_box: Box,
    sub_box: Box,
    sub_image: np.ndarray,
    fill: float = 0,
    dtype: npt.DTypeLike = None,
) -> np.ndarray:
    """Insert an image given both an image box and sub-image box

    Parameters
    ----------
    image_box: Box
        The bounding box that will contain the full image.
    sub_box: Box
        The bounding box of the sub-image that is inserted into the full image.
    sub_image: np.ndarray
        The image that will be inserted.
    dtype: npt.DTypeLike
        The dtype of the resulting image.
    fill: float
        The fill value of the image for pixels outside of `sub_image`.

    Returns
    -------
    image: np.ndarray
        An image of `fill` with the pixels in `sub_box` replaced
        with `sub_image`.
    """
    if dtype is None:
        dtype = sub_image.dtype
    if fill != 0:
        image = np.full(image_box.shape, fill, dtype=dtype)
    else:
        image = np.zeros(image_box.shape, dtype=dtype)
    slices = overlapped_slices(image_box, sub_box)
    image[slices[0]] = sub_image[slices[1]]
    return image


def project_morph_to_center(
    morph: np.ndarray,
    center: Sequence[int],
    bbox: Box,
    fullbox: Box,
    boxsize: int = None,
) -> tuple[np.ndarray, Box]:
    """Project an uncentered morphology into a box that is centered on it

    Since most astrophysical sources are roughly symmetric,
    assuming that there will be an equal area of flux on opposing
    sides of the center is true in most cases.
    Projecting the morphology to the center of a square box
    makes it easier to update the flux without resizing with
    a minimal waste of memory.

    Parameters
    ----------
    morph: np.ndarray
        The 2D morphology that is to be centered.
    center: Squence[int]
        The center pixel of `morph` in `fullbox`.
    bbox: Box
        The bounding box of `morph`.
    fullbox: Box
        The bounding box of the full `image` in which
        `center` describes the center of the source.
    boxsize: int
        The size of the centered morphology.
        If `boxsize` is `None` then the minimal box needed to
        contain the centered morphology with an odd number of
        pixels in x and y (so that there is a center) is used.

    Returns
    -------
    centered: np.ndarray
        The centered morphology.
    centered_box: `scarlet.bbox.Box`
        The bounding box that contains the centered morphology
        in coordinates of the `fullbaox`.
    """
    # find fitting bbox
    if bbox.contains(center):
        size = 2 * max(
            (
                center[0] - bbox.start[-2],
                bbox.stop[0] - center[-2],
                center[1] - bbox.start[-1],
                bbox.stop[1] - center[-1],
            )
        )
    else:
        size = 0

    # define new box and cut morphology accordingly
    if boxsize is None:
        boxsize = get_minimal_boxsize(size)

    bottom = center[0] - boxsize // 2
    top = center[0] + boxsize // 2 + 1
    left = center[1] - boxsize // 2
    right = center[1] + boxsize // 2 + 1
    centered_box = Box.from_bounds((bottom, top), (left, right))

    centered = np.zeros(centered_box.shape, dtype=morph.dtype)
    slices = overlapped_slices(centered_box, fullbox)
    centered[slices[0]] = morph[slices[1]]

    return centered, centered_box


def integrated_gaussian_psf(x: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian function evaluated at `x`

    Parameters
    ----------
    x: np.ndarray
        The coordinates to evaluate the integrated Gaussian.
    sigma: float
        The standard deviation of the Gaussian.

    Returns
    -------
    gaussian: np.ndarray
        A Gaussian function integrated over `x`
    """
    sqrt2 = np.sqrt(2)
    lhs = erfc((0.5 - x) / (sqrt2 * sigma))
    rhs = erfc((2 * x + 1) / (2 * sqrt2 * sigma))
    return np.sqrt(np.pi / 2) * sigma * (1 - lhs + 1 - rhs)


def integrated_circular_gaussian(
    x: npt.ArrayLike = None, y: npt.ArrayLike = None, sigma: float = 0.8
) -> np.ndarray:
    """Create a circular Gaussian that is integrated over pixels

    This is typically used for the model PSF,
    working well with the default parameters.

    Parameters
    ----------
    x, y: npt.ArrayLike
        The x,y-coordinates to evaluate the integrated Gaussian.
        If `X` and `Y` are `None` then they will both be given the
        default value `numpy.arange(-7, 8)`, resulting in a
        `15x15` centered image.
    sigma: `float`
        The standard deviation of the Gaussian.

    Returns
    -------
    image: np.ndarray
        A Gaussian function integrated over `X` and `Y`.
    """
    if x is None:
        if y is None:
            x = np.arange(-7, 8)
            y = x
        else:
            raise Exception(
                f"Either X and Y must be specified, or neither must be specified, got X={x} and Y={y}"
            )
    result = (
        integrated_gaussian_psf(x, sigma)[None, :]
        * integrated_gaussian_psf(y, sigma)[:, None]
    )
    return result / np.sum(result)


def get_circle_mask(diameter: int, dtype: npt.DTypeLike = np.float64):
    """Get a boolean image of a circle

    Parameters
    ----------
    diameter: int
        The diameter of the circle and width
        of the image.
    dtype: npt.DTypeLike
        The `dtype` of the image.

    Returns
    -------
    circle: `numpy.ndarray`
        A boolean array with ones for the
        inside of the circle and zeros
        outside of the circle.
    """
    c = (diameter - 1) / 2
    # The center of the circle and its radius are
    # off by half a pixel for circles with
    # even numbered diameter
    if diameter % 2 == 0:
        radius = diameter / 2
    else:
        radius = c
    x = np.arange(diameter)
    x, y = np.meshgrid(x, x)
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2)

    circle = np.ones((diameter, diameter), dtype=dtype)
    circle[r > radius] = 0
    return circle


INTRINSIC_SPECIAL_ATTRIBUTES = frozenset(
    (
        "__qualname__",
        "__module__",
        "__metaclass__",
        "__dict__",
        "__weakref__",
        "__class__",
        "__subclasshook__",
        "__name__",
        "__doc__",
    )
)


def is_attribute_safe_to_transfer(name, value):
    """Return True if an attribute is safe to monkeypatch-transfer to another
    class.
    This rejects special methods that are defined automatically for all
    classes, leaving only those explicitly defined in a class decorated by
    `continueClass` or registered with an instance of `TemplateMeta`.
    """
    if name.startswith("__") and (
        value is getattr(object, name, None) or name in INTRINSIC_SPECIAL_ATTRIBUTES
    ):
        return False
    return True


def continue_class(cls):
    """Re-open the decorated class, adding any new definitions into the
    original.
    For example:
    .. code-block:: python
        class Foo:
            pass
        @continueClass
        class Foo:
            def run(self):
                return None
    is equivalent to:
    .. code-block:: python
        class Foo:
            def run(self):
                return None
    .. warning::
        Python's built-in `super` function does not behave properly in classes
        decorated with `continue_class`.  Base class methods must be invoked
        directly using their explicit types instead.

    This is copied directly from lsst.utils. If any additional functions are
    used from that repo we should remove this function and make lsst.utils
    a dependency. But for now, it is easier to copy this single wrapper
    than to include lsst.utils and all of its dependencies.
    """
    orig = getattr(sys.modules[cls.__module__], cls.__name__)
    for name in dir(cls):
        # Common descriptors like classmethod and staticmethod can only be
        # accessed without invoking their magic if we use __dict__; if we use
        # getattr on those we'll get e.g. a bound method instance on the dummy
        # class rather than a classmethod instance we can put on the target
        # class.
        attr = cls.__dict__.get(name, None) or getattr(cls, name)
        if is_attribute_safe_to_transfer(name, attr):
            setattr(orig, name, attr)
    return orig
