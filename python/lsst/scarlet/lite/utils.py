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

import sys
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import erfc

ScalarLike = bool | int | float | complex
ScalarTypes = (bool, int, float, complex)


sqrt2 = np.sqrt(2)
sqrt_pi = np.sqrt(np.pi)


def integrated_gaussian_value(x: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian function evaluated at `x`

    Parameters
    ----------
    x:
        The coordinates to evaluate the integrated Gaussian
        (ie. the centers of pixels).
    sigma:
        The standard deviation of the Gaussian.

    Returns
    -------
    gaussian:
        A Gaussian function integrated over `x`
    """
    lhs = erfc((x - 0.5) / (sqrt2 * sigma))
    rhs = erfc((x + 0.5) / (sqrt2 * sigma))
    return sqrt_pi * 0.5 * sigma * (lhs - rhs)


def integrated_circular_gaussian(
    x: np.ndarray | None = None, y: np.ndarray | None = None, sigma: float = 0.8
) -> np.ndarray:
    """Create a circular Gaussian that is integrated over pixels

    This is typically used for the model PSF,
    working well with the default parameters.

    Parameters
    ----------
    x, y:
        The x,y-coordinates to evaluate the integrated Gaussian.
        If `X` and `Y` are `None` then they will both be given the
        default value `numpy.arange(-7, 8)`, resulting in a
        `15x15` centered image.
    sigma:
        The standard deviation of the Gaussian.

    Returns
    -------
    image:
        A Gaussian function integrated over `X` and `Y`.
    """
    if x is None:
        if y is None:
            x = np.arange(-7, 8)
            y = x
        else:
            raise ValueError(
                f"Either X and Y must be specified, or neither must be specified, got {x=} and {y=}"
            )
    elif y is None:
        raise ValueError(f"Either X and Y must be specified, or neither must be specified, got {x=} and {y=}")

    _x = integrated_gaussian_value(np.abs(x), sigma)[None, :]
    _y = integrated_gaussian_value(np.abs(y), sigma)[:, None]
    result = _x * _y
    return result / np.sum(result)


def get_circle_mask(diameter: int, dtype: npt.DTypeLike = np.float64):
    """Get a boolean image of a circle

    Parameters
    ----------
    diameter:
        The diameter of the circle and width
        of the image.
    dtype:
        The `dtype` of the image.

    Returns
    -------
    circle:
        A boolean array with ones for the pixels with centers
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
    _x = np.arange(diameter)
    x, y = np.meshgrid(_x, _x)
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


def convert_indices(sequence: Sequence, indices: Any, inclusive: bool = True) -> tuple[int, ...] | slice:
    """Get either a tuple of indices or a slice object from the given sequence.

    Parameters
    ----------
    sequence : Sequence
        The sequence to get the indices from. This sequence should have
        unique hashable elements.

    indices : Any
        The indices or slice to use. Can be:
        - A single element from sequence
        - A slice with start/stop elements from sequence
        - A sequence of elements from sequence

    inclusive : bool, optional
        If True, the stop element of a slice is inclusive.

    Returns
    -------
    tuple[int, ...] | slice
        A tuple of indices or a slice object.

    Raises
    ------
    TypeError :
        If `sequence` does not support `index` and `in` operations.
    IndexError :
        If a single element is not found in `sequence`.
    """
    # Validate that sequence has the required methods
    if not hasattr(sequence, "index") or not hasattr(sequence, "__contains__"):
        raise TypeError(f"'sequence' must support 'index' and 'in' operations, got {type(sequence)}")

    # Handle slice objects
    if isinstance(indices, slice):
        # Convert a slice of objects into a slice of array indices
        try:
            start = None if indices.start is None else sequence.index(indices.start)
        except ValueError as e:
            raise IndexError(f"Element {indices.start} not found in sequence {sequence}.") from e
        try:
            stop = None if indices.stop is None else sequence.index(indices.stop) + (1 if inclusive else 0)
        except ValueError as e:
            raise IndexError(f"Element {indices.stop} not found in sequence {sequence}.") from e
        return slice(start, stop, indices.step)

    # Try to handle as a single element first
    if indices in sequence:
        return (sequence.index(indices),)

    # Validate that indices is iterable
    if not hasattr(indices, "__iter__"):
        raise IndexError(f"Element {indices} not found in sequence {sequence}.")

    # Handle sequence of indices
    index_map = {value: idx for idx, value in enumerate(sequence)}
    new_indices = []
    for i in indices:
        try:
            if i not in index_map:
                raise IndexError(f"Element {i} not found in sequence {sequence}.")
        except TypeError as e:
            # If the
            raise IndexError(f"Element {i} not found in sequence {sequence}.") from e
        new_indices.append(index_map[i])

    return tuple(new_indices)


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
