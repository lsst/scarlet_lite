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

__all__ = ["Box", "overlapped_slices"]

from typing import Sequence, TypeVar

import numpy as np

TBox = TypeVar("TBox", bound="Box")


class Box:
    """Bounding Box for an object

    A Bounding box describes the location of a data unit in the
    global/model coordinate system, using the numpy/C++ ordering convention.
    So, for example, a 2D image will have shape ``(height, width)``,
    however the bounding `Box` code is agnostic as to number of dimensions
    or the meaning of those dimensions.

    Examples
    --------

    At a minimum a new `Box` can be initialized using the ``shape`` of the
    region it describes:

    >>> from lsst.scarlet.lite import Box
    >>> bbox = Box((3, 4, 5, 6))
    >>> print(bbox)
    <Box shape=(3, 4, 5, 6), origin=(0, 0, 0, 0)>

    If the region described by the `Box` is offset from the zero origin,
    a new ``origin`` can be passed to the constructor

    >>> bbox = Box((3, 4, 5, 6), (2, 4, 7, 9))
    >>> print(bbox)
    <Box shape=(3, 4, 5, 6), origin=(2, 4, 7, 9)>

    It is also possible to initialize a `Box` from a collection of tuples,
    where tuple is a pair of integers representing the
    first and last index in each dimension. For example:

    >>> bbox = Box.from_bounds((3, 6), (11, 21))
    >>> print(bbox)
    <Box shape=(3, 10), origin=(3, 11)>

    It is also possible to initialize a `Box` by thresholding a numpy array
    and including only the region of the image above the threshold in the
    resulting `Box`. For example

    >>> from lsst.scarlet.lite.utils import integrated_circular_gaussian
    >>> data = integrated_circular_gaussian(sigma=1.0)
    >>> bbox = Box.from_data(data, 1e-2)
    >>> print(bbox)
    <Box shape=(5, 5), origin=(5, 5)>

    The `Box` class contains a number of convenience methods that can be used
    to extract subsets of an array, combine bounding boxes, etc.

    For example, using the ``data`` and ``bbox`` from the end of the previous
    section, the portion of the data array that is contained in the bounding
    box can be extraced usng the `Box.slices` method:

    >>> subset = data[bbox.slices]

    The intersection of two boxes can be calcualted using the ``&`` operator,
    for example

    >>> bbox = Box((5, 5)) & Box((5, 5), (2, 2))
    >>> print(bbox)
    <Box shape=(3, 3), origin=(2, 2)>

    Similarly, the union of two boxes can be calculated using the ``|``
    operator:

    >>> bbox = Box((5, 5)) | Box((5, 5), (2, 2))
    >>> print(bbox)
    <Box shape=(7, 7), origin=(0, 0)>

    To find out of a point is located in a `Box` use

    >>> contains = bbox.contains((3, 3))
    >>> print(contains)
    True

    To find out if two boxes intersect (in other words ``box1 & box2`` has a
    non-zero size) use

    >>> intersects = bbox.intersects(Box((10, 10), (100, 100)))
    >>> print(intersects)
    False

    It is also possible to shift a box by a vector (sequence):

    >>> bbox = bbox + (50, 60)
    >>> print(bbox)
    <Box shape=(7, 7), origin=(50, 60)>

    which can also be negative

    >>> bbox = bbox - (5, -5)
    >>> print(bbox)
    <Box shape=(7, 7), origin=(45, 65)>

    Boxes can also be converted into higher dimensions using the
    ``@`` operator:

    >>> bbox1 = Box((10,), (3, ))
    >>> bbox2 = Box((101, 201), (18, 21))
    >>> bbox = bbox1 @ bbox2
    >>> print(bbox)
    <Box shape=(10, 101, 201), origin=(3, 18, 21)>

    Boxes are equal when they have the same shape and the same origin, so

    >>> print(Box((10, 10), (5, 5)) == Box((10, 10), (5, 5)))
    True

    >>> print(Box((10, 10), (5, 5)) == Box((10, 10), (4, 4)))
    False

    Finally, it is common to insert one array into another when their bounding
    boxes only partially overlap.
    In order to correctly insert the overlapping portion of the array it is
    convenient to calculate the slices from each array that overlap.
    For example

    >>> import numpy as np
    >>> x = np.arange(12).reshape(3, 4)
    >>> y = np.arange(9).reshape(3, 3)
    >>> print(x)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    >>> print(y)
    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    >>> x_box = Box.from_data(x) + (3, 4)
    >>> y_box = Box.from_data(y) + (1, 3)
    >>> slices = x_box.overlapped_slices(y_box)
    >>> x[slices[0]] += y[slices[1]]
    >>> print(x)
    [[ 7  9  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]

    Parameters
    ----------
    shape:
        Size of the box
    origin:
        Minimum corner coordinate of the box
    """

    def __init__(self, shape: tuple[int, ...], origin: tuple[int, ...] = None):
        self.shape = shape
        if origin is None:
            origin = (0,) * len(shape)
        assert len(origin) == len(shape)
        self.origin = tuple(origin)

    @staticmethod
    def from_bounds(*bounds: tuple[int, int]) -> TBox:
        """Initialize a box from its bounds

        Parameters
        ----------
        bounds:
            Min/Max coordinate for every dimension

        Returns
        -------
        bbox:
            A new box bounded by the input bounds.
        """
        shape = tuple(max(0, cmax - cmin) for cmin, cmax in bounds)
        origin = tuple(cmin for cmin, cmax in bounds)
        return Box(shape, origin=origin)

    @staticmethod
    def from_data(x: np.ndarray, min_value: float = 0) -> TBox:
        """Define range of `x` above `min_value`

        Parameters
        ----------
        x:
            Data to threshold
        min_value:
            Minimum value of the result.

        Returns
        -------
        bbox:
            Bounding box for the thresholded `x`
        """
        sel = x > min_value
        if sel.any():
            nonzero = np.where(sel)
            bounds = []
            for dim in range(len(x.shape)):
                bounds.append((nonzero[dim].min(), nonzero[dim].max() + 1))
        else:
            bounds = [[0, 0]] * len(x.shape)
        return Box.from_bounds(*bounds)

    def contains(self, p: Sequence[int]) -> bool:
        """Whether the box contains a given coordinate `p`"""
        if len(p) != self.dimensions:
            raise ValueError(f"Dimension mismatch in {p} and {self.dimensions}")

        for d in range(self.dimensions):
            if p[d] < self.origin[d] or p[d] >= self.origin[d] + self.shape[d]:
                return False
        return True

    @property
    def dimensions(self) -> int:
        """Dimensionality of this BBox"""
        return len(self.shape)

    @property
    def start(self) -> tuple[int, ...]:
        """Tuple of start coordinates"""
        return self.origin

    @property
    def stop(self) -> tuple[int, ...]:
        """Tuple of stop coordinates"""
        return tuple(o + s for o, s in zip(self.origin, self.shape))

    @property
    def center(self) -> tuple[float, ...]:
        """Tuple of center coordinates"""
        return tuple(o + s / 2 for o, s in zip(self.origin, self.shape))

    @property
    def bounds(self) -> tuple[tuple[int, int], ...]:
        """Bounds of the box"""
        return tuple((o, o + s) for o, s in zip(self.origin, self.shape))

    @property
    def slices(self) -> tuple[slice, ...]:
        """Bounds of the box as slices"""
        return tuple([slice(o, o + s) for o, s in zip(self.origin, self.shape)])

    def grow(self, radius: int | tuple[tuple[int, ...], ...]) -> TBox:
        """Grow the Box by the given radius in each direction"""
        if not hasattr(radius, "__iter__"):
            radius = [radius] * self.dimensions
        origin = tuple([self.origin[d] - radius[d] for d in range(self.dimensions)])
        shape = tuple([self.shape[d] + 2 * radius[d] for d in range(self.dimensions)])
        return Box(shape, origin=origin)

    def shifted_by(self, shift: Sequence[int]) -> TBox:
        """Generate a shifted copy of this box

        Parameters
        ----------
        shift:
            The amount to shift each axis to create the new box

        Returns
        -------
        result:
            The resulting bounding box.
        """
        origin = tuple(o + shift[i] for i, o in enumerate(self.origin))
        return Box(self.shape, origin=origin)

    def intersects(self, other: TBox) -> bool:
        """Check if two boxes overlap

        Parameters
        ----------
        other:
            The boxes to check for overlap

        Returns
        -------
        result:
            True when the two boxes overlap.
        """
        overlap = self & other
        return np.all(np.array(overlap.shape) != 0)

    def overlapped_slices(
        self, other: TBox
    ) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        """`slice` for the box that contains the overlap of this and
        another `Box`

        Parameters
        ----------
        other:

        Returns
        -------
        slices:
            The slice of an array bounded by `self` and
            the slice of an array bounded by `other` in the
            overlapping region.
        """
        return overlapped_slices(self, other)

    def __or__(self, other: TBox) -> TBox:
        """Union of two bounding boxes

        Parameters
        ----------
        other:
            The other bounding box in the union

        Returns
        -------
        result:
            The smallest rectangular box that contains *both* boxes.
        """
        if other.dimensions != self.dimensions:
            raise ValueError(f"Dimension mismatch in the boxes {other} and {self}")
        bounds = []
        for d in range(self.dimensions):
            bounds.append(
                (min(self.start[d], other.start[d]), max(self.stop[d], other.stop[d]))
            )
        return Box.from_bounds(*bounds)

    def __and__(self, other: TBox) -> TBox:
        """Intersection of two bounding boxes

        If there is no intersection between the two bounding
        boxes then an empty bounding box is returned.

        Parameters
        ----------
        other:
            The other bounding box in the intersection

        Returns
        -------
        result:
            The rectangular box that is in the overlap region
            of both boxes.
        """
        if other.dimensions != self.dimensions:
            raise ValueError(f"Dimension mismatch in the boxes {other} and {self}")
        assert other.dimensions == self.dimensions
        bounds = []
        for d in range(self.dimensions):
            bounds.append(
                (max(self.start[d], other.start[d]), min(self.stop[d], other.stop[d]))
            )
        return Box.from_bounds(*bounds)

    def __getitem__(self, index: int | slice | tuple[int, ...]) -> TBox:
        try:
            iter(index)
            # If I is a Sequence then select the indices in `index`, in order
            s_ = tuple(self.shape[i] for i in index)
            o_ = tuple(self.origin[i] for i in index)
        except TypeError:
            # The index is an integer or slice
            s_ = self.shape[index]
            o_ = self.origin[index]
            if not hasattr(s_, "__iter__"):
                s_ = (s_,)
                o_ = (o_,)
        return Box(s_, origin=o_)

    def __repr__(self) -> str:
        result = "<Box shape={0}, origin={1}>"
        return result.format(self.shape, self.origin)

    def _offset_to_tuple(self, offset: int | Sequence[int]) -> tuple[int, ...]:
        """Expand an integer offset into a tuple

        Parameters
        ----------
        offset:
            The offset to (potentially) convert into a tuple.

        Returns
        -------
        offset:
            The offset as a tuple.
        """
        if not hasattr(offset, "__iter__"):
            offset = (offset,) * self.dimensions
        return offset

    def __add__(self, offset: int | Sequence[int]) -> TBox:
        """Generate a new Box with a shifted offset

        Parameters
        ----------
        offset:
            The amount to shift the current offset

        Returns
        -------
        result:
            The shifted box.
        """
        return self.shifted_by(self._offset_to_tuple(offset))

    def __sub__(self, offset: int | Sequence[int]) -> TBox:
        """Generate a new Box with a shifted offset in the negative direction

        Parameters
        ----------
        offset:
            The amount to shift the current offset

        Returns
        -------
        result:
            The shifted box.
        """
        offset = self._offset_to_tuple(offset)
        offset = tuple(-o for o in offset)
        return self.shifted_by(offset)

    def __matmul__(self, bbox: TBox) -> TBox:
        """Combine two Boxes into a higher dimensional box

        Parameters
        ----------
        bbox:
            The box to append to this box.

        Returns
        -------
        result:
            The combined Box.
        """
        bounds = self.bounds + bbox.bounds
        result = Box.from_bounds(*bounds)
        return result

    def __copy__(self) -> TBox:
        """Copy of the box"""
        return Box(self.shape, origin=self.origin)

    def copy(self) -> TBox:
        """Copy of the box"""
        return self.__copy__()

    def __eq__(self, other: TBox) -> bool:
        """Check for equality.

        Two boxes are equal when they have the same shape and origin.
        """
        if not hasattr(other, "shape") and not hasattr(other, "origin"):
            return False
        return self.shape == other.shape and self.origin == other.origin

    def __hash__(self) -> int:
        return hash((self.shape, self.origin))


def overlapped_slices(
    bbox1: Box, bbox2: Box
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Slices of bbox1 and bbox2 that overlap

    Parameters
    ----------
    bbox1:
        The first box.
    bbox2:
        The second box.

    Returns
    -------
    slices: tuple[Sequence[slice], Sequence[slice]]
        The slice of an array bounded by `bbox1` and
        the slice of an array bounded by `bbox` in the
        overlapping region.
    """
    overlap = bbox1 & bbox2
    if np.all(np.array(overlap.shape) == 0):
        # There was no overlap, so return empty slices
        return (slice(0, 0),) * len(overlap.shape), (slice(0, 0),) * len(overlap.shape)
    _bbox1 = overlap - bbox1.origin
    _bbox2 = overlap - bbox2.origin
    slices = (
        _bbox1.slices,
        _bbox2.slices,
    )
    return slices
