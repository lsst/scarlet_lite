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
    global/model coordinate system.
    It is used to identify spatial and channel overlap and to map from model
    to observed frames and back.

    The `BBox` code is agnostic about the meaning of the dimensions.
    We generally use this convention:

    - 2D shapes denote (Height, Width)
    - 3D shapes denote (Channels, Height, Width)

    Parameters
    ----------
    shape: tuple[int]
        Size of the box
    origin: tuple[int]
        Minimum corner coordinate of the box
    """

    def __init__(self, shape: Sequence[int], origin: Sequence[int] = None):
        """
        Parameters
        ----------
        shape: Sequence[int]
            Size of the box
        origin: Sequence[int]
            Minimum corner coordinate of the box
        """
        self.shape = tuple(shape)
        if origin is None:
            origin = (0,) * len(shape)
        assert len(origin) == len(shape)
        self.origin = tuple(origin)

    @staticmethod
    def from_bounds(*bounds: Sequence[int]) -> TBox:
        """Initialize a box from its bounds

        Parameters
        ----------
        bounds: Sequence[Sequence[int, int]]
            Min/Max coordinate for every dimension

        Returns
        -------
        bbox: Box
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
        x: np.ndarray
            Data to threshold
        min_value: float
            Minimum value of the result.

        Returns
        -------
        bbox: Box
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

    def extract_from(self, image: np.ndarray, sub: np.ndarray = None) -> np.ndarray:
        """Extract sub-image described by this bbox from image

        Parameters
        ----------
        image: np.ndarray
            Full image
        sub: np.ndarray
            Extracted image

        Returns
        -------
        sub: np.ndarray
        """
        imbox = Box(image.shape)

        if sub is None:
            sub = np.zeros(self.shape, dtype=image.dtype)

        im_slices, sub_slices = overlapped_slices(imbox, self)
        sub[sub_slices] = image[im_slices]
        return sub

    def insert_into(self, image: np.ndarray, sub: np.ndarray) -> np.ndarray:
        """Insert `sub` into `image` according to this bbox

        Inverse operation to :func:`~scarlet.bbox.Box.extract_from`.

        Parameters
        ----------
        image: np.ndarray
            Full image
        sub: np.ndarray
            Extracted sub-image

        Returns
        -------
        image: `~numpy.array`
        """
        imbox = Box(image.shape)

        im_slices, sub_slices = overlapped_slices(imbox, self)
        image[im_slices] = sub[sub_slices]
        return image

    @property
    def dimensions(self) -> int:
        """Dimensionality of this BBox"""
        return len(self.shape)

    @property
    def start(self) -> Sequence[int]:
        """Tuple of start coordinates"""
        return self.origin

    @property
    def stop(self) -> tuple[int]:
        """Tuple of stop coordinates"""
        return tuple(o + s for o, s in zip(self.origin, self.shape))

    @property
    def center(self) -> tuple[float]:
        """Tuple of center coordinates"""
        return tuple(o + s / 2 for o, s in zip(self.origin, self.shape))

    @property
    def bounds(self) -> tuple[tuple[int, int]]:
        """Bounds of the box"""
        return tuple((o, o + s) for o, s in zip(self.origin, self.shape))

    @property
    def slices(self) -> tuple[slice]:
        """Bounds of the box as slices"""
        return tuple([slice(o, o + s) for o, s in zip(self.origin, self.shape)])

    def grow(self, radius: int) -> TBox:
        """Grow the Box by the given radius in each direction"""
        if not hasattr(radius, "__iter__"):
            radius = [radius] * self.dimensions
        origin = tuple([self.origin[d] - radius[d] for d in range(self.dimensions)])
        shape = tuple([self.shape[d] + 2 * radius[d] for d in range(self.dimensions)])
        return Box(shape, origin=origin)

    def shift(self, shift: tuple[int]):
        """Shift this box in-place

        Parameters
        ----------
        shift: tuple[int]
            A tuple the same shape as `origin` to shift this box
            along each axis.
        """
        self.origin = tuple(o + shift[i] for i, o in enumerate(self.origin))

    def shifted_by(self, shift: tuple[int]) -> TBox:
        """Generate a shifted copy of this box

        Parameters
        ----------
        shift: tuple[int]
            The amount to shift each axis to create the new box

        Returns
        -------
        result: `Box`
            The resulting bounding box.
        """
        origin = tuple(o + shift[i] for i, o in enumerate(self.origin))
        return Box(self.shape, origin=origin)

    def intersects(self, other: TBox) -> bool:
        """Check if two boxes overlap

        Parameters
        ----------
        other: Box
            The boxes to check for overlap

        Returns
        -------
        result: bool
            True when the two boxes overlap.
        """
        overlap = self & other
        return np.all(np.array(overlap.shape) != 0)

    def overlapped_slices(self, other: TBox) -> tuple[tuple[slice], tuple[slice]]:
        """`slice` for the box that contains the overlap of this and
        another `Box`

        Parameters
        ----------
        other: Box

        Returns
        -------
        slices: tuple[tuple[slice], tuple[slice]]
            The slice of an array bounded by `self` and
            the slice of an array bounded by `other` in the
            overlapping region.
        """
        return overlapped_slices(self, other)

    def __or__(self, other: TBox) -> TBox:
        """Union of two bounding boxes

        Parameters
        ----------
        other: Box
            The other bounding box in the union

        Returns
        -------
        result: Box
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
        other: Box
            The other bounding box in the intersection

        Returns
        -------
        result: Box
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

    def __getitem__(self, i: int | slice) -> TBox:
        s_ = self.shape[i]
        o_ = self.origin[i]
        if not hasattr(s_, "__iter__"):
            s_ = (s_,)
            o_ = (o_,)
        return Box(s_, origin=o_)

    def __repr__(self) -> str:
        result = "<Box shape={0}, origin={1}>"
        return result.format(self.shape, self.origin)

    def __iadd__(self, offset: int | Sequence[int]) -> TBox:
        if not hasattr(offset, "__iter__"):
            offset = (offset,) * self.dimensions
        self.origin = tuple([a + o for a, o in zip(self.origin, offset)])
        return self

    def __add__(self, offset: int | Sequence[int]) -> TBox:
        return self.copy().__iadd__(offset)

    def __isub__(self, offset: int | Sequence[int]) -> TBox:
        if not hasattr(offset, "__iter__"):
            offset = (offset,) * self.dimensions
        self.origin = tuple([a - o for a, o in zip(self.origin, offset)])
        return self

    def __sub__(self, offset: int | Sequence[int]) -> TBox:
        return self.copy().__isub__(offset)

    def __imatmul__(self, bbox: TBox) -> TBox:
        bounds = self.bounds + bbox.bounds
        result = Box.from_bounds(*bounds)
        return result

    def __matmul__(self, bbox: TBox) -> TBox:
        return self.copy().__imatmul__(bbox)

    def __copy__(self) -> TBox:
        return Box(self.shape, origin=self.origin)

    def copy(self) -> TBox:
        """Copy of the box"""
        return self.__copy__()

    def __eq__(self, other: TBox) -> bool:
        return self.shape == other.shape and self.origin == other.origin

    def __hash__(self) -> int:
        return hash((self.shape, self.origin))


def get_minimal_boxsize(size: int, min_size: int = 21, increment: int = 10) -> int:
    """Calculate the smallest box that will contain a source with `size`

    Parameters
    ----------
    size: int
        The size of the source.
    min_size: int
        The minimum size of a box.
    increment: int
        The step size for the box size.
    """
    boxsize = min_size
    while boxsize < size:
        boxsize += increment  # keep box sizes quite small
    return boxsize


def overlapped_slices(bbox1: Box, bbox2: Box) -> tuple[tuple[slice], tuple[slice]]:
    """Slices of bbox1 and bbox2 that overlap

    Parameters
    ----------
    bbox1: Box
    bbox2: Box

    Returns
    -------
    slices: tuple[tuple[slice], tuple[slice]]
        The slice of an array bounded by `bbox1` and
        the slice of an array bounded by `bbox` in the
        overlapping region.
    """
    overlap = bbox1 & bbox2
    _bbox1 = overlap - bbox1.origin
    _bbox2 = overlap - bbox2.origin
    slices = (
        _bbox1.slices,
        _bbox2.slices,
    )
    return slices
