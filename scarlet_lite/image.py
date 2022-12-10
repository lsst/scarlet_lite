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

import operator
from typing import Callable, Sequence, TypeVar

import numpy as np
from numpy.typing import DTypeLike

from .bbox import Box
from .utils import ScalarLike, ScalarTypes

__all__ = ["Image", "MismatchedBoxError", "MismatchedBandsError"]

TImage = TypeVar("TImage", bound="Image")


class MismatchedBandsError(Exception):
    pass


class MismatchedBoxError(Exception):
    pass


def get_combined_dtype(*arrays: np.ndarray) -> DTypeLike:
    """Get the combined dtype for a collection of arrays to prevent loss
    of precesion.

    Parameters
    ----------
    arrays:
        The arrays to use for calculating the dtype

    Returns
    -------
    result: DTypeLike
        The resulting dtype.
    """
    dtype = arrays[0].dtype
    for array in arrays[1:]:
        if array.dtype > dtype:
            dtype = array.dtype
    return dtype


class Image(np.ndarray):
    """A numpy array with an origin and (optional) bands"""

    def __new__(
        cls,
        array: np.ndarray,
        bands: Sequence | None = None,
        yx0: tuple[int, int] = None,
        indices: dict[tuple, tuple[tuple, tuple]] = None,
        slices: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[slice, ...], tuple[slice, ...]],
        ] = None,
    ) -> TImage:
        if bands is None:
            bands = ()
            assert len(array.shape) == 2
        else:
            bands = tuple(bands)
            assert len(array.shape) == 3
            if array.shape[0] != len(bands):
                raise ValueError(
                    f"Array has spectral size {array.shape[0]}, but {bands} bands"
                )
        if yx0 is None:
            yx0 = (0, 0)
        if indices is None:
            indices = {}
        if slices is None:
            slices = {}
        obj = np.asarray(array, dtype=array.dtype).view(cls)
        obj.yx0 = yx0
        obj.bands = bands
        obj.indices = indices
        obj.slices = slices
        return obj

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def height(self) -> int:
        return self.shape[-2]

    @property
    def width(self) -> int:
        return self.shape[-1]

    @property
    def y0(self) -> int:
        return self.yx0[0]

    @property
    def x0(self) -> int:
        return self.yx0[1]

    @property
    def bbox(self) -> Box:
        return Box(self.shape[-2:], self.yx0)

    @property
    def array(self):
        return self.view(np.ndarray)

    def spectral_indices(self, bands: Sequence) -> tuple[int]:
        band_indices = tuple(
            self.bands.index(band) for band in bands if band in self.bands
        )
        return band_indices

    def matched_spectral_indices(
        self, other: TImage, save: bool = False
    ) -> tuple[tuple[int, ...] | slice, tuple[int, ...] | slice]:
        if other.bands in self.indices:
            # The indices have already been calculated
            return self.indices[other.bands]
        if self.bands == other.bands and self.n_bands != 0:
            # The bands match
            return slice(None), slice(None)
        if self.n_bands == 0 and other.n_bands == 0:
            # The images are 2D, so no spectral slicing is necessary
            return (), ()
        if self.n_bands == 0 and other.n_bands > 1:
            err = f"Attempted to insert a monochromatic image into a mutli-band image"
            raise ValueError(err)
        if other.n_bands == 0:
            err = f"Attempted to insert a multi-band image into a monochromatic image"
            raise ValueError(err)
        self_indices = self.spectral_indices(other.bands)
        matched_bands = tuple(self.bands[bidx] for bidx in self_indices)
        other_indices = other.spectral_indices(matched_bands)
        if save:
            self.indices[other.bands] = (other_indices, self_indices)
        return other_indices, self_indices

    def matched_slices(
        self, bbox: Box, save: bool = False
    ) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        key = (bbox.shape, bbox.origin)
        if key in self.slices:
            return self.slices[key]
        if self.bbox == bbox:
            # No need to slice, since the boxes match
            _slice = (slice(None),) * bbox.dimensions
            return _slice, _slice

        slices = self.bbox.overlapped_slices(bbox)
        if save:
            self.slices[key] = slices
        return slices

    def project(
        self, bands: object | tuple[object] = None, yx0: tuple[int, int] = None
    ) -> TImage:
        if bands is None:
            bands = self.bands
        if yx0 is None:
            yx0 = self.yx0
        indices = self.spectral_indices(bands)
        return Image(self.view(np.ndarray)[indices], bands=bands, yx0=yx0)

    def insert_into(
        self,
        image: TImage,
        op: Callable = operator.add,
        save: bool = False,
    ):
        band_indices = self.matched_spectral_indices(image)
        slices = self.matched_slices(image.bbox, save)

        image_slices = band_indices[0] + slices[0]
        self_slices = band_indices[1] + slices[1]

        image[image_slices] = op(image[image_slices, self[self_slices]])
        return image

    def copy(self, order=None) -> TImage:
        return self.copy_with(order=order)

    def copy_with(self, data=None, order=None, bands=None, yx0=None):
        if order is None:
            order = "C"
        if data is None:
            data = self.array.copy(order)
        if bands is None:
            bands = self.bands
        if yx0 is None:
            yx0 = self.yx0
        return Image(data, bands, yx0)

    def _i_update(self, op: Callable, other: TImage) -> TImage:
        dtype = get_combined_dtype(self, other)
        if self.dtype != dtype:
            msg = f"Cannot update an array with type {self.dtype} with {other.dtype}"
            raise ValueError(msg)
        result = op(other)
        self[:] = result
        self.bands = result.bands
        self.yx0 = result.yx0
        return self

    def _check_equality(self, other: TImage, op: Callable) -> TImage:
        if (
            isinstance(other, Image)
            and other.bands == self.bands
            and other.bbox == self.bbox
        ):
            return self.copy_with(data=op(self.array, other.array))

        if not isinstance(other, Image):
            if type(other) in ScalarTypes:
                return self.copy_with(data=op(self.array, other))
            raise TypeError(f"Cannot compare images to {type(other)}")

        if other.bands != self.bands:
            msg = f"Cannot compare images with mismatched bands: {self.bands} vs {other.bands}"
            raise MismatchedBandsError(msg)

        raise MismatchedBoxError(
            f"Cannot compare images with different bounds boxes: {self.bbox} vs. {other.bbox}"
        )

    def __eq__(self, other: TImage | ScalarLike) -> TImage:
        return self._check_equality(other, operator.eq)

    def __ne__(self, other: TImage | ScalarLike) -> TImage:
        return ~self.__eq__(other)

    def __ge__(self, other: TImage | ScalarLike) -> TImage:
        return self._check_equality(other, operator.ge)

    def __le__(self, other: TImage | ScalarLike) -> TImage:
        return self._check_equality(other, operator.le)

    def __add__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.add)

    def __iadd__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__add__, other)

    def __radd__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other+self.array)
        return other.__add__(self)

    def __sub__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.sub)

    def __isub__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__sub__, other)

    def __rsub__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other-self.array)
        return other.__sub__(self)

    def __mul__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.mul)

    def __imul__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__mul__, other)

    def __rmul__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other*self.array)
        return other.__mul__(self)

    def __truediv__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.truediv)

    def __itruediv__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__truediv__, other)

    def __rtruediv__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other / self.array)
        return other.__truediv__(self)

    def __floordiv__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.floordiv)

    def __ifloordiv__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__floordiv__, other)

    def __rfloordiv__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other // self.array)
        return other.__floordiv__(self)

    def __pow__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.pow)

    def __ipow__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__pow__, other)

    def __rpow__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other**self.array)
        return other.__pow__(self)

    def __mod__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.mod)

    def __imod__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__mod__, other)

    def __rmod__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other % self.array)
        return other.__mod__(self)

    def __and__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.and_)

    def __iand__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__and__, other)

    def __rand__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other & self.array)
        return other.__and__(self)

    def __or__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.or_)

    def __ior__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__or__, other)

    def __ror__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other | self.array)
        return other.__or__(self)

    def __xor__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.xor)

    def __ixor__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__xor__, other)

    def __rxor__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other ^ self.array)
        return other.__xor__(self)

    def __lshift__(self, other: int | ScalarLike) -> TImage:
        if not issubclass(np.dtype(type(other)).type, np.integer):
            raise TypeError("Bit shifting an image can only be done with integers")
        return self.copy_with(data=self.array << other)

    def __ilshift__(self, other: int | ScalarLike) -> TImage:
        self[:] = self.__lshift__(other)
        return self

    def __rshift__(self, other: int | ScalarLike) -> TImage:
        if not issubclass(np.dtype(type(other)).type, np.integer):
            raise TypeError("Bit shifting an image can only be done with integers")
        return self.copy_with(data=self.array >> other)

    def __irshift__(self, other: int | ScalarLike) -> TImage:
        self[:] = self.__rshift__(other)
        return self

    def __matmul__(self, other: TImage | ScalarLike) -> TImage:
        raise TypeError("Images do not support matrix mutliplication")

    def __str__(self):
        return "Image:\n" + super().__str__() + "\n"

    def __array_finalize__(self, obj):
        self.yx0 = getattr(obj, "yx0", (0, 0))
        self.bands = getattr(obj, "bands", ())
        self.indices = getattr(obj, "indices", {})
        self.slices = getattr(obj, "slices", {})


def _operate_on_images(image1: Image | ScalarLike, image2: Image | ScalarLike, op: Callable) -> Image:
    if type(image2) in ScalarTypes:
        return image1.copy_with(data=op(image1.array, image2))
    if type(image1) in ScalarTypes:
        return image2.copy_with(data=op(image1, image2.array))
    if image1.bands == image2.bands and image1.bbox == image2.bbox:
        # The images perfectly overlap, so just combine their results
        return Image(op(image1.array, image2.array), bands=image1.bands, yx0=image1.yx0)

    # Use all of the bands in the first image
    bands = image1.bands
    # Add on any bands from the second image not contained in the first image
    bands = bands + tuple(band for band in image2.bands if band not in bands)
    # Create a box that contains both images
    bbox = image1.bbox | image2.bbox
    # Create an image that will contain both images
    shape = (len(bands),) + bbox.shape
    dtype = get_combined_dtype(image1, image2)
    result = Image(np.zeros(shape, dtype=dtype), bands=bands, yx0=bbox.origin)
    # Add the first image in place
    image1.insert_into(result, operator.add)
    # Use the operator to insert the second image
    image2.insert_into(result, op)
    return result
