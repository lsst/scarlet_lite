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
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
from numpy.typing import DTypeLike, ArrayLike

from .bbox import Box
from .utils import ScalarLike, ScalarTypes

__all__ = ["Image", "MismatchedBoxError", "MismatchedBandsError"]

TImage = TypeVar("TImage", bound="Image")


class MismatchedBandsError(Exception):
    """Attempt to compare images with different bands"""

    pass


class MismatchedBoxError(Exception):
    """Attempt to compare images in different bounding boxes"""

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


class Image:
    """A numpy array with an origin and (optional) bands"""

    def __init__(
        self,
        data: np.ndarray,
        bands: Sequence | None = None,
        yx0: tuple[int, int] = None,
        indices: dict[tuple, tuple[tuple, tuple]] = None,
        slices: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[tuple[slice, ...], tuple[slice, ...]],
        ] = None,
    ):
        """Create a new image

        Parameters
        ----------
        data:
            The array data for the image.
        bands:
            The bands coving the image.
        yx0:
            The (y, x) offset for the lower left of the image.
        indices:
            Dictionary of cached indices with the bands to project this image
            into as the key, and the slices for the projected image and this
            image as the values.
        slices:
            Dictionary of cached slices to insert this image into another
            image.
        """
        if bands is None or len(bands) == 0:
            # Using an empty tuple for the bands will result in a 2D image
            bands = ()
            assert len(data.shape) == 2
        else:
            bands = tuple(bands)
            assert len(data.shape) == 3
            if data.shape[0] != len(bands):
                raise ValueError(
                    f"Array has spectral size {data.shape[0]}, but {bands} bands"
                )
        if yx0 is None:
            yx0 = (0, 0)
        if indices is None:
            indices = {}
        if slices is None:
            slices = {}
        self._data = data
        self._yx0 = yx0
        self._bands = bands
        self._indices = indices
        self._slices = slices

    @staticmethod
    def from_box(bbox: Box, bands: tuple = None, dtype: DTypeLike = float):
        if bands is not None and len(bands) > 0:
            shape = (len(bands),) + bbox.shape
        else:
            shape = bbox.shape
        data = np.zeros(shape, dtype=dtype)
        return Image(data, bands=bands, yx0=bbox.origin)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def bands(self):
        return self._bands

    @property
    def n_bands(self) -> int:
        """Number of bands in the image

        If `n_bands == 0` then the image is 2D and does not have a spectral
        dimension.
        """
        return len(self._bands)

    @property
    def is_multiband(self):
        return self.n_bands > 0

    @property
    def height(self) -> int:
        """Height of the image"""
        return self.shape[-2]

    @property
    def width(self) -> int:
        """Width of the image"""
        return self.shape[-1]

    @property
    def yx0(self) -> tuple[int, int]:
        return self._yx0

    @property
    def y0(self) -> int:
        """location of the y-offset"""
        return self._yx0[0]

    @property
    def x0(self) -> int:
        """Location of the x-offset"""
        return self._yx0[1]

    @property
    def bbox(self) -> Box:
        """Bounding box for the special dimensions in the image"""
        return Box(self.shape[-2:], self._yx0)

    @property
    def data(self) -> np.ndarray:
        """The image viewed as a numpy array"""
        return self._data

    @property
    def indices(self) -> dict[tuple, tuple[tuple, tuple]]:
        """Dictionary of cached indices

        The bands to project this image into are the keys,
        and the slices for the projected image and this image are the values.
        """
        return self._indices

    @property
    def slices(
        self,
    ) -> dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[slice, ...], tuple[slice, ...]],
    ]:
        """Dictionary of cached slices to insert this image into another
        image.
        """
        return self._slices

    def spectral_indices(self, bands: Sequence | slice) -> tuple[int, ...] | slice:
        """The indices to extract each band in `bands` in order from the image

        Paramters
        ---------
        bands:
            If `bands` is a list of band names, then the result will be an
            index corresponding to each band, in order.
            If `bands` is a slice, then the ``start`` and ``stop`` properties
            should be band names, and the result will be a slice with the
            appropriate indices to start at `bands.start` and end at
            `bands.end`.

        Returns
        -------
        band_indices: tuple[int, ...]
            Tuple of indices for each band in this image.
        """
        if isinstance(bands, slice):
            # Convert a slice of band names into a slice of array indices
            # to select the appropriate slice.
            if bands.start is None:
                start = None
            else:
                start = self.bands.index(bands.start)
            if bands.stop is None:
                stop = None
            else:
                stop = self.bands.index(bands.stop) + 1
            return slice(start, stop, bands.step)

        if isinstance(bands, str):
            return (self.bands.index(bands),)

        band_indices = tuple(
            self.bands.index(band) for band in bands if band in self.bands
        )
        return band_indices

    def matched_spectral_indices(
        self, other: TImage, save: bool = False
    ) -> tuple[tuple[int, ...] | slice, tuple[int, ...] | slice]:
        """Match bands between two images

        Parameters
        ----------
        other:
            The other image to match spectral indices to.
        save:
            Whether or not to save the mapping between the two sets of bands
            in this image. Saving will cached the result to slice this image
            more quickly.

        Returns
        -------
        result: tuple[tuple[int, ...] | slice, tuple[int, ...] | slice]
            A tuple with a tuple of indices/slices for each dimension,
            including the spectral dimension.
        """
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
            err = "Attempted to insert a monochromatic image into a mutli-band image"
            raise ValueError(err)
        if other.n_bands == 0:
            err = "Attempted to insert a multi-band image into a monochromatic image"
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
        """Get the slices to match this image to a given bounding box

        Parameters
        ----------
        bbox:
            The bounding box to match this image to.
        save:
            Whether or not to sae this spatial mapping to sae compute
            time later.

        Returns
        -------
        result: tuple[tuple[slice, ...], tuple[slice, ...]]
            Tuple of indices/slices to match this image to the given bbox.
        """
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
        self,
        bands: object | tuple[object] = None,
        bbox: Box = None,
    ) -> TImage:
        """Project this image into a differnt set of bands

         Parameters
         ----------
         bands:
            Spctral bands to project this image into.
            Not all bands have to be contained in the image, and not all
            bands contained in the image have to be used in the projection.
         bbox:
            A bounding box to project the image into.

        Results
        -------
        image: Image
            The resulting image.
        """
        if bands is None:
            bands = self.bands
        if self.is_multiband:
            indices = self.spectral_indices(bands)
            data = self.data[
                indices,
            ]
        else:
            data = self.data

        if bbox is None:
            return Image(data, bands=bands, yx0=self.yx0)

        if self.is_multiband:
            image = np.zeros((len(bands),) + bbox.shape, dtype=data.dtype)
            slices = bbox.overlapped_slices(self.bbox)
            # Insert a slice for the spectral dimension
            image[(slice(None),) + slices[0]] = data[(slice(None),) + slices[1]]
            return Image(image, bands=bands, yx0=bbox.origin)

        image = np.zeros(bbox.shape, dtype=data.dtype)
        slices = bbox.overlapped_slices(self.bbox)
        image[slices[0]] = data[slices[1]]
        return Image(image, bands=bands, yx0=bbox.origin)

    @property
    def multiband_slices(self) -> tuple[tuple | slice, slice, slice]:
        """Return the

        :return:
        """
        return (self.spectral_indices(self.bands),) + self.bbox.slices

    def insert_into(
        self,
        image: TImage,
        op: Callable = operator.add,
        save: bool = False,
    ):
        return insert_image(image, self, op, save)

    def insert(self, image: TImage, op: Callable = operator.add, save: bool = False):
        return insert_image(self, image, op, save)

    def repeat(self, bands: tuple) -> TImage:
        """Project a 2D image into the spectral dimension

        Parameters
        ----------
        bands:
            The bands in the projected image.

        Returns
        -------
        result: Image
            The 2D image repeated in each band in the spectral dimension.
        """
        if self.is_multiband:
            raise ValueError("Image.repeat only works with 2D images")
        return self.copy_with(
            np.repeat(self.data[None, :, :], len(bands), axis=0),
            bands=bands,
            yx0=self.yx0,
        )

    def copy(self, order=None) -> TImage:
        """Make a copy of this image

        Parameters
        ----------
        order:
            The ordering to use for storing the bytes.
            This is unlikely to be needed, and just defaults to
            the numpy behavior (C) ordering.

        Returns
        -------
        iamge: Image
            The copy of this image.
        """
        return self.copy_with(order=order)

    def copy_with(
        self,
        data: np.ndarray = None,
        order: str = None,
        bands: tuple[str, ...] = None,
        yx0: tuple[int, int] = None,
    ):
        """Copy of this image with some parameters updated.

        Any parameters not specified by the user will be copied from the
        current image.

        Parameters
        ----------
        data:
            An update for the data in the image.
        order:
            The ordering for stored bytes, from numpy.copy.
        bands:
            The bands that the resulting image will have.
            The number of bands must be the same as the first dimension
            in the data array.
        yx0:
            The lower-left of the image bounding box.

        Returns
        -------
        image: Image
            The copied image.
        """
        if order is None:
            order = "C"
        if data is None:
            data = self.data.copy(order)
        if bands is None:
            bands = self.bands
        if yx0 is None:
            yx0 = self.yx0
        return Image(data, bands, yx0)

    def _i_update(self, op: Callable, other: TImage) -> TImage:
        """Update the data array in place

        This is typically implemented by `__i<op>__` methods,
        like `__iadd__`, to apply an operator and update this image
        with the data in place.

        Parameters
        ----------
        op:
            Operator used to combine this image with the `other` image.
        other:
            The other image that is combined with this one using the operator
            `op`.

        Returns
        -------
        image: Image
            This image, after being updated by the operator
        """
        dtype = get_combined_dtype(self.data, other)
        if self.dtype != dtype:
            msg = f"Cannot update an array with type {self.dtype} with {other.dtype}"
            raise ValueError(msg)
        result = op(other)
        self._data[:] = result.data
        self._bands = result.bands
        self._yx0 = result.yx0
        return self

    def _check_equality(self, other: TImage, op: Callable) -> TImage:
        """Compare this array to another

        This performs an element by element equality check and will raise
        a `TypeError` if `other` is not an `Image`, a `MismatchedBandsError`,
        if the other image has different bands, and a `MismatchedBoxError` if
        the other image exists in a different bounding box.

        Parameters
        ----------
        other:
            The image to compare this image to.
        op:
            The operator used for the comparision (==, !=, >=, <=).

        Returns
        -------
        image: Image
            An image made by checking all of the elements in this array with
            another.
        """
        if (
            isinstance(other, Image)
            and other.bands == self.bands
            and other.bbox == self.bbox
        ):
            return self.copy_with(data=op(self.data, other.data))

        if not isinstance(other, Image):
            if type(other) in ScalarTypes:
                return self.copy_with(data=op(self.data, other))
            raise TypeError(f"Cannot compare images to {type(other)}")

        if other.bands != self.bands:
            msg = f"Cannot compare images with mismatched bands: {self.bands} vs {other.bands}"
            raise MismatchedBandsError(msg)

        raise MismatchedBoxError(
            f"Cannot compare images with different bounds boxes: {self.bbox} vs. {other.bbox}"
        )

    def __eq__(self, other: TImage | ScalarLike) -> TImage:
        """Check if this image is equal to another"""
        return self._check_equality(other, operator.eq)

    def __ne__(self, other: TImage | ScalarLike) -> TImage:
        """Check if this image is not equal to another"""
        return ~self.__eq__(other)

    def __ge__(self, other: TImage | ScalarLike) -> TImage:
        """Check if this image is greater than or equal to another"""
        if type(other) in ScalarTypes:
            return self.copy_with(data=self.data >= other)
        return self._check_equality(other, operator.ge)

    def __le__(self, other: TImage | ScalarLike) -> TImage:
        """Check if this image is less than or equal to another"""
        if type(other) in ScalarTypes:
            return self.copy_with(data=self.data <= other)
        return self._check_equality(other, operator.le)

    def __gt__(self, other: TImage | ScalarLike) -> TImage:
        """Check if this image is greater than or equal to another"""
        if type(other) in ScalarTypes:
            return self.copy_with(data=self.data > other)
        return self._check_equality(other, operator.ge)

    def __lt__(self, other: TImage | ScalarLike) -> TImage:
        """Check if this image is less than or equal to another"""
        if type(other) in ScalarTypes:
            return self.copy_with(data=self.data < other)
        return self._check_equality(other, operator.le)

    def __neg__(self):
        """Take the negative of the image"""
        return self.copy_with(data=-self._data)

    def __pos__(self):
        """Make a copy using of the image"""
        return self.copy()

    def __invert__(self):
        """Take the inverse (~) of the image"""
        return self.copy_with(data=~self._data)

    def __add__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using addition"""
        return _operate_on_images(self, other, operator.add)

    def __iadd__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using addition and update
        in place
        """
        return self._i_update(self.__add__, other)

    def __radd__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using addition,
        with this image on the right
        """
        if type(other) in ScalarTypes:
            return self.copy_with(data=other + self.data)
        return other.__add__(self)

    def __sub__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using subtraction"""
        return _operate_on_images(self, other, operator.sub)

    def __isub__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using subtraction,
        with this image on the right
        """
        return self._i_update(self.__sub__, other)

    def __rsub__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using subtraction,
        with this image on the right
        """
        if type(other) in ScalarTypes:
            return self.copy_with(data=other - self.data)
        return other.__sub__(self)

    def __mul__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using multiplication"""
        return _operate_on_images(self, other, operator.mul)

    def __imul__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using multiplication,
        with this image on the right
        """
        return self._i_update(self.__mul__, other)

    def __rmul__(self, other: TImage | ScalarLike) -> TImage:
        """Combine this image and another image using multiplication,
        with this image on the right
        """
        if type(other) in ScalarTypes:
            return self.copy_with(data=other * self.data)
        return other.__mul__(self)

    def __truediv__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.truediv)

    def __itruediv__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__truediv__, other)

    def __rtruediv__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other / self.data)
        return other.__truediv__(self)

    def __floordiv__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.floordiv)

    def __ifloordiv__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__floordiv__, other)

    def __rfloordiv__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other // self.data)
        return other.__floordiv__(self)

    def __pow__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.pow)

    def __ipow__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__pow__, other)

    def __rpow__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other**self.data)
        return other.__pow__(self)

    def __mod__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.mod)

    def __imod__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__mod__, other)

    def __rmod__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other % self.data)
        return other.__mod__(self)

    def __and__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.and_)

    def __iand__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__and__, other)

    def __rand__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other & self.data)
        return other.__and__(self)

    def __or__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.or_)

    def __ior__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__or__, other)

    def __ror__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other | self.data)
        return other.__or__(self)

    def __xor__(self, other: TImage | ScalarLike) -> TImage:
        return _operate_on_images(self, other, operator.xor)

    def __ixor__(self, other: TImage | ScalarLike) -> TImage:
        return self._i_update(self.__xor__, other)

    def __rxor__(self, other: TImage | ScalarLike) -> TImage:
        if type(other) in ScalarTypes:
            return self.copy_with(data=other ^ self.data)
        return other.__xor__(self)

    def __lshift__(self, other: ScalarLike) -> TImage:
        if not issubclass(np.dtype(type(other)).type, np.integer):
            raise TypeError("Bit shifting an image can only be done with integers")
        return self.copy_with(data=self.data << other)

    def __ilshift__(self, other: ScalarLike) -> TImage:
        self[:] = self.__lshift__(other)
        return self

    def __rlshift__(self, other: ScalarLike) -> TImage:
        return self.copy_with(data=other << self.data)

    def __rshift__(self, other: ScalarLike) -> TImage:
        if not issubclass(np.dtype(type(other)).type, np.integer):
            raise TypeError("Bit shifting an image can only be done with integers")
        return self.copy_with(data=self.data >> other)

    def __irshift__(self, other: ScalarLike) -> TImage:
        self[:] = self.__rshift__(other)
        return self

    def __rrshift__(self, other: ScalarLike) -> TImage:
        return self.copy_with(data=other >> self.data)

    def __matmul__(self, other: TImage | ScalarLike) -> TImage:
        raise TypeError("Images do not support matrix mutliplication")

    def __str__(self):
        return f"Image:\n {str(self.data)}\n  bands={self.bands}\n  bbox={self.bbox}"

    def _is_spectral_index(self, index: Any):
        """Check to see if an index is a spectral index.

        Parameters
        ----------
        index:
            Either a slice, a tuple, or an element in `Image.bands`.
        """
        bands = self.bands
        if isinstance(index, slice):
            if (
                index.start in bands
                or index.stop in bands
                or (index.start is None and index.stop is None)
            ):
                return True
            return False
        if index in self.bands:
            return True
        if isinstance(index, tuple) and index[0] in self.bands:
            return True
        return False

    def _get_box_slices(self, index):
        overlap = self.bbox & index
        if overlap != index:
            raise IndexError("Bounding box is outside of the image")
        origin = index.origin
        shape = index.shape
        y_start = origin[0] - self.yx0[0]
        y_stop = origin[0] + shape[0] - self.yx0[0]
        x_start = origin[1] - self.yx0[1]
        x_stop = origin[1] + shape[1] - self.yx0[1]
        y_index = slice(y_start, y_stop)
        x_index = slice(x_start, x_stop)
        return y_index, x_index

    def _get_spatial_slices(self, indices):
        if isinstance(indices[0], Box):
            y_index, x_index = self._get_box_slices(indices[0])
        else:
            y_index = indices[0]
            if len(indices) > 1:
                if len(indices) > 2:
                    raise IndexError(f"Unable to parse indices {indices}")
                x_index = indices[1]
            else:
                x_index = slice(None)
        return y_index, x_index

    def _get_sliced(self, indices, value: TImage = None) -> TImage:
        """Select a subset of an image

        Parameters
        ----------
        indices:
            The indices to select a subsection of the image.
            The spectral index can either be a tuple of indices,
            a slice of indices, or a single index used to select a
            single-band 2D image.
            The y and x indices must be slices or None, since the image
            must be continuous and defined by a spatial bounding box.

        Returns
        -------
        result: Image | np.ndarray
            The resulting image obtained by selecting subsets of the iamge
            based on the `indices`.
        """
        if not isinstance(indices, tuple):
            indices = (indices,)

        if self.is_multiband:
            if len(indices) > 1 and indices[1] in self.bands:
                # The indices are all band names,
                # so use them all as a spectral
                spectral_index = self.spectral_indices(indices)
                y_index = x_index = slice(None)
            else:
                if self._is_spectral_index(indices[0]):
                    spectral_index = self.spectral_indices(indices[0])
                    indices = indices[1:]
                else:
                    spectral_index = slice(None)

                if len(indices) > 0:
                    y_index, x_index = self._get_spatial_slices(indices)
                else:
                    y_index = x_index = slice(None)

            if isinstance(spectral_index, slice):
                bands = self.bands[spectral_index]
                full_index = (spectral_index, y_index, x_index)
            elif len(spectral_index) == 1:
                bands = ()
                full_index = (spectral_index[0], y_index, x_index)
            else:
                bands = tuple(self.bands[idx] for idx in spectral_index)
                full_index = (spectral_index, y_index, x_index)
        else:
            y_index, x_index = self._get_spatial_slices(indices)
            bands = None
            full_index = (y_index, x_index)

        if not isinstance(y_index, slice):
            raise IndexError(
                f"Images can only be sliced over the spatial dimensions, got {y_index} for the y-dimension"
            )
        else:
            y0 = y_index.start
            if y0 is None:
                y0 = 0
        if not isinstance(x_index, slice):
            raise IndexError(
                f"Images can only be sliced over the spatial dimensions, got {x_index} for the x-dimension"
            )
        else:
            x0 = x_index.start
            if x0 is None:
                x0 = 0

        if value is None:
            # This is a getter,
            # so return an image with the data sliced properly
            yx0 = (y0 + self.yx0[0], x0 + self.yx0[1])

            data = self.data[full_index]

            if len(data.shape) == 2:
                # Only a single band was selected, so return that band
                return Image(data, yx0=yx0)
            return Image(data, bands=bands, yx0=yx0)

        # Set the data
        self._data[full_index] = value.data
        return self

    def overlapped_slices(
        self, bbox: Box
    ) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        overlap = self.bbox.overlapped_slices(bbox)
        if self.is_multiband:
            overlap = (slice(None),) + overlap[0], (slice(None),) + overlap[1]
        return overlap

    def __getitem__(self, indices) -> TImage:
        """Get the subset of an image

        Parameters
        ----------
        indices:
            The indices to select a subsection of the image.

        Returns
        -------
        result: Image | np.ndarray
            The resulting image obtained by selecting subsets of the iamge
            based on the `indices`.
        """
        return self._get_sliced(indices)

    def __setitem__(self, indices, value: ArrayLike) -> TImage:
        """Set a subset of an image to a given value

        Parameters
        ----------
        indices:
            The indices to select a subsection of the image.
        value:
            The value to use for the subset of the image.

        Returns
        -------
        result: Image | np.ndarray
            The resulting image obtained by selecting subsets of the iamge
            based on the `indices`.
        """
        return self._get_sliced(indices, value)


def _operate_on_images(
    image1: Image, image2: Image | ScalarLike, op: Callable
) -> Image:
    """Perform an operation on two images, that may or may not be spectrally
    and spatially aligned.

    Parameters
    ----------
    image1:
        The image on the LHS of the operation
    image2:
        The image on the RHS of the operation
    op:
        The operation used to combine the images.

    Returns
    -------
    image:
        The resulting combined image.
    """
    if type(image2) in ScalarTypes:
        return image1.copy_with(data=op(image1.data, image2))
    if image1.bands == image2.bands and image1.bbox == image2.bbox:
        # The images perfectly overlap, so just combine their results
        return Image(op(image1.data, image2.data), bands=image1.bands, yx0=image1.yx0)

    # Use all of the bands in the first image
    bands = image1.bands
    # Add on any bands from the second image not contained in the first image
    bands = bands + tuple(band for band in image2.bands if band not in bands)
    # Create a box that contains both images
    bbox = image1.bbox | image2.bbox
    # Create an image that will contain both images
    if len(bands) > 0:
        shape = (len(bands),) + bbox.shape
    else:
        shape = bbox.shape
    dtype = get_combined_dtype(image1, image2)
    result = Image(np.zeros(shape, dtype=dtype), bands=bands, yx0=bbox.origin)
    # Add the first image in place
    image1.insert_into(result, operator.add)
    # Use the operator to insert the second image
    image2.insert_into(result, op)
    return result


def insert_image(
    main_image: Image,
    sub_image: Image,
    op: Callable = operator.add,
    save: bool = False,
) -> Image:
    """Insert one image into another image

    Parameters
    ----------
    main_image:
        The image that will have `sub_image` insertd.
    sub_image:
        The image that is inserted into `main_image`.
    op:
        The operator to use for insertion
        (addition, subtraction, multiplication, etc.).
    save:
        Whether or not to save the speectral and spatial indices/slices
        in order to match everything to

    Returns
    -------
    image: Image
        The image, with the other image inserted in place.
    """
    if len(main_image.bands) == 0 and len(sub_image.bands) == 0:
        slices = sub_image.matched_slices(main_image.bbox, save)
        image_slices = slices[1]
        self_slices = slices[0]
    else:
        band_indices = sub_image.matched_spectral_indices(main_image, save)
        slices = sub_image.matched_slices(main_image.bbox, save)
        image_slices = (band_indices[0],) + slices[1]
        self_slices = (band_indices[1],) + slices[0]

    main_image._data[image_slices] = op(
        main_image.data[image_slices], sub_image.data[self_slices]
    )
    return main_image
