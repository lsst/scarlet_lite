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

import operator
from typing import Any, Callable, Sequence, cast

import numpy as np
from numpy.typing import DTypeLike

from .bbox import Box
from .utils import ScalarLike, ScalarTypes

__all__ = ["Image", "MismatchedBoxError", "MismatchedBandsError"]


class MismatchedBandsError(Exception):
    """Attempt to compare images with different bands"""


class MismatchedBoxError(Exception):
    """Attempt to compare images in different bounding boxes"""


def get_dtypes(*data: np.ndarray | Image | ScalarLike) -> list[DTypeLike]:
    """Get a list of dtypes from a list of arrays, images, or scalars

    Parameters
    ----------
    data:
        The arrays to use for calculating the dtype

    Returns
    -------
    result:
        A list of datatypes.
    """
    dtypes: list[DTypeLike] = [None] * len(data)
    for d, element in enumerate(data):
        if hasattr(element, "dtype"):
            dtypes[d] = cast(np.ndarray, element).dtype
        else:
            dtypes[d] = np.dtype(type(element))
    return dtypes


def get_combined_dtype(*data: np.ndarray | Image | ScalarLike) -> DTypeLike:
    """Get the combined dtype for a collection of arrays to prevent loss
    of precision.

    Parameters
    ----------
    data:
        The arrays to use for calculating the dtype

    Returns
    -------
    result: np.dtype
        The resulting dtype.
    """
    dtypes = get_dtypes(*data)
    return max(dtypes)  # type: ignore


class Image:
    """A numpy array with an origin and (optional) bands

    This class contains a 2D numpy array with the addition of an
    origin (``yx0``) and an optional first index (``bands``) that
    allows an immutable named index to be used.

    Notes
    -----
    One of the main limitations of using numpy arrays to store image data
    is the lack of an ``origin`` attribute that allows an array to retain
    knowledge of it's location in a larger scene.
    For example, if a numpy array ``x`` is sliced, eg. ``x[10:20, 30:40]``
    the result will be a new ``10x10`` numpy array that has no meta
    data to inform the user that it was sliced from a larger image.
    In addition, astrophysical images are also multi-band data cubes,
    with a 2D image in each band (in fact this is the simplifying
    assumption that distinguishes scarlet lite from scarlet main).
    However, the ordering of the bands during processing might differ from
    the ordering of the bands to display multiband data.
    So a mechanism was also desired to simplify the sorting and index of
    an image by band name.

    Thus, scarlet lite creates a numpy-array like class with the additional
    ``bands`` and ``yx0`` attributes to keep track of the bands contained
    in an array and the origin of that array (we specify ``yx0`` as opposed
    to ``xy0`` to be consistent with the default numpy/C++ ``(y, x)``
    ordering of arrays as opposed to the traditional cartesian ``(x, y)``
    ordering used in astronomy and other modules in the science pipelines.
    While this may be a small source of confusion for the user,
    it is consistent with the ordering in the original scarlet package and
    ensures the consistency of scarlet lite images and python index slicing.

    Examples
    --------

    The easiest way to create a new image is to use ``Image(numpy_array)``,
    for example

    >>> import numpy as np
    >>> from lsst.scarlet.lite import Image
    >>>
    >>> x = np.arange(12).reshape(3, 4)
    >>> image = Image(x)
    >>> print(image)
    Image:
     [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
      bands=()
      bbox=Box(shape=(3, 4), origin=(0, 0))

    This will create a single band :py:class:`~lsst.scarlet.lite.Image` with
    origin ``(0, 0)``.
    To create a multi-band image the input array must have 3 dimensions and
    the ``bands`` property must be specified:

    >>> x = np.arange(24).reshape(2, 3, 4)
    >>> image = Image(x, bands=("i", "z"))
    >>> print(image)
    Image:
     [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    <BLANKLINE>
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]
      bands=('i', 'z')
      bbox=Box(shape=(3, 4), origin=(0, 0))

    It is also possible to create an empty single-band image using the
    ``from_box`` static method:

    >>> from lsst.scarlet.lite import Box
    >>> image = Image.from_box(Box((3, 4), (100, 120)))
    >>> print(image)
    Image:
     [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
      bands=()
      bbox=Box(shape=(3, 4), origin=(100, 120))

    Similarly, an empty multi-band image can be created by passing a tuple
    of ``bands``:

    >>> image = Image.from_box(Box((3, 4)), bands=("r", "i"))
    >>> print(image)
    Image:
     [[[0. 0. 0. 0.]
      [0. 0. 0. 0.]
      [0. 0. 0. 0.]]
    <BLANKLINE>
     [[0. 0. 0. 0.]
      [0. 0. 0. 0.]
      [0. 0. 0. 0.]]]
      bands=('r', 'i')
      bbox=Box(shape=(3, 4), origin=(0, 0))

    To select a sub-image use a ``Box`` to select a spatial region in either a
    single-band or multi-band image:

    >>> x = np.arange(60).reshape(3, 4, 5)
    >>> image = Image(x, bands=("g", "r", "i"), yx0=(20, 30))
    >>> bbox = Box((2, 2), (21, 32))
    >>> print(image[bbox])
    Image:
     [[[ 7  8]
      [12 13]]
    <BLANKLINE>
     [[27 28]
      [32 33]]
    <BLANKLINE>
     [[47 48]
      [52 53]]]
      bands=('g', 'r', 'i')
      bbox=Box(shape=(2, 2), origin=(21, 32))


    To select a single-band image from a multi-band image,
    pass the name of the band as an index:

    >>> print(image["r"])
    Image:
     [[20 21 22 23 24]
     [25 26 27 28 29]
     [30 31 32 33 34]
     [35 36 37 38 39]]
      bands=()
      bbox=Box(shape=(4, 5), origin=(20, 30))

    Multi-band images can also be sliced in the spatial dimension, for example

    >>> print(image["g":"r"])
    Image:
     [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]
      [10 11 12 13 14]
      [15 16 17 18 19]]
    <BLANKLINE>
     [[20 21 22 23 24]
      [25 26 27 28 29]
      [30 31 32 33 34]
      [35 36 37 38 39]]]
      bands=('g', 'r')
      bbox=Box(shape=(4, 5), origin=(20, 30))

    and

    >>> print(image["r":"r"])
    Image:
     [[[20 21 22 23 24]
      [25 26 27 28 29]
      [30 31 32 33 34]
      [35 36 37 38 39]]]
      bands=('r',)
      bbox=Box(shape=(4, 5), origin=(20, 30))

    both extract a slice of a multi-band image.

    .. warning::
        Unlike numerical indices, where ``slice(x, y)`` will select the
        subset of an array from ``x`` to ``y-1`` (excluding ``y``),
        a spectral slice of an ``Image`` will return the image slice
        including band ``y``.

    It is also possible to change the order or index a subset of bands
    in an image. For example:

    >>> print(image[("r", "g", "i")])
    Image:
     [[[20 21 22 23 24]
      [25 26 27 28 29]
      [30 31 32 33 34]
      [35 36 37 38 39]]
    <BLANKLINE>
     [[ 0  1  2  3  4]
      [ 5  6  7  8  9]
      [10 11 12 13 14]
      [15 16 17 18 19]]
    <BLANKLINE>
     [[40 41 42 43 44]
      [45 46 47 48 49]
      [50 51 52 53 54]
      [55 56 57 58 59]]]
      bands=('r', 'g', 'i')
      bbox=Box(shape=(4, 5), origin=(20, 30))


    will return a new image with the bands re-ordered.

    Images can be combined using the standard arithmetic operations similar to
    numpy arrays, including ``+, -, *, /, **`` etc, however, if two images are
    combined with different bounding boxes, the _union_ of the two
    boxes is used for the result. For example:

    >>> image1 = Image(np.ones((2, 3, 4)), bands=tuple("gr"))
    >>> image2 = Image(np.ones((2, 3, 4)), bands=tuple("gr"), yx0=(2, 3))
    >>> result = image1 + image2
    >>> print(result)
    Image:
     [[[1. 1. 1. 1. 0. 0. 0.]
      [1. 1. 1. 1. 0. 0. 0.]
      [1. 1. 1. 2. 1. 1. 1.]
      [0. 0. 0. 1. 1. 1. 1.]
      [0. 0. 0. 1. 1. 1. 1.]]
    <BLANKLINE>
     [[1. 1. 1. 1. 0. 0. 0.]
      [1. 1. 1. 1. 0. 0. 0.]
      [1. 1. 1. 2. 1. 1. 1.]
      [0. 0. 0. 1. 1. 1. 1.]
      [0. 0. 0. 1. 1. 1. 1.]]]
      bands=('g', 'r')
      bbox=Box(shape=(5, 7), origin=(0, 0))

    If instead you want to additively ``insert`` image 1 into image 2,
    so that they have the same bounding box as image 2, use

    >>> _ = image2.insert(image1)
    >>> print(image2)
    Image:
     [[[2. 1. 1. 1.]
      [1. 1. 1. 1.]
      [1. 1. 1. 1.]]
    <BLANKLINE>
     [[2. 1. 1. 1.]
      [1. 1. 1. 1.]
      [1. 1. 1. 1.]]]
      bands=('g', 'r')
      bbox=Box(shape=(3, 4), origin=(2, 3))

    To insert an image using a different operation use

    >>> from operator import truediv
    >>> _ = image2.insert(image1, truediv)
    >>> print(image2)
    Image:
     [[[2. 1. 1. 1.]
      [1. 1. 1. 1.]
      [1. 1. 1. 1.]]
    <BLANKLINE>
     [[2. 1. 1. 1.]
      [1. 1. 1. 1.]
      [1. 1. 1. 1.]]]
      bands=('g', 'r')
      bbox=Box(shape=(3, 4), origin=(2, 3))


    However, depending on the operation you may get unexpected results
    since now there could be ``NaN`` and ``inf`` values due to the zeros
    in the non-overlapping regions.
    Instead, to select only the overlap region one can use

    >>> result = image1 / image2
    >>> print(result[image1.bbox & image2.bbox])
    Image:
     [[[0.5]]
    <BLANKLINE>
     [[0.5]]]
      bands=('g', 'r')
      bbox=Box(shape=(1, 1), origin=(2, 3))


    Parameters
    ----------
    data:
        The array data for the image.
    bands:
        The bands coving the image.
    yx0:
        The (y, x) offset for the lower left of the image.
    """

    def __init__(
        self,
        data: np.ndarray,
        bands: Sequence | None = None,
        yx0: tuple[int, int] | None = None,
    ):
        if bands is None or len(bands) == 0:
            # Using an empty tuple for the bands will result in a 2D image
            bands = ()
            assert data.ndim == 2
        else:
            bands = tuple(bands)
            assert data.ndim == 3
            if data.shape[0] != len(bands):
                raise ValueError(f"Array has spectral size {data.shape[0]}, but {bands} bands")
        if yx0 is None:
            yx0 = (0, 0)
        self._data = data
        self._yx0 = yx0
        self._bands = bands

    @staticmethod
    def from_box(bbox: Box, bands: tuple | None = None, dtype: DTypeLike = float) -> Image:
        """Initialize an empty image from a bounding Box and optional bands

        Parameters
        ----------
        bbox:
            The bounding box that contains the image.
        bands:
            The bands for the image.
            If bands is `None` then a 2D image is created.
        dtype:
            The numpy dtype of the image.

        Returns
        -------
        image:
            An empty image contained in ``bbox`` with ``bands`` bands.
        """
        if bands is not None and len(bands) > 0:
            shape = (len(bands),) + bbox.shape
        else:
            shape = bbox.shape
        data = np.zeros(shape, dtype=dtype)
        return Image(data, bands=bands, yx0=cast(tuple[int, int], bbox.origin))

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the image.

        This includes the spectral dimension, if there is one.
        """
        return self._data.shape

    @property
    def dtype(self) -> DTypeLike:
        """The numpy dtype of the image."""
        return self._data.dtype

    @property
    def bands(self) -> tuple:
        """The bands used in the image."""
        return self._bands

    @property
    def n_bands(self) -> int:
        """Number of bands in the image.

        If `n_bands == 0` then the image is 2D and does not have a spectral
        dimension.
        """
        return len(self._bands)

    @property
    def is_multiband(self) -> bool:
        """Whether or not the image has a spectral dimension."""
        return self.n_bands > 0

    @property
    def height(self) -> int:
        """Height of the image."""
        return self.shape[-2]

    @property
    def width(self) -> int:
        """Width of the image."""
        return self.shape[-1]

    @property
    def yx0(self) -> tuple[int, int]:
        """Origin of the image, in numpy/C++ y,x ordering."""
        return self._yx0

    @property
    def y0(self) -> int:
        """location of the y-offset."""
        return self._yx0[0]

    @property
    def x0(self) -> int:
        """Location of the x-offset."""
        return self._yx0[1]

    @property
    def bbox(self) -> Box:
        """Bounding box for the special dimensions in the image."""
        return Box(self.shape[-2:], self._yx0)

    @property
    def data(self) -> np.ndarray:
        """The image viewed as a numpy array."""
        return self._data

    @property
    def ndim(self) -> int:
        """Number of dimensions in the image."""
        return self._data.ndim

    def spectral_indices(self, bands: Sequence | slice) -> tuple[int, ...] | slice:
        """The indices to extract each band in `bands` in order from the image

        This converts a band name, or list of band names,
        into numerical indices that can be used to slice the internal numpy
        `data` array.

        Parameters
        ---------
        bands:
            If `bands` is a list of band names, then the result will be an
            index corresponding to each band, in order.
            If `bands` is a slice, then the ``start`` and ``stop`` properties
            should be band names, and the result will be a slice with the
            appropriate indices to start at `bands.start` and end at
            `bands.stop`.

        Returns
        -------
        band_indices:
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

        band_indices = tuple(self.bands.index(band) for band in bands if band in self.bands)
        return band_indices

    def matched_spectral_indices(
        self,
        other: Image,
    ) -> tuple[tuple[int, ...] | slice, tuple[int, ...] | slice]:
        """Match bands between two images

        Parameters
        ----------
        other:
            The other image to match spectral indices to.

        Returns
        -------
        result:
            A tuple with a tuple of indices/slices for each dimension,
            including the spectral dimension.
        """
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

        self_indices = cast(tuple[int, ...], self.spectral_indices(other.bands))
        matched_bands = tuple(self.bands[bidx] for bidx in self_indices)
        other_indices = cast(tuple[int, ...], other.spectral_indices(matched_bands))
        return other_indices, self_indices

    def matched_slices(self, bbox: Box) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        """Get the slices to match this image to a given bounding box

        Parameters
        ----------
        bbox:
            The bounding box to match this image to.

        Returns
        -------
        result:
            Tuple of indices/slices to match this image to the given bbox.
        """
        if self.bbox == bbox:
            # No need to slice, since the boxes match
            _slice = (slice(None),) * bbox.ndim
            return _slice, _slice

        slices = self.bbox.overlapped_slices(bbox)
        return slices

    def project(
        self,
        bands: object | tuple[object] | None = None,
        bbox: Box | None = None,
    ) -> Image:
        """Project this image into a different set of bands

         Parameters
         ----------
         bands:
            Spectral bands to project this image into.
            Not all bands have to be contained in the image, and not all
            bands contained in the image have to be used in the projection.
         bbox:
            A bounding box to project the image into.

        Results
        -------
        image:
            A new image creating by projecting this image into
            `bbox` and `bands`.
        """
        if bands is None:
            bands = self.bands
        if not isinstance(bands, tuple):
            bands = (bands,)
        if self.is_multiband:
            indices = self.spectral_indices(bands)
            data = self.data[indices, :]
        else:
            data = self.data

        if bbox is None:
            return Image(data, bands=bands, yx0=self.yx0)

        if self.is_multiband:
            image = np.zeros((len(bands),) + bbox.shape, dtype=data.dtype)
            slices = bbox.overlapped_slices(self.bbox)
            # Insert a slice for the spectral dimension
            image[(slice(None),) + slices[0]] = data[(slice(None),) + slices[1]]
            return Image(image, bands=bands, yx0=cast(tuple[int, int], bbox.origin))

        image = np.zeros(bbox.shape, dtype=data.dtype)
        slices = bbox.overlapped_slices(self.bbox)
        image[slices[0]] = data[slices[1]]
        return Image(image, bands=bands, yx0=cast(tuple[int, int], bbox.origin))

    @property
    def multiband_slices(self) -> tuple[tuple[int, ...] | slice, slice, slice]:
        """Return the slices required to slice a multiband image"""
        return (self.spectral_indices(self.bands),) + self.bbox.slices  # type: ignore

    def insert_into(
        self,
        image: Image,
        op: Callable = operator.add,
    ) -> Image:
        """Insert this image into another image in place.

        Parameters
        ----------
        image:
            The image to insert this image into.
        op:
            The operator to use when combining the images.

        Returns
        -------
        result:
            `image` updated by inserting this instance.
        """
        return insert_image(image, self, op)

    def insert(self, image: Image, op: Callable = operator.add) -> Image:
        """Insert another image into this image in place.

        Parameters
        ----------
        image:
            The image to insert this image into.
        op:
            The operator to use when combining the images.

        Returns
        -------
        result:
            This instance with `image` inserted.
        """
        return insert_image(self, image, op)

    def repeat(self, bands: tuple) -> Image:
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

    def copy(self, order=None) -> Image:
        """Make a copy of this image.

        Parameters
        ----------
        order:
            The ordering to use for storing the bytes.
            This is unlikely to be needed, and just defaults to
            the numpy behavior (C) ordering.

        Returns
        -------
        image: Image
            The copy of this image.
        """
        return self.copy_with(order=order)

    def copy_with(
        self,
        data: np.ndarray | None = None,
        order: str | None = None,
        bands: tuple[str, ...] | None = None,
        yx0: tuple[int, int] | None = None,
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
            data = self.data.copy(order)  # type: ignore
        if bands is None:
            bands = self.bands
        if yx0 is None:
            yx0 = self.yx0
        return Image(data, bands, yx0)

    def trimmed(self, threshold: float = 0) -> Image:
        """Return a copy of the image trimmed to a threshold.

        This is essentially the smallest image that contains all of the
        pixels above the threshold.

        Parameters
        ----------
        threshold:
            The threshold to use for trimming the image.

        Returns
        -------
        image:
            A copy of the image trimmed to the threshold.
        """
        data = self.data.copy()
        bbox = Box.from_data(data, threshold=threshold)
        data = data[bbox.slices]
        y0, x0 = bbox.origin

        return Image(data, yx0=(y0 + self.y0, x0 + self.x0))

    def at(self, y: int, x: int) -> ScalarLike | np.ndarray:
        """The value of the image at a given location.

        Image does not implment single index access because the
        result is a scalar, while indexing an image returns another image.

        Parameters
        ----------
        y:
            The y-coordinate of the location.
        x:
            The x-coordinate of the location.

        Returns
        -------
        value:
            The value of the image at the given location.
        """
        _y = y - self.y0
        _x = x - self.x0
        if self.ndim == 2:
            return self.data[_y, _x]
        return self.data[:, _y, _x]

    def _i_update(self, op: Callable, other: Image | ScalarLike) -> Image:
        """Update the data array in place.

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
            if hasattr(other, "dtype"):
                _dtype = cast(np.ndarray, other).dtype
            else:
                _dtype = type(other)
            msg = f"Cannot update an array with type {self.dtype} with {_dtype}"
            raise ValueError(msg)
        result = op(other)
        self._data[:] = result.data
        self._bands = result.bands
        self._yx0 = result.yx0
        return self

    def _check_equality(self, other: Image | ScalarLike, op: Callable) -> Image:
        """Compare this array to another.

        This performs an element by element equality check.

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

        Raises
        ------
        TypeError:
            If `other` is not an `Image`.
        MismatchedBandsError:
            If `other` has different bands.
        MismatchedBoxError:
            if `other` exists in a different bounding box.
        """
        if isinstance(other, Image) and other.bands == self.bands and other.bbox == self.bbox:
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

    def __eq__(self, other: object) -> Image:  # type: ignore
        """Check if this image is equal to another."""
        if not isinstance(other, Image) and not isinstance(other, ScalarTypes):
            raise TypeError(f"Cannot compare an Image to {type(other)}.")
        return self._check_equality(other, operator.eq)  # type: ignore

    def __ne__(self, other: object) -> Image:  # type: ignore
        """Check if this image is not equal to another."""
        return ~self.__eq__(other)

    def __ge__(self, other: Image | ScalarLike) -> Image:
        """Check if this image is greater than or equal to another."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=self.data >= other)
        return self._check_equality(other, operator.ge)

    def __le__(self, other: Image | ScalarLike) -> Image:
        """Check if this image is less than or equal to another."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=self.data <= other)
        return self._check_equality(other, operator.le)

    def __gt__(self, other: Image | ScalarLike) -> Image:
        """Check if this image is greater than or equal to another."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=self.data > other)
        return self._check_equality(other, operator.ge)

    def __lt__(self, other: Image | ScalarLike) -> Image:
        """Check if this image is less than or equal to another."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=self.data < other)
        return self._check_equality(other, operator.le)

    def __neg__(self):
        """Take the negative of the image."""
        return self.copy_with(data=-self._data)

    def __pos__(self):
        """Make a copy using of the image."""
        return self.copy()

    def __invert__(self):
        """Take the inverse (~) of the image."""
        return self.copy_with(data=~self._data)

    def __add__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using addition."""
        return _operate_on_images(self, other, operator.add)

    def __iadd__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using addition and update
        in place.
        """
        return self._i_update(self.__add__, other)

    def __radd__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using addition,
        with this image on the right.
        """
        if type(other) in ScalarTypes:
            return self.copy_with(data=other + self.data)
        return cast(Image, other).__add__(self)

    def __sub__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using subtraction."""
        return _operate_on_images(self, other, operator.sub)

    def __isub__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using subtraction,
        with this image on the right.
        """
        return self._i_update(self.__sub__, other)

    def __rsub__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using subtraction,
        with this image on the right.
        """
        if type(other) in ScalarTypes:
            return self.copy_with(data=other - self.data)
        return cast(Image, other).__sub__(self)

    def __mul__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using multiplication."""
        return _operate_on_images(self, other, operator.mul)

    def __imul__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using multiplication,
        with this image on the right.
        """
        return self._i_update(self.__mul__, other)

    def __rmul__(self, other: Image | ScalarLike) -> Image:
        """Combine this image and another image using multiplication,
        with this image on the right.
        """
        if type(other) in ScalarTypes:
            return self.copy_with(data=other * self.data)
        return cast(Image, other).__mul__(self)

    def __truediv__(self, other: Image | ScalarLike) -> Image:
        """Divide this image by `other`."""
        return _operate_on_images(self, other, operator.truediv)

    def __itruediv__(self, other: Image | ScalarLike) -> Image:
        """Divide this image by `other` in place."""
        return self._i_update(self.__truediv__, other)

    def __rtruediv__(self, other: Image | ScalarLike) -> Image:
        """Divide this image by `other` with this on the right."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=other / self.data)
        return cast(Image, other).__truediv__(self)

    def __floordiv__(self, other: Image | ScalarLike) -> Image:
        """Floor divide this image by `other` in place."""
        return _operate_on_images(self, other, operator.floordiv)

    def __ifloordiv__(self, other: Image | ScalarLike) -> Image:
        """Floor divide this image by `other` in place."""
        return self._i_update(self.__floordiv__, other)

    def __rfloordiv__(self, other: Image | ScalarLike) -> Image:
        """Floor divide this image by `other` with this on the right."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=other // self.data)
        return cast(Image, other).__floordiv__(self)

    def __pow__(self, other: Image | ScalarLike) -> Image:
        """Raise this image to the `other` power."""
        return _operate_on_images(self, other, operator.pow)

    def __ipow__(self, other: Image | ScalarLike) -> Image:
        """Raise this image to the `other` power in place."""
        return self._i_update(self.__pow__, other)

    def __rpow__(self, other: Image | ScalarLike) -> Image:
        """Raise this other to the power of this image."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=other**self.data)
        return cast(Image, other).__pow__(self)

    def __mod__(self, other: Image | ScalarLike) -> Image:
        """Take the modulus of this % other."""
        return _operate_on_images(self, other, operator.mod)

    def __imod__(self, other: Image | ScalarLike) -> Image:
        """Take the modulus of this % other in place."""
        return self._i_update(self.__mod__, other)

    def __rmod__(self, other: Image | ScalarLike) -> Image:
        """Take the modulus of other % this."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=other % self.data)
        return cast(Image, other).__mod__(self)

    def __and__(self, other: Image | ScalarLike) -> Image:
        """Take the bitwise and of this and other."""
        return _operate_on_images(self, other, operator.and_)

    def __iand__(self, other: Image | ScalarLike) -> Image:
        """Take the bitwise and of this and other in place."""
        return self._i_update(self.__and__, other)

    def __rand__(self, other: Image | ScalarLike) -> Image:
        """Take the bitwise and of other and this."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=other & self.data)
        return cast(Image, other).__and__(self)

    def __or__(self, other: Image | ScalarLike) -> Image:
        """Take the binary or of this or other."""
        return _operate_on_images(self, other, operator.or_)

    def __ior__(self, other: Image | ScalarLike) -> Image:
        """Take the binary or of this or other in place."""
        return self._i_update(self.__or__, other)

    def __ror__(self, other: Image | ScalarLike) -> Image:
        """Take the binary or of other or this."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=other | self.data)
        return cast(Image, other).__or__(self)

    def __xor__(self, other: Image | ScalarLike) -> Image:
        """Take the binary xor of this xor other."""
        return _operate_on_images(self, other, operator.xor)

    def __ixor__(self, other: Image | ScalarLike) -> Image:
        """Take the binary xor of this xor other in place."""
        return self._i_update(self.__xor__, other)

    def __rxor__(self, other: Image | ScalarLike) -> Image:
        """Take the binary xor of other xor this."""
        if type(other) in ScalarTypes:
            return self.copy_with(data=other ^ self.data)
        return cast(Image, other).__xor__(self)

    def __lshift__(self, other: ScalarLike) -> Image:
        """Shift this image to the left by other bits."""
        if not issubclass(np.dtype(type(other)).type, np.integer):
            raise TypeError("Bit shifting an image can only be done with integers")
        return self.copy_with(data=self.data << other)

    def __ilshift__(self, other: ScalarLike) -> Image:
        """Shift this image to the left by other bits in place."""
        self[:] = self.__lshift__(other)
        return self

    def __rlshift__(self, other: ScalarLike) -> Image:
        """Shift other to the left by this image bits."""
        return self.copy_with(data=other << self.data)

    def __rshift__(self, other: ScalarLike) -> Image:
        """Shift this image to the right by other bits."""
        if not issubclass(np.dtype(type(other)).type, np.integer):
            raise TypeError("Bit shifting an image can only be done with integers")
        return self.copy_with(data=self.data >> other)

    def __irshift__(self, other: ScalarLike) -> Image:
        """Shift this image to the right by other bits in place."""
        self[:] = self.__rshift__(other)
        return self

    def __rrshift__(self, other: ScalarLike) -> Image:
        """Shift other to the right by this image bits."""
        return self.copy_with(data=other >> self.data)

    def __str__(self):
        """Display the image array, bands, and bounding box."""
        return f"Image:\n {str(self.data)}\n  bands={self.bands}\n  bbox={self.bbox}"

    def _is_spectral_index(self, index: Any) -> bool:
        """Check to see if an index is a spectral index.

        Parameters
        ----------
        index:
            Either a slice, a tuple, or an element in `Image.bands`.

        Returns
        -------
        result:
            ``True`` if `index` is band or tuple of bands.
        """
        bands = self.bands
        if isinstance(index, slice):
            if index.start in bands or index.stop in bands or (index.start is None and index.stop is None):
                return True
            return False
        if index in self.bands:
            return True
        if isinstance(index, tuple) and index[0] in self.bands:
            return True
        return False

    def _get_box_slices(self, bbox: Box) -> tuple[slice, slice]:
        """Get the slices of the image to insert it into the overlapping
        region with `bbox`."""
        overlap = self.bbox & bbox
        if overlap != bbox:
            raise IndexError("Bounding box is outside of the image")
        origin = bbox.origin
        shape = bbox.shape
        y_start = origin[0] - self.yx0[0]
        y_stop = origin[0] + shape[0] - self.yx0[0]
        x_start = origin[1] - self.yx0[1]
        x_stop = origin[1] + shape[1] - self.yx0[1]
        y_index = slice(y_start, y_stop)
        x_index = slice(x_start, x_stop)
        return y_index, x_index

    def _get_sliced(self, indices: Any, value: Image | None = None) -> Image:
        """Select a subset of an image

        Parameters
        ----------
        indices:
            The indices to select a subsection of the image.
            The spectral index can either be a tuple of indices,
            a slice of indices, or a single index used to select a
            single-band 2D image.
            The spatial index (if present) is a `Box`.

        value:
            The value used to set this slice of the image.
            This allows the single `_get_sliced` method to be used for
            both getting a slice of an image and setting it.

        Returns
        -------
        result: Image | np.ndarray
            The resulting image obtained by selecting subsets of the iamge
            based on the `indices`.
        """
        if not isinstance(indices, tuple):
            indices = (indices,)

        if self.is_multiband:
            if self._is_spectral_index(indices[0]):
                if len(indices) > 1 and indices[1] in self.bands:
                    # The indices are all band names,
                    # so use them all as a spectral indices
                    bands = indices
                    spectral_index = self.spectral_indices(bands)
                    y_index = x_index = slice(None)
                elif self._is_spectral_index(indices[0]):
                    # The first index is a spectral index
                    spectral_index = self.spectral_indices(indices[0])
                    if isinstance(spectral_index, slice):
                        bands = self.bands[spectral_index]
                    elif len(spectral_index) == 1:
                        bands = ()
                        spectral_index = spectral_index[0]  # type: ignore
                    else:
                        bands = tuple(self.bands[idx] for idx in spectral_index)
                    indices = indices[1:]
                    if len(indices) == 1:
                        # The spatial index must be a bounding box
                        if not isinstance(indices[0], Box):
                            raise IndexError(f"Expected a Box for the spatial index but got {indices[1]}")
                        y_index, x_index = self._get_box_slices(indices[0])
                    elif len(indices) == 0:
                        y_index = x_index = slice(None)
                    else:
                        raise IndexError(f"Too many spatial indices, expeected a Box bot got {indices}")
                full_index = (spectral_index, y_index, x_index)
            elif isinstance(indices[0], Box):
                bands = self.bands
                y_index, x_index = self._get_box_slices(indices[0])
                full_index = (slice(None), y_index, x_index)
            else:
                error = f"3D images can only be indexed by spectral indices or bounding boxes, got {indices}"
                raise IndexError(error)
        else:
            if len(indices) != 1 or not isinstance(indices[0], Box):
                raise IndexError(f"2D images can only be sliced by bounding box, got {indices}")
            bands = ()
            y_index, x_index = self._get_box_slices(indices[0])
            full_index = (y_index, x_index)  # type: ignore

        y0 = y_index.start
        if y0 is None:
            y0 = 0

        x0 = x_index.start
        if x0 is None:
            x0 = 0

        if value is None:
            # This is a getter,
            # so return an image with the data sliced properly
            yx0 = (y0 + self.yx0[0], x0 + self.yx0[1])

            data = self.data[full_index]

            if data.ndim == 2:
                # Only a single band was selected, so return that band
                return Image(data, yx0=yx0)
            return Image(data, bands=bands, yx0=yx0)

        # Set the data
        self._data[full_index] = value.data
        return self

    def overlapped_slices(self, bbox: Box) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
        """Get the slices needed to insert this image into a bounding box.

        Parameters
        ----------
        bbox:
            The region to insert this image into.

        Returns
        -------
        overlap:
            The slice of this image and the slice of the `bbox` required to
            insert the overlapping portion of this image.

        """
        overlap = self.bbox.overlapped_slices(bbox)
        if self.is_multiband:
            overlap = (slice(None),) + overlap[0], (slice(None),) + overlap[1]
        return overlap

    def __getitem__(self, indices: Any) -> Image:
        """Get the subset of an image

        Parameters
        ----------
        indices:
            The indices to select a subsection of the image.

        Returns
        -------
        result:
            The resulting image obtained by selecting subsets of the iamge
            based on the `indices`.
        """
        return self._get_sliced(indices)

    def __setitem__(self, indices, value: Image) -> Image:
        """Set a subset of an image to a given value

        Parameters
        ----------
        indices:
            The indices to select a subsection of the image.
        value:
            The value to use for the subset of the image.

        Returns
        -------
        result:
            The resulting image obtained by selecting subsets of the image
            based on the `indices`.
        """
        return self._get_sliced(indices, value)


def _operate_on_images(image1: Image, image2: Image | ScalarLike, op: Callable) -> Image:
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
    image2 = cast(Image, image2)
    if image1.bands == image2.bands and image1.bbox == image2.bbox:
        # The images perfectly overlap, so just combine their results
        with np.errstate(divide="ignore", invalid="ignore"):
            result = op(image1.data, image2.data)
        return Image(result, bands=image1.bands, yx0=image1.yx0)

    if op != operator.add and op != operator.sub and image1.bands != image2.bands:
        msg = "Images with different bands can only be combined using addition and subtraction, "
        msg += f"got {op}, with bands {image1.bands}, {image2.bands}"
        raise ValueError(msg)

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

    if op == operator.add or op == operator.sub:
        dtype = get_combined_dtype(image1, image2)
        result = Image(np.zeros(shape, dtype=dtype), bands=bands, yx0=cast(tuple[int, int], bbox.origin))
        # Add the first image in place
        image1.insert_into(result, operator.add)
        # Use the operator to insert the second image
        image2.insert_into(result, op)
    else:
        # Project both images into the full bbox
        image1 = image1.project(bbox=bbox)
        image2 = image2.project(bbox=bbox)
        result = op(image1, image2)
    return result


def insert_image(
    main_image: Image,
    sub_image: Image,
    op: Callable = operator.add,
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

    Returns
    -------
    main_image: Image
        The `main_image`, with the `sub_image` inserted in place.
    """
    if len(main_image.bands) == 0 and len(sub_image.bands) == 0:
        slices = sub_image.matched_slices(main_image.bbox)
        image_slices = slices[1]
        self_slices = slices[0]
    else:
        band_indices = sub_image.matched_spectral_indices(main_image)
        slices = sub_image.matched_slices(main_image.bbox)
        image_slices = (band_indices[0],) + slices[1]  # type: ignore
        self_slices = (band_indices[1],) + slices[0]  # type: ignore

    main_image._data[image_slices] = op(main_image.data[image_slices], sub_image.data[self_slices])
    return main_image
