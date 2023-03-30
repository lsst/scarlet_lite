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

__all__ = ["Fourier"]

import operator
from typing import Callable, Sequence

import numpy as np
from scipy import fftpack


def centered(arr: np.ndarray, newshape: Sequence[int]) -> np.ndarray:
    """Return the central newshape portion of the array.

    Parameters
    ----------
    arr:
        The array to center.
    newshape:
        The new shape of the array.

    Notes
    -----
    If the array shape is odd and the target is even,
    the center of `arr` is shifted to the center-right
    pixel position.
    This is slightly different than the scipy implementation,
    which uses the center-left pixel for the array center.
    The reason for the difference is that we have
    adopted the convention of `np.fft.fftshift` in order
    to make sure that changing back and forth from
    fft standard order (0 frequency and position is
    in the bottom left) to 0 position in the center.
    """
    _newshape = np.array(newshape)
    currshape = np.array(arr.shape)

    if not np.all(_newshape <= currshape):
        msg = f"arr must be larger than newshape in both dimensions, received {arr.shape}, and {_newshape}"
        raise ValueError(msg)

    startind = (currshape - _newshape + 1) // 2
    endind = startind + _newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]


def fast_zero_pad(arr: np.ndarray, pad_width: Sequence[Sequence[int]]) -> np.ndarray:
    """Fast version of numpy.pad when `mode="constant"`

    Executing `numpy.pad` with zeros is ~1000 times slower
    because it doesn't make use of the `zeros` method for padding.

    Parameters
    ---------
    arr:
        The array to pad
    pad_width:
        Number of values padded to the edges of each axis.
        See numpy.pad docs for more.

    Returns
    -------
    result: np.ndarray
        The array padded with `constant_values`
    """
    newshape = tuple([a + ps[0] + ps[1] for a, ps in zip(arr.shape, pad_width)])

    result = np.zeros(newshape, dtype=arr.dtype)
    slices = tuple([slice(start, s - end) for s, (start, end) in zip(result.shape, pad_width)])
    result[slices] = arr
    return result


def _pad(
    arr: np.ndarray,
    newshape: Sequence[int],
    axes: int | Sequence[int] | None = None,
    mode: str = "constant",
    constant_values: float = 0,
) -> np.ndarray:
    """Pad an array to fit into newshape

    Pad `arr` with zeros to fit into newshape,
    which uses the `np.fft.fftshift` convention of moving
    the center pixel of `arr` (if `arr.shape` is odd) to
    the center-right pixel in an even shaped `newshape`.

    Parameters
    ----------
    arr:
        The arrray to pad.
    newshape:
        The new shape of the array.
    axes:
        The axes that are being reshaped.
    mode:
        The numpy mode used to pad the array.
        In other words, how to fill the new padded elements.
        See ``numpy.pad`` for details.
    constant_values:
        If `mode` == "constant" then this is the value to set all of
        the new padded elements to.
    """
    _newshape = np.asarray(newshape)
    if axes is None:
        currshape = np.array(arr.shape)
        diff = _newshape - currshape
        startind = (diff + 1) // 2
        endind = diff - startind
        pad_width = list(zip(startind, endind))
    else:
        # only pad the axes that will be transformed
        pad_width = [(0, 0) for _ in arr.shape]
        if isinstance(axes, int):
            axes = [axes]
        for a, axis in enumerate(axes):
            diff = _newshape[a] - arr.shape[axis]
            startind = (diff + 1) // 2
            endind = diff - startind
            pad_width[axis] = (startind, endind)
    if mode == "constant" and constant_values == 0:
        result = fast_zero_pad(arr, pad_width)
    else:
        result = np.pad(arr, tuple(pad_width), mode=mode)  # type: ignore
    return result


def get_fft_shape(
    im_or_shape1: np.ndarray | Sequence[int],
    im_or_shape2: np.ndarray | Sequence[int],
    padding: int = 3,
    axes: int | Sequence[int] | None = None,
    use_max: bool = False,
) -> tuple:
    """Return the fast fft shapes for each spatial axis

    Calculate the fast fft shape for each dimension in
    axes.

    Parameters
    ----------
    im_or_shape1:
        The left image or shape of an image.
    im_or_shape2:
        The right image or shape of an image.
    padding:
        Any additional padding to add to the final shape.
    axes:
        The axes that are being transformed.
    use_max:
        Whether or not to use the maximum of the two shapes,
        or the sum of the two shapes.

    Returns
    -------
    shape:
        Tuple of the shape to use when the two images are transformed
        into k-space.
    """
    if isinstance(im_or_shape1, np.ndarray):
        shape1 = np.asarray(im_or_shape1.shape)
    else:
        shape1 = np.asarray(im_or_shape1)
    if isinstance(im_or_shape2, np.ndarray):
        shape2 = np.asarray(im_or_shape2.shape)
    else:
        shape2 = np.asarray(im_or_shape2)
    # Make sure the shapes are the same size
    if len(shape1) != len(shape2):
        msg = (
            "img1 and img2 must have the same number of dimensions, "
            f"but got {len(shape1)} and {len(shape2)}"
        )
        raise ValueError(msg)
    # Set the combined shape based on the total dimensions
    if axes is None:
        if use_max:
            shape = np.max([shape1, shape2], axis=0)
        else:
            shape = shape1 + shape2
    else:
        if isinstance(axes, int):
            axes = [axes]
        shape = np.zeros(len(axes), dtype="int")
        for n, ax in enumerate(axes):
            shape[n] = shape1[ax] + shape2[ax]
            if use_max:
                shape[n] = np.max([shape1[ax], shape2[ax]])

    shape += padding
    # Use the next fastest shape in each dimension
    shape = [fftpack.next_fast_len(s) for s in shape]
    return tuple(shape)


class Fourier:
    """An array that stores its Fourier Transform

    The `Fourier` class is used for images that will make
    use of their Fourier Transform multiple times.
    In order to prevent numerical artifacts the same image
    convolved with different images might require different
    padding, so the FFT for each different shape is stored
    in a dictionary.

    Parameters
    ----------
    image: np.ndarray
        The real space image.
    image_fft: dict[Sequence[int], np.ndarray]
        A dictionary of {shape: fft_value} for which each different
        shape has a precalculated FFT.
    """

    def __init__(
        self,
        image: np.ndarray,
        image_fft: dict[Sequence[Sequence[int]], np.ndarray] | None = None,
    ):
        if image_fft is None:
            self._fft: dict[Sequence[Sequence[int]], np.ndarray] = {}
        else:
            self._fft = image_fft
        self._image = image

    @staticmethod
    def from_fft(
        image_fft: np.ndarray,
        fft_shape: Sequence[int],
        image_shape: Sequence[int],
        axes: int | Sequence[int] | None = None,
    ) -> Fourier:
        """Generate a new Fourier object from an FFT dictionary

        If the fft of an image has been generated but not its
        real space image (for example when creating a convolution kernel),
        this method can be called to create a new `Fourier` instance
        from the k-space representation.

        Parameters
        ----------
        image_fft:
            The FFT of the image.
        fft_shape:
            "Fast" shape of the image used to generate the FFT.
            This will be different than `image_fft.shape` if
            any of the dimensions are odd, since `np.fft.rfft`
            requires an even number of dimensions (for symmetry),
            so this tells `np.fft.irfft` how to go from
            complex k-space to real space.
        image_shape:
            The shape of the image *before padding*.
            This will regenerate the image with the extra
            padding stripped.
        axes:
            The dimension(s) of the array that will be transformed.

        Returns
        -------
        result:
            A `Fourier` object generated from the FFT.
        """
        if axes is None:
            axes = range(len(image_shape))
        if isinstance(axes, int):
            axes = [axes]
        all_axes = range(len(image_shape))
        image = np.fft.irfftn(image_fft, fft_shape, axes=axes)
        # Shift the center of the image from the bottom left to the center
        image = np.fft.fftshift(image, axes=axes)
        # Trim the image to remove the padding added
        # to reduce fft artifacts
        image = centered(image, image_shape)
        key = (tuple(fft_shape), tuple(axes), tuple(all_axes))

        return Fourier(image, {key: image_fft})

    @property
    def image(self) -> np.ndarray:
        """The real space image"""
        return self._image

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the real space image"""
        return self._image.shape

    def fft(self, fft_shape: Sequence[int], axes: int | Sequence[int]) -> np.ndarray:
        """The FFT of an image for a given `fft_shape` along desired `axes`

        Parameters
        ----------
        fft_shape:
            "Fast" shape of the image used to generate the FFT.
            This will be different than `image_fft.shape` if
            any of the dimensions are odd, since `np.fft.rfft`
            requires an even number of dimensions (for symmetry),
            so this tells `np.fft.irfft` how to go from
            complex k-space to real space.
        axes:
            The dimension(s) of the array that will be transformed.
        """
        if isinstance(axes, int):
            axes = (axes,)
        all_axes = range(len(self.image.shape))
        fft_key = (tuple(fft_shape), tuple(axes), tuple(all_axes))

        # If this is the first time calling `fft` for this shape,
        # generate the FFT.
        if fft_key not in self._fft:
            if len(fft_shape) != len(axes):
                msg = f"fft_shape self.axes must have the same number of dimensions, got {fft_shape}, {axes}"
                raise ValueError(msg)
            image = _pad(self.image, fft_shape, axes)
            self._fft[fft_key] = np.fft.rfftn(np.fft.ifftshift(image, axes), axes=axes)
        return self._fft[fft_key]

    def __len__(self) -> int:
        """Length of the image"""
        return len(self.image)

    def __getitem__(self, index: int | Sequence[int] | slice) -> Fourier:
        # Make the index a tuple
        if isinstance(index, int):
            index = tuple([index])

        # Axes that are removed from the shape of the new object
        if isinstance(index, slice):
            removed = np.array([])
        else:
            removed = np.array([n for n, idx in enumerate(index) if idx is not None])

        # Create views into the fft transformed values, appropriately adjusting
        # the shapes for the new axes

        fft_kernels = {
            (
                tuple([s for idx, s in enumerate(key[0]) if key[0][idx] not in removed]),
                tuple([s for idx, s in enumerate(key[1]) if key[1][idx] not in removed]),
                tuple([s for idx, s in enumerate(key[2]) if key[2][idx] not in removed]),
            ): kernel[index]
            for key, kernel in self._fft.items()
        }
        # mpypy doesn't recognize that tuple[int, ...]
        # is a valid Sequence[int] for some reason
        return Fourier(self.image[index], fft_kernels)  # type: ignore


def _kspace_operation(
    image1: Fourier,
    image2: Fourier,
    padding: int,
    op: Callable,
    shape: Sequence[int],
    axes: int | Sequence[int],
) -> Fourier:
    """Combine two images in k-space using a given `operator`

    Parameters
    ----------
    image1:
        The LHS of the equation.
    image2:
        The RHS of the equation.
    padding:
        The amount of padding to add before transforming into k-space.
    op:
        The operator used to combine the two images.
        This is either ``operator.mul`` for a convolution
        or ``operator.truediv`` for deconvolution.
    shape:
        The shape of the output image.
    axes:
        The dimension(s) of the array that will be transformed.
    """
    if len(image1.shape) != len(image2.shape):
        msg = (
            "Both images must have the same number of axes, "
            f"got {len(image1.shape)} and {len(image2.shape)}"
        )
        raise ValueError(msg)

    fft_shape = get_fft_shape(image1.image, image2.image, padding, axes)
    if (
        op == operator.truediv
        or op == operator.floordiv
        or op == operator.itruediv
        or op == operator.ifloordiv
    ):
        # prevent divide by zero
        lhs = image1.fft(fft_shape, axes)
        rhs = image2.fft(fft_shape, axes)

        # Broadcast, if necessary
        if rhs.shape[0] == 1 and lhs.shape[0] != rhs.shape[0]:
            rhs = np.tile(rhs, (lhs.shape[0],) + (1,) * len(rhs.shape[1:]))
        if lhs.shape[0] == 1 and lhs.shape[0] != rhs.shape[0]:
            lhs = np.tile(lhs, (rhs.shape[0],) + (1,) * len(lhs.shape[1:]))
        # only select non-zero elements for the denominator
        cuts = rhs != 0
        transformed_fft = np.zeros(lhs.shape, dtype=lhs.dtype)
        transformed_fft[cuts] = op(lhs[cuts], rhs[cuts])
    else:
        transformed_fft = op(image1.fft(fft_shape, axes), image2.fft(fft_shape, axes))
    return Fourier.from_fft(transformed_fft, fft_shape, shape, axes)


def match_kernel(
    kernel1: np.ndarray | Fourier,
    kernel2: np.ndarray | Fourier,
    padding: int = 3,
    axes: int | Sequence[int] = (-2, -1),
    return_fourier: bool = True,
    normalize: bool = False,
) -> Fourier | np.ndarray:
    """Calculate the difference kernel to match kernel1 to kernel2

    Parameters
    ----------
    kernel1:
        The first kernel, either as array or as `Fourier` object
    kernel2:
        The second kernel, either as array or as `Fourier` object
    padding:
        Additional padding to use when generating the FFT
        to supress artifacts.
    axes:
        Axes that contain the spatial information for the kernels.
    return_fourier:
        Whether to return `Fourier` or array
    normalize:
        Whether or not to normalize the input kernels.

    Returns
    -------
    result:
        The difference kernel to go from `kernel1` to `kernel2`.
    """
    if not isinstance(kernel1, Fourier):
        kernel1 = Fourier(kernel1)
    if not isinstance(kernel2, Fourier):
        kernel2 = Fourier(kernel2)

    if kernel1.shape[0] < kernel2.shape[0]:
        shape = kernel2.shape
    else:
        shape = kernel1.shape

    diff = _kspace_operation(kernel1, kernel2, padding, operator.truediv, shape, axes=axes)
    if return_fourier:
        return diff
    else:
        return np.real(diff.image)


def convolve(
    image: np.ndarray | Fourier,
    kernel: np.ndarray | Fourier,
    padding: int = 3,
    axes: int | Sequence[int] = (-2, -1),
    return_fourier: bool = True,
    normalize: bool = False,
) -> np.ndarray | Fourier:
    """Convolve image with a kernel

    Parameters
    ----------
    image:
        Image either as array or as `Fourier` object
    kernel:
        Convolution kernel either as array or as `Fourier` object
    padding:
        Additional padding to use when generating the FFT
        to suppress artifacts.
    axes:
        Axes that contain the spatial information for the PSFs.
    return_fourier:
        Whether to return `Fourier` or array
    normalize:
        Whether or not to normalize the input kernels.

    Returns
    -------
    result:
        The convolution of the image with the kernel.
    """
    if not isinstance(image, Fourier):
        image = Fourier(image)
    if not isinstance(kernel, Fourier):
        kernel = Fourier(kernel)

    convolved = _kspace_operation(image, kernel, padding, operator.mul, image.shape, axes=axes)
    if return_fourier:
        return convolved
    else:
        return np.real(convolved.image)
