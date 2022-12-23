from functools import partial
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from .cache import Cache
from .bbox import Box
from . import fft
from . import interpolation


class Monotonicity:
    """Class to implement Monotonicity

    This differes from the monotonicity in the main scarlet branch because
    this stores a single monotonicity operator to set the weights for all
    of the pixels up to the size of the largest shape expected,
    and only needs to be created once _per blend_, as oppossed to
    once _per source_..
    This class is then called with the source morphology
    to make monotonic and the location of the "center" of the image,
    and the full weight matrix is sliced accordingly.
    """
    def __init__(self, shape: tuple[int, int], dtype: npt.DTypeLike = float):
        """Initialize the monotonicity operator

        Parameters
        ----------
        shape:
            The shape of the full operator.
            This must be larger than the largest possible object size in the blend.
        dtype:
            The numpy ``dtype`` of the output image.
        """
        # Use the center of the operator as the center
        # and calculate the distance to each pixel from the center
        cx = (shape[1] - 1) >> 1
        cy = (shape[0] - 1) >> 1
        x = np.arange(shape[1], dtype=dtype) - cx
        y = np.arange(shape[0], dtype=dtype) - cy
        x, y = np.meshgrid(x, y)
        distance = np.sqrt(x ** 2 + y ** 2)

        # Calculate the distance from each pixel to its 8 nearest neighbors
        neighbor_dist = np.zeros((9,) + distance.shape, dtype=dtype)
        neighbor_dist[0, 1:, 1:] = distance[1:, 1:] - distance[:-1, :-1]
        neighbor_dist[1, 1:, :] = distance[1:, :] - distance[:-1, :]
        neighbor_dist[2, 1:, :-1] = distance[1:, :-1] - distance[:-1, 1:]
        neighbor_dist[3, :, 1:] = distance[:, 1:] - distance[:, :-1]
        # For the center pixel, set the distance to 1 just so that it is
        # non-zero
        neighbor_dist[4, cy, cx] = 1
        neighbor_dist[5, :, :-1] = distance[:, :-1] - distance[:, 1:]
        neighbor_dist[6, :-1, 1:] = distance[:-1, 1:] - distance[1:, :-1]
        neighbor_dist[7, :-1, :] = distance[:-1, :] - distance[1:, :]
        neighbor_dist[8, :-1, :-1] = distance[:-1, :-1] - distance[1:, 1:]

        # Calculate the difference in angle to the center
        # from each pixel to its 8 nearest neighbors
        angles = np.arctan2(y, x)
        angle_diff = np.zeros((9,) + angles.shape, dtype=dtype)
        angle_diff[0, 1:, 1:] = angles[1:, 1:] - angles[:-1, :-1]
        angle_diff[1, 1:, :] = angles[1:, :] - angles[:-1, :]
        angle_diff[2, 1:, :-1] = angles[1:, :-1] - angles[:-1, 1:]
        angle_diff[3, :, 1:] = angles[:, 1:] - angles[:, :-1]
        # For the center pixel, on the center will have a non-zero cosine,
        # which is used as the weight.
        angle_diff[4] = 1
        angle_diff[4, cy, cx] = 0
        angle_diff[5, :, :-1] = angles[:, :-1] - angles[:, 1:]
        angle_diff[6, :-1, 1:] = angles[:-1, 1:] - angles[1:, :-1]
        angle_diff[7, :-1, :] = angles[:-1, :] - angles[1:, :]
        angle_diff[8, :-1, :-1] = angles[:-1, :-1] - angles[1:, 1:]

        # Use cos(theta) to set the weights, then normalize
        weights = np.cos(angle_diff)
        weights[neighbor_dist <= 0] = 0
        # Adjust for the discontinuity at theta = 2pi
        weights[weights < 0] = -weights[weights < 0]
        weights = weights / np.sum(weights, axis=0)[None, :, :]

        # Store the parameters needed later
        self.weights = weights
        self.distance = distance
        self.center = (cy, cx)

    def __call__(self, image: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        """Make an input image monotonic about a center pixel

        Parameters
        ----------
        image:
            The image to make monotonic.
        center:
            The ``(y, x)`` location _in image coordinates__ to make the
            center of the monotonic region.

        Returns
        -------
        result: np.ndarray
            The input image is updated in place, but also returned from this
            method.
        """
        from scarlet_lite.operators_pybind11 import new_monotonicity

        # Create the bounding box to slice the weights and distance as needed
        cy, cx = self.center
        py, px = center
        bbox = Box((9,) + image.shape, origin=(0, cy - py, cx - px))
        weights = self.weights[bbox.slices]
        indices = np.argsort(self.distance[bbox.slices[1:]].flatten())
        coords = np.unravel_index(indices, image.shape)

        # Pad the image by 1 so that we don't have to worry about
        # weights on the edges.
        result_shape = (image.shape[0] + 2, image.shape[1] + 2)
        result = np.zeros(result_shape, dtype=image.dtype)
        result[1:-1, 1:-1] = image
        new_monotonicity(coords[0], coords[1], [w for w in weights], result)
        image[:] = result[1:-1, 1:-1]
        return image


def get_center(
    image: np.ndarray, center: tuple[int, int], radius: int = 1
) -> tuple[int, int]:
    """Search around a location for the maximum flux

    For monotonicity it is important to start at the brightest pixel
    in the center of the source. This may be off by a pixel or two,
    so we search for the correct center before applying
    monotonic_tree.

    Parameters
    ----------
    image: np.ndarray
        The image of the source.
    center: Sequence[int, int]
        The suggested center of the source.
    radius: int
        The number of pixels around the `center` to search
        for a higher flux value.

    Returns
    -------
    new_center: tuple[int, int]
        The true center of the source.
    """
    cy, cx = int(center[0]), int(center[1])
    y0 = np.max([cy - radius, 0])
    x0 = np.max([cx - radius, 0])
    y_slice = slice(y0, cy + radius + 1)
    x_slice = slice(x0, cx + radius + 1)
    subset = image[y_slice, x_slice]
    center = np.unravel_index(np.argmax(subset), subset.shape)
    return center[0] + y0, center[1] + x0


def prox_monotonic_mask(
    x: np.ndarray,
    center: tuple[int, int],
    center_radius: int = 1,
    variance: float = 0.0,
    max_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Apply monotonicity from any path from the center

    Parameters
    ----------
    x: np.ndarray
        The input image that the mask is created for.
    center: Sequence[int, int]
        The location of the center of the mask.
    center_radius: float
        Radius from the center pixel to search for a better center
        (ie. a pixel in `X` with higher flux than the pixel given by
         `center`).
        If `center_radius == 0` then the `center` pixel is assumed to be correct.
    variance: float
        The average variance in the image.
        This is used to allow pixels to be non-monotonic up to `variance`,
        so setting `variance=0` will force strict monotonicity in the mask.
    max_iter: int
        Maximum number of iterations to interpolate non-monotonic pixels.

    Returns
    -------
    result: tuple[np.ndarray, np.ndarray, np.ndarray[int, int, int, int]]
        The result is the tuple:
            - valid: Boolean array of pixels that are monotonic
            - model: the model with invalid pixels masked out
            - bounds: the bounds of the valid monotonic pixels
    """
    from scarlet_lite.operators_pybind11 import (
        get_valid_monotonic_pixels,
        linear_interpolate_invalid_pixels,
    )

    if center_radius > 0:
        i, j = get_center(x, center, center_radius)
    else:
        i, j = int(np.round(center[0])), int(np.round(center[1]))
    unchecked = np.ones(x.shape, dtype=bool)
    unchecked[i, j] = False
    orphans = np.zeros(x.shape, dtype=bool)
    # This is the bounding box of the result
    bounds = np.array([i, i, j, j], dtype=np.int32)
    # Get all of the monotonic pixels
    get_valid_monotonic_pixels(i, j, x, unchecked, orphans, variance, bounds, 0)
    # Set the initial model to the exact input in the valid pixels
    model = x.copy()

    it = 0

    while np.sum(orphans & unchecked) > 0 and it < max_iter:
        it += 1
        all_i, all_j = np.where(orphans)
        linear_interpolate_invalid_pixels(
            all_i, all_j, unchecked, model, orphans, variance, True, bounds
        )
    valid = ~unchecked & ~orphans
    # Clear all of the invalid pixels from the input image
    model = model * valid
    return valid, model, tuple(bounds)


# TODO: remove all code below this before putting this into peer review


def sort_by_radius(
    shape: tuple[int, int], center: tuple[float, float] = None
) -> np.ndarray:
    """Sort indices distance from the center

    Given a shape, calculate the distance of each
    pixel from the center and return the indices
    of each pixel, sorted by radial distance from
    the center, which need not be in the center
    of the image.

    Parameters
    ----------
    shape: tuple[int, int]
        Shape (y,x) of the source frame.

    center: Sequence[float, float]
        Location of the center pixel.

    Returns
    -------
    didx: np.ndarray
        Indices of elements in an image with shape `shape`,
        sorted by distance from the center.
    """
    # Get the center pixels
    if center is None:
        cx = (shape[1] - 1) >> 1
        cy = (shape[0] - 1) >> 1
    else:
        cy, cx = int(center[0]), int(center[1])
    # Calculate the distance between each pixel and the peak
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    x, y = np.meshgrid(x, y)
    x = x - cx
    y = y - cy
    distance = np.sqrt(x**2 + y**2)
    # Get the indices of the pixels sorted by distance from the peak
    didx = np.argsort(distance.flatten())
    return didx


def _prox_weighted_monotonic(
    x: np.ndarray,
    weights: np.ndarray,
    didx: Sequence[int],
    offsets: Sequence[int],
    min_gradient: float = 0.1,
):
    """Force an intensity profile to be monotonic based on weighting neighbors"""
    from scarlet_lite.operators_pybind11 import prox_weighted_monotonic

    prox_weighted_monotonic(x.reshape(-1), weights, offsets, didx, min_gradient)
    return x


def prox_weighted_monotonic(
    shape: tuple[int, int],
    neighbor_weight: str = "flat",
    min_gradient: float = 0.1,
    center: tuple[int, int] = None,
) -> Callable:
    """Build the prox_monotonic operator

    Parameters
    ----------
    shape: Sqeuence[int, int]
        Shape of the monotonic array.
    neighbor_weight: str
        Which weighting scheme ('flat', 'angle', 'nearest') to use for
        averaging all neighbor pixels towards `center`
        as reference for the monotonicty test.
    min_gradient: float
        Forced gradient. A `thresh` of zero will allow a pixel to be the
        same value as its reference pixels, while a `thresh` of one
        will force the pixel to zero.
    center: Sequence[int, int]
        Location of the central (highest-value) pixel.

    Returns
    -------
    result: Callable
        The monotonicity function.
    """
    height, width = shape
    didx = sort_by_radius(shape, center)
    coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    offsets = np.array([width * y + x for y, x in coords])
    weights = _get_radial_monotonic_weights(
        shape, neighbor_weight=neighbor_weight, center=center
    )
    result = partial(
        _prox_weighted_monotonic,
        weights=weights,
        didx=didx[1:],
        offsets=offsets,
        min_gradient=min_gradient,
    )
    return result



def uncentered_operator(
    x: np.ndarray,
    func: Callable,
    center: tuple[float, float] = None,
    fill: float = None,
    **kwargs
) -> np.ndarray:
    """Only apply the operator on a centered patch

    In some cases, for example symmetry, an operator might not make
    sense outside of a centered box. This operator only updates
    the portion of `X` inside the centered region.

    Parameters
    ----------
    x: np.ndarray
        The parameter to update.
    func: Callable
        The function (or operator) to apply to `x`.
    center: Sequence[float, float]
        The location of the center of the sub-region to
        apply `func` to `x`.
    fill: float
        The value to fill the region outside of centered
        `sub-region`, for example `0`. If `fill` is `None`
        then only the subregion is updated and the rest of
        `x` remains unchanged.

    Returns
    -------
    result : np.ndarray
        `x`, with an operator applied based on the shifted center.
    """
    if center is None:
        py, px = np.unravel_index(np.argmax(x), x.shape)
    else:
        py, px = center
    cy, cx = np.array(x.shape) // 2

    if py == cy and px == cx:
        return func(x, **kwargs)

    dy = int(2 * (py - cy))
    dx = int(2 * (px - cx))
    if not x.shape[0] % 2:
        dy += 1
    if not x.shape[1] % 2:
        dx += 1
    if dx < 0:
        xslice = slice(None, dx)
    else:
        xslice = slice(dx, None)
    if dy < 0:
        yslice = slice(None, dy)
    else:
        yslice = slice(dy, None)

    if fill is not None:
        _X = np.ones(x.shape, x.dtype) * fill
        _X[yslice, xslice] = func(x[yslice, xslice], **kwargs)
        x[:] = _X
    else:
        x[yslice, xslice] = func(x[yslice, xslice], **kwargs)

    return x


def prox_sdss_symmetry(x: np.ndarray):
    """SDSS/HSC symmetry operator

    This function uses the *minimum* of the two
    symmetric pixels in the update.

    Parameters
    ----------
    x: np.ndarray
        The array to make symmetric.

    Returns
    -------
    result: np.ndarray
        The updated `x`.
    """
    symmetric = np.fliplr(np.flipud(x))
    x[:] = np.min([x, symmetric], axis=0)
    return x


def prox_soft_symmetry(x: np.ndarray, strength: float = 1):
    """Soft version of symmetry
    Using a `strength` that varies from 0 to 1,
    with 0 meaning no symmetry enforced at all and
    1  being completely symmetric, the user can customize
    the level of symmetry required for a component

    Parameters
    ----------
    x: np.ndarray
        The array to make symmetric.
    strength: float
        The fraction of flux allowed to be non-symmetric
        (between [0, 1].

    Returns
    -------
    result: np.ndarray
        The updated `x`.
    """
    pads = [[0, 0], [0, 0]]
    slices = [slice(None), slice(None)]
    if x.shape[0] % 2 == 0:
        pads[0][1] = 1
        slices[0] = slice(0, x.shape[0])
    if x.shape[1] % 2 == 0:
        pads[1][1] = 1
        slices[1] = slice(0, x.shape[1])

    x = fft.fast_zero_pad(x, pads)
    symmetric = np.fliplr(np.flipud(x))
    x = 0.5 * strength * (x + symmetric) + (1 - strength) * x
    return x[tuple(slices)]


def prox_kspace_symmetry(
    x: np.ndarray, shift: Sequence[float] = None, padding: int = 10
) -> np.ndarray:
    """Symmetry in Fourier Space

    This algorithm by Nate Lust uses the fact that throwing
    away the imaginary part in Fourier space leaves a symmetric
    soution in real space. So `X` is transformed to Fourier space,
    shifted by the fractional amount `shift=(dy, dx)`,
    the imaginary part is discarded, shited back to its original position,
    then transformed back to real space.

    Parameters
    ----------
    x:
        The array to make symmetric.
    shift:
        The amount ``(dy, dx)`` to shift `x` before making it symmetric.
    padding:
        The amount of padding to use to limit FFT artifacts.

    Returns
    -------
    result: np.ndarray
        The updated `x` after being made symmetric.
    """
    # Get fast shapes
    fft_shape = fft.get_fft_shape(x, x, padding=padding)
    dy, dx = shift

    x = fft.Fourier(x)
    x_fft = x.fft(fft_shape, (0, 1))

    zero_mask = x.image <= 0

    # Compute shift operator
    shifter_y, shifter_x = interpolation.mk_shifter(fft_shape)

    # Apply shift in Fourier
    result_fft = x_fft * np.exp(shifter_y[:, np.newaxis] * (-dy))
    result_fft *= np.exp(shifter_x[np.newaxis, :] * (-dx))

    # symmetrize
    result_fft = result_fft.real

    # Unshift
    result_fft = result_fft * np.exp(shifter_y[:, np.newaxis] * dy)
    result_fft = result_fft * np.exp(shifter_x[np.newaxis, :] * dx)

    result = fft.Fourier.from_fft(result_fft, fft_shape, x.image.shape, [0, 1])

    result.image[zero_mask] = 0
    return np.real(result.image)


def prox_uncentered_symmetry(
    x: np.ndarray,
    center: tuple[int, int] = None,
    algorithm: str = "kspace",
    fill: float = None,
    shift: Sequence[float] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Symmetry with off-center peak

    Symmetrize X for all pixels with a symmetric partner.

    Parameters
    ----------
    x: np.ndarray
        The parameter to update.
    center: Sequence[int, int]
        The center pixel coordinates to apply the symmetry operator.
    algorithm: str
        The algorithm to use for symmetry.
        * If `algorithm = "kspace" then `X` is shifted by `shift` and
          symmetry is performed in kspace. This is the only symmetry algorithm
          in scarlet that works for fractional pixel shifts.
        * If `algorithm = "sdss" then the SDSS symmetry is used,
          namely the source is made symmetric around the `center` pixel
          by taking the minimum of each pixel and its symmetric partner.
          This is the algorithm used when initializing an `ExtendedSource`
          because it keeps the morphologies small, but during optimization
          the penalty is much stronger than the gradient
          and often leads to vanishing sources.
        * If `algorithm = "soft" then soft symmetry is used,
          meaning `X` will be allowed to differ from symmetry by the fraction
          `strength` from a perfectly symmetric solution. It is advised against
          using this algorithm because it does not work in general for sources
          shifted by a fractional amount, however it is used internally if
          a source is centered perfectly on a pixel.
    fill: float
        The value to fill the region that cannot be made symmetric.
        When `fill` is `None` then the region of `X` that is not symmetric
        is not constrained.
    shift: Sequence[float, float]
        The amount ``(dy, dx)`` to shift `x` before making it symmetric.
    strength: float
        The amount that symmetry is enforced. If `strength=0` then no
        symmetry is enforced, while `strength=1` enforces strict symmetry
        (ie. the mean of the two symmetric pixels is used for both of them).
        This parameter is only used when `algorithm = "soft"`.

    Returns
    -------
    result: Callable
        The update function based on the specified parameters.
    """
    if algorithm == "kspace" and (shift is None or np.all(shift == 0)):
        algorithm = "soft"
        strength = 1
    if algorithm == "kspace":
        return uncentered_operator(
            x, prox_kspace_symmetry, center, shift=shift, fill=fill
        )
    if algorithm == "sdss":
        return uncentered_operator(x, prox_sdss_symmetry, center, fill=fill)
    if algorithm == "soft" or algorithm == "kspace" and shift is None:
        # If there is no shift then the symmetry is exact and we can just use
        # the soft symmetry algorithm
        return uncentered_operator(
            x, prox_soft_symmetry, center, strength=strength, fill=fill
        )

    msg = "algorithm must be one of 'soft', 'sdss', 'kspace', recieved '{0}''"
    raise ValueError(msg.format(algorithm))


def _get_offsets(
    width: int, coords: Sequence[Sequence[int]] = None
) -> tuple[list[int], list[slice], list[slice]]:
    """Get the offset and slices for a sparse band diagonal array
    For an operator that interacts with its neighbors we want a band diagonal matrix,
    where each row describes the 8 pixels that are neighbors for the reference pixel
    (the diagonal). Regardless of the operator, these 8 bands are always the same,
    so we make a utility function that returns the offsets (passed to scipy.sparse.diags).
    See `diagonalizeArray` for more on the slices and format of the array used to create
    NxN operators that act on a data vector.
    """
    # Use the neighboring pixels by default
    if coords is None:
        coords = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    offsets = [width * y + x for y, x in coords]
    slices = [slice(None, s) if s < 0 else slice(s, None) for s in offsets]
    slices_inverse = [slice(-s, None) if s < 0 else slice(None, -s) for s in offsets]
    return offsets, slices, slices_inverse


def _diagonalize_array(
    arr: np.ndarray, shape: tuple[int, int] = None, dtype: npt.DTypeLike = np.float64
) -> tuple[np.ndarray, np.ndarray[bool]]:
    """Convert an array to a matrix that compares each pixel to its neighbors
    Given an array with length N, create an 8xN array, where each row will be a
    diagonal in a diagonalized array. Each column in this matrix is a row in the larger
    NxN matrix used for an operator, except that this 2D array only contains the values
    used to create the bands in the band diagonal matrix.
    Because the off-diagonal bands have less than N elements, ``getOffsets`` is used to
    create a mask that will set the elements of the array that are outside of the matrix to zero.
    ``arr`` is the vector to diagonalize, for example the distance from each pixel to the peak,
    or the angle of the vector to the peak.
    ``shape`` is the shape of the original image.
    """
    if shape is None:
        height, width = arr.shape
        data = arr.flatten()
    elif len(arr.shape) == 1:
        height, width = shape
        data = np.copy(arr)
    else:
        raise ValueError("Expected either a 2D array or a 1D array and a shape")
    size = width * height

    # We hard code 8 rows, since each row corresponds to a neighbor
    # of each pixel.
    diagonals = np.zeros((8, size), dtype=dtype)
    mask = np.ones((8, size), dtype=bool)
    offsets, slices, slices_inverse = _get_offsets(width)
    for n, s in enumerate(slices):
        diagonals[n][slices_inverse[n]] = data[s]
        mask[n][slices_inverse[n]] = 0

    # Create a mask to hide false neighbors for pixels on the edge
    # (for example, a pixel on the left edge should not be connected to the
    # pixel to its immediate left in the flattened vector, since that pixel
    # is actual the far right pixel on the row above it).
    mask[0][np.arange(1, height) * width] = 1
    mask[2][np.arange(height) * width - 1] = 1
    mask[3][np.arange(1, height) * width] = 1
    mask[4][np.arange(1, height) * width - 1] = 1
    mask[5][np.arange(height) * width] = 1
    mask[7][np.arange(1, height - 1) * width - 1] = 1

    return diagonals, mask


def _get_radial_monotonic_weights(
    shape: tuple[int, int],
    neighbor_weight: str = "flat",
    center: tuple[int, int] = None,
) -> np.ndarray:
    """Create the weights used for the Radial Monotonicity Operator
    This version of the radial monotonicity operator selects all of the pixels closer to the peak
    for each pixel and weights their flux based on their alignment with a vector from the pixel
    to the peak. In order to quickly create this using sparse matrices, its construction is a bit opaque.
    """
    assert neighbor_weight in ["flat", "angle", "nearest"]

    # Center on the center pixel
    if center is None:
        center = ((shape[0] - 1) // 2, (shape[1] - 1) // 2)
    py, px = int(center[0]), int(center[1])

    # Calculate the distance between each pixel and the peak
    _x = np.arange(shape[1]) - px
    _y = np.arange(shape[0]) - py
    x, y = np.meshgrid(_x, _y)
    distance = np.sqrt(x**2 + y**2)

    # Find each pixels neighbors further from the peak and mark them as invalid
    # (to be removed later)
    dist_arr, mask = _diagonalize_array(distance, dtype=np.float64)
    relative_dist = (distance.flatten()[:, None] - dist_arr.T).T
    invalid_pix = relative_dist <= 0

    # Calculate the angle between each pixel and the x axis, relative to the peak position
    # (also avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually)
    inf = x == 0
    x_copy = x.copy()
    x_copy[inf] = 1
    angles = np.arctan2(-y, -x_copy)
    angles[inf & (y != 0)] = 0.5 * np.pi * np.sign(angles[inf & (y != 0)])

    # Calculate the angle between each pixel and its neighbors
    x_arr, m = _diagonalize_array(x)
    y_arr, m = _diagonalize_array(y)
    dx = (x_arr.T - x.flatten()[:, None]).T
    dy = (y_arr.T - y.flatten()[:, None]).T
    # Avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually
    inf = dx == 0
    dx[inf] = 1
    relative_angles = np.arctan2(dy, dx)
    relative_angles[inf & (dy != 0)] = (
        0.5 * np.pi * np.sign(relative_angles[inf & (dy != 0)])
    )

    # Find the difference between each pixels angle with the peak
    # and the relative angles to its neighbors, and take the
    # cos to find its neighbors weight
    delta_angles = (angles.flatten()[:, None] - relative_angles.T).T
    cos_weight = np.cos(delta_angles)
    # Mask edge pixels, array elements outside the operator (for offdiagonal bands with < N elements),
    # and neighbors further from the peak than the reference pixel
    cos_weight[invalid_pix] = 0
    cos_weight[mask] = 0

    if neighbor_weight == "nearest":
        # Only use a single pixel most in line with peak
        cos_norm = np.zeros_like(cos_weight)
        column_indices = np.arange(cos_weight.shape[1])
        max_indices = np.argmax(cos_weight, axis=0)
        indices = max_indices * cos_norm.shape[1] + column_indices
        indices = np.unravel_index(indices, cos_norm.shape)
        cos_norm[indices] = 1
        # Remove the reference for the peak pixel
        cos_norm[:, px + py * shape[1]] = 0
    else:
        if neighbor_weight == "flat":
            cos_weight[cos_weight != 0] = 1

        # Normalize the cos weights for each pixel
        normalize = np.sum(cos_weight, axis=0)
        normalize[normalize == 0] = 1
        cos_norm = (cos_weight.T / normalize[:, None]).T
        cos_norm[mask] = 0

    return cos_norm


def prox_connected(morph: np.ndarray, centers: Sequence[Sequence[int]]) -> np.ndarray:
    """Remove all pixels not connected to the center of a source.

    Parameters
    ----------
    morph: np.ndarray
        The morphology that is being constrained.
    centers: Sequence[Sequence[int, int]]
        The `(cy, cx)` center of any sources that all pixels must be
        connected to.

    Returns
    -------
    result: np.ndarray
        The morphology with all pixels not connected to a center postion set
        to zero.
    """
    # Import here to avoid circular dependency
    from scarlet_lite.detect_pybind11 import get_connected_pixels

    result = np.zeros(morph.shape, dtype=bool)

    for center in centers:
        unchecked = np.ones(morph.shape, dtype=bool)
        cy, cx = center
        cy = int(cy)
        cx = int(cx)
        bounds = np.array([cy, cy, cx, cx]).astype(np.int32)
        # Update the result in place with the pixels connected to this center
        get_connected_pixels(cy, cx, morph, unchecked, result, bounds, 0)

    return result * morph


class MonotonicityConstraint:
    """Make morphology monotonically decrease from the center

    See `prox_monotonic`
    for a description of the parameters.
    """

    def __init__(
        self,
        neighbor_weight: str = "flat",
        min_gradient: float = 0.1,
        use_mask: bool = False,
        fit_center_radius: int = 0,
    ):
        self.neighbor_weight = neighbor_weight
        self.min_gradient = min_gradient
        self.use_mask = use_mask
        self.fit_center = fit_center_radius > 0
        self.fit_center_radius = fit_center_radius

    def __call__(self, morph: np.ndarray):
        shape = morph.shape
        center = (shape[0] // 2, shape[1] // 2)
        if self.fit_center:
            center = get_center(morph, center, radius=self.fit_center_radius)

        # get prox from the cache
        prox_name = "operator.prox_weighted_monotonic"
        key = (shape, center, self.neighbor_weight, self.min_gradient)
        # The creation of this operator is expensive,
        # so load it from memory if possible.
        try:
            prox = Cache.check(prox_name, key)
        except KeyError:
            prox = prox_weighted_monotonic(
                shape,
                neighbor_weight=self.neighbor_weight,
                min_gradient=self.min_gradient,
                center=center,
            )
            Cache.set(prox_name, key, prox)

        # apply the prox
        _morph = morph.copy()
        result = prox(morph)
        if self.use_mask:
            valid, _morph, _bounds = prox_monotonic_mask(
                _morph,
                center=center,
                center_radius=0,
                variance=0,
                max_iter=0,
            )
            result[valid] = _morph[valid]

        return result
