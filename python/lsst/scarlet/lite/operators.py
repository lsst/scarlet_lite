from typing import Callable, Sequence, cast

import numpy as np
import numpy.typing as npt
from lsst.scarlet.lite.detect_pybind11 import get_connected_pixels  # type: ignore
from lsst.scarlet.lite.operators_pybind11 import new_monotonicity  # type: ignore

from .bbox import Box


def prox_connected(morph: np.ndarray, centers: Sequence[Sequence[int]]) -> np.ndarray:
    """Remove all pixels not connected to the center of a source.

    Parameters
    ----------
    morph:
        The morphology that is being constrained.
    centers:
        The `(cy, cx)` center of any sources that all pixels must be
        connected to.

    Returns
    -------
    result:
        The morphology with all pixels that are not connected to a center
        postion set to zero.
    """
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


class Monotonicity:
    """Class to implement Monotonicity

    Callable class that applies monotonicity as a pseudo proximal
    operator (actually a projection operator) to *a* radially
    monotonic solution.

    Notes
    -----
    This differs from monotonicity in the main scarlet branch because
    this stores a single monotonicity operator to set the weights for all
    of the pixels up to the size of the largest shape expected,
    and only needs to be created once _per blend_, as opposed to
    once _per source_..
    This class is then called with the source morphology
    to make monotonic and the location of the "center" of the image,
    and the full weight matrix is sliced accordingly.

    Parameters
    ----------
    shape:
        The shape of the full operator.
        This must be larger than the largest possible object size
        in the blend.
    dtype:
        The numpy ``dtype`` of the output image.
    auto_update:
        If ``True`` the operator will update its shape if a image is
        too big to fit in the current operator.
    fit_radius:
        Pixels within `fit_radius` of the center of the array to make
        monotonic are checked to see if they have more flux than the center
        pixel. If they do, the pixel with larger flux is used as the center.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: npt.DTypeLike = float,
        auto_update: bool = True,
        fit_radius: int = 1,
    ):
        # Initialize defined variables
        self.weights: np.ndarray | None = None
        self.distance: np.ndarray | None = None
        self.sizes: tuple[int, int, int, int] | None = None
        self.dtype = dtype
        self.auto_update = auto_update
        self.fit_radius = fit_radius
        self.update(shape)

    @property
    def shape(self) -> tuple[int, int]:
        """The 2D shape of the largest component that can be made monotonic

        Returns
        -------
        result:
            The shape of the oeprator.
        """
        return cast(tuple[int, int], cast(np.ndarray, self.weights).shape[1:])

    @property
    def center(self) -> tuple[int, int]:
        """The center of the full operator

        Returns
        -------
        result:
            The center of the full operator.
        """
        shape = self.shape
        cx = (shape[1] - 1) // 2
        cy = (shape[0] - 1) // 2
        return cy, cx

    def update(self, shape: tuple[int, int]):
        """Update the operator with a new shape

        Parameters
        ----------
        shape:
            The new shape
        """
        if len(shape) != 2:
            msg = f"Monotonicity is a 2D operator but received shape with {len(shape)} dimensions"
            raise ValueError(msg)
        if shape[0] % 2 == 0 or shape[1] % 2 == 0:
            raise ValueError(f"The shape must be odd, got {shape}")
        # Use the center of the operator as the center
        # and calculate the distance to each pixel from the center
        cx = (shape[1] - 1) // 2
        cy = (shape[0] - 1) // 2
        _x = np.arange(shape[1], dtype=self.dtype) - cx
        _y = np.arange(shape[0], dtype=self.dtype) - cy
        x, y = np.meshgrid(_x, _y)
        distance = np.sqrt(x**2 + y**2)

        # Calculate the distance from each pixel to its 8 nearest neighbors
        neighbor_dist = np.zeros((9,) + distance.shape, dtype=self.dtype)
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
        angle_diff = np.zeros((9,) + angles.shape, dtype=self.dtype)
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
        # This gives more weight to neighboring pixels that are more closely
        # aligned with the vector pointing toward the center.
        weights = np.cos(angle_diff)
        weights[neighbor_dist <= 0] = 0
        # Adjust for the discontinuity at theta = 2pi
        weights[weights < 0] = -weights[weights < 0]
        weights = weights / np.sum(weights, axis=0)[None, :, :]

        # Store the parameters needed later
        self.weights = weights
        self.distance = distance
        self.sizes = (cy, cx, shape[0] - cy, shape[1] - cx)

    def check_size(self, shape: tuple[int, int], center: tuple[int, int], update: bool = True):
        """Check to see if the operator can be applied

        Parameters
        ----------
        shape:
            The shape of the image to apply monotonicity.
        center:
            The location (in `shape`) of the point where the monotonicity will
            be taken from.
        update:
            When ``True`` the operator will update itself so that an image
            with shape `shape` can be made monotonic about the `center`.

        Raises
        ------
        ValueError:
            Raised when an array with shape `shape` does not fit in the
            current operator and `update` is `False`.
        """
        sizes = np.array(tuple(center) + (shape[0] - center[0], shape[1] - center[1]))
        if np.any(sizes > self.sizes):
            if update:
                size = 2 * np.max(sizes) + 1
                self.update((size, size))
            else:
                raise ValueError(f"Cannot apply monotonicity to image with shape {shape} at {center}")

    def __call__(self, image: np.ndarray, center: tuple[int, int]) -> np.ndarray:
        """Make an input image monotonic about a center pixel

        Parameters
        ----------
        image:
            The image to make monotonic.
        center:
            The ``(y, x)`` location _in image coordinates_ to make the
            center of the monotonic region.

        Returns
        -------
        result:
            The input image is updated in place, but also returned from this
            method.
        """
        # Check for a better center
        center = get_peak(image, center, self.fit_radius)

        # Check that the operator can fit the image
        self.check_size(cast(tuple[int, int], image.shape), center, self.auto_update)

        # Create the bounding box to slice the weights and distance as needed
        cy, cx = self.center
        py, px = center
        bbox = Box((9,) + image.shape, origin=(0, cy - py, cx - px))
        weights = cast(np.ndarray, self.weights)[bbox.slices]
        indices = np.argsort(cast(np.ndarray, self.distance)[bbox.slices[1:]].flatten())
        coords = np.unravel_index(indices, image.shape)

        # Pad the image by 1 so that we don't have to worry about
        # weights on the edges.
        result_shape = (image.shape[0] + 2, image.shape[1] + 2)
        result = np.zeros(result_shape, dtype=image.dtype)
        result[1:-1, 1:-1] = image
        new_monotonicity(coords[0], coords[1], [w for w in weights], result)
        image[:] = result[1:-1, 1:-1]
        return image


def get_peak(image: np.ndarray, center: tuple[int, int], radius: int = 1) -> tuple[int, int]:
    """Search around a location for the maximum flux

    For monotonicity it is important to start at the brightest pixel
    in the center of the source. This may be off by a pixel or two,
    so we search for the correct center before applying
    monotonic_tree.

    Parameters
    ----------
    image:
        The image of the source.
    center:
        The suggested center of the source.
    radius:
        The number of pixels around the `center` to search
        for a higher flux value.

    Returns
    -------
    new_center:
        The true center of the source.
    """
    cy, cx = int(round(center[0])), int(round(center[1]))
    y0 = np.max([cy - radius, 0])
    x0 = np.max([cx - radius, 0])
    y_slice = slice(y0, cy + radius + 1)
    x_slice = slice(x0, cx + radius + 1)
    subset = image[y_slice, x_slice]
    center = cast(tuple[int, int], np.unravel_index(np.argmax(subset), subset.shape))
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
    x:
        The input image that the mask is created for.
    center:
        The location of the center of the mask.
    center_radius:
        Radius from the center pixel to search for a better center
        (ie. a pixel in `X` with higher flux than the pixel given by
         `center`).
        If `center_radius == 0` then the `center` pixel is assumed
        to be correct.
    variance:
        The average variance in the image.
        This is used to allow pixels to be non-monotonic up to `variance`,
        so setting `variance=0` will force strict monotonicity in the mask.
    max_iter:
        Maximum number of iterations to interpolate non-monotonic pixels.

    Returns
    -------
    valid:
        Boolean array of pixels that are monotonic.
    model:
        The model with invalid pixels masked out.
    bounds:
        The bounds of the valid monotonic pixels.
    """
    from lsst.scarlet.lite.operators_pybind11 import (
        get_valid_monotonic_pixels,
        linear_interpolate_invalid_pixels,
    )

    if center_radius > 0:
        i, j = get_peak(x, center, center_radius)
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
        linear_interpolate_invalid_pixels(all_i, all_j, unchecked, model, orphans, variance, True, bounds)
    valid = ~unchecked & ~orphans
    # Clear all of the invalid pixels from the input image
    model = model * valid
    return valid, model, tuple(bounds)  # type: ignore


def uncentered_operator(
    x: np.ndarray,
    func: Callable,
    center: tuple[int, int] | None = None,
    fill: float | None = None,
    **kwargs,
) -> np.ndarray:
    """Only apply the operator on a centered patch

    In some cases, for example symmetry, an operator might not make
    sense outside of a centered box. This operator only updates
    the portion of `X` inside the centered region.

    Parameters
    ----------
    x:
        The parameter to update.
    func:
        The function (or operator) to apply to `x`.
    center:
        The location of the center of the sub-region to
        apply `func` to `x`.
    fill:
        The value to fill the region outside of centered
        `sub-region`, for example `0`. If `fill` is `None`
        then only the subregion is updated and the rest of
        `x` remains unchanged.

    Returns
    -------
    result:
        `x`, with an operator applied based on the shifted center.
    """
    if center is None:
        py, px = cast(tuple[int, int], np.unravel_index(np.argmax(x), x.shape))
    else:
        py, px = center
    cy, cx = np.array(x.shape) // 2

    if py == cy and px == cx:
        return func(x, **kwargs)

    dy = int(round(2 * (py - cy)))
    dx = int(round(2 * (px - cx)))
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
        _x = np.ones(x.shape, x.dtype) * fill
        _x[yslice, xslice] = func(x[yslice, xslice], **kwargs)
        x[:] = _x
    else:
        x[yslice, xslice] = func(x[yslice, xslice], **kwargs)

    return x


def prox_sdss_symmetry(x: np.ndarray):
    """SDSS/HSC symmetry operator

    This function uses the *minimum* of the two
    symmetric pixels in the update.

    Parameters
    ----------
    x:
        The array to make symmetric.

    Returns
    -------
    result:
        The updated `x`.
    """
    symmetric = np.fliplr(np.flipud(x))
    x[:] = np.min([x, symmetric], axis=0)
    return x


def prox_uncentered_symmetry(
    x: np.ndarray,
    center: tuple[int, int] | None = None,
    fill: float | None = None,
) -> np.ndarray:
    """Symmetry with off-center peak

    Symmetrize X for all pixels with a symmetric partner.

    Parameters
    ----------
    x:
        The parameter to update.
    center:
        The center pixel coordinates to apply the symmetry operator.
    fill:
        The value to fill the region that cannot be made symmetric.
        When `fill` is `None` then the region of `X` that is not symmetric
        is not constrained.

    Returns
    -------
    result:
        The update function based on the specified parameters.
    """
    return uncentered_operator(x, prox_sdss_symmetry, center, fill=fill)
