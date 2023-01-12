import numpy as np
import numpy.typing as npt


def get_filter_coords(
    filter_values: np.ndarray, center: tuple[int, int] = None
) -> np.ndarray:
    """Create filter coordinate grid needed for the apply filter function

    Parameters
    ----------
    filter_values: np.ndarray
        The 2D array of the filter to apply.
    center: Sequence[int, int]
        The center (y,x) of the filter. If `center` is `None` then
        `filter_values` must have an odd number of rows and columns
        and the center will be set to the center of `filter_values`.

    Returns
    -------
    coords: np.ndarray
        The coordinates of the pixels in `filter_values`,
        where the coordinates of the `center` pixel are `(0,0)`.
    """
    if len(filter_values.shape) != 2:
        raise ValueError("`filter_values` must be 2D")
    if center is None:
        if filter_values.shape[0] % 2 == 0 or filter_values.shape[1] % 2 == 0:
            msg = """Ambiguous center of the `filter_values` array,
                     you must use a `filter_values` array
                     with an odd number of rows and columns or
                     calculate `coords` on your own."""
            raise ValueError(msg)
        center = [filter_values.shape[0] // 2, filter_values.shape[1] // 2]
    x = np.arange(filter_values.shape[1])
    y = np.arange(filter_values.shape[0])
    x, y = np.meshgrid(x, y)
    x -= center[1]
    y -= center[0]
    coords = np.dstack([y, x])
    return coords


def get_filter_bounds(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Get the slices in x and y to apply a filter

    Parameters
    ----------
    coords: np.ndarray
        The coordinates of the filter,
        defined by `get_filter_coords`.

    Returns
    -------
    y_start, y_end, x_start, x_end: Sequence[int, int, int, int]
        The start and end of each slice that is passed to `apply_filter`.
    """
    z = np.zeros((len(coords),), dtype=int)
    # Set the y slices
    y_start = np.max([z, coords[:, 0]], axis=0)
    y_end = -np.min([z, coords[:, 0]], axis=0)
    # Set the x slices
    x_start = np.max([z, coords[:, 1]], axis=0)
    x_end = -np.min([z, coords[:, 1]], axis=0)
    return y_start, y_end, x_start, x_end


def get_projection_slices(
    image: np.ndarray, shape: tuple[int, int], yx0: tuple[int, int] = None
) -> tuple[tuple[slice, slice], tuple[slice, slice], tuple[int, int, int, int]]:
    """Get slices needed to project an image

    This method returns the bounding boxes needed to
    project `image` into a larger image with `shape`.
    The results can be used with
    `projection[bb] = image[ibb]`.

    Parameters
    ----------
    image: np.ndarray.
        2D input image.
    shape: Sequence[int, int]
        Shape of the new image.
    yx0: Sequence[int, int]
        Location of the lower left corner of the image in
        the projection.
        If `yx0` is `None` then the image is centered in
        the projection.

    Returns
    -------
    bb: tuple[slice, slice]
        `(yslice, xslice)` of the projected image to place `image`.
    ibb: tuple[slice, slice]
        `(iyslice, ixslice)` of `image` to insert into the projection.
    bounds: tuple[int, int, int, int]
        `(bottom, top, left, right)` locations of the corners of `image`
        in the projection. While this isn't needed for slicing it can be
        useful for calculating information about the image before projection.
    """
    ny, nx = shape
    i_ny, i_nx = image.shape
    if yx0 is None:
        y0 = i_ny // 2
        x0 = i_nx // 2
        yx0 = (-y0, -x0)
    bottom, left = yx0
    bottom += ny >> 1
    left += nx >> 1

    top = bottom + i_ny
    yslice = slice(max(0, bottom), min(ny, top))
    iyslice = slice(max(0, -bottom), max(ny - bottom, -top))

    right = left + i_nx
    xslice = slice(max(0, left), min(nx, right))
    ixslice = slice(max(0, -left), max(nx - left, -right))
    return (yslice, xslice), (iyslice, ixslice), (bottom, top, left, right)


def project_image(
    image: np.ndarray, shape: tuple[int, int], yx0: tuple[int, int] = None
) -> np.ndarray:
    """Project an image centered in a larger image

    The projection pads the image with zeros if
    necessary or trims the edges if img is larger
    than shape in a given dimension.

    Parameters
    ----------
    image: np.ndarray
        2D input image.
    shape: Sequence[int, int]
        Shape of the new image.
    yx0: Sequence[int, int]
        Location of the lower left corner of the image in
        the projection.
        If `yx0` is `None` then the image is centered in
        the projection.

    Returns
    -------
    result: np.ndarray
        The result of projecting `image`.
    """
    result = np.zeros(shape)
    bb, ibb, _ = get_projection_slices(image, shape, yx0)
    result[bb] = image[ibb]
    return result


def common_projections(
    img1: np.ndarray, img2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project two images to a common frame

    It is assumed that the two images have the same center.
    This is mainly used for FFT convolutions of source components,
    where the convolution kernel is a different size than the morphology.

    Parameters
    ----------
    img1: np.ndarray
        1st 2D image to project
    img2: np.ndarray
        2nd 2D image to project

    Returns
    -------
    img1: np.ndarray
        Projection of 1st image
    img2: np.ndarray
        Projection of 2nd image
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    shape = (max(h1, h2), max(w1, w2))
    return project_image(img1, shape), project_image(img2, shape)


def bilinear(dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear interpolation kernel

    Interpolate between neighboring pixels to shift
    by a fractional amount.

    Parameters
    ----------
    dx: float
        Fractional amount that the kernel will be shifted.

    Returns
    -------
    result: np.ndarray
        2x2 linear kernel to use nearest neighbor interpolation.
    window: np.ndarray
        The pixel values for the window containing the kernel.
    """
    if np.abs(dx) > 1:
        raise ValueError("The fractional shift dx must be between -1 and 1")
    if dx >= 0:
        window = np.arange(2)
        y = np.array([1 - dx, dx])
    else:
        window = np.array([-1, 0])
        y = np.array([-dx, 1 + dx])
    return y, window


def cubic_spline(
    dx: float, a: float = 1, b: float = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a cubix spline centered on `dx`.

    Parameters
    ----------
    dx: float
        Fractional amount that the kernel will be shifted
    a: float
        Cubic spline sharpness paremeter
    b: float
        Cubic spline shape parameter

    Returns
    -------
    result: np.ndarray
        Cubic Spline kernel in a window from floor(dx)-1 to floor(dx) + 3
    window: np.ndarray
        The pixel values for the window containing the kernel
    """
    if np.abs(dx) > 1:
        raise ValueError("The fractional shift dx must be between -1 and 1")

    def inner(x: float):
        """Cubic from 0<=abs(x)<=1"""
        third = (-6 * a - 9 * b + 12) * x**3
        second = (6 * a + 12 * b - 18) * x**2
        zero = -2 * b + 6
        return (zero + second + third) / 6

    def outer(x: float):
        """Cubic from 1<=abs(x)<=2"""
        third = (-6 * a - b) * x**3
        second = (30 * a + 6 * b) * x**2
        first = (-48 * a - 12 * b) * x
        zero = 24 * a + 8 * b
        return (zero + first + second + third) / 6

    window = np.arange(-1, 3) + np.floor(dx)
    _x = np.abs(dx - window)
    result = np.piecewise(
        _x, [_x <= 1, (_x > 1) & (_x < 2)], [lambda x: inner(x), lambda x: outer(x)]
    )

    return result, np.array(window).astype(int)


def catmull_rom(dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Cubic spline with a=0.5, b=0

    See `cubic_spline` for details.
    """
    return cubic_spline(dx, a=0.5, b=0)


def mitchel_netravali(dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Cubic spline with a=1/3, b=1/3

    See `cubic_spline` for details.
    """
    ab = 1 / 3
    return cubic_spline(dx, a=ab, b=ab)


def lanczos(dx: float, a: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Lanczos kernel

    Parameters
    ----------
    dx: float
        amount to shift image
    a: int
        Lanczos window size parameter

    Returns
    -------
    result: np.ndarray
        1D Lanczos kernel
    windoe: np.ndarray
    """
    if np.abs(dx) > 1:
        raise ValueError("The fractional shift dx must be between -1 and 1")
    window = np.arange(-a + 1, a + 1) + np.floor(dx)
    y = np.sinc(dx - window) * np.sinc((dx - window) / a)
    return y, window.astype(int)


def quintic_spline(
    dx: float, dtype: npt.DTypeLike = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    def inner(x):
        return 1 + x**3 / 12 * (-95 + 138 * x - 55 * x**2)

    def middle(x):
        return (x - 1) * (x - 2) / 24 * (-138 + 348 * x - 249 * x**2 + 55 * x**3)

    def outer(x):
        return (x - 2) * (x - 3) ** 2 / 24 * (-54 + 50 * x - 11 * x**2)

    window = np.arange(-3, 4)
    _x = np.abs(dx - window)
    result = np.piecewise(
        _x,
        [_x <= 1, (_x > 1) & (_x <= 2), (_x > 2) & (_x <= 3)],
        [lambda x: inner(x), lambda x: middle(x), lambda x: outer(x)],
        dtype=dtype,
    )
    return result, window
