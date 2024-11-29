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

__all__ = [
    "starlet_transform",
    "starlet_reconstruction",
    "multiband_starlet_transform",
    "multiband_starlet_reconstruction",
    "get_multiresolution_support",
]

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


def bspline_convolve(image: np.ndarray, scale: int) -> np.ndarray:
    """Convolve an image with a bspline at a given scale.

    This uses the spline
    `h1d = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16])`
    from Starck et al. 2011.

    Parameters
    ----------
    image:
        The 2D image or wavelet coefficients to convolve.
    scale:
        The wavelet scale for the convolution. This sets the
        spacing between adjacent pixels with the spline.

    Returns
    -------
    result:
        The result of convolving the `image` with the spline.
    """
    # Filter for the scarlet transform. Here bspline
    h1d = np.array([1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4, 1.0 / 16]).astype(image.dtype)
    j = scale

    slice0 = slice(None, -(2 ** (j + 1)))
    slice1 = slice(None, -(2**j))
    slice3 = slice(2**j, None)
    slice4 = slice(2 ** (j + 1), None)
    # row
    col = image * h1d[2]
    col[slice4] += image[slice0] * h1d[0]
    col[slice3] += image[slice1] * h1d[1]
    col[slice1] += image[slice3] * h1d[3]
    col[slice0] += image[slice4] * h1d[4]

    # column
    result = col * h1d[2]
    result[:, slice4] += col[:, slice0] * h1d[0]
    result[:, slice3] += col[:, slice1] * h1d[1]
    result[:, slice1] += col[:, slice3] * h1d[3]
    result[:, slice0] += col[:, slice4] * h1d[4]
    return result


def get_starlet_scales(image_shape: Sequence[int], scales: int | None = None) -> int:
    """Get the number of scales to use in the starlet transform.

    Parameters
    ----------
    image_shape:
        The 2D shape of the image that is being transformed
    scales:
        The number of scales to transform with starlets.
        The total dimension of the starlet will have
        `scales+1` dimensions, since it will also hold
        the image at all scales higher than `scales`.

    Returns
    -------
    result:
        Number of scales, adjusted for the size of the image.
    """
    # Number of levels for the Starlet decomposition
    max_scale = int(np.log2(np.min(image_shape[-2:]))) - 1
    if (scales is None) or scales > max_scale:
        scales = max_scale
    return int(scales)


def starlet_transform(
    image: np.ndarray,
    scales: int | None = None,
    generation: int = 2,
    convolve2d: Callable | None = None,
) -> np.ndarray:
    """Perform a starlet transform, or 2nd gen starlet transform.

    Parameters
    ----------
    image:
        The image to transform into starlet coefficients.
    scales:
        The number of scale to transform with starlets.
        The total dimension of the starlet will have
        `scales+1` dimensions, since it will also hold
        the image at all scales higher than `scales`.
    generation:
        The generation of the transform.
        This must be `1` or `2`.
    convolve2d:
        The filter function to use to convolve the image
        with starlets in 2D.

    Returns
    -------
    starlet:
        The starlet dictionary for the input `image`.
    """
    if len(image.shape) != 2:
        raise ValueError(f"Image should be 2D, got {len(image.shape)}")
    if generation not in (1, 2):
        raise ValueError(f"generation should be 1 or 2, got {generation}")

    scales = get_starlet_scales(image.shape, scales)
    c = image
    if convolve2d is None:
        convolve2d = bspline_convolve

    # wavelet set of coefficients.
    starlet = np.zeros((scales + 1,) + image.shape, dtype=image.dtype)
    for j in range(scales):
        gen1 = convolve2d(c, j)

        if generation == 2:
            gen2 = convolve2d(gen1, j)
            starlet[j] = c - gen2
        else:
            starlet[j] = c - gen1

        c = gen1

    starlet[-1] = c
    return starlet


def multiband_starlet_transform(
    image: np.ndarray,
    scales: int | None = None,
    generation: int = 2,
    convolve2d: Callable | None = None,
) -> np.ndarray:
    """Perform a starlet transform of a multiband image.

    See `starlet_transform` for a description of the parameters.
    """
    if len(image.shape) != 3:
        raise ValueError(f"Image should be 3D (bands, height, width), got shape {len(image.shape)}")
    if generation not in (1, 2):
        raise ValueError(f"generation should be 1 or 2, got {generation}")
    scales = get_starlet_scales(image.shape, scales)

    wavelets = np.empty((scales + 1,) + image.shape, dtype=image.dtype)
    for b, image in enumerate(image):
        wavelets[:, b] = starlet_transform(image, scales=scales, generation=generation, convolve2d=convolve2d)
    return wavelets


def starlet_reconstruction(
    starlets: np.ndarray,
    generation: int = 2,
    convolve2d: Callable | None = None,
) -> np.ndarray:
    """Reconstruct an image from a dictionary of starlets

    Parameters
    ----------
    starlets:
        The starlet dictionary used to reconstruct the image
        with dimension (scales+1, Ny, Nx).
    generation:
        The generation of the starlet transform (either ``1`` or ``2``).
    convolve2d:
        The filter function to use to convolve the image
        with starlets in 2D.

    Returns
    -------
    image:
        The 2D image reconstructed from the input `starlet`.
    """
    if generation == 1:
        return np.sum(starlets, axis=0)
    if convolve2d is None:
        convolve2d = bspline_convolve
    scales = len(starlets) - 1

    c = starlets[-1]
    for i in range(1, scales + 1):
        j = scales - i
        cj = convolve2d(c, j)
        c = cj + starlets[j]
    return c


def multiband_starlet_reconstruction(
    starlets: np.ndarray,
    generation: int = 2,
    convolve2d: Callable | None = None,
) -> np.ndarray:
    """Reconstruct a multiband image.

    See `starlet_reconstruction` for a description of the
    remainder of the parameters.
    """
    _, bands, width, height = starlets.shape
    result = np.zeros((bands, width, height), dtype=starlets.dtype)
    for band in range(bands):
        result[band] = starlet_reconstruction(starlets[:, band], generation=generation, convolve2d=convolve2d)
    return result


@dataclass
class MultiResolutionSupport:
    support: np.ndarray
    sigma: np.ndarray


def get_multiresolution_support(
    image: np.ndarray,
    starlets: np.ndarray,
    sigma: np.floating,
    sigma_scaling: float = 3,
    epsilon: float = 1e-1,
    max_iter: int = 20,
    image_type: str = "ground",
) -> MultiResolutionSupport:
    """Calculate the multi-resolution support for a
    dictionary of starlet coefficients.

    This is different for ground and space based telescopes.
    For space-based telescopes the procedure in Starck and Murtagh 1998
    iteratively calculates the multi-resolution support.
    For ground based images, where the PSF is much wider and there are no
    pixels with no signal at all scales, we use a modified method that
    estimates support at each scale independently.

    Parameters
    ----------
    image:
        The image to transform into starlet coefficients.
    starlets:
        The starlet dictionary used to reconstruct `image` with
        dimension (scales+1, Ny, Nx).
    sigma:
        The standard deviation of the `image`.
    sigma_scaling:
        The multiple of `sigma` to use to calculate significance.
        Coefficients `w` where `|w| > K*sigma_j`, where `sigma_j` is
        standard deviation at the jth scale, are considered significant.
    epsilon:
        The convergence criteria of the algorithm.
        Once `|new_sigma_j - sigma_j|/new_sigma_j < epsilon` the
        algorithm has completed.
    max_iter:
        Maximum number of iterations to fit `sigma_j` at each scale.
    image_type:
        The type of image that is being used.
        This should be "ground" for ground based images with wide PSFs or
        "space" for images from space-based telescopes with a narrow PSF.

    Returns
    -------
    M:
        Mask with significant coefficients in `starlets` set to `True`.
    """
    if image_type not in ("ground", "space"):
        raise ValueError(f"image_type must be 'ground' or 'space', got {image_type}")

    if image_type == "space":
        # Calculate sigma_je, the standard deviation at
        # each scale due to gaussian noise
        noise_img = np.random.normal(size=image.shape)
        noise_starlet = starlet_transform(noise_img, generation=1, scales=len(starlets) - 1)
        sigma_je = np.zeros((len(noise_starlet),))
        for j, star in enumerate(noise_starlet):
            sigma_je[j] = np.std(star)
        noise = image - starlets[-1]

        last_sigma_i = sigma
        for it in range(max_iter):
            m = np.abs(starlets) > sigma_scaling * sigma * sigma_je[:, None, None]
            s = np.sum(m, axis=0) == 0
            sigma_i = np.std(noise * s)
            if np.abs(sigma_i - last_sigma_i) / sigma_i < epsilon:
                break
            last_sigma_i = sigma_i
        sigma_j = sigma_je
    else:
        # Sigma to use for significance at each scale
        # Initially we use the input `sigma`
        sigma_j = np.full(len(starlets), sigma, dtype=image.dtype)
        last_sigma_j = sigma_j
        for it in range(max_iter):
            m = np.abs(starlets) > sigma_scaling * sigma_j[:, None, None]
            # Take the standard deviation of the current
            # insignificant coeffs at each scale
            s = ~m
            sigma_j = np.std(starlets * s.astype(int), axis=(1, 2))
            # At lower scales all of the pixels may be significant,
            # so sigma is effectively zero. To avoid infinities we
            # only check the scales with non-zero sigma
            cut = sigma_j > 0
            if np.all(np.abs(sigma_j[cut] - last_sigma_j[cut]) / sigma_j[cut] < epsilon):
                break

            last_sigma_j = sigma_j
    # noinspection PyUnboundLocalVariable
    return MultiResolutionSupport(support=m.astype(int), sigma=sigma_j)


def apply_wavelet_denoising(
    image: np.ndarray,
    sigma: np.floating | None = None,
    sigma_scaling: float = 3,
    epsilon: float = 1e-1,
    max_iter: int = 20,
    image_type: str = "ground",
    positive: bool = True,
) -> np.ndarray:
    """Apply wavelet denoising

    Uses the algorithm and notation from Starck et al. 2011, section 4.1

    Parameters
    ----------
    image:
        The image to denoise
    sigma:
        The standard deviation of the image
    sigma_scaling:
        The threshold in units of sigma to declare a coefficient significant
    epsilon:
        Convergence criteria for determining the support
    max_iter:
        The maximum number of iterations.
        This applies to both finding the support and the denoising loop.
    image_type:
        The type of image that is being used.
        This should be "ground" for ground based images with wide PSFs or
        "space" for images from space-based telescopes with a narrow PSF.
    positive:
        Whether or not the expected result should be positive

    Returns
    -------
    result:
        The resulting denoised image after `max_iter` iterations.
    """
    image_coeffs = starlet_transform(image)
    if sigma is None:
        sigma = np.median(np.absolute(image - np.median(image)))
    coeffs = image_coeffs.copy()
    support = get_multiresolution_support(
        image=image,
        starlets=coeffs,
        sigma=sigma,
        sigma_scaling=sigma_scaling,
        epsilon=epsilon,
        max_iter=max_iter,
        image_type=image_type,
    )
    x = starlet_reconstruction(coeffs)

    for n in range(max_iter):
        coeffs = starlet_transform(x)
        x = x + starlet_reconstruction(support.support * (image_coeffs - coeffs))
        if positive:
            x[x < 0] = 0
    return x
