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

import logging
from dataclasses import dataclass
from math import comb
from typing import Sequence

import numpy as np
from deprecated.sphinx import deprecated  # type: ignore
from lsst.scarlet.lite.detect_pybind11 import Footprint, get_footprints  # type: ignore
from scipy import stats
from scipy.optimize import brentq

from .bbox import Box, overlapped_slices
from .image import Image
from .utils import continue_class
from .wavelet import (
    get_multiresolution_support,
    get_starlet_scales,
    multiband_starlet_reconstruction,
    multiband_starlet_transform,
    starlet_transform,
)

logger = logging.getLogger("scarlet.detect")


def bounds_to_bbox(bounds: tuple[int, int, int, int]) -> Box:
    """Convert the bounds of a Footprint into a Box

    Notes
    -----
    Unlike slices, the bounds are _inclusive_ of the end points.

    Parameters
    ----------
    bounds:
        The bounds of the `Footprint` as a `tuple` of
        ``(bottom, top, left, right)``.
    Returns
    -------
    result:
        The `Box` created from the bounds
    """
    return Box(
        (bounds[1] + 1 - bounds[0], bounds[3] + 1 - bounds[2]),
        origin=(bounds[0], bounds[2]),
    )


def bbox_to_bounds(bbox: Box) -> tuple[int, int, int, int]:
    """Convert a Box into the bounds of a Footprint

    Parameters
    ----------
    bbox:
        The `Box` to convert into bounds.

    Returns
    -------
    result:
        The bounds of the `Footprint` as a `tuple` of
        ``(bottom, top, left, right)``.

    Notes
    -----
    Unlike slices, the bounds are _inclusive_ of the end points.
    """
    bounds = (
        bbox.origin[0],
        bbox.origin[0] + bbox.shape[0] - 1,
        bbox.origin[1],
        bbox.origin[1] + bbox.shape[1] - 1,
    )
    return bounds


@continue_class
class Footprint:  # type: ignore # noqa
    @property
    def bbox(self) -> Box:
        """Bounding box for the Footprint

        Returns
        -------
        bbox:
            The minimal `Box` that contains the entire `Footprint`.
        """
        return bounds_to_bbox(self.bounds)  # type: ignore

    @property
    def yx0(self) -> tuple[int, int]:
        """Origin in y, x of the lower left corner of the footprint"""
        return self.bounds[0], self.bounds[2]  # type: ignore

    def intersection(self, other: Footprint) -> Image | None:
        """The intersection of two footprints

        Parameters
        ----------
        other:
            The other footprint to compare.

        Returns
        -------
        intersection:
            The intersection of two footprints.
        """
        footprint1 = Image(self.data, yx0=self.yx0)  # type: ignore
        footprint2 = Image(other.data, yx0=other.yx0)  # type: ignore # noqa
        return footprint1 & footprint2

    def union(self, other: Footprint) -> Image | None:
        """The union of two footprints

        Parameters
        ----------
        other:
            The other footprint to compare.

        Returns
        -------
        union:
            The union of two footprints.
        """
        footprint1 = Image(self.data, yx0=self.yx0)  # type: ignore
        footprint2 = Image(other.data, yx0=other.yx0)
        return footprint1 | footprint2


def footprints_to_image(footprints: Sequence[Footprint], bbox: Box) -> Image:
    """Convert a set of scarlet footprints to a pixelized image.

    Parameters
    ----------
    footprints:
        The footprints to convert into an image.
    box:
        The full box of the image that will contain the footprints.

    Returns
    -------
    result:
        The image created from the footprints.
    """
    result = Image.from_box(bbox, dtype=int)
    for k, footprint in enumerate(footprints):
        slices = overlapped_slices(result.bbox, footprint.bbox)
        result.data[slices[0]] += footprint.data[slices[1]] * (k + 1)
    return result


def get_wavelets(
    images: np.ndarray,
    variance: np.ndarray,
    scales: int | None = None,
    generation: int = 2,
) -> np.ndarray:
    """Calculate wavelet coefficents given a set of images and their variances

    Parameters
    ----------
    images:
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance:
        An array of variances with the same shape as `images`.
    scales:
        The maximum number of wavelet scales to use.

    Returns
    -------
    coeffs:
        The array of coefficents with shape `(scales+1, bands, Ny, Nx)`.
        Note that the result has `scales+1` total arrays,
        since the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.
    """
    sigma = np.median(np.sqrt(variance), axis=(1, 2))
    # Create the wavelet coefficients for the significant pixels
    scales = get_starlet_scales(images[0].shape, scales)
    coeffs = np.empty((scales + 1,) + images.shape, dtype=images.dtype)
    for b, image in enumerate(images):
        _coeffs = starlet_transform(image, scales=scales, generation=generation)
        support = get_multiresolution_support(
            image=image,
            starlets=_coeffs,
            sigma=sigma[b],
            sigma_scaling=3,
            epsilon=1e-1,
            max_iter=20,
        )
        coeffs[:, b] = (support.support * _coeffs).astype(images.dtype)
    return coeffs


def get_detect_wavelets(images: np.ndarray, variance: np.ndarray, scales: int = 3) -> np.ndarray:
    """Get an array of wavelet coefficents to use for detection

    Parameters
    ----------
    images:
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance:
        An array of variances with the same shape as `images`.
    scales:
        The maximum number of wavelet scales to use.
        Note that the result will have `scales+1` total arrays,
        where the last set of coefficients is the image of all
        flux with frequency greater than the last wavelet scale.

    Returns
    -------
    starlets:
        The array of wavelet coefficients for pixels with siignificant
        amplitude in each scale.
    """
    sigma = np.median(np.sqrt(variance))
    # Create the wavelet coefficients for the significant pixels
    detect = np.sum(images, axis=0)
    _coeffs = starlet_transform(detect, scales=scales)
    support = get_multiresolution_support(
        image=detect,
        starlets=_coeffs,
        sigma=sigma,  # type: ignore
        sigma_scaling=3,
        epsilon=1e-1,
        max_iter=20,
    )
    return (support.support * _coeffs).astype(images.dtype)


@deprecated(
    reason="detect_footprints is replaced by detect_peaks and will be removed after v31.0. "
    "Use detect_peaks instead.",
    version="v31.0",
    category=FutureWarning,
)
def detect_footprints(
    images: np.ndarray,
    variance: np.ndarray,
    scales: int = 1,
    generation: int = 2,
    origin: tuple[int, int] | None = None,
    min_separation: float = 4,
    min_area: int = 4,
    peak_thresh: float = 5,
    footprint_thresh: float = 5,
    find_peaks: bool = True,
    remove_high_freq: bool = True,
    min_pixel_detect: int = 1,
) -> list[Footprint]:
    """Detect footprints in an image

    Parameters
    ----------
    images:
        The array of images with shape `(bands, Ny, Nx)` for which to
        calculate wavelet coefficients.
    variance:
        An array of variances with the same shape as `images`.
    scales:
        The maximum number of wavelet scales to use.
        If `remove_high_freq` is `False`, then this argument is ignored.
    generation:
        The generation of the starlet transform to use.
        If `remove_high_freq` is `False`, then this argument is ignored.
    origin:
        The location (y, x) of the lower corner of the image.
    min_separation:
        The minimum separation between peaks in pixels.
    min_area:
        The minimum area of a footprint in pixels.
    peak_thresh:
        The threshold for peak detection, in (approximate) sigma units.
        The detection image is calibrated empirically to have unit
        noise std for the LSST 6-band case with ``remove_high_freq=True``;
        for other band counts or wavelet settings the effective sigma
        differs (see the note on ``sigma`` in the function body and
        ticket DM-54860 for the full fix).
    footprint_thresh:
        The threshold for footprint detection. Same calibration caveat
        as ``peak_thresh``.
    find_peaks:
        If `True`, then detect peaks in the detection image,
        otherwise only the footprints are returned.
    remove_high_freq:
        If `True`, then remove high frequency wavelet coefficients
        before detecting peaks.
    min_pixel_detect:
        The minimum number of bands that must be above the
        detection threshold for a pixel to be included in a footprint.
    """

    if origin is None:
        origin = (0, 0)
    if remove_high_freq:
        # Build the wavelet coefficients
        wavelets = get_wavelets(
            images,
            variance,
            scales=scales,
            generation=generation,
        )
        # Remove the high frequency wavelets.
        # This has the effect of preventing high frequency noise
        # from interfering with the detection of peak positions.
        wavelets[0] = 0
        # Reconstruct the image from the remaining wavelet coefficients
        _images = multiband_starlet_reconstruction(
            wavelets,
            generation=generation,
        )
    else:
        _images = images
    # Build a SNR weighted detection image.
    #
    # The ``/ 2`` is an empirical calibration, not a noise estimate.
    # Zeroing the highest-frequency starlet scale and reconstructing
    # smooths the per-band image by ``h*h`` along each axis (gen-2
    # B-spline filter), reducing the per-pixel noise std by a factor
    # ``F = sum((h*h)**2) ~= 0.196``. The principled per-band sigma is
    # therefore ``median(sqrt(variance)) * F * sqrt(N_bands)``; for the
    # LSST 6-band case that's ``~ median(sqrt(variance)) * 0.481``,
    # which the ``/ 2`` (i.e. ``* 0.5``) approximates within ~4%, so
    # ``peak_thresh=5`` corresponds to a real ~5 sigma peak. For other
    # band counts or with ``remove_high_freq=False`` this approximation
    # is wrong by a band-count-dependent factor.
    #
    # TODO: DM-54860 for the full fix (proper chi^2 coadd + an
    # analytic noise calibration), which is deferred because the
    # detection code is not used in the LSST science pipelines
    # production runs.
    sigma = np.median(np.sqrt(variance), axis=(1, 2)) / 2
    detection = np.sum(_images / sigma[:, None, None], axis=0)
    if min_pixel_detect > 1:
        mask = np.sum(images > 0, axis=0) >= min_pixel_detect
        detection[~mask] = 0
    # Detect peaks on the detection image
    footprints = get_footprints(
        detection,
        min_separation,
        min_area,
        peak_thresh,
        footprint_thresh,
        find_peaks,
        origin[0],
        origin[1],
    )

    return footprints


# Record type for a peak candidate
CANDIDATE_DTYPE = np.dtype(
    [
        ("y", int),
        ("x", int),
        ("band", int),
        ("scale", int),
        ("flux", float),
    ]
)


@dataclass
class PeakCandidateResult:
    """Peak candidates and the products used to detect them.

    Attributes
    ----------
    candidates :
        Structured array of peak candidates with dtype `CANDIDATE_DTYPE`.
        The same source is expected to appear multiple times, once per band
        and scale it is significant in.
    significance_map :
        The per-scale detection map from ``_build_significance_map``, with
        shape ``(n_scales, n_bands + 1, Ny, Nx)``. Single-band planes are in
        sigma; the final plane is the chi coadd.
    starlets :
        The multiband starlet coefficients, with shape
        ``(scales + 1, n_bands, Ny, Nx)``.
    sigma :
        The per-scale, per-band coefficient noise std, with shape
        ``(scales + 1, n_bands)``.
    """

    candidates: np.ndarray
    significance_map: np.ndarray
    starlets: np.ndarray
    sigma: np.ndarray


def _starlet_scale_factors(scales: int, generation: int = 2) -> np.ndarray:
    """Sum of squared effective-kernel weights at each starlet scale.

    Parameters
    ----------
    scales :
        The number of wavelet scales. The result has ``scales + 1`` entries
        to account for the coarse scale.
    generation :
        The generation of the starlet transform, either ``1`` or ``2``.

    Returns
    -------
    factors :
        The factor ``F_j = sum(K_j**2)`` for each scale, where ``K_j`` is the
        effective a trous kernel. White noise of variance ``v`` transforms to
        coefficients of variance ``F_j * v`` at scale ``j``.

    Notes
    -----
    Because starlets change the scale of the data, we have to change the
    scale of the variance by a related factor in order to properly calculate
    the significance of detections. This algorithm computes the sum of squared
    effective-kernel weights at each starlet scale.

    The factors are read off a delta image transformed through the same fast
    a trous transform, so they match its generation and boundary handling.
    """
    size = 2 ** (scales + 3) + 1
    delta = np.zeros((size, size))
    delta[size // 2, size // 2] = 1.0
    coeffs = starlet_transform(delta, scales=scales, generation=generation)
    return np.sum(coeffs**2, axis=(1, 2))


def _build_detection_starlets(
    images: np.ndarray,
    variance: np.ndarray,
    scales: int = 3,
    generation: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a multiband image and estimate its per-scale noise.

    Parameters
    ----------
    images :
        The multiband image with shape ``(n_bands, Ny, Nx)``.
    variance :
        The per-pixel variance, with the same shape as ``images``.
    scales :
        The maximum number of wavelet scales to use.
    generation :
        The generation of the starlet transform, either ``1`` or ``2``.

    Returns
    -------
    starlets :
        The multiband starlet coefficients with shape
        ``(scales + 1, n_bands, Ny, Nx)``.
    sigma :
        The coefficient noise std at each scale and band, with shape
        ``(scales + 1, n_bands)``.

    Raises
    ------
    ValueError
        Raised if ``images`` and ``variance`` differ in shape, or if
        ``images`` is not 3D.

    Notes
    -----
    The coefficient noise is propagated from a single variance per band,
    ``sigma_{j,b} = sqrt(F_j * nanmedian(var_b))``, where ``F_j`` is the
    per-scale factor from ``_starlet_scale_factors``. This treats the noise as
    stationary within a band, which is accurate when the variance plane is
    slowly varying and avoids convolving the variance with the full, dense,
    per-scale kernels.
    """
    if images.shape != variance.shape:
        raise ValueError("images and variance must have the same shape")
    if images.ndim != 3:
        raise ValueError("images and variance must be 3D (bands, Ny, Nx)")

    starlets = multiband_starlet_transform(images, scales=scales, generation=generation)
    # The transform caps the scale count at the image size, so read the
    # realized number of scales back off the coefficients.
    realized_scales = starlets.shape[0] - 1
    factors = _starlet_scale_factors(realized_scales, generation=generation)
    band_variance = np.nanmedian(variance, axis=(1, 2))
    sigma = np.sqrt(factors[:, None] * band_variance[None, :]).astype(starlets.dtype)
    return starlets, sigma


def _clipped_chi2_survival(chi2: np.ndarray | float, n_bands: int) -> np.ndarray | float:
    """Survival function of the clipped chi-squared coadd under noise.

    Parameters
    ----------
    chi2 :
        The clipped chi-squared coadd value(s); scalar or array.
    n_bands :
        The number of bands combined into ``chi2``.

    Returns
    -------
    survival :
        ``P(C > chi2)`` for pure noise, with the same shape as ``chi2``.

    Notes
    -----
    See ``_chi2_to_sigma`` for the derivation of this mixture of ``chi^2_k``
    survival functions weighted by the binomial coefficients ``C(n, k)/2**n``.
    """
    return sum(comb(n_bands, k) * 2.0 ** (-n_bands) * stats.chi2.sf(chi2, k) for k in range(1, n_bands + 1))


def _chi2_to_sigma(chi2: np.ndarray | float, n_bands: int) -> np.ndarray:
    """Convert a clipped chi-squared coadd to Gaussian sigma.

    Parameters
    ----------
    chi2 :
        The chi-squared coadd ``sum_b max(w_b, 0)**2`` of per-band
        coefficients standardized to unit noise variance. This can also be
        a single pixel, like the value of a peak, to detect its significance.
    n_bands :
        The number of bands combined into ``chi2``.

    Returns
    -------
    sigma :
        The equivalent Gaussian significance (upper-tail), with the same
        shape as ``chi2``.

    Notes
    -----
    For an unclipped chi-squared coadd with ``n_bands`` degrees of freedom,
    ``scipy.stats.chi2.sf(chi2, n)`` gives the probability that a pure-noise
    pixel exceeds ``chi2``. However, ``_build_detection_starlets`` clips each
    band's coefficients at zero before squaring to supress false detections
    from negative structure (wavelet sidelobes, oversubtracted sky, etc).
    This changes the noise distribution: each standardized coeffficent is in
    a Gaussian with mean 0 and unit variance, so in any band a noise pixel is
    clipped to zero with probability 1/2, and the number of bands that
    contribute to the sum is binomially distributed. If ``k`` bands contribute,
    the sum is a ``chi^2_k`` deviate, so the probability that a noise pixel
    exceeds ``chi2`` is the sum over ``k`` of ``stats.chi2.sf(chi2, k)``
    weighted by the binomial probability ``C(n, k) / 2**n``. That probability
    is converted to an equivalent Gaussian sigma with ``stats.norm.isf`` so
    that thresholds are directly comparable to a single-band ``n``-sigma cut.
    """
    # Floor the survival function to keep saturated cores finite. The Gaussian
    # inverse saturates near 37 sigma, so this mapping is for reporting and
    # display only; peaks are detected on the chi statistic itself
    # (see `_find_peak_candidates`), which stays strictly monotonic.
    survival = np.clip(_clipped_chi2_survival(chi2, n_bands), np.finfo(float).tiny, 1.0)
    return stats.norm.isf(survival)


def _sigma_to_chi2(sigma: float, n_bands: int) -> float:
    """Convert a Gaussian sigma threshold to a chi-squared coadd threshold.

    Parameters
    ----------
    sigma :
        The desired detection threshold in Gaussian sigma (upper-tail).
    n_bands :
        The number of bands combined into the chi-squared coadd.

    Returns
    -------
    chi2 :
        The chi-squared coadd value with the same upper-tail probability as
        ``sigma``, i.e. the inverse of ``_chi2_to_sigma``.

    Notes
    -----
    The clipped chi-squared survival function (see ``_chi2_to_sigma``)
    decreases monotonically, so the matching ``chi2`` is found by bracketing
    the target upper-tail probability and refining with a root search.
    Inverting the threshold once avoids mapping every pixel through
    ``_chi2_to_sigma``, along with that mapping's saturation in bright cores.
    """
    target = stats.norm.sf(sigma)
    hi = 2.0
    # Brent's method requires an interval [a, b] where the function changes
    # sign, so the root is guaranteed to lie between 0 and `hi`.
    # This loop ensures that the upper bound `hi` is large enough so that
    # the surivival function at `hi` is below the target sigma.
    while _clipped_chi2_survival(hi, n_bands) > target and hi < 1e12:
        hi *= 2
    return float(brentq(lambda chi2: _clipped_chi2_survival(chi2, n_bands) - target, 0.0, hi))


def _build_significance_map(
    starlets: np.ndarray,
    sigma: np.ndarray,
    first_scale: int = 1,
) -> np.ndarray:
    """Build a per-scale detection map from multiband starlet coefficients.

    Parameters
    ----------
    starlets :
        The multiband starlet coefficients with shape
        ``(scales + 1, n_bands, Ny, Nx)``.
    sigma :
        The per-scale, per-band coefficient noise std with shape
        ``(scales + 1, n_bands)``, from ``_build_detection_starlets``.
    first_scale :
        The first starlet scale to keep. Scales below this (the highest
        frequencies) and the final residual scale are dropped.

    Returns
    -------
    significance_map :
        The detection map with shape ``(n_scales, n_bands + 1, Ny, Nx)``,
        where ``n_scales = scales - first_scale``. The first ``n_bands``
        planes are the standardized single-band coefficients, in units of
        Gaussian sigma. The last plane is the clipped chi coadd
        ``sqrt(sum_b max(w_b, 0)**2)``.

    Notes
    -----
    We clip the standardized single-band coefficients at zero before computing
    the chi coadd, to avoid negative contributions. This adds computational
    complexity (see `chi2_t_sigma`) at the benefit of reducing our false
    positive detections.
    """
    _, n_bands, height, width = starlets.shape
    scale_slice = slice(first_scale, -1)
    coeffs = starlets[scale_slice]
    scale_sigma = sigma[scale_slice]
    n_scales = coeffs.shape[0]

    significance_map = np.zeros((n_scales, n_bands + 1, height, width), dtype=np.float32)
    standardized = coeffs / scale_sigma[..., None, None]
    significance_map[:, :n_bands] = standardized
    significance_map[:, n_bands] = np.sqrt(np.sum(np.clip(standardized, 0, None) ** 2, axis=1))
    return significance_map


def _find_peak_candidates(
    significance_map: np.ndarray,
    min_separation: float = 0,
    min_area: int = 4,
    peak_thresh: float = 3,
    footprint_thresh: float = 2,
    first_scale: int = 1,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Find peak candidates in each plane of a significance map.

    Parameters
    ----------
    significance_map :
        A detection map from ``_build_significance_map``. The single-band
        planes are in sigma and the final plane is the chi coadd.
    min_separation :
        The minimum separation between peaks in pixels.
        By default there is no minimum separation, relying on our linking
        procedure to merge detections within the same band and scale as
        well across other bands and scales.
    min_area :
        The minimum area of a footprint in pixels.
    peak_thresh :
        The peak detection threshold, in sigma.
    footprint_thresh :
        The footprint detection threshold, in sigma.
    first_scale :
        The first starlet scale in ``significance_map``, used to label the
        ``scale`` field of each candidate.
    origin :
        The ``(y, x)`` location of the lower corner of the image.

    Returns
    -------
    candidates :
        Structured array of candidates with dtype `CANDIDATE_DTYPE`. The
        ``flux`` field is the peak significance in sigma for every plane.
    """
    y0, x0 = origin
    n_bands = significance_map.shape[1] - 1
    # The chi plane is thresholded in chi units so its bright cores stay
    # strictly peaked; single-band planes are already in sigma.
    chi_peak = np.sqrt(_sigma_to_chi2(peak_thresh, n_bands))
    chi_footprint = np.sqrt(_sigma_to_chi2(footprint_thresh, n_bands))

    candidates = []
    for scale_index, scale_planes in enumerate(significance_map):
        scale = scale_index + first_scale
        for band, plane in enumerate(scale_planes):
            is_chi = band == n_bands
            footprints = get_footprints(
                plane,
                min_separation,
                min_area,
                chi_peak if is_chi else peak_thresh,
                chi_footprint if is_chi else footprint_thresh,
                True,
                y0,
                x0,
            )
            for footprint in footprints:
                for peak in footprint.peaks:
                    # Report every plane's peak in sigma; the chi value maps
                    # back through the coadd survival function.
                    flux = float(_chi2_to_sigma(peak.flux**2, n_bands)) if is_chi else peak.flux
                    candidates.append((peak.y, peak.x, band, scale, flux))
    return np.array(candidates, dtype=CANDIDATE_DTYPE)


def detect_peaks(
    images: np.ndarray,
    variance: np.ndarray,
    scales: int = 3,
    generation: int = 2,
    first_scale: int = 1,
    origin: tuple[int, int] | None = None,
    min_separation: float = 0,
    min_area: int = 4,
    peak_thresh: float = 3,
    footprint_thresh: float = 2,
) -> PeakCandidateResult:
    """Detect peak candidates across bands and starlet scales.

    Parameters
    ----------
    images :
        The multiband image with shape ``(n_bands, Ny, Nx)``.
    variance :
        The per-pixel variance, with the same shape as ``images``.
    scales :
        The maximum number of wavelet scales to use.
    generation :
        The generation of the starlet transform, either ``1`` or ``2``.
    first_scale :
        The first starlet scale to search for peaks.
    origin :
        The ``(y, x)`` location of the lower corner of the image.
    min_separation :
        The minimum separation between peaks in pixels.
    min_area :
        The minimum area of a footprint in pixels.
    peak_thresh :
        The peak detection threshold, in sigma.
    footprint_thresh :
        The footprint detection threshold, in sigma.

    Returns
    -------
    result :
        The peak candidates and the intermediate detection products.
    """
    if origin is None:
        origin = (0, 0)
    starlets, sigma = _build_detection_starlets(
        images,
        variance,
        scales=scales,
        generation=generation,
    )
    significance_map = _build_significance_map(starlets, sigma, first_scale=first_scale)
    candidates = _find_peak_candidates(
        significance_map,
        min_separation,
        min_area,
        peak_thresh,
        footprint_thresh,
        first_scale,
        origin,
    )
    return PeakCandidateResult(
        candidates=candidates,
        significance_map=significance_map,
        starlets=starlets,
        sigma=sigma,
    )
