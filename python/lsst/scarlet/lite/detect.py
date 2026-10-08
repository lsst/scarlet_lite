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
from math import comb, log
from typing import Sequence, cast

import numpy as np
from deprecated.sphinx import deprecated  # type: ignore
from lsst.scarlet.lite.detect_pybind11 import Footprint, Peak, get_footprints  # type: ignore
from scipy import special
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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


@deprecated(
    reason="get_wavelets is only used by the deprecated detect_footprints "
    "and will be removed after v31.0.",
    version="v31.0",
    category=FutureWarning,
)
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


@deprecated(
    reason="get_detect_wavelets is superseded by detect_peaks and will be " "removed after v31.0.",
    version="v31.0",
    category=FutureWarning,
)
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


# Record type for a peak candidate: one local maximum in one plane
# (band, scale) of the significance map that survived the plane's
# contrast-limited watershed.
CANDIDATE_DTYPE = np.dtype(
    [
        ("y", int),
        ("x", int),
        ("band", int),
        ("scale", int),
        ("flux", float),
        ("saddle", float),
        ("position", int),
    ]
)

# Record type for a unique candidate position, the intermediate between the
# candidates and the final peaks.
POSITION_DTYPE = np.dtype(
    [
        ("y", int),
        ("x", int),
        ("scale", int),
        ("flux", float),
        ("peak_sigma", float),
        ("band_flags", int),
        ("scale_flags", int),
        ("plane_flags", np.int64),
        ("n_candidates", int),
        ("n_linked", int),
        ("peak", int),
        ("component", int),
        ("ambiguous", bool),
        ("max_linked_fraction", float),
        ("max_linked_peak", int),
    ]
)

# Record type for a final detection, one row per source.
DETECTION_DTYPE = np.dtype(
    [
        ("y", int),
        ("x", int),
        ("scale", int),
        ("flux", float),
        ("peak_sigma", float),
        ("band_flags", int),
        ("scale_flags", int),
        ("n_candidates", int),
        ("n_positions", int),
        ("n_ambiguous", int),
        ("footprint", int),
        ("location_fallback", bool),
        ("search_capped", bool),
        ("nearest_distance", float),
        ("nearest_distance_fwhm", float),
        ("stat_err_y", float),
        ("stat_err_x", float),
        ("moment_yy", float),
        ("moment_xx", float),
        ("moment_xy", float),
        ("scatter_y", float),
        ("scatter_x", float),
        ("err_y", float),
        ("err_x", float),
    ]
)

# Record type for a pair of positions within the largest link radius, linked
# or not.
PAIR_DTYPE = np.dtype(
    [
        ("i", int),
        ("j", int),
        ("distance", float),
        ("linked", bool),
        ("shares_plane", bool),
        ("within_min", bool),
        ("within_max", bool),
    ]
)

# Record type for a connected component of the position graph.
COMPONENT_DTYPE = np.dtype(
    [
        ("n_nodes", int),
        ("lower_bound", int),
        ("upper_bound", int),
        ("k", int),
        ("search_ran", bool),
        ("n_visited", int),
        ("cap_hit", bool),
        ("n_min_covers", int),
    ]
)

# The PAIR_DTYPE field that decides whether a pair is close enough to link,
# for each value of ``radius_rule``.
RADIUS_RULES = {"min": "within_min", "max": "within_max"}

# Minimum covers are only counted for components this small, and at most this
# many of them.
COUNT_COVERS_MAX_NODES = 12
COUNT_COVERS_MAX = 100
SEARCH_CAP = 100_000

SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
MAD_TO_SIGMA = 1.4826


@dataclass
class PeakDetectionResult:
    """Detected peaks and the products used to detect them.

    Attributes
    ----------
    peaks :
        Structured array of detections with dtype `DETECTION_DTYPE`, one row
        per source. Row ids link back to ``positions`` via
        ``positions["peak"]``; ``footprint`` is the index into
        ``footprints`` of the footprint containing the peak. The location,
        flags and errors are described in ``_build_detections``.
    footprints :
        The detected footprints, each holding the `Peak` objects of the
        detections inside it, brightest first. Every peak lies in exactly
        one footprint and every footprint holds at least one peak (see
        ``_build_footprints``).
    positions :
        Structured array of unique candidate positions with dtype
        `POSITION_DTYPE`, the intermediate between candidates and peaks. Row
        ids link back to ``candidates`` via ``candidates["position"]``.
        ``plane_flags`` is a bitmask of the planes (band, scale) the position
        was a distinct peak in. ``peak`` and ``component`` index ``peaks``
        and ``components``; ``n_linked``, ``ambiguous`` and the
        ``max_linked_*`` fields describe the links to other peaks (see
        ``_link_fractions``).
    candidates :
        Structured array of peak candidates with dtype `CANDIDATE_DTYPE`.
        The same source is expected to appear multiple times, once per band
        and scale it is significant in. ``saddle`` is the level at which the
        candidate's basin joined a brighter candidate's basin in its plane
        (NaN for the brightest candidate in a footprint).
    significance_map :
        The per-scale detection map from ``_build_significance_map``, with
        shape ``(n_scales, n_bands + 1, Ny, Nx)``. Every plane is in units of
        Gaussian sigma; the final plane is the clipped chi coadd mapped to
        sigma.
    starlets :
        The multiband starlet coefficients, with shape
        ``(scales + 1, n_bands, Ny, Nx)``.
    sigma :
        The coefficient noise std, shaped to broadcast against ``starlets``.
        See ``_build_detection_starlets`` for the two shapes it takes.
    pairs :
        Structured array of position pairs with dtype `PAIR_DTYPE`. ``i`` and
        ``j`` index ``positions``.
    components :
        Structured array of the connected components of the position graph
        with dtype `COMPONENT_DTYPE` (see ``_clique_cover``).
    metadata :
        The settings the peaks were grouped with: the per-band
        ``psf_fwhm``, the ``link_radius`` at each of ``link_radius_scales``,
        ``min_separation``, ``radius_rule`` and ``search_cap``.
    """

    peaks: np.ndarray
    footprints: list[Footprint]
    positions: np.ndarray
    candidates: np.ndarray
    significance_map: np.ndarray
    starlets: np.ndarray
    sigma: np.ndarray
    pairs: np.ndarray
    components: np.ndarray
    metadata: dict


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
    variance_mode: str = "median",
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
    variance_mode :
        How ``variance`` is reduced to a coefficient noise. ``"median"`` takes
        one value per band, ``"pixel"`` keeps the variance plane per pixel.

    Returns
    -------
    starlets :
        The multiband starlet coefficients with shape
        ``(scales + 1, n_bands, Ny, Nx)``.
    sigma :
        The coefficient noise std at each scale and band, shaped to broadcast
        against ``starlets``: ``(scales + 1, n_bands, 1, 1)`` for
        ``variance_mode="median"`` and ``(scales + 1, n_bands, Ny, Nx)`` for
        ``variance_mode="pixel"``.

    Raises
    ------
    ValueError
        Raised if ``images`` and ``variance`` differ in shape, if ``images`` is
        not 3D, or if ``variance_mode`` is not one of the two allowed values.

    Notes
    -----
    The coefficient noise is ``sigma_{j,b} = sqrt(F_j * var_b)``, where ``F_j``
    is the per-scale factor from ``_starlet_scale_factors``.

    With ``variance_mode="median"`` the variance is a single
    ``nanmedian(var_b)`` per band, which treats the noise as stationary within
    a band. With ``"pixel"`` the variance plane is used as it stands, which
    follows a varying depth at the memory cost of a ``sigma`` that is as
    large as ``starlets``.

    Neither is the exact coefficient noise, which is ``var_b`` convolved with
    the square of the effective kernel at each scale. Both approximate that by
    the scalar ``F_j``, so ``"pixel"`` is the better estimate only where the
    variance is smooth across a kernel, and the kernel doubles in width with
    every scale.
    """
    if images.shape != variance.shape:
        raise ValueError("images and variance must have the same shape")
    if images.ndim != 3:
        raise ValueError("images and variance must be 3D (bands, Ny, Nx)")
    if variance_mode not in ("median", "pixel"):
        raise ValueError(f"variance_mode must be 'median' or 'pixel', not {variance_mode!r}")

    starlets = multiband_starlet_transform(images, scales=scales, generation=generation)
    # The transform caps the scale count at the image size, so read the
    # realized number of scales back off the coefficients.
    realized_scales = starlets.shape[0] - 1
    factors = _starlet_scale_factors(realized_scales, generation=generation)
    if variance_mode == "median":
        band_variance = np.nanmedian(variance, axis=(1, 2))[:, None, None]
    else:
        band_variance = variance
    sigma = np.sqrt(factors[:, None, None, None] * band_variance[None]).astype(starlets.dtype)
    return starlets, sigma


def _log_gammaincc(a: float, z: np.ndarray) -> np.ndarray:
    """Log of the regularized upper incomplete gamma function ``Q(a, z)``.

    Parameters
    ----------
    a :
        The shape parameter, ``a > 0``.
    z :
        The lower limit(s) of integration, ``z >= 0``.

    Returns
    -------
    log_q :
        ``log(Q(a, z))``, with the same shape as ``z``.

    Notes
    -----
    ``scipy.special.gammaincc`` underflows to zero for ``z`` above a few
    hundred, which would map every sufficiently bright pixel to the same
    saturated significance. Above ``z = 500`` this uses the asymptotic
    expansion
    ``Q(a, z) ~ z**(a-1) exp(-z) / Gamma(a) * sum_m prod_{i<=m}(a-i) / z**m``,
    whose truncation error at ``z >= 500`` is far below double precision for
    any ``a`` of interest here (``a = k/2`` for ``k`` bands).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape, dtype=float)
    small = z < 500
    out[small] = np.log(special.gammaincc(a, z[small]))
    zl = z[~small]
    if zl.size:
        series = np.ones_like(zl)
        term = np.ones_like(zl)
        for m in range(1, 8):
            term = term * (a - m) / zl
            series += term
        out[~small] = (a - 1) * np.log(zl) - zl - special.gammaln(a) + np.log(series)
    return out


def _chi2_log_survival(chi2: np.ndarray | float, n_bands: int) -> np.ndarray:
    """Log survival function of the clipped chi-squared coadd under noise.

    Parameters
    ----------
    chi2 :
        The clipped chi-squared coadd value(s); scalar or array.
    n_bands :
        The number of bands combined into ``chi2``.

    Returns
    -------
    log_survival :
        ``log P(C > chi2)`` for pure noise, with the shape of ``chi2``.

    Notes
    -----
    See ``_chi2_to_sigma`` for the derivation of this mixture of ``chi^2_k``
    survival functions weighted by the binomial coefficients ``C(n, k)/2**n``.
    The mixture is evaluated in log space so it stays finite for arbitrarily
    bright pixels; ``chi2_k.sf(x) = Q(k/2, x/2)``.
    """
    z = np.asarray(chi2, dtype=float) / 2
    log_norm = -n_bands * log(2)
    terms = np.stack(
        [log(comb(n_bands, k)) + log_norm + _log_gammaincc(k / 2, z) for k in range(1, n_bands + 1)]
    )
    return special.logsumexp(terms, axis=0)


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
    pixel exceeds ``chi2``. However, ``_build_significance_map`` clips each
    band's coefficients at zero before squaring to supress false detections
    from negative structure (wavelet sidelobes, oversubtracted sky, etc).
    This changes the noise distribution: each standardized coeffficent is in
    a Gaussian with mean 0 and unit variance, so in any band a noise pixel is
    clipped to zero with probability 1/2, and the number of bands that
    contribute to the sum is binomially distributed. If ``k`` bands contribute,
    the sum is a ``chi^2_k`` deviate, so the probability that a noise pixel
    exceeds ``chi2`` is the sum over ``k`` of ``chi2_k.sf(chi2)`` weighted by
    the binomial probability ``C(n, k) / 2**n``. That probability is converted
    to an equivalent Gaussian sigma so that thresholds are directly comparable
    to a single-band ``n``-sigma cut.

    Due to divergence for large ``chi2``, instead of using scipy.stats.chi2
    function directly, we calculate in log space (``_chi2_log_survival`` and
    ``scipy.special.ndtri_exp``), so the mapping is strictly monotonic with no
    saturation, however bright the pixel.
    """
    return -special.ndtri_exp(_chi2_log_survival(chi2, n_bands))


def _chi_to_sigma(chi: np.ndarray, n_bands: int) -> np.ndarray:
    """Map a clipped chi coadd plane to Gaussian sigma through a lookup table.

    Parameters
    ----------
    chi :
        The chi coadd ``sqrt(sum_b max(w_b, 0)**2)``; any shape.
    n_bands :
        The number of bands combined into ``chi``.

    Returns
    -------
    sigma :
        ``_chi2_to_sigma(chi**2, n_bands)`` with the shape and dtype of
        ``chi``, evaluated by linear interpolation of a table.

    Notes
    -----
    ``chi -> sigma`` is a smooth, monotonic, one-dimensional function for a
    fixed band count, so evaluating it on a few thousand grid points and
    interpolating is far cheaper than evaluating the survival-function mixture
    per pixel, and accurate to well below 1e-4 sigma. The grid is dense where
    the function has curvature (``chi < 64``) and geometric beyond, where
    ``sigma`` approaches ``chi`` minus a slowly varying offset.
    """
    chi_max = float(np.nanmax(chi)) if chi.size else 1.0
    n_dense = 4096
    dense_max = 64.0
    grid = np.linspace(0.0, dense_max, n_dense + 1)
    table = _chi2_to_sigma(grid**2, n_bands)

    # The dense grid is uniform, so interpolate by direct indexing (a bin
    # search per pixel with np.interp is ~10x slower on a full image).
    invalid = np.isnan(chi)
    t = np.clip(chi, 0, dense_max) * (n_dense / dense_max)
    if invalid.any():
        t[invalid] = 0.0
    idx = t.astype(np.intp)
    np.minimum(idx, n_dense - 1, out=idx)
    frac = t - idx
    sigma = table[idx]
    sigma += frac * (table[idx + 1] - sigma)
    if invalid.any():
        sigma[invalid] = np.nan

    if chi_max > dense_max:
        # Bright pixels are rare; a geometric grid and np.interp are fine here.
        bright = chi > dense_max
        grid_hi = np.geomspace(dense_max, chi_max * 1.01, 513)
        table_hi = _chi2_to_sigma(grid_hi**2, n_bands)
        sigma[bright] = np.interp(chi[bright], grid_hi, table_hi)
    return sigma.astype(chi.dtype, copy=False)


def build_chi2_significance(
    standardized: np.ndarray,
) -> np.ndarray:
    """Coadd standardized single-band planes into a map of Gaussian sigma.

    Parameters
    ----------
    standardized :
        The single-band planes with shape ``(n_bands, Ny, Nx)``, each
        standardized to unit noise variance. These are starlet coefficients
        divided by their per-scale noise std, or an image divided by the
        square root of its variance.

    Returns
    -------
    significance :
        The coadd with shape ``(Ny, Nx)``, in units of Gaussian sigma.

    Notes
    -----
    The standardized single-band planes are clipped at zero before computing
    the chi coadd, to avoid negative contributions. This changes the noise
    distribution of the coadd (see ``_chi2_to_sigma``), which is why the coadd
    is mapped to sigma rather than used directly.
    """
    chi = np.sqrt(np.sum(np.clip(standardized, 0, None) ** 2, axis=0))
    return _chi_to_sigma(chi, standardized.shape[0])


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
        The coefficient noise std from ``_build_detection_starlets``, shaped
        to broadcast against ``starlets``.
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
        ``sqrt(sum_b max(w_b, 0)**2)`` mapped to Gaussian sigma with
        ``_chi_to_sigma``, so every plane is on the same footing and a
        single threshold or contrast applies to all of them.
    """
    _, n_bands, height, width = starlets.shape
    scale_slice = slice(first_scale, -1)
    coeffs = starlets[scale_slice]
    scale_sigma = sigma[scale_slice]
    n_scales = coeffs.shape[0]

    significance_map = np.zeros((n_scales, n_bands + 1, height, width), dtype=np.float32)
    standardized = coeffs / scale_sigma
    significance_map[:, :n_bands] = standardized
    for i in range(n_scales):
        significance_map[i, n_bands] = build_chi2_significance(standardized[i])
    return significance_map


def _dilate_bad_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    """Grow a bad-pixel mask by a circular structuring element.

    Parameters
    ----------
    mask :
        A 2D boolean mask, `True` where a pixel is bad.
    radius :
        The dilation radius in pixels.

    Returns
    -------
    dilated :
        The dilated mask. ``mask`` is returned unchanged when ``radius`` is
        not positive or no pixel is set.

    Notes
    -----
    Starlet ringing around a bad region reaches beyond the flagged pixels, so
    the peaks it seeds have to be suppressed over a slightly wider region than
    the mask itself.
    """
    if radius <= 0 or not mask.any():
        return mask
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    element = yy**2 + xx**2 <= radius**2
    return binary_dilation(mask, structure=element)


def _find_peak_candidates(
    significance_map: np.ndarray,
    min_separation: float = 0,
    min_area: int = 4,
    peak_thresh: float = 3,
    footprint_thresh: float = 2,
    kappa: float = 3,
    first_scale: int = 1,
    origin: tuple[int, int] = (0, 0),
    bad_pixel_mask: np.ndarray | None = None,
    dilation_radius: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find peak candidates in each plane of a significance map.

    Parameters
    ----------
    significance_map :
        A detection map from ``_build_significance_map``, with every plane
        in sigma.
    min_separation :
        A hard floor on the separation between peaks in a plane, in pixels.
        Peaks closer than this to a brighter peak are dropped regardless of
        the contrast test; ``0`` disables it.
    min_area :
        The minimum area of a footprint in pixels.
    peak_thresh :
        The peak detection threshold, in sigma.
    footprint_thresh :
        The footprint detection threshold, in sigma.
    kappa :
        The minimum prominence of a peak, in sigma: the peak must rise at
        least ``kappa`` above the saddle connecting it to any brighter peak
        in its footprint, otherwise it is a bump on that peak's flank and is
        culled. This is what makes two candidates in the same plane distinct
        sources (see ``_build_pairs``).
    first_scale :
        The first starlet scale in ``significance_map``, used to label the
        ``scale`` field of each candidate.
    origin :
        The ``(y, x)`` location of the lower corner of the image.
    bad_pixel_mask :
        Per-band boolean mask with shape ``(n_bands, Ny, Nx)``, `True` where a
        band is bad. Peaks inside the dilated mask of their band are dropped;
        the chi coadd plane (``band == n_bands``) uses the union across bands.
        `None` disables the suppression.
    dilation_radius :
        The base radius in pixels by which ``bad_pixel_mask`` is grown before
        it suppresses peaks, to catch the starlet ringing just outside a bad
        region. The starlet kernel doubles in width with each scale, so the
        radius applied at a plane is ``dilation_radius * 2 ** (scale -
        first_scale)``.

    Returns
    -------
    candidates :
        Structured array of candidates with dtype `CANDIDATE_DTYPE`. The
        ``flux`` field is the peak significance in sigma for every plane.
    footprint_mask :
        Boolean image with the spatial shape of ``significance_map`` that is
        `True` in every pixel of any plane's footprint that has a peak.
    """
    y0, x0 = origin
    height, width = significance_map.shape[-2:]
    n_bands = significance_map.shape[1] - 1
    footprint_mask = np.zeros((height, width), dtype=bool)
    # Dilated bad mask per (band, scale), filled lazily.
    # The ring widens with scale, so each scale grows the mask by its own
    # radius. The chi coadd plane is the union of all bands.
    dilated_masks: dict[tuple[int, int], np.ndarray] = {}
    candidates = []
    for scale_index, scale_planes in enumerate(significance_map):
        scale = scale_index + first_scale
        radius = dilation_radius * 2 ** (scale - first_scale)
        for band, plane in enumerate(scale_planes):
            bad = None
            if bad_pixel_mask is not None:
                if (band, scale) not in dilated_masks:
                    raw = bad_pixel_mask.any(axis=0) if band == n_bands else bad_pixel_mask[band]
                    dilated_masks[(band, scale)] = _dilate_bad_mask(raw, radius)
                bad = dilated_masks[(band, scale)]
            footprints = get_footprints(
                plane,
                min_separation,
                min_area,
                peak_thresh,
                footprint_thresh,
                True,
                y0,
                x0,
                kappa,
            )
            for footprint in footprints:
                bottom, top, left, right = footprint.bounds
                footprint_mask[bottom - y0 : top - y0 + 1, left - x0 : right - x0 + 1] |= footprint.data
                for peak in footprint.peaks:
                    # Do not add peaks that fall on bad pixels.
                    if bad is not None and bad[peak.y - y0, peak.x - x0]:
                        continue
                    # position is -1 until _collapse_positions fills it.
                    candidates.append((peak.y, peak.x, band, scale, peak.flux, peak.saddle, -1))
    return np.array(candidates, dtype=CANDIDATE_DTYPE), footprint_mask


def _plane_flags(candidates: np.ndarray, n_bands: int, first_scale: int) -> np.ndarray:
    """One bit per plane ``(band, scale)`` for each candidate.

    Parameters
    ----------
    candidates :
        Structured array of candidates with dtype `CANDIDATE_DTYPE`.
    n_bands :
        The number of single-band planes (the chi plane is ``band ==
        n_bands``).
    first_scale :
        The first starlet scale in the significance map.

    Returns
    -------
    flags :
        ``1 << plane`` for each candidate, where
        ``plane = (scale - first_scale) * (n_bands + 1) + band``.

    Raises
    ------
    ValueError
        Raised if there are more than 63 planes.
    """
    n_planes = int(candidates["scale"].max() - first_scale + 1) * (n_bands + 1) if len(candidates) else 0
    if n_planes > 63:
        raise ValueError(f"plane_flags supports at most 63 planes, got {n_planes}")
    plane = (candidates["scale"] - first_scale) * (n_bands + 1) + candidates["band"]
    return np.left_shift(np.int64(1), plane.astype(np.int64))


def _collapse_positions(candidates: np.ndarray, n_bands: int, first_scale: int) -> np.ndarray:
    """Collapse candidates to unique positions, filling their ``position``.

    Parameters
    ----------
    candidates :
        Structured array of candidates with dtype `CANDIDATE_DTYPE`. The
        ``position`` field is filled in place.
    n_bands :
        The number of single-band planes.
    first_scale :
        The first starlet scale in the significance map.

    Returns
    -------
    positions :
        The unique positions with dtype `POSITION_DTYPE`, sorted by
        ``(y, x)``. The grouping fields are left for ``_group_peaks``.

    Notes
    -----
    Everything here is a grouped reduction over the candidates, done with
    ``np.ufunc.at`` and one lexsort rather than a loop over positions, so
    the cost is linear in the number of candidates.
    """
    yx = np.column_stack([candidates["y"], candidates["x"]])
    unique_yx, inverse, counts = np.unique(yx, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    n = len(unique_yx)
    candidates["position"] = inverse

    positions = np.zeros(n, dtype=POSITION_DTYPE)
    positions["y"] = unique_yx[:, 0]
    positions["x"] = unique_yx[:, 1]
    positions["n_candidates"] = counts
    positions["peak"] = -1

    band_flags = np.zeros(n, dtype=np.int64)
    np.bitwise_or.at(band_flags, inverse, np.left_shift(np.int64(1), candidates["band"].astype(np.int64)))
    scale_flags = np.zeros(n, dtype=np.int64)
    np.bitwise_or.at(scale_flags, inverse, np.left_shift(np.int64(1), candidates["scale"].astype(np.int64)))
    plane_flags = np.zeros(n, dtype=np.int64)
    np.bitwise_or.at(plane_flags, inverse, _plane_flags(candidates, n_bands, first_scale))
    positions["band_flags"] = band_flags
    positions["scale_flags"] = scale_flags
    positions["plane_flags"] = plane_flags

    scale = np.full(n, np.iinfo(int).max, dtype=int)
    np.minimum.at(scale, inverse, candidates["scale"])
    positions["scale"] = scale
    peak_sigma = np.full(n, -np.inf)
    np.maximum.at(peak_sigma, inverse, candidates["flux"])
    positions["peak_sigma"] = peak_sigma

    # flux is the brightest candidate at the finest scale of each position:
    # sort by (position, scale, -flux) and take the first row per position.
    order = np.lexsort((-candidates["flux"], candidates["scale"], inverse))
    sorted_inverse = inverse[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
    positions["flux"][sorted_inverse[first]] = candidates["flux"][order][first]
    return positions


def _link_radius(scale: int | np.ndarray, psf_fwhm: float) -> float | np.ndarray:
    """Linking radius for a position at a given starlet scale.

    Parameters
    ----------
    scale :
        The absolute starlet scale of the position (scalar or array).
    psf_fwhm :
        The PSF FWHM in pixels, used as a floor at the finest scales.

    Returns
    -------
    radius :
        The radius in pixels, ``max(psf_fwhm, 2**scale)``.
    """
    return np.maximum(psf_fwhm, 2.0 ** np.asarray(scale, dtype=float))


def _starlet_kernel_fwhm(psf_fwhm: float | np.ndarray, scale: int | np.ndarray) -> float | np.ndarray:
    """Effective FWHM of a point source at a starlet scale.

    Parameters
    ----------
    psf_fwhm :
        The PSF FWHM in pixels (scalar or array).
    scale :
        The absolute starlet scale (scalar or array).

    Returns
    -------
    fwhm :
        The FWHM in pixels of the PSF convolved with the smoothing kernel that
        precedes ``scale``.

    Notes
    -----
    Scale ``j`` is the difference between the image smoothed ``j`` times and
    its further smoothing. Pass ``i`` of the B3 spline has a variance of
    ``4**i`` per axis, so the smoothing before scale ``j`` has a variance of
    ``(4**j - 1) / 3``. Both kernels are treated as Gaussians and added in
    quadrature.
    """
    kernel_variance = (4.0 ** np.asarray(scale, dtype=float) - 1) / 3
    return np.sqrt(np.asarray(psf_fwhm, dtype=float) ** 2 + SIGMA_TO_FWHM**2 * kernel_variance)


def _build_pairs(positions: np.ndarray, psf_fwhm: float, radius_rule: str) -> np.ndarray:
    """Find every pair of positions within the largest link radius.

    Parameters
    ----------
    positions :
        The unique positions with dtype `POSITION_DTYPE`.
    psf_fwhm :
        The PSF FWHM in pixels, passed to ``_link_radius``.
    radius_rule :
        Which link radius of a pair bounds its separation, a key of
        `RADIUS_RULES`.

    Returns
    -------
    pairs :
        The pairs with dtype `PAIR_DTYPE`, sorted by ``(i, j)`` with
        ``i < j``.

    Notes
    -----
    Two positions are linked if they are within the link radius and share no
    plane (starlet scale and band). Positions that were both peaks in one
    plane were separated by that plane's contrast-limited watershed,
    so they are distinct sources however close they are.
    """
    if len(positions) < 2:
        return np.zeros(0, dtype=PAIR_DTYPE)
    yx = np.column_stack([positions["y"], positions["x"]]).astype(float)
    radii = cast(np.ndarray, _link_radius(positions["scale"], psf_fwhm))
    ij = cKDTree(yx).query_pairs(float(radii.max()), output_type="ndarray")
    ij = ij[np.lexsort((ij[:, 1], ij[:, 0]))]
    i, j = ij[:, 0], ij[:, 1]
    flags = positions["plane_flags"]

    pairs = np.zeros(len(ij), dtype=PAIR_DTYPE)
    pairs["i"] = i
    pairs["j"] = j
    pairs["distance"] = np.hypot(yx[i, 0] - yx[j, 0], yx[i, 1] - yx[j, 1])
    pairs["shares_plane"] = (flags[i] & flags[j]) != 0
    pairs["within_min"] = pairs["distance"] <= np.minimum(radii[i], radii[j])
    pairs["within_max"] = pairs["distance"] <= np.maximum(radii[i], radii[j])
    pairs["linked"] = pairs[RADIUS_RULES[radius_rule]] & ~pairs["shares_plane"]
    return pairs


def _full_cliques(neighbors: list[int], clique: list[int], sizes: list[int]) -> list[int]:
    """Cliques that a node is adjacent to every member of, in ascending order.

    Parameters
    ----------
    neighbors :
        The neighbors of the node.
    clique :
        The clique of every node, ``-1`` if unassigned.
    sizes :
        The number of members of each clique.

    Returns
    -------
    cliques :
        The cliques the node could join.
    """
    counts: dict[int, int] = {}
    for u in neighbors:
        g = clique[u]
        if g >= 0:
            counts[g] = counts.get(g, 0) + 1
    return sorted(g for g, count in counts.items() if count == sizes[g])


def _greedy_independent_set(adjacency: list[list[int]], order: list[int]) -> int:
    """Size of a greedy independent set, visiting nodes in ``order``.

    Parameters
    ----------
    adjacency :
        The neighbors of each node.
    order :
        The order in which nodes are offered to the set.

    Returns
    -------
    size :
        The number of nodes in the set.

    Notes
    -----
    Two adjacent nodes (positions) can never be in the same clique (peak),
    so the size of any independent set is a lower bound on the number of
    cliques.
    """
    chosen = [False] * len(adjacency)
    for v in order:
        if not any(chosen[u] for u in adjacency[v]):
            chosen[v] = True
    return sum(chosen)


def _greedy_cover(adjacency: list[list[int]], order: list[int]) -> list[int]:
    """Cover a graph with cliques greedily, visiting nodes in ``order``.

    Parameters
    ----------
    adjacency :
        The neighbors of each node.
    order :
        The order in which nodes are assigned.

    Returns
    -------
    clique :
        The clique of each node. Each node joins the first clique it is
        adjacent to every member of, or opens a new one.
    """
    clique = [-1] * len(adjacency)
    sizes: list[int] = []
    for v in order:
        eligible = _full_cliques(adjacency[v], clique, sizes)
        if eligible:
            clique[v] = eligible[0]
            sizes[eligible[0]] += 1
        else:
            clique[v] = len(sizes)
            sizes.append(1)
    return clique


def _search_covers(
    adjacency: list[list[int]],
    order: list[int],
    max_cliques: int,
    max_visits: float,
    max_covers: int,
) -> tuple[list[list[int]], int, bool]:
    """Search for covers of a graph by at most ``k`` cliques.

    Parameters
    ----------
    adjacency :
        The neighbors of each node.
    order :
        The order in which nodes are assigned.
    max_cliques :
        The maximum number of cliques.
    max_visits :
        Stop after visiting this many nodes.
    max_covers :
        Stop after finding this many covers.

    Returns
    -------
    covers :
        The clique of each node, for each cover found.
    n_visited :
        The number of nodes visited.
    cap_hit :
        `True` if the search stopped at ``max_visits`` before finishing.

    Notes
    -----
    A depth-first search that assigns the nodes in ``order``: each node may
    join any clique it is adjacent to every member of, or open a new one
    while fewer than ``max_cliques`` exist. Cliques are numbered in the order
    they are opened, so every partition is reached by exactly one path.
    The stack is explicit because components can be far deeper than the
    recursion limit.
    """
    n = len(order)
    clique = [-1] * n
    sizes: list[int] = []
    covers: list[list[int]] = []

    def options(v: int) -> list[int]:
        eligible = _full_cliques(adjacency[v], clique, sizes)
        if len(sizes) < max_cliques:
            eligible.append(len(sizes))
        # Options are popped from the end, so the oldest clique is tried first.
        return eligible[::-1]

    def unassign(v: int) -> None:
        g = clique[v]
        clique[v] = -1
        sizes[g] -= 1
        if sizes[g] == 0:
            # Only the newest clique can empty, since later ones were opened
            # deeper in the search and have already been unwound.
            sizes.pop()

    stack = [options(order[0])]
    n_visited = 1
    while stack:
        depth = len(stack) - 1
        if not stack[-1]:
            stack.pop()
            if depth > 0:
                unassign(order[depth - 1])
            continue
        v = order[depth]
        g = stack[-1].pop()
        if g == len(sizes):
            sizes.append(1)
        else:
            sizes[g] += 1
        clique[v] = g
        if depth + 1 == n:
            covers.append(clique.copy())
            if len(covers) >= max_covers:
                return covers, n_visited, False
            unassign(v)
            continue
        if n_visited >= max_visits:
            return covers, n_visited, True
        n_visited += 1
        stack.append(options(order[depth + 1]))
    return covers, n_visited, False


def _cover_component(
    adjacency: list[list[int]],
    plane_bound: int,
    search_cap: int,
    count_covers: bool,
) -> tuple[list[int], dict[str, int | bool]]:
    """Find a minimum clique cover of one connected component.

    Parameters
    ----------
    adjacency :
        The neighbors of each node.
    plane_bound :
        The largest number of nodes that share a plane. They are pairwise
        unlinked, so this is a lower bound on the cover.
    search_cap :
        The number of nodes the exact search may visit, summed over every
        ``k`` it tries.
    count_covers :
        Count the distinct minimum covers of a small component.
        This is a diagnostic that should always be ``False`` in production.

    Returns
    -------
    clique :
        The clique of each node.
    stats :
        The values of the `COMPONENT_DTYPE` fields.

    Notes
    -----
    Finding the true clique cover is an NP-hard problem, so this function uses
    a combination of greedy heuristics and exact search within a limited search
    cap to find a minimum cover for small components.
    """
    n = len(adjacency)
    order = sorted(range(n), key=lambda v: (len(adjacency[v]), v))
    # The independent set is a lower bound on the number of cliques,
    # as no two nodes in an independent set can be in the same clique.
    # However, it is dependent on the order of nodes considered and is
    # not guaranteed to be the maximum.
    # It's a cheap way to get a lower bound that might be higher than the
    # number of components.
    lower = max(_greedy_independent_set(adjacency, order), plane_bound)

    # Estimate the upper bound on the number of cliques using a greedy cover.
    clique = _greedy_cover(adjacency, order)
    upper = max(clique) + 1
    max_cliques = upper
    n_visited = 0
    cap_hit = False
    for trial in range(lower, upper):
        if n_visited >= search_cap:
            cap_hit = True
            break
        # Try an exact search for a minimum clique cover
        covers, visited, cap_hit = _search_covers(adjacency, order, trial, search_cap - n_visited, 1)
        n_visited += visited
        if covers:
            # A valid cover was found, so update the clique and the maximum
            # number of cliques.
            clique = covers[0]
            max_cliques = trial
            break
        if cap_hit:
            break
    n_min_covers = -1
    if count_covers and n <= COUNT_COVERS_MAX_NODES and not cap_hit:

        covers, _, _ = _search_covers(adjacency, order, max_cliques, np.inf, COUNT_COVERS_MAX)
        n_min_covers = len(covers)
    stats: dict[str, int | bool] = {
        "lower_bound": lower,
        "upper_bound": upper,
        "k": max_cliques,
        "search_ran": lower < upper,
        "n_visited": n_visited,
        "cap_hit": cap_hit,
        "n_min_covers": n_min_covers,
    }
    return clique, stats


def _max_shared_plane(plane_flags: np.ndarray, component: np.ndarray, n_components: int) -> np.ndarray:
    """Largest number of positions in each component that share a plane.

    Parameters
    ----------
    plane_flags :
        The plane bitmask of each position.
    component :
        The component of each position.
    n_components :
        The number of components.

    Returns
    -------
    count :
        The count for each component.
    """
    flags = plane_flags.astype(np.int64)
    count = np.zeros(n_components, dtype=int)
    n_bits = int(flags.max()).bit_length() if len(flags) else 0
    for bit in range(n_bits):
        has_plane = ((flags >> bit) & 1).astype(bool)
        count = np.maximum(count, np.bincount(component[has_plane], minlength=n_components))
    return count


def _clique_cover(
    plane_flags: np.ndarray,
    edges: np.ndarray,
    search_cap: int = 100_000,
    count_covers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition the position graph into a minimum number of cliques.

    Parameters
    ----------
    plane_flags :
        The plane bitmask of each position, used for a lower bound.
    edges :
        The linked pairs, with shape ``(n_edges, 2)``.
    search_cap :
        The number of nodes the exact search may visit in a component.
    count_covers :
        Count the distinct minimum covers of components with at most
        `COUNT_COVERS_MAX_NODES` nodes, up to `COUNT_COVERS_MAX`.

    Returns
    -------
    clique :
        The clique (peak) of each position. Cliques are numbered in the
        order of their first member.
    component :
        The connected component of each position, numbered in the
        order of their first member. A component is a set of linked positions
        that is split into one or more peaks (cliques).
    components :
        One row per component with dtype `COMPONENT_DTYPE`.

    Notes
    -----
    A missing edge is evidence of two sources, so every peak is a clique and
    the number of peaks is the size of the minimum clique cover. Each
    connected component is covered on its own. A complete component is a
    single clique. Otherwise nodes are visited fewest neighbors first (ties by
    index) to build a greedy cover, an upper bound, and a greedy independent
    set, which together with ``plane_flags`` gives a lower bound. When the
    bounds differ, ``_search_covers`` tries each ``k`` from the lower bound
    up. If it hits ``search_cap`` the greedy cover is kept and ``cap_hit`` is
    set.
    """
    n = len(plane_flags)
    i, j = edges[:, 0], edges[:, 1]
    # The adjacency array needs both directions for each edge, meaning we need
    # i, j and j, i. We do this by concatenating them and switching the
    # order.
    # The format of a CSR array has three attributes:
    # - `data`: the non-zero values of the array. In this case all True.
    # - `indices`: the column indices of the non-zero elements. These are in
    #   the same order as the `data` array, arranged row by row, and by
    #   increasing column within each row (after `sort_indices`).
    # - `indptr`: points to the start of each row in the `indices` and `data`
    #   arrays. It has one entry per position plus a final entry marking the
    #   end of the last row, so the neighbors of position ``v`` are
    #   ``indices[indptr[v]:indptr[v + 1]]``. A row with no non-zero
    #   elements has ``indptr[v] == indptr[v + 1]``.
    adjacency = csr_array(
        (np.ones(2 * len(edges), dtype=bool), (np.concatenate([i, j]), np.concatenate([j, i]))),
        shape=(n, n),
    )
    adjacency.sort_indices()
    indptr, indices = adjacency.indptr, adjacency.indices

    # Split the graph into components.
    # Each component is at a minimum a single peak.
    n_components, component = connected_components(adjacency, directed=False)
    # Count the number of nodes and edges in each component.
    sizes = np.bincount(component, minlength=n_components)
    n_edges = np.bincount(component[i], minlength=n_components)
    # The largest number of positions in each component that share a single
    # (scale, band) plane. Positions in the same plane are always distinct,
    # so this gives us a lower bound on the number of peaks (cliques) in the
    # component.
    plane_bound = _max_shared_plane(plane_flags, component, n_components)

    components = np.zeros(n_components, dtype=COMPONENT_DTYPE)
    components["n_nodes"] = sizes
    complete = n_edges == sizes * (sizes - 1) // 2
    components["lower_bound"] = 1
    components["upper_bound"] = 1
    # Number of cliques (peaks) that each component is divided into.
    components["k"] = 1
    components["n_min_covers"] = -1
    if count_covers:
        components["n_min_covers"][complete & (sizes <= COUNT_COVERS_MAX_NODES)] = 1

    # Nodes (positions) grouped by component, in index order within each one.
    by_component = np.argsort(component, kind="stable")
    # Starts gives the starting index of each component in the sorted array
    # of nodes, so component `c`'s nodes are
    # `by_component[starts[c] : starts[c + 1]]`.
    starts = np.concatenate([[0], np.cumsum(sizes)])
    local = np.empty(n, dtype=int)
    # The local index of each node (candidate) within its component.
    local[by_component] = np.arange(n) - starts[component[by_component]]
    # The local clique index of each node within its component.
    local_clique = np.zeros(n, dtype=int)
    for c in np.flatnonzero(~complete):
        nodes = by_component[starts[c] : starts[c + 1]]
        # The local indices of the neighbors of each node
        # These are the other positions that can potentially be merged
        # with the node.
        neighbors = [local[indices[indptr[v] : indptr[v + 1]]].tolist() for v in nodes]
        # Subdivide the component into cliques (peaks), where each clique is
        # a set of nodes all connected to each other.
        component_clique, stats = _cover_component(neighbors, int(plane_bound[c]), search_cap, count_covers)
        local_clique[nodes] = component_clique
        for name, value in stats.items():
            components[name][c] = value

    # The starting offset of each component's cliques in the raw numbering,
    # where cliques are numbered component by component.
    offsets = np.cumsum(components["k"]) - components["k"]
    # A unique but not yet ordered clique label for each position
    raw_position_cliques = offsets[component] + local_clique
    n_cliques = int(components["k"].sum())
    # Determine the first occurrence of each clique in the global ordering.
    first = np.full(n_cliques, n, dtype=int)
    np.minimum.at(first, raw_position_cliques, np.arange(n))
    # The global index of each clique, determined by the first occurrence
    # of any of its nodes.
    global_clique = np.empty(n_cliques, dtype=int)
    global_clique[np.argsort(first)] = np.arange(n_cliques)
    # The clique that each position (node) belongs to.
    clique = global_clique[raw_position_cliques]
    return clique, component, components


def _link_fractions(
    clique: np.ndarray, edges: np.ndarray, n_cliques: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure how strongly each position links to the other cliques.

    Parameters
    ----------
    clique :
        The clique of each position.
    edges :
        The linked pairs, with shape ``(n_edges, 2)``.
    n_cliques :
        The number of cliques.

    Returns
    -------
    n_linked :
        The number of cliques the position is adjacent to every member of,
        counting its own.
    ambiguous :
        `True` where ``n_linked > 1``: the position could move to another
        clique without breaking it.
    max_fraction :
        The largest fraction of another clique's members the position is
        adjacent to, ``0`` if none.
    max_clique :
        The clique of ``max_fraction`` (the lowest id on a tie), ``-1`` if
        none.
    """
    n = len(clique)
    n_linked = np.ones(n, dtype=int)
    max_fraction = np.zeros(n)
    max_clique = np.full(n, -1, dtype=int)
    # Edges are not directed and only stored once.
    # We concatenate to give us both directions.
    # Note that positions with multiple edges will appear multiple times.
    node = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    # We swap the edge order in the concatenate, so each node is paired
    # with its linked partner. `other` is the clique of the partner node,
    # which tells us which cliques each node's partners belong to.
    other = clique[np.concatenate([edges[:, 1], edges[:, 0]])]
    # If the node and its partner are in the same clique,
    # the link carries no information so we ignore those.
    keep = other != clique[node]
    # Pack each (position, foreign clique) pair into one integer.
    # This works because other < n_cliques.
    # np.unique counts how many members of that clique the position is
    # linked to.
    key, count = np.unique(node[keep] * n_cliques + other[keep], return_counts=True)
    if len(key) == 0:
        # This happens if each component has only a single clique (peak)
        # or if there are no edges at all.
        return n_linked, n_linked > 1, max_fraction, max_clique
    # Unpack the position and its partner's clique from the key.
    node, other = np.divmod(key, n_cliques)
    # The number of members in the partner's clique of each pair.
    size = np.bincount(clique, minlength=n_cliques)[other]
    fraction = count / size
    # A position linked to every member of a foreign clique could join
    # that clique.
    n_linked += np.bincount(node[count == size], minlength=n)

    # lexsort uses its last key first: group by position, then descending
    # fraction, then the lowest clique id on a tie.
    order = np.lexsort((other, -fraction, node))
    # The first row of each position's run is its strongest foreign clique.
    first = np.ones(len(order), dtype=bool)
    first[1:] = node[order[1:]] != node[order[:-1]]
    best = order[first]
    # Positions with no foreign links are absent from node and keep the
    # defaults.
    max_fraction[node[best]] = fraction[best]
    max_clique[node[best]] = other[best]
    return n_linked, n_linked > 1, max_fraction, max_clique


def _clique_median(values: np.ndarray, cliques: np.ndarray, n_cliques: int) -> np.ndarray:
    """Median of ``values`` within each clique.

    Parameters
    ----------
    values :
        The values.
    cliques :
        The clique of each value.
    n_cliques :
        The number of cliques.

    Returns
    -------
    median :
        The median of each clique, NaN for an empty clique.
    """
    # Sort by clique first, then by value, so the values of each clique
    # form one contiguous sorted run.
    ordered = values[np.lexsort((values, cliques))].astype(float)
    counts = np.bincount(cliques, minlength=n_cliques)
    # The offset in `ordered` where each clique's run begins.
    starts = np.cumsum(counts) - counts
    median = np.full(n_cliques, np.nan)
    has = counts > 0
    # The two middle elements of each run. For an odd count they are the
    # same element, so averaging them is the usual median either way.
    lower = starts[has] + (counts[has] - 1) // 2
    upper = starts[has] + counts[has] // 2
    median[has] = 0.5 * (ordered[lower] + ordered[upper])
    return median


def _unambiguous_members(positions: np.ndarray, n_peaks: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the positions that describe each peak.

    Parameters
    ----------
    positions :
        The positions with ``peak`` and ``ambiguous`` filled.
    n_peaks :
        The number of peaks.

    Returns
    -------
    use :
        `True` for the unambiguous positions, and for every position of a
        peak that has none.
    fallback :
        `True` for each peak that has no unambiguous position.
    """
    peak = positions["peak"]
    clear = ~positions["ambiguous"]
    # Peaks that have no clear positions get a fallback.
    fallback = np.bincount(peak[clear], minlength=n_peaks) == 0
    return clear | fallback[peak], fallback


def _locate_peaks(
    positions: np.ndarray,
    candidates: np.ndarray,
    use: np.ndarray,
    n_peaks: int,
    psf_fwhm: np.ndarray,
) -> np.ndarray:
    """Choose the position that locates each peak.

    Parameters
    ----------
    positions :
        The positions with ``peak`` filled.
    candidates :
        The candidates with ``position`` filled.
    use :
        The positions a peak may be located at, from
        ``_unambiguous_members``.
    n_peaks :
        The number of peaks.
    psf_fwhm :
        The PSF FWHM of each single band, in pixels.

    Returns
    -------
    location :
        The index of the chosen position for each peak.

    Notes
    -----
    The chosen position is the one closest to the median ``x`` and ``y`` of
    the peak's candidates, so a position counts once per candidate. Ties go to
    the finest scale, then the smallest PSF FWHM among the bands of the
    position's finest-scale candidates (the chi^2 plane ranks last), then the
    lowest position index.
    """
    position_of = candidates["position"]
    # Work at the candidate level so a position detected in several planes
    # pulls the median toward itself once per plane.
    in_use = use[position_of]
    candidate_peak = positions["peak"][position_of[in_use]]
    median_y = _clique_median(candidates["y"][in_use], candidate_peak, n_peaks)
    median_x = _clique_median(candidates["x"][in_use], candidate_peak, n_peaks)

    # A position's scale is the finest of its candidates, so this selects the
    # candidates at that scale.
    finest = candidates["scale"] == positions["scale"][position_of]
    # Map a candidate's band to its PSF FWHM. The chi^2 plane is band index
    # n_bands and maps to inf, so it ranks after every single band.
    band_rank = np.append(psf_fwhm, np.inf)
    # The smallest PSF FWHM among the finest-scale candidates of each
    # position.
    finest_fwhm = np.full(len(positions), np.inf)
    np.minimum.at(finest_fwhm, position_of[finest], band_rank[candidates["band"][finest]])

    index = np.flatnonzero(use)
    peak = positions["peak"][index]
    # The medians are multiples of 0.5, so distances that tie compare equal.
    distance2 = (positions["y"][index] - median_y[peak]) ** 2 + (positions["x"][index] - median_x[peak]) ** 2
    # lexsort uses its last key first: group by peak, then the tie breakers
    # in the order listed in the Notes.
    order = np.lexsort((index, finest_fwhm[index], positions["scale"][index], distance2, peak))
    # The first row of each peak's run is its chosen position. Every peak
    # has at least one row because _unambiguous_members never leaves a peak
    # without a usable position.
    sorted_peak = peak[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_peak[1:] != sorted_peak[:-1]
    location = np.empty(n_peaks, dtype=int)
    location[sorted_peak[first]] = index[order][first]
    return location


def _fill_position_errors(
    peaks: np.ndarray,
    positions: np.ndarray,
    candidates: np.ndarray,
    use: np.ndarray,
    psf_fwhm: np.ndarray,
) -> None:
    """Fill the position error fields of each peak.

    Parameters
    ----------
    peaks :
        The detections with ``y`` and ``x`` set. The error fields are filled
        in place.
    positions :
        The positions with ``peak`` filled.
    candidates :
        The candidates with ``position`` filled.
    use :
        The positions that describe each peak, from ``_unambiguous_members``.
    psf_fwhm :
        The PSF FWHM of each single band, in pixels.

    Notes
    -----
    Every term is computed over the candidates of the ``use`` positions. The
    statistical error of a candidate is its effective width from
    ``_starlet_kernel_fwhm`` over ``2.355 * significance``. Scales within a
    band are not independent, so each band contributes its best candidate and
    the bands are combined by inverse variance. The chi^2 plane is a
    combination of the bands and is used only for a peak with no single-band
    candidate. The provisional error is the largest of the statistical error,
    the robust scatter, and the ``1 / sqrt(12)`` quantization floor.
    """
    n_peaks = len(peaks)
    n_bands = len(psf_fwhm)
    position_of = candidates["position"]
    keep = use[position_of]
    peak = positions["peak"][position_of[keep]]
    band = candidates["band"][keep]
    scale = candidates["scale"][keep]
    y = candidates["y"][keep].astype(float)
    x = candidates["x"][keep].astype(float)
    # The chi^2 plane takes the widest band's PSF.
    plane_fwhm = np.append(psf_fwhm, psf_fwhm.max())

    width = _starlet_kernel_fwhm(plane_fwhm[band], scale)
    sigma = width / (SIGMA_TO_FWHM * candidates["flux"][keep])
    best = np.full((n_peaks, n_bands + 1), np.inf)
    np.minimum.at(best, (peak, band), sigma)
    inverse_variance = np.sum(best[:, :n_bands] ** -2.0, axis=1)
    stat = best[:, n_bands].copy()
    has_band = inverse_variance > 0
    stat[has_band] = inverse_variance[has_band] ** -0.5
    peaks["stat_err_y"] = stat
    peaks["stat_err_x"] = stat

    count = np.bincount(peak, minlength=n_peaks)
    dy = y - peaks["y"][peak]
    dx = x - peaks["x"][peak]
    peaks["moment_yy"] = np.bincount(peak, weights=dy * dy, minlength=n_peaks) / count
    peaks["moment_xx"] = np.bincount(peak, weights=dx * dx, minlength=n_peaks) / count
    peaks["moment_xy"] = np.bincount(peak, weights=dx * dy, minlength=n_peaks) / count

    for axis, values in (("y", y), ("x", x)):
        median = _clique_median(values, peak, n_peaks)
        scatter = MAD_TO_SIGMA * _clique_median(np.abs(values - median[peak]), peak, n_peaks)
        scatter[count < 3] = np.nan
        peaks[f"scatter_{axis}"] = scatter
        # fmax ignores the NaN scatter of a peak with too few candidates.
        peaks[f"err_{axis}"] = np.fmax(np.fmax(stat, scatter), 1 / np.sqrt(12))

    nearest = np.full(n_peaks, np.inf)
    if n_peaks > 1:
        yx = np.column_stack([peaks["y"], peaks["x"]]).astype(float)
        nearest = cKDTree(yx).query(yx, k=2)[0][:, 1]
    min_fwhm = np.full(n_peaks, np.inf)
    np.minimum.at(min_fwhm, peak, plane_fwhm[band])
    peaks["nearest_distance"] = nearest
    peaks["nearest_distance_fwhm"] = nearest / min_fwhm


def _build_detections(
    positions: np.ndarray,
    candidates: np.ndarray,
    components: np.ndarray,
    psf_fwhm: np.ndarray,
) -> np.ndarray:
    """Reduce grouped positions to one detection record per peak.

    Parameters
    ----------
    positions :
        The positions with the grouping fields filled by ``_group_peaks``.
    candidates :
        The candidates with ``position`` filled.
    components :
        The components from ``_clique_cover``.
    psf_fwhm :
        The PSF FWHM of each single band, in pixels.

    Returns
    -------
    peaks :
        The detections with dtype `DETECTION_DTYPE`. The location and
        ``flux`` come from the position chosen by ``_locate_peaks``;
        ``scale``, ``peak_sigma`` and the flags are reductions over the
        unambiguous positions (all positions for a peak with none), while the
        counts include every position.
    """
    n_peaks = int(positions["peak"].max()) + 1 if len(positions) else 0
    peaks = np.zeros(n_peaks, dtype=DETECTION_DTYPE)
    if n_peaks == 0:
        return peaks
    peak_of = positions["peak"]
    # To determine the location of a peak we only use positions that are
    # unambiguously contained in its clique.
    use, fallback = _unambiguous_members(positions, n_peaks)
    location = _locate_peaks(positions, candidates, use, n_peaks, psf_fwhm)
    members = positions[use]
    member_peak = peak_of[use]

    peaks["y"] = positions["y"][location]
    peaks["x"] = positions["x"][location]
    peaks["flux"] = positions["flux"][location]
    scale = np.full(n_peaks, np.iinfo(int).max, dtype=int)
    np.minimum.at(scale, member_peak, members["scale"])
    peaks["scale"] = scale
    peak_sigma = np.full(n_peaks, -np.inf)
    np.maximum.at(peak_sigma, member_peak, members["peak_sigma"])
    peaks["peak_sigma"] = peak_sigma
    band_flags = np.zeros(n_peaks, dtype=np.int64)
    np.bitwise_or.at(band_flags, member_peak, members["band_flags"].astype(np.int64))
    peaks["band_flags"] = band_flags
    scale_flags = np.zeros(n_peaks, dtype=np.int64)
    np.bitwise_or.at(scale_flags, member_peak, members["scale_flags"].astype(np.int64))
    peaks["scale_flags"] = scale_flags

    peaks["n_candidates"] = np.bincount(peak_of, weights=positions["n_candidates"], minlength=n_peaks)
    peaks["n_positions"] = np.bincount(peak_of, minlength=n_peaks)
    peaks["n_ambiguous"] = np.bincount(peak_of, weights=positions["ambiguous"], minlength=n_peaks)
    peaks["location_fallback"] = fallback
    peaks["search_capped"] = components["cap_hit"][positions["component"][location]]
    _fill_position_errors(peaks, positions, candidates, use, psf_fwhm)
    return peaks


def _build_footprints(
    footprint_mask: np.ndarray,
    peaks: np.ndarray,
    origin: tuple[int, int] = (0, 0),
) -> list[Footprint]:
    """Split the footprint mask into footprints and place the peaks in them.

    Parameters
    ----------
    footprint_mask :
        The union of the per-plane footprints from ``_find_peak_candidates``.
    peaks :
        The detections with dtype `DETECTION_DTYPE`. The ``footprint`` field
        is filled in place.
    origin :
        The ``(y, x)`` location of the lower corner of ``footprint_mask``.

    Returns
    -------
    footprints :
        The connected components of ``footprint_mask`` that contain at least
        one peak, each holding a `Peak` per detection inside it, brightest
        first. The `Peak` flux is the detection's ``peak_sigma``.

    Raises
    ------
    RuntimeError
        Raised if a peak lies outside every footprint. Every peak is located
        at a candidate inside one of the per-plane footprints, so this
        indicates a bug upstream.

    Notes
    -----
    Per-plane footprints of the same source differ in extent (and may be
    split or merged) from one plane to the next, so a footprint is identified
    with a source only after the peaks are grouped: the mask is relabeled into
    4-connected components and each peak is looked up in the component that
    contains it. A component whose candidates were all grouped with a peak
    located in another component ends up with no peak and is dropped, so
    ``footprints`` never holds an empty footprint.
    """
    y0, x0 = origin
    if len(peaks) == 0:
        return []
    components = get_footprints(
        footprint_mask.astype(np.float32),
        min_separation=0,
        min_area=1,
        peak_thresh=0,
        footprint_thresh=0.5,
        find_peaks=False,
        y0=y0,
        x0=x0,
    )
    bbox = Box(footprint_mask.shape, origin=origin)
    # Labels are the component index plus one, zero outside every footprint.
    labels = footprints_to_image(components, bbox).data
    component_of = labels[peaks["y"] - y0, peaks["x"] - x0] - 1
    if np.any(component_of < 0):
        raise RuntimeError("A detected peak lies outside every footprint")

    # Brightest first within a footprint, matching get_footprints.
    order = np.lexsort((-peaks["peak_sigma"], component_of))
    footprints: list[Footprint] = []
    footprint_of = np.empty(len(peaks), dtype=int)
    last = -1
    for row in order:
        component = component_of[row]
        if component != last:
            footprints.append(components[component])
            last = component
        footprint_of[row] = len(footprints) - 1
        footprints[-1].add_peak(
            Peak(int(peaks["y"][row]), int(peaks["x"][row]), float(peaks["peak_sigma"][row]))
        )
    peaks["footprint"] = footprint_of
    return footprints


def _group_peaks(
    candidates: np.ndarray,
    n_bands: int,
    first_scale: int,
    psf_fwhm: np.ndarray,
    radius_rule: str = "min",
    search_cap: int = 100_000,
    count_covers: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse candidates to positions and group positions into detections.

    Parameters
    ----------
    candidates :
        Structured array of candidates with dtype `CANDIDATE_DTYPE`. The
        ``position`` field is filled in place.
    n_bands :
        The number of single-band planes.
    first_scale :
        The first starlet scale in the significance map.
    psf_fwhm :
        The PSF FWHM of each single band, in pixels. The largest sets the link
        radius.
    radius_rule :
        Passed to ``_build_pairs``.
    search_cap :
        Passed to ``_clique_cover``.
    count_covers :
        Passed to ``_clique_cover``.

    Returns
    -------
    positions :
        The unique positions with dtype `POSITION_DTYPE`.
    peaks :
        The detections with dtype `DETECTION_DTYPE`.
    pairs :
        The position pairs with dtype `PAIR_DTYPE`.
    components :
        The connected components of the position graph with dtype
        `COMPONENT_DTYPE`.
    """
    if len(candidates) == 0:
        return (
            np.empty(0, dtype=POSITION_DTYPE),
            np.empty(0, dtype=DETECTION_DTYPE),
            np.empty(0, dtype=PAIR_DTYPE),
            np.empty(0, dtype=COMPONENT_DTYPE),
        )
    positions = _collapse_positions(candidates, n_bands, first_scale)
    pairs = _build_pairs(positions, float(np.max(psf_fwhm)), radius_rule)
    linked = pairs[pairs["linked"]]
    edges = np.column_stack([linked["i"], linked["j"]])
    peak, component, components = _clique_cover(positions["plane_flags"], edges, search_cap, count_covers)
    positions["peak"] = peak
    positions["component"] = component
    n_linked, ambiguous, max_fraction, max_peak = _link_fractions(peak, edges, int(peak.max()) + 1)
    positions["n_linked"] = n_linked
    positions["ambiguous"] = ambiguous
    positions["max_linked_fraction"] = max_fraction
    positions["max_linked_peak"] = max_peak
    peaks = _build_detections(positions, candidates, components, psf_fwhm)
    return positions, peaks, pairs, components


def detect_peaks(
    images: np.ndarray,
    variance: np.ndarray,
    scales: int = 3,
    generation: int = 2,
    first_scale: int = 1,
    origin: tuple[int, int] | None = None,
    min_separation: float | None = None,
    min_area: int = 4,
    peak_thresh: float = 3,
    footprint_thresh: float = 2,
    psf_fwhm: float | Sequence[float] = 3.5,
    kappa: float = 3.0,
    variance_mode: str = "median",
    bad_pixel_mask: np.ndarray | None = None,
    dilation_radius: float = 0,
    radius_rule: str = "min",
    search_cap: int = SEARCH_CAP,
    count_covers: bool = False,
) -> PeakDetectionResult:
    """Detect peaks across bands and starlet scales.

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
        A hard floor on the separation between peaks within a plane, in
        pixels; ``0`` disables it and relies on ``kappa`` alone.
        `None` uses the smallest PSF FWHM to calculate the minimum separation
        based on two Gaussians of equal amplitude.
    min_area :
        The minimum area of a footprint in pixels.
    peak_thresh :
        The peak detection threshold, in sigma.
    footprint_thresh :
        The footprint detection threshold, in sigma.
    psf_fwhm :
        The PSF FWHM in pixels, either one value for every band or one per
        band. The largest is the floor on the linking radius; the per-band
        values break ties in the peak location and set the position errors.
    kappa :
        The minimum prominence in sigma of a peak above its saddle to a
        brighter peak in the same plane; smaller bumps are culled.
    variance_mode :
        How ``variance`` is reduced to a coefficient noise, either ``"median"``
        for one value per band or ``"pixel"`` to follow the variance plane; see
        ``_build_detection_starlets``.
    bad_pixel_mask :
        Per-band boolean mask with the same shape as ``images``, `True` for
        pixels that are bad in that band. Candidate peaks that fall within
        ``dilation_radius`` of a bad pixel in a given band are suppressed
        in that band only. This allows eg. saturated sources in one band to
        be detected in other, clean, bands.
        `None` disables the suppression.
    dilation_radius :
        The base radius in pixels by which ``bad_pixel_mask`` is grown before
        it suppresses peaks, to catch the starlet ringing just outside a bad
        region. The radius doubles with each starlet scale, since the kernel
        does; see ``_find_peak_candidates``.
    radius_rule :
        Whether two positions link within the smaller (``"min"``) or larger
        (``"max"``) of their link radii; see ``_build_pairs``.
    search_cap :
        The number of nodes the exact clique cover search may visit in a
        component before it falls back to the greedy cover; see
        ``_clique_cover``.
    count_covers :
        Count the distinct minimum covers of small components; see
        ``_clique_cover``. This is a diagnostic that should always be
        ``False`` in production.

    Returns
    -------
    result :
        The detected peaks, their footprints, and the intermediate detection
        products.

    Raises
    ------
    ValueError
        Raised if ``bad_pixel_mask`` is given and does not match the shape of
        ``images``, if ``psf_fwhm`` is a sequence without one value per
        band, or if ``radius_rule`` is not a key of `RADIUS_RULES`.
    """
    band_fwhm = np.asarray(psf_fwhm, dtype=float)
    if band_fwhm.ndim == 0:
        band_fwhm = np.full(len(images), float(band_fwhm))
    elif band_fwhm.shape != (len(images),):
        raise ValueError(f"psf_fwhm must be a scalar or have one value per band, got {band_fwhm.shape}")
    if radius_rule not in RADIUS_RULES:
        raise ValueError(f"radius_rule must be one of {list(RADIUS_RULES)}, got {radius_rule!r}")
    link_fwhm = float(band_fwhm.max())
    if min_separation is None:
        min_separation = band_fwhm.min() * 0.84932
    if origin is None:
        origin = (0, 0)
    if bad_pixel_mask is not None and bad_pixel_mask.shape != images.shape:
        raise ValueError("bad_pixel_mask must have the same shape as images")
    starlets, sigma = _build_detection_starlets(
        images,
        variance,
        scales=scales,
        generation=generation,
        variance_mode=variance_mode,
    )
    significance_map = _build_significance_map(starlets, sigma, first_scale=first_scale)
    candidates, footprint_mask = _find_peak_candidates(
        significance_map,
        min_separation,
        min_area,
        peak_thresh,
        footprint_thresh,
        kappa,
        first_scale,
        origin,
        bad_pixel_mask,
        dilation_radius,
    )
    n_bands = significance_map.shape[1] - 1
    positions, peaks, pairs, components = _group_peaks(
        candidates, n_bands, first_scale, band_fwhm, radius_rule, search_cap, count_covers
    )
    footprints = _build_footprints(footprint_mask, peaks, origin)
    link_scales = np.arange(first_scale, first_scale + significance_map.shape[0])
    metadata = {
        "psf_fwhm": band_fwhm.tolist(),
        "link_radius_scales": link_scales.tolist(),
        "link_radius": np.atleast_1d(_link_radius(link_scales, link_fwhm)).tolist(),
        "min_separation": float(min_separation),
        "radius_rule": radius_rule,
        "search_cap": int(search_cap),
    }
    return PeakDetectionResult(
        peaks=peaks,
        footprints=footprints,
        positions=positions,
        candidates=candidates,
        significance_map=significance_map,
        starlets=starlets,
        sigma=sigma,
        pairs=pairs,
        components=components,
        metadata=metadata,
    )
