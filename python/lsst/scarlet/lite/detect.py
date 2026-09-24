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
    ]
)


@dataclass
class PeakDetectionResult:
    """Detected peaks and the products used to detect them.

    Attributes
    ----------
    peaks :
        Structured array of detections with dtype `DETECTION_DTYPE`, one row
        per source. Row ids link back to ``positions`` via
        ``positions["peak"]``; ``footprint`` is the index into
        ``footprints`` of the footprint containing the peak.
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
        was a distinct peak in; ``n_linked`` is the number of peaks the
        position could have joined (see ``_assign_peaks``), and ``peak`` is
        -1 for a position that was not assigned to any peak.
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
    """

    peaks: np.ndarray
    footprints: list[Footprint]
    positions: np.ndarray
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


def _find_peak_candidates(
    significance_map: np.ndarray,
    min_separation: float = 0,
    min_area: int = 4,
    peak_thresh: float = 3,
    footprint_thresh: float = 2,
    kappa: float = 3,
    first_scale: int = 1,
    origin: tuple[int, int] = (0, 0),
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
        sources (see ``_assign_peaks``).
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
    footprint_mask :
        Boolean image with the spatial shape of ``significance_map`` that is
        `True` in every pixel of any plane's footprint that has a peak.
    """
    y0, x0 = origin
    height, width = significance_map.shape[-2:]
    footprint_mask = np.zeros((height, width), dtype=bool)
    candidates = []
    for scale_index, scale_planes in enumerate(significance_map):
        scale = scale_index + first_scale
        for band, plane in enumerate(scale_planes):
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
        ``(y, x)``. ``n_linked`` and ``peak`` are left for ``_assign_peaks``.

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


def _assign_peaks(
    positions: np.ndarray,
    psf_fwhm: float,
    blend_policy: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    """Assign positions to peaks, filling their ``peak`` and ``n_linked``.

    Parameters
    ----------
    positions :
        The unique positions with dtype `POSITION_DTYPE`. ``peak`` and
        ``n_linked`` are filled in place.
    psf_fwhm :
        The PSF FWHM in pixels, passed to ``_link_radius``.
    blend_policy :
        What to do with a position that could join two or more existing
        peaks: ``"nearest"`` joins the one with the nearest seed, ``"drop"``
        leaves it unassigned (``peak == -1``).

    Returns
    -------
    seeds :
        For each peak, the index of the position that seeded it. This is the
        peak's finest-scale, brightest position and defines its location.
    peak_of :
        ``positions["peak"]``, for convenience.

    Notes
    -----
    Two positions are *distinct* if they were both peaks in the same plane
    ``(band, scale)`` (the plane's contrast-limited watershed already
    established a significant saddle between them), i.e. if their
    ``plane_flags`` share a bit. Positions are visited finest scale first,
    brightest first within a scale, and each one joins the nearest existing
    peak that lies within the linking radius and contains no position it is
    distinct from; if there is none it seeds a new peak.

    Because a position never joins a peak containing a position it is
    distinct from, two distinct positions can never share a peak, however
    many other positions either of them merges with. Together with the
    watershed this gives the invariant that matters: a pair resolved as two
    peaks in any plane is two peaks in the output.

    ``n_linked`` counts the peaks a position was eligible to join. A value of
    two or more means the position (typically a coarse-scale peak sitting
    between two finer ones) is ambiguous between several sources, and its
    assignment was decided by ``blend_policy``; downstream code can treat it
    as a blend rather than a measurement of whichever peak it landed in.

    The loop is sequential (each decision depends on the peaks already
    seeded) and costs a few tens of microseconds per position in Python; if
    it ever dominates it ports directly to C++ with the same structure.
    """
    n = len(positions)
    peak_of = np.full(n, -1, dtype=int)
    n_linked = np.zeros(n, dtype=int)
    seeds: list[int] = []
    if n == 0:
        positions["peak"] = peak_of
        positions["n_linked"] = n_linked
        return np.array(seeds, dtype=int), peak_of
    if blend_policy not in ("nearest", "drop"):
        raise ValueError(f"blend_policy must be 'nearest' or 'drop', got {blend_policy!r}")

    yx = np.column_stack([positions["y"], positions["x"]]).astype(float)
    radii: np.ndarray = cast(np.ndarray, _link_radius(positions["scale"], psf_fwhm))
    flags = positions["plane_flags"]
    neighbors = cKDTree(yx).query_ball_point(yx, float(radii.max()))
    order = np.lexsort((-positions["peak_sigma"], positions["scale"]))
    is_seed = np.zeros(n, dtype=bool)
    members: list[list[int]] = []

    for i in order:
        yx_i = yx[i]
        radius = radii[i]
        flag = flags[i]
        # Eligible peaks, keyed by peak id, with squared distance to the seed.
        eligible: dict[int, float] = {}
        for k in neighbors[i]:
            if not is_seed[k]:
                continue
            d = yx[k] - yx_i
            d2 = d[0] ** 2 + d[1] ** 2
            r = max(radius, radii[k])
            if d2 > r**2:
                continue
            pk = peak_of[k]
            if any(flag & flags[m] for m in members[pk]):
                continue  # distinct from a member of this peak
            eligible[pk] = d2
        n_linked[i] = len(eligible)
        if len(eligible) == 0:
            peak_of[i] = len(members)
            is_seed[i] = True
            seeds.append(i)
            members.append([i])
        elif len(eligible) == 1 or blend_policy == "nearest":
            pk = min(eligible, key=eligible.__getitem__)
            peak_of[i] = pk
            members[pk].append(i)
        # else: "drop" with several eligible peaks; leave unassigned.

    positions["peak"] = peak_of
    positions["n_linked"] = n_linked
    return np.array(seeds, dtype=int), peak_of


def _build_detections(positions: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Reduce assigned positions to one detection record per peak.

    Parameters
    ----------
    positions :
        The positions with ``peak`` and ``n_linked`` filled by
        ``_assign_peaks``.
    seeds :
        The seed position of each peak, from ``_assign_peaks``.

    Returns
    -------
    peaks :
        The detections with dtype `DETECTION_DTYPE`. The position, finest
        scale and ``flux`` come from the seed; the flags and counts are
        reductions over all of the peak's positions.
    """
    n_peaks = len(seeds)
    peaks = np.zeros(n_peaks, dtype=DETECTION_DTYPE)
    if n_peaks == 0:
        return peaks
    assigned = positions["peak"] >= 0
    members = positions[assigned]
    peak_of = members["peak"]

    peaks["y"] = positions["y"][seeds]
    peaks["x"] = positions["x"][seeds]
    peaks["scale"] = positions["scale"][seeds]
    peaks["flux"] = positions["flux"][seeds]

    peak_sigma = np.full(n_peaks, -np.inf)
    np.maximum.at(peak_sigma, peak_of, members["peak_sigma"])
    peaks["peak_sigma"] = peak_sigma
    band_flags = np.zeros(n_peaks, dtype=np.int64)
    np.bitwise_or.at(band_flags, peak_of, members["band_flags"].astype(np.int64))
    peaks["band_flags"] = band_flags
    scale_flags = np.zeros(n_peaks, dtype=np.int64)
    np.bitwise_or.at(scale_flags, peak_of, members["scale_flags"].astype(np.int64))
    peaks["scale_flags"] = scale_flags
    peaks["n_candidates"] = np.bincount(peak_of, weights=members["n_candidates"], minlength=n_peaks).astype(
        int
    )
    peaks["n_positions"] = np.bincount(peak_of, minlength=n_peaks)
    peaks["n_ambiguous"] = np.bincount(peak_of, weights=members["n_linked"] > 1, minlength=n_peaks).astype(
        int
    )
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
        Raised if a peak lies outside every footprint. Every peak is seeded by
        a candidate inside one of the per-plane footprints, so this indicates
        a bug upstream.

    Notes
    -----
    Per-plane footprints of the same source differ in extent (and may be
    split or merged) from one plane to the next, so a footprint is identified
    with a source only after the peaks are grouped: the mask is relabeled into
    4-connected components and each peak is looked up in the component that
    contains it. A component whose candidates were all linked to a peak
    seeded in another component ends up with no peak and is dropped, so
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
    psf_fwhm: float,
    blend_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse candidates to positions and link positions into detections.

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
        The PSF FWHM in pixels, passed to ``_link_radius``.
    blend_policy :
        Passed to ``_assign_peaks``.

    Returns
    -------
    positions :
        The unique positions with dtype `POSITION_DTYPE`.
    peaks :
        The detections with dtype `DETECTION_DTYPE`.
    """
    if len(candidates) == 0:
        return np.empty(0, dtype=POSITION_DTYPE), np.empty(0, dtype=DETECTION_DTYPE)
    positions = _collapse_positions(candidates, n_bands, first_scale)
    seeds, _ = _assign_peaks(positions, psf_fwhm, blend_policy)
    peaks = _build_detections(positions, seeds)
    return positions, peaks


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
    psf_fwhm: float = 3.5,
    kappa: float = 3.0,
    blend_policy: str = "nearest",
    variance_mode: str = "median",
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
    min_area :
        The minimum area of a footprint in pixels.
    peak_thresh :
        The peak detection threshold, in sigma.
    footprint_thresh :
        The footprint detection threshold, in sigma.
    psf_fwhm :
        The PSF FWHM in pixels, used as a floor on the linking radius.
    kappa :
        The minimum prominence in sigma of a peak above its saddle to a
        brighter peak in the same plane; smaller bumps are culled.
    blend_policy :
        How to assign a position that could join several peaks; see
        ``_assign_peaks``.
    variance_mode :
        How ``variance`` is reduced to a coefficient noise, either ``"median"``
        for one value per band or ``"pixel"`` to follow the variance plane; see
        ``_build_detection_starlets``.

    Returns
    -------
    result :
        The detected peaks, their footprints, and the intermediate detection
        products.
    """
    if origin is None:
        origin = (0, 0)
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
    )
    n_bands = significance_map.shape[1] - 1
    positions, peaks = _group_peaks(candidates, n_bands, first_scale, psf_fwhm, blend_policy)
    footprints = _build_footprints(footprint_mask, peaks, origin)
    return PeakDetectionResult(
        peaks=peaks,
        footprints=footprints,
        positions=positions,
        candidates=candidates,
        significance_map=significance_map,
        starlets=starlets,
        sigma=sigma,
    )
