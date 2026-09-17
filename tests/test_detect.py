# This file is part of lsst.scarlet.lite.
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

import os

import numpy as np
from lsst.scarlet.lite import Box, Image
from lsst.scarlet.lite.detect import (
    CANDIDATE_DTYPE,
    _build_detection_starlets,
    _build_significance_map,
    _chi2_to_sigma,
    _clipped_chi2_survival,
    _find_peak_candidates,
    _sigma_to_chi2,
    _starlet_scale_factors,
    bbox_to_bounds,
    bounds_to_bbox,
    detect_footprints,
    footprints_to_image,
    get_detect_wavelets,
    get_wavelets,
)
from lsst.scarlet.lite.detect_pybind11 import (
    Footprint,
    Peak,
    get_connected_multipeak,
    get_connected_pixels,
    get_footprints,
)
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from lsst.scarlet.lite.wavelet import starlet_transform
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats
from utils import ScarletTestCase


class TestDetect(ScarletTestCase):
    def setUp(self):
        centers = (
            (17, 9),
            (27, 14),
            (41, 25),
            (10, 42),
        )
        sigmas = (1.0, 0.95, 0.9, 1.5)

        sources = []
        for sigma, center in zip(sigmas, centers):
            yx0 = center[0] - 7, center[1] - 7
            source = Image(integrated_circular_gaussian(sigma=sigma).astype(np.float32), yx0=yx0)
            sources.append(source)

        image = Image.from_box(Box((51, 51)))
        for source in sources:
            image += source
        image.data[30:32, 40] = 0.5

        self.image = image
        self.centers = centers
        self.sources = sources

        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        self.hsc_data = np.load(filename)

    def tearDown(self):
        del self.hsc_data

    def test_connected(self):
        image = self.image.copy()

        # Check that the first 3 footprints are all connected
        # with thresholding at zero
        truth = self.sources[0] + self.sources[1] + self.sources[2]
        bbox = truth.bbox
        truth = truth.data > 0

        unchecked = np.ones(self.image.shape, dtype=bool)
        footprint = np.zeros(self.image.shape, dtype=bool)
        y, x = self.centers[0]
        get_connected_pixels(
            y,
            x,
            image.data,
            unchecked,
            footprint,
            np.array([y, y, x, x]).astype(np.int32),
            0,
        )
        assert_array_equal(footprint[bbox.slices], truth)

        # Check that only the first 2 footprints are all connected
        # with thresholding at 1e-15
        truth = self.sources[0] + self.sources[1]
        bbox = truth.bbox
        truth = truth.data > 1e-15

        unchecked = np.ones(self.image.shape, dtype=bool)
        footprint = np.zeros(self.image.shape, dtype=bool)
        y, x = self.centers[0]
        get_connected_pixels(
            y,
            x,
            image.data,
            unchecked,
            footprint,
            np.array([y, y, x, x]).astype(np.int32),
            1e-15,
        )
        assert_array_equal(footprint[bbox.slices], truth)

        # Test finding all peaks
        footprint = get_connected_multipeak(self.image.data, self.centers, 1e-15)
        truth = self.image.data > 1e-15
        truth[30:32, 40] = False
        assert_array_equal(footprint, truth)

    def _check_footprints(self, footprints):
        self.assertEqual(len(footprints), 3)

        # The first footprint has a single peak
        assert_array_equal(footprints[0].data, self.sources[3].data > 1e-15)
        self.assertEqual(len(footprints[0].peaks), 1)
        self.assertBoxEqual(footprints[0].bbox, self.sources[3].bbox)
        self.assertEqual(footprints[0].peaks[0].y, self.centers[3][0])
        self.assertEqual(footprints[0].peaks[0].x, self.centers[3][1])

        # The second footprint has two peaks
        truth = self.sources[0] + self.sources[1]
        assert_array_equal(footprints[1].data, truth.data > 1e-15)
        self.assertEqual(len(footprints[1].peaks), 2)
        self.assertBoxEqual(footprints[1].bbox, truth.bbox)
        self.assertEqual(footprints[1].peaks[0].y, self.centers[1][0])
        self.assertEqual(footprints[1].peaks[0].x, self.centers[1][1])
        self.assertEqual(footprints[1].peaks[1].y, self.centers[0][0])
        self.assertEqual(footprints[1].peaks[1].x, self.centers[0][1])

        # The third footprint has a single peak
        assert_array_equal(footprints[2].data, self.sources[2].data > 1e-15)
        self.assertEqual(len(footprints[2].peaks), 1)
        self.assertBoxEqual(footprints[2].bbox, self.sources[2].bbox)
        self.assertEqual(footprints[2].peaks[0].y, self.centers[2][0])
        self.assertEqual(footprints[2].peaks[0].x, self.centers[2][1])

        truth = 1 * self.sources[3] + 2 * (self.sources[0] + self.sources[1]) + 3 * self.sources[2]
        truth.data[truth.data < 1e-15] = 0
        fp_image = footprints_to_image(footprints, truth.bbox)
        assert_array_equal(fp_image, truth.data)

    def test_get_footprints(self):
        footprints = get_footprints(self.image.data, 1, 4, 1e-15, 1e-15, True)
        self._check_footprints(footprints)

    def test_get_footprints_min_area_boundary(self):
        """A footprint that exactly meets ``min_area`` in a tight
        bounding box of the same area must be kept.

        Audit finding D-1: the C++ pre-filter on the bounding-box
        area used strict ``>`` while the actual pixel count check
        uses ``>=``. A 2x2 filled square with ``min_area=4`` was
        therefore rejected by the pre-filter (4 > 4 is false) before
        the true area check (4 >= 4) ever ran.
        """
        img = np.zeros((10, 10), dtype=np.float32)
        img[3:5, 5:7] = 1.0  # 2x2 filled square: area=4, bbox=2x2=4

        # ``find_peaks=False`` to keep the test focused on the
        # min_area logic; with ``find_peaks=True`` an ambiguous
        # plateau can fail the peak check for unrelated reasons.
        footprints = get_footprints(img, 1, 4, 1e-15, 1e-15, False)
        self.assertEqual(len(footprints), 1)

        # Sanity: with ``min_area=5`` the same footprint must be
        # rejected, confirming the boundary is tight.
        footprints = get_footprints(img, 1, 5, 1e-15, 1e-15, False)
        self.assertEqual(len(footprints), 0)

    def _check_peaks(self, peaks):
        matched_peaks = []
        for center in self.centers:
            for peak in peaks:
                if peak.y == center[0] and peak.x == center[1]:
                    matched_peaks.append(peak)
                    break
        self.assertEqual(len(matched_peaks), len(self.centers))

    def test_detect_footprints(self):
        # This method doesn't test for accurracy, since
        # there is no variance, so we set it to ones.
        # detect_footprints is deprecated in favor of detect_peaks;
        # the calls assert the FutureWarning while still covering the
        # legacy behavior.
        variance = np.ones(self.image.shape, dtype=self.image.dtype)

        with self.assertWarns(FutureWarning):
            footprints = detect_footprints(
                self.image.data[None, :, :],
                variance[None, :, :],
                scales=1,
                generation=2,
                origin=(0, 0),
                min_separation=1,
                min_area=4,
                peak_thresh=1e-15,
                footprint_thresh=1e-15,
                find_peaks=True,
                remove_high_freq=False,
                min_pixel_detect=1,
            )

        self.assertEqual(len(footprints), 3)
        peaks = [peak for footprint in footprints for peak in footprint.peaks]
        self._check_peaks(peaks)

        with self.assertWarns(FutureWarning):
            footprints = detect_footprints(
                self.image.data[None, :, :],
                variance[None, :, :],
                scales=1,
                generation=1,
                min_separation=1,
                min_area=4,
                peak_thresh=1e-15,
                footprint_thresh=1e-15,
                find_peaks=True,
                remove_high_freq=True,
                min_pixel_detect=1,
            )

        self.assertEqual(len(footprints), 2)
        peaks = [peak for footprint in footprints for peak in footprint.peaks]
        self._check_peaks(peaks)

    def test_detect_footprints_min_pixel_detect(self):
        """``min_pixel_detect`` requires the detection pixel to be
        above zero in at least N bands. Verify both that single-band
        input is filtered out entirely when ``min_pixel_detect=2``,
        and that multi-band input filters selectively.
        """
        variance = np.ones(self.image.shape, dtype=self.image.dtype)

        # Single-band: with min_pixel_detect=2 every pixel fails the
        # "at least 2 bands above 0" check, so nothing survives.
        with self.assertWarns(FutureWarning):
            footprints = detect_footprints(
                self.image.data[None, :, :],
                variance[None, :, :],
                scales=1,
                generation=2,
                origin=(0, 0),
                min_separation=1,
                min_area=4,
                peak_thresh=1e-15,
                footprint_thresh=1e-15,
                find_peaks=True,
                remove_high_freq=False,
                min_pixel_detect=2,
            )
        self.assertEqual(len(footprints), 0)

        # Two-band: band 0 only contains sources 0+1, band 1 only
        # contains sources 2+3. With min_pixel_detect=2 no pixel is
        # above zero in *both* bands, so nothing survives.
        band0 = self.sources[0] + self.sources[1]
        band1 = self.sources[2] + self.sources[3]
        full = Image.from_box(Box((51, 51)))
        b0 = (full + band0).data
        b1 = (full + band1).data
        images = np.stack([b0, b1])
        variance2 = np.ones(images.shape, dtype=images.dtype)
        with self.assertWarns(FutureWarning):
            footprints = detect_footprints(
                images,
                variance2,
                scales=1,
                generation=2,
                origin=(0, 0),
                min_separation=1,
                min_area=4,
                peak_thresh=1e-15,
                footprint_thresh=1e-15,
                find_peaks=True,
                remove_high_freq=False,
                min_pixel_detect=2,
            )
        self.assertEqual(len(footprints), 0)

        # Sanity: with min_pixel_detect=1 the same multi-band input
        # produces the union of both bands' footprints.
        with self.assertWarns(FutureWarning):
            footprints = detect_footprints(
                images,
                variance2,
                scales=1,
                generation=2,
                origin=(0, 0),
                min_separation=1,
                min_area=4,
                peak_thresh=1e-15,
                footprint_thresh=1e-15,
                find_peaks=True,
                remove_high_freq=False,
                min_pixel_detect=1,
            )
        self.assertGreater(len(footprints), 0)

    def test_bounds_to_bbox(self):
        bounds = (3, 27, 11, 52)
        truth = Box((25, 42), (3, 11))
        bbox = bounds_to_bbox(bounds)
        self.assertBoxEqual(bbox, truth)

        # Check that the reverse operation also works
        new_bounds = bbox_to_bounds(bbox)
        self.assertTupleEqual(new_bounds, bounds)

    def test_footprint(self):
        footprint = self.sources[0].data
        footprint[footprint < 1e-15] = 0
        bounds = [
            self.sources[0].bbox.start[0],
            self.sources[0].bbox.stop[0] - 1,
            self.sources[0].bbox.start[1],
            self.sources[0].bbox.stop[1] - 1,
        ]
        print(bounds)
        peaks = [Peak(self.centers[0][0], self.centers[0][1], self.image.data[self.centers[0]])]
        footprint1 = Footprint(footprint, peaks, bounds)
        footprint = self.sources[1].data
        footprint[footprint < 1e-15] = 0
        bounds = [
            self.sources[1].bbox.start[0],
            self.sources[1].bbox.stop[0] - 1,
            self.sources[1].bbox.start[1],
            self.sources[1].bbox.stop[1] - 1,
        ]
        print(bounds)
        peaks = [Peak(self.centers[1][0], self.centers[1][1], self.image.data[self.centers[1]])]
        footprint2 = Footprint(footprint, peaks, bounds)

        truth = self.sources[0] + self.sources[1]
        truth.data[truth.data < 1e-15] = 0
        image = footprints_to_image([footprint1, footprint2], truth.bbox)
        assert_array_equal(image, truth.data)

        # Test intersection
        truth = (self.sources[0] > 1e-15) & (self.sources[1] > 1e-15)
        intersection = footprint1.intersection(footprint2)
        self.assertImageEqual(intersection, truth)

        # Test union
        truth = (self.sources[0] > 1e-15) | (self.sources[1] > 1e-15)
        union = footprint1.union(footprint2)
        self.assertImageEqual(union, truth)

    def test_get_wavelets(self):
        images = self.hsc_data["images"]
        variance = self.hsc_data["variance"]
        with self.assertWarns(FutureWarning):
            wavelets = get_wavelets(images, variance)

        self.assertTupleEqual(wavelets.shape, (5, 5, 58, 48))
        self.assertEqual(wavelets.dtype, np.float32)

    def test_get_detect_wavelets(self):
        images = self.hsc_data["images"]
        variance = self.hsc_data["variance"]
        with self.assertWarns(FutureWarning):
            wavelets = get_detect_wavelets(images, variance)

        self.assertTupleEqual(wavelets.shape, (4, 58, 48))


class TestPeakDetection(ScarletTestCase):
    def setUp(self):
        # Three isotropic Gaussian sources present in every band on top of
        # per-band white noise. The bands have different noise levels so the
        # per-band standardization is actually exercised.
        rng = np.random.default_rng(42)
        self.n_bands = 3
        self.shape = (64, 64)
        self.centers = [(20, 15), (40, 45), (12, 50)]
        band_std = np.array([1.5, 2.0, 3.0], dtype=np.float32)

        yy, xx = np.mgrid[0 : self.shape[0], 0 : self.shape[1]]
        images = rng.standard_normal((self.n_bands,) + self.shape) * band_std[:, None, None]
        for cy, cx in self.centers:
            bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5**2))
            images += 40.0 * bump
        self.images = images.astype(np.float32)
        self.variance = np.tile((band_std**2)[:, None, None], (1,) + self.shape)
        self.band_std = band_std

    def test_starlet_scale_factors(self):
        factors = _starlet_scale_factors(3, generation=2)
        self.assertEqual(factors.shape, (4,))
        self.assertTrue(np.all(factors > 0))
        # Coarser scales spread the kernel wider, so each successive
        # sum(K_j**2) is smaller.
        self.assertTrue(np.all(np.diff(factors) < 0))

        # White noise of variance ``v`` produces coefficients of variance
        # ``F_j * v`` at scale ``j``; this is the whole reason the factors
        # exist. Measure it on the interior to avoid boundary effects.
        rng = np.random.default_rng(0)
        v = 4.0
        noise = rng.standard_normal((512, 512)) * np.sqrt(v)
        coeffs = starlet_transform(noise, scales=3, generation=2)
        measured = np.array([c[50:-50, 50:-50].var() for c in coeffs])
        assert_allclose(measured, factors * v, rtol=0.05)

    def test_build_detection_starlets(self):
        starlets, sigma = _build_detection_starlets(self.images, self.variance, scales=3)
        self.assertEqual(starlets.shape, (4, self.n_bands) + self.shape)
        self.assertEqual(sigma.shape, (4, self.n_bands))

        # sigma_{j,b} = sqrt(F_j * nanmedian(var_b)).
        factors = _starlet_scale_factors(3)
        band_var = np.nanmedian(self.variance, axis=(1, 2))
        expected = np.sqrt(factors[:, None] * band_var[None, :])
        assert_allclose(sigma, expected, rtol=1e-4)

    def test_build_detection_starlets_ignores_nan_variance(self):
        variance = self.variance.copy()
        variance[:, :10, :10] = np.nan
        _, sigma = _build_detection_starlets(self.images, variance, scales=3)
        self.assertFalse(np.any(np.isnan(sigma)))

    def test_build_detection_starlets_validation(self):
        with self.assertRaises(ValueError):
            _build_detection_starlets(self.images, self.variance[:, :10, :10])
        with self.assertRaises(ValueError):
            _build_detection_starlets(self.images[0], self.variance[0])

    def test_clipped_chi2_survival(self):
        c = np.array([0.5, 2.0, 8.0, 20.0])
        # For one band the coadd is zero half the time and a chi^2_1 deviate
        # otherwise, so the survival function is 0.5 * chi2_1.
        assert_allclose(_clipped_chi2_survival(c, 1), 0.5 * stats.chi2.sf(c, 1))
        # Two bands: binomial mixture of chi^2_1 and chi^2_2.
        expected = 0.5 * stats.chi2.sf(c, 1) + 0.25 * stats.chi2.sf(c, 2)
        assert_allclose(_clipped_chi2_survival(c, 2), expected)

        # At zero the survival equals the probability that at least one band
        # is positive, 1 - 2**-n.
        for n in (1, 2, 3, 6):
            self.assertAlmostEqual(float(_clipped_chi2_survival(0.0, n)), 1 - 2.0**-n)

        # Strictly decreasing.
        grid = np.linspace(0, 30, 200)
        self.assertTrue(np.all(np.diff(_clipped_chi2_survival(grid, 3)) < 0))

    def test_clipped_chi2_survival_matches_simulation(self):
        rng = np.random.default_rng(1)
        n = 3
        y = rng.standard_normal((2_000_000, n))
        coadd = np.sum(np.clip(y, 0, None) ** 2, axis=1)
        for thr in (2.0, 8.0, 12.0):
            empirical = np.mean(coadd > thr)
            assert_allclose(empirical, float(_clipped_chi2_survival(thr, n)), rtol=0.06)

    def test_chi2_to_sigma(self):
        n = 3
        c = np.array([1.0, 5.0, 20.0])
        assert_allclose(_chi2_to_sigma(c, n), stats.norm.isf(_clipped_chi2_survival(c, n)))
        # Works on a scalar as well as an array.
        self.assertAlmostEqual(float(_chi2_to_sigma(5.0, n)), float(_chi2_to_sigma(c, n)[1]))
        # Monotonically increasing.
        grid = np.linspace(0.1, 50, 100)
        self.assertTrue(np.all(np.diff(_chi2_to_sigma(grid, n)) > 0))
        # A bright core stays finite instead of overflowing to infinity.
        saturated = float(_chi2_to_sigma(1e6, n))
        self.assertTrue(np.isfinite(saturated))
        self.assertGreater(saturated, 30.0)

    def test_sigma_to_chi2_inverts_chi2_to_sigma(self):
        n = 3
        thresholds = (2.0, 3.0, 5.0, 8.0)
        for s in thresholds:
            c = _sigma_to_chi2(s, n)
            self.assertAlmostEqual(float(_chi2_to_sigma(c, n)), s, places=4)
            # The threshold carries the requested upper-tail probability.
            self.assertAlmostEqual(float(_clipped_chi2_survival(c, n)), float(stats.norm.sf(s)), places=6)
        # Increasing in sigma.
        values = [_sigma_to_chi2(s, n) for s in thresholds]
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_build_significance_map(self):
        starlets, sigma = _build_detection_starlets(self.images, self.variance, scales=3)
        smap = _build_significance_map(starlets, sigma, first_scale=1)

        # The finest ``first_scale`` scales and the coarse residual are
        # dropped, and a chi coadd plane is appended to the bands.
        self.assertEqual(smap.shape, (2, self.n_bands + 1) + self.shape)
        self.assertEqual(smap.dtype, np.float32)

        # The single-band planes are the standardized coefficients.
        standardized = starlets[1:-1] / sigma[1:-1, :, None, None]
        assert_allclose(smap[:, : self.n_bands], standardized, rtol=1e-5)

        # The final plane is the clipped chi coadd of those planes.
        chi = np.sqrt(np.sum(np.clip(standardized, 0, None) ** 2, axis=1))
        assert_allclose(smap[:, self.n_bands], chi, rtol=1e-5)

    def test_find_peak_candidates(self):
        ny, nx = 40, 40
        n_bands = 2
        significance_map = np.zeros((1, n_bands + 1, ny, nx), dtype=np.float32)
        yy, xx = np.mgrid[0:ny, 0:nx]

        # A 10-sigma bump in band 0.
        band0_center = (25, 12)
        significance_map[0, 0] = 10.0 * np.exp(
            -((yy - band0_center[0]) ** 2 + (xx - band0_center[1]) ** 2) / (2 * 2.0**2)
        )
        # A bump in the chi plane whose peak maps back to 9 sigma.
        chi_center = (8, 30)
        chi_amp = np.sqrt(_sigma_to_chi2(9.0, n_bands))
        significance_map[0, n_bands] = chi_amp * np.exp(
            -((yy - chi_center[0]) ** 2 + (xx - chi_center[1]) ** 2) / (2 * 2.0**2)
        )

        candidates = _find_peak_candidates(
            significance_map,
            min_separation=1,
            min_area=1,
            peak_thresh=5,
            footprint_thresh=3,
            first_scale=1,
        )
        self.assertEqual(candidates.dtype, CANDIDATE_DTYPE)
        # The empty middle band contributes nothing.
        self.assertEqual(np.sum(candidates["band"] == 1), 0)
        # Candidates are labeled with the offset starlet scale.
        assert_array_equal(np.unique(candidates["scale"]), [1])

        band0 = candidates[candidates["band"] == 0]
        self.assertEqual(len(band0), 1)
        self.assertEqual((int(band0["y"][0]), int(band0["x"][0])), band0_center)
        self.assertAlmostEqual(float(band0["flux"][0]), 10.0, places=4)

        chi = candidates[candidates["band"] == n_bands]
        self.assertEqual(len(chi), 1)
        self.assertEqual((int(chi["y"][0]), int(chi["x"][0])), chi_center)
        # The chi peak is reported in sigma.
        self.assertAlmostEqual(float(chi["flux"][0]), 9.0, places=3)

    def test_find_peak_candidates_empty(self):
        significance_map = np.zeros((2, 3, 32, 32), dtype=np.float32)
        candidates = _find_peak_candidates(significance_map, peak_thresh=5, footprint_thresh=3)
        self.assertEqual(len(candidates), 0)
