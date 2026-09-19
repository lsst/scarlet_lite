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
    DETECTION_DTYPE,
    POSITION_DTYPE,
    _assign_peaks,
    _build_detection_starlets,
    _build_footprints,
    _build_significance_map,
    _chi2_log_survival,
    _chi2_to_sigma,
    _chi_to_sigma,
    _collapse_positions,
    _find_peak_candidates,
    _link_radius,
    _plane_flags,
    _starlet_scale_factors,
    bbox_to_bounds,
    bounds_to_bbox,
    detect_footprints,
    detect_peaks,
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

        # Three footprints: the two source blends plus the flat 0.5 patch at
        # [30:32, 40]. The watershed seeds a peak on the patch's plateau, so
        # the footprint is kept rather than dropped for having no peak.
        self.assertEqual(len(footprints), 3)
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

    def test_chi2_log_survival(self):
        c = np.array([0.5, 2.0, 8.0, 20.0])
        # For one band the coadd is zero half the time and a chi^2_1 deviate
        # otherwise, so the survival function is 0.5 * chi2_1.
        assert_allclose(np.exp(_chi2_log_survival(c, 1)), 0.5 * stats.chi2.sf(c, 1))
        # Two bands: binomial mixture of chi^2_1 and chi^2_2.
        expected = 0.5 * stats.chi2.sf(c, 1) + 0.25 * stats.chi2.sf(c, 2)
        assert_allclose(np.exp(_chi2_log_survival(c, 2)), expected)

        # At zero the survival equals the probability that at least one band
        # is positive, 1 - 2**-n.
        for n in (1, 2, 3, 6):
            self.assertAlmostEqual(float(np.exp(_chi2_log_survival(0.0, n))), 1 - 2.0**-n)

        # Strictly decreasing, and finite in log space where the survival
        # function itself has underflowed to zero.
        grid = np.linspace(0, 30, 200)
        self.assertTrue(np.all(np.diff(_chi2_log_survival(grid, 3)) < 0))
        self.assertTrue(np.isfinite(float(_chi2_log_survival(1e6, 3))))

    def test_chi2_log_survival_matches_simulation(self):
        rng = np.random.default_rng(1)
        n = 3
        y = rng.standard_normal((2_000_000, n))
        coadd = np.sum(np.clip(y, 0, None) ** 2, axis=1)
        for thr in (2.0, 8.0, 12.0):
            empirical = np.mean(coadd > thr)
            assert_allclose(empirical, float(np.exp(_chi2_log_survival(thr, n))), rtol=0.06)

    def test_chi2_to_sigma(self):
        n = 3
        c = np.array([1.0, 5.0, 20.0])
        assert_allclose(_chi2_to_sigma(c, n), stats.norm.isf(np.exp(_chi2_log_survival(c, n))))
        # Works on a scalar as well as an array.
        self.assertAlmostEqual(float(_chi2_to_sigma(5.0, n)), float(_chi2_to_sigma(c, n)[1]))
        # Monotonically increasing.
        grid = np.linspace(0.1, 50, 100)
        self.assertTrue(np.all(np.diff(_chi2_to_sigma(grid, n)) > 0))
        # A bright core keeps climbing rather than saturating: the log-space
        # mapping stays finite and monotonic however bright the pixel.
        bright = float(_chi2_to_sigma(1e6, n))
        self.assertTrue(np.isfinite(bright))
        self.assertGreater(bright, 900.0)

    def test_chi_to_sigma(self):
        n = 3
        chi = np.array([0.0, 1.0, 3.0, 10.0, 80.0], dtype=np.float32)
        # The lookup table reproduces the exact chi**2 -> sigma mapping, in
        # both the dense (< 64) and the geometric (> 64) regimes.
        assert_allclose(_chi_to_sigma(chi, n), _chi2_to_sigma(chi.astype(float) ** 2, n), atol=1e-4)
        # The dtype of the input is preserved.
        self.assertEqual(_chi_to_sigma(chi, n).dtype, np.float32)
        # NaN in, NaN out; finite pixels are untouched.
        chi_nan = chi.copy()
        chi_nan[2] = np.nan
        out = _chi_to_sigma(chi_nan, n)
        self.assertTrue(np.isnan(out[2]))
        self.assertFalse(np.any(np.isnan(out[[0, 1, 3, 4]])))

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

        # The final plane is the clipped chi coadd mapped to sigma, so it is
        # on the same footing as the single-band planes.
        chi = np.sqrt(np.sum(np.clip(standardized, 0, None) ** 2, axis=1))
        assert_allclose(smap[:, self.n_bands], _chi_to_sigma(chi, self.n_bands), rtol=1e-5)

    def test_find_peak_candidates(self):
        ny, nx = 40, 40
        n_bands = 2
        # Every plane of a significance map is already in sigma, so a bump
        # is read back at its own amplitude.
        significance_map = np.zeros((1, n_bands + 1, ny, nx), dtype=np.float32)
        yy, xx = np.mgrid[0:ny, 0:nx]

        # A 10-sigma bump in band 0.
        band0_center = (25, 12)
        significance_map[0, 0] = 10.0 * np.exp(
            -((yy - band0_center[0]) ** 2 + (xx - band0_center[1]) ** 2) / (2 * 2.0**2)
        )
        # A 9-sigma bump in the chi plane.
        chi_center = (8, 30)
        significance_map[0, n_bands] = 9.0 * np.exp(
            -((yy - chi_center[0]) ** 2 + (xx - chi_center[1]) ** 2) / (2 * 2.0**2)
        )

        candidates, footprint_mask = _find_peak_candidates(
            significance_map,
            min_separation=1,
            min_area=1,
            peak_thresh=5,
            footprint_thresh=3,
            first_scale=1,
        )
        self.assertEqual(candidates.dtype, CANDIDATE_DTYPE)
        # The mask is the union of both planes' footprints above
        # footprint_thresh.
        self.assertEqual(footprint_mask.shape, (ny, nx))
        assert_array_equal(footprint_mask, np.any(significance_map[0] > 3, axis=0))
        # The empty middle band contributes nothing.
        self.assertEqual(np.sum(candidates["band"] == 1), 0)
        # Candidates are labeled with the offset starlet scale.
        assert_array_equal(np.unique(candidates["scale"]), [1])

        band0 = candidates[candidates["band"] == 0]
        self.assertEqual(len(band0), 1)
        self.assertEqual((int(band0["y"][0]), int(band0["x"][0])), band0_center)
        self.assertAlmostEqual(float(band0["flux"][0]), 10.0, places=4)
        # A single peak in a footprint never joins a brighter basin, so its
        # saddle is NaN.
        self.assertTrue(np.isnan(band0["saddle"][0]))

        chi = candidates[candidates["band"] == n_bands]
        self.assertEqual(len(chi), 1)
        self.assertEqual((int(chi["y"][0]), int(chi["x"][0])), chi_center)
        self.assertAlmostEqual(float(chi["flux"][0]), 9.0, places=3)

    def test_find_peak_candidates_kappa_culls_flank(self):
        # A bright bump with a low-prominence bump on its flank, in one plane.
        ny, nx = 5, 40
        x = np.arange(nx)
        profile = 10.0 * np.exp(-((x - 8) ** 2) / (2 * 2.0**2)) + 4.0 * np.exp(
            -((x - 14) ** 2) / (2 * 2.0**2)
        )
        significance_map = np.zeros((1, 2, ny, nx), dtype=np.float32)
        significance_map[0, 0, 2] = profile

        # kappa=0 keeps both local maxima.
        low, _ = _find_peak_candidates(
            significance_map, min_separation=0, min_area=1, peak_thresh=1, footprint_thresh=0.5, kappa=0.0
        )
        self.assertEqual(np.sum(low["band"] == 0), 2)

        # kappa=3 culls the flank bump; it rises less than 3 sigma above the
        # saddle to the brighter peak.
        high, _ = _find_peak_candidates(
            significance_map, min_separation=0, min_area=1, peak_thresh=1, footprint_thresh=0.5, kappa=3.0
        )
        band0 = high[high["band"] == 0]
        self.assertEqual(len(band0), 1)
        self.assertEqual(int(band0["x"][0]), 8)

    def test_find_peak_candidates_empty(self):
        significance_map = np.zeros((2, 3, 32, 32), dtype=np.float32)
        candidates, footprint_mask = _find_peak_candidates(
            significance_map, peak_thresh=5, footprint_thresh=3
        )
        self.assertEqual(len(candidates), 0)
        self.assertEqual(footprint_mask.shape, (32, 32))
        self.assertFalse(footprint_mask.any())

    def test_find_peak_candidates_mask_excludes_peakless(self):
        # A significant blob with no pixel above peak_thresh yields no
        # candidate and contributes nothing to the mask.
        significance_map = np.zeros((1, 1, 20, 20), dtype=np.float32)
        significance_map[0, 0, 5:9, 5:9] = 4.0
        significance_map[0, 0, 12:16, 12:16] = 8.0
        candidates, footprint_mask = _find_peak_candidates(
            significance_map, min_area=1, peak_thresh=5, footprint_thresh=3
        )
        self.assertEqual(len(candidates), 1)
        truth = np.zeros((20, 20), dtype=bool)
        truth[12:16, 12:16] = True
        assert_array_equal(footprint_mask, truth)

    def _make_peaks(self, rows):
        """Build detections from ``(y, x, peak_sigma)`` rows."""
        peaks = np.zeros(len(rows), dtype=DETECTION_DTYPE)
        for i, (y, x, peak_sigma) in enumerate(rows):
            peaks[i]["y"] = y
            peaks[i]["x"] = x
            peaks[i]["peak_sigma"] = peak_sigma
            peaks[i]["footprint"] = -1
        return peaks

    def test_build_footprints(self):
        origin = (100, 200)
        mask = np.zeros((20, 30), dtype=bool)
        # Two components: one holding two peaks, one holding a single peak,
        # and a third with no peak at all.
        mask[2:6, 3:12] = True
        mask[10:14, 20:25] = True
        mask[15:18, 2:5] = True
        peaks = self._make_peaks(
            [
                (100 + 3, 200 + 5, 6.0),
                (100 + 11, 200 + 22, 9.0),
                (100 + 4, 200 + 10, 7.5),
            ]
        )

        footprints = _build_footprints(mask, peaks, origin)

        # The peakless component is dropped.
        self.assertEqual(len(footprints), 2)
        # Every peak is placed in the footprint that contains it.
        for row, peak in enumerate(peaks):
            footprint = footprints[peak["footprint"]]
            self.assertTrue(footprint.bbox.contains((peak["y"], peak["x"])))
            self.assertTrue(footprint.data[peak["y"] - footprint.yx0[0], peak["x"] - footprint.yx0[1]])
        self.assertEqual(peaks["footprint"][0], peaks["footprint"][2])
        self.assertNotEqual(peaks["footprint"][0], peaks["footprint"][1])

        # Peaks within a footprint are brightest first with the detection's
        # peak_sigma as their flux.
        pair = footprints[peaks["footprint"][0]]
        self.assertEqual([(p.y, p.x) for p in pair.peaks], [(104, 210), (103, 205)])
        assert_allclose([p.flux for p in pair.peaks], [7.5, 6.0])
        single = footprints[peaks["footprint"][1]]
        self.assertEqual([(p.y, p.x) for p in single.peaks], [(111, 222)])

        # The footprint pixels are the component of the mask, in absolute
        # coordinates.
        self.assertBoxEqual(pair.bbox, Box((4, 9), origin=(102, 203)))
        assert_array_equal(pair.data, mask[2:6, 3:12])
        self.assertBoxEqual(single.bbox, Box((4, 5), origin=(110, 220)))

    def test_build_footprints_empty(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:4, 2:4] = True
        self.assertEqual(_build_footprints(mask, np.empty(0, dtype=DETECTION_DTYPE)), [])

    def test_build_footprints_peak_outside_mask(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:4, 2:4] = True
        peaks = self._make_peaks([(6, 6, 5.0)])
        with self.assertRaises(RuntimeError):
            _build_footprints(mask, peaks)

    def _make_candidates(self, rows):
        """Build a candidate array from ``(y, x, band, scale, flux)`` rows."""
        candidates = np.zeros(len(rows), dtype=CANDIDATE_DTYPE)
        for i, (y, x, band, scale, flux) in enumerate(rows):
            candidates[i] = (y, x, band, scale, flux, np.nan, -1)
        return candidates

    def _make_positions(self, rows):
        """Build positions from ``(y, x, scale, peak_sigma, plane_flags)``."""
        positions = np.zeros(len(rows), dtype=POSITION_DTYPE)
        for i, (y, x, scale, peak_sigma, plane_flags) in enumerate(rows):
            positions[i]["y"] = y
            positions[i]["x"] = x
            positions[i]["scale"] = scale
            positions[i]["peak_sigma"] = peak_sigma
            positions[i]["flux"] = peak_sigma
            positions[i]["plane_flags"] = plane_flags
            positions[i]["peak"] = -1
        return positions

    def test_link_radius(self):
        # The PSF FWHM floors the radius at fine scales.
        self.assertEqual(_link_radius(1, 3.5), 3.5)
        # The 2**scale term dominates at coarse scales.
        self.assertEqual(_link_radius(3, 3.5), 8.0)
        # Vectorized over an array of scales.
        assert_array_equal(_link_radius(np.array([1, 3]), 3.5), [3.5, 8.0])

    def test_plane_flags(self):
        # plane = (scale - first_scale) * (n_bands + 1) + band, so band 0 at
        # first_scale is plane 0, the chi plane (band 3) is plane 3, and band 0
        # at the next scale is plane 4.
        candidates = self._make_candidates(
            [
                (0, 0, 0, 1, 1.0),
                (0, 0, 3, 1, 1.0),
                (0, 0, 0, 2, 1.0),
            ]
        )
        flags = _plane_flags(candidates, n_bands=3, first_scale=1)
        assert_array_equal(flags, [1 << 0, 1 << 3, 1 << 4])

        # More than 63 planes cannot fit in an int64 bitmask.
        too_many = self._make_candidates([(0, 0, 0, 20, 1.0)])
        with self.assertRaisesRegex(ValueError, "at most 63 planes"):
            _plane_flags(too_many, n_bands=3, first_scale=1)

    def test_collapse_positions(self):
        # Two candidates at the same pixel in different bands and scales
        # collapse to one position; a third at another pixel stays separate.
        candidates = self._make_candidates(
            [
                (10, 10, 0, 1, 8.0),
                (10, 10, 2, 2, 5.0),
                (10, 14, 1, 1, 6.0),
            ]
        )
        positions = _collapse_positions(candidates, n_bands=3, first_scale=1)
        self.assertEqual(positions.dtype, POSITION_DTYPE)
        self.assertEqual(len(positions), 2)

        merged = positions[(positions["y"] == 10) & (positions["x"] == 10)][0]
        self.assertEqual(merged["band_flags"], (1 << 0) | (1 << 2))
        self.assertEqual(merged["scale_flags"], (1 << 1) | (1 << 2))
        self.assertEqual(merged["n_candidates"], 2)
        # The finest scale wins the position flux; peak_sigma is the brightest.
        self.assertEqual(merged["scale"], 1)
        self.assertAlmostEqual(float(merged["flux"]), 8.0)
        self.assertAlmostEqual(float(merged["peak_sigma"]), 8.0)
        # ``position`` is filled in place, sending the two co-located
        # candidates to one position and the third to another.
        self.assertEqual(candidates["position"][0], candidates["position"][1])
        self.assertNotEqual(candidates["position"][0], candidates["position"][2])

    def test_assign_peaks_distinct(self):
        # Two positions within the link radius that were both peaks in the same
        # plane (they share a plane_flags bit) are distinct and stay two peaks.
        positions = self._make_positions([(10, 10, 1, 8.0, 0b1), (10, 12, 1, 7.0, 0b1)])
        seeds, peak_of = _assign_peaks(positions, psf_fwhm=3.5)
        self.assertEqual(len(seeds), 2)
        assert_array_equal(np.sort(peak_of), [0, 1])

        # The same two positions sharing no plane bit merge into one peak.
        positions = self._make_positions([(10, 10, 1, 8.0, 0b1), (10, 12, 1, 7.0, 0b10)])
        seeds, peak_of = _assign_peaks(positions, psf_fwhm=3.5)
        self.assertEqual(len(seeds), 1)
        assert_array_equal(peak_of, [0, 0])

        # Positions beyond the link radius never merge, distinct or not.
        positions = self._make_positions([(10, 10, 1, 8.0, 0b1), (10, 40, 1, 7.0, 0b10)])
        seeds, _ = _assign_peaks(positions, psf_fwhm=3.5)
        self.assertEqual(len(seeds), 2)

    def test_assign_peaks_blend_policy(self):
        # Two distinct fine-scale peaks with a coarse-scale position between
        # them, eligible to join either. ``n_linked`` records the ambiguity.
        rows = [(10, 10, 1, 10.0, 0b1), (10, 16, 1, 10.0, 0b1), (10, 13, 2, 5.0, 0b100)]

        nearest = self._make_positions(rows)
        seeds, peak_of = _assign_peaks(nearest, psf_fwhm=3.5, blend_policy="nearest")
        self.assertEqual(len(seeds), 2)
        # The ambiguous position joins the nearest seed, not dropped.
        self.assertGreaterEqual(peak_of[2], 0)
        self.assertEqual(nearest["n_linked"][2], 2)

        drop = self._make_positions(rows)
        seeds, peak_of = _assign_peaks(drop, psf_fwhm=3.5, blend_policy="drop")
        self.assertEqual(len(seeds), 2)
        # With "drop" the ambiguous position is left unassigned.
        self.assertEqual(peak_of[2], -1)
        self.assertEqual(drop["n_linked"][2], 2)

        with self.assertRaisesRegex(ValueError, "blend_policy"):
            _assign_peaks(self._make_positions([(1, 1, 1, 1.0, 0b1)]), psf_fwhm=3.5, blend_policy="bogus")

    def test_detect_peaks_groups_sources(self):
        result = detect_peaks(self.images, self.variance, scales=3, peak_thresh=5, footprint_thresh=3)
        self.assertEqual(result.peaks.dtype, DETECTION_DTYPE)
        self.assertEqual(result.positions.dtype, POSITION_DTYPE)

        # Every injected source is recovered exactly once, within a pixel.
        self.assertEqual(len(result.peaks), len(self.centers))
        for cy, cx in self.centers:
            matched = [p for p in result.peaks if abs(p["y"] - cy) <= 1 and abs(p["x"] - cx) <= 1]
            self.assertEqual(len(matched), 1)

        # The many per-band, per-scale candidates collapse into each source.
        peak = result.peaks[0]
        self.assertGreater(peak["n_candidates"], 1)
        self.assertGreaterEqual(peak["peak_sigma"], peak["flux"])
        # These bright sources are seen in all 3 bands and the chi coadd.
        expected_bands = (1 << self.n_bands + 1) - 1
        self.assertEqual(peak["band_flags"], expected_bands)
        self.assertNotEqual(peak["scale_flags"], 0)

        # Each well-separated source gets its own footprint holding its peak.
        self.assertEqual(len(result.footprints), len(self.centers))
        assert_array_equal(np.sort(result.peaks["footprint"]), np.arange(len(self.centers)))
        for peak in result.peaks:
            footprint = result.footprints[peak["footprint"]]
            self.assertEqual(len(footprint.peaks), 1)
            self.assertEqual((footprint.peaks[0].y, footprint.peaks[0].x), (peak["y"], peak["x"]))
            self.assertEqual(footprint.peaks[0].flux, peak["peak_sigma"])
            self.assertTrue(footprint.data[peak["y"] - footprint.yx0[0], peak["x"] - footprint.yx0[1]])

    def test_detect_peaks_footprints_offset_origin(self):
        origin = (1000, 2000)
        result = detect_peaks(
            self.images, self.variance, scales=3, peak_thresh=5, footprint_thresh=3, origin=origin
        )
        reference = detect_peaks(self.images, self.variance, scales=3, peak_thresh=5, footprint_thresh=3)
        self.assertEqual(len(result.footprints), len(reference.footprints))
        for footprint, ref in zip(result.footprints, reference.footprints):
            self.assertBoxEqual(footprint.bbox, ref.bbox + origin)
            assert_array_equal(footprint.data, ref.data)
            self.assertEqual(
                [(p.y, p.x) for p in footprint.peaks],
                [(p.y + origin[0], p.x + origin[1]) for p in ref.peaks],
            )

    def test_detect_peaks_blend_shares_footprint(self):
        # Two sources close enough that their footprints touch but far
        # enough apart to be resolved land in one footprint with two peaks.
        rng = np.random.default_rng(7)
        shape = (48, 48)
        yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
        images = rng.standard_normal((self.n_bands,) + shape).astype(np.float32)
        for cy, cx in [(24, 18), (24, 30)]:
            images += 40.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5**2)).astype(np.float32)
        variance = np.ones_like(images)
        result = detect_peaks(images, variance, scales=3, peak_thresh=5, footprint_thresh=3)
        self.assertEqual(len(result.peaks), 2)
        self.assertEqual(len(result.footprints), 1)
        assert_array_equal(result.peaks["footprint"], 0)
        self.assertEqual(len(result.footprints[0].peaks), 2)

    def test_detect_peaks_retraceable(self):
        result = detect_peaks(self.images, self.variance, scales=3, peak_thresh=5, footprint_thresh=3)
        # Every position is assigned to a peak in this clean blend.
        assert_array_equal(result.candidates["position"] >= 0, True)
        for peak_id, peak in enumerate(result.peaks):
            member_positions = np.nonzero(result.positions["peak"] == peak_id)[0]
            self.assertEqual(len(member_positions), peak["n_positions"])

            member_candidates = np.concatenate(
                [np.nonzero(result.candidates["position"] == pid)[0] for pid in member_positions]
            )
            self.assertEqual(len(member_candidates), peak["n_candidates"])

    def test_detect_peaks_empty(self):
        # A pure-noise floor with a threshold nothing clears yields no peaks
        # but still well-formed tables.
        images = np.zeros((self.n_bands,) + self.shape, dtype=np.float32)
        variance = np.ones_like(images)
        result = detect_peaks(images, variance, scales=3, peak_thresh=5, footprint_thresh=3)
        self.assertEqual(len(result.candidates), 0)
        self.assertEqual(len(result.positions), 0)
        self.assertEqual(len(result.peaks), 0)
        self.assertEqual(result.footprints, [])
        self.assertEqual(result.peaks.dtype, DETECTION_DTYPE)
        self.assertEqual(result.positions.dtype, POSITION_DTYPE)
