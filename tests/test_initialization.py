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
from unittest.mock import patch

import numpy as np
from deprecated.sphinx import deprecated
from lsst.scarlet.lite import Box, Image, Observation
from lsst.scarlet.lite.initialization import (
    FactorizedInitialization,
    FactorizedWaveletInitialization,
    init_monotonic_morph,
    multifit_spectra,
    trim_morphology,
)
from lsst.scarlet.lite.operators import Monotonicity, prox_monotonic_mask
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.signal import convolve as scipy_convolve
from utils import ObservationData, ScarletTestCase


class TestInitialization(ScarletTestCase):
    def setUp(self) -> None:
        yx0 = (1000, 2000)
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
        model_psf = integrated_circular_gaussian(sigma=0.8)
        self.detect = np.sum(data["images"], axis=0)
        self.centers = np.array([data["catalog"]["y"], data["catalog"]["x"]]).T + np.array(yx0)
        bands = data["filters"]
        self.observation = Observation(
            Image(data["images"], bands=bands, yx0=yx0),
            Image(data["variance"], bands=bands, yx0=yx0),
            Image(1 / data["variance"], bands=bands, yx0=yx0),
            data["psfs"],
            model_psf[None],
            bands=bands,
        )

    def test_trim_morphology(self):
        # Default parameters: returns a tight bbox around the non-zero
        # support of the input.
        morph = np.zeros((50, 50)).astype(np.float32)
        morph[10:15, 12:27] = 1
        trimmed, trimmed_box = trim_morphology(morph)
        assert_array_equal(trimmed, morph)
        self.assertTupleEqual(trimmed_box.origin, (10, 12))
        self.assertTupleEqual(trimmed_box.shape, (5, 15))
        self.assertEqual(trimmed.dtype, np.float32)

        # With a threshold: pixels at or below the threshold are zeroed,
        # and the bbox is the tight box around what remains.
        morph = np.full((50, 50), 0.1).astype(np.float32)
        morph[10:15, 12:27] = 1
        truth = np.zeros(morph.shape)
        truth[10:15, 12:27] = 1
        trimmed, trimmed_box = trim_morphology(morph, 0.5)
        assert_array_equal(trimmed, truth)
        self.assertTupleEqual(trimmed_box.origin, (10, 12))
        self.assertTupleEqual(trimmed_box.shape, (5, 15))
        self.assertEqual(trimmed.dtype, np.float32)

    def test_init_monotonic_mask(self):
        full_box = self.observation.bbox
        center = self.centers[0]
        local_center = (center[0] - full_box.origin[0], center[1] - full_box.origin[1])

        # Default parameters
        bbox, morph = init_monotonic_morph(self.detect.copy(), center, full_box)
        self.assertBoxEqual(bbox, Box((38, 29), (1014, 2000)))
        _, masked_morph, _ = prox_monotonic_mask(self.detect.copy(), local_center, max_iter=0)
        assert_array_equal(morph, masked_morph / np.max(masked_morph))
        self.assertEqual(morph.dtype, np.float32)

        # Non-zero threshold AND non-zero padding. This combination
        # exercises the path-1 trim_morphology call AND the post-trim
        # padding step; if those ever double up again (audit I-2), the
        # bbox below would grow to (34, 28) at origin (1017, 2000)
        # rather than (30, 25) at (1019, 2001).
        bbox, morph = init_monotonic_morph(
            self.detect.copy(),
            center,
            full_box,
            2,  # padding
            False,  # normalize
            None,  # monotonicity
            0.2,  # threshold
        )
        self.assertBoxEqual(bbox, Box((30, 25), (1019, 2001)))
        # Remove pixels below the threshold
        truth = masked_morph.copy()
        truth[truth < 0.2] = 0
        assert_array_equal(morph, truth)
        self.assertEqual(morph.dtype, np.float32)

        # Test an empty morphology
        bbox, morph = init_monotonic_morph(np.zeros(self.detect.shape), center, full_box)
        self.assertBoxEqual(bbox, Box((0, 0)))
        self.assertIsNone(morph)

    def test_init_monotonic_weighted(self):
        full_box = self.observation.bbox
        center = self.centers[0]
        local_center = (center[0] - full_box.origin[0], center[1] - full_box.origin[1])
        monotonicity = Monotonicity((101, 101))

        # Default parameters
        bbox, morph = init_monotonic_morph(self.detect.copy(), center, full_box, monotonicity=monotonicity)
        truth = monotonicity(self.detect.copy(), local_center)
        truth[truth < 0] = 0
        truth = truth / np.max(truth)
        self.assertBoxEqual(bbox, Box((58, 48), origin=(1000, 2000)))
        assert_array_equal(morph, truth)
        self.assertEqual(morph.dtype, np.float32)

        # Non-zero threshold AND non-zero padding. trim_morphology is
        # always called on path 2; pairing that with padding > 0 makes
        # the regression for audit I-2 (double padding) observable
        # rather than masked by clipping or padding=0.
        bbox, morph = init_monotonic_morph(
            self.detect.copy(),
            center,
            full_box,
            2,  # padding
            False,  # normalize
            monotonicity,  # monotonicity
            0.2,  # threshold
        )
        truth = monotonicity(self.detect.copy(), local_center)
        truth[truth < 0.2] = 0
        self.assertBoxEqual(bbox, Box((49, 47), origin=(1008, 2001)))
        assert_array_equal(morph, truth)
        self.assertEqual(morph.dtype, np.float32)

        # Test zero morphology
        zeros = np.zeros(self.detect.shape)
        bbox, morph = init_monotonic_morph(zeros, center, full_box, monotonicity=monotonicity)
        self.assertBoxEqual(bbox, Box((0, 0), (1000, 2000)))
        self.assertIsNone(morph)

    def test_multifit_spectra(self):
        bands = ("g", "r", "i")
        variance = np.ones((3, 35, 35), dtype=np.float32)
        weights = 1 / variance
        psfs = np.array([integrated_circular_gaussian(sigma=sigma) for sigma in [1.05, 0.9, 1.2]])
        psfs = psfs.astype(np.float32)
        model_psf = integrated_circular_gaussian(sigma=0.8).astype(np.float32)

        # The spectrum of each source
        spectra = np.array(
            [
                [31, 10, 0],
                [0, 5, 20],
                [15, 8, 3],
                [20, 3, 4],
                [0, 30, 60],
            ],
            dtype=np.float32,
        )

        # Use a point source for all of the sources
        morphs = [
            integrated_circular_gaussian(sigma=sigma).astype(np.float32)
            for sigma in [0.8, 3.1, 1.1, 2.1, 1.5]
        ]
        # Make the second component a disk component
        morphs[1] = scipy_convolve(morphs[1], model_psf, mode="same")

        # Give the first two components the same center, and unique centers
        # for the remaining sources
        centers = [
            (10, 12),
            (10, 12),
            (20, 23),
            (20, 10),
            (25, 20),
        ]

        # Create the Observation
        test_data = ObservationData(bands, psfs, spectra, morphs, centers, model_psf, dtype=np.float32)
        observation = Observation(
            test_data.convolved,
            variance,
            weights,
            psfs,
            model_psf[None],
            bands=bands,
        )

        fit_spectra = multifit_spectra(observation, test_data.morphs)
        self.assertEqual(fit_spectra.dtype, spectra.dtype)
        assert_almost_equal(fit_spectra, spectra, decimal=5)

    def test_psf_component_at_boundary(self):
        """``get_psf_component`` must extract the surviving region of
        the model PSF when the source center is close enough to the
        observation boundary that the PSF box is clipped.

        Audit finding I-3: the original code created the Image with
        ``yx0=bbox.origin`` (the *intersection's* origin) instead of
        the original PSF origin, so ``[bbox]`` returned the top-left
        of the PSF rather than the portion of the PSF that survived
        the clip. The PSF was therefore spatially misaligned with the
        actual source center.
        """
        init = FactorizedInitialization(self.observation, self.centers)
        model_psf = self.observation.model_psf[0]

        # Case 1: positive psf_bbox.origin. Center at the top-left
        # corner of the observation (origin (1000, 2000)): the 15x15
        # model PSF (py=px=7) extends 7 rows above and 3 columns to
        # the left of the observation bbox, so 7 rows and 3 columns
        # are clipped. psf_bbox.origin = (993, 1997) — both positive.
        center = (1000, 2004)
        component = init.get_psf_component(center)
        self.assertBoxEqual(component.bbox, Box((8, 12), origin=(1000, 2000)))
        # The surviving region is psf[7:15, 3:15] — the bottom-right
        # of the PSF, not the top-left.
        assert_array_equal(component.morph, model_psf[7:15, 3:15])

        # Case 2: negative psf_bbox.origin. Build a synthetic
        # observation at origin (0, 0) and place a center near the
        # corner so psf_bbox.origin = (-5, -2) — both negative. The
        # negative-origin path must still produce the correct
        # surviving region. ``Box.slices`` rejects negative origins,
        # so this also guards against future refactors that would
        # call ``.slices`` on ``psf_bbox`` directly.
        bands = ("r",)
        shape = (30, 30)
        images = np.ones((1,) + shape, dtype=np.float32)
        variance = np.ones((1,) + shape, dtype=np.float32)
        psfs = np.array([integrated_circular_gaussian(sigma=1.0)], dtype=np.float32)
        small_model_psf = integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        small_obs = Observation(
            Image(images, bands=bands, yx0=(0, 0)),
            Image(variance, bands=bands, yx0=(0, 0)),
            Image(1 / variance, bands=bands, yx0=(0, 0)),
            psfs,
            small_model_psf[None],
            bands=bands,
        )
        small_init = FactorizedInitialization(small_obs, [(2, 5)])
        component = small_init.get_psf_component((2, 5))
        self.assertBoxEqual(component.bbox, Box((10, 13), origin=(0, 0)))
        assert_array_equal(component.morph, small_model_psf[5:15, 2:15])

        # Case 3: PSF footprint does not overlap the observation at
        # all -> raise an informative error rather than silently
        # producing a degenerate component.
        with self.assertRaises(ValueError):
            small_init.get_psf_component((-100, -100))

    def test_get_single_component_zero_convolved(self):
        """``get_single_component`` must produce a finite spectrum
        even when the convolved detection image is zero at the source
        center.

        Audit finding I-4: ``spectrum = images / convolved`` produces
        ``inf`` (or ``nan`` for 0/0) at zero-convolved pixels, and
        the subsequent ``spectrum[spectrum < 0] = 0`` does not catch
        either, so the component is initialized with non-finite
        flux.
        """
        init = FactorizedInitialization(self.observation, self.centers)
        center = (int(self.centers[0][0]), int(self.centers[0][1]))
        local_center = (
            center[0] - init.observation.bbox.origin[0],
            center[1] - init.observation.bbox.origin[1],
        )
        # Force the convolved detection image to zero at the source
        # center across all bands. The detection image still has flux
        # at this pixel, so ``init_monotonic_morph`` returns a valid
        # morph and the spectrum branch is exercised.
        init.convolved.data[:, local_center[0], local_center[1]] = 0

        thresh = np.mean(self.observation.noise_rms) * init.initial_bg_thresh
        component = init.get_single_component(center, init.detect.copy(), thresh, init.padding)
        assert component is not None
        np.testing.assert_array_equal(np.isfinite(component.spectrum), True)
        np.testing.assert_array_equal(component.spectrum >= 0, True)

    def test_factorized_chi2_init(self):
        # Test default parameters
        init = FactorizedInitialization(self.observation, self.centers)
        self.assertEqual(init.observation, self.observation)
        self.assertEqual(init.min_snr, 50)
        self.assertIsNone(init.monotonicity)
        self.assertEqual(init.disk_percentile, 25)
        self.assertEqual(init.thresh, 0.5)
        self.assertTupleEqual((init.py, init.px), (7, 7))
        self.assertEqual(len(init.sources), 7)
        for src in init.sources:
            self.assertEqual(src.get_model().dtype, np.float32)

        centers = tuple(tuple(center.astype(int)) for center in self.centers) + ((1000, 2004),)
        init = FactorizedInitialization(self.observation, centers)
        self.assertEqual(len(init.sources), 8)
        for src in init.sources:
            self.assertEqual(src.get_model().dtype, np.float32)

    @deprecated(
        version="v29.0",
        reason="FactorizedWaveletInitialization is deprecated and will be removed after v29.0",
    )
    def test_wavelet_init_source_falls_back_to_psf(self):
        """init_source must always return a Source with at least one
        component, even when individual init paths fail.

        Audit finding I-1: when get_single_component returned None or
        the two-component path produced all-zero spectra, ``components``
        was either left unbound (UnboundLocalError) or set to an empty
        list. Both cases must now fall back to a PSF component.
        """
        init = FactorizedWaveletInitialization(self.observation, self.centers)
        int_centers = [(int(round(c[0])), int(round(c[1]))) for c in self.centers]

        # Failure mode 1: get_single_component always returns None.
        # Centers hitting the single-component branch or the
        # two-component fallback must fall back to PSF rather than
        # raising or producing an empty Source.
        with patch.object(FactorizedWaveletInitialization, "get_single_component", return_value=None):
            for center in int_centers:
                source = init.init_source(center)
                self.assertGreater(len(source.components), 0)

        # Failure mode 2: two-component path returns all-zero spectra
        # for both bulge and disk -> empty components list, also a fall
        # back case.
        n_bands = len(self.observation.bands)
        with patch(
            "lsst.scarlet.lite.initialization.multifit_spectra",
            return_value=np.zeros((2, n_bands), dtype=np.float32),
        ):
            for center in int_centers:
                source = init.init_source(center)
                self.assertGreater(len(source.components), 0)

    @deprecated(
        version="v29.0",
        reason="FactorizedWaveletInitialization is deprecated and will be removed after v29.0",
    )
    def test_factorized_wavelet_init(self):
        # Test default parameters
        init = FactorizedWaveletInitialization(self.observation, self.centers)
        self.assertEqual(init.observation, self.observation)
        self.assertEqual(init.min_snr, 50)
        self.assertIsNone(init.monotonicity)
        self.assertTupleEqual((init.py, init.px), (7, 7))
        self.assertEqual(len(init.sources), 7)
        components = np.sum([len(src.components) for src in init.sources])
        self.assertEqual(components, 8)
        for src in init.sources:
            self.assertEqual(src.get_model().dtype, np.float32)
