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
from lsst.scarlet.lite import Box, Image, Observation
from lsst.scarlet.lite.initialization import (
    FactorizedChi2Initialization,
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
        # Test default parameters
        morph = np.zeros((50, 50)).astype(np.float32)
        morph[10:15, 12:27] = 1
        trimmed, trimmed_box = trim_morphology(morph)
        assert_array_equal(trimmed, morph)
        self.assertTupleEqual(trimmed_box.origin, (5, 7))
        self.assertTupleEqual(trimmed_box.shape, (15, 25))
        self.assertEqual(trimmed.dtype, np.float32)

        # Test with parameters specified
        morph = np.full((50, 50), 0.1).astype(np.float32)
        morph[10:15, 12:27] = 1
        truth = np.zeros(morph.shape)
        truth[10:15, 12:27] = 1
        trimmed, trimmed_box = trim_morphology(morph, 0.5, 1)
        assert_array_equal(trimmed, truth)
        self.assertTupleEqual(trimmed_box.origin, (9, 11))
        self.assertTupleEqual(trimmed_box.shape, (7, 17))
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

        # Specifying parameters
        bbox, morph = init_monotonic_morph(
            self.detect.copy(),
            center,
            full_box,
            0,  # padding
            False,  # normalizae
            None,  # monotonicity
            0.2,  # threshold
        )
        self.assertBoxEqual(bbox, Box((26, 21), (1021, 2003)))
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

        # Specify parameters
        bbox, morph = init_monotonic_morph(
            self.detect.copy(),
            center,
            full_box,
            0,  # padding
            False,  # normalize
            monotonicity,  # monotonicity
            0.2,  # threshold
        )
        truth = monotonicity(self.detect.copy(), local_center)
        truth[truth < 0.2] = 0
        self.assertBoxEqual(bbox, Box((45, 44), origin=(1010, 2003)))
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

    def test_factorized_chi2_init(self):
        # Test default parameters
        init = FactorizedChi2Initialization(self.observation, self.centers)
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
        init = FactorizedChi2Initialization(self.observation, centers)
        self.assertEqual(len(init.sources), 8)
        for src in init.sources:
            self.assertEqual(src.get_model().dtype, np.float32)

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
