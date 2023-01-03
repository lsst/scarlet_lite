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

import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from scarlet_lite.operators import prox_monotonic_mask, Monotonicity
from scarlet_lite import Box
from scarlet_lite.initialization import trim_morphology, init_monotonic_morph


class TestInitialization(unittest.TestCase):
    def setUp(self) -> None:
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
        self.detect = np.sum(data["images"], axis=0)
        self.centers = np.array([data["catalog"]["y"], data["catalog"]["x"]]).T

    def test_trim_morphology(self):
        # Test default parameters
        morph = np.zeros((50, 50))
        morph[10:15, 12:27] = 1
        trimmed, trimmed_box = trim_morphology(morph)
        assert_array_equal(trimmed, morph)
        self.assertTupleEqual(trimmed_box.origin, (5, 7))
        self.assertTupleEqual(trimmed_box.shape, (15, 25))

        # Test with parameters specified
        morph = np.full((50, 50), 0.1)
        morph[10:15, 12:27] = 1
        truth = np.zeros(morph.shape)
        truth[10:15, 12:27] = 1
        trimmed, trimmed_box = trim_morphology(morph, 0.5, 1)
        assert_array_equal(trimmed, truth)
        self.assertTupleEqual(trimmed_box.origin, (9, 11))
        self.assertTupleEqual(trimmed_box.shape, (7, 17))

    def test_init_monotonic_mask(self):
        full_box = Box(self.detect.shape)
        center = self.centers[0]

        # Default parameters
        bbox, morph = init_monotonic_morph(self.detect.copy(), center, full_box)
        self.assertEqual(bbox, Box((38, 29), (14, 0)))
        _, masked_morph, _ = prox_monotonic_mask(self.detect.copy(), center, max_iter=0)
        assert_array_equal(morph, masked_morph / np.max(masked_morph))

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
        self.assertEqual(bbox, Box((26, 21), (21, 3)))
        # Remove pixels below the threshold
        truth = masked_morph.copy()
        truth[truth < 0.2] = 0
        assert_array_equal(morph, truth)

    def test_init_monotonic_weighted(self):
        full_box = Box(self.detect.shape)
        center = self.centers[0]
        monotonicity = Monotonicity((101, 101))

        # Default parameters
        bbox, morph = init_monotonic_morph(
            self.detect.copy(), center, full_box, monotonicity=monotonicity
        )
        truth = monotonicity(self.detect.copy(), center)
        truth[truth < 0] = 0
        truth = truth / np.max(truth)
        self.assertEqual(bbox, Box((58, 48), origin=(0, 0)))
        assert_array_equal(morph, truth)

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
        truth = monotonicity(self.detect.copy(), center)
        truth[truth < 0.2] = 0
        self.assertEqual(bbox, Box((45, 44), origin=(10, 3)))
        assert_array_equal(morph, truth)
