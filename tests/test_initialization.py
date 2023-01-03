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

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from scarlet_lite.initialization import trim_morphology


class TestInitialization(unittest.TestCase):
    def test_trim_morphology(self):
        # Test default parameters
        morph = np.zeros((50, 50))
        morph[10:15, 12:27] = 1
        trimmed, trimmed_box = trim_morphology(morph)
        assert_array_equal(trimmed, morph)
        self.assertTupleEqual(trimmed_box.origin, (5, 7))
        self.assertTupleEqual(trimmed_box.shape, (15, 25))

        # Test with parameters specified
        morph = np.full((50, 50), .1)
        morph[10:15, 12:27] = 1
        truth = np.zeros(morph.shape)
        truth[10:15, 12:27] = 1
        trimmed, trimmed_box = trim_morphology(morph, 0.5, 1)
        assert_array_equal(trimmed, truth)
        self.assertTupleEqual(trimmed_box.origin, (9, 11))
        self.assertTupleEqual(trimmed_box.shape, (7, 17))
