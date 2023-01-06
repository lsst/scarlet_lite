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
from numpy.testing import assert_array_equal, assert_almost_equal
from scipy.signal import convolve as scipy_convolve

from scarlet_lite import Box, Image, Observation, Source
from scarlet_lite.initialization import (
    trim_morphology,
    init_monotonic_morph,
    multifit_spectra,
    FactorizedChi2Initialization,
)
from scarlet_lite.operators import prox_monotonic_mask, Monotonicity
from scarlet_lite.utils import integrated_circular_gaussian

from utils import ObservationData


class TestInitialization(unittest.TestCase):
    def test_source(self):
        morph = integrated_circular_gaussian(sigma=0.8)
