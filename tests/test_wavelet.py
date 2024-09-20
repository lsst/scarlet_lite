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
from lsst.scarlet.lite.wavelet import (
    apply_wavelet_denoising,
    get_multiresolution_support,
    multiband_starlet_reconstruction,
    multiband_starlet_transform,
    starlet_reconstruction,
    starlet_transform,
)
from numpy.testing import assert_almost_equal
from utils import ScarletTestCase


class TestWavelet(ScarletTestCase):
    def setUp(self) -> None:
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        self.data = np.load(filename)

    def tearDown(self) -> None:
        del self.data

    def test_transform_inverse(self):
        image = np.sum(self.data["images"], axis=0)
        starlets = starlet_transform(image, scales=3)
        self.assertEqual(starlets.dtype, np.float32)

        # Test number of levels
        self.assertTupleEqual(starlets.shape, (4, 58, 48))

        # Test inverse
        inverse = starlet_reconstruction(starlets)
        assert_almost_equal(inverse, image, decimal=5)
        self.assertEqual(inverse.dtype, starlets.dtype)

        # Test using gen1 starlets
        starlets = starlet_transform(image, scales=3, generation=1)

        # Test number of levels
        self.assertTupleEqual(starlets.shape, (4, 58, 48))

        # Test inverse
        inverse = starlet_reconstruction(starlets, generation=1)
        assert_almost_equal(inverse, image, decimal=5)

    def test_multiband_transform(self):
        image = self.data["images"]
        starlets = multiband_starlet_transform(image, scales=3)
        self.assertEqual(starlets.dtype, np.float32)

        # Test number of levels
        self.assertTupleEqual(starlets.shape, (4, 5, 58, 48))

        # Test inverse
        inverse = multiband_starlet_reconstruction(starlets)
        assert_almost_equal(inverse, image, decimal=5)
        self.assertEqual(inverse.dtype, np.float32)

    def test_extras(self):
        # This is code that is not used in production,
        # but that might be used in the future,
        # so we test to prevent bitrot
        image = np.sum(self.data["images"].astype(float), axis=0)
        starlets = starlet_transform(image, scales=3)

        # Execute to ensure that the code runs
        get_multiresolution_support(image, starlets, 0.1)
        get_multiresolution_support(image, starlets, 0.1, image_type="space")
        apply_wavelet_denoising(image)
