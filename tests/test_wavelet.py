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

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from scarlet_lite.wavelet import starlet_transform, starlet_reconstruction
from utils import ScarletTestCase


class TestWavelet(ScarletTestCase):
    def get_image(self) -> np.ndarray:
        x = np.linspace(-10, 10, 129)
        y = np.linspace(-10, 10, 129)
        x, y = np.meshgrid(x, y)
        return np.exp(-(x**2 + y**2))

    """Test the wavelet object"""

    def test_transform_inverse(self):
        image = self.get_image()
        starlets = starlet_transform(image, scales=3)

        # Test number of levels
        assert_equal(starlets.shape[0], 4)

        # Test inverse
        inverse = starlet_reconstruction(starlets)
        assert_almost_equal(inverse, image)
