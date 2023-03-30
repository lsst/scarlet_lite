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

import numpy as np
from lsst.scarlet.lite.utils import (
    continue_class,
    get_circle_mask,
    integrated_circular_gaussian,
    integrated_gaussian_value,
)
from numpy.testing import assert_array_almost_equal, assert_array_equal
from utils import ScarletTestCase


class DummyClass:
    """A class to test the continue_class decorator"""

    def __init__(self, x):
        self.x = x


@continue_class
class DummyClass:  # noqa: F811
    """Update to the DummyClass"""

    def square(self):
        return self.x**2


class TestUtils(ScarletTestCase):
    def test_integrated_gaussians(self):
        result = integrated_circular_gaussian()
        self.assertTupleEqual(result.shape, (15, 15))

        x = np.arange(-5, 6)
        y = np.arange(-3, 4)
        result = integrated_circular_gaussian(x, y, sigma=1.2)
        x_psf = integrated_gaussian_value(x, sigma=1.2)
        y_psf = integrated_gaussian_value(y, sigma=1.2)
        truth = x_psf[None, :] * y_psf[:, None]
        truth /= np.sum(truth)
        assert_array_almost_equal(result, truth)

        with self.assertRaises(ValueError):
            integrated_circular_gaussian(x)

        with self.assertRaises(ValueError):
            integrated_circular_gaussian(y=y)

    def test_circle_mask(self):
        truth = [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]
        x = get_circle_mask(7, dtype=int)
        assert_array_equal(x, truth)

        truth = [
            [0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ]
        x = get_circle_mask(6, dtype=int)
        assert_array_equal(x, truth)

    def test_continue_class(self):
        test = DummyClass(5)
        self.assertEqual(test.square(), 25)
