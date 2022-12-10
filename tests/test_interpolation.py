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

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import scarlet_lite


class TestProjections(object):
    """Test project_image

    Because the behavior of projections is dependent on
    whether the input image and the output image have an
    even or odd number of pixels, we have tests for all
    four different cases (odd-odd, even-even, odd-even, even-odd).
    """

    def test_odd2odd(self):
        project_image = scarlet_lite.interpolation.project_image
        img = np.arange(35).reshape(5, 7)

        # samller to bigger
        shape = (11, 9)
        result = project_image(img, shape)
        truth = np.zeros(shape)
        truth[3:-3, 1:-1] = img
        assert_array_equal(result, truth)

        # bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape)
        truth = img[1:-1, 2:-2]
        assert_array_equal(result, truth)

        # lower left smaller to bigger
        shape = (11, 9)
        result = project_image(img, shape, (-6, -6))
        truth = np.zeros(shape)
        truth[:4, :5] = img[-4:, -5:]
        assert_array_equal(result, truth)

        # lower left bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape, (-4, -6))
        truth = np.zeros(shape)
        truth[:2, :2] = img[-2:, -2:]
        assert_array_equal(result, truth)

        # upper right smaller to bigger
        shape = (11, 9)
        result = project_image(img, shape, (4, 0))
        truth = np.zeros(shape)
        truth[-2:, -5:] = img[:2, :5]
        assert_array_equal(result, truth)

        # upper right bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape, (0, 1))
        truth = np.zeros(shape)
        truth[-2:, -1:] = img[:2, :1]
        assert_array_equal(result, truth)

    def test_even2even(self):
        project_image = scarlet_lite.interpolation.project_image
        img = np.arange(48).reshape(8, 6)

        # samller to bigger
        shape = (12, 8)
        result = project_image(img, shape)
        truth = np.zeros(shape)
        truth[2:-2, 1:-1] = img
        assert_array_equal(result, truth)

        # bigger to smaller
        shape = (6, 4)
        result = project_image(img, shape)
        truth = img[1:-1, 1:-1]
        assert_array_equal(result, truth)

        # lower left smaller to bigger
        shape = (14, 18)
        result = project_image(img, shape, (-10, -11))
        truth = np.zeros(shape)
        truth[:5, :4] = img[-5:, -4:]
        assert_array_equal(result, truth)

        # lower left bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape, (-1, -1))
        truth = np.zeros(shape)
        truth[-3:, -3:] = img[:3, :3]
        assert_array_equal(result, truth)

        # upper right smaller to bigger
        shape = (12, 10)
        result = project_image(img, shape, (3, 1))
        truth = np.zeros(shape)
        truth[-3:, -4:] = img[:3, :4]
        assert_array_equal(result, truth)

        # upper right bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape, (0, -1))
        truth = np.zeros(shape)
        truth[-2:, -3:] = img[:2, :3]
        assert_array_equal(result, truth)

    def test_odd2even(self):
        project_image = scarlet_lite.interpolation.project_image
        img = np.arange(35).reshape(5, 7)

        # samller to bigger
        shape = (10, 8)
        result = project_image(img, shape)
        truth = np.zeros(shape)
        truth[3:8, 1:] = img
        assert_array_equal(result, truth)

        # bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape)
        truth = img[:4, 1:-2]
        assert_array_equal(result, truth)

        # lower left smaller to bigger
        shape = (14, 18)
        result = project_image(img, shape, (-9, -11))
        truth = np.zeros(shape)
        truth[:3, :5] = img[-3:, -5:]
        assert_array_equal(result, truth)

        # lower left bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape, (-4, -5))
        truth = np.zeros(shape)
        truth[:3, :4] = img[-3:, -4:]
        assert_array_equal(result, truth)

        # upper right smaller to bigger
        shape = (12, 10)
        result = project_image(img, shape, (3, 1))
        truth = np.zeros(shape)
        truth[-3:, -4:] = img[:3, :4]

        # upper right bigger to smaller
        shape = (4, 4)
        result = project_image(img, shape, (1, 0))
        truth = np.zeros(shape)
        truth[-1:, -2:] = img[:1, :2]
        assert_array_equal(result, truth)

    def test_even2odd(self):
        project_image = scarlet_lite.interpolation.project_image
        img = np.arange(48).reshape(8, 6)

        # samller to bigger
        shape = (11, 9)
        result = project_image(img, shape)
        truth = np.zeros(shape)
        truth[1:-2, 1:-2] = img
        assert_array_equal(result, truth)

        # bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape)
        truth = img[3:-2, 2:-1]
        assert_array_equal(result, truth)

        # lower left smaller to bigger
        shape = (11, 9)
        result = project_image(img, shape, (-9, -5))
        truth = np.zeros(shape)
        truth[:4, :5] = img[-4:, -5:]
        assert_array_equal(result, truth)

        # lower left bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape, (-7, -5))
        truth = np.zeros(shape)
        truth[:2, :2] = img[-2:, -2:]
        assert_array_equal(result, truth)

        # upper right smaller to bigger
        shape = (11, 9)
        result = project_image(img, shape, (4, 0))
        truth = np.zeros(shape)
        truth[-2:, -5:] = img[:2, :5]
        assert_array_equal(result, truth)

        # upper right bigger to smaller
        shape = (3, 3)
        result = project_image(img, shape, (0, 1))
        truth = np.zeros(shape)
        truth[-2:, -1:] = img[:2, :1]
        assert_array_equal(result, truth)

    def test_zoom(self):
        # Test that zomming out and in keeps a consistent center
        kernel = np.arange(4).reshape(2, 2) + 1
        p3 = scarlet_lite.interpolation.project_image(kernel, (3, 3))
        p6 = scarlet_lite.interpolation.project_image(p3, (6, 6))
        p5 = scarlet_lite.interpolation.project_image(p6, (5, 5))
        p2 = scarlet_lite.interpolation.project_image(p3, (2, 2))
        assert_array_equal(p2, kernel)
        truth = [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]
        assert_array_equal(p3, truth)
        truth = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        assert_array_equal(p6, truth)
        truth = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0, 0.0],
            [0.0, 3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        assert_array_equal(p5, truth)


def interpolate_comparison(func, zero_truth, positive_truth, **kwargs):
    # zero shift
    result = func(0, **kwargs)
    truth = zero_truth
    assert_almost_equal(result[0], truth[0])
    assert_array_equal(result[1], truth[1])

    # positive shift
    result = func(0.103, **kwargs)
    truth = positive_truth
    assert_almost_equal(result[0], truth[0])
    assert_array_equal(result[1], truth[1])

    # negative shift
    result = func(-0.103, **kwargs)
    truth = (truth[0][::-1], -truth[1][::-1])
    assert_almost_equal(result[0], truth[0])
    assert_array_equal(result[1], truth[1])

    with pytest.raises(ValueError):
        scarlet_lite.interpolation.lanczos(1.1)
    with pytest.raises(ValueError):
        scarlet_lite.interpolation.lanczos(-1.1)


class TestConvolutions:
    """Test FFT convolutions and interpolation algorithms"""

    def test_bilinear(self):
        zero_truth = (np.array([1, 0]), np.array([0, 1]))
        positive_truth = (np.array([1 - 0.103, 0.103]), np.array([0, 1]))
        interpolate_comparison(
            scarlet_lite.interpolation.bilinear, zero_truth, positive_truth
        )

    def test_cubic_spline(self):
        zero_truth = (np.array([0.0, 1.0, 0.0, 0.0]), np.array([-1, 0, 1, 2]))
        positive_truth = (
            np.array([-0.08287473, 0.97987473, 0.11251627, -0.00951627]),
            np.array([-1, 0, 1, 2]),
        )
        interpolate_comparison(
            scarlet_lite.interpolation.cubic_spline, zero_truth, positive_truth
        )

    def test_catmull_rom(self):
        # Catmull Rom should be the same as the cubic spline
        # with a=0.5 and b=0
        zero_truth = scarlet_lite.interpolation.cubic_spline(0, a=0.5)
        positive_truth = scarlet_lite.interpolation.cubic_spline(0.103, a=0.5)
        interpolate_comparison(
            scarlet_lite.interpolation.catmull_rom, zero_truth, positive_truth
        )

    def test_mitchel_netravali(self):
        # Mitchel Netravali should be the same as the cubic spline
        # with a=1/3 and b=1/3
        zero_truth = scarlet_lite.interpolation.cubic_spline(0, a=1 / 3, b=1 / 3)
        positive_truth = scarlet_lite.interpolation.cubic_spline(
            0.103, a=1 / 3, b=1 / 3
        )
        interpolate_comparison(
            scarlet_lite.interpolation.mitchel_netravali, zero_truth, positive_truth
        )

    def test_lanczos(self):
        # test Lanczos 3
        zero_truth = (np.array([0, 0, 1, 0, 0, 0]), np.arange(6) - 2)
        positive_truth = (
            np.array(
                [
                    0.01763955,
                    -0.07267534,
                    0.98073579,
                    0.09695747,
                    -0.0245699,
                    0.00123974,
                ]
            ),
            np.array([-2, -1, 0, 1, 2, 3]),
        )
        interpolate_comparison(
            scarlet_lite.interpolation.lanczos, zero_truth, positive_truth
        )

        # test Lanczos 5
        _truth = np.zeros((10,))
        _truth[4] = 1
        zero_truth = (_truth, np.arange(10) - 4)
        positive_truth = (
            np.array(
                [
                    5.11187895e-03,
                    -1.55432491e-02,
                    3.52955166e-02,
                    -8.45895745e-02,
                    9.81954247e-01,
                    1.06954413e-01,
                    -4.15882547e-02,
                    1.85994926e-02,
                    -6.77652513e-03,
                    4.34415682e-04,
                ]
            ),
            np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        )
        interpolate_comparison(
            scarlet_lite.interpolation.lanczos, zero_truth, positive_truth, a=5
        )
