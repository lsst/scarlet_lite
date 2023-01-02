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
from numpy.testing import assert_array_equal  # , assert_almost_equal

from scarlet_lite import Image
from scarlet_lite.operators import prox_connected, Monotonicity, prox_monotonic_mask

# from utils import assert_image_equal


class TestOperators(unittest.TestCase):
    def setUp(self) -> None:
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
        self.detect = np.sum(data["images"], axis=0)
        self.centers = np.array([data["catalog"]["y"], data["catalog"]["x"]]).T

    def test_prox_connected(self):
        image1 = Image(np.full((5, 10), 1), yx0=(5, 2))
        image2 = Image(np.full((3, 9), 2), yx0=(20, 16))
        image3 = Image(np.full((5, 12), 3), yx0=(10, 30))
        image4 = Image(np.full((7, 3), 4), yx0=(29, 5))
        image5 = Image(np.full((11, 15), 5), yx0=(30, 30))

        image = Image(np.zeros((50, 50), dtype=float))
        image += image1 + image2 + image3
        full_image = image + image4 + image5

        # Include 3 of the 5 box centers
        centers = [
            (7, 7),
            (21, 20),
            (12, 36)
        ]

        result = prox_connected(full_image.data, centers)
        assert_array_equal(result, image.data)

    def test_monotonicity(self):
        shape = (201, 201)
        cx = (shape[1] - 1) >> 1
        cy = (shape[0] - 1) >> 1
        x = np.arange(shape[1], dtype=float) - cx
        y = np.arange(shape[0], dtype=float) - cy
        x, y = np.meshgrid(x, y)
        distance = np.sqrt(x ** 2 + y ** 2)

        neighbor_dist = np.zeros((9,) + distance.shape, dtype=float)
        neighbor_dist[0, 1:, 1:] = distance[1:, 1:] - distance[:-1, :-1]
        neighbor_dist[1, 1:, :] = distance[1:, :] - distance[:-1, :]
        neighbor_dist[2, 1:, :-1] = distance[1:, :-1] - distance[:-1, 1:]
        neighbor_dist[3, :, 1:] = distance[:, 1:] - distance[:, :-1]
        # For the center pixel, set the distance to 1 just so that it is
        # non-zero
        neighbor_dist[4, cy, cx] = 1
        neighbor_dist[5, :, :-1] = distance[:, :-1] - distance[:, 1:]
        neighbor_dist[6, :-1, 1:] = distance[:-1, 1:] - distance[1:, :-1]
        neighbor_dist[7, :-1, :] = distance[:-1, :] - distance[1:, :]
        neighbor_dist[8, :-1, :-1] = distance[:-1, :-1] - distance[1:, 1:]

        monotonicity = Monotonicity(shape)
        assert_array_equal(monotonicity.distance, distance)
        assert_array_equal(monotonicity.weights > 0, neighbor_dist > 0)

        # Since the monotonicity operators _are_ the test for monotonicity,
        # we just check that the two different monotonicty operators run,
        # and that the weighted monotonicity operator is still monotonic
        # according to the monotonicity mask operator.
        morph = self.detect.copy()
        cy, cx = self.centers[1].astype(int)
        morph = monotonicity(morph, (cy, cx))
        # Add zero threshold
        morph[morph < 0] = 0
        # Get the monotonicc mask soluton
        _, masked, _ = prox_monotonic_mask(
            morph.copy(),
            (cy, cx),
            0,
            0,
            0,
        )
        # The operators are not exactly equal, since the weighted monotonicity
        # uses diagonal pixels and the monotonic mask does not take those
        # into account. So we allow for known pixels to be different.
        diff = np.abs(morph - masked)
        self.assertEqual(np.sum(diff > 0), 198)

        # Remove all of the diagonal weights and check that the
        # weighted monotonic solution agrees with the monotonic mask solution.
        weights = monotonicity.weights
        weights[0] = 0
        weights[2] = 0
        weights[6] = 0
        weights[8] = 0

        morph = self.detect.copy()
        cy, cx = self.centers[1].astype(int)
        morph = monotonicity(morph, (cy, cx))
        # Add zero threshold
        morph[morph < 0] = 0
        # Get the monotonicc mask soluton
        _, masked, _ = prox_monotonic_mask(
            morph.copy(),
            (cy, cx),
            0,
            0,
            0,
        )
        assert_array_equal(morph, masked)
