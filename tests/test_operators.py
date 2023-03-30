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
from lsst.scarlet.lite import Image
from lsst.scarlet.lite.operators import (
    Monotonicity,
    prox_connected,
    prox_monotonic_mask,
    prox_sdss_symmetry,
    prox_uncentered_symmetry,
)
from numpy.testing import assert_array_equal
from utils import ScarletTestCase


class TestOperators(ScarletTestCase):
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
        centers = [(7, 7), (21, 20), (12, 36)]

        result = prox_connected(full_image.data, centers)
        assert_array_equal(result, image.data)

    def test_monotonicity(self):
        shape = (201, 201)
        cx = (shape[1] - 1) >> 1
        cy = (shape[0] - 1) >> 1
        x = np.arange(shape[1], dtype=float) - cx
        y = np.arange(shape[0], dtype=float) - cy
        x, y = np.meshgrid(x, y)
        distance = np.sqrt(x**2 + y**2)

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
        self.assertTupleEqual(monotonicity.shape, (201, 201))
        self.assertTupleEqual(monotonicity.center, (100, 100))

        # Since the monotonicity operators _are_ the test for monotonicity,
        # we just check that the two different monotonicty operators run,
        # and that the weighted monotonicity operator is still monotonic
        # according to the monotonicity mask operator.
        morph = self.detect.copy()
        cy, cx = self.centers[1].astype(int)
        morph = monotonicity(morph, (cy, cx))
        # Add zero threshold
        morph[morph < 0] = 0
        # Get the monotonic mask soluton
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

        # Test that interpolating edge pixels is working
        _, interpolated, _ = prox_monotonic_mask(
            morph.copy(),
            (cy, cx),
            0,
            0,
            3,
        )

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
        # Get the monotonic mask soluton
        _, masked, _ = prox_monotonic_mask(
            morph.copy(),
            (cy, cx),
            0,
            0,
            0,
        )
        assert_array_equal(morph, masked)

    def test_resize_monotonicity(self):
        monotonicity = Monotonicity((101, 101))
        morph = self.detect.copy()
        cy, cx = self.centers[1].astype(int)
        morph = monotonicity(morph, (cy, cx))
        self.assertTupleEqual(monotonicity.shape, (101, 101))

        monotonicity.update((201, 201))
        morph2 = monotonicity(morph, (cy, cx))
        self.assertTupleEqual(monotonicity.shape, (201, 201))
        assert_array_equal(morph, morph2)

        with self.assertRaises(ValueError):
            # Even shapes not allowed
            Monotonicity((100, 100))

        with self.assertRaises(ValueError):
            # The shape should only have 2 dimensions
            Monotonicity((101, 101, 101))  # type: ignore

        with self.assertRaises(ValueError):
            # The shape should have exactly 2 dimensions
            Monotonicity((101,))  # type: ignore

    def test_check_size(self):
        monotonicity = Monotonicity((11, 11))
        self.assertTupleEqual(monotonicity.shape, (11, 11))
        self.assertTupleEqual(monotonicity.sizes, (5, 5, 6, 6))
        morph = self.detect.copy()
        cy, cx = self.centers[1].astype(int)
        monotonicity(morph, (cy, cx))
        self.assertTupleEqual(monotonicity.shape, (73, 73))
        self.assertTupleEqual(monotonicity.sizes, (36, 36, 37, 37))

        monotonicity = Monotonicity((11, 11), auto_update=False)
        with self.assertRaises(ValueError):
            monotonicity(morph, (cy, cx))

    def test_off_center_monotonicity(self):
        monotonicity = Monotonicity((101, 101))
        morph = self.detect.copy()
        cy, cx = self.centers[1].astype(int)
        truth = monotonicity(morph.copy(), (cy, cx))
        morph = monotonicity(morph, (cy + 1, cx))
        assert_array_equal(morph, truth)

        # Shift by 2 pixels and confirm that the morphologies are not equal
        morph = self.detect.copy()
        morph = monotonicity(morph, (cy + 2, cx))
        with self.assertRaises(AssertionError):
            assert_array_equal(morph, truth)

        # Now increase the search radius and try again
        monotonicity = Monotonicity((101, 101), fit_radius=2)
        morph = self.detect.copy()
        morph = monotonicity(morph, (cy + 2, cx))
        assert_array_equal(morph, truth)

        monotonicity = Monotonicity((101, 101), fit_radius=2)
        morph = self.detect.copy()
        morph = monotonicity(morph, (cy + 1, cx + 1))
        assert_array_equal(morph, truth)

    def test_symmetry(self):
        # Test simple symmetry
        morph = np.arange(27).reshape(3, 9)
        truth = np.array(list(range(14)) + list(range(13)[::-1])).reshape(3, 9)
        morph = prox_sdss_symmetry(morph)
        assert_array_equal(morph, truth)

        # Test uncentered symmetry
        morph = np.arange(50).reshape(5, 10)

        symmetric = [
            [25, 26, 27, 28, 29],
            [35, 36, 37, 36, 35],
            [29, 28, 27, 26, 25],
        ]

        # Test leaving the non-symmetric part of the morphology
        truth = morph.copy()
        truth[2:, 5:] = symmetric
        center = (3, 7)
        symmetric_morph = prox_uncentered_symmetry(morph.copy(), center)
        assert_array_equal(symmetric_morph, truth)

        # Test setting the non-symmetric part of the morphology to zero
        truth = np.zeros(morph.shape, dtype=int)
        truth[2:, 5:] = symmetric
        symmetric_morph = prox_uncentered_symmetry(morph.copy(), center, 0)
        assert_array_equal(symmetric_morph, truth)

        # Test skipping re-centering if the center of the source
        # is the center of the image
        _morph = morph[2:, 5:]
        symmetric_morph = prox_uncentered_symmetry(_morph.copy(), (1, 2))
        assert_array_equal(symmetric_morph, symmetric)

        # Test using the default center of the image
        _morph = morph[2:, 5:]
        symmetric_morph = prox_uncentered_symmetry(_morph.copy())
        assert_array_equal(symmetric_morph, _morph)
