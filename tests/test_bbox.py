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
from lsst.scarlet.lite import Box
from utils import ScarletTestCase


class TestBox(ScarletTestCase):
    def check_bbox(self, bbox: Box, shape: tuple[int, ...], origin: tuple[int, ...]):
        """Check the attributes and properties of a Box

        Parameters
        ----------
        bbox:
            The box to test.
        shape:
            The shape of the Box.
        origin:
            The origin of the Box.
        """
        self.assertTupleEqual(bbox.shape, shape)
        self.assertTupleEqual(bbox.origin, origin)
        self.assertEqual(bbox.ndim, len(shape))
        self.assertTupleEqual(bbox.start, origin)

    def test_constructors(self):
        shape = (2, 3, 4)
        origin = (0, 0, 0)
        bbox = Box(shape)
        self.check_bbox(bbox, shape=shape, origin=origin)

        shape = (2, 4)
        origin = (5, 6)
        bbox = Box(shape, origin)
        self.check_bbox(bbox, shape=shape, origin=origin)

        bbox = Box.from_bounds((5, 7), (6, 10))
        self.check_bbox(bbox, shape, origin)

    def test_from_data(self):
        x = np.arange(25).reshape(5, 5)
        x[0] = 0
        x[:, -2:] = 0
        bbox = Box.from_data(x)
        self.assertBoxEqual(bbox, Box((4, 3), origin=(1, 0)))

        x += 10
        bbox = Box.from_data(x)
        self.assertBoxEqual(bbox, Box((5, 5), origin=(0, 0)))

        bbox = Box.from_data(x, threshold=10)
        self.assertBoxEqual(bbox, Box((4, 3), origin=(1, 0)))

        bbox = Box.from_data(x, threshold=100)
        self.assertBoxEqual(bbox, Box((0, 0), origin=(0, 0)))

    def test_contains(self):
        bbox = Box((6, 4, 3), origin=(0, 1, 0))
        p = (2, 2, 2)
        self.assertTrue(bbox.contains(p))

        p = (3, 0, 3)
        self.assertFalse(bbox.contains(p))

        p = (7, 3, 3)
        self.assertFalse(bbox.contains(p))

        p = (3, 3, -1)
        self.assertFalse(bbox.contains(p))

        with self.assertRaises(ValueError):
            bbox.contains((1, 2))

    def test_properties(self):
        shape = (10, 3, 8)
        origin = (2, 7, 5)
        bbox = Box(shape, origin)
        self.assertTupleEqual(bbox.stop, (12, 10, 13))
        self.assertTupleEqual(bbox.center, (7, 8.5, 9))
        self.assertTupleEqual(bbox.bounds, ((2, 12), (7, 10), (5, 13)))
        self.assertTupleEqual(bbox.shape, shape)
        self.assertTupleEqual(bbox.origin, origin)
        self.assertEqual(len(bbox.slices), 3)
        self.assertEqual(bbox.slices[0], slice(2, 12))
        self.assertEqual(bbox.slices[1], slice(7, 10))
        self.assertEqual(bbox.slices[2], slice(5, 13))
        self.assertEqual(hash(bbox), hash((shape, origin)))

    def test_simple_methods(self):
        shape = (2, 4, 8, 16)
        origin = (9, 5, 3, 9)
        bbox = Box(shape, origin)
        self.check_bbox(bbox, shape, origin)

        # Grow the box
        grown = bbox.grow(3)
        self.check_bbox(grown, (8, 10, 14, 22), (6, 2, 0, 6))

        # Shift the box
        shifted = bbox.shifted_by((0, 5, 2, 10))
        self.check_bbox(shifted, shape, (9, 10, 5, 19))

    def test_union(self):
        bbox1 = Box((3, 4), (20, 34))
        bbox2 = Box((10, 15), (1, 2))
        bbox3 = Box((20, 30, 40), (10, 20, 30))

        result = bbox1 | bbox2
        truth = Box((22, 36), (1, 2))
        self.assertBoxEqual(result, truth)

        with self.assertRaises(ValueError):
            bbox1 | bbox3

    def test_intersection(self):
        bbox1 = Box((3, 4), (20, 34))
        bbox2 = Box((20, 30), (10, 20))
        bbox3 = Box((20, 30, 40), (10, 20, 30))

        result = bbox1 & bbox2
        truth = Box((3, 4), (20, 34))
        self.assertBoxEqual(result, truth)

        with self.assertRaises(ValueError):
            bbox1 & bbox3

    def test_intersections(self):
        bbox1 = Box((3, 4), (20, 34))
        bbox2 = Box((10, 15), (1, 2))
        bbox3 = Box((20, 30), (10, 20))

        # Test intersection test
        self.assertFalse(bbox1.intersects(bbox2))
        self.assertTrue(bbox1.intersects(bbox3))
        self.assertFalse(bbox2.intersects(bbox1))
        self.assertFalse(bbox2.intersects(bbox3))
        self.assertTrue(bbox3.intersects(bbox1))

        # Test overlapping slices
        slices = bbox1.overlapped_slices(bbox2)
        self.assertTupleEqual(slices, ((slice(0, 0), slice(0, 0)), (slice(0, 0), slice(0, 0))))
        slices = bbox1.overlapped_slices(bbox3)
        self.assertTupleEqual(slices, ((slice(0, 3), slice(0, 4)), (slice(10, 13), (slice(14, 18)))))

    def test_offset(self):
        shape = (2, 5, 7)
        origin = (82, 34, 15)
        bbox = Box(shape, origin)
        bbox = bbox + 1
        self.assertBoxEqual(bbox, Box(shape, (83, 35, 16)))

    def test_arithmetic(self):
        shape = (2, 5, 7)
        origin = (82, 34, 15)
        bbox = Box(shape, origin)

        # Check addition
        shifted = bbox + (2, 4, 6)
        self.check_bbox(bbox, shape, origin)
        self.check_bbox(shifted, shape, (84, 38, 21))

        # Check subtraction
        shifted = bbox - (2, 4, 6)
        self.check_bbox(bbox, shape, origin)
        self.check_bbox(shifted, shape, (80, 30, 9))

        # Check "matrix multiplication"
        prebox = Box((2, 5), (3, 4))
        new_box = prebox @ bbox
        self.check_bbox(new_box, (2, 5, 2, 5, 7), (3, 4, 82, 34, 15))

        # Check equality
        bbox1 = Box((1, 2, 3), (2, 4, 6))
        bbox2 = Box((1, 2, 3), (2, 4, 6))
        bbox3 = Box((1, 2), (5, 6))

        self.assertBoxEqual(bbox1, bbox2)
        with self.assertRaises(AssertionError):
            self.assertBoxEqual(bbox2, bbox3)

        # Check a copy
        bbox2 = bbox.copy()
        self.assertBoxEqual(bbox, bbox2)

        self.assertFalse(bbox1 == shape)
        self.assertNotEqual(bbox1, bbox2)
        self.assertEqual(bbox1, bbox1)

    def test_slicing(self):
        bbox = Box((1, 2, 3, 4), (2, 4, 6, 8))
        # Check integer index
        self.assertBoxEqual(bbox[2], Box((3,), (6,)))
        # check slice index
        self.assertBoxEqual(bbox[:3], Box((1, 2, 3), (2, 4, 6)))
        # check tuple index
        self.assertBoxEqual(bbox[(3, 1)], Box((4, 2), (8, 4)))
