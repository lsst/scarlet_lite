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
from typing import Sequence

from scarlet_lite import Box


def check_bbox(bbox: Box, shape: Sequence[int], origin: Sequence[int]):
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
    assert bbox.shape == shape
    assert bbox.origin == origin
    assert bbox.dimensions == len(shape)
    assert bbox.start == origin


class TestBox(object):
    def test_constuctors(self):
        shape = (2, 3, 4)
        origin = (0, 0, 0)
        bbox = Box(shape)
        check_bbox(bbox, shape=shape, origin=origin)

        shape = (2, 4)
        origin = (5, 6)
        bbox = Box(shape, origin)
        check_bbox(bbox, shape=shape, origin=origin)

        bbox = Box.from_bounds((5, 7), (6, 10))
        check_bbox(bbox, shape, origin)

    def test_from_data(self):
        x = np.arange(25).reshape(5, 5)
        x[0] = 0
        x[:, -2:] = 0
        bbox = Box.from_data(x)
        assert bbox == Box((4, 3), origin=(1, 0))

        x += 10
        bbox = Box.from_data(x)
        assert bbox == Box((5, 5), origin=(0, 0))

        bbox = Box.from_data(x, min_value=10)
        assert bbox == Box((4, 3), origin=(1, 0))

    def test_contains(self):
        bbox = Box((6, 4, 3), origin=(0, 1, 0))
        p = (2, 2, 2)
        assert bbox.contains(p)

        p = (3, 0, 3)
        assert not bbox.contains(p)

        p = (7, 3, 3)
        assert not bbox.contains(p)

        p = (3, 3, -1)
        assert not bbox.contains(p)

    def test_extract_from(self):
        image = np.zeros((3, 5, 5))
        image[1, 1, 1] = 1

        # simple one pixel box extraction
        bbox = Box.from_data(image)
        extracted = bbox.extract_from(image)
        assert extracted.shape == (1, 1, 1) and extracted[0, 0, 0] == 1

        # offset box extraction past boundary of image
        bbox = Box.from_bounds((0, 3), (-2, 3), (-3, 2))
        extracted = bbox.extract_from(image)
        assert extracted.shape == (3, 5, 5) and extracted[1, 3, 4] == 1

    def test_insert_into(self):
        image = np.zeros((3, 5, 5))
        sub = np.zeros((3, 5, 5))
        sub[1, 3, 4] = 1
        bbox = Box.from_bounds((0, 3), (-2, 3), (-3, 2))
        image = bbox.insert_into(image, sub)
        assert image.shape == (3, 5, 5) and image[1, 1, 1] == 1

    def test_properties(self):
        shape = (10, 3, 8)
        origin = (2, 7, 5)
        bbox = Box(shape, origin)
        assert bbox.stop == (12, 10, 13)
        assert bbox.center == (7, 8.5, 9)
        assert bbox.bounds == ((2, 12), (7, 10), (5, 13))
        assert len(bbox.slices) == 3
        assert bbox.slices[0] == slice(2, 12)
        assert bbox.slices[1] == slice(7, 10)
        assert bbox.slices[2] == slice(5, 13)

    def test_simple_methods(self):
        shape = (2, 4, 8, 16)
        origin = (9, 5, 3, 9)
        bbox = Box(shape, origin)
        check_bbox(bbox, shape, origin)

        # Grow the box
        grown = bbox.grow(3)
        check_bbox(grown, (8, 10, 14, 22), (6, 2, 0, 6))

        # Shift the box
        shifted = bbox.shifted_by((0, 5, 2, 10))
        check_bbox(shifted, shape, (9, 10, 5, 19))

    def test_intersections(self):
        bbox1 = Box((3, 4), (20, 34))
        bbox2 = Box((10, 15), (1, 2))
        bbox3 = Box((20, 30), (10, 20))

        # Test intersection test
        assert not bbox1.intersects(bbox2)
        assert bbox1.intersects(bbox3)
        assert not bbox2.intersects(bbox1)
        assert not bbox2.intersects(bbox3)
        assert bbox3.intersects(bbox1)

        # Test overlapping slices
        slices = bbox1.overlapped_slices(bbox2)
        assert slices == ((slice(0, 0), slice(0, 0)), (slice(0, 0), slice(0, 0)))
        slices = bbox1.overlapped_slices(bbox3)
        assert slices == ((slice(0, 3), slice(0, 4)), (slice(10, 13), (slice(14, 18))))

    def test_arithmetic(self):
        shape = (2, 5, 7)
        origin = (82, 34, 15)
        bbox = Box(shape, origin)

        # Check addition
        shifted = bbox + (2, 4, 6)
        check_bbox(bbox, shape, origin)
        check_bbox(shifted, shape, (84, 38, 21))

        # Check subtraction
        shifted = bbox - (2, 4, 6)
        check_bbox(bbox, shape, origin)
        check_bbox(shifted, shape, (80, 30, 9))

        # Check "matrix multiplication"
        prebox = Box((2, 5), (3, 4))
        new_box = prebox @ bbox
        check_bbox(new_box, (2, 5, 2, 5, 7), (3, 4, 82, 34, 15))

        # Check equality
        bbox1 = Box((1, 2, 3), (2, 4, 6))
        bbox2 = Box((1, 2, 3), (2, 4, 6))
        bbox3 = Box((1, 2), (5, 6))

        assert bbox1 == bbox2
        assert bbox2 != bbox3

        # Check a copy
        bbox2 = bbox.copy()
        assert bbox == bbox2

    def test_slicing(self):
        bbox = Box((1, 2, 3, 4), (2, 4, 6, 8))
        # Check integer index
        assert bbox[2] == Box((3,), (6,))
        # check slice index
        assert bbox[:3] == Box((1, 2, 3), (2, 4, 6))
        # check tuple index
        assert bbox[(3, 1)] == Box((4, 2), (8, 4))
