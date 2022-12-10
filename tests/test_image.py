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

import operator
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from scarlet_lite import Image, Box
from scarlet_lite.image import MismatchedBoxError, MismatchedBandsError

from utils import assert_image_equal


class TestImage(unittest.TestCase):
    def test_constructors(self):
        # Default constructor
        data = np.arange(12).reshape(3, 4)  # type: ignore
        image = Image(data)
        self.assertEqual(image.dtype, int)
        self.assertEqual(image.bands, ())
        self.assertEqual(image.n_bands, 0)
        assert_array_equal(image.shape, (3, 4))
        self.assertEqual(image.height, 3)
        self.assertEqual(image.width, 4)
        assert_array_equal(image.yx0, (0, 0))
        self.assertEqual(image.y0, 0)
        self.assertEqual(image.x0, 0)
        self.assertEqual(image.indices, {})
        self.assertEqual(image.slices, {})
        self.assertEqual(image.bbox, Box((3, 4), (0, 0)))
        assert_array_equal(image.array, data)
        self.assertIsInstance(image.array, np.ndarray)
        self.assertNotIsInstance(image.array, Image)

        # Test constructor with all parameters
        data = np.arange(24, dtype=float).reshape(2, 3, 4)  # type: ignore
        bands = ("g", "i")
        y0, x0 = 10, 15
        indices = {("g", "r", "i", "z", "y"): ((0, 2), (0, 1))}
        slices = {((100, 100), (0, 0)): ((slice(10, 13), slice(15, 19)), (slice(None), slice(None)))}
        image = Image(
            data,
            bands=bands,
            yx0=(y0, x0),
            indices=indices,
            slices=slices,
        )
        self.assertEqual(image.dtype, float)
        assert_array_equal(image.bands, bands)
        self.assertEqual(image.n_bands, 2)
        assert_array_equal(image.shape, (2, 3, 4))
        self.assertEqual(image.height, 3)
        self.assertEqual(image.width, 4)
        assert_array_equal(image.yx0, (10, 15))
        self.assertEqual(image.y0, 10)
        self.assertEqual(image.x0, 15)
        self.assertEqual(len(image.indices.keys()), 1)
        self.assertEqual(indices[("g", "r", "i", "z", "y")], ((0, 2), (0, 1)))
        self.assertEqual(len(image.slices.keys()), 1)
        self.assertEqual(
            image.slices[((100, 100), (0, 0))],
            ((slice(10, 13), slice(15, 19)), (slice(None), slice(None)))
        )
        self.assertEqual(image.bbox, Box((3, 4), (10, 15)))
        assert_array_equal(image.array, data)
        self.assertIsInstance(image.array, np.ndarray)
        self.assertNotIsInstance(image.array, Image)

    def _binary_operation_test(
        self,
        lower_data: np.ndarray,
        higher_data: np.ndarray,
        lower_image: Image,
        higher_image: Image,
        op_name: str,
    ) -> None:
        lower = lower_image.copy()
        higher = higher_image.copy()
        op = getattr(operator, op_name)

        # Test operation with constants
        for constant in (3, 3.14, 3.14+3j):
            if op_name in ("floordiv", "mod", "rshift", "lshift") and constant != 3:
                # Cannot use floats or complex numbers for some operations,
                # so skip them
                continue
            print(op_name, constant)
            truth = op(lower_data, constant)
            truth_image = Image(truth, bands=lower.bands)
            result = op(lower, constant)
            assert_array_equal(result.array, truth)
            assert_image_equal(result, truth_image)

            if op_name not in ("eq", "ne", "ge", "le") and (op_name != "pow" or constant == 3.14):
                truth = op(constant, lower_data)
                truth_image = Image(truth, bands=lower.bands)
                result = getattr(lower, f"__r{op_name}__")(constant)
                assert_array_equal(result.array, truth)
                assert_image_equal(result, truth_image)

        if op_name in ["rshift", "lshift"]:
            # Shifting cannot be done with non-integer arrays
            return

        # Test lower * higher
        truth = op(lower_data, higher_data)
        truth_image = Image(truth, bands=higher_image.bands)
        result = op(lower, higher)
        assert_array_equal(result.array, truth)
        assert_image_equal(result, truth_image)

        if op_name not in ("eq", "ne", "ge", "le"):
            result = getattr(higher, f"__r{op_name}__")(lower)
            assert_array_equal(result.array, truth)
            assert_image_equal(result, truth_image)

            truth = op(higher_data, lower_data)
            truth_image = Image(truth, bands=higher_image.bands)
            iop = getattr(operator, "i" + op_name)
            iop(higher, lower)
            assert_array_equal(higher.array, truth)
            assert_image_equal(higher, truth_image)

            with self.assertRaises(ValueError):
                iop(lower_image, higher_image)

    def test_simple_arithmetic(self):
        np.random.seed(1)
        data_bool = np.random.choice((True, False), size=(2, 3, 4))
        data_int = np.random.randint(-10, 10, (2, 3, 4))
        data_int[data_int == 0] = 1
        data_float = (np.random.random((2, 3, 4)) - 0.5) * 10

        image_bool = Image(data_bool, bands=("g", "i"))
        image_int = Image(data_int, bands=("g", "i"))
        image_float = Image(data_float, bands=("g", "i"))

        self.assertEqual(data_bool.dtype, bool)
        self.assertEqual(data_int.dtype, int)
        self.assertEqual(data_float.dtype, float)

        # test casting for bool + int
        self._binary_operation_test(
            data_bool,
            data_int,
            image_bool,
            image_int,
            "add",
        )

        # Test binary operations
        binary_operations = (
            "add",
            "sub",
            "mul",
            "truediv",
            "floordiv",
            "pow",
            "mod",
            "eq",
            "ne",
            "ge",
            "le",
            "rshift",
            "lshift",
        )
        for op_name in binary_operations:
            self._binary_operation_test(
                data_int,
                data_float,
                image_int,
                image_float,
                op_name,
            )

        # Test negation
        assert_image_equal(-image_float, Image(-data_float, bands=("g", "i")))  # type: ignore
        # Test unary positive operator
        assert_image_equal(+image_float, image_float)

        # Test that matric multiplication is not supported
        with self.assertRaises(TypeError):
            image_int @ image_float

        with self.assertRaises(TypeError):
            image_int @= image_float

    def test_image_equality(self):
        # Note: equality of the arrays is tested in other tests.
        # This just checks that comparing non-images to images,
        # or images with different bounding boxes or bands raises
        # the appropriate exception.
        np.random.seed(1)
        bands = ("g", "r")
        data1 = np.random.randint(-10, 10, (2, 3, 4))
        data2 = data1.astype(float)
        data3 = np.random.randint(-10, 10, (2, 3, 4)).astype(float)

        image1 = Image(data1, bands=bands)
        image2 = Image(data2, bands=bands)
        image3 = Image(data3, bands=bands)

        for op in (operator.eq, operator.ne):
            with self.assertRaises(TypeError):
                op(image1, data1)
            with self.assertRaises(MismatchedBandsError):
                op(image1, image2.copy_with(bands=("g", "i")))
            with self.assertRaises(MismatchedBandsError):
                op(image1, image3.copy_with(bands=("g", "i")))
            with self.assertRaises(MismatchedBoxError):
                op(image1, image2.copy_with(yx0=(30, 35)))
            with self.assertRaises(MismatchedBoxError):
                op(image1, image3.copy_with(yx0=(30, 35)))

    def test_simple_boolean_arithmetic(self):
        np.random.seed(1)
        # Test boolean operations
        boolean_operations = (
            "and_",
            "or_",
            "xor",
        )
        data1 = np.random.choice((True, False), size=(2, 3, 4))
        data2 = np.random.choice((True, False), size=(2, 3, 4))
        _image1 = Image(data1, bands=("g", "i"))
        _image2 = Image(data2, bands=("g", "i"))
        for op_name in boolean_operations:
            image1 = _image1.copy()
            image2 = _image2.copy()
            op = getattr(operator, op_name)
            result = op(image1, image2)
            data_result = op(data1, data2)
            assert_image_equal(result, Image(data_result, bands=("g", "i")))

            if op_name[-1] == "_":
                # Trim the underscore after `or` and `and` in operator
                op_name = op_name[:-1]

            result = getattr(image2, f"__r{op_name}__")(image1)
            assert_image_equal(result, Image(data_result, bands=("g", "i")))

            iop = getattr(operator, "i" + op_name)
            iop(image1, image2)
            assert_image_equal(image1, Image(data_result, bands=("g", "i")))

        # Test inversion
        assert_image_equal(~_image1, Image(~data1, bands=("g", "i")))  # type: ignore

    def test_mismatchd_arithmetic(self):
        pass
