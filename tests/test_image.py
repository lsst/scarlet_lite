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
from numpy.testing import assert_array_equal, assert_almost_equal

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
        assert_array_equal(image.data, data)
        self.assertIsInstance(image.data, np.ndarray)
        self.assertNotIsInstance(image.data, Image)

        # Test constructor with all parameters
        data = np.arange(24, dtype=float).reshape(2, 3, 4)  # type: ignore
        bands = ("g", "i")
        y0, x0 = 10, 15
        indices = {("g", "r", "i", "z", "y"): ((0, 2), (0, 1))}
        slices = {
            ((100, 100), (0, 0)): (
                (slice(10, 13), slice(15, 19)),
                (slice(None), slice(None)),
            )
        }
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
            ((slice(10, 13), slice(15, 19)), (slice(None), slice(None))),
        )
        self.assertEqual(image.bbox, Box((3, 4), (10, 15)))
        assert_array_equal(image.data, data)
        self.assertIsInstance(image.data, np.ndarray)
        self.assertNotIsInstance(image.data, Image)

        # test initializing an empty image from a bounding box
        image = Image.from_box(Box((10, 10), (13, 50)))
        assert_image_equal(image, Image(np.zeros((10, 10), dtype=float), bands=(), yx0=(13, 50)))
        bands = ("g", "r", "i")
        image = Image.from_box(Box((10, 10), (13, 50)), bands=bands)
        assert_image_equal(image, Image(np.zeros((3, 10, 10), dtype=float), bands=bands, yx0=(13, 50)))

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
        for constant in (3, 3.14, 3.14 + 3j):
            if op_name in ("floordiv", "mod", "rshift", "lshift") and constant != 3:
                # Cannot use floats or complex numbers for some operations,
                # so skip them
                continue
            truth = op(lower_data, constant)
            truth_image = Image(truth, bands=lower.bands)
            result = op(lower, constant)
            assert_array_equal(result.data, truth)
            assert_image_equal(result, truth_image)

            if op_name not in ("eq", "ne", "ge", "le") and (
                op_name != "pow" or constant == 3.14
            ):
                truth = op(constant, lower_data)
                truth_image = Image(truth, bands=lower.bands)
                result = getattr(lower, f"__r{op_name}__")(constant)
                assert_array_equal(result.data, truth)
                assert_image_equal(result, truth_image)

        if op_name in ["rshift", "lshift"]:
            # Shifting cannot be done with non-integer arrays
            return

        # Test lower * higher
        truth = op(lower_data, higher_data)
        truth_image = Image(truth, bands=higher_image.bands)
        result = op(lower, higher)
        assert_array_equal(result.data, truth)
        assert_image_equal(result, truth_image)

        if op_name not in ("eq", "ne", "ge", "le"):
            result = getattr(higher, f"__r{op_name}__")(lower)
            assert_array_equal(result.data, truth)
            assert_image_equal(result, truth_image)

            truth = op(higher_data, lower_data)
            truth_image = Image(truth, bands=higher_image.bands)
            iop = getattr(operator, "i" + op_name)
            iop(higher, lower)
            assert_array_equal(higher.data, truth)
            assert_image_equal(higher, truth_image)

            with self.assertRaises(ValueError):
                iop(lower_image, higher_image)

    def check_simple_arithmetic(self, data_bool, data_int, data_float, bands):
        image_bool = Image(data_bool, bands=bands)
        image_int = Image(data_int, bands=bands)
        image_float = Image(data_float, bands=bands)

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
            if op_name == "pow":
                _data_int = np.abs(data_int)
                _data_int[_data_int == 0] = 1
                _image_int = image_int.copy()
                _image_int.data[:] = _data_int
                _data_float = np.abs(data_float)
                _data_float[_data_float == 0] = 1
                _image_float = image_float.copy()
                _image_float.data[:] = _data_float
            else:
                _data_float = data_float
                _image_float = image_float
                _data_int = data_int
                _image_int = image_int
            self._binary_operation_test(
                _data_int,
                _data_float,
                _image_int,
                _image_float,
                op_name,
            )

        # Test negation
        assert_image_equal(-image_float, Image(-data_float, bands=bands))  # type: ignore
        # Test unary positive operator
        assert_image_equal(+image_float, image_float)

        # Test that matric multiplication is not supported
        with self.assertRaises(TypeError):
            image_int @ image_float

        with self.assertRaises(TypeError):
            image_int @= image_float

    def test_simple_3d_arithmetic(self):
        np.random.seed(1)
        data_bool = np.random.choice((True, False), size=(2, 3, 4))
        data_int = np.random.randint(-10, 10, (2, 3, 4))
        data_int[data_int == 0] = 1
        data_float = (np.random.random((2, 3, 4)) - 0.5) * 10
        self.check_simple_arithmetic(data_bool, data_int, data_float, bands=("g", "r"))

    def test_simple_2d_arithmetic(self):
        np.random.seed(1)
        data_bool = np.random.choice((True, False), size=(3, 4))
        data_int = np.random.randint(-10, 10, (3, 4))
        data_int[data_int == 0] = 1
        data_float = (np.random.random((3, 4)) - 0.5) * 10
        self.check_simple_arithmetic(data_bool, data_int, data_float, bands=None)

    def test_3d_image_equality(self):
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

    def test_2d_image_equality(self):
        # Note: equality of the arrays is tested in other tests.
        # This just checks that comparing non-images to images,
        # or images with different bounding boxes or bands raises
        # the appropriate exception.
        np.random.seed(1)
        data1 = np.random.randint(-10, 10, (3, 4))
        data2 = data1.astype(float)
        data3 = np.random.randint(-10, 10, (3, 4)).astype(float)

        image1 = Image(data1)
        image2 = Image(data2)
        image3 = Image(data3)

        for op in (operator.eq, operator.ne):
            with self.assertRaises(TypeError):
                op(image1, data1)
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
        assert_image_equal(~_image1, Image(~data1, bands=("g", "i")))

    def _3d_mismatched_images_test(
        self,
        op_name: str,
    ):
        np.random.seed(1)
        op = getattr(operator, op_name)
        grizy = ("g", "r", "i", "z", "y")
        gir = ("g", "i", "r")
        igy = ("i", "g", "y")

        # Test band insert
        if op_name == "pow":
            data1 = np.random.random((5, 3, 4)) + 1
            data2 = np.random.random((3, 3, 4)) + 1
        else:
            data1 = (np.random.random((5, 3, 4)) - 0.5) * 10
            data2 = (np.random.random((3, 3, 4)) - 0.5) * 10
        image1 = Image(data1, bands=grizy)
        image2 = Image(data2, bands=gir)
        result = op(image1, image2)
        truth = np.zeros((5, 3, 4), dtype=float)
        truth += data1
        truth[(0, 2, 1), :, :] = op(truth[(0, 2, 1), :, :], data2)
        assert_almost_equal(result.data, truth)
        assert_image_equal(result, Image(truth, bands=grizy))

        # Test band mixture
        if op_name == "pow":
            data1 = np.random.random((3, 3, 4)) + 1
            data2 = np.random.random((3, 3, 4)) + 1
        else:
            data1 = (np.random.random((3, 3, 4)) - 0.5) * 10
            data2 = (np.random.random((3, 3, 4)) - 0.5) * 10
        image1 = Image(data1, bands=gir)
        image2 = Image(data2, bands=igy)
        result = op(image1, image2)
        truth = np.zeros((4, 3, 4), dtype=float)
        truth[(0, 1, 2), :, :] = data1
        truth[(1, 0, 3), :, :] = op(truth[(1, 0, 3), :, :], data2)
        assert_almost_equal(result.data, truth)
        assert_image_equal(result, Image(truth, bands=("g", "i", "r", "y")))

        # Test spatial offsets
        if op_name == "pow":
            data1 = np.random.random((3, 3, 4)) + 1
            data2 = np.random.random((3, 3, 4)) + 1
        else:
            data1 = (np.random.random((3, 3, 4)) - 0.5) * 10
            data2 = (np.random.random((3, 3, 4)) - 0.5) * 10
        image1 = Image(data1, bands=gir, yx0=(10, 20))
        image2 = Image(data2, bands=gir, yx0=(11, 17))
        result = op(image1, image2)
        truth = np.zeros((3, 4, 7), dtype=float)
        truth[:, :3, 3:] = data1
        truth[:, 1:, :4] = op(truth[:, 1:, :4], data2)
        assert_almost_equal(result.data, truth)
        assert_image_equal(result, Image(truth, bands=gir, yx0=(10, 17)))

    def _2d_mismatched_images_test(
        self,
        op_name: str,
    ):
        np.random.seed(1)
        op = getattr(operator, op_name)

        # Test spatial offsets
        if op_name == "pow":
            data1 = np.random.random((3, 4)) + 1
            data2 = np.random.random((3, 4)) + 1
        else:
            data1 = (np.random.random((3, 4)) - 0.5) * 10
            data2 = (np.random.random((3, 4)) - 0.5) * 10
        image1 = Image(data1, yx0=(10, 20))
        image2 = Image(data2, yx0=(11, 17))
        result = op(image1, image2)
        truth = np.zeros((4, 7), dtype=float)
        truth[:3, 3:] = data1
        truth[1:, :4] = op(truth[1:, :4], data2)
        assert_almost_equal(result.data, truth)
        assert_image_equal(result, Image(truth, yx0=(10, 17)))

    def test_mismatchd_arithmetic(self):
        binary_operations = (
            "add",
            "sub",
            "mul",
            "truediv",
            "floordiv",
            "pow",
            "mod",
        )

        for op_name in binary_operations:
            self._3d_mismatched_images_test(op_name)
            self._2d_mismatched_images_test(op_name)

    def test_slicing(self):
        bands = ("g", "r", "i", "z", "y")
        yx0 = (27, 82)
        data = (np.random.random((5, 30, 40)) - 0.5) * 10
        image = Image(data, bands=bands, yx0=yx0)

        # test band slicing
        sub_img = image["g"]
        assert_image_equal(sub_img, Image(data[0], yx0=yx0))

        sub_img = image["g":"i"]
        assert_image_equal(sub_img, Image(data[0:2], bands=("g", "r"), yx0=yx0))

        sub_img = image["r":"y"]
        assert_image_equal(sub_img, Image(data[1:4], bands=("r", "i", "z"), yx0=yx0))

        sub_img = image["z":]
        assert_image_equal(sub_img, Image(data[-2:], bands=("z", "y"), yx0=yx0))

        sub_img = image[("z", "i", "y")]
        assert_image_equal(sub_img, Image(data[(3, 2, 4), :, :], bands=("z", "i", "y"), yx0=yx0))

        assert_image_equal(image[:], image)

        # test spatial slicing
        sub_img = image[:, :10, :10]
        assert_image_equal(sub_img, Image(data[:, :10, :10], bands=bands, yx0=yx0))

        sub_img = image[:, 10:20, 5:10]
        assert_image_equal(sub_img, Image(data[:, 10:20, 5:10], bands=bands, yx0=(37, 87)))

        # Test bounding box slicing
        sub_img = image[:, Box((10, 5), (37, 87))]
        assert_image_equal(sub_img, Image(data[:, 10:20, 5:10], bands=bands, yx0=(37, 87)))

        with self.assertRaises(IndexError):
            # Cannot index a single row, since it would not return an image
            _ = image["g", 0]

        with self.assertRaises(IndexError):
            # Cannot index a single column, since it would not return an image
            _ = image[:, :, 0]

        with self.assertRaises(IndexError):
            # Cannot use a tuple to select rows/columns
            _ = image[("r", "i"), (1, 2)]

        with self.assertRaises(IndexError):
            # Cannot use a bounding box outside of the image
            _ = image[:, Box((10, 10), (0, 0))]

        with self.assertRaises(IndexError):
            # Cannot use a bounding box partially outside of the image
            _ = image[:, Box((40, 40), (20, 80))]

    def test_overlap_detection(self):
        # Test 2D image
        image = Image(np.zeros((5, 6)), yx0=(10, 15))
        slices = image.overlapped_slices(Box((8, 9), (7, 18)))
        truth = (
            (slice(0, 5), slice(3, 6)),
            (slice(3, 8), slice(0, 3))
        )
        self.assertTupleEqual(slices, truth)

        # Test 3D image
        image = Image(np.zeros((3, 10, 12)), bands=("g", "r", "i"), yx0=(13, 21))
        slices = image.overlapped_slices(Box((8, 9), (15, 18)))
        truth = (
            (slice(None), slice(2, 10), slice(0, 6)),
            (slice(None), slice(0, 8), slice(3, 9))
        )
        self.assertTupleEqual(slices, truth)

        # Test no overlap
        slices = image.overlapped_slices(Box((8, 9), (115, 118)))
        truth = (
            (slice(None), slice(0, 0), slice(0, 0)),
            (slice(None), slice(0, 0), slice(0, 0)),
        )
        self.assertTupleEqual(slices, truth)
