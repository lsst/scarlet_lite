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

import operator

import numpy as np
from lsst.scarlet.lite import Box, Image
from lsst.scarlet.lite.image import MismatchedBandsError, MismatchedBoxError
from numpy.testing import assert_almost_equal, assert_array_equal
from utils import ScarletTestCase


class TestImage(ScarletTestCase):
    def test_constructors(self):
        # Default constructor
        data = np.arange(12).reshape(3, 4)  # type: ignore
        image = Image(data)
        self.assertEqual(image.dtype, int)
        self.assertTupleEqual(image.bands, ())
        self.assertEqual(image.n_bands, 0)
        assert_array_equal(image.shape, (3, 4))
        self.assertEqual(image.height, 3)
        self.assertEqual(image.width, 4)
        assert_array_equal(image.yx0, (0, 0))
        self.assertEqual(image.y0, 0)
        self.assertEqual(image.x0, 0)
        self.assertBoxEqual(image.bbox, Box((3, 4), (0, 0)))
        assert_array_equal(image.data, data)
        self.assertIsInstance(image.data, np.ndarray)
        self.assertNotIsInstance(image.data, Image)

        # Test constructor with all parameters
        data = np.arange(24, dtype=float).reshape(2, 3, 4)  # type: ignore
        bands = ("g", "i")
        y0, x0 = 10, 15
        image = Image(
            data,
            bands=bands,
            yx0=(y0, x0),
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
        self.assertBoxEqual(image.bbox, Box((3, 4), (10, 15)))
        assert_array_equal(image.data, data)
        self.assertIsInstance(image.data, np.ndarray)
        self.assertNotIsInstance(image.data, Image)

        # test initializing an empty image from a bounding box
        image = Image.from_box(Box((10, 10), (13, 50)))
        self.assertImageEqual(image, Image(np.zeros((10, 10), dtype=float), bands=(), yx0=(13, 50)))
        bands = ("g", "r", "i")
        image = Image.from_box(Box((10, 10), (13, 50)), bands=bands)
        self.assertImageEqual(image, Image(np.zeros((3, 10, 10), dtype=float), bands=bands, yx0=(13, 50)))

        with self.assertRaises(ValueError):
            Image(np.zeros((3, 4, 5)), bands=tuple("gr"))

        truth = "Image:\n [[[0 1 2]\n  [3 4 5]]]\n  bands=('g',)\n  bbox=Box(shape=(2, 3), origin=(3, 2))"
        data = np.arange(6).reshape(1, 2, 3)
        bands = tuple("g")
        yx0 = (3, 2)
        image = Image(data, bands=bands, yx0=yx0)
        self.assertEqual(str(image), truth)

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
            self.assertImageEqual(result, truth_image)

            if op_name not in ("eq", "ne", "ge", "le", "lt", "gt") and (op_name != "pow" or constant == 3.14):
                truth = op(constant, lower_data)
                truth_image = Image(truth, bands=lower.bands)
                result = getattr(lower, f"__r{op_name}__")(constant)
                assert_array_equal(result.data, truth)
                self.assertImageEqual(result, truth_image)

        if op_name in ["rshift", "lshift"]:
            # Shifting cannot be done with non-integer arrays
            return

        # Test lower * higher
        truth = op(lower_data, higher_data)
        truth_image = Image(truth, bands=higher_image.bands)
        result = op(lower, higher)
        assert_array_equal(result.data, truth)
        self.assertImageEqual(result, truth_image)

        if op_name not in ("eq", "ne", "ge", "le", "gt", "lt"):
            result = getattr(higher, f"__r{op_name}__")(lower)
            assert_array_equal(result.data, truth)
            self.assertImageEqual(result, truth_image)

            truth = op(higher_data, lower_data)
            truth_image = Image(truth, bands=higher_image.bands)
            iop = getattr(operator, "i" + op_name)
            iop(higher, lower)
            assert_array_equal(higher.data, truth)
            self.assertImageEqual(higher, truth_image)

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
            "gt",
            "lt",
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
        self.assertImageEqual(-image_float, Image(-data_float, bands=bands))  # type: ignore
        # Test unary positive operator
        self.assertImageEqual(+image_float, image_float)

        # Test that matrix multiplication is not supported
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
            self.assertImageEqual(result, Image(data_result, bands=("g", "i")))

            if op_name[-1] == "_":
                # Trim the underscore after `or` and `and` in operator
                op_name = op_name[:-1]

            result = getattr(image2, f"__r{op_name}__")(image1)
            self.assertImageEqual(result, Image(data_result, bands=("g", "i")))

            iop = getattr(operator, "i" + op_name)
            iop(image1, image2)
            self.assertImageEqual(image1, Image(data_result, bands=("g", "i")))

        # Test inversion
        self.assertImageEqual(~_image1, Image(~data1, bands=("g", "i")))

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
        if op_name == "add" or op_name == "subtract":
            data1 = (np.random.random((5, 3, 4)) - 0.5) * 10
            data2 = (np.random.random((3, 3, 4)) - 0.5) * 10
            image1 = Image(data1, bands=grizy)
            image2 = Image(data2, bands=gir)
            result = op(image1, image2)
            truth = np.zeros((5, 3, 4), dtype=float)
            truth += data1
            truth[(0, 2, 1), :, :] = op(truth[(0, 2, 1), :, :], data2)
            assert_almost_equal(result.data, truth)
            self.assertImageEqual(result, Image(truth, bands=grizy))

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
            self.assertImageEqual(result, Image(truth, bands=("g", "i", "r", "y")))

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

        _data1 = np.zeros((3, 4, 7), dtype=float)
        _data2 = np.zeros((3, 4, 7), dtype=float)
        _data1[:, :3, 3:] = data1
        _data2[:, 1:, :4] = data2
        with np.errstate(divide="ignore", invalid="ignore"):
            truth = op(_data1, _data2)
        assert_almost_equal(result.data, truth)
        self.assertImageEqual(result, Image(truth, bands=gir, yx0=(10, 17)))

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

        _data1 = np.zeros((4, 7), dtype=float)
        _data2 = np.zeros((4, 7), dtype=float)
        _data1[:3, 3:] = data1
        _data2[1:, :4] = data2
        with np.errstate(divide="ignore", invalid="ignore"):
            truth = op(_data1, _data2)
        assert_almost_equal(result.data, truth)
        self.assertImageEqual(result, Image(truth, yx0=(10, 17)))

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

    def test_scalar_arithmetic(self):
        data = np.arange(6).reshape(1, 2, 3)
        bands = tuple("g")
        yx0 = (3, 2)
        image = Image(data, bands=bands, yx0=yx0)
        self.assertImageEqual(2 & image, Image(2 & data, bands=bands, yx0=yx0))
        self.assertImageEqual(2 | image, Image(2 | data, bands=bands, yx0=yx0))
        self.assertImageEqual(2 ^ image, Image(2 ^ data, bands=bands, yx0=yx0))

        with self.assertRaises(TypeError):
            image << 2.0
        with self.assertRaises(TypeError):
            image >> 2.0

        image2 = image.copy()
        image2 <<= 2
        self.assertImageEqual(image2, Image(data << 2, bands=bands, yx0=yx0))

        image2 = image.copy()
        image2 >>= 2
        self.assertImageEqual(image2, Image(data >> 2, bands=bands, yx0=yx0))

    def test_slicing(self):
        bands = ("g", "r", "i", "z", "y")
        yx0 = (27, 82)
        data = (np.random.random((5, 30, 40)) - 0.5) * 10
        image = Image(data, bands=bands, yx0=yx0)
        image_2d = Image(data[0], yx0=yx0)

        # test band slicing
        sub_img = image["g"]
        self.assertImageEqual(sub_img, Image(data[0], yx0=yx0))

        sub_img = image[:"g"]
        self.assertImageEqual(sub_img, Image(data[:1], bands=("g",), yx0=yx0))

        sub_img = image["g":"r"]
        self.assertImageEqual(sub_img, Image(data[0:2], bands=("g", "r"), yx0=yx0))

        sub_img = image["r":"z"]
        self.assertImageEqual(sub_img, Image(data[1:4], bands=("r", "i", "z"), yx0=yx0))

        sub_img = image["z":]
        self.assertImageEqual(sub_img, Image(data[-2:], bands=("z", "y"), yx0=yx0))

        sub_img = image[("z", "i", "y")]
        self.assertImageEqual(sub_img, Image(data[(3, 2, 4), :, :], bands=("z", "i", "y"), yx0=yx0))

        self.assertImageEqual(image[:], image)

        # Test bounding box slicing
        sub_img = image[:, Box((10, 5), (37, 87))]
        self.assertImageEqual(sub_img, Image(data[:, 10:20, 5:10], bands=bands, yx0=(37, 87)))

        sub_img = image[Box((10, 5), (37, 87))]
        self.assertImageEqual(sub_img, Image(data[:, 10:20, 5:10], bands=bands, yx0=(37, 87)))

        sub_img = image_2d[Box((10, 5), (37, 87))]
        self.assertImageEqual(sub_img, Image(data[0, 10:20, 5:10], yx0=(37, 87)))

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

        with self.assertRaises(IndexError):
            # Too many spatial indices
            _ = image[:, :, :, :]

        truth = (
            (0, 1, 2, 3, 4),
            slice(27, 57),
            slice(82, 122),
        )
        self.assertTupleEqual(image.multiband_slices, truth)

    def test_overlap_detection(self):
        # Test 2D image
        image = Image(np.zeros((5, 6)), yx0=(10, 15))
        slices = image.overlapped_slices(Box((8, 9), (7, 18)))
        truth = ((slice(0, 5), slice(3, 6)), (slice(3, 8), slice(0, 3)))
        self.assertTupleEqual(slices, truth)

        # Test 3D image
        image = Image(np.zeros((3, 10, 12)), bands=("g", "r", "i"), yx0=(13, 21))
        slices = image.overlapped_slices(Box((8, 9), (15, 18)))
        truth = (
            (slice(None), slice(2, 10), slice(0, 6)),
            (slice(None), slice(0, 8), slice(3, 9)),
        )
        self.assertTupleEqual(slices, truth)

        # Test no overlap
        slices = image.overlapped_slices(Box((8, 9), (115, 118)))
        truth = (
            (slice(None), slice(0, 0), slice(0, 0)),
            (slice(None), slice(0, 0), slice(0, 0)),
        )
        self.assertTupleEqual(slices, truth)

    def test_insertion(self):
        img1 = Image.from_box(Box((20, 20)), bands=tuple("gri"))
        img2 = Image.from_box(Box((5, 5), (11, 12)), bands=tuple("gi"))
        img2.data[:] = np.arange(1, 3)[:, None, None]
        img1.insert(img2)

        truth = img1.copy()
        truth.data[0, 11:16, 12:17] = 1
        truth.data[2, 11:16, 12:17] = 2
        self.assertImageEqual(img1, truth)

    def test_matched_spectral_indices(self):
        img1 = Image.from_box(Box((5, 5)))
        img2 = Image.from_box(Box((5, 5)))
        indices = img1.matched_spectral_indices(img2)
        self.assertTupleEqual(indices, ((), ()))

        img3 = Image.from_box(Box((5, 5)), bands=tuple("gri"))
        with self.assertRaises(ValueError):
            img1.matched_spectral_indices(img3)

        with self.assertRaises(ValueError):
            img3.matched_spectral_indices(img1)

    def test_project(self):
        data = np.arange(30).reshape(5, 6)
        img = Image(data, yx0=(11, 15))

        result = img.project(bbox=Box((20, 20), (2, 3)))
        truth = np.zeros((20, 20))
        truth[9:14, 12:18] = data
        truth = Image(truth, yx0=(2, 3))
        self.assertImageEqual(result, truth)

        data = np.arange(60).reshape(3, 4, 5)
        img = Image(data, bands=tuple("gri"))
        result = img.project(tuple("gi"))
        truth = data[(0, 2), :]
        self.assertImageEqual(result, Image(truth, bands=tuple("gi")))

    def test_repeat(self):
        data = np.arange(18).reshape(3, 6)
        image = Image(data, yx0=(15, 32))
        result = image.repeat(tuple("grizy"))
        truth = np.array([data, data, data, data, data])
        truth = Image(truth, bands=tuple("grizy"), yx0=(15, 32))
        self.assertImageEqual(result, truth)

        with self.assertRaises(ValueError):
            result.repeat(tuple("ubv"))
