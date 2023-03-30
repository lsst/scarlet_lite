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

import lsst.scarlet.lite.fft as fft
import numpy as np
from lsst.scarlet.lite import Fourier
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.signal import convolve as scipy_convolve
from utils import ScarletTestCase, get_psfs


class TestFourier(ScarletTestCase):
    """Test the centering and padding algorithms"""

    def test_shift(self):
        """Test that padding and fft shift/unshift are consistent"""
        a0 = np.ones((1, 1))
        a_pad = fft._pad(a0, (5, 4))
        truth = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        assert_array_equal(a_pad, truth)

        a_shift = np.fft.ifftshift(a_pad)
        truth = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        assert_array_equal(a_shift, truth)

        # Shifting back should give us a_pad again
        a_shift_back = np.fft.fftshift(a_shift)
        assert_array_equal(a_shift_back, a_pad)

    def test_center(self):
        """Test that centered method is compatible with shift/unshift"""
        shape = (5, 2)
        a0 = np.arange(10).reshape(shape)
        a_pad = fft._pad(a0, (9, 11))
        truth = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 6, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        assert_array_equal(a_pad, truth)

        a_shift = np.fft.ifftshift(a_pad)
        truth = [
            [4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        assert_array_equal(a_shift, truth)

        # Shifting back should give us a_pad again
        a_shift_back = np.fft.fftshift(a_shift)
        assert_array_equal(a_shift_back, a_pad)

        # _centered should undo the padding, returning the original array
        a_final = fft.centered(a_pad, shape)
        assert_array_equal(a_final, a0)

        with self.assertRaises(ValueError):
            fft.centered(a_final, (20, 20))

    def test_pad(self):
        x = np.arange(40).reshape(2, 4, 5)
        truth = np.zeros((2, 10, 11), dtype=int)
        truth[:, 3:7, 3:8] = x.copy()
        result = fft._pad(x, (10, 11), axes=(1, 2))
        assert_array_equal(result, truth)

        truth = np.zeros((4, 4, 5))
        truth[1:3] = x
        result = fft._pad(x, (4,), axes=0)
        assert_array_equal(result, truth)

        truth = np.pad(x, 5, mode="edge")
        result = fft._pad(x, (12, 14, 15), mode="edge")
        assert_array_equal(result, truth)

    def test_get_fft_shape(self):
        shape1 = (3, 11)
        shape2 = (5, 10)
        shape = tuple(fft.get_fft_shape(shape1, shape2))
        self.assertTupleEqual(shape, (12, 24))

        shape = fft.get_fft_shape(shape1, shape2, use_max=True)
        self.assertTupleEqual(shape, (8, 15))

        shape = fft.get_fft_shape(shape1, shape2, axes=1)
        self.assertTupleEqual(shape, (24,))

        shape = fft.get_fft_shape(shape1, shape2, axes=1, use_max=True)
        self.assertTupleEqual(shape, (15,))

        with self.assertRaises(ValueError):
            fft.get_fft_shape((1, 2), (1, 2, 3))

    def test_2d_psf_matching(self):
        """Test matching two 2D psfs"""
        # Narrow PSF
        psf1 = Fourier(get_psfs(1))
        # Wide PSF
        psf2 = Fourier(get_psfs(2))

        # Test narrow to wide
        kernel_1to2 = fft.match_kernel(psf2, psf1)
        img2 = fft.convolve(psf1, kernel_1to2)
        assert_almost_equal(img2.image, psf2.image)

        # Test wide to narrow
        kernel_2to1 = fft.match_kernel(psf1, psf2)
        img1 = fft.convolve(psf2, kernel_2to1)
        assert_almost_equal(img1.image, psf1.image)

    def test_from_fft(self):
        x = integrated_circular_gaussian(sigma=1.0)
        _x = np.pad(x, 3, mode="constant")
        fft_x = np.fft.rfftn(np.fft.ifftshift(_x))
        fourier = Fourier.from_fft(fft_x, (21, 21), (15, 15))
        assert_almost_equal(fourier.image, x)
        self.assertEqual(len(fourier), 15)

    def test_fourier(self):
        x = integrated_circular_gaussian(sigma=1.0)
        fourier = Fourier(x)
        assert_almost_equal(fourier.image, x)
        _x = np.pad(x, 3, mode="constant")
        fft_x = np.fft.rfftn(np.fft.ifftshift(_x))
        assert_almost_equal(fourier.fft((21, 21), (0, 1)), fft_x)
        self.assertEqual(len(fourier), 15)

        _x = np.pad(x, ((0, 0), (3, 3)), mode="constant")
        fft_x = np.fft.rfftn(np.fft.ifftshift(_x, axes=1), axes=(1,))
        assert_almost_equal(fourier.fft((21,), 1), fft_x)

        with self.assertRaises(ValueError):
            fourier.fft((3, 4, 5), (2, 3))

    def test_convolutions(self):
        x = integrated_circular_gaussian(sigma=1.0)
        y = integrated_circular_gaussian(sigma=1.3)

        with self.assertRaises(ValueError):
            fft._kspace_operation(Fourier(x), Fourier(y[None, :, :]), 3, operator.mul, (15, 15), (0, 1))

        convolved = fft.convolve(x, y, return_fourier=False)
        truth = scipy_convolve(x, y, mode="same", method="direct")
        assert_almost_equal(convolved, truth)

    def test_multiband_psf_matching(self):
        """Test matching two PSFs with a spectral dimension"""
        # Narrow PSF
        psf1 = Fourier(get_psfs(1))
        # Wide PSF
        psf2 = Fourier(get_psfs((1, 2, 3)))

        # Narrow to wide
        kernel_1to2 = fft.match_kernel(psf2, psf1)
        image = fft.convolve(kernel_1to2, psf1)
        assert_almost_equal(psf2.image, image.image)

        kernel_array = fft.match_kernel(psf2, psf1, return_fourier=False)
        assert_almost_equal(kernel_array, kernel_1to2.image)

        # Wide to narrow
        kernel_2to1 = fft.match_kernel(psf1, psf2)
        image = fft.convolve(kernel_2to1, psf2).image

        for img in image:
            assert_almost_equal(img, psf1.image[0])
