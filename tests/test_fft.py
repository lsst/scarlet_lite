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
from numpy.testing import assert_array_equal, assert_almost_equal

import scarlet_lite.fft as fft
from scarlet_lite import Fourier
from utils import get_psfs


class TestCentering(object):
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
        """Test that _centered method is compatible with shift/unshift"""
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
        a_final = fft._centered(a_pad, shape)
        assert_array_equal(a_final, a0)


class TestFourier(object):
    def test_2d_psf_matching(self):
        """Test matching two 2D psfs"""
        # Narrow PSF
        psf1 = Fourier(get_psfs(1))
        # Wide PSF
        psf2 = Fourier(get_psfs(2))

        # Test narrow to wide
        kernel_1to2 = fft.match_psf(psf2, psf1)
        img2 = fft.convolve(psf1, kernel_1to2)
        assert_almost_equal(img2.image, psf2.image)

        # Test wide to narrow
        kernel_2to1 = fft.match_psf(psf1, psf2)
        img1 = fft.convolve(psf2, kernel_2to1)
        assert_almost_equal(img1.image, psf1.image)

    def test_multiband_psf_matching(self):
        """Test matching two PSFs with a spectral dimension"""
        # Narrow PSF
        psf1 = Fourier(get_psfs(1))
        # Wide PSF
        psf2 = Fourier(get_psfs((1, 2, 3)))

        # Nawrrow to wide
        kernel_1to2 = fft.match_psf(psf2, psf1)
        image = fft.convolve(kernel_1to2, psf1)
        assert_almost_equal(psf2.image, image.image)

        # Wide to narrow
        kernel_2to1 = fft.match_psf(psf1, psf2)
        image = fft.convolve(kernel_2to1, psf2).image

        for img in image:
            assert_almost_equal(img, psf1.image[0])
