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

import warnings

import numpy as np
from lsst.scarlet.lite import Image, ImagePsf, Observation, Psf
from lsst.scarlet.lite.fft import match_kernel
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal, assert_array_equal
from utils import ScarletTestCase, get_psfs


class TestImagePsf(ScarletTestCase):
    def setUp(self):
        self.bands = ("g", "r", "i")
        self.psf_data = get_psfs([1.05, 0.9, 1.2])
        self.model_psf_data = integrated_circular_gaussian(sigma=0.8)[None]
        self.psf = ImagePsf(self.psf_data, bands=self.bands)
        self.model_psf = ImagePsf(self.model_psf_data)

    def test_properties(self):
        self.assertEqual(self.psf.bands, self.bands)
        self.assertEqual(self.psf.dtype, self.psf_data.dtype)
        self.assertTupleEqual(self.psf.shape, self.psf_data.shape[-2:])
        assert_array_equal(self.psf.data, self.psf_data)
        self.assertIsInstance(self.psf, Psf)

    def test_get_image(self):
        image = self.psf.get_image()
        self.assertIsInstance(image, Image)
        self.assertEqual(image.bands, self.bands)
        assert_array_equal(image.data, self.psf_data)

    def test_get_image_bandless(self):
        """A band-less PSF returns a 2D image, dropping a singleton broadcast
        axis, but refuses to silently collapse a genuine multi-band cube."""
        # The model PSF is stored as a (1, y, x) broadcast cube with no bands.
        bandless = ImagePsf(self.model_psf_data)
        image = bandless.get_image()
        self.assertEqual(image.bands, ())
        self.assertEqual(image.data.ndim, 2)
        assert_array_equal(image.data, self.model_psf_data[0])

        # A genuine multi-band cube without band labels is ambiguous.
        ambiguous = ImagePsf(np.ones((3, 5, 5)))
        with self.assertRaises(ValueError):
            ambiguous.get_image()

    def test_astype(self):
        psf32 = self.psf.astype(np.float32)
        self.assertEqual(psf32.dtype, np.float32)
        self.assertEqual(psf32.bands, self.bands)
        assert_array_equal(psf32.data, self.psf_data.astype(np.float32))

    def test_getitem(self):
        sub = self.psf[("i", "g")]
        self.assertEqual(sub.bands, ("i", "g"))
        assert_array_equal(sub.data[0], self.psf_data[2])
        assert_array_equal(sub.data[1], self.psf_data[0])

    def test_match(self):
        """``match`` must reproduce ``fft.match_kernel``."""
        diff = self.psf.match(self.model_psf)
        self.assertIsInstance(diff, ImagePsf)
        self.assertEqual(diff.bands, self.bands)
        truth = match_kernel(self.psf_data, self.model_psf_data, padding=3)
        assert_array_equal(diff.data, truth.image)

    def test_convolve_fft_matches_real(self):
        """FFT and real-space convolution must agree."""
        diff = self.psf.match(self.model_psf)
        rng = np.random.RandomState(1)
        image = Image(rng.normal(size=(3, 35, 35)), bands=self.bands)
        fft = diff.convolve(image, mode="fft")
        real = diff.convolve(image, mode="real")
        self.assertImageAlmostEqual(fft, real, decimal=5)
        # The result keeps the image's bands and origin.
        self.assertEqual(fft.bands, image.bands)
        self.assertEqual(fft.yx0, image.yx0)

    def test_grad_is_flipped_convolution(self):
        """``grad`` convolves with the spatially-flipped kernel."""
        diff = self.psf.match(self.model_psf)
        rng = np.random.RandomState(2)
        image = Image(rng.normal(size=(3, 35, 35)), bands=self.bands)
        grad = diff.grad(image, mode="fft")
        flipped = ImagePsf(diff.data[:, ::-1, ::-1], bands=self.bands)
        truth = flipped.convolve(image, mode="fft")
        self.assertImageAlmostEqual(grad, truth)
        # The adjoint kernel is the flipped difference kernel.
        assert_array_equal(diff.adjoint.data, diff.data[:, ::-1, ::-1])

    def test_adjoint_property(self):
        """``grad`` must be the exact transpose of ``convolve``:
        <A x, y> == <x, A^T y> for all x, y.
        """
        diff = self.psf.match(self.model_psf)
        rng = np.random.RandomState(3)
        x = Image(rng.normal(size=(3, 35, 35)), bands=self.bands)
        y = Image(rng.normal(size=(3, 35, 35)), bands=self.bands)
        ax = diff.convolve(x, mode="fft")
        aty = diff.grad(y, mode="fft")
        lhs = np.sum(ax.data * y.data)
        rhs = np.sum(x.data * aty.data)
        assert_almost_equal(lhs, rhs, decimal=6)

    def test_cache_off_by_default(self):
        """The FFT cache must not grow unless ``cache=True``."""
        diff = self.psf.match(self.model_psf)
        diff.fourier._fft.clear()
        image = Image(np.zeros((3, 35, 35)), bands=self.bands)
        diff.convolve(image)
        diff.convolve(Image(np.zeros((3, 21, 21)), bands=self.bands))
        self.assertEqual(len(diff.fourier._fft), 0)

    def test_cache_opt_in_and_separate_adjoint(self):
        """``cache=True`` grows the kernel's dict; the adjoint kernel keeps
        its own independent cache."""
        diff = self.psf.match(self.model_psf)
        diff.fourier._fft.clear()
        diff.adjoint.fourier._fft.clear()
        image = Image(np.zeros((3, 35, 35)), bands=self.bands)

        diff.convolve(image, cache=True)
        self.assertEqual(len(diff.fourier._fft), 1)
        # Forward cache untouched by the gradient pass; adjoint grows instead.
        diff.grad(image, cache=True)
        self.assertEqual(len(diff.fourier._fft), 1)
        self.assertEqual(len(diff.adjoint.fourier._fft), 1)

    def test_observation_accepts_psf(self):
        """``Observation`` accepts a ``Psf`` or, deprecated, an ndarray."""

        rng = np.random.RandomState(4)
        images = rng.normal(size=(3, 35, 35))
        variance = np.ones((3, 35, 35))

        from_psf = Observation(images, variance, 1 / variance, self.psf, self.model_psf, bands=self.bands)
        # The ndarray path still works, but is deprecated.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            from_array = Observation(
                images, variance, 1 / variance, self.psf_data, self.model_psf_data, bands=self.bands
            )
        self.assertIsInstance(from_psf.psf, ImagePsf)
        self.assertIsInstance(from_array.psf, ImagePsf)
        # Both routes build identical difference kernels and ndarray psfs.
        assert_array_equal(from_psf.psf.data, self.psf_data)
        assert_array_equal(from_psf.diff_kernel.data, from_array.diff_kernel.data)
