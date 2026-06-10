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
from lsst.scarlet.lite import Box, Image, ImagePsf, Observation
from lsst.scarlet.lite.observation import convolve as scarlet_convolve
from lsst.scarlet.lite.observation import get_filter_bounds, get_filter_coords
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.signal import convolve as scipy_convolve
from utils import ObservationData, ScarletTestCase


class TestObservation(ScarletTestCase):
    def setUp(self):
        bands = ("g", "r", "i")
        variance = np.ones((3, 35, 35), dtype=float)
        weights = 1 / variance
        psfs = np.array([integrated_circular_gaussian(sigma=sigma) for sigma in [1.05, 0.9, 1.2]])
        model_psf = integrated_circular_gaussian(sigma=0.8)

        # The spectrum of each source
        spectra = np.array(
            [
                [31, 10, 0],
                [0, 5, 20],
                [15, 8, 3],
                [20, 3, 4],
                [0, 30, 60],
            ]
        )

        # Use a point source for all of the sources
        morphs = [integrated_circular_gaussian(sigma=sigma) for sigma in [0.8, 3.1, 1.1, 2.1, 1.5]]
        # Make the second component a disk component
        morphs[1] = scipy_convolve(morphs[1], model_psf, mode="same")

        # Give the first two components the same center, and unique centers
        # for the remaining sources
        centers = [
            (10, 12),
            (10, 12),
            (20, 23),
            (20, 10),
            (25, 20),
        ]

        # Create the Observation
        test_data = ObservationData(bands, psfs, spectra, morphs, centers, model_psf)
        self.observation = Observation(
            test_data.convolved,
            variance,
            weights,
            ImagePsf(psfs, bands=bands),
            ImagePsf(model_psf[None]),
            bands=bands,
        )
        self.data = test_data
        self.spectra = spectra
        self.centers = centers
        self.morphs = morphs
        self.psfs = psfs
        self.bands = bands

    def tearDown(self):
        del self.data

    def test_real_convolution_function(self):
        """Test that real space convolution works"""
        images = self.data.images
        true_convolved = np.array(
            [
                scipy_convolve(images.data[b], self.observation.psf.data[b], mode="same")
                for b in range(len(images.data))
            ]
        )
        coords = get_filter_coords(self.observation.psf.data[0])
        bounds = get_filter_bounds(coords.reshape(-1, 2))
        convolved = scarlet_convolve(images.data, self.observation.psf.data, bounds)
        assert_almost_equal(convolved, true_convolved)

        with self.assertRaises(ValueError):
            get_filter_coords(np.arange(10))

        with self.assertRaises(ValueError):
            get_filter_coords(np.arange(16).reshape(4, 4))

    def test_constructors(self):
        np.random.seed(1)
        variance = np.random.normal(size=self.data.convolved.shape) ** 2
        observation = Observation(
            self.data.convolved,
            variance,
            1 / variance,
            ImagePsf(self.psfs, bands=self.bands),
            bands=self.bands,
        )
        self.assertImageEqual(observation.images, self.data.convolved)
        self.assertImageEqual(observation.variance, Image(variance, bands=self.bands))
        self.assertImageEqual(observation.weights, Image(1 / variance, bands=self.bands))
        assert_array_equal(observation.psf.data, self.psfs)
        self.assertIsNone(observation.model_psf)
        self.assertIsNone(observation.diff_kernel)
        assert_array_equal(observation.noise_rms, np.mean(np.sqrt(variance), axis=(1, 2)))
        self.assertBoxEqual(observation.bbox, Box(variance.shape[-2:]))
        self.assertIn(observation.mode, ["fft", "real"])

        # Set all of the pixels in the model to 1 more than the images,
        # so that images - model = np.ones, meaning the log_likelihood
        # is just half the sum of the weights.
        model = self.observation.images - 1
        self.assertEqual(observation.log_likelihood(model), -0.5 * np.sum(observation.weights.data))
        self.assertTupleEqual(observation.shape, (3, 35, 35))
        self.assertEqual(observation.n_bands, 3)
        self.assertEqual(observation.dtype, float)

    def test_psf_deprecation(self):
        """``psf`` is the supported argument; ndarray ``psf`` and the ``psfs``
        alias are deprecated, and ambiguous/missing PSFs raise."""
        images = self.data.convolved
        variance = np.ones(images.shape)
        weights = 1 / variance
        psf = ImagePsf(self.psfs, bands=self.bands)

        # A `Psf` is the supported path and emits no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            observation = Observation(images, variance, weights, psf, bands=self.bands)
        assert_array_equal(observation.psf.data, self.psfs)

        # Passing an ndarray as `psf` is deprecated but still works.
        with self.assertWarns(FutureWarning):
            observation = Observation(images, variance, weights, self.psfs, bands=self.bands)
        assert_array_equal(observation.psf.data, self.psfs)

        # The `psfs` alias is deprecated but still works.
        with self.assertWarns(FutureWarning):
            observation = Observation(images, variance, weights, psfs=self.psfs, bands=self.bands)
        assert_array_equal(observation.psf.data, self.psfs)

        # Providing both `psf` and `psfs` is ambiguous.
        with self.assertRaises(ValueError):
            Observation(images, variance, weights, psf, psfs=self.psfs, bands=self.bands)

        # A PSF is required.
        with self.assertRaises(ValueError):
            Observation(images, variance, weights, bands=self.bands)

        # Passing an ndarray as `model_psf` is deprecated, but a `Psf` is not.
        model_psf = integrated_circular_gaussian(sigma=0.8)[None]
        with self.assertWarns(FutureWarning):
            Observation(images, variance, weights, psf, model_psf, bands=self.bands)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            Observation(images, variance, weights, psf, ImagePsf(model_psf), bands=self.bands)

    def test_empty(self):
        """``Observation.empty`` resolves `psf`/`psfs`, re-imposes its required
        arguments, and deprecates ndarray PSFs."""
        bands = self.bands
        bbox = self.observation.bbox
        psf = ImagePsf(self.psfs, bands=bands)
        model_psf = ImagePsf(integrated_circular_gaussian(sigma=0.8)[None])

        # A `Psf` is the supported path and emits no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            observation = Observation.empty(bands=bands, psf=psf, model_psf=model_psf, bbox=bbox, dtype=float)
        self.assertBoxEqual(observation.bbox, bbox)
        assert_array_equal(observation.images.data, np.zeros((len(bands),) + bbox.shape))
        assert_array_equal(observation.psf.data, self.psfs)

        # The deprecated `psfs` alias still works, with a warning.
        with self.assertWarns(FutureWarning):
            observation = Observation.empty(
                bands=bands, psfs=self.psfs, model_psf=model_psf, bbox=bbox, dtype=float
            )
        assert_array_equal(observation.psf.data, self.psfs)

        # An ndarray `model_psf` is deprecated.
        with self.assertWarns(FutureWarning):
            Observation.empty(
                bands=bands,
                psf=psf,
                model_psf=integrated_circular_gaussian(sigma=0.8)[None],
                bbox=bbox,
                dtype=float,
            )

        # Providing both `psf` and `psfs` is ambiguous.
        with self.assertRaises(ValueError):
            Observation.empty(
                bands=bands, psf=psf, psfs=self.psfs, model_psf=model_psf, bbox=bbox, dtype=float
            )

        # Missing required arguments raise a clear ``TypeError`` (regression:
        # this previously raised ``AttributeError`` from building the dummy
        # image before validating the arguments).
        with self.assertRaises(TypeError):
            Observation.empty(bands=bands, psf=psf)

    def test_convolve(self):
        # Test the default initialization with no model psf,
        # menaing observation.convolve is a pass-through operation.
        np.random.seed(1)
        variance = np.random.normal(size=self.data.convolved.shape) ** 2
        observation = Observation(
            self.data.convolved,
            variance,
            1 / variance,
            ImagePsf(self.psfs, bands=self.bands),
            bands=self.bands,
        )
        assert_array_equal(observation.convolve(observation.images), observation.images)

        # Use an observation with a model_psf and difference kernel and check
        # convolution.
        observation = self.observation
        assert_array_equal(observation.diff_kernel.data, self.data.diff_kernel.image)
        assert_almost_equal(observation.convolve(self.data.images).data, observation.images.data)

        # Test real conversions
        deconvolved = self.data.images
        observation.mode = "real"
        self.assertImageAlmostEqual(observation.convolve(deconvolved), observation.images)

        # Test convolution with the gradient
        grad_convolved = np.array(
            [
                scipy_convolve(
                    deconvolved.data[band],
                    observation.diff_kernel.adjoint.data[band],
                    mode="same",
                )
                for band in range(len(deconvolved.data))
            ]
        )
        assert_almost_equal(observation.convolve(deconvolved, grad=True).data, grad_convolved)

        # Test that overriding the mode works
        real = observation.convolve(deconvolved, mode="real")
        self.assertImageAlmostEqual(real, observation.images)

        with self.assertRaises(ValueError):
            observation.convolve(deconvolved, mode="fake")

        # Test convolving a small image
        x = np.linspace(-3, 3, 7)
        small_array = integrated_circular_gaussian(x=x, y=x, sigma=0.8)
        small_psf = Image(np.array([small_array, small_array, small_array]), bands=observation.bands)
        truth = Image(
            np.array(
                [
                    scipy_convolve(
                        small_psf[band].data,
                        observation.diff_kernel.data[observation.bands.index(band)],
                        method="direct",
                        mode="same",
                    )
                    for band in observation.bands
                ]
            ),
            bands=observation.bands,
        )
        convolved = observation.convolve(small_psf)
        self.assertImageAlmostEqual(convolved, truth)

    def test_convolve_cache_default_off(self):
        """``Observation.convolve`` must default to ``cache=False`` so
        that ad-hoc downstream calls (per-source models, components
        with varying shapes) don't grow the long-lived difference and
        adjoint kernel FFT dicts. Regression for audit O-4.
        """
        observation = self.observation
        observation.diff_kernel.fourier._fft.clear()
        observation.diff_kernel.adjoint.fourier._fft.clear()

        # Convolve at the full-blend shape, then at a smaller shape.
        observation.convolve(self.data.images)
        coords = np.linspace(-3, 3, 7)
        small_array = integrated_circular_gaussian(x=coords, y=coords, sigma=0.8)
        small = Image(np.array([small_array, small_array, small_array]), bands=self.bands)
        observation.convolve(small)
        observation.convolve(self.data.images, grad=True)

        self.assertEqual(len(observation.diff_kernel.fourier._fft), 0)
        self.assertEqual(len(observation.diff_kernel.adjoint.fourier._fft), 0)

    def test_convolve_cache_opt_in(self):
        """``cache=True`` must populate the kernel's FFT dict and reuse
        the entry on subsequent calls at the same shape.
        """
        observation = self.observation
        observation.diff_kernel.fourier._fft.clear()
        observation.diff_kernel.adjoint.fourier._fft.clear()

        observation.convolve(self.data.images, cache=True)
        self.assertEqual(len(observation.diff_kernel.fourier._fft), 1)
        # Repeat call at the same shape: cache hit, dict size unchanged.
        observation.convolve(self.data.images, cache=True)
        self.assertEqual(len(observation.diff_kernel.fourier._fft), 1)
        # Gradient convolve uses the *separate* adjoint kernel's dict.
        observation.convolve(self.data.images, grad=True, cache=True)
        self.assertEqual(len(observation.diff_kernel.fourier._fft), 1)
        self.assertEqual(len(observation.diff_kernel.adjoint.fourier._fft), 1)

    def test_index_extraction(self):
        alpha_bands = ("g", "i", "r", "y", "z")
        images = np.arange(60).reshape(5, 3, 4)
        image_g = images[0]
        image_i = images[1]
        image_r = images[2]
        variance = np.arange(5)[:, None, None] * np.ones((5, 3, 4)) + 1
        weights = 1 / variance
        psfs = np.arange(5)[:, None, None] * np.ones((5, 2, 2))
        model_psf = np.zeros(9).reshape(3, 3)
        model_psf[1] = 1
        model_psf[:, 1] = 1
        model_psf[1, 1] = 2
        observation = Observation(
            images,
            variance,
            weights,
            ImagePsf(psfs, bands=alpha_bands),
            ImagePsf(model_psf[None]),
            bands=alpha_bands,
        )

        bands = "i"
        truth = Image(
            np.arange(12, 24).reshape(3, 4),
        )
        self.assertImageEqual(observation.images[bands], truth)

        bands = ("g", "r", "i")
        indices = observation.images.spectral_indices(bands)
        assert_array_equal(indices, (0, 2, 1))
        self.assertImageEqual(
            observation.images[bands],
            Image(np.array([image_g, image_r, image_i]), bands=("g", "r", "i")),
        )

    def test_slicing(self):
        np.random.seed(25)
        all_bands = tuple("grizy")
        images = np.random.random((5, 11, 13))
        variance = np.arange(5)[:, None, None] * np.ones((5, 11, 13)) + 1
        weights = 1 / variance
        psfs = np.arange(5)[:, None, None] * np.ones((5, 2, 2))
        model_psf = np.zeros(9).reshape(3, 3)
        model_psf[1] = 1
        model_psf[:, 1] = 1
        model_psf[1, 1] = 2
        bbox = Box((11, 13), origin=(12, 10))
        observation = Observation(
            images,
            variance,
            weights,
            ImagePsf(psfs, bands=all_bands),
            ImagePsf(model_psf[None]),
            bands=all_bands,
            bbox=bbox,
        )

        new_box = Box((3, 4), origin=(15, 13))
        sliced_bands = tuple("riz")
        sliced_observation = observation[sliced_bands, new_box]
        self.assertImageEqual(
            sliced_observation.images["r":"z"],
            Image(images[1:4, 3:6, 3:7], bands=sliced_bands, yx0=new_box.origin),
        )
        self.assertImageEqual(
            sliced_observation.variance["r":"z"],
            Image(variance[1:4, 3:6, 3:7], bands=sliced_bands, yx0=new_box.origin),
        )
        self.assertImageEqual(
            sliced_observation.weights["r":"z"],
            Image(weights[1:4, 3:6, 3:7], bands=sliced_bands, yx0=new_box.origin),
        )
        np.testing.assert_array_equal(
            sliced_observation.psf.data,
            psfs[1:4],
        )
        np.testing.assert_array_almost_equal(
            sliced_observation.model_psf.data,
            model_psf[None, :, :],
        )
        np.testing.assert_array_almost_equal(
            sliced_observation.noise_rms,
            observation.noise_rms[1:4],
        )
        self.assertBoxEqual(sliced_observation.bbox, new_box)

    def test_shallow_copy(self):
        observation_copy = self.observation.copy()
        self.assertObservationEqual(observation_copy, self.observation)

    def test_deep_copy(self):
        observation_copy = self.observation.copy(deep=True)
        self.assertObservationEqual(observation_copy, self.observation)

        # Modify the copy and check that the original is unchanged
        observation_copy.images._data += 1
        with self.assertRaises(AssertionError):
            self.assertImageEqual(observation_copy.images, self.observation.images)
        observation_copy.variance._data += 1
        with self.assertRaises(AssertionError):
            self.assertImageEqual(observation_copy.variance, self.observation.variance)
        observation_copy.weights._data += 1
        with self.assertRaises(AssertionError):
            self.assertImageEqual(observation_copy.weights, self.observation.weights)
