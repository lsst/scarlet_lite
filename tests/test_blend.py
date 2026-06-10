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

from __future__ import annotations

from typing import cast

import numpy as np
from lsst.scarlet.lite import Blend, Box, Image, ImagePsf, Observation, Source
from lsst.scarlet.lite.component import CubeComponent, FactorizedComponent, default_adaprox_parameterization
from lsst.scarlet.lite.initialization import FactorizedInitialization
from lsst.scarlet.lite.operators import Monotonicity
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal, assert_raises
from scipy.signal import convolve as scipy_convolve
from utils import ObservationData, ScarletTestCase


class TestBlend(ScarletTestCase):
    def setUp(self):
        bands = ("g", "r", "i")
        yx0 = (1000, 2000)
        # The PSF in each band of the "observation"
        psfs = np.array([integrated_circular_gaussian(sigma=sigma) for sigma in [1.05, 0.9, 1.2]])
        # The PSF of the model
        model_psf = integrated_circular_gaussian(sigma=0.8)

        # The spectrum of each source
        spectra = np.array(
            [
                [40, 10, 0],
                [0, 25, 40],
                [15, 8, 3],
                [20, 3, 4],
                [0, 30, 60],
            ],
            dtype=float,
        )

        # Use a point source for all of the sources
        morphs = [integrated_circular_gaussian(sigma=sigma) for sigma in [0.8, 2.5, 1.1, 2.1, 1.5]]
        # Make the second component a disk component
        morphs[1] = scipy_convolve(morphs[1], model_psf, mode="same")

        # Give the first two components the same center, and unique centers
        # for the remaining sources
        centers = [
            (1010, 2012),
            (1010, 2012),
            (1020, 2023),
            (1020, 2010),
            (1025, 2020),
        ]

        # Create the simulated image and associated data products
        test_data = ObservationData(bands, psfs, spectra, morphs, centers, model_psf, yx0=yx0)

        # Create the Observation
        variance = np.ones((3, 35, 35), dtype=float) * 1e-2
        weights = 1 / variance
        weights = weights / np.max(weights)
        self.observation = Observation(
            test_data.convolved,
            variance,
            weights,
            ImagePsf(psfs, bands=bands),
            ImagePsf(model_psf[None]),
            bands=bands,
            bbox=Box(variance.shape[-2:], origin=yx0),
        )
        self.data = test_data
        self.spectra = spectra
        self.centers = centers
        self.morphs = morphs

        components = []
        for spectrum, center, morph, data_morph in zip(
            self.spectra, self.centers, self.morphs, self.data.morphs
        ):
            components.append(
                FactorizedComponent(
                    bands=bands,
                    spectrum=spectrum,
                    morph=morph,
                    bbox=data_morph.bbox,
                    peak=center,
                )
            )

        sources = [Source(components[:2])]
        sources += [Source([component]) for component in components[2:]]

        self.blend = Blend(sources, self.observation)

    def test_exact(self):
        """Test that a blend model initialized with the exact solution
        builds the model correctly
        """
        blend = self.blend
        self.assertEqual(len(blend.components), 5)
        self.assertEqual(len(blend.sources), 4)
        self.assertBoxEqual(blend.bbox, Box(self.data.images.shape[1:], self.observation.bbox.origin))
        self.assertImageAlmostEqual(blend.get_model(), self.data.images)
        self.assertImageAlmostEqual(blend.get_model(convolve=True), self.observation.images)
        self.assertImageAlmostEqual(
            self.observation.convolve(blend.get_model(), mode="real"),
            self.observation.images,
        )

        # Test that the log likelihood is very small
        assert_almost_equal([blend.log_likelihood], [0])

        # Test that grad_log_likelihood updates the loss
        self.assertListEqual(blend.loss, [])
        blend._grad_log_likelihood()
        assert_almost_equal(blend.loss, [0])

        # Remove one of the sources and calculate the non-zero log_likelihood
        del blend.sources[-1]
        # Update the loss function and check that the loss changed
        blend._grad_log_likelihood()
        assert_almost_equal(blend.log_likelihood, -60.011720889007485)
        assert_almost_equal(blend.loss, [0, -60.011720889007485])

    def test_fit_spectra(self):
        """Test that fitting the spectra with exact morphologies is
        identical to the multiband image
        """
        np.random.seed(0)
        blend = self.blend

        # Change the initial spectra so that they can be fit later
        for component in blend.components:
            c = cast(FactorizedComponent, component)
            c.spectrum[:] = np.random.rand(3) * 10

        with assert_raises(AssertionError):
            # Since the spectra have not yet been fit,
            # the model and images should not be equal
            self.assertImageEqual(blend.get_model(), self.data.images)

        # We initialized all of the morphologies exactly,
        # so fitting the spectra should give a nearly exact solution
        blend.fit_spectra()

        self.assertEqual(len(blend.components), 5)
        self.assertEqual(len(blend.sources), 4)
        self.assertBoxEqual(blend.bbox, self.observation.bbox)
        self.assertImageAlmostEqual(blend.get_model(), self.data.images)
        self.assertImageAlmostEqual(blend.get_model(convolve=True), self.observation.images)

    def test_fit(self):
        observation = self.observation
        np.random.seed(0)
        images = observation.images.copy()
        noise = np.random.normal(size=observation.images.shape) * 1e-2
        observation.images._data += noise

        monotonicity = Monotonicity((101, 101))
        init = FactorizedInitialization(observation, self.centers, monotonicity=monotonicity)

        blend = Blend(init.sources, self.observation).fit_spectra()
        blend.parameterize(default_adaprox_parameterization)
        blend.fit(100)

        self.assertImageAlmostEqual(blend.get_model(convolve=True), images, decimal=1)

    def test_non_factorized(self):
        np.random.seed(1)
        blend = self.blend
        # Remove the disk component from the first source
        model = self.spectra[1][:, None, None] * self.morphs[1][None, :, :]
        yx0 = blend.sources[0].components[1].bbox.origin
        blend.sources[0].components = blend.sources[0].components[:1]

        # Change the initial spectra so that they can be fit later
        for component in blend.components:
            c = cast(FactorizedComponent, component)
            c.spectrum[:] = np.random.rand(3) * 10

        with assert_raises(AssertionError):
            # Since the spectra have not yet been fit,
            # the model and images should not be equal
            self.assertImageEqual(blend.get_model(), self.data.images)

        # Remove the disk component from the first source
        blend.sources[0].components = blend.sources[0].components[:1]
        # Create a new source for the disk with a non-factorized component
        component = CubeComponent(Image(model, bands=self.blend.observation.bands, yx0=yx0), (0, 0))
        blend.sources.append(Source([component]))

        blend.fit_spectra()

        self.assertEqual(len(blend.components), 5)
        self.assertEqual(len(blend.sources), 5)
        self.assertImageAlmostEqual(blend.get_model(), self.data.images)

    def test_clipping(self):
        blend = self.blend

        # Change the initial spectra so that they can be fit later
        for component in blend.components:
            c = cast(FactorizedComponent, component)
            c.spectrum[:] = np.random.rand(3) * 10

        with assert_raises(AssertionError):
            # Since the spectra have not yet been fit,
            # the model and images should not be equal
            self.assertImageEqual(blend.get_model(), self.data.images)

        # Add an empty source
        zero_model = Image.from_box(Box((5, 5), (30, 0)), bands=blend.observation.bands)
        component = CubeComponent(zero_model, (0, 0))
        blend.sources.append(Source([component]))

        blend.fit_spectra(clip=True)

        self.assertEqual(len(blend.components), 5)
        self.assertEqual(len(blend.sources), 5)
        self.assertImageAlmostEqual(blend.get_model(), self.data.images)

    def test_shallow_copy(self):
        blend = self.blend
        blend.metadata = {"test": "value"}
        blend_copy = blend.copy()

        self.assertIsNot(blend_copy, blend)
        self.assertEqual(len(blend_copy.sources), len(blend.sources))
        for source_copy, source in zip(blend_copy.sources, blend.sources):
            self.assertSourceEqual(source_copy, source)

        self.assertObservationEqual(blend_copy.observation, blend.observation)

        self.assertDictEqual(blend_copy.metadata, blend.metadata)

    def test_deepcopy(self):
        blend = self.blend
        blend.metadata = {"test": "value"}
        blend_copy = blend.copy(deep=True)

        self.assertIsNot(blend_copy, blend)
        self.assertEqual(len(blend_copy.sources), len(blend.sources))
        for source_copy, source in zip(blend_copy.sources, blend.sources):
            self.assertSourceEqual(source_copy, source)

            with self.assertRaises(AssertionError):
                source_copy.components[0]._spectrum.x += 1
                self.assertSourceEqual(source_copy, source)

        self.assertObservationEqual(blend_copy.observation, blend.observation)
        self.assertDictEqual(blend_copy.metadata, blend.metadata)
        blend_copy.metadata["test"] = "new_value"
        with self.assertRaises(AssertionError):
            self.assertDictEqual(blend_copy.metadata, blend.metadata)

    def test_slice(self):
        blend = self.blend
        blend.metadata = {"test": "value"}
        blend_sliced = blend["g":"r"]
        self.assertEqual(len(blend.sources), len(blend_sliced.sources))

        for source_sliced, source in zip(blend_sliced.sources, blend.sources):
            self.assertSourceEqual(source_sliced, source["g":"r"])

        self.assertObservationEqual(blend_sliced.observation, blend.observation["g":"r"])
        self.assertDictEqual(blend_sliced.metadata, blend.metadata)

    def test_reorder(self):
        blend = self.blend
        blend.metadata = {"test": "value"}
        indices = ("i", "g", "r")
        blend_reordered = blend[indices]
        self.assertEqual(len(blend.sources), len(blend_reordered.sources))

        for source_reordered, source in zip(blend_reordered.sources, blend.sources):
            self.assertSourceEqual(source_reordered, source[indices])

        self.assertObservationEqual(blend_reordered.observation, blend.observation[indices])
        self.assertDictEqual(blend_reordered.metadata, blend.metadata)

    def test_subset(self):
        blend = self.blend
        blend.metadata = {"test": "value"}
        blend_subset = blend[("r",)]
        self.assertEqual(len(blend.sources), len(blend_subset.sources))

        for source_subset, source in zip(blend_subset.sources, blend.sources):
            self.assertSourceEqual(source_subset, source["r"])

        self.assertObservationEqual(blend_subset.observation, blend.observation["r"])
        self.assertDictEqual(blend_subset.metadata, blend.metadata)

    def test_indexing_errors(self):
        blend = self.blend

        with self.assertRaises(IndexError):
            blend["x"]

        with self.assertRaises(IndexError):
            blend[("r", "x")]

        with self.assertRaises(IndexError):
            blend["r":"x"]

        with self.assertRaises(IndexError):
            blend["x":"i"]

        with self.assertRaises(IndexError):
            blend["g", "x", "i"]

        with self.assertRaises(IndexError):
            blend[Box((0, 0), (10, 10))]

        with self.assertRaises(IndexError):
            blend[:, 10:20, 10:20]

        with self.assertRaises(IndexError):
            blend[1:]

        with self.assertRaises(IndexError):
            blend[1]

        with self.assertRaises(IndexError):
            blend[0, 1]
