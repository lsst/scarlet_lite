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

from typing import cast
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.signal import convolve as scipy_convolve

from scarlet_lite import Blend, Source, FactorizedComponent, Observation, Box
from scarlet_lite.component import default_adaprox_parameterization
from scarlet_lite.initialization import init_all_sources_chi2
from scarlet_lite.utils import integrated_circular_gaussian
from utils import ObservationData


class TestBlend(unittest.TestCase):
    def setUp(self):
        bands = ("g", "r", "i")
        # The PSF in each band of the "obseration"
        psfs = np.array(
            [integrated_circular_gaussian(sigma=sigma) for sigma in [1.05, 0.9, 1.2]]
        )
        # The PSF of the model
        model_psf = integrated_circular_gaussian(sigma=0.8)

        # The spectrum of each source
        spectra = np.array(
            [
                [40, 10, 0],
                [0, 5, 20],
                [15, 8, 3],
                [20, 3, 4],
                [0, 30, 60],
            ],
            dtype=float,
        )

        # Use a point source for all of the sources
        morphs = [
            integrated_circular_gaussian(sigma=sigma)
            for sigma in [0.8, 3.1, 1.1, 2.1, 1.5]
        ]
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

        # Create the simulated image and associated data products
        test_data = ObservationData(bands, psfs, spectra, morphs, centers, model_psf)

        # Create the Observation
        variance = np.ones((3, 35, 35), dtype=float)
        weights = 1 / variance
        self.observation = Observation(
            test_data.convolved, variance, weights, psfs, model_psf[None], bands=bands,
        )
        self.data = test_data
        self.spectra = spectra
        self.centers = centers
        self.morphs = morphs

        components = []
        for spectrum, center, morph, bbox in zip(
            self.spectra, self.centers, self.morphs, self.data.boxes
        ):
            components.append(
                FactorizedComponent(
                    bands=bands,
                    sed=spectrum,
                    morph=morph,
                    bbox=bbox,
                    model_bbox=self.observation.bbox,
                    center=center,
                )
            )

        sources = [Source(components[:2], float)]
        sources += [Source([component], float) for component in components[2:]]

        self.blend = Blend(sources, self.observation)

    def test_exact(self):
        """Test that a blend model initialized with the exact solution
        builds the model correctly
        """
        blend = self.blend
        assert len(blend.components) == 5
        assert len(blend.sources) == 4
        assert blend.bbox == Box(self.data.images.shape)
        assert_almost_equal(blend.get_model(), self.data.images)
        assert_almost_equal(blend.get_model(convolve=True), self.observation.images)
        assert_almost_equal(
            self.observation.convolve(blend.get_model(), mode="real"),
            self.observation.images,
        )

        # Test that the log likelihood is very small
        assert_almost_equal([blend.log_likelihood], [0])

        # Test that grad_log_likelihood updates the loss
        assert blend.loss == []
        blend._grad_log_likelihood()
        assert_almost_equal(blend.loss, [0])

        # Remove one of the sources and calculate the non-zero log_likelihood
        del blend._components[-1]
        del blend.sources[-1]
        # Check that the log-likelihood is unchanged because no gradient
        # step has been performed
        assert_almost_equal(blend.log_likelihood, 0)
        # Update the loss function and check that the loss changed
        blend._grad_log_likelihood()
        assert_almost_equal(blend.log_likelihood, -60.011720889007485)
        assert_almost_equal(blend.loss, [0, -60.011720889007485])

    def test_fit_spectra(self):
        """Test that fitting the spectra with exact morpholigies is
        identical to the mutliband image
        """
        np.random.seed(0)
        blend = self.blend
        # Change the initial SEDs so that they can be fit later
        for component in blend.components:
            c = cast(FactorizedComponent, component)
            c.sed[:] = np.random.rand(3) * 10

        with np.testing.assert_raises(AssertionError):
            # Since the spectra have not yet been fit,
            # the model and images should not be equal
            np.testing.assert_array_equal(blend.get_model(), self.data.images)

        # We initialized all of the morphologies exactly,
        # so fitting the spectra should give a nearly exact solution
        blend.fit_spectra()

        self.assertEqual(len(blend.components), 5)
        self.assertEqual(len(blend.sources), 4)
        self.assertEqual(blend.bbox, Box(self.observation.bbox.shape))
        assert_almost_equal(blend.get_model(), self.data.images)
        assert_almost_equal(blend.get_model(convolve=True), self.observation.images)

    def test_fit(self):
        observation = self.observation
        np.random.seed(0)
        images = observation.images.copy()
        noise = np.random.normal(size=observation.images.shape) * 1e-2
        observation.images._data += noise

        sources = init_all_sources_chi2(
            observation,
            self.centers,
        )

        for src in sources:
            src.parameterize(default_adaprox_parameterization)

        # The default initialization only assigns one component to
        # each source, so we add an additional disk component for the
        # first source
        disk_center = sources[0].components[0].center  # type: ignore
        disk_origin = (0, disk_center[0] - 7, disk_center[1] - 7)
        disk_component = FactorizedComponent(
            np.random.rand(3) * 10,
            integrated_circular_gaussian(sigma=2.5),
            Box((3, 15, 15), disk_origin),
            observation.bbox,
            disk_center,
        )
        sources[0].components.append(disk_component)
        blend = Blend(sources, self.observation).fit_spectra()
        blend.fit(100, min_iter=100)
        assert_almost_equal(blend.get_model(convolve=True), images, decimal=1)
