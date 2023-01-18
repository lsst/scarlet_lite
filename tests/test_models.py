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

import os

import numpy as np

from scarlet_lite import Blend, Image, Observation, Source
from scarlet_lite.component import default_adaprox_parameterization
from scarlet_lite.models import FreeFormComponent
from scarlet_lite.utils import integrated_circular_gaussian

from utils import ScarletTestCase


class TestFreeForm(ScarletTestCase):
    def setUp(self) -> None:
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
        self.data = data
        model_psf = integrated_circular_gaussian(sigma=0.8)
        self.detect = np.sum(data["images"], axis=0)
        self.centers = np.array([data["catalog"]["y"], data["catalog"]["x"]]).T
        bands = data["filters"]
        self.observation = Observation(
            Image(data["images"], bands=bands),
            Image(data["variance"], bands=bands),
            Image(1 / data["variance"], bands=bands),
            data["psfs"],
            model_psf[None],
            bands=bands,
        )

    def test_free_form_component(self):
        images = self.data["images"]

        # Test with no thresholding (sparsity)
        sources = []
        for i in range(5):
            component = FreeFormComponent(
                self.observation.bands,
                np.ones(5),
                images[i].copy(),
                self.observation.bbox,
            )
            sources.append(Source([component]))

        blend = Blend(sources, self.observation).fit_spectra()
        blend.parameterize(default_adaprox_parameterization)
        blend.fit(12, e_rel=1e-6)

        # Test with thresholding (sparsity)
        sources = []
        for i in range(5):
            component = FreeFormComponent(
                self.observation.bands,
                np.ones(5),
                images[i].copy(),
                self.observation.bbox,
                bg_rms=self.observation.noise_rms,
                bg_thresh=0.25,
                min_area=4,
            )
            sources.append(Source([component]))

        blend = Blend(sources, self.observation).fit_spectra()
        blend.parameterize(default_adaprox_parameterization)
        blend.fit(12, e_rel=1e-6)

        # Test with peak centers specified
        sources = []
        peaks = list(np.array([self.data["catalog"]["y"], self.data["catalog"]["x"]]).T.astype(int))
        for i in range(5):
            component = FreeFormComponent(
                self.observation.bands,
                np.ones(5),
                images[i].copy(),
                self.observation.bbox,
                peaks=peaks,
            )
            sources.append(Source([component]))

        blend = Blend(sources, self.observation).fit_spectra()
        blend.parameterize(default_adaprox_parameterization)
        blend.fit(12, e_rel=1e-6)

        # Tests for code blocks that are difficult to reach,
        # to complete test coverage
        component = blend.sources[-1].components[0]
        self.assertFalse(component.resize())
        self.assertEqual(str(component), "FreeFormComponent")
        self.assertEqual(repr(component), "FreeFormComponent")
        component.morph[:] = 0
        component.prox_morph(component.morph)
