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

import json
import os

import numpy as np
from lsst.scarlet.lite import Blend, Image, Observation, io
from lsst.scarlet.lite.initialization import FactorizedChi2Initialization
from lsst.scarlet.lite.models.free_form import FactorizedFreeFormComponent
from lsst.scarlet.lite.operators import Monotonicity
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal
from utils import ScarletTestCase


class TestIo(ScarletTestCase):
    def setUp(self) -> None:
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
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
        monotonicity = Monotonicity((101, 101))
        init = FactorizedChi2Initialization(self.observation, self.centers, monotonicity=monotonicity)
        self.blend = Blend(init.sources, self.observation)

    def test_json(self):
        blend = self.blend
        for i in range(len(blend.sources)):
            blend.sources[i].record_id = i * 10
            blend.sources[i].peak_id = i
        blend_data = io.ScarletBlendData.from_blend(blend, (51, 67))
        model_data = io.ScarletModelData(
            psf=self.observation.model_psf,
            blends={1: blend_data},
        )

        # Get the json string for the model
        model_str = model_data.json()
        # Load the model string from the json
        model_dict = json.loads(model_str)
        # Load the full set of model data classes from the json string
        model_data = io.ScarletModelData.parse_obj(model_dict)
        # Convert the model data into scarlet models
        loaded_blend = model_data.blends[1].minimal_data_to_blend(
            model_psf=model_data.psf,
            dtype=blend.observation.dtype,
        )

        self.assertEqual(len(blend.sources), len(loaded_blend.sources))
        self.assertEqual(len(blend.components), len(loaded_blend.components))
        self.assertImageAlmostEqual(blend.get_model(), loaded_blend.get_model())

        for sidx in range(len(blend.sources)):
            source1 = blend.sources[sidx]
            source2 = loaded_blend.sources[sidx]
            self.assertTupleEqual(source1.center, source2.center)
            self.assertEqual(len(source1.components), len(source2.components))
            self.assertBoxEqual(source1.bbox, source2.bbox)
            for cidx in range(len(source1.components)):
                component1 = source1.components[cidx]
                component2 = source2.components[cidx]
                self.assertEqual(component1.peak, component2.peak)
                assert_almost_equal(component1.spectrum, component2.spectrum)
                assert_almost_equal(component1.morph, component2.morph)
                self.assertBoxEqual(component1.bbox, component2.bbox)

    def test_cube_component(self):
        blend = self.blend
        for i in range(len(blend.sources)):
            blend.sources[i].record_id = i * 10
            blend.sources[i].peak_id = i
        component = blend.sources[-1].components[-1]
        # Replace one of the components with a Free-Form component.
        blend.sources[-1].components[-1] = FactorizedFreeFormComponent(
            bands=self.observation.bands,
            spectrum=component.spectrum,
            morph=component.morph,
            model_bbox=self.observation.bbox,
        )

        blend_data = io.ScarletBlendData.from_blend(blend, (51, 67))
        model_data = io.ScarletModelData(
            psf=self.observation.model_psf,
            blends={1: blend_data},
        )

        # Get the json string for the model
        model_str = model_data.json()
        # Load the model string from the json
        model_dict = json.loads(model_str)
        # Load the full set of model data classes from the json string
        model_data = io.ScarletModelData.parse_obj(model_dict)
        # Convert the model data into scarlet models
        loaded_blend = model_data.blends[1].minimal_data_to_blend(
            model_psf=model_data.psf,
            dtype=blend.observation.dtype,
        )

        self.assertEqual(len(blend.sources), len(loaded_blend.sources))
        self.assertEqual(len(blend.components), len(loaded_blend.components))
        self.assertImageAlmostEqual(blend.get_model(), loaded_blend.get_model())
