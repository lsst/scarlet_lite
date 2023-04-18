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

import numpy as np
from lsst.scarlet.lite import Box, Image, Source
from lsst.scarlet.lite.component import FactorizedComponent
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from utils import ScarletTestCase


class TestSource(ScarletTestCase):
    def test_constructor(self):
        # Test empty source
        source = Source([])

        self.assertEqual(source.n_components, 0)
        self.assertIsNone(source.center)
        self.assertIsNone(source.source_center)
        self.assertTrue(source.is_null)
        self.assertBoxEqual(source.bbox, Box((0, 0)))
        self.assertTupleEqual(source.bands, ())

        # Test a source with a single component
        bands = tuple("grizy")
        center = (27, 32)
        morph1 = integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        spectrum1 = np.arange(5).astype(np.float32)
        component_box1 = Box((15, 15), (20, 25))
        components = [
            FactorizedComponent(
                bands,
                spectrum1,
                morph1,
                component_box1,
                center,
            ),
        ]
        source = Source(components)
        self.assertEqual(source.n_components, 1)
        self.assertTupleEqual(source.center, center)
        self.assertTupleEqual(source.source_center, (7, 7))
        self.assertFalse(source.is_null)
        self.assertBoxEqual(source.bbox, component_box1)
        self.assertTupleEqual(source.bands, bands)
        self.assertImageEqual(
            source.get_model(),
            Image(
                spectrum1[:, None, None] * morph1[None, :, :],
                yx0=component_box1.origin,
                bands=bands,
            ),
        )
        self.assertIsNone(source.get_model(True))
        self.assertEqual(source.get_model().dtype, np.float32)

        # Test a source with multiple components
        morph2 = integrated_circular_gaussian(sigma=2.1).astype(np.float32)
        spectrum2 = np.arange(5)[::-1].astype(np.float32)
        component_box2 = Box((15, 15), (10, 35))

        components = [
            FactorizedComponent(
                bands,
                spectrum1,
                morph1,
                component_box1,
                center,
            ),
            FactorizedComponent(
                bands,
                spectrum2,
                morph2,
                component_box2,
                center,
            ),
        ]
        source = Source(components)
        self.assertEqual(source.n_components, 2)
        self.assertTupleEqual(source.center, center)
        self.assertTupleEqual(source.source_center, (17, 7))
        self.assertFalse(source.is_null)
        self.assertBoxEqual(source.bbox, Box((25, 25), (10, 25)))
        self.assertTupleEqual(source.bands, bands)
        self.assertEqual(str(source), "Source<2>")
        self.assertEqual(source.get_model().dtype, np.float32)

        model = np.zeros((5, 25, 25), dtype=np.float32)
        model[:, 10:25, :15] = spectrum1[:, None, None] * morph1[None, :, :]
        model[:, :15, 10:25] += spectrum2[:, None, None] * morph2[None, :, :]
        model = Image(model, yx0=(10, 25), bands=tuple("grizy"))

        self.assertImageEqual(
            source.get_model(),
            model,
        )
        self.assertIsNone(source.get_model(True))

        source = Source([])
        result = source.get_model()
        self.assertEqual(result, 0)
