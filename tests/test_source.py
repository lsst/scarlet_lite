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
    def setUp(self) -> None:
        super().setUp()
        self.bands = tuple("grizy")
        self.center = (27, 32)
        self.morph1 = integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        self.spectrum1 = np.arange(5).astype(np.float32)
        self.component_box1 = Box((15, 15), (20, 25))
        self.morph2 = integrated_circular_gaussian(sigma=2.1).astype(np.float32)
        self.spectrum2 = np.arange(5)[::-1].astype(np.float32)
        self.component_box2 = Box((15, 15), (10, 35))
        self.components = [
            FactorizedComponent(
                self.bands,
                self.spectrum1,
                self.morph1,
                self.component_box1,
                self.center,
            ),
            FactorizedComponent(
                self.bands,
                self.spectrum2,
                self.morph2,
                self.component_box2,
                self.center,
            ),
        ]

    def test_empty_constructor(self):
        source = Source([])

        self.assertEqual(source.n_components, 0)
        self.assertIsNone(source.center)
        self.assertIsNone(source.source_center)
        self.assertTrue(source.is_null)
        self.assertBoxEqual(source.bbox, Box((0, 0)))
        self.assertTupleEqual(source.bands, ())

    def test_single_component_constructor(self):
        source = Source(self.components[:1])
        self.assertEqual(source.n_components, 1)
        self.assertTupleEqual(source.center, self.center)
        self.assertTupleEqual(source.source_center, (7, 7))
        self.assertFalse(source.is_null)
        self.assertBoxEqual(source.bbox, self.component_box1)
        self.assertTupleEqual(source.bands, self.bands)
        self.assertImageEqual(
            source.get_model(),
            Image(
                self.spectrum1[:, None, None] * self.morph1[None, :, :],
                yx0=self.component_box1.origin,
                bands=self.bands,
            ),
        )
        self.assertIsNone(source.get_model(True))
        self.assertEqual(source.get_model().dtype, np.float32)

    def test_multiple_component_constructor(self):
        # Test a source with multiple components
        source = Source(self.components)
        self.assertEqual(source.n_components, 2)
        self.assertTupleEqual(source.center, self.center)
        self.assertTupleEqual(source.source_center, (17, 7))
        self.assertFalse(source.is_null)
        self.assertBoxEqual(source.bbox, Box((25, 25), (10, 25)))
        self.assertTupleEqual(source.bands, self.bands)
        self.assertEqual(str(source), "Source<2>")
        self.assertEqual(source.get_model().dtype, np.float32)

        model = np.zeros((5, 25, 25), dtype=np.float32)
        model[:, 10:25, :15] = self.spectrum1[:, None, None] * self.morph1[None, :, :]
        model[:, :15, 10:25] += self.spectrum2[:, None, None] * self.morph2[None, :, :]
        model = Image(model, yx0=(10, 25), bands=self.bands)

        self.assertImageEqual(
            source.get_model(),
            model,
        )
        self.assertIsNone(source.get_model(True))

        source = Source([])
        result = source.get_model()
        self.assertEqual(result, 0)

    def test_shallow_copy(self):
        source = Source(self.components)
        source_copy = source.copy()

        self.assertIsNot(source, source_copy)
        self.assertEqual(source.n_components, 2)
        self.assertEqual(source.n_components, source_copy.n_components)
        self.assertFactorizedComponentEqual(source.components[0], source_copy.components[0])
        self.assertFactorizedComponentEqual(source.components[1], source_copy.components[1])
        self.assertIs(source.flux_weighted_image, source_copy.flux_weighted_image)
        self.assertIs(source.metadata, source_copy.metadata)

    def test_deepcopy(self):
        source = Source(self.components)
        source_deepcopy = source.copy(deep=True)

        self.assertIsNot(source, source_deepcopy)
        self.assertEqual(source.n_components, source_deepcopy.n_components)
        for comp, comp_deepcopy in zip(source.components, source_deepcopy.components):
            self.assertIsNot(comp, comp_deepcopy)
            self.assertFactorizedComponentEqual(comp, comp_deepcopy)
            comp_deepcopy._spectrum.x += 1
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(comp._spectrum.x, comp_deepcopy._spectrum.x)
            comp_deepcopy._morph.x += 1
            with self.assertRaises(AssertionError):
                np.testing.assert_array_equal(comp._morph.x, comp_deepcopy._morph.x)

    def test_slice(self):
        source = Source(self.components)
        source_sliced = source["g":"r"]
        self.assertTupleEqual(source_sliced.bands, ("g", "r"))
        self.assertEqual(source.n_components, source_sliced.n_components)

        for comp, comp_sliced in zip(source.components, source_sliced.components):
            self.assertFactorizedComponentEqual(comp["g":"r"], comp_sliced)

    def test_reorder(self):
        source = Source(self.components)
        indices = ("i", "g", "r")
        source_reordered = source[indices]
        self.assertTupleEqual(source_reordered.bands, indices)
        self.assertEqual(source.n_components, source_reordered.n_components)
        for comp, comp_reordered in zip(source.components, source_reordered.components):
            self.assertFactorizedComponentEqual(comp[indices], comp_reordered)

        source_reordered = source["igr"]
        self.assertTupleEqual(source_reordered.bands, indices)
        self.assertEqual(source.n_components, source_reordered.n_components)
        for comp, comp_reordered in zip(source.components, source_reordered.components):
            self.assertFactorizedComponentEqual(comp["igr"], comp_reordered)

    def test_subset(self):
        source = Source(self.components)
        source_subset = source[("r",)]
        self.assertTupleEqual(source_subset.bands, ("r",))
        self.assertEqual(source.n_components, source_subset.n_components)
        for comp, comp_subset in zip(source.components, source_subset.components):
            self.assertFactorizedComponentEqual(comp["r"], comp_subset)

        source = source.copy(deep=True)
        for comp in source.components:
            comp._bands = ("ab", "cd", "ef")
        source_subset = source["ab"]
        self.assertTupleEqual(source_subset.bands, ("ab",))
        self.assertEqual(source.n_components, source_subset.n_components)
        for comp, comp_subset in zip(source.components, source_subset.components):
            self.assertFactorizedComponentEqual(comp["ab"], comp_subset)

    def test_indexing_errors(self):
        source = Source(self.components)

        with self.assertRaises(IndexError):
            source["x"]

        with self.assertRaises(IndexError):
            source["r":"x"]

        with self.assertRaises(IndexError):
            source["x":"i"]

        with self.assertRaises(IndexError):
            source["g", "x", "i"]

        with self.assertRaises(IndexError):
            source[Box((0, 0), (10, 10))]

        with self.assertRaises(IndexError):
            source[:, 10:20, 10:20]

        with self.assertRaises(IndexError):
            source[1:]

        with self.assertRaises(IndexError):
            source[1]

        with self.assertRaises(IndexError):
            source[0, 1]
