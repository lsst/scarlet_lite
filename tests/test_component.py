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

from abc import ABC
from typing import Any, Callable

import numpy as np
from lsst.scarlet.lite import Box, Image, Parameter
from lsst.scarlet.lite.component import (
    Component,
    CubeComponent,
    FactorizedComponent,
    default_adaprox_parameterization,
    default_fista_parameterization,
)
from lsst.scarlet.lite.operators import Monotonicity
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal, assert_array_equal
from utils import ScarletTestCase


class DummyComponent(Component):
    def resize(self) -> bool:
        pass

    def update(self, it: int, input_grad: np.ndarray):
        pass

    def get_model(self) -> Image:
        pass

    def parameterize(self, parameterization: Callable) -> None:
        parameterization(self)

    def to_data(self) -> DummyComponent:
        pass

    def __getitem__(self, indices: Any) -> DummyComponent:
        pass

    def __copy__(self) -> DummyComponent:
        pass

    def __deepcopy__(self, memo: dict[int, Any]) -> DummyComponent:
        pass


class _ComponentTestBase(ABC):
    def test_slice(self):
        component = self.component
        component_sliced = component["g":"r"]
        self.assertTupleEqual(component_sliced.bands, ("g", "r"))
        np.testing.assert_array_equal(component_sliced.get_model(), component.get_model().data[0:2])

    def test_reorder(self):
        component = self.component
        indices = ("i", "g", "r")
        component_reordered = component["i", "g", "r"]
        self.assertTupleEqual(component_reordered.bands, indices)
        np.testing.assert_array_equal(
            component_reordered.get_model(),
            component.get_model().data[(2, 0, 1),],
        )

        component_reordered = component["igr"]
        self.assertTupleEqual(component_reordered.bands, indices)
        np.testing.assert_array_equal(
            component_reordered.get_model(),
            component.get_model().data[(2, 0, 1),],
        )

    def test_subset(self):
        component = self.component
        indices = ("r",)
        component_subset = component["r"]
        self.assertTupleEqual(component_subset.bands, indices)
        np.testing.assert_array_equal(
            component_subset.get_model(),
            component.get_model().data[1:2,],
        )

        component = self.component.copy(deep=True)
        component._bands = ("ab", "cd", "ef")
        indices = "ab"
        component_reordered = component["ab"]
        self.assertTupleEqual(component_reordered.bands, (indices,))
        np.testing.assert_array_equal(
            component_reordered.get_model(),
            component.get_model().data[0:1,],
        )

    def test_indexing_errors(self):
        component = self.component
        print("bands", component.bands)
        with self.assertRaises(IndexError):
            component["z"]

        with self.assertRaises(IndexError):
            component["r":"z"]

        with self.assertRaises(IndexError):
            component["z":"i"]

        with self.assertRaises(IndexError):
            component["g", "z", "i"]

        with self.assertRaises(IndexError):
            component[Box((0, 0), (10, 10))]

        with self.assertRaises(IndexError):
            component[:, 10:20, 10:20]

        with self.assertRaises(IndexError):
            component[1:]

        with self.assertRaises(IndexError):
            component[1]

        with self.assertRaises(IndexError):
            component[0, 1]


class TestFactorizedComponent(_ComponentTestBase, ScarletTestCase):
    def setUp(self) -> None:
        spectrum = np.arange(3).astype(np.float32)
        morph = np.arange(20).reshape(4, 5).astype(np.float32)
        bands = ("g", "r", "i")
        bbox = Box((4, 5), (22, 31))
        self.model_box = Box((100, 100))
        center = (24, 33)

        self.component = FactorizedComponent(
            bands,
            spectrum,
            morph,
            bbox,
            center,
        )

        self.bands = bands
        self.spectrum = spectrum
        self.morph = morph
        self.full_shape = (3, 100, 100)

    def test_constructor(self):
        # Test with only required parameters
        component = FactorizedComponent(
            self.bands,
            self.spectrum,
            self.morph,
            self.component.bbox,
        )

        self.assertIsInstance(component._spectrum, Parameter)
        assert_array_equal(component.spectrum, self.spectrum)
        self.assertIsInstance(component._morph, Parameter)
        assert_array_equal(component.morph, self.morph)
        self.assertBoxEqual(component.bbox, self.component.bbox)
        self.assertIsNone(component.peak)
        self.assertIsNone(component.bg_rms)
        self.assertEqual(component.bg_thresh, 0.25)
        self.assertEqual(component.floor, 1e-20)
        self.assertTupleEqual(component.shape, (3, 4, 5))

        # Test that parameters are passed through
        center = self.component.peak
        bg_rms = np.arange(5) / 10
        bg_thresh = 0.9
        floor = 1e-10

        component = FactorizedComponent(
            self.bands,
            self.spectrum,
            self.morph,
            self.component.bbox,
            center,
            bg_rms,
            bg_thresh,
            floor,
        )

        self.assertTupleEqual(component.peak, center)
        assert_array_equal(component.bg_rms, bg_rms)  # type: ignore
        self.assertEqual(component.bg_thresh, bg_thresh)
        self.assertEqual(component.floor, floor)
        self.assertEqual(component.get_model().dtype, np.float32)

    def test_get_model(self):
        component = self.component
        assert_array_equal(component.get_model(), self.spectrum[:, None, None] * self.morph[None, :, :])

        # Insert component into a larger model
        full_model = np.zeros(self.full_shape)
        full_model[:, 22:26, 31:36] = self.spectrum[:, None, None] * self.morph[None, :, :]

        test_model = Image(np.zeros(self.full_shape), bands=self.bands)
        test_model += component.get_model()

        assert_array_equal(test_model.data, full_model)

    def test_gradients(self):
        component = self.component
        morph = self.morph
        spectrum = self.spectrum

        input_grad = np.array([morph, 2 * morph, 3 * morph])
        true_spectrum_grad = np.array(
            [
                np.sum(morph**2),
                np.sum(2 * morph**2),
                np.sum(3 * morph**2),
            ]
        )
        assert_almost_equal(component.grad_spectrum(input_grad, spectrum, morph), true_spectrum_grad)

        true_morph_grad = np.sum(input_grad * spectrum[:, None, None], axis=0)
        assert_almost_equal(component.grad_morph(input_grad, morph, spectrum), true_morph_grad)

    def test_proximal_operators(self):
        # Test spectrum positivity, morph threshold, and monotonicity
        spectrum = np.array([-1, 2, 3], dtype=float)
        morph = np.array([[10, 2, 1], [1, 5, 3], [0.1, 4, -1]], dtype=float)
        bbox = Box((3, 3), (10, 10))
        morph_bbox = Box((100, 100))
        center = (11, 11)
        monotonicity = Monotonicity((101, 101), fit_radius=0)

        component = FactorizedComponent(
            self.bands,
            spectrum.copy(),
            morph.copy(),
            bbox,
            center,
            bg_rms=np.array([1, 1, 1]),
            bg_thresh=0.5,
            monotonicity=monotonicity,
        )

        proxed_spectrum = np.array([1e-20, 2, 3])
        proxed_morph = np.array([[2.6666666666666667, 2, 1], [1, 5, 3], [0, 4, 0]])
        proxed_morph = proxed_morph / 5

        component.prox_spectrum(component.spectrum)
        component.prox_morph(component.morph)

        assert_array_equal(component.spectrum, proxed_spectrum)
        assert_array_equal(component.morph, proxed_morph)

        component = FactorizedComponent(
            self.bands,
            spectrum.copy(),
            morph.copy(),
            bbox,
            None,
        )

        proxed_spectrum = np.array([1e-20, 2, 3])
        proxed_morph = np.array([[10, 2, 1], [1, 5, 3], [0.1, 4, 0]])
        proxed_morph = proxed_morph / 10

        component.prox_spectrum(component.spectrum)
        component.prox_morph(component.morph)

        assert_array_equal(component.spectrum, proxed_spectrum)
        assert_array_equal(component.morph, proxed_morph)

        self.assertFalse(component.resize(morph_bbox))

    def test_resize(self):
        spectrum = np.array([1, 2, 3], dtype=float)
        morph = np.zeros((10, 10), dtype=float)
        morph[3:6, 5:8] = np.arange(9).reshape(3, 3)
        bbox = Box((10, 10), (3, 5))

        morph_bbox = Box((100, 100))
        monotonicity = Monotonicity((101, 101), fit_radius=0)

        component = FactorizedComponent(
            self.bands,
            spectrum.copy(),
            morph.copy(),
            bbox,
            None,
            bg_rms=np.array([1, 1, 1]),
            bg_thresh=0.5,
            monotonicity=monotonicity,
            padding=1,
        )

        self.assertTupleEqual(component.morph.shape, (10, 10))
        self.assertIsNone(component.component_center)

        component.resize(morph_bbox)
        self.assertTupleEqual(component.morph.shape, (5, 5))
        self.assertTupleEqual(component.bbox.origin, (5, 9))
        self.assertTupleEqual(component.bbox.shape, (5, 5))
        self.assertIsNone(component.component_center)

    def test_parameterization(self):
        component = self.component
        assert_array_equal(component.get_model(), self.spectrum[:, None, None] * self.morph[None, :, :])

        component.parameterize(default_fista_parameterization)
        helpers = set(component._morph.helpers.keys())
        self.assertSetEqual(helpers, {"z"})
        component.parameterize(default_adaprox_parameterization)
        helpers = set(component._morph.helpers.keys())
        self.assertSetEqual(helpers, {"m", "v", "vhat"})

        params = (tuple("grizy"), Box((5, 5)))
        with self.assertRaises(NotImplementedError):
            default_fista_parameterization(DummyComponent(*params))

        with self.assertRaises(NotImplementedError):
            default_adaprox_parameterization(DummyComponent(*params))

    def test_shallow_copy(self):
        component = self.component
        component.monotonicity = Monotonicity((11, 11), fit_radius=0)

        component_copy = component.copy()

        self.assertIsNot(component, component_copy)
        np.testing.assert_array_equal(component._spectrum.x, component_copy._spectrum.x)
        np.testing.assert_array_equal(component._morph.x, component_copy._morph.x)
        self.assertIs(component.bbox, component_copy.bbox)
        self.assertIs(component.peak, component_copy.peak)
        self.assertIs(component.bg_thresh, component_copy.bg_thresh)
        self.assertIs(component.monotonicity, component_copy.monotonicity)

    def test_deep_copy(self):
        component = self.component
        component.monotonicity = Monotonicity((11, 11), fit_radius=0)
        component_deepcopy = component.copy(deep=True)

        self.assertIsNot(component, component_deepcopy)

        np.testing.assert_array_equal(component._spectrum.x, component_deepcopy._spectrum.x)
        component_deepcopy._spectrum.x += 1
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(component._spectrum.x, component_deepcopy._spectrum.x)

        np.testing.assert_array_equal(component._morph.x, component_deepcopy._morph.x)
        component_deepcopy._morph.x += 1
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(component._morph.x, component_deepcopy._morph.x)

        self.assertIsNot(component.bbox, component_deepcopy.bbox)
        self.assertBoxEqual(component.bbox, component_deepcopy.bbox)

        self.assertTupleEqual(component.peak, component_deepcopy.peak)
        self.assertEqual(component.bg_thresh, component_deepcopy.bg_thresh)
        self.assertIsNot(component.monotonicity, component_deepcopy.monotonicity)


class TestCubeComponent(_ComponentTestBase, ScarletTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.bands = tuple("gri")
        peak = (27, 32)
        bbox = Box((15, 15), (20, 25))
        morph = integrated_circular_gaussian(sigma=0.8).astype(np.float32)
        spectrum = np.arange(3, dtype=np.float32)
        model = morph[None, :, :] * spectrum[:, None, None]
        model_image = Image(model, yx0=bbox.origin, bands=self.bands)
        self.component = CubeComponent(model=model_image, peak=peak)

    def test_constructor(self):
        component = self.component
        self.assertIsInstance(component._model, Image)
        np.testing.assert_array_equal(component._model.data, self.component._model.data)
        self.assertTupleEqual(component.bands, self.bands)
        self.assertBoxEqual(component.bbox, Box((15, 15), (20, 25)))
        self.assertTupleEqual(component.peak, (27, 32))

    def test_shallow_copy(self):
        component = self.component
        component_copy = component.copy()

        self.assertIsNot(component_copy, component)
        self.assertTupleEqual(component_copy.peak, component.peak)
        self.assertImageEqual(component_copy._model, component._model)

    def test_deep_copy(self):
        component = self.component
        component_copy = component.copy(deep=True)

        self.assertIsNot(component, component_copy)

        self.assertTupleEqual(component_copy.peak, component.peak)
        self.assertImageEqual(component_copy._model, component._model)
        with self.assertRaises(AssertionError):
            component_copy._model._data -= 1
            self.assertImageEqual(component_copy._model, component._model)
