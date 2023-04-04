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

from typing import Callable

import numpy as np
from lsst.scarlet.lite import Box, Image, Parameter
from lsst.scarlet.lite.component import (
    Component,
    FactorizedComponent,
    default_adaprox_parameterization,
    default_fista_parameterization,
)
from lsst.scarlet.lite.operators import Monotonicity
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


class TestFactorizedComponent(ScarletTestCase):
    def setUp(self) -> None:
        spectrum = np.arange(3)
        morph = np.arange(20).reshape(4, 5)
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
