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

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from scarlet_lite import Box, Image, Parameter
from scarlet_lite.component import FactorizedComponent
from scarlet_lite.operators import Monotonicity
from utils import ScarletTestCase


class TestFactorizedComponent(ScarletTestCase):
    def setUp(self) -> None:
        sed = np.arange(3)
        morph = np.arange(20).reshape(4, 5)
        bands = ("g", "r", "i")
        bbox = Box((4, 5), (22, 31))
        model_bbox = Box((100, 100))
        center = (24, 33)

        self.component = FactorizedComponent(
            bands,
            sed,
            morph,
            bbox,
            model_bbox,
            center,
        )

        self.bands = bands
        self.sed = sed
        self.morph = morph
        self.full_shape = (3, 100, 100)

    def test_constructor(self):
        # Test with only required parameters
        component = FactorizedComponent(
            self.bands,
            self.sed,
            self.morph,
            self.component.bbox,
            self.component.model_bbox,
        )

        self.assertIsInstance(component._sed, Parameter)
        assert_array_equal(component.sed, self.sed)
        self.assertIsInstance(component._morph, Parameter)
        assert_array_equal(component.morph, self.morph)
        self.assertBoxEqual(component.bbox, self.component.bbox)
        self.assertBoxEqual(component.model_bbox, self.component.model_bbox)
        self.assertIsNone(component.center)
        self.assertIsNone(component.bg_rms)
        self.assertEqual(component.bg_thresh, 0.25)
        self.assertEqual(component.floor, 1e-20)
        self.assertTupleEqual(component.shape, (3, 4, 5))
        self.assertEqual(str(component), "FactorizedComponent")
        self.assertEqual(repr(component), "FactorizedComponent")

        # Test that parameters are passed through
        center = self.component.center
        bg_rms = np.arange(5) / 10
        bg_thresh = 0.9
        floor = 1e-10

        component = FactorizedComponent(
            self.bands,
            self.sed,
            self.morph,
            self.component.bbox,
            self.component.model_bbox,
            center,
            bg_rms,
            bg_thresh,
            floor,
        )

        self.assertTupleEqual(component.center, center)
        assert_array_equal(component.bg_rms, bg_rms)  # type: ignore
        self.assertEqual(component.bg_thresh, bg_thresh)
        self.assertEqual(component.floor, floor)

    def test_get_model(self):
        component = self.component
        assert_array_equal(
            component.get_model(), self.sed[:, None, None] * self.morph[None, :, :]
        )

        # Insert component into a larger model
        full_model = np.zeros(self.full_shape)
        full_model[:, 22:26, 31:36] = self.sed[:, None, None] * self.morph[None, :, :]

        test_model = Image(np.zeros(self.full_shape), bands=self.bands)
        test_model += component.get_model()

        assert_array_equal(test_model.data, full_model)

    def test_gradients(self):
        component = self.component
        morph = self.morph
        sed = self.sed

        input_grad = np.zeros(self.full_shape)
        input_grad[:, 22:26, 31:36] = np.array([morph, 2 * morph, 3 * morph])
        true_sed_grad = np.array(
            [
                np.sum(morph**2),
                np.sum(2 * morph**2),
                np.sum(3 * morph**2),
            ]
        )
        assert_almost_equal(component.grad_sed(input_grad, sed, morph), true_sed_grad)

        true_morph_grad = np.sum(input_grad * sed[:, None, None], axis=0)[22:26, 31:36]
        assert_almost_equal(
            component.grad_morph(input_grad, morph, sed), true_morph_grad
        )

    def test_proximal_operators(self):
        # Test SED positivity, morph threshold, and monotonicity
        sed = np.array([-1, 2, 3], dtype=float)
        morph = np.array([[10, 2, 1], [1, 5, 3], [0.1, 4, -1]], dtype=float)
        bbox = Box((3, 3), (10, 10))
        morph_bbox = Box((100, 100))
        center = (11, 11)
        monotonicity = Monotonicity((101, 101), fit_radius=0)

        component = FactorizedComponent(
            self.bands,
            sed.copy(),
            morph.copy(),
            bbox,
            morph_bbox,
            center,
            bg_rms=np.array([1, 1, 1]),
            bg_thresh=0.5,
            monotonicity=monotonicity,
        )

        proxed_sed = np.array([1e-20, 2, 3])
        proxed_morph = np.array([[2.6666666666666667, 2, 1], [1, 5, 3], [0, 4, 0]])
        proxed_morph = proxed_morph / 5

        component.prox_sed(component.sed)
        component.prox_morph(component.morph)

        assert_array_equal(component.sed, proxed_sed)
        assert_array_equal(component.morph, proxed_morph)

        component = FactorizedComponent(
            self.bands,
            sed.copy(),
            morph.copy(),
            bbox,
            morph_bbox,
            None,
        )

        proxed_sed = np.array([1e-20, 2, 3])
        proxed_morph = np.array([[10, 2, 1], [1, 5, 3], [0.1, 4, 0]])
        proxed_morph = proxed_morph / 5

        component.prox_sed(component.sed)
        component.prox_morph(component.morph)

        assert_array_equal(component.sed, proxed_sed)
        assert_array_equal(component.morph, proxed_morph)

        self.assertFalse(component.resize())

    def test_resize(self):
        sed = np.array([1, 2, 3], dtype=float)
        morph = np.zeros((10, 10), dtype=float)
        morph[3:6, 5:8] = np.arange(9).reshape(3, 3)
        bbox = Box((10, 10), (3, 5))

        morph_bbox = Box((100, 100))
        monotonicity = Monotonicity((101, 101), fit_radius=0)

        component = FactorizedComponent(
            self.bands,
            sed.copy(),
            morph.copy(),
            bbox,
            morph_bbox,
            None,
            bg_rms=np.array([1, 1, 1]),
            bg_thresh=0.5,
            monotonicity=monotonicity,
            padding=1,
        )

        self.assertTupleEqual(component.morph.shape, (10, 10))
        self.assertIsNone(component.component_center)

        component.resize()
        self.assertTupleEqual(component.morph.shape, (5, 5))
        self.assertTupleEqual(component.bbox.origin, (5, 9))
        self.assertTupleEqual(component.bbox.shape, (5, 5))
        self.assertIsNone(component.component_center)
