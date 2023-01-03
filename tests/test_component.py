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

import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from scarlet_lite import Box, FactorizedComponent, Image, Parameter
from scarlet_lite.operators import Monotonicity


class TestFactorizedComponent(unittest.TestCase):
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

        assert isinstance(component._sed, Parameter)
        assert_array_equal(component.sed, self.sed)
        assert isinstance(component._morph, Parameter)
        assert_array_equal(component.morph, self.morph)
        assert component.bbox == self.component.bbox
        assert component.model_bbox == self.component.model_bbox
        assert component.center is None
        assert component.bg_rms is None
        assert component.bg_thresh == 0.25
        assert component.floor == 1e-20

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

        assert component.center == center
        assert_array_equal(component.bg_rms, bg_rms)  # type: ignore
        assert component.bg_thresh == bg_thresh
        assert component.floor == floor

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

        assert_array_equal(
            test_model.data, full_model
        )

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
