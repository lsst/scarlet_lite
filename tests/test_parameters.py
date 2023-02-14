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
from lsst.scarlet.lite import Box
from lsst.scarlet.lite.parameters import (
    AdaproxParameter,
    FistaParameter,
    FixedParameter,
    Parameter,
    parameter,
    phi_psi,
)
from numpy.testing import assert_array_equal
from utils import ScarletTestCase


def prox_ceiling(x, thresh: float = 20):
    """Test prox for testing parameters"""
    x[x > thresh] = thresh
    return x


def grad(input_grad: np.ndarray, x: np.ndarray, *args):
    """Test gradient for testing parameters"""
    return 2 * x * input_grad


class TestParameters(ScarletTestCase):
    def test_parameter_class(self):
        x = np.arange(15, dtype=float).reshape(3, 5)
        param = parameter(x)
        self.assertIsInstance(param, Parameter)
        assert_array_equal(param.x, x)
        self.assertTupleEqual(param.shape, (3, 5))
        self.assertEqual(param.dtype, float)

        with self.assertRaises(NotImplementedError):
            param.update(1, np.zeros((3, 5)))

        # Test copy method
        y = np.zeros((3, 5), dtype=float)
        y[1, 3] = 1
        param = Parameter(x, {"y": y}, 0)
        self.assertIsNot(param.copy().x, x)
        assert_array_equal(param.copy().x, x)
        self.assertIsNot(param.copy().helpers["y"], y)
        assert_array_equal(param.copy().helpers["y"], y)

        param2 = parameter(param)
        self.assertIs(param2, param)

    def test_growing(self):
        x = np.arange(15, dtype=float).reshape(3, 5)
        y = np.zeros((3, 5), dtype=float)
        y[1, 3] = 1
        param = Parameter(x, {"y": y}, 0)

        # Test growing in all dimensions
        old_box = Box((3, 5), (21, 15))
        new_box = Box((11, 20), (19, 10))
        param.resize(old_box, new_box)
        truth = np.zeros((11, 20), dtype=float)
        truth[2:5, 5:10] = x
        assert_array_equal(param.x, truth)

        # Test shrinking in all directions
        param = Parameter(x, {"y": y}, 0)
        old_box = Box((3, 5), (21, 15))
        new_box = Box((1, 3), (22, 16))
        param.resize(old_box, new_box)
        truth = x[1:2, 1:4]
        assert_array_equal(param.x, truth)

    def test_fista_parameter(self):
        x = np.arange(10, dtype=float)
        x2 = x**2
        param = FistaParameter(
            x2,
            0.1,
            grad,
            prox_ceiling,
        )

        assert_array_equal(param.x, x2)
        assert_array_equal(param.grad(np.full(x.shape, 0.1), x), 0.2 * x)
        truth = x2.copy()
        truth[truth > 20] = 20
        assert_array_equal(param.prox(x2), truth)
        param.update(10, x, x2)

    def test_adprox_parameter(self):
        x = np.arange(10, dtype=float)
        x2 = x**2
        param = AdaproxParameter(
            x2,
            0.1,
            grad,
            prox_ceiling,
        )

        assert_array_equal(param.x, x2)
        assert_array_equal(param.grad(np.full(x.shape, 0.1), x), 0.2 * x)
        truth = x2.copy()
        truth[truth > 20] = 20
        assert_array_equal(param.prox(x2), truth)
        param.update(10, x, x2)

        schemes = tuple(phi_psi.keys())
        for scheme in schemes:
            param = AdaproxParameter(
                x2,
                0.1,
                grad,
                prox_ceiling,
                scheme=scheme,
            )
            param.update(10, x, x2)

    def test_fixed_parameter(self):
        x = np.arange(10, dtype=float)
        param = FixedParameter(x)
        param.update(10, np.arange(10) * 2)
        assert_array_equal(param.x, x)
