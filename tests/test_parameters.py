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

        # Audit finding O-1: ``update`` must work when ``prox`` is
        # None (the default), matching ``FistaParameter.update``.
        # Previously ``self.prox(_x)`` was called unconditionally and
        # raised ``TypeError`` on ``None``.
        param = AdaproxParameter(x2.copy(), 0.1, grad)
        param.update(10, x, x2)
        param.update(0, x, x2)

    def test_adaprox_variants_converge(self):
        """Each ADAM variant must drive a simple quadratic loss to
        its optimum.

        The loss is ``0.5 * sum((x - target)**2)`` with gradient
        ``x - target``. Every scheme should reach ``target`` within
        a small tolerance after a fixed iteration budget. This
        catches semantic regressions in any of the per-iteration
        update formulas (the kinds of bug in O-2).
        """
        target = np.array([3.0, -2.0, 5.0])

        def quad_grad(input_grad, x):
            return x - target

        for scheme in tuple(phi_psi.keys()):
            param = AdaproxParameter(
                np.zeros_like(target),
                step=0.1,
                grad=quad_grad,
                scheme=scheme,
            )
            for it in range(2000):
                param.update(it, np.zeros_like(target))
            np.testing.assert_allclose(
                param.x,
                target,
                atol=1e-3,
                err_msg=f"AdaproxParameter scheme={scheme!r} failed to converge",
            )

    def test_adamx_first_iteration(self):
        """``_adamx_phi_psi`` must treat ``factor`` as 1 on the first
        iteration rather than indexing ``b1[it-1] = b1[-1]``.

        Audit finding O-2: at ``it=0`` the formula
        ``(1 - b1[it])**2 / (1 - b1[it-1])**2`` accidentally reads
        the *last* element of a varying ``b1`` schedule. With the
        default ``SingleItemArray`` (constant ``b1``) this returns
        the right value by coincidence; with a real array of varying
        decay rates the factor is wrong on the very first step.
        """
        adamx = phi_psi["adamx"]
        # b1[-1] differs sharply from b1[0], so the buggy and fixed
        # branches diverge.
        b1 = np.array([0.9, 0.5])
        g = np.array([1.0])
        m = np.array([0.0])
        v = np.array([0.0])
        # Non-default ``vhat`` so the factor multiplies a finite
        # value (the default ``-inf`` would absorb any positive
        # factor).
        vhat = np.array([1.0])
        _, psi = adamx(0, g, m, v, vhat, b1, 0.999, 0, 0.5)
        # v after update: (1-0.999)*1 = 0.001
        # Fixed: vhat = max(1.0 * 1.0, 0.001) = 1.0; psi = sqrt(1.0) = 1.0
        # Buggy: factor = (0.1)**2/(0.5)**2 = 0.04
        #        vhat = max(0.04 * 1.0, 0.001) = 0.04; psi = sqrt(0.04) = 0.2
        np.testing.assert_allclose(psi, 1.0)

    def test_fixed_parameter(self):
        x = np.arange(10, dtype=float)
        param = FixedParameter(x)
        param.update(10, np.arange(10) * 2)
        assert_array_equal(param.x, x)

    def test_shallow_copy(self):
        x = np.arange(10, dtype=float)

        # FistaParameter
        param = FistaParameter(x, 0.1)
        param_copy = param.copy()
        self.assertIsInstance(param_copy, FistaParameter)

        assert_array_equal(param.x, param_copy.x)
        assert_array_equal(param.helpers["z"], param_copy.helpers["z"])

        # AdaproxParameter
        param = AdaproxParameter(x, 0.1)
        param_copy = param.copy()
        self.assertIsInstance(param_copy, AdaproxParameter)

        assert_array_equal(param.x, param_copy.x)
        assert_array_equal(param.helpers["m"], param_copy.helpers["m"])
        assert_array_equal(param.helpers["v"], param_copy.helpers["v"])
        assert_array_equal(param.helpers["vhat"], param_copy.helpers["vhat"])

        # FixedParameter
        param = FixedParameter(x)
        param_copy = param.copy()
        self.assertIsInstance(param_copy, FixedParameter)
        assert_array_equal(param.x, param_copy.x)

    def test_deep_copy(self):
        x = np.arange(10, dtype=float)

        # FistaParameter
        param = FistaParameter(x, 0.1)
        param_deepcopy = param.copy(deep=True)
        self.assertIsInstance(param_deepcopy, FistaParameter)

        assert_array_equal(param.x, param_deepcopy.x)
        param_deepcopy.x += 1
        with self.assertRaises(AssertionError):
            assert_array_equal(param.x, param_deepcopy.x)

        assert_array_equal(param.helpers["z"], param_deepcopy.helpers["z"])
        param_deepcopy.helpers["z"] += 1
        with self.assertRaises(AssertionError):
            assert_array_equal(param.helpers["z"], param_deepcopy.helpers["z"])

        # AdaproxParameter
        param = AdaproxParameter(x, 0.1)
        param_deepcopy = param.copy(deep=True)
        self.assertIsInstance(param_deepcopy, AdaproxParameter)

        assert_array_equal(param.x, param_deepcopy.x)
        param_deepcopy.x += 1
        with self.assertRaises(AssertionError):
            assert_array_equal(param.x, param_deepcopy.x)

        assert_array_equal(param.helpers["m"], param_deepcopy.helpers["m"])
        param_deepcopy.helpers["m"] = -1
        with self.assertRaises(AssertionError):
            assert_array_equal(param.helpers["m"], param_deepcopy.helpers["m"])

        assert_array_equal(param.helpers["v"], param_deepcopy.helpers["v"])
        param_deepcopy.helpers["v"] = -1
        with self.assertRaises(AssertionError):
            assert_array_equal(param.helpers["v"], param_deepcopy.helpers["v"])

        assert_array_equal(param.helpers["vhat"], param_deepcopy.helpers["vhat"])
        param_deepcopy.helpers["vhat"] = -1
        with self.assertRaises(AssertionError):
            assert_array_equal(param.helpers["vhat"], param_deepcopy.helpers["vhat"])

        # FixedParameter
        param = FixedParameter(x)
        param_deepcopy = param.copy(deep=True)
        self.assertIsInstance(param_deepcopy, FixedParameter)
        assert_array_equal(param.x, param_deepcopy.x)
        param_deepcopy.x += 1
        with self.assertRaises(AssertionError):
            assert_array_equal(param.x, param_deepcopy.x)
