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

__all__ = [
    "parameter",
    "Parameter",
    "FistaParameter",
    "AdaproxParameter",
    "FixedParameter",
    "relative_step",
    "DEFAULT_ADAPROX_FACTOR",
]

from typing import Callable, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from .bbox import Box

TParameter = TypeVar("TParameter", bound="Parameter")
TFistaParameter = TypeVar("TFistaParameter", bound="FistaParameter")
TAdaproxParameter = TypeVar("TAdaproxParameter", bound="AdaproxParameter")


# The default factor used for adaprox parameter steps
DEFAULT_ADAPROX_FACTOR = 1e-2


class Parameter:
    """A parameter in a `Component`"""

    def __init__(self, x, helpers: dict[str, np.ndarray]):
        self.x = x
        self.helpers = helpers

    @property
    def shape(self) -> tuple[int, ...]:
        return self.x.shape

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.x.dtype

    def copy(self) -> TParameter:
        helpers = {k: v.copy() for k, v in self.helpers.items()}
        return Parameter(self.x.copy(), helpers)

    def update(self, it: int, input_grad: np.ndarray, *args):
        """Update the parameter in one iteration.

        This includes the gradient update, proximal update,
        and any meta parameters that are stored as class
        attributes to update the parameter.

        Parameters
        ----------
        it: int
            The current iteration
        input_grad: np.ndarray
            The gradient from the full model, passed to the parameter.
        """
        raise NotImplementedError("Base Parameters cannot be updated")

    def grow(self, old_box: Box, new_box: Box):
        """Grow the parameter and all of the meta parameters

        Parameters
        ----------
        old_box:
            The old bounding box for the parameter.
        new_box:
            The new bounding box for the parameter.
        """
        slices = new_box.overlapped_slices(old_box)
        x = np.zeros(new_box.shape, dtype=self.dtype)
        x[slices[0]] = self.x[slices[1]]
        self.x = x

        for name, value in self.helpers.items():
            result = np.zeros(new_box.shape, dtype=self.dtype)
            result[slices[0]] = value[slices[1]]
            self.helpers[name] = result


def parameter(x: np.ndarray | Parameter) -> Parameter:
    """Convert a `np.ndarray` into a `Parameter`.

    Parameters
    ----------
    x: np.ndarray | Parameter
        The array or parameter to convert into a `Parameter`.

    Returns
    -------
    result: Parameter
        `x`, converted into a `Parameter` if necessary.
    """
    if isinstance(x, Parameter):
        return x
    return Parameter(x, {})


class FistaParameter(Parameter):
    """A `LiteParameter` that updates itself using the Beck-Teboulle 2009
    FISTA proximal gradient method.

    See https://www.ceremade.dauphine.fr/~carlier/FISTA
    """

    def __init__(
        self,
        x: np.ndarray,
        step: float,
        grad: Callable = None,
        prox: Callable = None,
        t0: float = 1,
        z0: np.ndarray = None,
    ):
        if z0 is None:
            z0 = x

        super().__init__(
            x,
            {"z": z0},
        )

        self.step = step
        self.grad = grad
        self.prox = prox
        self.t = t0

    def update(self, it: int, input_grad: np.ndarray, *args):
        """Update the parameter and meta-parameters using the PGM

        See `Parameter` for more.
        """
        step = self.step / np.sum(args[0] * args[0])
        _x = self.x
        _z = self.helpers["z"]

        y = _z - step * self.grad(input_grad, _x, *args)
        x = self.prox(y)
        t = 0.5 * (1 + np.sqrt(1 + 4 * self.t**2))
        omega = 1 + (self.t - 1) / t
        self.helpers["z"] = _x + omega * (x - _x)
        _x[:] = x
        self.t = t


# noinspection PyUnusedLocal
def _adam_phi_psi(it, g, m, v, vhat, b1, b2, eps, p):
    # moving averages
    m[:] = (1 - b1[it]) * g + b1[it] * m
    v[:] = (1 - b2) * (g**2) + b2 * v

    # bias correction
    t = it + 1
    phi = m / (1 - b1[it] ** t)
    psi = np.sqrt(v / (1 - b2**t)) + eps
    return phi, psi


# noinspection PyUnusedLocal
def _nadam_phi_psi(it, g, m, v, vhat, b1, b2, eps, p):
    # moving averages
    m[:] = (1 - b1[it]) * g + b1[it] * m
    v[:] = (1 - b2) * (g**2) + b2 * v

    # bias correction
    t = it + 1
    phi = (b1[it] * m[:] + (1 - b1[it]) * g) / (1 - b1[it] ** t)
    psi = np.sqrt(v / (1 - b2**t)) + eps
    return phi, psi


# noinspection PyUnusedLocal
def _amsgrad_phi_psi(it, g, m, v, vhat, b1, b2, eps, p):
    # moving averages
    m[:] = (1 - b1[it]) * g + b1[it] * m
    v[:] = (1 - b2) * (g**2) + b2 * v

    phi = m
    if vhat is None:
        vhat = v
    else:
        vhat[:] = np.maximum(vhat, v)
    # sanitize zero-gradient elements
    if eps > 0:
        vhat = np.maximum(vhat, eps)
    psi = np.sqrt(vhat)
    return phi, psi


def _padam_phi_psi(it, g, m, v, vhat, b1, b2, eps, p):
    # moving averages
    m[:] = (1 - b1[it]) * g + b1[it] * m
    v[:] = (1 - b2) * (g**2) + b2 * v

    phi = m
    if vhat is None:
        vhat = v
    else:
        vhat[:] = np.maximum(vhat, v)
    # sanitize zero-gradient elements
    if eps > 0:
        vhat = np.maximum(vhat, eps)
    psi = vhat**p
    return phi, psi


# noinspection PyUnusedLocal
def _adamx_phi_psi(it, g, m, v, vhat, b1, b2, eps, p):
    # moving averages
    m[:] = (1 - b1[it]) * g + b1[it] * m
    v[:] = (1 - b2) * (g**2) + b2 * v

    phi = m
    if vhat is None:
        vhat = v
    else:
        factor = (1 - b1[it]) ** 2 / (1 - b1[it - 1]) ** 2
        vhat[:] = np.maximum(factor * vhat, v)
    # sanitize zero-gradient elements
    if eps > 0:
        vhat = np.maximum(vhat, eps)
    psi = np.sqrt(vhat)
    return phi, psi


# noinspection PyUnusedLocal
def _radam_phi_psi(it, g, m, v, vhat, b1, b2, eps, p):
    rho_inf = 2 / (1 - b2) - 1

    # moving averages
    m[:] = (1 - b1[it]) * g + b1[it] * m
    v[:] = (1 - b2) * (g**2) + b2 * v

    # bias correction
    t = it + 1
    phi = m / (1 - b1[it] ** t)
    rho = rho_inf - 2 * t * b2**t / (1 - b2**t)

    if rho > 4:
        psi = np.sqrt(v / (1 - b2**t))
        r = np.sqrt(
            (rho - 4) * (rho - 2) * rho_inf / (rho_inf - 4) / (rho_inf - 2) / rho
        )
        psi /= r
    else:
        psi = np.ones(g.shape, g.dtype)
    # sanitize zero-gradient elements
    if eps > 0:
        psi = np.maximum(psi, np.sqrt(eps))
    return phi, psi


phi_psi = {
    "adam": _adam_phi_psi,
    "nadam": _nadam_phi_psi,
    "amsgrad": _amsgrad_phi_psi,
    "padam": _padam_phi_psi,
    "adamx": _adamx_phi_psi,
    "radam": _radam_phi_psi,
}


class SingleItemArray:
    """Mock an array with only a single item"""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, item):
        return self.value


class AdaproxParameter(Parameter):
    """Operator updated using te Proximal ADAM algorithm

    Uses multiple variants of adaptive quasi-Newton gradient descent
        * Adam (Kingma & Ba 2015)
        * NAdam (Dozat 2016)
        * AMSGrad (Reddi, Kale & Kumar 2018)
        * PAdam (Chen & Gu 2018)
        * AdamX (Phuong & Phong 2019)
        * RAdam (Liu et al. 2019)
    See details of the algorithms in the respective papers.
    """

    def __init__(
        self,
        x: np.ndarray,
        step: Callable | float,
        grad: Callable = None,
        prox: Callable = None,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        p: float = 0.25,
        m0: np.ndarray = None,
        v0: np.ndarray = None,
        vhat0: np.ndarray = None,
        scheme: str = "amsgrad",
        prox_e_rel: float = 1e-6,
    ):
        shape = x.shape
        dtype = x.dtype
        if m0 is None:
            m0 = np.zeros(shape, dtype=dtype)

        if v0 is None:
            v0 = np.zeros(shape, dtype=dtype)

        if vhat0 is None:
            vhat0 = np.ones(shape, dtype=dtype) * -np.inf

        super().__init__(
            x,
            {
                "m": m0,
                "v": v0,
                "vhat": vhat0,
            },
        )

        if not hasattr(b1, "__getitem__"):
            b1 = SingleItemArray(b1)

        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.p = p

        if not hasattr(step, "__call__"):

            def _step(x):
                return step

            self.step = _step
        else:
            self.step = step
        self.grad = grad
        self.prox = prox

        self.phi_psi = phi_psi[scheme]
        self.e_rel = prox_e_rel

    def update(self, it: int, input_grad: np.ndarray, *args):
        """Update the parameter and meta-parameters using the PGM

        See `~LiteParameter` for more.
        """
        _x = self.x
        # Calculate the gradient
        grad = self.grad(input_grad, _x, *args)
        # Get the update for the parameter
        phi, psi = self.phi_psi(
            it,
            grad,
            self.helpers["m"],
            self.helpers["v"],
            self.helpers["vhat"],
            self.b1,
            self.b2,
            self.eps,
            self.p,
        )
        # Calculate the step size
        step = self.step(_x)
        if it > 0:
            _x += -step * phi / psi
        else:
            # This is a scheme that Peter Melchior and I came up with to
            # dampen the known affect of ADAM, where the first iteration
            # is often much larger than desired.
            _x += -step * phi / psi / 10

        self.x = self.prox(_x)


class FixedParameter(Parameter):
    """A parameter that is not updated"""

    def __init__(self, x: np.ndarray):
        super().__init__(x, {})

    def update(self, it: int, input_grad: np.ndarray, *args):
        pass


def relative_step(
    x: np.ndarray,
    factor: float = 0.1,
    minimum: float = 0,
    axis: int | Sequence[int] = None,
):
    """Step size set at `factor` times the mean of `X` in direction `axis`"""
    return np.maximum(minimum, factor * x.mean(axis=axis))
