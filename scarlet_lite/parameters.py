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


TParameter = TypeVar("TParameter", bound="Parameter")
TFistaParameter = TypeVar("TFistaParameter", bound="FistaParameter")
TAdaproxParameter = TypeVar("TAdaproxParameter", bound="AdaproxParameter")


# The default factor used for adaprox parameter steps
DEFAULT_ADAPROX_FACTOR = 1e-2


def grow_array(x: np.ndarray, new_shape: Sequence[int], dist: int) -> np.ndarray:
    """grow an array and pad it with zeros

    This is faster than `numpy.pad` by a factor of ~20.

    Parameters
    ----------
     x: np.ndarray
        The array to grow
    new_shape: Sequence[int]
        This is the new shape of the array.
        It would be trivial to calculate in this function,
        however in most cases this is already calculated for
        other purposes, so it might as well be reused.
    dist: int
        The amount to pad each side of the input array
        (so the new shape is extended by 2*dist on each axis).

    Returns
    -------
    result: np.ndarray
        The larger array that contains `x`.
    """
    result = np.zeros(new_shape, dtype=x.dtype)
    result[dist:-dist, dist:-dist] = x
    return result


class Parameter(np.ndarray):
    """A parameter in a `Component`"""

    is_parameter = True

    def __new__(cls, array: np.ndarray, name: str | None = None) -> TParameter:
        obj = np.asarray(array, dtype=array.dtype).view(cls)
        obj.name = name
        return obj

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
        pass

    def grow(self, new_shape: Sequence[int], dist: int):
        """Grow the parameter and all of the meta parameters

        Parameters
        ----------
        new_shape: Sequence[int]
            The new shape of the parameter.
        dist: int
            The amount to extend the array in each direction
        """
        pass

    def shrink(self, dist: int):
        """Shrink the parameter and all of the meta parameters

        Parameters
        ----------
        dist: int
            The amount to shrink the array in each direction
        """
        pass

    def __array_finalize__(self, obj):
        self.name = getattr(obj, "name", None)


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
    if hasattr(x, "is_parameter") and x.is_parameter:
        return x
    return Parameter(x)


class FistaParameter(Parameter):
    """A `LiteParameter` that updates itself using the Beck-Teboulle 2009
    FISTA proximal gradient method.

    See https://www.ceremade.dauphine.fr/~carlier/FISTA
    """

    def __init__(
        self,
        array: np.ndarray,
        step: float,
        grad: Callable = None,
        prox: Callable = None,
        t0: float = 1,
        z0: np.ndarray = None,
    ):
        super().__init__(array.shape)
        if z0 is None:
            z0 = array
        self.step = step
        self.grad = grad
        self.prox = prox
        self.z = z0
        self.t = t0

    def __new__(
        cls,
        array: np.ndarray,
        step: float,
        grad: Callable = None,
        prox: Callable = None,
        t0: float = 1,
        z0: np.ndarray = None,
        name: str | None = None,
    ) -> TFistaParameter:
        """Initialize the parameter

        Parameters
        ----------
        array: np.ndarray
            The initial guess for the parameter.
        step: float
            The step size for the parameter.
            This is scaled in each step by the first argument to
            `update` after the `input_grad`.
        grad: Callable
            The function to use to calculate the gradient.
            `grad` should accept the `input_grad` and a list
            of arguments.
        prox: Callable
            The function that acts as a proximal operator.
            This function should take `x` as an input, however
            the input `x` might not be the same as the input
            parameter, but a meta parameter instead.
        t0: float
            The initial value of the acceleration parameter.
        z0: np.ndarray
            The initial value of the meta parameter `z`.
            If this is `None` then `z` is initialized to the
            initial `x`.
        name: str | None
            The name of this parameter.
        """

        return super().__new__(
            cls,
            array,
            name,
        )

    def __array_finalize__(self, obj):
        """Required by sublclasses of ndarray to handle different ways
        of creating ndarrays

        See
        https://numpy.org/doc/stable/user/basics.subclassing.html#the-role-of-array-finalize  # noqa: W505
        for more on why this method is necessary
        """
        if obj is None:
            return

        self.name = getattr(obj, "name", None)
        self.step = getattr(obj, "step", np.nan)
        self.grad = getattr(obj, "grad", None)
        self.prox = getattr(obj, "prox", None)
        self.t = getattr(obj, "t", 1)
        self.z = getattr(obj, "z", obj.view(np.ndarray))

    def update(self, it: int, input_grad: np.ndarray, *args):
        """Update the parameter and meta-parameters using the PGM

        See `Parameter` for more.
        """
        step = self.step / np.sum(args[0] * args[0])
        _x = self.view(np.ndarray)

        y = self.z - step * self.grad(input_grad, _x, *args)
        x = self.prox(y, step)
        t = 0.5 * (1 + np.sqrt(1 + 4 * self.t**2))
        omega = 1 + (self.t - 1) / t
        self.z = _x + omega * (x - _x)
        _x[:] = x
        self.t = t

    def grow(self, new_shape: Sequence[int], dist: int):
        _x = self.view(np.ndarray)
        _x[:] = grow_array(_x, new_shape, dist)
        self.z = grow_array(self.z, new_shape, dist)
        # self.t = 1

    def shrink(self, dist: int):
        _x = self.view(np.ndarray)
        _x[:] = _x[dist:-dist, dist:-dist]
        self.z = self.z[dist:-dist, dist:-dist]
        # self.t = 1


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
    and PGM sub-iterations to satisfy feasibility and optimality.
    See details of the algorithms in the respective papers.
    """

    def __init__(
        self,
        array: np.ndarray,
        step: Callable,
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
        max_prox_iter: int = 1,
        prox_e_rel: float = 1e-6,
    ):
        super().__init__(array.shape)

        if not hasattr(b1, "__getitem__"):
            b1 = SingleItemArray(b1)

        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.p = p

        if not hasattr(step, "__call__"):
            # noinspection PyUnusedLocal
            def _step(x):
                return step

            self.step = _step
        else:
            self.step = step
        self.grad = grad
        self.prox = prox
        if m0 is None:
            m0 = np.zeros(self.shape, dtype=self.dtype)
        self.m = m0
        if v0 is None:
            v0 = np.zeros(self.shape, dtype=self.dtype)
        self.v = v0
        if vhat0 is None:
            vhat0 = np.ones(self.shape, dtype=self.dtype) * -np.inf
        self.vhat = vhat0
        self.phi_psi = phi_psi[scheme]
        self.max_prox_iter = max_prox_iter
        self.e_rel = prox_e_rel

    def __array_finalize__(self, obj):
        """Required by sublclasses of ndarray to handle different ways
        of creating ndarrays

        See
        https://numpy.org/doc/stable/user/basics.subclassing.html#the-role-of-array-finalize  # noqa: W505
        for more on why this method is necessary
        """
        if obj is None:
            return

        self.name = getattr(obj, "name", None)
        self.step = getattr(obj.step, np.nan)
        self.grad = getattr(obj, "grad", None)
        self.prox = getattr(obj, "prox", None)
        self.b1 = getattr(obj, "b1", 0.9)
        self.b2 = getattr(obj, "b2", 0.999)
        self.eps = getattr(obj, "eps", 1e-8)
        self.p = getattr(obj, "p", 0.25)
        self.m0 = getattr(obj, "m0", None)
        self.v0 = getattr(obj, "v0", None)
        self.vhat = getattr(obj, "vhat", None)
        self.scheme = getattr(obj, "scheme", "amsgrad")
        self.max_prox_iter = getattr(obj, "max_prox_iter", 1)
        self.e_rel = getattr(obj, "e_rel", 1e-6)

    def __new__(
        cls,
        array: np.ndarray,
        step: Callable,
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
        max_prox_iter: int = 1,
        prox_e_rel: float = 1e-6,
        name: str | None = None,
    ) -> TAdaproxParameter:
        """Create a new instance of a FistaParameter

        See `__init__` for a description of the parameters.
        """
        return super().__new__(
            cls,
            array,
            name,
        )

    def update(self, it: int, input_grad: np.ndarray, *args):
        """Update the parameter and meta-parameters using the PGM

        See `~LiteParameter` for more.
        """
        _x = self.view(np.ndarray)
        # Calculate the gradient
        grad = self.grad(input_grad, _x, *args)
        # Get the update for the parameter
        phi, psi = self.phi_psi(
            it, grad, self.m, self.v, self.vhat, self.b1, self.b2, self.eps, self.p
        )
        # Calculate the step size
        step = self.step(_x, it)
        if it > 0:
            _x[:] = _x - step * phi / psi
        else:
            _x[:] = _x - step * phi / psi / 10

        # Iterate over the proximal operators until convergence
        if self.prox is not None:
            z = _x.copy()
            gamma = step / np.max(psi)
            for tau in range(1, self.max_prox_iter + 1):
                _z = self.prox(z - gamma / step * psi * (z - _x), gamma)
                converged = ((_z - z) ** 2).sum() <= self.e_rel**2 * (z**2).sum()
                z = _z

                if converged:
                    break

            _x[:] = z

    def grow(self, new_shape: Sequence[int], dist: int):
        _x = self.view(np.ndarray)
        _x[:] = grow_array(_x, new_shape, dist)
        self.m = grow_array(self.m, new_shape, dist)
        self.v = grow_array(self.v, new_shape, dist)
        self.vhat = grow_array(self.vhat, new_shape, dist)

    def shrink(self, dist: int):
        _x = self.view(np.ndarray)
        _x[:] = _x[dist:-dist, dist:-dist]
        self.m = self.m[dist:-dist, dist:-dist]
        self.v = self.v[dist:-dist, dist:-dist]
        self.vhat = self.vhat[dist:-dist, dist:-dist]


class FixedParameter(Parameter):
    """A parameter that is not updated"""

    def __init__(self, array: np.ndarray):
        super().__init__(array.shape)

    def update(self, it: int, input_grad: np.ndarray, *args):
        pass

    def grow(self, new_shape: Sequence[int], dist: int):
        pass

    def shrink(self, dist: int):
        pass


def relative_step(
    x: np.ndarray,
    factor: float = 0.1,
    minimum: float = 0,
    axis: int | Sequence[int] = None,
):
    """Step size set at `factor` times the mean of `X` in direction `axis`"""
    return np.maximum(minimum, factor * x.mean(axis=axis))
