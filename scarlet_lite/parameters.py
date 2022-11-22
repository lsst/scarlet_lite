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
    "LiteParameter",
    "FistaParameter",
    "AdaproxParameter",
    "FixedParameter",
    "relative_step",
]

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np


# The default factor used for adaprox parameter steps
DEFAULT_FACTOR = 1e-2


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


class LiteParameter(ABC):
    """A parameter in a `LiteComponent`

    Unlike the main scarlet `Parameter` class,
    a `LiteParameter` also contains methods to update the parameter,
    using any given optimizer, provided the abstract methods
    are all implemented. The main parameter should always be
    stored as `LiteParameter.x`, but the names of the meta parameters
    can be different.
    """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def shrink(self, dist: int):
        """Shrink the parameter and all of the meta parameters

        Parameters
        ----------
        dist: int
            The amount to shrink the array in each direction
        """
        pass


class FistaParameter(LiteParameter):
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
        """Initialize the parameter

        Parameters
        ----------
        x: np.ndarray
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
        """
        if z0 is None:
            z0 = x
        self.x = x
        self.step = step
        self.grad = grad
        self.prox = prox
        self.z = z0
        self.t = t0

    def update(self, it: int, input_grad: np.ndarray, *args):
        """Update the parameter and meta-parameters using the PGM

        See `~LiteParameter` for more.
        """
        step = self.step / np.sum(args[0] * args[0])

        y = self.z - step * self.grad(input_grad, self.x, *args)
        x = self.prox(y, step)
        t = 0.5 * (1 + np.sqrt(1 + 4 * self.t**2))
        omega = 1 + (self.t - 1) / t
        self.z = self.x + omega * (x - self.x)
        self.x = x
        self.t = t

    def grow(self, new_shape: Sequence[int], dist: int):
        self.x = grow_array(self.x, new_shape, dist)
        self.z = grow_array(self.z, new_shape, dist)
        # self.t = 1

    def shrink(self, dist: int):
        self.x = self.x[dist:-dist, dist:-dist]
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


class AdaproxParameter(LiteParameter):
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
        x: np.ndarray,
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
        """Initialize the parameter

         NOTE:
        Setting `m`, `v`, `vhat` allows to continue a previous run,
        e.g. for a warm start of a slightly changed problem.
        If not set, they will be initialized with 0.

        Parameter
        ---------
        x: np.ndarray
            The initial guess for the parameter.
        step: Callable
            The step size for the parameter that takes the
            parameter `x` and the iteration `it` as arguments.
        grad: Callable
            The function to use to calculate the gradient.
            `grad` should accept the `input_grad` and a list
            of arguments.
        prox: Callable
            The function that acts as a proximal operator.
            This function should take `x` as an input, however
            the input `x` might not be the same as the input
            parameter, but a meta parameter instead.
        b1: float
            The strength parameter for the weighted gradient
            (`m`) update.
        b2: float
            The strength for the weighted gradient squared
            (`v`) update.
        eps: float
            Minimum value of the cumulative gradient squared
            (`vhat`) meta paremeter.
        p: float
            Meta parameter used by some of the ADAM schemes
        m0: np.ndarray
            Initial value of the weighted gradient (`m`) parameter
            for a warm start.
        v0: np.ndarray
            Initial value of the weighted gradient squared(`v`) parameter
            for a warm start.
        vhat0: np.ndarray
            Initial value of the
            cumulative weighted gradient squared (`vhat`) parameter
            for a warm start.
        scheme: str
            Name of the ADAM scheme to use.
            One of ["adam", "nadam", "adamx", "amsgrad", "padam", "radam"]
        """
        self.x = x

        if not hasattr(b1, "__getitem__"):
            b1 = SingleItemArray(b1)

        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.p = p

        if not hasattr(step, "__call__"):
            # noinspection PyUnusedLocal
            def _step(dummy_x, it):
                return step

            self.step = _step
        else:
            self.step = step
        self.grad = grad
        self.prox = prox
        if m0 is None:
            m0 = np.zeros(x.shape, dtype=x.dtype)
        self.m = m0
        if v0 is None:
            v0 = np.zeros(x.shape, dtype=x.dtype)
        self.v = v0
        if vhat0 is None:
            vhat0 = np.ones(x.shape, dtype=x.dtype) * -np.inf
        self.vhat = vhat0
        self.phi_psi = phi_psi[scheme]
        self.max_prox_iter = max_prox_iter
        self.e_rel = prox_e_rel

    def update(self, it: int, input_grad: np.ndarray, *args):
        """Update the parameter and meta-parameters using the PGM

        See `~LiteParameter` for more.
        """
        # Calculate the gradient
        grad = self.grad(input_grad, self.x, *args)
        # Get the update for the parameter
        phi, psi = self.phi_psi(
            it, grad, self.m, self.v, self.vhat, self.b1, self.b2, self.eps, self.p
        )
        # Calculate the step size
        step = self.step(self.x, it)
        if it > 0:
            self.x -= step * phi / psi
        else:
            self.x -= step * phi / psi / 10

        # Iterate over the proximal operators until convergence
        if self.prox is not None:
            z = self.x.copy()
            gamma = step / np.max(psi)
            for tau in range(1, self.max_prox_iter + 1):
                _z = self.prox(z - gamma / step * psi * (z - self.x), gamma)
                converged = ((_z - z) ** 2).sum() <= self.e_rel**2 * (z**2).sum()
                z = _z

                if converged:
                    break

            self.x = z

    def grow(self, new_shape: Sequence[int], dist: int):
        self.x = grow_array(self.x, new_shape, dist)
        self.m = grow_array(self.m, new_shape, dist)
        self.v = grow_array(self.v, new_shape, dist)
        self.vhat = grow_array(self.vhat, new_shape, dist)

    def shrink(self, dist: int):
        self.x = self.x[dist:-dist, dist:-dist]
        self.m = self.m[dist:-dist, dist:-dist]
        self.v = self.v[dist:-dist, dist:-dist]
        self.vhat = self.vhat[dist:-dist, dist:-dist]


class FixedParameter(LiteParameter):
    """A parameter that is not updated"""

    def __init__(self, x):
        self.x = x

    def update(self, it: int, input_grad: np.ndarray, *args):
        pass

    def grow(self, new_shape: Sequence[int], dist: int):
        pass

    def shrink(self, dist: int):
        pass


# noinspection PyUnusedLocal
def relative_step(
    x: np.ndarray,
    it: int,
    factor: float = 0.1,
    minimum: float = 0,
    axis=None | int | Sequence[int],
):
    """Step size set at `factor` times the mean of `X` in direction `axis`"""
    return np.maximum(minimum, factor * x.mean(axis=axis))
