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

__all__ = ["default_fista_parameterization", "default_adaprox_parameterization"]

from functools import partial
from .base import Component
from .factorized import FactorizedComponent
from ..parameters import FistaParameter, AdaproxParameter, relative_step


def default_fista_parameterization(component: Component):
    """Initialize a factorized component to use FISTA PGM for optimization"""
    if isinstance(component, FactorizedComponent):
        component._spectrum = FistaParameter(component.spectrum, step=0.5)
        component._morph = FistaParameter(component.morph, step=0.5)
    else:
        raise NotImplementedError(f"Unrecognized component type {component}")


def default_adaprox_parameterization(component: Component, noise_rms: float = None):
    """Initialize a factorized component to use Proximal ADAM
    for optimization
    """
    if noise_rms is None:
        noise_rms = 1e-16
    if isinstance(component, FactorizedComponent):
        component._spectrum = AdaproxParameter(
            component.spectrum,
            step=partial(relative_step, factor=1e-2, minimum=noise_rms),
        )
        component._morph = AdaproxParameter(
            component.morph,
            step=1e-2,
        )
    else:
        raise NotImplementedError(f"Unrecognized component type {component}")
