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

from typing import Sequence

import numpy as np

from scarlet_lite.utils import integrated_circular_gaussian


def get_psfs(sigmas: float | Sequence[float]) -> np.ndarray:
    try:
        iter(sigmas)
    except TypeError:
        sigmas = (sigmas,)
    psf = [integrated_circular_gaussian(sigma=sigma) for sigma in sigmas]
    return np.array(psf)
