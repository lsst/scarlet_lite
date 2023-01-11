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

__all__ = ["SedComponent"]

import numpy as np

from ..bbox import Box
from . import FactorizedComponent
from ..parameters import Parameter

from ..detect import scarlet_footprints_to_image


class SedComponent(FactorizedComponent):
    """Implements a free-form component

    With no constraints this component is typically either a garbage collector,
    or part of a set of components to deconvolve an image by separating out
    the different spectral components.
    """

    def __init__(
        self,
        bands: tuple,
        sed: np.ndarray | Parameter,
        morph: np.ndarray | Parameter,
        model_bbox: Box,
        bg_thresh: float = None,
        bg_rms: np.ndarray = None,
        floor: float = 1e-20,
        peaks: list[tuple[int, int]] = None,
        min_area: float = 0,
    ):
        """Initialize the component.

        See `FactorizedComponent` for a list of parameters not shown here.

        Parameters
        ----------
        peaks: `list` of `tuple`
            A set of ``(cy, cx)`` peaks for detected sources.
            If peak is not ``None`` then only pixels in the same "footprint"
            as one of the peaks are included in the morphology.
            If `peaks` is ``None`` then there is no constraint applied.
        min_area: float
            The minimum area for a peak.
            If `min_area` is not `None` then all regions of the morphology
            with fewer than `min_area` connected pixels are removed.
        """
        super().__init__(
            bands=bands,
            sed=sed,
            morph=morph,
            bbox=model_bbox,
            model_bbox=model_bbox,
            center=None,
            bg_rms=bg_rms,
            bg_thresh=bg_thresh,
            floor=floor,
        )

        self.peaks = peaks
        self.min_area = min_area

    def prox_sed(self, sed: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the SED

        This differs from `FactorizedComponent` because an
        `SedComponent` has the SED normalized to unity.
        """
        # prevent divergent SED
        sed[sed < self.floor] = self.floor
        # Normalize the SED
        sed = sed / np.sum(sed)
        return sed

    def prox_morph(self, morph: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the morphology

        This is the main difference between an `SedComponent` and a
        `FactorizedComponent`, since this component has fewer constraints.
        """
        from scarlet_lite.detect_pybind11 import get_connected_multipeak, get_footprints

        if self.bg_thresh is not None:
            bg_thresh = self.bg_rms * self.bg_thresh
            # Enforce background thresholding
            model = self.sed[:, None, None] * morph[None, :, :]
            morph[np.all(model < bg_thresh[:, None, None], axis=0)] = 0
        else:
            # enforce positivity
            morph[morph < 0] = 0

        if self.peaks is not None:
            morph = morph * get_connected_multipeak(morph > 0, self.peaks, 0)

        if self.min_area > 0:
            footprints = get_footprints(morph > 0, 4.0, self.min_area, 0, False)
            morph = morph * (scarlet_footprints_to_image(footprints, morph.shape) > 0)

        if np.all(morph == 0):
            morph[0, 0] = self.floor

        return morph

    def resize(self) -> bool:
        return False

    def __str__(self):
        return "SedComponent"

    def __repr__(self):
        return "SedComponent"
