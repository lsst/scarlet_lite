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

__all__ = ["FreeFormComponent"]

from typing import cast

import numpy as np

from ..bbox import Box
from ..component import FactorizedComponent
from ..detect import footprints_to_image
from ..parameters import Parameter


class FreeFormComponent(FactorizedComponent):
    """Implements a free-form component

    With no constraints this component is typically either a garbage collector,
    or part of a set of components to deconvolve an image by separating out
    the different spectral components.

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

    def __init__(
        self,
        bands: tuple,
        spectrum: np.ndarray | Parameter,
        morph: np.ndarray | Parameter,
        model_bbox: Box,
        bg_thresh: float | None = None,
        bg_rms: np.ndarray | None = None,
        floor: float = 1e-20,
        peaks: list[tuple[int, int]] | None = None,
        min_area: float = 0,
    ):
        super().__init__(
            bands=bands,
            spectrum=spectrum,
            morph=morph,
            bbox=model_bbox,
            peak=None,
            bg_rms=bg_rms,
            bg_thresh=bg_thresh,
            floor=floor,
        )

        self.peaks = peaks
        self.min_area = min_area

    def prox_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the spectrum

        This differs from `FactorizedComponent` because an
        `SedComponent` has the spectrum normalized to unity.
        """
        # prevent divergent spectrum
        spectrum[spectrum < self.floor] = self.floor
        # Normalize the spectrum
        spectrum = spectrum / np.sum(spectrum)
        return spectrum

    def prox_morph(self, morph: np.ndarray) -> np.ndarray:
        """Apply a prox-like update to the morphology

        This is the main difference between an `SedComponent` and a
        `FactorizedComponent`, since this component has fewer constraints.
        """
        from lsst.scarlet.lite.detect_pybind11 import get_connected_multipeak, get_footprints  # type: ignore

        if self.bg_thresh is not None and isinstance(self.bg_rms, np.ndarray):
            bg_thresh = self.bg_rms * self.bg_thresh
            # Enforce background thresholding
            model = self.spectrum[:, None, None] * morph[None, :, :]
            morph[np.all(model < bg_thresh[:, None, None], axis=0)] = 0
        else:
            # enforce positivity
            morph[morph < 0] = 0

        if self.peaks is not None:
            morph = morph * get_connected_multipeak(morph > 0, self.peaks, 0)

        if self.min_area > 0:
            footprints = get_footprints(morph > 0, 4.0, self.min_area, 0, False)
            footprint_image = footprints_to_image(footprints, cast(tuple[int, int], morph.shape))
            morph = morph * (footprint_image > 0).data

        if np.all(morph == 0):
            morph[0, 0] = self.floor

        return morph

    def resize(self, model_box: Box) -> bool:
        return False

    def __str__(self):
        return (
            f"FreeFormComponent(\n    bands={self.bands}\n    "
            f"spectrum={self.spectrum})\n    center={self.peak}\n    "
            f"morph_shape={self.morph.shape}"
        )

    def __repr__(self):
        return self.__str__()
