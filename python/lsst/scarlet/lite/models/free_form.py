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
from __future__ import annotations

__all__ = ["FactorizedFreeFormComponent"]

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from ..bbox import Box
from ..component import Component, FactorizedComponent
from ..detect import footprints_to_image
from ..detect_pybind11 import get_connected_multipeak, get_footprints  # type: ignore
from ..image import Image
from ..parameters import Parameter, parameter

if TYPE_CHECKING:
    from ..io.component import ScarletComponentBaseData


class FactorizedFreeFormComponent(FactorizedComponent):
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
            footprint = get_connected_multipeak(morph, self.peaks, 0)
            morph = morph * footprint

        if self.min_area > 0:
            footprints = get_footprints(morph, 4.0, self.min_area, 0, 0, False)
            bbox = self.bbox.copy()
            bbox.origin = (0, 0)
            footprint_image = footprints_to_image(footprints, bbox)
            morph = morph * (footprint_image > 0).data

        if np.all(morph == 0):
            morph[0, 0] = self.floor

        return morph

    def resize(self, model_box: Box) -> bool:
        return False

    def __str__(self):
        return (
            f"FactorizedFreeFormComponent(\n    bands={self.bands}\n    "
            f"spectrum={self.spectrum})\n    center={self.peak}\n    "
            f"morph_shape={self.morph.shape}"
        )

    def __repr__(self):
        return self.__str__()


class FreeFormComponent(Component):
    """Implements a component with no spectral or monotonicty constraints

    This is a FreeFormComponent that is not factorized into a
    spectrum and morphology with no monotonicity constraint.
    """

    def __init__(
        self,
        bands: tuple,
        model: np.ndarray | Parameter,
        model_bbox: Box,
        bg_thresh: float | None = None,
        bg_rms: np.ndarray | None = None,
        floor: float = 1e-20,
        peaks: list[tuple[int, int]] | None = None,
        min_area: float = 0,
    ):
        super().__init__(bands=bands, bbox=model_bbox)
        self._model = parameter(model)
        self.bg_rms = bg_rms
        self.bg_thresh = bg_thresh
        self.floor = floor
        self.peaks = peaks
        self.min_area = min_area

    @property
    def model(self) -> np.ndarray:
        return self._model.x

    def get_model(self) -> Image:
        return Image(self.model, bands=self.bands, yx0=cast(tuple[int, int], self.bbox.origin))

    @property
    def shape(self) -> tuple:
        return self.model.shape

    def grad_model(self, input_grad: np.ndarray, model: np.ndarray) -> np.ndarray:
        return input_grad

    def prox_model(self, model: np.ndarray) -> np.ndarray:
        if self.bg_thresh is not None and isinstance(self.bg_rms, np.ndarray):
            bg_thresh = self.bg_rms * self.bg_thresh
            # Enforce background thresholding
            model[model < bg_thresh[:, None, None]] = 0
        else:
            # enforce positivity
            model[model < 0] = 0

        if self.peaks is not None:
            # Remove pixels not connected to one of the peaks
            model2d = np.sum(model, axis=0)
            footprint = get_connected_multipeak(model2d, self.peaks, 0)
            model = model * footprint[None, :, :]

        if self.min_area > 0:
            # Remove regions with fewer than min_area connected pixels
            model2d = np.sum(model, axis=0)
            footprints = get_footprints(model2d, 4.0, self.min_area, 0, 0, False)
            bbox = self.bbox.copy()
            bbox.origin = (0, 0)
            footprint_image = footprints_to_image(footprints, bbox)
            model = model * (footprint_image > 0).data[None, :, :]

        if np.all(model == 0):
            # If the model is all zeros, set a single pixel to the floor
            model[0, 0] = self.floor

        return model

    def resize(self, model_box: Box) -> bool:
        return False

    def update(self, it: int, grad_log_likelihood: np.ndarray):
        self._model.update(it, grad_log_likelihood)

    def parameterize(self, parameterization: Callable) -> None:
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization: Callable
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        # Update the spectrum and morph in place
        parameterization(self)
        # update the parameters
        self._model.grad = self.grad_model
        self._model.prox = self.prox_model

    def __str__(self):
        result = f"FreeFormComponent<bands={self.bands}, shape={self.shape}>"
        return result

    def __repr__(self):
        return self.__str__()

    def to_data(self) -> ScarletComponentBaseData:
        raise NotImplementedError("Serialization not implemented for FreeFormComponent")

    def __getitem__(self, indices: Any) -> FreeFormComponent:
        """Get a sub-component corresponding to the given indices.

        Parameters
        ----------
        indices: Any
            The indices to use to slice the component model.

        Returns
        -------
        component: FreeFormComponent
            A new component that is a sub-component of this one.

        Raises
        ------
        IndexError :
            If the index includes a ``Box`` or spatial indices.
        """
        if indices in self.bands:
            bands = (indices,)
        else:
            bands = tuple(indices)

        return FreeFormComponent(
            bands=bands,
            model=self.model[indices],
            model_bbox=self.bbox,
            bg_thresh=self.bg_thresh,
            bg_rms=self.bg_rms,
            floor=self.floor,
            peaks=self.peaks,
            min_area=self.min_area,
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> FreeFormComponent:
        """Create a deep copy of this component.

        Parameters
        ----------
        memo: dict[int, Any]
            A dictionary to keep track of already copied objects.

        Returns
        -------
        component : FreeFormComponent
            A new component that is a deep copy of this one.
        """
        if id(self) in memo:
            return memo[id(self)]

        component = FreeFormComponent.__new__(FreeFormComponent)
        memo[id(self)] = component

        component.__init__(  # type: ignore[misc]
            bands=deepcopy(self.bands),
            model=deepcopy(self.model),
            model_bbox=deepcopy(self.bbox),
            bg_thresh=self.bg_thresh,
            bg_rms=deepcopy(self.bg_rms),
            floor=self.floor,
            peaks=deepcopy(self.peaks),
            min_area=self.min_area,
        )
        return component

    def __copy__(self) -> FreeFormComponent:
        """Create a copy of this component.

        Returns
        -------
        component : FreeFormComponent
            A new component that is a copy of this one.
        """
        return FreeFormComponent(
            bands=self.bands,
            model=self.model,
            model_bbox=self.bbox,
            bg_thresh=self.bg_thresh,
            bg_rms=self.bg_rms,
            floor=self.floor,
            peaks=self.peaks,
            min_area=self.min_area,
        )
