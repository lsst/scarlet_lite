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

__all__ = ["Blend"]

from typing import Callable, Sequence, TypeVar, cast

import numpy as np

from .bbox import Box
from .component import Component, FactorizedComponent
from .image import Image
from .observation import Observation
from .source import Source

TBlend = TypeVar("TBlend", bound="Blend")


class Blend:
    """A single blend.

    This is effectively a combination of the `Blend`, `Observation`,
    and `Renderer` classes in pmelchior/scarlet,
    greatly simplified due to the assumptions that the
    observations are all resampled onto the same pixel grid and that the
    `images` contain all of the information for all of the model bands.

    This is still agnostic to the component type, so new custom classes
    are allowed as long as they posses the `get_model`, `update`, and
    `resize` methods, but all components should be contained in sources.
    The only underlying assumption is that all of the components inserted
    into the model by addition. If the components require a more
    complicated insertion, for example multiplication of a dust lane,
    then a new blend class will need to be created.
    """

    def __init__(self, sources: Sequence[Source], observation: Observation):
        """Initialize the class.

        Parameters
        ----------
        sources:
            The sources to fit.
        observation:
            The observation that contains the images,
            PSF, etc. that are being fit.
        """
        self.sources = list(sources)
        self.observation = observation

        # Initialzie the iteration count and loss function
        self.it = 0
        self.loss = []
        for component in self.components:
            component.overlap = component.bbox & self.bbox

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the model for the entire `Blend`."""
        return self.observation.shape

    @property
    def bbox(self) -> Box:
        """The bounding box of the entire blend."""
        return self.observation.bbox

    @property
    def components(self) -> list[Component]:
        """The list of all components in the blend.

        Since the list of sources might change,
        this is always built on the fly.
        """
        return [c for src in self.sources for c in src.components]

    def get_model(self, convolve: bool = False, use_flux: bool = False) -> Image:
        """Generate a model of the entire blend.

        Parameters
        ----------
        convolve:
            Whether to convolve the model with the observed PSF in each band.
        use_flux:
            Whether to use the re-distributed flux associated with the source
            instead of the component models.

        Returns
        -------
        model:
            The model created by combining all of the source models.
        """
        model = Image(
            np.zeros(self.shape, dtype=self.observation.images.dtype),
            bands=self.observation.bands,
            yx0=self.observation.bbox.origin[-2:],
        )

        if use_flux:
            for src in self.sources:
                src.flux.insert_into(model)
        else:
            for component in self.components:
                component.get_model().insert_into(model)
            if convolve:
                return self.observation.convolve(model)
        return model

    def _grad_log_likelihood(self) -> Image:
        """Gradient of the likelihood wrt the unconvolved model"""
        model = self.get_model(convolve=True)
        # Update the loss
        self.loss.append(self.observation.log_likelihood(model))
        # Calculate the gradient wrt the model d(logL)/d(model)
        result = self.observation.weights * (model - self.observation.images)
        result = self.observation.convolve(result, grad=True)
        return result

    @property
    def log_likelihood(self) -> float:
        """The current log-likelihood"""
        if len(self.loss) == 0:
            return self.observation.log_likelihood(self.get_model(convolve=True))
        return self.loss[-1]

    def fit_spectra(self, clip: bool = False) -> TBlend:
        """Fit all of the spectra given their current morphologies

        Parameters
        ----------
        clip:
            Whether or not to clip components that were not
            assigned any flux during the fit.

        Returns
        -------
        blend:
            The blend with updated components is returned.
        """
        from .initialization import multifit_spectra

        morphs = []
        spectra = []
        factorized_indices = []
        model = Image.from_box(
            self.observation.bbox,
            bands=self.observation.bands,
            dtype=self.observation.dtype,
        )
        for idx, component in enumerate(self.components):
            if hasattr(component, "morph") and hasattr(component, "spectrum"):
                morphs.append(component.morph)
                spectra.append(component.spectrum)
                factorized_indices.append(idx)
            else:
                if not hasattr(component, "overlap"):
                    component.overlap = component.bbox & self.bbox
                model[component.overlap] += component.get_model()[component.overlap]
        model = self.observation.convolve(model, mode="real")

        boxes = [c.bbox for c in self.components]
        fit_spectra = multifit_spectra(
            self.observation,
            [Image(morph, yx0=bbox.origin) for morph, bbox in zip(morphs, boxes)],
            model,
        )
        for idx in range(len(morphs)):
            component = cast(
                FactorizedComponent, self.components[factorized_indices[idx]]
            )
            component.spectrum[:] = fit_spectra[idx]
            component.spectrum[component.spectrum < 0] = 0

        # Run the proxes for all of the components to make sure that the
        # spectra are consistent with the constraints.
        # In practice this usually means making sure that they are
        # non-negative.
        for src in self.sources:
            for component in src.components:
                if (
                    hasattr(component, "spectrum")
                    and hasattr(component, "prox_spectrum")
                    and component.prox_spectrum is not None  # type: ignore
                ):
                    component.prox_spectrum(component.spectrum)

        if clip:
            # Remove components with no positive flux
            for src in self.sources:
                _components = []
                for component in src.components:
                    component_model = component.get_model()
                    component_model.data[component_model.data < 0] = 0
                    if np.sum(component_model.data) > 0:
                        _components.append(component)
                src.components = _components

        return self

    def fit(
        self,
        max_iter: int,
        e_rel: float = 1e-4,
        min_iter: int = 15,
        resize: int = 10,
    ) -> tuple[int, float]:
        """Fit all of the parameters

        Parameters
        ----------
        max_iter:
            The maximum number of iterations
        e_rel:
            The relative error to use for determining convergence.
        min_iter:
            The minimum number of iterations.
        resize:
            Number of iterations before attempting to resize the
            resizable components. If `resize` is `None` then
            no resizing is ever attempted.

        Returns
        -------
        it:
            Number if iterations.
        loss:
            Loss for the last solution
        """
        while self.it < max_iter:
            # Calculate the gradient wrt the on-convolved model
            grad_log_likelihood = self._grad_log_likelihood()
            # Update each component given the current gradient
            for component in self.components:
                if not hasattr(component, "overlap"):
                    component.overlap = component.bbox & self.bbox
                component.update(self.it, grad_log_likelihood[component.overlap].data)
            # Check to see if any components need to be resized
            if resize is not None and self.it > 0 and self.it % resize == 0:
                for component in self.components:
                    if component.resize(self.bbox):
                        component.overlap = component.bbox & self.bbox
            # Stopping criteria
            if self.it > min_iter and np.abs(
                self.loss[-1] - self.loss[-2]
            ) < e_rel * np.abs(self.loss[-1]):
                break
            self.it += 1
        return self.it, self.loss[-1]

    def parameterize(self, parameterization: Callable):
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization:
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        for source in self.sources:
            source.parameterize(parameterization)
