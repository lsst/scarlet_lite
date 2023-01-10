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

__all__ = ["Blend", "FitPsfBlend"]

from typing import cast, Callable, Sequence, TypeVar

import numpy as np

from .bbox import Box
from .component import Component, FactorizedComponent
from .image import Image
from .measure import conserve_flux
from .observation import Observation, FitPsfObservation
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
        sources: Sequence[Source]
            The sources to fit.
        observation: Observation
            The observation that contains the images,
            PSF, etc. that are being fit.
        """
        self.sources = list(sources)
        self._components = []
        for source in sources:
            self._components.extend(source.components)
        self.observation = observation

        # Initialzie the iteration count and loss function
        self.it = 0
        self.loss = []

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.observation.shape

    @property
    def bbox(self) -> Box:
        """The bounding box of the entire blend"""
        return self.observation.bbox

    @property
    def components(self) -> list[Component]:
        return self._components

    def get_model(self, convolve: bool = False, use_flux: bool = False) -> Image:
        """Generate a model of the entire blend

        Parameters
        ----------
        convolve: bool
            Whether to convolve the model with the observed PSF in each band.
        use_flux: bool
            Whether to use the re-distributed flux associated with the source
            instead of the component models.
        """
        model = Image(
            np.zeros(self.shape, dtype=self.observation.images.dtype),
            bands=self.observation.bands,
            yx0=self.observation.bbox.origin[-2:],
        )

        if use_flux:
            for src in self.sources:
                src.flux.insert_into(model, save=True)
        else:
            for component in self.components:
                component.get_model().insert_into(model, save=True)
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
        clip: bool
            Whether or not to clip components that were not
            assigned any flux during the fit.

        Returns
        -------
        blend: Blend
            The blend with updated components is returned.
        """
        from .initialization import multifit_spectra

        morphs = []
        seds = []
        factorized_indices = []
        model = np.zeros(
            self.observation.images.shape, dtype=self.observation.images.dtype
        )
        for idx, component in enumerate(self.components):
            if hasattr(component, "morph") and hasattr(component, "sed"):
                morphs.append(component.morph)
                seds.append(component.sed)
                factorized_indices.append(idx)
            else:
                model[component.slices[0]] += component.get_model()[component.slices[1]]

        boxes = [c.bbox for c in self.components]
        fit_seds = multifit_spectra(
            self.observation,
            [Image(morph, yx0=bbox.origin) for morph, bbox in zip(morphs, boxes)],
        )
        for idx in range(len(morphs)):
            component = cast(
                FactorizedComponent, self.components[factorized_indices[idx]]
            )
            component.sed[:] = fit_seds[idx]
            component.sed[component.sed < 0] = 0

        if clip:
            components = []
            # Remove components with no sed or morphology
            for src in self.sources:
                _components = []
                for component in src.components:
                    if np.any(component.sed) > 0 and np.any(component.morph) > 0:
                        components.append(component)
                        _components.append(component)
                src.components = _components
            self._components = components
        else:
            for src in self.sources:
                for component in src.components:
                    if (
                        hasattr(component, "sed")
                        and hasattr(component, "prox_sed")
                        and component.prox_sed is not None  # type: ignore
                    ):
                        component.prox_sed(component.sed)

        return self

    def fit(
        self,
        max_iter: int,
        e_rel: float = 1e-4,
        min_iter: int = 15,
        resize: int = 10,
        do_conserve_flux: bool = True,
    ) -> tuple[int, float]:
        """Fit all of the parameters

        Parameters
        ----------
        max_iter: int
            The maximum number of iterations
        e_rel: float
            The relative error to use for determining convergence.
        min_iter: int
            The minimum number of iterations.
        resize: int
            Number of iterations before attempting to resize the
            resizable components. If `resize` is `None` then
            no resizing is ever attempted.
        do_conserve_flux: bool
            Whether or not to reweight the flux using the source
            models as templates.

        Returns
        -------
        it: int
            Number if iterations.
        loss: float
            Loss for the last solution
        """
        it = self.it
        while it < max_iter:
            # Calculate the gradient wrt the on-convolved model
            grad_log_likelihood = self._grad_log_likelihood()
            # Update each component given the current gradient
            for component in self.components:
                component.update(it, grad_log_likelihood.data)
            # Check to see if any components need to be resized
            if resize is not None and it > 0 and it % resize == 0:
                for component in self.components:
                    if hasattr(component, "resize"):
                        component.resize()
            # Stopping criteria
            if it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(
                self.loss[-1]
            ):
                break
            it += 1
        self.it = it
        if do_conserve_flux:
            conserve_flux(self)
        return it, self.loss[-1]

    def parameterize(self, parameterization: Callable):
        """Convert the component parameter arrays into Parameter instances

        Parameters
        ----------
        parameterization: Callable
            A function to use to convert parameters of a given type into
            a `Parameter` in place. It should take a single argument that
            is the `Component` or `Source` that is to be parameterized.
        """
        for component in self.components:
            component.parameterize(parameterization)


class FitPsfBlend(Blend):
    """A blend that attempts to fit the PSF along with the source models."""

    def _grad_log_likelihood(self) -> np.ndarray:
        """Gradient of the likelihood wrt the unconvolved model"""
        model = self.get_model(convolve=True)
        # Update the loss
        self.loss.append(
            0.5
            * -np.sum(self.observation.weights * (self.observation.images - model) ** 2)
        )
        # Calculate the gradient wrt the model d(logL)/d(model)
        result = self.observation.weights * (model - self.observation.images)
        return result

    def fit(
        self,
        max_iter: int,
        e_rel: float = 1e-4,
        min_iter: int = 1,
        resize: int = 10,
        do_conserve_flux: bool = False,
    ) -> tuple[int, float]:
        """Fit all of the parameters

        Parameters
        ----------
        max_iter: int
            The maximum number of iterations
        e_rel: float
            The relative error to use for determining convergence.
        min_iter: int
            The minimum number of iterations.
        resize: int
            Number of iterations before attempting to resize the
            resizable components. If `resize` is `None` then
            no resizing is ever attempted.
        do_conserve_flux: bool
            Whether or not to reweight the flux using the source
            models as templates.
        """
        it = self.it
        while it < max_iter:
            # Calculate the gradient wrt the on-convolved model
            grad_log_likelihood = self._grad_log_likelihood()
            _grad_log_likelihood = self.observation.convolve(
                grad_log_likelihood, grad=True
            )
            # Update each component given the current gradient
            for component in self.components:
                component.update(it, _grad_log_likelihood)
            # Check to see if any components need to be resized
            if resize is not None and it > 0 and it % resize == 0:
                for component in self.components:
                    if hasattr(component, "resize"):
                        component.resize()

            # Update the PSF
            cast(self.observation, FitPsfObservation).update(
                it, grad_log_likelihood, self.get_model()
            )
            # Stopping criteria
            if it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(
                self.loss[-1]
            ):
                break
            it += 1
        self.it = it
        if do_conserve_flux:
            conserve_flux(self)
        return it, self.loss[-1]
