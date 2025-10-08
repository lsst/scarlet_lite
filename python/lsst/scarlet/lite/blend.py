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

__all__ = ["Blend"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import numpy as np

from .bbox import Box
from .component import Component, FactorizedComponent
from .image import Image
from .observation import Observation
from .source import Source

if TYPE_CHECKING:
    from .io import ScarletBlendData, ScarletSourceBaseData


class BlendBase(ABC):
    """A base class for blends that can be extended to add additional
    functionality.

    This class holds all of the sources and observation that are to be fit,
    as well as performing fitting and joint initialization of the
    spectral components (when applicable).

    Parameters
    ----------
    sources:
        The sources to fit.
    observation:
        The observation that contains the images,
            PSF, etc. that are being fit.
    metadata:
        Additional metadata to store with the blend.
    """

    sources: list[Source]
    observation: Observation
    metadata: dict | None

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

    @abstractmethod
    def get_model(self, convolve: bool = False, use_flux: bool = False) -> Image:
        """Generate a model of the entire blend.

        Parameters
        ----------
        convolve:
            Whether to convolve the model with the observed PSF in each band.
        use_flux:
            Whether to use the re-distributed flux associated with the sources
            instead of the component models.

        Returns
        -------
        model:
            The model created by combining all of the source models.
        """

    @abstractmethod
    def to_blend_data(self) -> ScarletBlendData:
        """Convert the blend into a serializable dictionary format.

        Returns
        -------
        data:
            A dictionary containing all of the information needed to
            reconstruct the blend.
        """


class Blend(BlendBase):
    """A single blend.

    This class holds all of the sources and observation that are to be fit,
    as well as performing fitting and joint initialization of the
    spectral components (when applicable).

    Parameters
    ----------
    sources:
        The sources to fit.
    observation:
        The observation that contains the images,
            PSF, etc. that are being fit.
    metadata:
        Additional metadata to store with the blend.
    """

    def __init__(self, sources: Sequence[Source], observation: Observation, metadata: dict | None = None):
        self.sources = list(sources)
        self.observation = observation
        if metadata is not None and len(metadata) == 0:
            metadata = None
        self.metadata = metadata

        # Initialize the iteration count and loss function
        self.it = 0
        self.loss: list[float] = []

    def get_model(self, convolve: bool = False, use_flux: bool = False) -> Image:
        """Generate a model of the entire blend.

        Parameters
        ----------
        convolve:
            Whether to convolve the model with the observed PSF in each band.
        use_flux:
            Whether to use the re-distributed flux associated with the sources
            instead of the component models.

        Returns
        -------
        model:
            The model created by combining all of the source models.
        """
        model = Image(
            np.zeros(self.shape, dtype=self.observation.images.dtype),
            bands=self.observation.bands,
            yx0=cast(tuple[int, int], self.observation.bbox.origin[-2:]),
        )

        if use_flux:
            for src in self.sources:
                if src.flux_weighted_image is None:
                    raise ValueError(
                        "Some sources do not have 'flux' attribute set. Run measure.conserve_flux"
                    )
                src.flux_weighted_image.insert_into(model)
        else:
            for component in self.components:
                component.get_model().insert_into(model)
            if convolve:
                return self.observation.convolve(model)
        return model

    def _grad_log_likelihood(self) -> tuple[Image, np.ndarray]:
        """Gradient of the likelihood wrt the unconvolved model

        Returns
        -------
        result:
            The gradient of the likelihood wrt the model
        model_data:
           The convol model data used to calculate the gradient.
           This can be useful for debugging but is not used in
           production.
        """
        model = self.get_model(convolve=True)
        # Update the loss
        self.loss.append(self.observation.log_likelihood(model))
        # Calculate the gradient wrt the model d(logL)/d(model)
        result = self.observation.weights * (model - self.observation.images)
        result = self.observation.convolve(result, grad=True)
        return result, model.data

    @property
    def log_likelihood(self) -> float:
        """The current log-likelihood

        This is calculated on the fly to ensure that it is always up to date
        with the current model parameters.
        """
        return self.observation.log_likelihood(self.get_model(convolve=True))

    def fit_spectra(self, clip: bool = False) -> Blend:
        """Fit all of the spectra given their current morphologies with a
        linear least squares algorithm.

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
        components = self.components
        for idx, component in enumerate(components):
            if hasattr(component, "morph") and hasattr(component, "spectrum"):
                component = cast(FactorizedComponent, component)
                morphs.append(component.morph)
                spectra.append(component.spectrum)
                factorized_indices.append(idx)
            else:
                model.insert(component.get_model())
        model = self.observation.convolve(model, mode="real")

        boxes = [c.bbox for c in components]
        fit_spectra = multifit_spectra(
            self.observation,
            [Image(morph, yx0=cast(tuple[int, int], bbox.origin)) for morph, bbox in zip(morphs, boxes)],
            model,
        )
        for idx in range(len(morphs)):
            component = cast(FactorizedComponent, components[factorized_indices[idx]])
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
                    component.prox_spectrum(component.spectrum)  # type: ignore

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
            Number of iterations.
        loss:
            Loss for the last solution
        """
        while self.it < max_iter:
            # Calculate the gradient wrt the on-convolved model
            grad_log_likelihood = self._grad_log_likelihood()
            if resize is not None and self.it > 0 and self.it % resize == 0:
                do_resize = True
            else:
                do_resize = False
            # Update each component given the current gradient
            for component in self.components:
                overlap = component.bbox & self.bbox
                component.update(self.it, grad_log_likelihood[0][overlap].data)
                # Check to see if any components need to be resized
                if do_resize:
                    component.resize(self.bbox)
            # Stopping criteria
            self.it += 1
            if self.it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(self.loss[-1]):
                break
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

    def conserve_flux(self, mask_footprint: bool = True, weight_image: Image | None = None) -> None:
        """Use the source models as templates to re-distribute flux
        from the data

        The source models are used as approximations to the data,
        which redistribute the flux in the data according to the
        ratio of the models for each source.
        There is no return value for this function,
        instead it adds (or modifies) a ``flux_weighted_image``
        attribute to each the sources with the flux attributed to
        that source.

        Parameters
        ----------
        blend:
            The blend that is being fit
        mask_footprint:
            Whether or not to apply a mask for pixels with zero weight.
        weight_image:
            The weight image to use for the redistribution.
            If `None` then the observation image is used.
        """
        observation = self.observation
        py = observation.psfs.shape[-2] // 2
        px = observation.psfs.shape[-1] // 2

        images = observation.images.copy()
        if mask_footprint:
            images.data[observation.weights.data == 0] = 0

        if weight_image is None:
            weight_image = self.get_model()
            # Always convolve in real space to avoid FFT artifacts
            weight_image = observation.convolve(weight_image, mode="real")

            # Due to ringing in the PSF, the convolved model can have
            # negative values. We take the absolute value to avoid
            # negative fluxes in the flux weighted images.
            weight_image.data[:] = np.abs(weight_image.data)

        for src in self.sources:
            if src.is_null:
                src.flux_weighted_image = Image.from_box(Box((0, 0)), bands=observation.bands)  # type: ignore
                continue
            src_model = src.get_model()

            # Grow the model to include the wings of the PSF
            src_box = src.bbox.grow((py, px))
            overlap = observation.bbox & src_box
            src_model = src_model.project(bbox=overlap)
            src_model = observation.convolve(src_model, mode="real")
            src_model.data[:] = np.abs(src_model.data)
            numerator = src_model.data
            denominator = weight_image[overlap].data
            cuts = denominator != 0
            ratio = np.zeros(numerator.shape, dtype=numerator.dtype)
            ratio[cuts] = numerator[cuts] / denominator[cuts]
            ratio[denominator == 0] = 0
            # sometimes numerical errors can cause a hot pixel to have a
            # slightly higher ratio than 1
            ratio[ratio > 1] = 1
            src.flux_weighted_image = src_model.copy_with(data=ratio) * images[overlap]

    def to_blend_data(self) -> ScarletBlendData:
        """Convert the Blend into a persistable data object

        Parameters
        ----------
        blend :
            The blend that is being persisted.

        Returns
        -------
        blend_data :
            The data model for a single blend.
        """
        from .io import ScarletBlendData

        sources: dict[Any, ScarletSourceBaseData] = {}
        for sidx, source in enumerate(self.sources):
            metadata = source.metadata or {}
            if "id" in metadata:
                sources[metadata["id"]] = source.to_source_data()
            else:
                sources[sidx] = source.to_source_data()

        blend_data = ScarletBlendData(
            origin=self.bbox.origin,  # type: ignore
            shape=self.bbox.shape,  # type: ignore
            sources=sources,
            metadata=self.metadata,
        )

        return blend_data
