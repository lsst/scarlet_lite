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


__all__ = ["LiteSource", "LiteBlend", "FitPsfBlend"]


import numpy as np
from .bbox import overlapped_slices, Box
from .measure import weight_sources
from .utils import insert_image


class LiteSource:
    """A container for components associated with the same astrophysical object

    A source can have a single component, or multiple components, and each can be
    contained in different bounding boxes.
    """
    def __init__(self, components, dtype):
        self.components = components
        self.dtype = dtype
        self._flux = None
        self.flux_bbox = None

    @property
    def n_components(self):
        """The number of components in this source"""
        return len(self.components)

    @property
    def center(self):
        if not self.is_null:
            return self.components[0].center
        return None

    @property
    def is_null(self):
        """True if the source does not have any components"""
        return self.n_components == 0

    @property
    def bbox(self):
        """The minimal bounding box to contain all of this sources components

        Null sources have a bounding box with shape `(0,0,0)`
        """
        if self.n_components == 0:
            return Box((0, 0, 0))
        bbox = self.components[0].bbox
        for component in self.components[1:]:
            bbox = bbox | component.bbox
        return bbox

    def get_model(self, bbox=None, use_flux=False):
        """Build the model for the source

        This is never called during optimization and is only used
        to generate a model of the source for investigative purposes.
        """
        if self.n_components == 0:
            return 0

        if use_flux:
            # Return the redistributed flux
            # (calculated by scarlet.lite.measure.weight_sources)
            if bbox is None:
                return self.flux
            return insert_image(bbox, self.flux_box, self.flux)

        if bbox is None:
            bbox = self.bbox
        model = np.zeros(bbox.shape, dtype=self.dtype)
        for component in self.components:
            slices = overlapped_slices(bbox, component.bbox)
            model[slices[0]] += component.get_model()[slices[1]]
        return model

    def __str__(self):
        return f"LiteSource<{','.join([str(c) for c in self.components])}>"

    def __repr__(self):
        return f"LiteSource<{len(self.components)}>"


class LiteBlend:
    """A single blend.

    This is effectively a combination of the `Blend`, `Observation`, and `Renderer`
    classes in main scarlet, greatly simplified due to the assumptions that the
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
    def __init__(self, sources, observation):
        """Initialize the class.

        Parameters
        ----------
        sources: `list` of `scarlet.lite.LiteSource`
            The sources to fit.
        `observation`: `scarlet.lite.LiteObservation`
            The observation that contains the images,
            PSF, etc. that are being fit.
        """
        self.sources = sources
        self.components = []
        for source in sources:
            self.components.extend(source.components)
        self.observation = observation

        # Initialzie the iteration count and loss function
        self.it = 0
        self.loss = []

    @property
    def bbox(self):
        """The bounding box of the entire blend"""
        return self.observation.bbox

    def get_model(self, convolve=False, use_flux=False):
        """Generate a model of the entire blend"""
        model = np.zeros(self.bbox.shape, dtype=self.observation.images.dtype)

        if use_flux:
            for src in self.sources:
                slices = overlapped_slices(self.bbox, src.flux_box)
                model[slices[0]] += src.flux
        else:
            for component in self.components:
                _model = component.get_model()
                model[component.slices[0]] += _model[component.slices[1]]
            if convolve:
                return self.observation.convolve(model)
        return model

    def grad_log_likelihood(self):
        """Gradient of the likelihood wrt the unconvolved model"""
        model = self.get_model(convolve=True)
        # Update the loss
        self.loss.append(0.5 * -np.sum(self.observation.weights * (self.observation.images - model)**2))
        # Calculate the gradient wrt the model d(logL)/d(model)
        result = self.observation.weights * (model - self.observation.images)
        result = self.observation.convolve(result, grad=True)
        return result

    def fit_spectra(self, clip=False):
        """Fit all of the spectra given their current morphologies

        Parameters
        ----------
        clip: `bool`
            Whether or not to clip components that were not
            assigned any flux during the fit.
        """
        from .initialization import multifit_seds

        morphs = [c.morph for c in self.components]
        boxes = [c.bbox[1:] for c in self.components]
        fit_seds = multifit_seds(self.observation, morphs, boxes)
        for idx, component in enumerate(self.components):
            component.sed[:] = fit_seds[idx]
            component.sed[component.sed < 0] = 0

        if clip:
            components = []
            # Remove components with no sed or morphology
            for src in self.sources:
                _components = []
                for c in src.components:
                    if np.any(c.sed) > 0 and np.any(c.morph) > 0:
                        components.append(c)
                        _components.append(c)
                src.components = _components
            self.components = components
        else:
            for src in self.sources:
                for c in src.components:
                    c.prox_sed(c.sed)

        return self

    @property
    def log_likelihood(self, model=None):
        if model is None:
            return np.array(self.loss)
        return 0.5 * -np.sum(self.observation.weights * (self.observation.images - model)**2)

    def fit(self, max_iter, e_rel=1e-4, min_iter=1, resize=10, reweight=True):
        """Fit all of the parameters

        Parameters
        ----------
        max_iter: `int`
            The maximum number of iterations
        e_rel: `float`
            The relative error to use for determining convergence.
        min_iter: `int`
            The minimum number of iterations.
        resize: `int`
            Number of iterations before attempting to resize the
            resizable components. If `resize` is `None` then
            no resizing is ever attempted.
        reweight: `bool`
            Whether or not to reweight the flux using the source
            models as templates.
        """
        it = self.it
        while it < max_iter:
            # Calculate the gradient wrt the on-convolved model
            grad_log_likelihood = self.grad_log_likelihood()
            # Update each component given the current gradient
            for component in self.components:
                component.update(it, grad_log_likelihood)
            # Check to see if any components need to be resized
            if resize is not None and it > 0 and it % resize == 0:
                for component in self.components:
                    if hasattr(component, "resize"):
                        component.resize()
            # Stopping criteria
            if it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(self.loss[-1]):
                break
            it += 1
        self.it = it
        if reweight:
            weight_sources(self)
        return it, self.loss[-1]


class FitPsfBlend(LiteBlend):
    def grad_log_likelihood(self):
        """Gradient of the likelihood wrt the unconvolved model"""
        model = self.get_model(convolve=True)
        # Update the loss
        self.loss.append(0.5 * -np.sum(self.observation.weights * (self.observation.images - model)**2))
        # Calculate the gradient wrt the model d(logL)/d(model)
        result = self.observation.weights * (model - self.observation.images)
        return result

    def fit(self, max_iter, e_rel=1e-4, min_iter=1, resize=10, reweight=False):
        """Fit all of the parameters

        Parameters
        ----------
        max_iter: `int`
            The maximum number of iterations
        e_rel: `float`
            The relative error to use for determining convergence.
        min_iter: `int`
            The minimum number of iterations.
        resize: `int`
            Number of iterations before attempting to resize the
            resizable components. If `resize` is `None` then
            no resizing is ever attempted.
        reweight: `bool`
            Whether or not to reweight the flux using the source
            models as templates.
        """
        it = self.it
        while it < max_iter:
            # Calculate the gradient wrt the on-convolved model
            grad_log_likelihood = self.grad_log_likelihood()
            _grad_log_likelihood = self.observation.convolve(grad_log_likelihood, grad=True)
            # Update each component given the current gradient
            for component in self.components:
                component.update(it, _grad_log_likelihood)
            # Check to see if any components need to be resized
            if resize is not None and it > 0 and it % resize == 0:
                for component in self.components:
                    if hasattr(component, "resize"):
                        component.resize()

            # Update the PSF
            self.observation.update(it, grad_log_likelihood, self.get_model())
            # Stopping criteria
            if it > min_iter and np.abs(self.loss[-1] - self.loss[-2]) < e_rel * np.abs(self.loss[-1]):
                break
            it += 1
        self.it = it
        if reweight:
            weight_sources(self)
        return it, self.loss[-1]
