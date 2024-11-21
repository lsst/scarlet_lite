# This file is part of lsst.scarlet.lite.
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

import os
from functools import partial
from typing import cast

import lsst.scarlet.lite.models as models
import numpy as np
from lsst.scarlet.lite import Blend, Box, FistaParameter, Image, Observation, Source
from lsst.scarlet.lite.component import Component, FactorizedComponent, default_adaprox_parameterization
from lsst.scarlet.lite.initialization import FactorizedChi2Initialization
from lsst.scarlet.lite.models import (
    CartesianFrame,
    EllipseFrame,
    EllipticalParametricComponent,
    FactorizedFreeFormComponent,
    FittedPsfBlend,
    FittedPsfObservation,
    ParametricComponent,
)
from lsst.scarlet.lite.operators import Monotonicity
from lsst.scarlet.lite.parameters import AdaproxParameter, parameter, relative_step
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_array_equal
from utils import ScarletTestCase


def parameterize(component: Component):
    assert isinstance(component, ParametricComponent)
    component._spectrum = AdaproxParameter(
        component.spectrum,
        step=partial(relative_step, factor=1e-2, minimum=1e-16),
    )
    component._params = AdaproxParameter(
        component._params.x,
        step=1e-2,
    )


class TestFreeForm(ScarletTestCase):
    def setUp(self) -> None:
        yx0 = (1000, 2000)
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
        self.data = data
        model_psf = integrated_circular_gaussian(sigma=0.8)
        self.detect = np.sum(data["images"], axis=0)
        self.centers = np.array([data["catalog"]["y"], data["catalog"]["x"]]).T + np.array(yx0)
        bands = data["filters"]
        self.observation = Observation(
            Image(data["images"], bands=bands, yx0=yx0),
            Image(data["variance"], bands=bands, yx0=yx0),
            Image(1 / data["variance"], bands=bands, yx0=yx0),
            data["psfs"],
            model_psf[None],
            bands=bands,
        )

    def tearDown(self):
        del self.data

    def test_free_form_component(self):
        images = self.data["images"]

        # Test with no thresholding (sparsity)
        sources = []
        for i in range(5):
            component = FactorizedFreeFormComponent(
                self.observation.bands,
                np.ones(5),
                images[i].copy(),
                self.observation.bbox,
            )
            sources.append(Source([component]))

        blend = Blend(sources, self.observation).fit_spectra()
        blend.parameterize(default_adaprox_parameterization)
        blend.fit(12, e_rel=1e-6)

        # Test with thresholding (sparsity)
        sources = []
        for i in range(5):
            component = FactorizedFreeFormComponent(
                self.observation.bands,
                np.ones(5),
                images[i].copy(),
                self.observation.bbox,
                bg_rms=self.observation.noise_rms,
                bg_thresh=0.25,
                min_area=4,
            )
            sources.append(Source([component]))

        blend = Blend(sources, self.observation).fit_spectra()
        blend.parameterize(default_adaprox_parameterization)
        blend.fit(12, e_rel=1e-6)

        # Test with peak centers specified
        sources = []
        peaks = list(np.array([self.data["catalog"]["y"], self.data["catalog"]["x"]]).T.astype(int))
        for i in range(5):
            component = FactorizedFreeFormComponent(
                self.observation.bands,
                np.ones(5),
                images[i].copy(),
                self.observation.bbox,
                peaks=peaks,
            )
            sources.append(Source([component]))

        blend = Blend(sources, self.observation).fit_spectra()
        blend.parameterize(default_adaprox_parameterization)
        blend.fit(12, e_rel=1e-6)

        # Tests for code blocks that are difficult to reach,
        # to complete test coverage
        component = blend.sources[-1].components[0]
        self.assertFalse(component.resize(self.observation.bbox))
        component.morph[:] = 0
        component.prox_morph(component.morph)


class TestParametric(ScarletTestCase):
    def setUp(self) -> None:
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
        self.data = data
        self.model_psf = integrated_circular_gaussian(sigma=0.8)
        self.detect = np.sum(data["images"], axis=0)
        self.centers = np.array([data["catalog"]["y"], data["catalog"]["x"]]).T
        bands = data["filters"]
        self.observation = Observation(
            Image(data["images"], bands=bands),
            Image(data["variance"], bands=bands),
            Image(1 / data["variance"], bands=bands),
            data["psfs"],
            self.model_psf[None],
            bands=bands,
        )

    def tearDown(self):
        del self.data

    def test_cartesian_frame(self):
        bbox = Box((31, 60), (1, 2))
        frame = CartesianFrame(bbox)
        y = np.linspace(1, 31, 31)
        x = np.linspace(2, 61, 60)
        x, y = np.meshgrid(x, y)

        self.assertBoxEqual(frame.bbox, bbox)
        self.assertTupleEqual(frame.shape, bbox.shape)
        assert_array_equal(frame.x_grid, x)
        assert_array_equal(frame.y_grid, y)
        self.assertIsNone(frame._r2)
        self.assertIsNone(frame._r)

    def test_ellipse_frame(self):
        y0 = 23
        x0 = 36
        major = 3
        minor = 2
        theta = 2 * np.pi / 3
        bbox = Box((50, 39), (2, 7))
        r_min = 1e-20
        frame = EllipseFrame(
            y0,
            x0,
            major,
            minor,
            theta,
            bbox,
        )

        self.assertEqual(frame.x0, x0)
        self.assertEqual(frame.y0, y0)
        self.assertEqual(frame.major, major)
        self.assertEqual(frame.minor, minor)
        self.assertEqual(frame.theta, theta)
        self.assertEqual(frame.bbox, bbox)

        y = np.linspace(2, 51, 50)
        x = np.linspace(7, 45, 39)
        x, y = np.meshgrid(x, y)
        r2 = frame._xa**2 + frame._yb**2 + r_min**2
        r = np.sqrt(r2)

        self.assertEqual(frame._sin, np.sin(theta))
        self.assertEqual(frame._cos, np.cos(theta))
        self.assertBoxEqual(frame.bbox, bbox)
        self.assertTupleEqual(frame.shape, bbox.shape)
        assert_array_equal(frame.x_grid, x)
        assert_array_equal(frame.y_grid, y)
        assert_array_equal(frame.r2_grid, r2)
        assert_array_equal(frame.r_grid, r)

        # There doesn't seem to be much utility to testing the
        # gradients exactly, however we test that they all run
        # properly.
        gradients = (
            frame.grad_x0,
            frame.grad_y0,
            frame.grad_major,
            frame.grad_minor,
            frame.grad_theta,
        )

        input_grad = np.ones(bbox.shape, dtype=float)
        original_grad = input_grad.copy()
        for grad in gradients:
            grad(input_grad, False)
            grad(input_grad, True)

        # Make sure that none of the methods changed the input gradient
        assert_array_equal(input_grad, original_grad)

    def test_parametric_component(self):
        observation = self.observation
        bands = observation.bands
        spectrum = np.ones((observation.n_bands,), dtype=float)
        frame = CartesianFrame(observation.bbox)

        # Test integrated Gaussian PSF and sersic
        sources = []
        for idx, center in enumerate(self.centers):
            # Get the integer center of the source
            cy, cx = int(np.round(center[0])), int(np.round(center[1]))
            # For now we use a fixed bounding box that is the size of
            # the observed PSF image
            bbox = Box((41, 41), origin=(cy - 20, cx - 20)) & observation.bbox
            # Keep track of the initial positions
            yi, xi = cy, cx
            # Restrict the values of the parameters
            _proxmin = np.array([yi - 2, xi - 2, 1e-1, 1e-1, -np.pi / 2, 1.1])
            _proxmax = np.array([yi + 2, xi + 2, frame.shape[-2] / 2, frame.shape[-1] / 2, np.pi / 2, 3])

            __proxmin = np.array([yi - 2, xi - 2, 0.8])
            __proxmax = np.array([yi + 2, xi + 2, 1.2])

            # Initialize a PSF-like component of the source using a
            # non-pixel integrated gaussian
            component = ParametricComponent(
                bands,
                bbox,
                spectrum=parameter(spectrum.copy()),
                morph_params=parameter(np.array([center[0], center[1], 0.8])),
                morph_func=models.integrated_gaussian,
                morph_grad=models.grad_integrated_gaussian,
                morph_prox=partial(models.bounded_prox, proxmin=__proxmin, proxmax=__proxmax),
                morph_step=np.array([1e-2, 1e-2, 1e-2]),
                prox_spectrum=lambda x: x,
            )
            # Define the component to use ADAPROX as the optimizer
            components = [component]

            # Initialize an n=1 sersic component
            component = EllipticalParametricComponent(
                bands,
                bbox,
                spectrum=parameter(spectrum.copy()),
                morph_params=parameter(np.array([center[0], center[1], 2 * 1.2**2, 2 * 1.2**2, 0.0, 1])),
                morph_func=models.sersic,
                morph_grad=models.grad_sersic,
                morph_prox=partial(models.bounded_prox, proxmin=_proxmin, proxmax=_proxmax),
                morph_step=np.array([1e-2, 1e-2, 1e-3, 1e-3, 1e-2, 1e-2]),
            )
            # Define the component to use ADAPROX as the optimizer
            components.append(component)

            component = EllipticalParametricComponent(
                bands,
                bbox,
                spectrum=parameter(spectrum.copy()),
                morph_params=parameter(np.array([center[0], center[1], 2 * 1.2**2, 2 * 1.2**2, 0.0, 1])),
                morph_func=models.sersic,
                morph_grad=models.grad_sersic,
                morph_prox=partial(models.bounded_prox, proxmin=_proxmin, proxmax=_proxmax),
                morph_step=np.array([1e-2, 1e-2, 1e-3, 1e-3, 1e-2, 1e-2]),
            )
            # Define the component to use ADAPROX as the optimizer
            components.append(component)

            # Create a new source using the two components
            sources.append(Source(components))

        # Fit the models
        blend = Blend(sources, observation)
        blend.parameterize(parameterize)
        blend.fit_spectra()

        # Check properties of a component
        component = cast(ParametricComponent, blend.components[0])
        self.assertTupleEqual(component.peak, tuple(self.centers[0]))
        self.assertEqual(component.y0, component._params.x[0])
        self.assertEqual(component.x0, component._params.x[1])
        assert_array_equal(component.morph_step, np.array([1e-2, 1e-2, 1e-2]))

        component = cast(EllipticalParametricComponent, blend.components[1])
        self.assertTupleEqual(component.peak, tuple(self.centers[0]))
        self.assertEqual(component.y0, component._params.x[0])
        self.assertEqual(component.x0, component._params.x[1])
        assert_array_equal(component.semi_major, component._params.x[2])
        assert_array_equal(component.semi_minor, component._params.x[3])
        assert_array_equal(component.theta, component._params.x[4])
        assert_array_equal(component.ellipse_params, component._params.x[:5])
        assert_array_equal(component.radial_params, component._params.x[5:])
        self.assertIsNotNone(component.morph_prox)
        self.assertIsNotNone(component.morph_grad)

        blend.fit(12)

        # Test elliptical and circular Gaussian models
        sources = []
        for idx, center in enumerate(self.centers):
            # Get the integer center of the source
            cy, cx = int(np.round(center[0])), int(np.round(center[1]))
            # For now we use a fixed bounding box that is the size of
            # the observed PSF image
            bbox = Box((41, 41), origin=(cy - 20, cx - 20)) & observation.bbox
            # Keep track of the initial positions
            yi, xi = cy, cx
            # Restrict the values of the parameters
            _proxmin = np.array([yi - 2, xi - 2, 1e-1, 1e-1, -np.pi / 2])
            _proxmax = np.array([yi + 2, xi + 2, frame.shape[-2] / 2, frame.shape[-1] / 2, np.pi / 2])

            __proxmin = np.array([yi - 2, xi - 2])
            __proxmax = np.array([yi + 2, xi + 2])

            # Initialize a PSF-like component of the source using a
            # non-pixel integrated gaussian
            component = ParametricComponent(
                bands,
                bbox,
                spectrum=parameter(spectrum.copy()),
                morph_params=parameter(np.array([center[0], center[1]])),
                morph_func=partial(models.circular_gaussian, sigma=0.8),
                morph_grad=partial(models.grad_circular_gaussian, sigma=0.8),
                morph_prox=partial(models.bounded_prox, proxmin=__proxmin, proxmax=__proxmax),
                morph_step=np.array([1e-2, 1e-2, 1e-2]),
            )
            # Define the component to use ADAPROX as the optimizer
            components = [component]

            # Initialize an n=1 sersic component
            component = EllipticalParametricComponent(
                bands,
                bbox,
                spectrum=parameter(spectrum.copy()),
                morph_params=parameter(np.array([center[0], center[1], 2 * 1.2**2, 2 * 1.2**2, 0.0])),
                morph_func=models.gaussian2d,
                morph_grad=models.grad_gaussian2,
                morph_prox=partial(models.bounded_prox, proxmin=_proxmin, proxmax=_proxmax),
                morph_step=np.array([1e-2, 1e-2, 1e-3, 1e-3, 1e-2, 1e-2]),
            )
            # Define the component to use ADAPROX as the optimizer
            components.append(component)
            # Create a new source using the two components
            sources.append(Source(components))

        # Fit the models
        blend = Blend(sources, observation)
        blend.parameterize(parameterize)
        blend.fit_spectra()
        blend.fit(12)

    def test_psf_fitting(self):
        # Use flat weights for FISTA optimization
        weights = np.ones(self.data["images"].shape)

        monotonicity = Monotonicity((101, 101))

        observation = FittedPsfObservation(
            self.data["images"],
            self.data["variance"],
            weights,
            self.data["psfs"],
            self.model_psf[None],
            bands=self.data["filters"],
        )

        def fista_parameterization(component: Component):
            if isinstance(component, FactorizedComponent):
                component._spectrum = FistaParameter(component.spectrum, step=0.5)
                component._morph = FistaParameter(component.morph, step=0.5)
            else:
                if isinstance(component, FittedPsfObservation):
                    component._fitted_kernel = FistaParameter(component._fitted_kernel.x, step=1e-2)

        init = FactorizedChi2Initialization(observation, self.centers, monotonicity=monotonicity)
        blend = FittedPsfBlend(init.sources, observation).fit_spectra()
        blend.parameterize(fista_parameterization)
        assert isinstance(cast(FittedPsfObservation, blend.observation)._fitted_kernel, FistaParameter)
        blend.fit(12, e_rel=1e-4)
