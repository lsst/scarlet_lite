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

import numpy as np
from lsst.scarlet.lite.bbox import Box
from lsst.scarlet.lite.image import Image
from lsst.scarlet.lite.models.parametric import EllipticalParametricComponent
from lsst.scarlet.lite.models.psf_fitter import (
    GaussianProfile,
    MoffatProfile,
    PsfFitResult,
    PsfFitter,
    PsfTarget,
    WingProfile,
    default_psf_adaprox_parameterization,
    default_psf_fista_parameterization,
)
from lsst.scarlet.lite.parameters import AdaproxParameter, FistaParameter
from numpy.testing import assert_array_equal
from utils import ScarletTestCase


def make_moffat_target(
    shape: tuple[int, int] = (41, 41),
    alpha: float = 5.0,
    beta: float = 2.5,
    axis_ratio: float = 0.8,
    angle: float = 0.3,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, MoffatProfile]:
    """Render a known elliptical Moffat to use as a fit target.

    Parameters
    ----------
    shape:
        The ``(height, width)`` of the rendered image.
    alpha, beta, axis_ratio, angle, amplitude:
        The ground-truth Moffat parameters.

    Returns
    -------
    data:
        The rendered single-band image.
    truth:
        The profile that produced ``data`` (its parameters are the truth).
    """
    bbox = Box.centered(shape)
    truth = MoffatProfile(
        bbox,
        alpha=alpha,
        beta=beta,
        axis_ratio=axis_ratio,
        angle=angle,
        amplitude=amplitude,
    )
    return truth.get_model().data[0].copy(), truth


class TestPsfTarget(ScarletTestCase):
    def setUp(self):
        self.data = make_moffat_target()[0]

    def test_default_weight(self):
        """A target with no weight gets an all-ones weight and centered
        geometry."""
        target = PsfTarget(self.data)
        assert_array_equal(target.data, self.data)
        assert_array_equal(target.weight, np.ones_like(self.data))
        self.assertBoxEqual(target.bbox, Box.centered(self.data.shape))
        self.assertEqual(target.center, target.bbox.int_center)
        self.assertEqual(target.peak, float(np.max(self.data)))

    def test_explicit_weight(self):
        """An explicit weight is stored verbatim."""
        weight = np.full_like(self.data, 2.0)
        target = PsfTarget(self.data, weight)
        assert_array_equal(target.weight, weight)

    def test_non_2d_data_raises(self):
        """Data that is not 2D is rejected."""
        with self.assertRaises(ValueError):
            PsfTarget(self.data[None])

    def test_weight_shape_mismatch_raises(self):
        """A weight whose shape differs from the data is rejected."""
        with self.assertRaises(ValueError):
            PsfTarget(self.data, np.ones((3, 3)))

    def test_from_image(self):
        """``from_image`` recenters a scarlet image, ignoring its origin."""
        image = Image(self.data, yx0=(100, 200))
        target = PsfTarget.from_image(image)
        assert_array_equal(target.data, self.data)
        self.assertBoxEqual(target.bbox, Box.centered(self.data.shape))


class TestWingProfiles(ScarletTestCase):
    def setUp(self):
        self.bbox = Box.centered((41, 41))
        self.cy, self.cx = self.bbox.int_center

    def test_moffat_construction(self):
        """A Moffat seeds its ellipse, radial, and step parameters from the
        constructor arguments."""
        profile = MoffatProfile(self.bbox, alpha=5.0, beta=2.5, axis_ratio=0.8, angle=0.1)
        self.assertIsInstance(profile, WingProfile)
        self.assertEqual(profile.y0, self.cy)
        self.assertEqual(profile.x0, self.cx)
        self.assertEqual(profile.semi_major, 5.0)
        self.assertAlmostEqual(profile.semi_minor, 5.0 * 0.8)
        self.assertEqual(profile.theta, 0.1)
        # The radial parameter that follows the ellipse is the Moffat index.
        self.assertEqual(profile._params.x[5], 2.5)
        self.assertEqual(profile.spectrum[0], 1.0)
        self.assertEqual(profile.fit_step, 0.2)
        self.assertEqual(profile.fit_flux_factor, 0.1)

    def test_gaussian_construction(self):
        """A Gaussian carries only the ellipse parameters (no radial term)."""
        profile = GaussianProfile(self.bbox, sigma=3.0, axis_ratio=0.9)
        self.assertEqual(profile.semi_major, 3.0)
        self.assertAlmostEqual(profile.semi_minor, 3.0 * 0.9)
        self.assertEqual(len(profile._params.x), 5)

    def test_max_size_default(self):
        """``max_size`` defaults to the larger box dimension."""
        bbox = Box((21, 31), origin=(-10, -15))
        profile = MoffatProfile(bbox, alpha=4.0)
        # The semi-axis upper bounds (proxmax) are the default max_size.
        self.assertEqual(profile.prox_morph.keywords["proxmax"][2], 31.0)
        self.assertEqual(profile.prox_morph.keywords["proxmax"][3], 31.0)

    def test_model_shape(self):
        """A profile renders a single-band model on its own frame."""
        profile = MoffatProfile(self.bbox, alpha=4.0)
        model = profile.get_model()
        self.assertEqual(model.shape, (1,) + tuple(self.bbox.shape))


class TestParameterization(ScarletTestCase):
    def setUp(self):
        self.bbox = Box.centered((31, 31))
        self.target = PsfTarget(make_moffat_target((31, 31))[0])

    def test_default_adaprox_parameterization(self):
        """The adaprox default wraps the arrays as ``AdaproxParameter`` while
        preserving their values."""
        profile = MoffatProfile(self.bbox, alpha=4.0, beta=3.0)
        params = profile._params.x.copy()
        spectrum = profile.spectrum.copy()

        default_psf_adaprox_parameterization(profile)

        self.assertIsInstance(profile._params, AdaproxParameter)
        self.assertIsInstance(profile._spectrum, AdaproxParameter)
        assert_array_equal(profile._params.x, params)
        assert_array_equal(profile.spectrum, spectrum)

    def test_default_fista_parameterization(self):
        """The FISTA default wraps the arrays as ``FistaParameter`` and uses
        the profile's stored steps."""
        profile = MoffatProfile(self.bbox, alpha=4.0, beta=3.0)

        default_psf_fista_parameterization(profile)

        self.assertIsInstance(profile._params, FistaParameter)
        self.assertIsInstance(profile._spectrum, FistaParameter)
        self.assertEqual(profile._params.step, profile.fit_step)
        self.assertEqual(profile._spectrum.step, profile.fit_flux_factor)

    def test_parameterize_default_adaprox(self):
        """``parameterize`` with no argument uses the adaprox default for an
        adaprox fitter and marks every profile parameterized."""
        fitter = PsfFitter(self.target, optimizer="adaprox")
        moffat = fitter.add_profile(MoffatProfile, alpha=4.0)
        gaussian = fitter.add_profile(GaussianProfile, sigma=2.0)
        # Before parameterization the arrays are plain (non-optimizer) params.
        self.assertNotIsInstance(moffat._params, (AdaproxParameter, FistaParameter))

        fitter.parameterize()

        for profile in (moffat, gaussian):
            self.assertIsInstance(profile._params, AdaproxParameter)
            self.assertIsInstance(profile._spectrum, AdaproxParameter)

    def test_parameterize_default_fista(self):
        """A FISTA fitter's default parameterization uses FISTA params."""
        fitter = PsfFitter(self.target, optimizer="fista")
        profile = fitter.add_profile(MoffatProfile, alpha=4.0)

        fitter.parameterize()

        self.assertIsInstance(profile._params, FistaParameter)
        self.assertIsInstance(profile._spectrum, FistaParameter)

    def test_parameterize_custom(self):
        """A custom parameterization is applied to every profile in turn."""
        fitter = PsfFitter(self.target, optimizer="adaprox")
        fitter.add_profile(MoffatProfile, alpha=4.0)
        fitter.add_profile(GaussianProfile, sigma=2.0)

        seen = []

        def custom(component: WingProfile) -> None:
            seen.append(component)
            default_psf_adaprox_parameterization(component)

        fitter.parameterize(custom)
        self.assertEqual(seen, list(fitter.profiles))


class TestPsfFitter(ScarletTestCase):
    def setUp(self):
        self.data, self.truth = make_moffat_target()
        self.target = PsfTarget(self.data)

    def test_invalid_optimizer_raises(self):
        """An unknown optimizer name is rejected at construction."""
        with self.assertRaises(ValueError):
            PsfFitter(self.target, optimizer="newton")

    def test_properties(self):
        """The fitter exposes its target, frame, optimizer, and profiles."""
        fitter = PsfFitter(self.target, optimizer="fista")
        self.assertIs(fitter.target, self.target)
        self.assertBoxEqual(fitter.bbox, self.target.bbox)
        self.assertEqual(fitter.optimizer, "fista")
        self.assertEqual(fitter.profiles, ())

    def test_add_profile(self):
        """``add_profile`` builds the component on the fitter's frame, appends
        it, and returns it."""
        fitter = PsfFitter(self.target)
        profile = fitter.add_profile(MoffatProfile, alpha=4.0, beta=3.0)
        self.assertIsInstance(profile, MoffatProfile)
        self.assertEqual(fitter.profiles, (profile,))
        self.assertBoxEqual(profile.bbox, self.target.bbox)

    def test_add_profile_type_check(self):
        """A class that is not a ``WingProfile`` subclass is rejected."""
        fitter = PsfFitter(self.target)
        with self.assertRaises(TypeError):
            fitter.add_profile(EllipticalParametricComponent)
        with self.assertRaises(TypeError):
            fitter.add_profile(int)  # type: ignore

    def test_fit_requires_profiles(self):
        """Fitting with no profiles is an error."""
        fitter = PsfFitter(self.target)
        with self.assertRaises(ValueError):
            fitter.fit()

    def test_fit_auto_parameterizes(self):
        """``fit`` wraps the profiles with the default parameterization when
        the caller has not done so, and recovers the ground-truth Moffat."""
        fitter = PsfFitter(self.target, optimizer="adaprox", max_iter=2000, e_rel=1e-8)
        profile = fitter.add_profile(MoffatProfile, alpha=4.0, beta=3.0)
        result = fitter.fit()

        self.assertIsInstance(profile._params, AdaproxParameter)
        # The loss drops by many orders of magnitude...
        self.assertLess(result.loss[-1], 1e-6 * result.loss[0])
        # ...and the recovered parameters match the truth.
        self.assertAlmostEqual(profile.semi_major, self.truth.semi_major, places=3)
        self.assertAlmostEqual(profile.semi_minor, self.truth.semi_minor, places=3)
        self.assertAlmostEqual(profile._params.x[5], self.truth._params.x[5], places=3)
        self.assertAlmostEqual(profile.spectrum[0], self.truth.spectrum[0], places=3)

    def test_fit_respects_prior_parameterization(self):
        """A parameterization installed before ``fit`` is not overwritten: the
        custom callable runs exactly once per profile."""
        fitter = PsfFitter(self.target, optimizer="adaprox", max_iter=20)
        fitter.add_profile(MoffatProfile, alpha=4.0)

        calls = []

        def custom(component: WingProfile) -> None:
            calls.append(component)
            default_psf_adaprox_parameterization(component)

        fitter.parameterize(custom)
        fitter.fit()
        # Called once during the explicit parameterize, not again inside fit.
        self.assertEqual(len(calls), 1)

    def test_add_profile_resets_parameterization(self):
        """Adding a profile after ``parameterize`` forces ``fit`` to wrap the
        new profile too."""
        fitter = PsfFitter(self.target, optimizer="adaprox", max_iter=5)
        fitter.parameterize()  # no profiles yet, but marks parameterized
        profile = fitter.add_profile(MoffatProfile, alpha=4.0)
        fitter.fit()
        self.assertIsInstance(profile._params, AdaproxParameter)

    def test_fit_converged_flag(self):
        """The convergence flag is set when the loss stops changing, and is
        ``False`` when the iteration budget is exhausted first."""
        converging = PsfFitter(self.target, optimizer="adaprox", max_iter=5000, e_rel=1e-6)
        converging.add_profile(MoffatProfile, alpha=4.0, beta=3.0)
        self.assertTrue(converging.fit().converged)

        capped = PsfFitter(self.target, optimizer="adaprox", max_iter=1)
        capped.add_profile(MoffatProfile, alpha=4.0, beta=3.0)
        self.assertFalse(capped.fit().converged)

    def test_fit_multiple_profiles(self):
        """A two-component fit runs and its model is the sum of profiles."""
        fitter = PsfFitter(self.target, optimizer="adaprox", max_iter=200)
        moffat = fitter.add_profile(MoffatProfile, alpha=4.0, beta=3.0)
        gaussian = fitter.add_profile(GaussianProfile, sigma=2.0)
        result = fitter.fit()

        self.assertEqual(len(result.components), 2)
        expected = moffat.get_model().data + gaussian.get_model().data
        rendered = result.evaluate(self.data.shape)
        assert_array_equal(rendered.data, expected[0])

    def test_core_radius_excludes_core(self):
        """A positive ``core_radius`` zeroes the central weight, so an
        unmodeled core no longer biases the wing fit."""
        spiked = self.data.copy()
        # The array index of the center (distinct from the centered-frame
        # coordinate, which is ``(0, 0)``).
        cy, cx = (size // 2 for size in self.data.shape)
        spiked[cy - 2 : cy + 3, cx - 2 : cx + 3] += 3.0

        def recover(core_radius: float) -> float:
            fitter = PsfFitter(PsfTarget(spiked), optimizer="adaprox", max_iter=2000, core_radius=core_radius)
            profile = fitter.add_profile(MoffatProfile, alpha=4.0, beta=3.0)
            fitter.fit()
            return profile.semi_major

        masked = recover(4.0)
        unmasked = recover(0.0)
        truth = self.truth.semi_major
        # Masking the corrupted core brings the fit closer to the truth.
        self.assertLess(abs(masked - truth), abs(unmasked - truth))
        self.assertLess(abs(masked - truth), 1.5)


class TestPsfFitResult(ScarletTestCase):
    def setUp(self):
        self.data, self.truth = make_moffat_target()
        fitter = PsfFitter(PsfTarget(self.data), optimizer="adaprox", max_iter=2000, e_rel=1e-8)
        fitter.add_profile(MoffatProfile, alpha=4.0, beta=3.0)
        self.result = fitter.fit()

    def test_result_fields(self):
        """The result bundles the fitted components, loss history, and flag."""
        self.assertIsInstance(self.result, PsfFitResult)
        self.assertEqual(len(self.result.components), 1)
        self.assertIsInstance(self.result.components[0], MoffatProfile)
        self.assertTrue(all(isinstance(value, float) for value in self.result.loss))
        self.assertIsInstance(self.result.converged, bool)

    def test_evaluate_shape_and_origin(self):
        """``evaluate`` renders the analytic model on a centered frame of the
        requested shape."""
        rendered = self.result.evaluate((61, 61))
        self.assertIsInstance(rendered, Image)
        self.assertEqual(rendered.shape, (61, 61))
        self.assertEqual(rendered.bbox.origin, Box.centered((61, 61)).origin)

    def test_evaluate_size_invariance(self):
        """Because the model is analytic, rendering on a larger frame and
        cropping back reproduces the fit-size render."""
        small = self.result.evaluate(self.data.shape)
        large = self.result.evaluate((61, 61))
        cropped = large[small.bbox]
        self.assertImageAlmostEqual(cropped, small)
