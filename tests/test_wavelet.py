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

import numpy as np
from lsst.scarlet.lite.wavelet import (
    apply_wavelet_denoising,
    get_multiresolution_support,
    multiband_starlet_reconstruction,
    multiband_starlet_transform,
    starlet_reconstruction,
    starlet_transform,
)
from numpy.testing import assert_almost_equal
from utils import ScarletTestCase


class TestWavelet(ScarletTestCase):
    def setUp(self) -> None:
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        self.data = np.load(filename)

    def tearDown(self) -> None:
        del self.data

    def test_transform_inverse(self):
        image = np.sum(self.data["images"], axis=0)
        starlets = starlet_transform(image, scales=3)
        self.assertEqual(starlets.dtype, np.float32)

        # Test number of levels
        self.assertTupleEqual(starlets.shape, (4, 58, 48))

        # Test inverse
        inverse = starlet_reconstruction(starlets)
        assert_almost_equal(inverse, image, decimal=5)
        self.assertEqual(inverse.dtype, starlets.dtype)

        # Test using gen1 starlets
        starlets = starlet_transform(image, scales=3, generation=1)

        # Test number of levels
        self.assertTupleEqual(starlets.shape, (4, 58, 48))

        # Test inverse
        inverse = starlet_reconstruction(starlets, generation=1)
        assert_almost_equal(inverse, image, decimal=5)

    def test_multiband_transform(self):
        image = self.data["images"]
        starlets = multiband_starlet_transform(image, scales=3)
        self.assertEqual(starlets.dtype, np.float32)

        # Test number of levels
        self.assertTupleEqual(starlets.shape, (4, 5, 58, 48))

        # Test inverse
        inverse = multiband_starlet_reconstruction(starlets)
        assert_almost_equal(inverse, image, decimal=5)
        self.assertEqual(inverse.dtype, np.float32)

    def test_extras(self):
        # This is code that is not used in production,
        # but that might be used in the future,
        # so we test to prevent bitrot
        image = np.sum(self.data["images"].astype(float), axis=0)
        starlets = starlet_transform(image, scales=3)

        # Execute to ensure that the code runs
        get_multiresolution_support(image, starlets, 0.1)
        get_multiresolution_support(image, starlets, 0.1, image_type="space")
        apply_wavelet_denoising(image)

    def test_ground_branch_unbiased_sigma(self):
        """Audit finding D-5: the per-scale noise estimate in the
        ``image_type='ground'`` branch must compute std over the
        insignificant pixels only, not over the full array with
        significant pixels zeroed (which pulls the variance down).

        Run the algorithm on a synthetic starlet image where a
        large fraction of pixels are above the significance
        threshold. ``sigma_j`` is the noise-only std at each scale,
        so even though most pixels are masked, the returned value
        must match ``np.std`` of the underlying noise pixels — not
        ``np.std`` of those pixels mixed with zeros.
        """
        rng = np.random.default_rng(0)
        # Build a single-scale "starlet" array where everything is
        # noise: half the pixels are unit-sigma noise, the other
        # half are very-large-amplitude pixels that the iterative
        # threshold will mask out. The unmasked-only std should
        # converge to ~1.0; the bug's zero-padded std would be
        # roughly sqrt(0.5) ~ 0.71.
        noise = rng.normal(scale=1.0, size=(64, 64)).astype(np.float32)
        starlets_per_scale = noise.copy()
        starlets_per_scale[:32] += 100.0  # half the array is "signal"
        # Stack one finest-scale band plus a coarse residual.
        starlets = np.stack([starlets_per_scale, np.zeros_like(noise)])
        # The image just needs a matching shape for the API.
        image = starlets.sum(axis=0)

        result = get_multiresolution_support(image, starlets, 1.0, image_type="ground")
        # The finest scale's converged sigma must match the std of
        # the unmasked noise pixels (~1.0 to within iteration
        # tolerance), not the bug's zero-padded ~0.71.
        self.assertGreater(result.sigma[0], 0.9)
        self.assertLess(result.sigma[0], 1.1)

    def test_space_branch_iterates_sigma(self):
        """Audit finding D-2: the ``image_type='space'`` branch of
        ``get_multiresolution_support`` implements the Starck &
        Murtagh 1998 multi-resolution support algorithm, which
        iteratively refines the global noise ``sigma_e`` from pixels
        that are insignificant at every scale. The iteration is
        meaningful only if each step's threshold uses the *previous*
        iteration's ``sigma``, otherwise the support never changes
        after iteration 0 and the loop is a no-op.

        With a deliberately wrong input ``sigma`` (3x the true noise
        level), the algorithm must still converge to a support
        close to what the correct-sigma run produces.
        """
        rng = np.random.default_rng(0)
        image = rng.normal(scale=1.0, size=(64, 64))
        starlets = starlet_transform(image, generation=1, scales=3)

        result_correct = get_multiresolution_support(image, starlets, 1.0, image_type="space")
        result_overestimate = get_multiresolution_support(image, starlets, 3.0, image_type="space")
        # With the bug, the overestimate run never re-thresholds the
        # mask and produces an essentially empty support (count = 0);
        # with the fix the iteration adapts and the support count is
        # within a small factor of the correct-sigma run.
        correct_count = result_correct.support.sum()
        overestimate_count = result_overestimate.support.sum()
        self.assertGreater(overestimate_count, 0)
        self.assertLess(abs(overestimate_count - correct_count), correct_count)
