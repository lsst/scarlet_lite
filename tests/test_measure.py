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
from lsst.scarlet.lite import Blend, Image, Observation, Source, io
from lsst.scarlet.lite.component import default_adaprox_parameterization
from lsst.scarlet.lite.initialization import FactorizedChi2Initialization
from lsst.scarlet.lite.measure import calculate_snr
from lsst.scarlet.lite.operators import Monotonicity
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from utils import ScarletTestCase


class TestMeasurements(ScarletTestCase):
    def test_snr(self):
        psfs = np.array(
            [
                integrated_circular_gaussian(sigma=1.0),
                integrated_circular_gaussian(sigma=0.85),
                integrated_circular_gaussian(sigma=1.8),
            ]
        )

        bands = tuple("gri")
        images = np.zeros((3, 51, 51), dtype=float)
        images[:, 20:35, 10:25] = psfs * np.arange(1, 4)[:, None, None]
        images = Image(images, bands=bands)
        variance = np.zeros(images.shape, dtype=images.dtype)
        variance[0] = 0.2
        variance[1] = 0.1
        variance[2] = 0.05
        variance = Image(variance, bands=bands)
        snr = calculate_snr(images, variance, psfs, (27, 17))

        numerator = np.sum(images.data[:, 20:35, 10:25] * psfs)
        denominator = np.sqrt(np.sum(psfs**2 * np.array([0.2, 0.1, 0.05])[:, None, None]))
        truth = numerator / denominator
        self.assertAlmostEqual(snr, truth)

    def test_conserve_flux(self):
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
        model_psf = integrated_circular_gaussian(sigma=0.8)
        centers = tuple(np.array([data["catalog"]["y"], data["catalog"]["x"]]).T.astype(int))
        bands = data["filters"]
        observation = Observation(
            Image(data["images"], bands=bands),
            Image(data["variance"], bands=bands),
            Image(1 / data["variance"], bands=bands),
            data["psfs"],
            model_psf[None],
            bands=bands,
        )
        monotonicity = Monotonicity((101, 101))
        init = FactorizedChi2Initialization(observation, centers, monotonicity=monotonicity)

        blend = Blend(init.sources, observation).fit_spectra()
        blend.sources.append(Source([]))
        blend.parameterize(default_adaprox_parameterization)
        blend.fit(100, e_rel=1e-4)

        # Insert a flat source to pickup the background flux
        blend.sources.append(
            Source(
                [
                    io.ComponentCube(
                        bands=observation.bands,
                        bbox=observation.bbox,
                        model=Image(
                            np.ones(observation.shape, dtype=observation.dtype),
                            observation.bands,
                            yx0=(0, 0),
                        ),
                        peak=(0, 0),
                    )
                ]
            )
        )

        blend.conserve_flux(blend)
        flux_model = blend.get_model(use_flux=True)

        self.assertImageAlmostEqual(flux_model, observation.images, decimal=1)
