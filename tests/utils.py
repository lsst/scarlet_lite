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

from typing import Sequence

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from scipy.signal import convolve as scipy_convolve

from scarlet_lite.bbox import Box
from scarlet_lite.fft import match_psf
from scarlet_lite.image import Image
from scarlet_lite.utils import integrated_circular_gaussian


__all__ = ["get_psfs", "ObservationData", "assert_image_equal"]


def get_psfs(sigmas: float | Sequence[float]) -> np.ndarray:
    try:
        iter(sigmas)
    except TypeError:
        sigmas = (sigmas,)
    psf = [integrated_circular_gaussian(sigma=sigma) for sigma in sigmas]
    return np.array(psf)


class ObservationData:
    """Generate an image an associated data used to create the image."""
    def __init__(
        self,
        psfs: np.ndarray,
        spectra: np.ndarray,
        morphs: Sequence[np.ndarray],
        centers: Sequence[tuple[int, int]],
        model_psf: np.ndarray = None,
    ):
        """Initialize the test dataset

        Parameters
        ----------
        psfs:
            The psf in each band as a (bands, Y, X) array.
        spectra:
            The spectrum of all the components in the image.
        morphs:
            The morphology for every component in the image.
        centers:
            The center of every component in the image
        model_psf:
            The 2D PSF of the model space.
        """
        assert len(spectra) == len(morphs) == len(centers)
        origins = [
            tuple([center[i] - (morph.shape[i]-1)//2 for i in range(len(center))])
            for center, morph in zip(centers, morphs)
        ]
        # Define the bounding box for each source based on its center
        boxes = [
            Box((3, 15, 15), (0,) + origin) for center, origin in zip(centers, origins)
        ]

        # Create the image with the sources placed according to their boxes
        images = np.zeros((3, 35, 35), dtype=float)
        for spectrum, center, morph, bbox in zip(spectra, centers, morphs, boxes):
            images[bbox.slices] += spectrum[:, None, None] * morph[None, :, :]

        diff_kernel = match_psf(psfs, model_psf[None], padding=3)
        convolved = np.array(
            [
                scipy_convolve(images[b], diff_kernel.image[b], mode="same")
                for b in range(3)
            ]
        )

        self.images = images
        self.convolved = convolved
        self.diff_kernel = diff_kernel
        self.boxes = boxes


def assert_image_equal(image, truth):
    if not isinstance(image, Image):
        raise AssertionError(f"image is a {type(image)}, not a scarlet_lite `Image`")
    if not isinstance(truth, Image):
        raise AssertionError(f"truth is a {type(truth)}, not a scarlet_lite `Image`")

    try:
        assert_array_equal(image.bands, truth.bands)
    except AssertionError:
        msg = f"Mismatched bands:\nimage: {image.bands}\ntruth: {truth.bands}"
        raise AssertionError(msg)

    try:
        assert_array_equal(image.bbox.shape, truth.bbox.shape)
        assert_array_equal(image.bbox.origin, truth.bbox.origin)
    except AssertionError:
        msg = f"Bounding boxes do not overlap:\nimage: {image.bbox}\ntruth: {truth.bbox}"
        raise AssertionError(msg)

    # The images overlap in multi-band image space, check the values of the images
    assert_array_equal(image.array, truth.array)
