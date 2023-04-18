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

import sys
import traceback
from typing import Sequence
from unittest import TestCase

import numpy as np
from lsst.scarlet.lite.bbox import Box
from lsst.scarlet.lite.fft import match_kernel
from lsst.scarlet.lite.image import Image
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.typing import DTypeLike
from scipy.signal import convolve as scipy_convolve

__all__ = ["get_psfs", "ObservationData", "ScarletTestCase"]


def get_psfs(sigmas: float | Sequence[float]) -> np.ndarray:
    try:
        iter(sigmas)
    except TypeError:
        sigmas = (sigmas,)
    psf = [integrated_circular_gaussian(sigma=sigma) for sigma in sigmas]
    return np.array(psf)


def execute_doc_scripts(filename: str):
    """Test python code in docstrings and document files.

    Any lines not containing code are replaced with a newline character,
    that way if any of the code blocks fail, the line with the error will
    match the linenumber in the .rst file or python file with the docstring.

    Parameters
    ----------
    filename:
        The name of the file to test.
    """
    with open(filename) as file:
        lines = file.readlines()

    full_script = ""
    script = ""
    whitespace = 0
    code_block_start = None
    for n, line in enumerate(lines):
        if ".. code-block:: python" in line:
            if code_block_start is not None:
                message = (
                    f"End of the previous code block starting at {code_block_start}"
                    f"was not detected by the new code block starting at {n}"
                )
                raise ValueError(message)
            code_block_start = n
            tab, directive = line.split("..")
            whitespace = len(tab) + 4
            full_script += f"# {n+1}: " + line
        elif code_block_start is not None:
            indent = len(line) - len(line.lstrip())
            if indent < whitespace and indent != 1:
                code_block_start = None
                whitespace = 0
                full_script += script + "\n"
                script = ""
            elif indent == 1:
                script += "\n"
            else:
                script += line[whitespace:]
        else:
            full_script += f"# {n+1}: " + line

    try:
        exec(full_script)
    except Exception:
        exc_info = sys.exc_info()
        try:
            msg = f"Error encountered in a docstring for the file {filename}."
            raise RuntimeError(msg)
        finally:
            traceback.print_exception(*exc_info)
            del exc_info


class ObservationData:
    """Generate an image an associated data used to create the image."""

    def __init__(
        self,
        bands: tuple,
        psfs: np.ndarray,
        spectra: np.ndarray,
        morphs: Sequence[np.ndarray],
        centers: Sequence[tuple[int, int]],
        model_psf: np.ndarray = None,
        yx0: tuple[int, int] = (0, 0),
        dtype: DTypeLike = float,
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
            tuple([center[i] - (morph.shape[i] - 1) // 2 for i in range(len(center))])
            for center, morph in zip(centers, morphs)
        ]
        # Define the bounding box for each source based on its center
        boxes = [Box((15, 15), origin) for center, origin in zip(centers, origins)]

        # Create the image with the sources placed according to their boxes
        images = np.zeros((3, 35, 35), dtype=dtype)
        spectral_box = Box((len(bands),))
        for spectrum, center, morph, bbox in zip(spectra, centers, morphs, boxes):
            images[(spectral_box @ (bbox - yx0)).slices] += spectrum[:, None, None] * morph[None, :, :]

        diff_kernel = match_kernel(psfs, model_psf[None], padding=3)
        convolved = np.array([scipy_convolve(images[b], diff_kernel.image[b], mode="same") for b in range(3)])
        convolved = convolved.astype(dtype)

        self.images = Image(images, bands=bands, yx0=yx0)
        self.convolved = Image(convolved, bands=bands, yx0=yx0)
        self.diff_kernel = diff_kernel
        self.morphs = [Image(morph, yx0=origin) for morph, origin in zip(morphs, origins)]

        assert self.images.dtype == dtype
        assert self.convolved.dtype == dtype
        assert self.diff_kernel.image.dtype == dtype
        for morph in self.morphs:
            assert morph.dtype == dtype


class ScarletTestCase(TestCase):
    def assertBoxEqual(self, bbox: Box, truth: Box):  # noqa: N802
        try:
            self.assertTupleEqual(bbox.shape, truth.shape)
        except AssertionError:
            msg = f"Box shapes differ: {bbox.shape}!={truth.shape}"
            raise AssertionError(msg)
        try:
            self.assertTupleEqual(bbox.origin, truth.origin)
        except AssertionError:
            msg = f"Box origins differ: {bbox.origin}!={truth.origin}"
            raise AssertionError(msg)

    def assertImageAlmostEqual(self, image: Image, truth: Image, decimal: int = 7):  # noqa: N802
        if not isinstance(image, Image):
            raise AssertionError(f"image is a {type(image)}, not a lsst.scarlet.lite `Image`")
        if not isinstance(truth, Image):
            raise AssertionError(f"truth is a {type(truth)}, not a lsst.scarlet.lite `Image`")

        try:
            self.assertTupleEqual(image.bands, truth.bands)
        except AssertionError:
            msg = f"Mismatched bands:{image.bands} != {truth.bands}"
            raise AssertionError(msg)

        try:
            self.assertTupleEqual(image.bbox.shape, truth.bbox.shape)
            self.assertTupleEqual(image.bbox.origin, truth.bbox.origin)
        except AssertionError:
            msg = f"Bounding boxes do not overlap:\nimage: {image.bbox}\ntruth: {truth.bbox}"
            raise AssertionError(msg)

        # The images overlap in multi-band image space,
        # check the values of the images
        assert_almost_equal(image.data, truth.data, decimal=decimal)

    def assertImageEqual(self, image: Image, truth: Image):  # noqa: N802
        self.assertImageAlmostEqual(image, truth)
        assert_array_equal(image.data, truth.data)
