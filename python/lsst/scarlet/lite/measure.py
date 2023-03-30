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

from typing import cast

import numpy as np

from .bbox import Box
from .image import Image


def calculate_snr(
    images: Image,
    variance: Image,
    psfs: np.ndarray,
    center: tuple[int, int],
) -> float:
    """Calculate the signal to noise for a point source

    This is done by weighting the image with the PSF in each band
    and dividing by the PSF weighted variance.

    Parameters
    ----------
    images:
        The 3D (bands, y, x) image containing the data.
    variance:
        The variance of `images`.
    psfs:
        The PSF in each band.
    center:
        The center of the signal.

    Returns
    -------
    snr:
        The signal to noise of the source.
    """
    py = psfs.shape[1] // 2
    px = psfs.shape[2] // 2
    bbox = Box(psfs[0].shape, origin=(-py + center[0], -px + center[1]))
    overlap = images.bbox & bbox
    noise = variance[overlap].data
    img = images[overlap].data
    _psfs = Image(psfs, bands=images.bands, yx0=cast(tuple[int, int], bbox.origin))[overlap].data
    numerator = img * _psfs
    denominator = (_psfs * noise) * _psfs
    return np.sum(numerator) / np.sqrt(np.sum(denominator))
