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


def conserve_flux(blend, mask_footprint: bool = True, images: Image | None = None) -> None:
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
    """
    observation = blend.observation
    py = observation.psfs.shape[-2] // 2
    px = observation.psfs.shape[-1] // 2

    if images is None:
        images = observation.images.copy()
        if mask_footprint:
            images.data[observation.weights.data == 0] = 0
        model = blend.get_model()
        bands = None
    else:
        bands = images.bands
        model = blend.get_model()[bands,]
    # Always convolve in real space to avoid FFT artifacts
    model = observation.convolve(model, mode="real")
    model.data[model.data < 0] = 0

    for src in blend.sources:
        if src.is_null:
            src.flux_weighted_image = Image.from_box(Box((0, 0)), bands=observation.bands)  # type: ignore
            continue
        src_model = src.get_model()

        # Grow the model to include the wings of the PSF
        src_box = src.bbox.grow((py, px))
        overlap = observation.bbox & src_box
        src_model = src_model.project(bbox=overlap)
        src_model = observation.convolve(src_model, mode="real")
        if bands is not None:
            src_model = src_model[bands,]
        src_model.data[src_model.data < 0] = 0
        numerator = src_model.data
        denominator = model[overlap].data
        cuts = denominator != 0
        ratio = np.zeros(numerator.shape, dtype=numerator.dtype)
        ratio[cuts] = numerator[cuts] / denominator[cuts]
        ratio[denominator == 0] = 0
        # sometimes numerical errors can cause a hot pixel to have a
        # slightly higher ratio than 1
        ratio[ratio > 1] = 1
        src.flux_weighted_image = src_model.copy_with(data=ratio) * images[overlap]
