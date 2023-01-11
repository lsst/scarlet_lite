import numpy as np

from .bbox import Box
from .blend import Blend
from .image import Image


def calculate_snr(
    images: Image,
    variance: Image,
    psfs: np.ndarray,
    center: tuple[int, int],
):
    """Calculate the signal to noise for a point source

    This is done by weighting the image with the PSF in each band
    and dividing by the PSF weighted variance.

    Parameters
    ----------
    images:
        The 3D (channels, y, x) image containing the data.
    variance:
        The variance of `images`.
    psfs:
        The PSF in each channel.
    center:
        The center of the signal.

    Returns
    -------
    snr: `numpy.ndarray`
        The signal to noise of the source.
    """
    py = psfs.shape[1] // 2
    px = psfs.shape[2] // 2
    bbox = Box(psfs[0].shape, origin=(-py + center[0], -px + center[1]))
    overlap = images.bbox & bbox
    noise = variance[overlap].data
    img = images[overlap].data
    _psfs = Image(psfs, bands=images.bands, yx0=bbox.origin)[overlap].data
    numerator = img * _psfs
    denominator = (_psfs * noise) * _psfs
    return np.sum(numerator) / np.sqrt(np.sum(denominator))


def conserve_flux(blend: Blend, mask_footprint: bool = True):
    """Use the source models as templates to re-weight the data

    This is the standard "deblending" trick, where the models are
    only used as approximations to the data and are used to re-distribute
    the flux in the data according to the ratio of the models for each source.
    There is no return value for this function, instead it adds (or modifies)
    a `flux` attribute and `flux_box` attributes for all of the sources
    that contain their flux and the bounding box containing that flux
    repectively.

    Parameters
    ----------
    blend: `scarlet.lite.LiteBlend`
        The blend that is being fit
    mask_footprint: `bool`
        Whether or not to apply a mask for pixels with zero weight.

    Returns
    -------
    None
    """
    observation = blend.observation
    py = observation.psfs.shape[-2] // 2
    px = observation.psfs.shape[-1] // 2

    images = observation.images.copy()
    if mask_footprint:
        images.data[observation.weights.data == 0] = 0
    model = blend.get_model()
    # Always convolve in real space to avoid FFT artifacts
    model = observation.convolve(model, mode="real")
    model.data[model.data < 0] = 0

    for src in blend.sources:
        if src.is_null:
            src.flux = Image.from_box(Box((0, 0)), bands=observation.bands)
            continue
        src_model = src.get_model()

        # Grow the model to include the wings of the PSF
        src_box = src.bbox.grow((py, px))
        overlap = observation.bbox & src_box
        src_model = src_model.project(bbox=overlap)
        src_model = observation.convolve(src_model, mode="real")
        src_model.data[src_model.data < 0] = 0
        numerator = src_model.data
        denominator = model[overlap].data
        cuts = denominator != 0
        ratio = np.zeros(numerator.shape, dtype=numerator.dtype)
        ratio[cuts] = numerator[cuts] / denominator[cuts]
        ratio[denominator == 0] = 0
        # sometimes numerical errors can cause a hot pixel to have a slightly
        # higher ratio than 1
        ratio[ratio > 1] = 1
        src.flux = src_model.copy_with(data=ratio) * images[overlap]