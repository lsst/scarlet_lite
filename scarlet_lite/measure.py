import numpy as np

from .bbox import Box, overlapped_slices
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
    slices = images.overlapped_slices(bbox)
    noise = variance[slices[0]].data
    img = images[slices[0]].data
    _psfs = psfs[slices[1]]
    numerator = img * _psfs
    denominator = (_psfs * noise) * _psfs
    return np.sum(numerator) / np.sqrt(np.sum(denominator))


def conserve_flux(blend, mask_footprint=True):
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
        images = images * (observation.weights > 0)
    model = blend.get_model()
    # Always convolve in real space to avoid FFT artifacts
    model = observation.convolve(model, mode="real")
    model[model < 0] = 0
    zero_shape = (model.n_bands, 0, 0)

    for src in blend.sources:
        if src.is_null:
            zero_flux = np.zeros(zero_shape)
            src.flux = Image(zero_flux, bands=observation.bands)
            src.flux_box = Box((0, 0))
            continue
        src_model = src.get_model()
        # Grow the model to include the wings of the PSF
        bbox = src.bbox.grow((0, py, px))
        src_model = insert_image(bbox, src.bbox, src_model)
        src_model = observation.convolve(src_model, mode="real")
        src_model[src_model < 0] = 0
        slices = overlapped_slices(observation.bbox, bbox)
        numerator = src_model[slices[1]]
        denominator = model[slices[0]]
        cuts = denominator != 0
        ratio = np.zeros(numerator.shape, dtype=numerator.dtype)
        ratio[cuts] = numerator[cuts] / denominator[cuts]
        ratio[denominator == 0] = 0
        # sometimes numerical errors can cause a hot pixel to have a slightly
        # higher ratio than 1
        ratio[ratio > 1] = 1
        src.flux = ratio * images[slices[0]]
        src.flux_box = observation.bbox & bbox
