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

import logging
import numpy as np
from typing import Sequence

from .bbox import Box, get_minimal_boxsize
from .component import FactorizedComponent
from .detect import bounds_to_bbox, get_detect_wavelets
from .image import Image
from .measure import calculate_snr
from .observation import Observation
from .operators import (
    prox_monotonic_mask,
    prox_uncentered_symmetry,
    prox_weighted_monotonic,
)

from .source import Source
from .utils import project_morph_to_center


logger = logging.getLogger("scarlet.lite.initialization")


def trim_morphology(
    center_index: tuple[int, int],
    morph: np.ndarray,
    bg_thresh: float = 0,
    boxsize: int = None,
) -> tuple[np.ndarray, Box]:
    """Trim the morphology up to pixels above a threshold

    Parameters
    ----------
    center_index: Sequence[int, int]
        The location of the center of the morphology.
    morph: np.ndarray
        The morphology to be trimmed.
    bg_thresh: float
        The morphology is trimed to pixels above the threshold.
    boxsize: int
        The size of the box that will contain the morphology.
        If `boxsize` is `None` then the smallest box that will fit the
        trimmed morphology is used.

    Returns
    -------
    morph: np.ndarray
        The trimmed morphology
    box: Box
        The box that contains the morphology.
    """
    # trim morph to pixels above threshold
    mask = morph > bg_thresh
    morph[~mask] = 0

    bbox = Box.from_data(morph, min_value=0)

    # find fitting bbox
    if bbox.contains(center_index):
        size = 2 * max(
            (
                center_index[0] - bbox.start[-2],
                bbox.stop[0] - center_index[-2],
                center_index[1] - bbox.start[-1],
                bbox.stop[1] - center_index[-1],
            )
        )
    else:
        size = 0

    # define new box and cut morphology accordingly
    if boxsize is None:
        boxsize = get_minimal_boxsize(size)

    bottom = center_index[0] - boxsize // 2
    top = center_index[0] + boxsize // 2 + 1
    left = center_index[1] - boxsize // 2
    right = center_index[1] + boxsize // 2 + 1
    bbox = Box.from_bounds((bottom, top), (left, right))
    morph = bbox.extract_from(morph)
    return morph, bbox


def get_min_psf(psfs: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    """Extract the significant portion of the PSF

    This function compares the PSF in each band and
    finds the minimum box needed to contain all pixels
    in the PSF model that differ by more than `thresh`
    in any two bands. The result is that all pixels
    outside of

    Parameters
    ----------
    psfs: np.ndarray
        The full 3D (bands, height, width) PSF model.
    thresh: float
        The minimal difference between two PSFs to be
        considered significant.

    Returns
    -------
    psfs: np.ndarray
        The extracted PSFs.
    """
    # The radius of the PSF in the X and Y directions
    py = psfs.shape[1] // 2
    px = psfs.shape[2] // 2

    # Get the radial coordinates of each pixel
    x = np.arange(psfs.shape[-1])
    y = np.arange(psfs.shape[-2])
    x, y = np.meshgrid(x, y)
    r = np.sqrt((x - px) ** 2 + (y - py) ** 2)

    max_radius = 0
    for p1 in range(len(psfs) - 1):
        for p2 in range(p1 + 1, len(psfs)):
            # Calculate the difference between the PSFs
            psf1 = psfs[p1]
            psf2 = psfs[p2]
            diff = (psf1 - psf2) / np.max([psf1, psf2])
            # keep all pixels greater than the threshold
            significant = np.abs(diff) > thresh
            # extract the radius for all of the significant pixels
            radius = int(np.max(r * significant))
            # Update the maximum radius (if necessary)
            if radius > max_radius:
                max_radius = radius
    # Create the slices to extract the PSF
    dy = py - max_radius
    dx = px - max_radius
    if dy > 0:
        sy = slice(dy, -dy)
    else:
        sy = slice(None)
    if dx > 0:
        sx = slice(dx, -dx)
    else:
        sx = slice(None)
    return psfs[:, sy, sx].copy()


def init_monotonic_morph(
    detect: np.ndarray,
    center: tuple[int, int],
    full_box: Box,
    grow: int = 0,
    normalize: bool = True,
    use_mask: bool = True,
    thresh: float = 0,
) -> tuple[Box, np.ndarray | None]:
    """Initialize a morphology for a monotonic source

    Parameters
    ----------
    detect: np.ndarray
        The 2D detection image contained in `full_box`.
    center: Sequence[int, int]
        The center of the monotonic source.
    full_box: Box
        The bounding box of `detect`.
    grow: int
        The number of pixels to grow the morphology in each direction.
        This can be useful if initializing a source with a kernel that
        is known to be narrower than the expected value of the source.
    normalize: bool
        Whether or not to normalize the morphology.
    use_mask: bool
        When `True` the component is initialized with only the
        monotonic pixels, otherwise the monotonicity operator is used to
        project the morphology to a monotonic solution.
    thresh: float
        The threshold (fraction above the background) to use for trimming the
        morphology.

    Returns
    -------
    bbox: Box
        The bounding box of the morphology.
    morph: np.ndarray
        The initialized morphology.
    """
    if use_mask:
        _, morph, bounds = prox_monotonic_mask(detect, center, max_iter=0)
        bbox = bounds_to_bbox(bounds)
        if bbox.shape == (1, 1) and morph[bbox.slices][0, 0] == 0:
            return bbox, None

        if grow is not None and grow > 0:
            bbox = bbox.grow(grow)
        morph, bbox = project_morph_to_center(morph, center, bbox, full_box)
    else:
        prox_monotonic = prox_weighted_monotonic(
            detect.shape,
            neighbor_weight="angle",
            center=center,
            min_gradient=0,
        )

        morph = prox_monotonic(detect).reshape(detect.shape)

        # truncate morph at thresh * bg_rms
        morph, bbox = trim_morphology(center, morph, bg_thresh=thresh)
        if np.max(morph) == 0:
            return Box((0, 0, 0)), None

    if normalize:
        morph /= np.max(morph)
    return bbox, morph


def multifit_seds(
    observation: Observation,
    morphs: Sequence[Image],
    model: Image = None,
) -> np.ndarray:
    """Fit the seds of multiple components simultaneously

    Parameters
    ----------
    observation:
        The class containing the observation data.
    morphs:
        The morphology of each component.
    model:
        An optional model for sources that are not factorized,
        and thus will not have their SEDs fit.
        This model is subtracted from the data before fittig the other
        SEDs.

    Returns
    -------
    seds: np.ndarray
        The SED for each component, in the same order as `morphs` and `boxes`.
    """
    bands = observation.images.shape[0]
    dtype = observation.images.dtype

    if model is not None:
        img = observation.images - model
    else:
        img = observation.images.copy()

    morph_images = np.zeros((bands, len(morphs), img[0].size), dtype=dtype)
    for idx, morph in enumerate(morphs):
        _img = Image.from_box(Box(img.shape[1:])).insert(morph)
        morph_images[:, idx] = observation.convolve(_img).data.reshape(bands, -1)

    seds = np.zeros((len(morphs), bands), dtype=dtype)

    for b in range(bands):
        a = np.vstack(morph_images[b]).T
        seds[:, b] = np.linalg.lstsq(a, img[b].flatten(), rcond=None)[0]
    seds[seds < 0] = 0
    return seds


def init_chi2_parameters(
    detect: np.ndarray,
    center: tuple[int, int],
    observation: Observation,
    convolved: Image = None,
    use_mask: bool = False,
    thresh: float = 0.5,
) -> tuple[Box, np.ndarray | None, np.ndarray | None]:
    """Initialize parameters using the same general algorithm as scarlet main

    This is currently up to date as of commit `6619736`, but might get out of
    date if the main initialization changes, but even now there are slight
    differences that have very little effect on the overal initialization.

    Parameters
    ----------
    detect:
        The monochromatic detection image (usually a chi^2 coadd,
        possibly weighted by the SED of the source being detected).
    center:
        The location of the center of the source to detect in the full image.
    observation:
        The observation that is being modeled.
    convolved:
        The convolved image in each band. Since the morphology of each source
        is close to the input images, this is a good approximation of the
        convolved morphologies and gives an SED within 1% without having
        to convolve the morphology of each source separately.
        If `convolved` is `None` then the result is accurate to
        machine precision.
    use_mask:
        Whether to use the monotonic mask constraint for initialization or
        the weighted monotonicity constraint.
    thresh:
        The fraction of the `noise_rms` used to trim the morphology.

    Returns
    -------
    bbox: Box
        The bounding box that contains the component.
    morph: np.ndarray
        The morphology of the component.
    sed: np.ndarray
        The SED of the component.
    """
    _detect = prox_uncentered_symmetry(detect.copy(), center, "sdss")
    thresh = np.mean(observation.noise_rms) * thresh

    bbox, morph = init_monotonic_morph(
        _detect,
        center,
        observation.bbox[1:],
        grow=0,
        normalize=False,
        use_mask=use_mask,
        thresh=thresh,
    )

    if morph is None:
        return bbox, None, None

    sed_center = (slice(None), center[0], center[1])
    images = observation.images

    if convolved is None:
        # Convolve the morphology to get the exact SED to match the image,
        # accurate to machine precision
        _morph = Image.from_box(observation.bbox[1:]).insert(morph)
        _morph = _morph.repeat(observation.bands)
        convolved = observation.convolve(_morph, mode="real")
    sed = images[sed_center] / convolved[sed_center]
    sed[sed < 0] = 0
    morph_max = np.max(morph)
    sed *= morph_max
    morph /= morph_max
    return bbox, morph, sed


class Chi2InitParameters:
    """Parameters used to initialize all sources with chi^2 detections

    There are a large number of parameters that are universal for all of the
    sources being initialized from the same set of observed images.
    To simplify the API those parameters are all initialized by this class
    and passed to `init_main_source` for each source.
    It also creates temporary objects that only need to be created once for
    all of the sources in a blend.
    """

    def __init__(
        self,
        observation: Observation,
        detect: np.ndarray = None,
        min_snr: float = 50,
        use_mask: bool = False,
        disk_percentile: float = 25,
        thresh: float = 0.5,
    ):
        """Initialize the class

        Parameters
        ----------
        observation:
            The observation containing the blend
        detect:
            The array that contains a 2D image used for detection.
        min_snr:
            The minimum SNR required per component.
            So a 2-component source requires at least `2*min_snr` while sources
            with SNR < `min_snr` will be initialized with the PSF.
        use_mask:
            Whether to use the monotonic mask or weighted monotonicity for
            initialization.
        disk_percentile:
            The percentage of the overall flux to attribute to the disk.
        thresh:
            The threshold used to trim the morphology,
            so all pixels below `thresh * bg_rms` are set to zero.
        """
        self.observation = observation
        if detect is None:
            # Build the morphology detection image
            detect = np.sum(
                observation.images / (observation.noise_rms**2)[:, None, None], axis=0
            )
        self.detect = detect
        detect = Image(detect)
        # Convolve the detection image.
        # This may seem counter-intuitive, since this is effectively growing the model,
        # but this is exactly what convolution will do to the model in each iteration.
        # So we create the convolved model in order to correctly set the SED.
        self.convolved = observation.convolve(detect.repeat(observation.bands), mode="real")
        # Get the model PSF
        # Convolve the PSF in order to set the SED of a point source correctly.
        self.model_psf = Image(observation.model_psf[0])
        self.convolved_psf = observation.convolve(self.model_psf.repeat(observation.bands), mode="real")
        # Get the "SED" of the PSF
        self.py = self.model_psf.shape[0] // 2
        self.px = self.model_psf.shape[1] // 2
        self.psf_sed = self.convolved_psf[:, self.py, self.px]
        # Set the input parameters
        self.min_snr = min_snr
        self.use_mask = use_mask
        self.disk_percentile = disk_percentile
        self.thresh = thresh


def init_chi2_source(
    center: tuple[int, int], init: Chi2InitParameters
) -> Source | None:
    """Initialize a source from a chi^2 detection.

    Parameter
    ---------
    center:
        The center of the source.
    init:
        The initialization parameters common to all of the sources.
    """
    # Calculate the signal to noise at the center of this source
    snr = np.floor(
        calculate_snr(
            init.observation.images,
            init.observation.variance,
            init.observation.psfs,
            center,
        )
    )
    component_snr = snr / init.min_snr

    # Initialize the bbox, morph, sed for a single component source
    bbox, morph, sed = init_chi2_parameters(
        init.detect,
        center,
        init.observation,
        init.convolved,
        init.use_mask,
        init.thresh,
    )

    if morph is None:
        # There wasn't sufficient flux for an extended source,
        # so create a PSF source.
        sed_center = (slice(None), center[0], center[1])
        sed = init.observation.images[sed_center] / init.psf_sed
        sed[sed < 0] = 0
        morph = init.model_psf.copy()
        morph = morph / np.max(morph)
        bbox = Box(
            init.model_psf.shape, origin=(center[0] - init.py, center[1] - init.px)
        )
        components = [
            FactorizedComponent(
                init.observation.bands,
                sed,
                morph,
                init.observation.bbox[0] @ bbox,
                init.observation.bbox,
                center,
            )
        ]
    elif component_snr >= 2:
        # There was enough flux for a 2-component source,
        # so split the single component model into two components, using the
        # same algorithm as scarlet main.
        bulge_morph = morph.copy()
        disk_morph = morph
        flux_thresh = init.disk_percentile / 100
        mask = disk_morph > flux_thresh
        disk_morph[mask] = flux_thresh
        bulge_morph -= flux_thresh
        bulge_morph[bulge_morph < 0] = 0

        if bulge_morph is None or disk_morph is None:
            if bulge_morph is None:
                if disk_morph is None:
                    return None
                morph = disk_morph
            else:
                morph = bulge_morph
            # One of the components was null,
            # so initialize as a single component
            components = [
                FactorizedComponent(
                    init.observation.bands,
                    sed,
                    morph,
                    init.observation.bbox[0] @ bbox,
                    init.observation.bbox,
                    center,
                )
            ]
        else:
            bulge_morph /= np.max(bulge_morph)
            disk_morph /= np.max(disk_morph)

            bulge_sed, disk_sed = multifit_seds(
                init.observation,
                [
                    Image(bulge_morph, yx0=bbox.origin),
                    Image(disk_morph, yx0=bbox.origin),
                ],
            )

            components = [
                FactorizedComponent(
                    init.observation.bands,
                    bulge_sed,
                    bulge_morph,
                    init.observation.bbox[0] @ bbox,
                    bbox,
                    center,
                ),
                FactorizedComponent(
                    init.observation.bands,
                    disk_sed,
                    disk_morph,
                    init.observation.bbox[0] @ bbox,
                    init.observation.bbox,
                    center,
                ),
            ]
    else:
        components = [
            FactorizedComponent(
                init.observation.bands,
                sed,
                morph,
                init.observation.bbox[0] @ bbox,
                init.observation.bbox,
                center,
            )
        ]

    return Source(components, init.observation.dtype)


def init_all_sources_chi2(
    observation: Observation,
    centers: Sequence[tuple[int, int]],
    detect: np.ndarray = None,
    min_snr: float = 50,
    use_mask: bool = False,
    disk_percentile: float = 25,
    thresh: float = 0.5,
) -> list[Source]:
    """Initialize all of the sources in a blend into factorized components

    This function uses a set of algorithms to give similar results to the
    algorithms in scarlet main to give nearly identical resulting sed and
    morphology arrays without creating all of the intermediate scarlet
    objects.

    See the parameters of `~Chi2InitParameters.__init__` for a description of
    the parameters.

    Returns
    -------
    sources: list[Source]
        The list of sources in the blend.
        This includes null sources that have no components.
    """
    init = Chi2InitParameters(
        observation,
        detect,
        min_snr=min_snr,
        use_mask=use_mask,
        disk_percentile=disk_percentile,
        thresh=thresh,
    )
    sources = []
    for center in centers:
        source = init_chi2_source(center, init)
        sources.append(source)
    return sources


class WaveletInitParameters:
    """Parameters used to initialize all sources with wavelet detections

    There are a large number of parameters that are universal for all of the
    sources being initialized from the same set of wavelet coefficients.
    To simplify the API those parameters are all initialized by this class
    and passed to `init_wavelet_source` for each source.
    """

    def __init__(
        self,
        observation: Observation,
        bulge_slice: slice = slice(None, 2),
        disk_slice: slice = slice(2, -1),
        bulge_grow: int = 5,
        disk_grow: int = 5,
        use_psf: bool = True,
        scales: int = 5,
        wavelets: np.ndarray = None,
    ):
        """Initialize the parameters.

        Parameters
        ----------
        observation:
            The multiband observation of the blend.
        bulge_slice, disk_slice:
            The slice used to select the wavelet scales used for the bulge/disk.
        bulge_grow, disk_grow:
            The number of pixels to grow the bounding box of the bulge/disk
            to leave extra room for growth in the first few iterations.
        use_psf:
            Whether or not to use the PSF for single component sources.
            If `use_psf` is `False` then only sources with low signal at all scales
            are initialized with the PSF morphology.
        scales:
            Number of wavelet scales to use.
        wavelets: `numpy.ndarray`
            The array of wavelet coefficients `(scale, y, x)` used for detection.
        """
        if wavelets is None:
            wavelets = get_detect_wavelets(
                observation.images, observation.variance, scales=scales
            )
        wavelets[wavelets < 0] = 0
        # The detection coadd for single component sources
        detectlets = np.sum(wavelets[:-1], axis=0)
        # The detection coadd for the bulge
        bulgelets = np.sum(wavelets[bulge_slice], axis=0)
        # The detection coadd for the disk
        disklets = np.sum(wavelets[disk_slice], axis=0)

        # useful extracted parameters
        images = observation.images

        # The convolve image, used to initialize the SED
        detect = Image(detectlets)
        convolved = observation.convolve(detect.repeat(observation.bands), mode="real")
        model_psf = Image(observation.model_psf[0])
        convolved_psf = observation.convolve(model_psf.repeat(observation.bands), mode="real")
        py = observation.model_psf.shape[1] // 2
        px = observation.model_psf.shape[2] // 2
        psf_sed = convolved_psf[:, py, px]

        self.observation = observation
        self.images = images
        self.convolved = convolved
        self.detectlets = detectlets
        self.bulgelets = bulgelets
        self.disklets = disklets
        self.bulge_grow = bulge_grow
        self.disk_grow = disk_grow
        self.psf_sed = psf_sed
        self.py = py
        self.px = px
        self.use_psf = use_psf


def init_wavelet_source(
    center: tuple[int, int], nbr_components: int, init: WaveletInitParameters
):
    """Initialize a single source with wavelet coefficients

    Parameters
    ----------
    center:
        The location of the source in the full image.
    nbr_components:
        The number of components of the source.
        If `nbr_components >= 2` then initialization with 2 components
        is attempted. If this fails, or if ` 2 > nbr_components >= 1`
        then initialization with 1 component is attempted.
        Otherwise the source is initialized with the PSF.
    init:
        Parameters used to initialize all sources.

    Returns
    -------
    source: Source
        The initialized source.
    """
    observation = init.observation
    model_psf = observation.model_psf[0]
    sed_center = (slice(None), center[0], center[1])

    if (
        nbr_components < 1
        and init.use_psf
        or init.detectlets[center[0], center[1]] <= 0
    ):
        sed = init.images[sed_center] / init.psf_sed
        sed[sed < 0] = 0
        morph = model_psf.copy()
        morph = morph / np.max(morph)
        bbox = Box(model_psf.shape, origin=(center[0] - init.py, center[1] - init.px))

        component = FactorizedComponent(
            observation.bands, sed, morph, observation.bbox[0] @ bbox, bbox, center
        )
        source = Source([component], observation.dtype)
    elif nbr_components < 2:
        bbox, morph = init_monotonic_morph(
            init.detectlets, center, observation.bbox[1:], init.disk_grow
        )
        if morph is None or np.max(morph) <= 0:
            return Source([], observation.dtype)

        sed = init.images[sed_center] / init.convolved[sed_center]
        sed[sed < 0] = 0
        morph = morph / np.max(morph)

        component = FactorizedComponent(
            observation.bands, sed, morph, observation.bbox[0] @ bbox, bbox, center
        )
        source = Source([component], observation.dtype)
    else:
        bulge_box, bulge_morph = init_monotonic_morph(
            init.bulgelets, center, observation.bbox[1:], init.bulge_grow
        )
        disk_box, disk_morph = init_monotonic_morph(
            init.disklets, center, observation.bbox[1:], init.disk_grow
        )

        if bulge_morph is None or disk_morph is None:
            if bulge_morph is None:
                if disk_morph is None:
                    return None
            # One of the components was null,
            # so initialize as a single component
            return init_wavelet_source(center, 1, init)
        else:
            bulge_sed, disk_sed = multifit_seds(
                observation,
                [
                    Image(bulge_morph, yx0=bulge_box.origin),
                    Image(disk_morph, yx0=disk_box.origin),
                ],
            )

            components = []
            if np.sum(bulge_sed != 0):
                components.append(
                    FactorizedComponent(
                        observation.bands,
                        bulge_sed,
                        bulge_morph,
                        observation.bbox[0] @ bulge_box,
                        observation.bbox,
                        center,
                    )
                )
            else:
                logger.debug("cut bulge")
            if np.sum(disk_sed) != 0:
                components.append(
                    FactorizedComponent(
                        observation.bands,
                        disk_sed,
                        disk_morph,
                        observation.bbox[0] @ disk_box,
                        observation.bbox,
                        center,
                    )
                )
            else:
                logger.debug("cut disk")

            source = Source(components, observation.dtype)
    return source


def init_all_sources_wavelets(
    observation: Observation,
    centers: Sequence[tuple[int, int]],
    min_snr: float = 50,
    bulge_grow: int = 5,
    disk_grow: int = 5,
    use_psf: bool = True,
    bulge_slice: slice = slice(None, 2),
    disk_slice: slice = slice(2, -1),
    scales: int = 5,
    wavelets: np.ndarray = None,
):
    """Initialize all sources using wavelet detection images.

    This does not initialize the SED and morpholgy parameters, so
    `parameterize_source` must still be run to select a parameterization
    (optimizer) that `LiteBlend` requires for fitting.

    See the parameters of `~WaveletInitParameters.__init__` for a description of
    the parameters.

    Parameters
    ----------
    observation:
            The multiband observation of the blend.
    centers:
        The center location for each source to be initialized.
    min_snr:
        Minimum signal to noise for each component. So if `min_snr=50`,
        a source must have SNR > 50 to be initialized with one component
        and SNR > 100 for 2 components.
    bulge_slice, disk_slice:
        The slice used to select the wavelet scales used for the bulge/disk.
    bulge_grow, disk_grow:
        The number of pixels to grow the bounding box of the bulge/disk
        to leave extra room for growth in the first few iterations.
    use_psf:
        Whether or not to use the PSF for single component sources.
        If `use_psf` is `False` then only sources with low signal at all scales
        are initialized with the PSF morphology.
    scales:
        Number of wavelet scales to use.
    wavelets: `numpy.ndarray`
        The array of wavelet coefficients `(scale, y, x)` used for detection.

    Returns
    -------
    sources: `list` of `scarlet.lite.LiteSource`
        The sources that have been initialized.
    """
    init = WaveletInitParameters(
        observation,
        bulge_slice,
        disk_slice,
        bulge_grow,
        disk_grow,
        use_psf,
        scales,
        wavelets,
    )
    sources = []
    for center in centers:
        snr = np.floor(
            calculate_snr(
                observation.images, observation.variance, observation.psfs, center
            )
        )
        component_snr = snr / min_snr
        source = init_wavelet_source(center, component_snr, init)
        sources.append(source)
    return sources
