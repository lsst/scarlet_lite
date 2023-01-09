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

from .bbox import Box
from .component import FactorizedComponent
from .detect import bounds_to_bbox, get_detect_wavelets
from .image import Image
from .measure import calculate_snr
from .observation import Observation
from .operators import (
    prox_monotonic_mask,
    prox_uncentered_symmetry,
    Monotonicity,
)

from .source import Source


logger = logging.getLogger("scarlet.lite.initialization")


def trim_morphology(
    morph: np.ndarray,
    bg_thresh: float = 0,
    padding: int = 5,
) -> tuple[np.ndarray, Box]:
    """Trim the morphology up to pixels above a threshold

    Parameters
    ----------
    morph:
        The morphology to be trimmed.
    bg_thresh:
        The morphology is trimed to pixels above the threshold.
    padding:
        The amount to pad each side to allow the source to grow.

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
    bbox = Box.from_data(morph, min_value=0).grow(padding)
    return morph, bbox


def init_monotonic_morph(
    detect: np.ndarray,
    center: tuple[int, int],
    full_box: Box,
    padding: int = 5,
    normalize: bool = True,
    monotonicity: Monotonicity = None,
    thresh: float = 0,
) -> tuple[Box, np.ndarray | None]:
    """Initialize a morphology for a monotonic source

    Parameters
    ----------
    detect:
        The 2D detection image contained in `full_box`.
    center:
        The center of the monotonic source.
    full_box:
        The bounding box of `detect`.
    padding:
        The number of pixels to grow the morphology in each direction.
        This can be useful if initializing a source with a kernel that
        is known to be narrower than the expected value of the source.
    normalize:
        Whether or not to normalize the morphology.
    monotonicity:
        When `monotonicity` is `None`,
        the component is initialized with only the
        monotonic pixels, otherwise the monotonicity operator is used to
        project the morphology to a monotonic solution.
    thresh:
        The threshold (fraction above the background) to use for trimming the
        morphology.

    Returns
    -------
    bbox: Box
        The bounding box of the morphology.
    morph: np.ndarray
        The initialized morphology.
    """
    if monotonicity is None:
        _, morph, bounds = prox_monotonic_mask(detect, center, max_iter=0)
        bbox = bounds_to_bbox(bounds)
        if bbox.shape == (1, 1) and morph[bbox.slices][0, 0] == 0:
            return bbox, None

        if padding is not None and padding > 0:
            # Pad the morphology to allow it to grow
            bbox = bbox.grow(padding)

        if thresh > 0:
            morph, bbox = trim_morphology(morph, bg_thresh=thresh, padding=padding)

    else:
        morph = monotonicity(detect, center)

        # truncate morph at thresh * bg_rms
        morph, bbox = trim_morphology(morph, bg_thresh=thresh, padding=padding)

    if np.max(morph) == 0:
        return Box((0, 0)), None

    if normalize:
        morph /= np.max(morph)

    # Ensure that the bounding box is inside the full box,
    # even after padding.
    bbox = bbox & full_box
    return bbox, morph


def multifit_spectra(
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
    _bands = observation.bands
    n_bands = len(_bands)
    dtype = observation.images.dtype

    if model is not None:
        image = observation.images - model
    else:
        image = observation.images.copy()

    morph_images = np.zeros((n_bands, len(morphs), image.data[0].size), dtype=dtype)
    for idx, morph in enumerate(morphs):
        _image = morph.repeat(observation.bands)
        _image = Image.from_box(image.bbox, bands=image.bands).insert(_image)
        morph_images[:, idx] = observation.convolve(_image).data.reshape(n_bands, -1)

    spectra = np.zeros((len(morphs), n_bands), dtype=dtype)

    for b in range(n_bands):
        a = np.vstack(morph_images[b]).T
        spectra[:, b] = np.linalg.lstsq(
            a, image[observation.bands[b]].data.flatten(), rcond=None
        )[0]
    spectra[spectra < 0] = 0
    return spectra


class FactorizedChi2Initialization:
    """Initialize all sources with chi^2 detections

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
        centers: Sequence[tuple[int, int]],
        detect: np.ndarray = None,
        min_snr: float = 50,
        monotonicity: Monotonicity | None = None,
        disk_percentile: float = 25,
        thresh: float = 0.5,
    ):
        """Initialize the class

        Parameters
        ----------
        observation:
            The observation containing the blend
        centers:
            The center of each source to initialize.
        detect:
            The array that contains a 2D image used for detection.
        min_snr:
            The minimum SNR required per component.
            So a 2-component source requires at least `2*min_snr` while sources
            with SNR < `min_snr` will be initialized with the PSF.
        monotonicity:
            When `monotonicity` is `None`,
            the component is initialized with only the
            monotonic pixels, otherwise the monotonicity operator is used to
            project the morphology to a monotonic solution.
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
                observation.images.data / (observation.noise_rms**2)[:, None, None],
                axis=0,
            )
        self.detect = detect
        detect = Image(detect)
        # Convolve the detection image.
        # This may seem counter-intuitive, since this is effectively growing the model,
        # but this is exactly what convolution will do to the model in each iteration.
        # So we create the convolved model in order to correctly set the SED.
        self.convolved = observation.convolve(
            detect.repeat(observation.bands), mode="real"
        )
        # Get the model PSF
        # Convolve the PSF in order to set the SED of a point source correctly.
        model_psf = Image(observation.model_psf[0])
        self.convolved_psf = observation.convolve(
            model_psf.repeat(observation.bands), mode="real"
        ).data
        # Get the "SED" of the PSF
        self.py = model_psf.shape[0] // 2
        self.px = model_psf.shape[1] // 2
        self.psf_sed = self.convolved_psf[:, self.py, self.px]
        # Set the input parameters
        self.min_snr = min_snr
        self.monotonicity = monotonicity
        self.disk_percentile = disk_percentile
        self.thresh = thresh

        sources = []
        for center in centers:
            source = self.init_source((int(center[0]), int(center[1])))
            sources.append(source)
        self.sources = sources

    def init_component(
        self, center: tuple[int, int]
    ) -> tuple[Box, np.ndarray | None, np.ndarray | None]:
        """Initialize parameters for a `FactorizedComponent`

        Parameters
        ----------
        center:
            The location of the center of the source to detect in the full image.

        Returns
        -------
        bbox: Box
            The bounding box that contains the component.
        morph: np.ndarray
            The morphology of the component.
        sed: np.ndarray
            The SED of the component.
        """
        _detect = prox_uncentered_symmetry(self.detect.copy(), center, fill=0)
        thresh = np.mean(self.observation.noise_rms) * self.thresh

        bbox, morph = init_monotonic_morph(
            _detect,
            center,
            self.observation.bbox,
            padding=0,
            normalize=False,
            monotonicity=self.monotonicity,
            thresh=thresh,
        )

        if morph is None:
            return bbox, None, None
        morph = morph[bbox.slices]

        sed_center = (slice(None), center[0], center[1])
        images = self.observation.images

        if self.convolved is None:
            # Convolve the morphology to get the exact SED to match the image,
            # accurate to machine precision
            _morph = Image.from_box(self.observation.bbox[1:]).insert(morph)
            _morph = _morph.repeat(self.observation.bands)
            convolved = self.observation.convolve(_morph, mode="real")
        else:
            convolved = self.convolved
        sed = images.data[sed_center] / convolved.data[sed_center]
        sed[sed < 0] = 0
        morph_max = np.max(morph)
        sed *= morph_max
        morph /= morph_max
        return bbox, morph, sed

    def init_source(self, center: tuple[int, int]) -> Source | None:
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
                self.observation.images,
                self.observation.variance,
                self.observation.psfs,
                center,
            )
        )
        component_snr = snr / self.min_snr

        # Initialize the bbox, morph, sed for a single component source
        bbox, morph, sed = self.init_component(center)

        if morph is None:
            # There wasn't sufficient flux for an extended source,
            # so create a PSF source.
            sed_center = (slice(None), center[0], center[1])
            sed = self.observation.images[sed_center] / self.psf_sed
            sed[sed < 0] = 0
            morph = self.observation.model_psf.copy()
            morph = morph / np.max(morph)
            bbox = Box(
                self.observation.model_psf.shape,
                origin=(center[0] - self.py, center[1] - self.px),
            )
            components = [
                FactorizedComponent(
                    self.observation.bands,
                    sed,
                    morph,
                    bbox,
                    self.observation.bbox,
                    center,
                    self.observation.noise_rms,
                    monotonicity=self.monotonicity,
                )
            ]
        elif component_snr >= 2:
            # There was enough flux for a 2-component source,
            # so split the single component model into two components, using the
            # same algorithm as scarlet main.
            bulge_morph = morph.copy()
            disk_morph = morph
            flux_thresh = self.disk_percentile / 100
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
                        self.observation.bands,
                        sed,
                        morph,
                        bbox,
                        self.observation.bbox,
                        center,
                        self.observation.noise_rms,
                    )
                ]
            else:
                bulge_morph /= np.max(bulge_morph)
                disk_morph /= np.max(disk_morph)

                bulge_sed, disk_sed = multifit_spectra(
                    self.observation,
                    [
                        Image(bulge_morph, yx0=bbox.origin),
                        Image(disk_morph, yx0=bbox.origin),
                    ],
                )

                components = [
                    FactorizedComponent(
                        self.observation.bands,
                        bulge_sed,
                        bulge_morph,
                        bbox,
                        self.observation.bbox,
                        center,
                        self.observation.noise_rms,
                        monotonicity=self.monotonicity,
                    ),
                    FactorizedComponent(
                        self.observation.bands,
                        disk_sed,
                        disk_morph,
                        bbox,
                        self.observation.bbox,
                        center,
                        self.observation.noise_rms,
                        monotonicity=self.monotonicity,
                    ),
                ]
        else:
            components = [
                FactorizedComponent(
                    self.observation.bands,
                    sed,
                    morph,
                    bbox,
                    self.observation.bbox,
                    center,
                    self.observation.noise_rms,
                    monotonicity=self.monotonicity,
                )
            ]

        return Source(components)


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
        convolved_psf = observation.convolve(
            model_psf.repeat(observation.bands), mode="real"
        )
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
        source = Source([component])
    elif nbr_components < 2:
        bbox, morph = init_monotonic_morph(
            init.detectlets, center, observation.bbox[1:], init.disk_grow
        )
        if morph is None or np.max(morph) <= 0:
            return Source([])

        sed = init.images[sed_center] / init.convolved[sed_center]
        sed[sed < 0] = 0
        morph = morph / np.max(morph)

        component = FactorizedComponent(
            observation.bands, sed, morph, observation.bbox[0] @ bbox, bbox, center
        )
        source = Source([component])
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
            bulge_sed, disk_sed = multifit_spectra(
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

            source = Source(components)
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
