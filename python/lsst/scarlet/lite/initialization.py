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
from abc import ABC, abstractmethod
from typing import Sequence, cast

import numpy as np

from .bbox import Box
from .component import FactorizedComponent
from .detect import bounds_to_bbox, get_detect_wavelets
from .image import Image
from .measure import calculate_snr
from .observation import Observation
from .operators import Monotonicity, prox_monotonic_mask, prox_uncentered_symmetry
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
        The morphology is trimmed to pixels above the threshold.
    padding:
        The amount to pad each side to allow the source to grow.

    Returns
    -------
    morph:
        The trimmed morphology
    box:
        The box that contains the morphology.
    """
    # trim morph to pixels above threshold
    mask = morph > bg_thresh
    morph[~mask] = 0
    bbox = Box.from_data(morph, threshold=0).grow(padding)
    return morph, bbox


def init_monotonic_morph(
    detect: np.ndarray,
    center: tuple[int, int],
    full_box: Box,
    padding: int = 5,
    normalize: bool = True,
    monotonicity: Monotonicity | None = None,
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
    bbox:
        The bounding box of the morphology.
    morph:
        The initialized morphology.
    """
    center: tuple[int, int] = tuple(center[i] - full_box.origin[i] for i in range(2))  # type: ignore

    if monotonicity is None:
        _, morph, bounds = prox_monotonic_mask(detect, center, max_iter=0)
        bbox = bounds_to_bbox(bounds)
        if bbox.shape == (1, 1) and morph[bbox.slices][0, 0] == 0:
            return Box((0, 0)), None

        if thresh > 0:
            morph, bbox = trim_morphology(morph, bg_thresh=thresh, padding=padding)

        # Shift the bounding box to account for the non-zero origin
        bbox += full_box.origin

    else:
        morph = monotonicity(detect, center)

        # truncate morph at thresh * bg_rms
        morph, bbox = trim_morphology(morph, bg_thresh=thresh, padding=padding)
        # Shift the bounding box to account for the non-zero origin
        bbox += full_box.origin

    if np.max(morph) == 0:
        return Box((0, 0), origin=full_box.origin), None

    if normalize:
        morph /= np.max(morph)

    if padding is not None and padding > 0:
        # Pad the morphology to allow it to grow
        bbox = bbox.grow(padding)

    # Ensure that the bounding box is inside the full box,
    # even after padding.
    bbox = bbox & full_box
    return bbox, morph


def multifit_spectra(
    observation: Observation,
    morphs: Sequence[Image],
    model: Image | None = None,
) -> np.ndarray:
    """Fit the spectra of multiple components simultaneously

    Parameters
    ----------
    observation:
        The class containing the observation data.
    morphs:
        The morphology of each component.
    model:
        An optional model for sources that are not factorized,
        and thus will not have their spectra fit.
        This model is subtracted from the data before fitting the other
        spectra.

    Returns
    -------
    spectra:
        The spectrum for each component, in the same order as `morphs`.
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
        spectra[:, b] = np.linalg.lstsq(a, image[observation.bands[b]].data.flatten(), rcond=None)[0]
    spectra[spectra < 0] = 0
    return spectra


class FactorizedInitialization(ABC):
    """Common variables and methods for both Factorized Component schemes

    Parameters
    ----------
    observation:
        The observation containing the blend
    centers:
        The center of each source to initialize.
    min_snr:
        The minimum SNR required per component.
        So a 2-component source requires at least `2*min_snr` while sources
        with SNR < `min_snr` will be initialized with the PSF.
    monotonicity:
        When `monotonicity` is `None`,
        the component is initialized with only the
        monotonic pixels, otherwise the monotonicity operator is used to
        project the morphology to a monotonic solution.
    use_sparse_init:
        Use a monotonic mask to prevent initial source models from growing
        too large.
    """

    def __init__(
        self,
        observation: Observation,
        convolved: Image,
        centers: Sequence[tuple[int, int]],
        min_snr: float = 50,
        monotonicity: Monotonicity | None = None,
        use_sparse_init: bool = True,
    ):
        self.observation = observation
        self.convolved = convolved
        self.centers = centers
        self.min_snr = min_snr
        self.monotonicity = monotonicity
        self.use_sparse_init = use_sparse_init

        # Get the model PSF
        # Convolve the PSF in order to set the spectrum
        # of a point source correctly.
        model_psf = Image(cast(np.ndarray, observation.model_psf)[0])
        convolved = model_psf.repeat(observation.bands)
        self.convolved_psf = observation.convolve(convolved, mode="real").data
        # Get the "spectrum" of the PSF
        self.py = model_psf.shape[0] // 2
        self.px = model_psf.shape[1] // 2
        self.psf_spectrum = self.convolved_psf[:, self.py, self.px]

        # Initalize all of the sources
        sources = []
        for center in centers:
            source = self.init_source((int(center[0]), int(center[1])))
            sources.append(source)
        self.sources = sources

    def get_snr(self, center: tuple[int, int]) -> float:
        """Get the SNR at the center of a component

        Parameters
        ----------
        center:
            The location of the center of the source.

        Returns
        -------
        result:
            The SNR at the center of the component.
        """
        snr = np.floor(
            calculate_snr(
                self.observation.images,
                self.observation.variance,
                self.observation.psfs,
                center,
            )
        )
        return snr / self.min_snr

    def get_psf_component(self, center: tuple[int, int]) -> FactorizedComponent:
        """Create a factorized component with a PSF morphology

        Parameters
        ----------
        center:
            The center of the component.

        Returns
        -------
        component:
            A `FactorizedComponent` with a PSF-like morphology.
        """
        local_center = (
            center[0] - self.observation.bbox.origin[0],
            center[1] - self.observation.bbox.origin[1],
        )
        # There wasn't sufficient flux for an extended source,
        # so create a PSF source.
        spectrum_center = (slice(None), local_center[0], local_center[1])
        spectrum = self.observation.images.data[spectrum_center] / self.psf_spectrum
        spectrum[spectrum < 0] = 0

        psf = cast(np.ndarray, self.observation.model_psf)[0].copy()
        py = psf.shape[0] // 2
        px = psf.shape[1] // 2
        bbox = Box(psf.shape, origin=(-py + center[0], -px + center[1]))
        bbox = self.observation.bbox & bbox
        morph = Image(psf, yx0=cast(tuple[int, int], bbox.origin))[bbox].data
        component = FactorizedComponent(
            self.observation.bands,
            spectrum,
            morph,
            bbox,
            center,
            self.observation.noise_rms,
            monotonicity=self.monotonicity,
        )
        return component

    def get_single_component(
        self,
        center: tuple[int, int],
        detect: np.ndarray,
        thresh: float,
        padding: int,
    ) -> FactorizedComponent | None:
        """Initialize parameters for a `FactorizedComponent`

        Parameters
        ----------
        center:
            The location of the center of the source to detect in the
            full image.
        detect:
            The image used for detection of the morphology.
        thresh:
            The lower cutoff threshold to use for the morphology.
        padding:
            The amount to pad the morphology to allow for extra flux
            in the first few iterations before resizing.

        Returns
        -------
        component:
            A `FactorizedComponent` created from the detection image.

        """
        local_center = (
            center[0] - self.observation.bbox.origin[0],
            center[1] - self.observation.bbox.origin[1],
        )

        if self.use_sparse_init:
            monotonicity = None
        else:
            monotonicity = self.monotonicity
        bbox, morph = init_monotonic_morph(
            detect,
            center,
            self.observation.bbox,
            padding=padding,
            normalize=False,
            monotonicity=monotonicity,
            thresh=thresh,
        )

        if morph is None:
            return None
        morph = morph[(bbox - self.observation.bbox.origin).slices]

        spectrum_center = (slice(None), local_center[0], local_center[1])
        images = self.observation.images

        convolved = self.convolved
        spectrum = images.data[spectrum_center] / convolved.data[spectrum_center]
        spectrum[spectrum < 0] = 0
        morph_max = np.max(morph)
        spectrum *= morph_max
        morph /= morph_max

        return FactorizedComponent(
            self.observation.bands,
            spectrum,
            morph,
            bbox,
            center,
            self.observation.noise_rms,
            monotonicity=self.monotonicity,
        )

    @abstractmethod
    def init_source(self, center: tuple[int, int]) -> Source | None:
        """Initialize a source

        Parameters
        ----------
        center:
            The center of the source.
        """


class FactorizedChi2Initialization(FactorizedInitialization):
    """Initialize all sources with chi^2 detections

    There are a large number of parameters that are universal for all of the
    sources being initialized from the same set of observed images.
    To simplify the API those parameters are all initialized by this class
    and passed to `init_main_source` for each source.
    It also creates temporary objects that only need to be created once for
    all of the sources in a blend.

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
    padding:
        The amount to pad the morphology to allow for extra flux
        in the first few iterations before resizing.
    """

    def __init__(
        self,
        observation: Observation,
        centers: Sequence[tuple[int, int]],
        detect: np.ndarray | None = None,
        min_snr: float = 50,
        monotonicity: Monotonicity | None = None,
        disk_percentile: float = 25,
        thresh: float = 0.5,
        padding: int = 2,
    ):
        if detect is None:
            # Build the morphology detection image
            detect = np.sum(
                observation.images.data / (observation.noise_rms**2)[:, None, None],
                axis=0,
            )
        self.detect = detect
        _detect = Image(detect)
        # Convolve the detection image.
        # This may seem counter-intuitive,
        # since this is effectively growing the model,
        # but this is exactly what convolution will do to the model
        # in each iteration.
        # So we create the convolved model in order
        # to correctly set the spectrum.
        convolved = observation.convolve(_detect.repeat(observation.bands), mode="real")

        # Set the input parameters
        self.disk_percentile = disk_percentile
        self.thresh = thresh
        self.padding = padding

        # Initialize the sources
        super().__init__(observation, convolved, centers, min_snr, monotonicity)

    def init_source(self, center: tuple[int, int]) -> Source | None:
        """Initialize a source from a chi^2 detection.

        Parameter
        ---------
        center:
            The center of the source.
        init:
            The initialization parameters common to all of the sources.
        max_components:
            The maximum number of components in the source.
        """
        # Some operators need the local center, not center in the full image
        local_center = (
            center[0] - self.observation.bbox.origin[0],
            center[1] - self.observation.bbox.origin[1],
        )

        # Calculate the signal to noise at the center of this source
        component_snr = self.get_snr(center)

        # Initialize the bbox, morph, and spectrum
        # for a single component source
        detect = prox_uncentered_symmetry(self.detect.copy(), local_center, fill=0)
        thresh = np.mean(self.observation.noise_rms) * self.thresh
        component = self.get_single_component(center, detect, thresh, self.padding)

        if component is None:
            components = [self.get_psf_component(center)]
        elif component_snr < 2:
            components = [component]
        else:
            # There was enough flux for a 2-component source,
            # so split the single component model into two components,
            # using the same algorithm as scarlet main.
            bulge_morph = component.morph.copy()
            disk_morph = component.morph
            # Set the threshold for the bulge.
            # Since the morphology is monotonic, this selects the inner
            # of the single component morphology and assigns it to the bulge.
            flux_thresh = self.disk_percentile / 100
            mask = disk_morph > flux_thresh
            # Remove the flux above the threshold so that the disk will have
            # a flat center.
            disk_morph[mask] = flux_thresh
            # Subtract off the thresholded flux (since we're normalizing the
            # morphology anyway) so that it does not have a sharp
            # discontinuity at the edge.
            bulge_morph -= flux_thresh
            bulge_morph[bulge_morph < 0] = 0

            bulge_morph /= np.max(bulge_morph)
            disk_morph /= np.max(disk_morph)

            # Fit the spectra assuming that all of the flux in the image
            # is due to both components. This is not true, but for the
            # vast majority of sources this is a good approximation.
            bulge_spectrum, disk_spectrum = multifit_spectra(
                self.observation,
                [
                    Image(bulge_morph, yx0=cast(tuple[int, int], component.bbox.origin)),
                    Image(disk_morph, yx0=cast(tuple[int, int], component.bbox.origin)),
                ],
            )

            components = [
                FactorizedComponent(
                    self.observation.bands,
                    bulge_spectrum,
                    bulge_morph,
                    component.bbox.copy(),
                    center,
                    self.observation.noise_rms,
                    monotonicity=self.monotonicity,
                ),
                FactorizedComponent(
                    self.observation.bands,
                    disk_spectrum,
                    disk_morph,
                    component.bbox.copy(),
                    center,
                    self.observation.noise_rms,
                    monotonicity=self.monotonicity,
                ),
            ]

        return Source(components)  # type: ignore


class FactorizedWaveletInitialization(FactorizedInitialization):
    """Parameters used to initialize all sources with wavelet detections

    There are a large number of parameters that are universal for all of the
    sources being initialized from the same set of wavelet coefficients.
    To simplify the API those parameters are all initialized by this class
    and passed to `init_wavelet_source` for each source.

    Parameters
    ----------
    observation:
        The multiband observation of the blend.
    centers:
        The center of each source to initialize.
    bulge_slice, disk_slice:
        The slice used to select the wavelet scales used for the
        bulge/disk.
    bulge_padding, disk_padding:
        The number of pixels to grow the bounding box of the bulge/disk
        to leave extra room for growth in the first few iterations.
    use_psf:
        Whether or not to use the PSF for single component sources.
        If `use_psf` is `False` then only sources with low signal
        at all scales are initialized with the PSF morphology.
    scales:
        Number of wavelet scales to use.
    wavelets:
        The array of wavelet coefficients `(scale, y, x)`
        used for detection.
    monotonicity:
        When `monotonicity` is `None`,
        the component is initialized with only the
        monotonic pixels, otherwise the monotonicity operator is used to
        project the morphology to a monotonic solution.
    min_snr:
        The minimum SNR required per component.
        So a 2-component source requires at least `2*min_snr` while sources
        with SNR < `min_snr` will be initialized with the PSF.
    """

    def __init__(
        self,
        observation: Observation,
        centers: Sequence[tuple[int, int]],
        bulge_slice: slice = slice(None, 2),
        disk_slice: slice = slice(2, -1),
        bulge_padding: int = 5,
        disk_padding: int = 5,
        use_psf: bool = True,
        scales: int = 5,
        wavelets: np.ndarray | None = None,
        monotonicity: Monotonicity | None = None,
        min_snr: float = 50,
    ):
        if wavelets is None:
            wavelets = get_detect_wavelets(
                observation.images.data,
                observation.variance.data,
                scales=scales,
            )
        wavelets[wavelets < 0] = 0
        # The detection coadd for single component sources
        detectlets = np.sum(wavelets[:-1], axis=0)
        # The detection coadd for the bulge
        bulgelets = np.sum(wavelets[bulge_slice], axis=0)
        # The detection coadd for the disk
        disklets = np.sum(wavelets[disk_slice], axis=0)

        # The convolved image, used to initialize the spectrum
        detect = Image(detectlets)
        convolved = observation.convolve(detect.repeat(observation.bands), mode="real")

        self.detectlets = detectlets
        self.bulgelets = bulgelets
        self.disklets = disklets
        self.bulge_grow = bulge_padding
        self.disk_grow = disk_padding
        self.use_psf = use_psf

        # Initialize the sources
        super().__init__(observation, convolved, centers, min_snr, monotonicity)

    def init_source(self, center: tuple[int, int]) -> Source | None:
        """Initialize a source from a chi^2 detection.

        Parameter
        ---------
        center:
            The center of the source.
        """
        local_center = (
            center[0] - self.observation.bbox.origin[0],
            center[1] - self.observation.bbox.origin[1],
        )
        nbr_components = self.get_snr(center)
        observation = self.observation

        if (nbr_components < 1 and self.use_psf) or self.detectlets[local_center[0], local_center[1]] <= 0:
            # Initialize the source as an PSF source
            components = [self.get_psf_component(center)]
        elif nbr_components < 2:
            # Inititialize with a single component
            component = self.get_single_component(center, self.detectlets, 0, self.disk_grow)
            if component is not None:
                components = [component]
        else:
            # Initialize with a 2 component model
            bulge_box, bulge_morph = init_monotonic_morph(
                self.bulgelets, center, observation.bbox, self.bulge_grow
            )
            disk_box, disk_morph = init_monotonic_morph(
                self.disklets, center, observation.bbox, self.disk_grow
            )
            if bulge_morph is None or disk_morph is None:
                if bulge_morph is None:
                    if disk_morph is None:
                        return None
                # One of the components was null,
                # so initialize as a single component
                component = self.get_single_component(center, self.detectlets, 0, self.disk_grow)
                if component is not None:
                    components = [component]
            else:
                local_bulge_box = bulge_box - self.observation.bbox.origin
                local_disk_box = disk_box - self.observation.bbox.origin
                bulge_morph = bulge_morph[local_bulge_box.slices]
                disk_morph = disk_morph[local_disk_box.slices]

                bulge_spectrum, disk_spectrum = multifit_spectra(
                    observation,
                    [
                        Image(bulge_morph, yx0=cast(tuple[int, int], bulge_box.origin)),
                        Image(disk_morph, yx0=cast(tuple[int, int], disk_box.origin)),
                    ],
                )

                components = []
                if np.sum(bulge_spectrum != 0):
                    components.append(
                        FactorizedComponent(
                            observation.bands,
                            bulge_spectrum,
                            bulge_morph,
                            bulge_box,
                            center,
                            monotonicity=self.monotonicity,
                        )
                    )
                else:
                    logger.debug("cut bulge")
                if np.sum(disk_spectrum) != 0:
                    components.append(
                        FactorizedComponent(
                            observation.bands,
                            disk_spectrum,
                            disk_morph,
                            disk_box,
                            center,
                            monotonicity=self.monotonicity,
                        )
                    )
                else:
                    logger.debug("cut disk")
        return Source(components)  # type: ignore
