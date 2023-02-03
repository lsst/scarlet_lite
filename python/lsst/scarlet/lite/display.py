from astropy.visualization.lupton_rgb import Mapping, LinearMapping, AsinhMapping
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import numpy.typing as npt

from .bbox import Box
from .blend import Blend
from .source import Source
from .image import Image


# Size of a single panel, used for generating figures with multiple sub-plots
panel_size = 4.0


def channels_to_rgb(channels: int) -> np.ndarray:
    """Get the linear mapping of multiple channels to RGB channels
    The mapping created here assumes the channels are ordered in wavelength
    direction, starting with the shortest wavelength.
    The mapping seeks to produce a relatively even weights for across
    all channels. It does not consider e.g.
    signal-to-noise variations across channels or human perception.
    Parameters
    ----------
    channels: int
        Number of channels (in range(0,7)).
    Returns
    -------
    channel_map: np.ndarray
        Array (3, `channels`) to map onto RGB.
    """
    assert channels in range(
        0, 8
    ), "No mapping has been implemented for more than {} channels".format(channels)

    channel_map = np.zeros((3, channels))
    if channels == 1:
        channel_map[0, 0] = channel_map[1, 0] = channel_map[2, 0] = 1
    elif channels == 2:
        channel_map[0, 1] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[1, 0] = 0.333
        channel_map[2, 0] = 0.667
        channel_map /= 0.667
    elif channels == 3:
        channel_map[0, 2] = 1
        channel_map[1, 1] = 1
        channel_map[2, 0] = 1
    elif channels == 4:
        channel_map[0, 3] = 1
        channel_map[0, 2] = 0.333
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.667
        channel_map[2, 1] = 0.333
        channel_map[2, 0] = 1
        channel_map /= 1.333
    elif channels == 5:
        channel_map[0, 4] = 1
        channel_map[0, 3] = 0.667
        channel_map[1, 3] = 0.333
        channel_map[1, 2] = 1
        channel_map[1, 1] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 1.667
    elif channels == 6:
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    elif channels == 7:
        channel_map[:, 6] = 2 / 3.0
        channel_map[0, 5] = 1
        channel_map[0, 4] = 0.667
        channel_map[0, 3] = 0.333
        channel_map[1, 4] = 0.333
        channel_map[1, 3] = 0.667
        channel_map[1, 2] = 0.667
        channel_map[1, 1] = 0.333
        channel_map[2, 2] = 0.333
        channel_map[2, 1] = 0.667
        channel_map[2, 0] = 1
        channel_map /= 2
    return channel_map


class LinearPercentileNorm(LinearMapping):
    def __init__(self, img: npt.ArrayLike, percentiles: tuple[int, int] = None):
        """Create norm that is linear between lower and upper percentile of img
        Parameters
        ----------
        img:
            Image to normalize
        percentiles:
            Lower and upper percentile to consider (default = ``(1,99)``).
            Pixel values below will be
            set to zero, above to saturated.
        """
        if percentiles is None:
            percentiles = (1, 99)
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        super().__init__(minimum=vmin, maximum=vmax)


class AsinhPercentileNorm(AsinhMapping):
    def __init__(self, img: npt.ArrayLike, percentiles: tuple[int, int] = None):
        """Create norm that is linear between lower and upper percentile of img
        Parameters
        ----------
        img:
            Image to normalize.
        percentiles:
            Lower and upper percentile to consider (default = ``(1,99)``).
            Pixel values below will be
            set to zero, above to saturated.
        """
        if percentiles is None:
            percentiles = (1, 99)
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        # solution for beta assumes flat spectrum at vmax
        stretch = vmax - vmin
        beta = stretch / np.sinh(1)
        super().__init__(minimum=vmin, stretch=stretch, Q=beta)


def img_to_3channel(
    img: np.ndarray, channel_map: np.ndarray = None, fill_value: float = 0
) -> np.ndarray:
    """Convert multi-band image cube into 3 RGB channels

    Parameters
    ----------
    img: np.ndarray
        This should be an array with dimensions (channels, height, width).
    channel_map: np.ndarray
        Linear mapping with dimensions (3, channels)
    fill_value: float
        Value to use for any masked pixels.
    Returns
    -------
    RGB: np.ndarray
    """
    # expand single img into cube
    assert len(img.shape) in [2, 3]
    if len(img.shape) == 2:
        ny, nx = img.shape
        img_ = img.reshape((1, ny, nx))
    elif len(img.shape) == 3:
        img_ = img
    else:
        raise ValueError(
            f"Image must have either 2 or 3 dimensions, got {len(img.shape)}"
        )
    dimensions = len(img_)

    # filterWeights: channel x band
    if channel_map is None:
        channel_map = channels_to_rgb(dimensions)
    else:
        assert channel_map.shape == (3, len(img))

    # map channels onto RGB channels
    _, ny, nx = img_.shape
    rgb = np.dot(channel_map, img_.reshape(dimensions, -1)).reshape((3, ny, nx))

    if hasattr(rgb, "mask"):
        rgb = rgb.filled(fill_value)

    return rgb


def img_to_rgb(
    img: np.ndarray | Image,
    channel_map: np.ndarray = None,
    fill_value: float = 0,
    norm: Mapping = None,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Convert images to normalized RGB.
    If normalized values are outside of the range [0..255], they will be
    truncated such as to preserve the corresponding color.
    Parameters
    ----------
    img: np.ndarray
        This should be an array with dimensions (channels, height, width).
    channel_map: np.ndarray
        Linear mapping with dimensions (3, channels)
    fill_value: float
        Value to use for any masked pixels.
    norm: Mapping
        Norm to use for mapping in the allowed range [0..255].
        If ``norm=None``, `LinearPercentileNorm` will be used.
    mask: np.ndarray
        A [0,1] binary mask to apply over the top of the image,
        where pixels with mask==1 are masked out.
    Returns
    -------
    rgb: np.ndarray
        RGB values with dimensions (3, height, width) and dtype uint8
    """
    if isinstance(img, Image):
        img = img.data
    _rgb = img_to_3channel(img, channel_map=channel_map, fill_value=fill_value)
    if norm is None:
        norm = LinearMapping(image=_rgb)
    rgb = norm.make_rgb_image(*_rgb)
    if mask is not None:
        rgb = np.dstack([rgb, ~mask * 255])
    return rgb


def show_likelihood(
    blend: Blend, figsize: tuple[float, float] = None, **kwargs
) -> matplotlib.pyplot.Figure:
    """Display a plot of the likelihood in each iteration for a blend

    Parameters
    ----------
    blend: Blend
        The blend to generate the likelihood plot for.
    figsize: tuple[float, float]
        The size of the figure.
    kwargs: dict
        Keyword arguements passed to `blend.log_likelihood`.

    Returns
    -------
    fig: matplotlib.Figure
        The figure containing the log-likelihood plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(blend.log_likelihood, **kwargs)
    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("log-Likelihood")
    return fig


def _add_markers(
    src: Source,
    extent: tuple[float, float, float, float],
    ax: matplotlib.pyplot.Axes,
    add_markers: bool,
    add_boxes: bool,
    marker_kwargs: dict,
    box_kwargs: dict,
):
    if add_markers and hasattr(src, "center") and src.center is not None:
        center = np.array(src.center)[::-1]
        ax.plot(*center, "wx", **marker_kwargs)

    if add_boxes:

        rect = Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            **box_kwargs,
        )
        ax.add_artist(rect)


def show_scene(
    blend: Blend,
    norm: Mapping = None,
    channel_map: np.ndarray = None,
    show_model: bool = True,
    show_observed: bool = False,
    show_rendered: bool = False,
    show_residual: bool = False,
    add_labels: bool = True,
    add_boxes: bool = False,
    figsize: tuple[float, float] = None,
    linear: bool = True,
    use_flux: bool = False,
    box_kwargs: dict = None,
) -> matplotlib.pyplot.Figure:
    """Plot all sources to recreate the scence.

    The functions provides a fast way of evaluating the quality
    of the entire model,
    i.e. the combination of all scences that seek to fit the observation.

    Parameters
    ----------
    blend: Blend
    norm: Mapping
        Norm to compress image intensity to the range [0,255].
    channel_map: np.ndarray
        Linear mapping with dimensions (3, channels).
    show_model: bool
        Whether the model is shown in the model frame.
    show_observed: bool
        Whether the observation is shown.
    show_rendered: bool
        Whether the model, rendered to match the observation, is shown.
    show_residual: bool
        Whether the residuals between rendered model and observation is shown.
    add_labels: bool
        Whether each source is labeled with its numerical
        index in the source list.
    add_boxes: bool
        Whether each source box is shown.
    figsize: tuple[float, float]
        Size of the final figure.
    linear: bool
        Whether or not to display the scene in a single line (`True`) or
        on multiple lines (`False`).
    use_flux: bool
        Whether to show the flux redistributed model (`source.flux`) or
        the model itself (`source.get_model()`) for each source.
    box_kwargs: dict
        Keyword arguments to create boxes (`matplotlib.patches.Rectangle`)
        around sources, if `add_boxes == True`.

    Returns
    -------
    fig: matplotlib.Figure
        The figure that is generated based on the parameters.
    """
    if box_kwargs is None:
        box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    panels = sum((show_model, show_observed, show_rendered, show_residual))
    if linear:
        if figsize is None:
            figsize = (panel_size * panels, panel_size)
        fig, ax = plt.subplots(1, panels, figsize=figsize)
    else:
        columns = int(np.ceil(panels / 2))
        if figsize is None:
            figsize = (panel_size * columns, panel_size * 2)
        fig = plt.figure(figsize=figsize)
        ax = [fig.add_subplot(2, columns, n + 1) for n in range(panels)]
    if not hasattr(ax, "__iter__"):
        ax = (ax,)

    observation = blend.observation
    sources = blend.sources
    model = blend.get_model(use_flux=use_flux)
    bbox = blend.bbox

    # Mask any pixels with zero weight in all bands
    if observation is not None:
        mask = np.sum(observation.weights, axis=0) == 0
        # if there are no masked pixels, do not use a mask
        if np.all(mask == 0):
            mask = None
    else:
        mask = None

    panel = 0
    if show_model:
        extent = get_extent(bbox)
        ax[panel].imshow(
            img_to_rgb(model.data, norm=norm, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Model")
        panel += 1

    if (show_rendered or show_residual) and not use_flux:
        model = observation.convolve(model)
    extent = get_extent(observation.bbox)

    if show_rendered:
        ax[panel].imshow(
            img_to_rgb(model.data, norm=norm, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Model Rendered")
        panel += 1

    if show_observed:
        ax[panel].imshow(
            img_to_rgb(
                observation.images.data, norm=norm, channel_map=channel_map, mask=mask
            ),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Observation")
        panel += 1

    if show_residual:
        residual = observation.images - model
        norm_ = LinearPercentileNorm(residual.data)
        ax[panel].imshow(
            img_to_rgb(residual.data, norm=norm_, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Residual")
        panel += 1

    for k, src in enumerate(sources):
        if add_boxes:
            panel = 0
            extent = get_extent(src.bbox)
            if show_model:
                rect = Rectangle(
                    (extent[0], extent[2]),
                    extent[1] - extent[0],
                    extent[3] - extent[2],
                    **box_kwargs,
                )
                ax[panel].add_artist(rect)
                panel = 1
            if observation is not None:
                for panel in range(panel, panels):
                    rect = Rectangle(
                        (extent[0], extent[2]),
                        extent[1] - extent[0],
                        extent[3] - extent[2],
                        **box_kwargs,
                    )
                    ax[panel].add_artist(rect)

        if add_labels and hasattr(src, "center") and src.center is not None:
            center = src.center
            panel = 0
            if show_model:
                ax[panel].text(*center[::-1], k, color="w", ha="center", va="center")
                panel = 1
            if observation is not None:
                for panel in range(panel, panels):
                    ax[panel].text(
                        *center[::-1], k, color="w", ha="center", va="center"
                    )

    fig.tight_layout()
    return fig


def get_extent(bbox: Box) -> tuple[int, int, int, int]:
    """Convert a `Box` into a list of bounds used in matplotlib

    Paramters
    ---------
    bbox: Box
       The box to convert into an extent list.
    """
    return bbox.start[-1], bbox.stop[-1], bbox.start[-2], bbox.stop[-2]


def show_sources(
    blend: Blend,
    sources: list[Source] = None,
    norm: Mapping = None,
    channel_map: np.ndarray = None,
    show_model: bool = True,
    show_observed: bool = False,
    show_rendered: bool = False,
    show_spectrum: bool = True,
    figsize: tuple[float, float] = None,
    model_mask: bool = True,
    add_markers: bool = True,
    add_boxes: bool = False,
    use_flux: bool = False,
) -> matplotlib.pyplot.Figure:
    """Plot individual source models

    The functions provides a fast way of evaluating the quality of
    individual sources.

    Parameters
    ----------
    blend: Blend
        The blend that contains the sources.
    sources: list[Source]
        The list of sources to plot.
        If `sources` is `None` then all of the sources in `blend` are
        displayed.
    norm: Mapping
        Norm to compress image intensity to the range [0,255].
    channel_map: np.ndarray
        Linear mapping with dimensions (3, channels).
    show_model: bool
        Whether the model is shown in the model frame.
    show_observed: bool
        Whether the observation is shown.
    show_rendered: bool
        Whether the model, rendered to match the observation, is shown.
    show_spectrum: bool
        Whether or not to show a plot for the spectrum of each component
        in each source.
    figsize: tuple[float, float]
        Size of the final figure.
    model_mask: bool
        Whether pixels with no flux in a model are masked.
    add_markers: bool
        Whether all of the sources are marked in each plot.
    add_boxes: bool
        Whether each source box is shown.
    use_flux: bool
        Whether to show the flux redistributed model (`source.flux`) or
        the model itself (`source.get_model()`) for each source.

    Returns
    -------
    fig: matplotlib.Figure
        The figure that is generated based on the parameters.
    """
    observation = blend.observation
    if sources is None:
        sources = blend.sources
    panels = sum((show_model, show_observed, show_rendered, show_spectrum))
    n_sources = len([src for src in sources if not src.is_null])
    if figsize is None:
        figsize = (panel_size * panels, panel_size * n_sources)

    fig, ax = plt.subplots(n_sources, panels, figsize=figsize, squeeze=False)

    marker_kwargs = {"mew": 1, "ms": 10}
    box_kwargs = {"facecolor": "none", "edgecolor": "w", "lw": 0.5}

    skipped = 0
    for k, src in enumerate(sources):
        if src.is_null:
            skipped += 1
            continue
        if use_flux:
            src_box = src.flux.bbox
        else:
            src_box = src.bbox

        extent = get_extent(src_box)

        # model in its bbox
        panel = 0
        model = src.get_model(use_flux=use_flux)

        if show_model:
            if model_mask:
                _model_mask = np.max(model.data, axis=0) <= 0
            else:
                _model_mask = None
            # Show the unrendered model in it's bbox
            ax[k - skipped][panel].imshow(
                img_to_rgb(
                    model.data, norm=norm, channel_map=channel_map, mask=_model_mask
                ),
                extent=extent,
                origin="lower",
            )
            ax[k - skipped][panel].set_title("Model Source {}".format(k))
            _add_markers(
                src,
                extent,
                ax[k - skipped][panel],
                add_markers,
                False,
                marker_kwargs,
                box_kwargs,
            )
            panel += 1

        # model in observation frame
        if show_rendered:
            # Center and show the rendered model
            model_ = Image(np.zeros(observation.shape), bands=observation.bands)
            model_.insert(src.get_model(use_flux=use_flux))
            if not use_flux:
                model_ = observation.convolve(model_)
            ax[k - skipped][panel].imshow(
                img_to_rgb(model_.data, norm=norm, channel_map=channel_map),
                extent=get_extent(observation.bbox),
                origin="lower",
            )
            ax[k - skipped][panel].set_title("Model Source {} Rendered".format(k))
            _add_markers(
                src,
                extent,
                ax[k - skipped][panel],
                add_markers,
                add_boxes,
                marker_kwargs,
                box_kwargs,
            )
            panel += 1

        if show_observed:
            # Center the observation on the source and display it
            _images = observation.images
            ax[k - skipped][panel].imshow(
                img_to_rgb(_images.data, norm=norm, channel_map=channel_map),
                extent=get_extent(observation.bbox),
                origin="lower",
            )
            ax[k - skipped][panel].set_title(f"Observation {k}")
            _add_markers(
                src,
                extent,
                ax[k - skipped][panel],
                add_markers,
                add_boxes,
                marker_kwargs,
                box_kwargs,
            )
            panel += 1

        if show_spectrum:
            spectra = [model.sum(axis=(1, 2))]

            for spectrum in spectra:
                ax[k - skipped][panel].plot(spectrum)
            ax[k - skipped][panel].set_xticks(range(len(spectra)))
            ax[k - skipped][panel].set_title("Spectrum")
            ax[k - skipped][panel].set_xlabel("Band")
            ax[k - skipped][panel].set_ylabel("Intensity")

    fig.tight_layout()
    return fig


def compare_spectra(
    use_flux: bool = True, use_template: bool = True, **all_sources: list[Source]
) -> matplotlib.pyplot.Figure:
    """Compare spectra from multiple different deblending results of the
    same sources.

    Parameters
    ----------
    use_flux: bool
        Whether or not to show the re-distributed flux version of the model.
    use_template: bool
        Whether or not to show the scarlet model templates.
    all_sources: dict[str, list[Source]]
        The list of sources for each different deblending model.
    """
    first_key = next(iter(all_sources.keys()))
    nbr_sources = len(all_sources[first_key])
    for key, sources in all_sources.items():
        if len(sources) != nbr_sources:
            msg = (
                "All source lists must have the same number of components."
                f"Received {nbr_sources} sources for the list {first_key} and {len(sources)}"
                f"for list {key}."
            )
            raise ValueError(msg)

    columns = 4
    rows = int(np.ceil(nbr_sources / columns))
    fig, ax = plt.subplots(rows, columns, figsize=(15, 15 * rows / columns))
    if rows == 1:
        ax = [ax[0], ax[1]]

    panel = 0
    for k in range(nbr_sources):
        row = panel // 4
        column = panel - row * 4
        ax[row][column].set_title(f"source {k}")
        for key, sources in all_sources.items():
            if sources[k].is_null:
                continue
            if use_template or not hasattr(sources[k], "flux"):
                spectrum = np.sum(sources[k].get_model(), axis=(1, 2))
                ax[row][column].plot(spectrum, ".-", label=key + " model")
            if use_flux and hasattr(sources[k], "flux"):
                spectrum = np.sum(sources[k].get_model(use_flux=True), axis=(1, 2))
                ax[row][column].plot(spectrum, ".--", label=key + " flux")
        panel += 1
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    return fig
