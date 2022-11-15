from astropy.visualization.lupton_rgb import LinearMapping, AsinhMapping
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


def channels_to_rgb(channels):
    """Get the linear mapping of multiple channels to RGB channels
    The mapping created here assumes the channels are ordered in wavelength
    direction, starting with the shortest wavelength. The mapping seeks to produce
    a relatively even weights for across all channels. It does not consider e.g.
    signal-to-noise variations across channels or human perception.
    Parameters
    ----------
    channels: int in range(0,7)
        Number of channels
    Returns
    -------
    array (3, channels) to map onto RGB
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
        channel_map[:, 6] = 2/3.
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
    def __init__(self, img, percentiles=[1, 99]):
        """Create norm that is linear between lower and upper percentile of img
        Parameters
        ----------
        img: array_like
            Image to normalize
        percentile: array_like, default=[1,99]
            Lower and upper percentile to consider. Pixel values below will be
            set to zero, above to saturated.
        """
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        super().__init__(minimum=vmin, maximum=vmax)


class AsinhPercentileNorm(AsinhMapping):
    def __init__(self, img, percentiles=[1, 99]):
        """Create norm that is linear between lower and upper percentile of img
        Parameters
        ----------
        img: array_like
            Image to normalize
        percentile: array_like, default=[1,99]
            Lower and upper percentile to consider. Pixel values below will be
            set to zero, above to saturated.
        """
        assert len(percentiles) == 2
        vmin, vmax = np.percentile(img, percentiles)
        # solution for beta assumes flat spectrum at vmax
        stretch = vmax - vmin
        beta = stretch / np.sinh(1)
        super().__init__(minimum=vmin, stretch=stretch, Q=beta)


def img_to_3channel(img, channel_map=None, fill_value=0):
    """Convert multi-band image cube into 3 RGB channels
    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (channels, height, width).
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
    fill_value: float, default=`0`
        Value to use for any masked pixels.
    Returns
    -------
    RGB: numpy array with dtype float
    """
    # expand single img into cube
    assert len(img.shape) in [2, 3]
    if len(img.shape) == 2:
        ny, nx = img.shape
        img_ = img.reshape(1, ny, nx)
    elif len(img.shape) == 3:
        img_ = img
    C = len(img_)

    # filterWeights: channel x band
    if channel_map is None:
        channel_map = channels_to_rgb(C)
    else:
        assert channel_map.shape == (3, len(img))

    # map channels onto RGB channels
    _, ny, nx = img_.shape
    rgb = np.dot(channel_map, img_.reshape(C, -1)).reshape(3, ny, nx)

    if hasattr(rgb, "mask"):
        rgb = rgb.filled(fill_value)

    return rgb


def img_to_rgb(img, channel_map=None, fill_value=0, norm=None, mask=None):
    """Convert images to normalized RGB.
    If normalized values are outside of the range [0..255], they will be
    truncated such as to preserve the corresponding color.
    Parameters
    ----------
    img: array_like
        This should be an array with dimensions (channels, height, width).
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
    fill_value: float, default=`0`
        Value to use for any masked pixels.
    norm: `scarlet.display.Norm`, default `None`
        Norm to use for mapping in the allowed range [0..255]. If `norm=None`,
        `scarlet.display.LinearPercentileNorm` will be used.
    mask: array_like
        A [0,1] binary mask to apply over the top of the image,
        where pixels with mask==1 are masked out.
    Returns
    -------
    rgb: numpy array with dimensions (3, height, width) and dtype uint8
    """
    RGB = img_to_3channel(img, channel_map=channel_map)
    if norm is None:
        norm = LinearMapping(image=RGB)
    rgb = norm.make_rgb_image(*RGB)
    if mask is not None:
        rgb = np.dstack([rgb, ~mask * 255])
    return rgb


panel_size = 4.0


def show_likelihood(blend, figsize=None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(blend.log_likelihood, **kwargs)
    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("log-Likelihood")
    return fig


def _add_markers(src, extent, ax, add_markers, add_boxes, marker_kwargs, box_kwargs):
    if add_markers and hasattr(src, "center") and src.center is not None:
        center = np.array(src.center)[::-1]
        ax.plot(*center, "wx", **marker_kwargs)

    if add_boxes:

        rect = Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            **box_kwargs
        )
        ax.add_artist(rect)


def show_scene(
    blend,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_residual=False,
    add_labels=True,
    add_boxes=False,
    figsize=None,
    linear=True,
    use_flux=False,
    box_kwargs=None,
):
    """Plot all sources to recreate the scence.
    The functions provides a fast way of evaluating the quality of the entire model,
    i.e. the combination of all scences that seek to fit the observation.
    Parameters
    ----------
    blend: `~scarlet.lite.models.LiteBlend`
    observation: `~scarlet.Observation`
    norm: norm to compress image intensity to the range [0,255]
    channel_map: array_like
        Linear mapping with dimensions (3, channels)
    show_model: bool
        Whether the model is shown in the model frame
    show_observed: bool
        Whether the observation is shown
    show_rendered: bool
        Whether the model, rendered to match the observation, is shown
    show_residual: bool
        Whether the residuals between rendered model and observation is shown
    add_label: bool
        Whether each source is labeled with its numerical index in the source list
    add_boxes: bool
        Whether each source box is shown
    figsize: matplotlib figsize argument
    linear: bool
        Whether or not to display the scene in a single line (`True`) or
        on multiple lines (`False`).
    use_flux: bool
        Whether to show the flux redistributed model (`source.flux`) or
        the model itself (`source.get_model()`) for each source.
    Returns
    -------
    matplotlib figure
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

    panel = 0
    if show_model:
        extent = get_extent(bbox)
        ax[panel].imshow(
            img_to_rgb(model, norm=norm, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Model")
        panel += 1

    if (show_rendered or show_residual) and not use_flux:
        model = observation.render(model)
    extent = get_extent(observation.bbox)

    if show_rendered:
        ax[panel].imshow(
            img_to_rgb(model, norm=norm, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Model Rendered")
        panel += 1

    if show_observed:
        ax[panel].imshow(
            img_to_rgb(observation.data, norm=norm, channel_map=channel_map, mask=mask),
            extent=extent,
            origin="lower",
        )
        ax[panel].set_title("Observation")
        panel += 1

    if show_residual:
        residual = observation.data - model
        norm_ = LinearPercentileNorm(residual)
        ax[panel].imshow(
            img_to_rgb(residual, norm=norm_, channel_map=channel_map, mask=mask),
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
                    **box_kwargs
                )
                ax[panel].add_artist(rect)
                panel = 1
            if observation is not None:
                start, stop = src.bbox.start[-2:][::-1], src.bbox.stop[-2:][::-1]
                points = (start, (start[0], stop[1]), stop, (stop[0], start[1]))
                for panel in range(panel, panels):
                    rect = Rectangle(
                        (extent[0], extent[2]),
                        extent[1] - extent[0],
                        extent[3] - extent[2],
                        **box_kwargs
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
                    ax[panel].text(*center[::-1], k, color="w", ha="center", va="center")

    fig.tight_layout()
    return fig


def get_extent(bbox):
    return [bbox.start[-1], bbox.stop[-1], bbox.start[-2], bbox.stop[-2]]


def show_sources(
    blend,
    sources=None,
    norm=None,
    channel_map=None,
    show_model=True,
    show_observed=False,
    show_rendered=False,
    show_spectrum=True,
    figsize=None,
    model_mask=True,
    add_markers=True,
    add_boxes=False,
    use_flux=False,
):
    observation = blend.observation
    if sources is None:
        sources = blend.sources
    panels = sum((show_model, show_observed, show_rendered, show_spectrum))
    n_sources = len([src for src in sources if not src.is_null])
    bbox = observation.bbox
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
            src_box = src.flux_box
        else:
            src_box = src.bbox

        extent = get_extent(src_box)

        # model in its bbox
        panel = 0
        model = src.get_model(use_flux=use_flux)

        if show_model:
            if model_mask:
                _model_mask = np.max(model, axis=0) <= 0
            # Show the unrendered model in it's bbox
            ax[k-skipped][panel].imshow(
                img_to_rgb(model, norm=norm, channel_map=channel_map, mask=_model_mask),
                extent=extent,
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Model Source {}".format(k))
            _add_markers(src, extent, ax[k-skipped][panel], add_markers, False, marker_kwargs, box_kwargs)
            panel += 1

        # model in observation frame
        if show_rendered:
            # Center and show the rendered model
            model_ = src.get_model(use_flux=use_flux, bbox=bbox)
            if not use_flux:
                model_ = observation.render(model_)
            ax[k-skipped][panel].imshow(
                img_to_rgb(model_, norm=norm, channel_map=channel_map),
                extent=get_extent(observation.bbox),
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Model Source {} Rendered".format(k))
            _add_markers(src, extent, ax[k-skipped][panel], add_markers, add_boxes, marker_kwargs, box_kwargs)
            panel += 1

        if show_observed:
            # Center the observation on the source and display it
            _images = observation.data
            ax[k-skipped][panel].imshow(
                img_to_rgb(_images, norm=norm, channel_map=channel_map),
                extent=get_extent(observation.bbox),
                origin="lower",
            )
            ax[k-skipped][panel].set_title("Observation".format(k))
            _add_markers(src, extent, ax[k-skipped][panel], add_markers, add_boxes, marker_kwargs, box_kwargs)
            panel += 1

        if show_spectrum:
            spectra = [model.sum(axis=(1, 2))]

            for spectrum in spectra:
                ax[k-skipped][panel].plot(spectrum)
            ax[k-skipped][panel].set_xticks(range(len(spectrum)))
            ax[k-skipped][panel].set_title("Spectrum")
            ax[k-skipped][panel].set_xlabel("Band")
            ax[k-skipped][panel].set_ylabel("Intensity")

    fig.tight_layout()
    return fig


def compare_spectra(use_flux=True, use_template=True, **all_sources):
    """Compare spectra from multiple different deblending results of the same sources.

    Parameters
    ----------
    use_flux: `bool`
        Whether or not to show the re-distributed flux version of the model.
    use_template: `bool`
        Whether or not to show the scarlet model templates.
    all_sources: `dict` of (`str, list)`
        The list of sources for each different deblending model.
    """
    first_key = next(iter(all_sources.keys()))
    K = len(all_sources[first_key])
    for key, sources in all_sources.items():
        if len(sources) != K:
            msg = (
                "All source lists must have the same number of components."
                f"Received {K} sources for the list {first_key} and {len(sources)}"
                f"for list {key}."
            )
            raise ValueError(msg)

    columns = 4
    rows = int(np.ceil(K / columns))
    fig, ax = plt.subplots(rows, columns, figsize=(15, 15*rows/columns))
    if rows == 1:
        ax = [ax[0], ax[1]]

    panel = 0
    for k in range(K):
        row = panel // 4
        column = panel - row*4
        ax[row][column].set_title(f"source {k}")
        for key, sources in all_sources.items():
            if sources[k].is_null:
                continue
            if use_template or not hasattr(sources[k], "flux"):
                sed = np.sum(sources[k].get_model(), axis=(1,2))
                ax[row][column].plot(sed, ".-", label=key+" model")
            if use_flux and hasattr(sources[k], "flux"):
                sed = np.sum(sources[k].get_model(use_flux=True), axis=(1,2))
                ax[row][column].plot(sed, ".--", label=key+" flux")
        panel += 1
    handles, labels = ax[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    plt.show()