from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from lsst.scarlet.lite import Blend, Box, Component, FactorizedComponent, Image, Observation, Source
from lsst.scarlet.lite.parameters import FixedParameter
from numpy.typing import DTypeLike

__all__ = [
    "ScarletComponentData",
    "ScarletFactorizedComponentData",
    "ScarletSourceData",
    "ScarletBlendData",
    "ScarletModelData",
    "ComponentCube",
    "data_to_scarlet",
    "scarlet_to_data",
    "DummyObservation",
]

logger = logging.getLogger(__name__)


@dataclass
class ScarletComponentData:
    """Data for a component expressed as a 3D data cube

    This is used for scarlet component models that are not factorized,
    storing their entire model as a 3D data cube (bands, y, x).

    Attributes
    ----------
    origin:
        The lower bound of the components bounding box.
    peak:
        The peak of the component.
    model:
        The model for the component.
    """

    origin: tuple[int, int]
    peak: tuple[float, float]
    model: np.ndarray

    @property
    def shape(self):
        return self.model.shape

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result:
            The object encoded as a JSON compatible dict
        """
        return {
            "origin": self.origin,
            "shape": self.model.shape,
            "peak": self.peak,
            "model": tuple(self.model.flatten().astype(float)),
        }

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletComponentData:
        """Reconstruct `ScarletComponentData` from JSON compatible dict

        Parameters
        ----------
        data:
            Dictionary representation of the object
        dtype:
            Datatype of the resulting model.

        Returns
        -------
        result:
            The reconstructed object
        """
        data_shallow_copy = dict(data)
        data_shallow_copy["origin"] = tuple(data["origin"])
        shape = tuple(data_shallow_copy.pop("shape"))
        data_shallow_copy["model"] = np.array(data["model"]).reshape(shape).astype(dtype)
        return cls(**data_shallow_copy)


@dataclass
class ScarletFactorizedComponentData:
    """Data for a factorized component

    Attributes
    ----------
    origin:
        The lower bound of the component's bounding box.
    peak:
        The ``(y, x)`` peak of the component.
    spectrum:
        The SED of the component.
    morph:
        The 2D morphology of the component.
    """

    origin: tuple[int, int]
    peak: tuple[float, float]
    spectrum: np.ndarray
    morph: np.ndarray

    @property
    def shape(self):
        return self.morph.shape

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result:
            The object encoded as a JSON compatible dict
        """
        return {
            "origin": tuple(int(o) for o in self.origin),
            "shape": tuple(int(s) for s in self.morph.shape),
            "peak": tuple(int(p) for p in self.peak),
            "spectrum": tuple(self.spectrum.astype(float)),
            "morph": tuple(self.morph.flatten().astype(float)),
        }

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletFactorizedComponentData:
        """Reconstruct `ScarletFactorizedComponentData` from JSON compatible
        dict.

        Parameters
        ----------
        data:
            Dictionary representation of the object
        dtype:
            Datatype of the resulting model.

        Returns
        -------
        result:
            The reconstructed object
        """
        data_shallow_copy = dict(data)
        data_shallow_copy["origin"] = tuple(data["origin"])
        shape = tuple(data_shallow_copy.pop("shape"))
        data_shallow_copy["spectrum"] = np.array(data["spectrum"]).astype(dtype)
        data_shallow_copy["morph"] = np.array(data["morph"]).reshape(shape).astype(dtype)
        return cls(**data_shallow_copy)


@dataclass
class ScarletSourceData:
    """Data for a scarlet source

    Attributes
    ----------
    components:
        The components contained in the source that are not factorized.
    factorized_components:
        The components contained in the source that are factorized.
    peak_id:
        The peak ID of the source in it's parent's footprint peak catalog.
    """

    components: list[ScarletComponentData]
    factorized_components: list[ScarletFactorizedComponentData]
    peak_id: int

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result:
            The object encoded as a JSON compatible dict
        """
        result = {
            "components": [component.as_dict() for component in self.components],
            "factorized": [component.as_dict() for component in self.factorized_components],
            "peak_id": self.peak_id,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletSourceData:
        """Reconstruct `ScarletSourceData` from JSON compatible
        dict.

        Parameters
        ----------
        data:
            Dictionary representation of the object
        dtype:
            Datatype of the resulting model.

        Returns
        -------
        result:
            The reconstructed object
        """
        data_shallow_copy = dict(data)
        del data_shallow_copy["factorized"]
        components = []
        for component in data["components"]:
            component = ScarletComponentData.from_dict(component, dtype=dtype)
            components.append(component)
        data_shallow_copy["components"] = components

        factorized = []
        for component in data["factorized"]:
            component = ScarletFactorizedComponentData.from_dict(component, dtype=dtype)
            factorized.append(component)
        data_shallow_copy["factorized_components"] = factorized
        data_shallow_copy["peak_id"] = int(data["peak_id"])
        return cls(**data_shallow_copy)


@dataclass
class ScarletBlendData:
    """Data for an entire blend.

    Attributes
    ----------
    origin:
        The lower bound of the blend's bounding box.
    shape:
        The shape of the blend's bounding box.
    sources:
        Data for the sources contained in the blend.
    psf_center:
        The location used for the center of the PSF for
        the blend.
    """

    origin: tuple[int, int]
    shape: tuple[int, int]
    sources: dict[int, ScarletSourceData]
    psf_center: tuple[float, float]
    psf: np.ndarray

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result:
            The object encoded as a JSON compatible dict
        """
        result = {
            "origin": self.origin,
            "shape": self.shape,
            "psf_center": self.psf_center,
            "psf_shape": self.psf.shape,
            "psf": tuple(self.psf.flatten().astype(float)),
            "sources": {bid: source.as_dict() for bid, source in self.sources.items()},
        }
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletBlendData:
        """Reconstruct `ScarletBlendData` from JSON compatible
        dict.

        Parameters
        ----------
        data:
            Dictionary representation of the object
        dtype:
            Datatype of the resulting model.

        Returns
        -------
        result:
            The reconstructed object
        """
        data_shallow_copy = dict(data)
        data_shallow_copy["origin"] = tuple(data["origin"])
        data_shallow_copy["shape"] = tuple(data["shape"])
        psf_shape = data_shallow_copy.pop("psf_shape")
        data_shallow_copy["psf_center"] = tuple(data["psf_center"])
        data_shallow_copy["psf"] = np.array(data["psf"]).reshape(psf_shape).astype(dtype)
        data_shallow_copy["sources"] = {
            int(bid): ScarletSourceData.from_dict(source, dtype=dtype)
            for bid, source in data["sources"].items()
        }
        return cls(**data_shallow_copy)


class ScarletModelData:
    """A container that propagates scarlet models for an entire catalog."""

    def __init__(self, bands: list[Any], psf: np.ndarray, blends: dict[int, ScarletBlendData] | None = None):
        """Initialize an instance

        Parameters
        ----------
        bands:
            The names of the bands.
            The order of the bands must be the same as the order of
            the multiband model arrays, and SEDs.
        psf:
            The 2D array of the PSF in scarlet model space.
            This is typically a narrow Gaussian integrated over the
            pixels in the exposure.
        blends:
            Map from parent IDs in the source catalog
            to scarlet model data for each parent ID (blend).
        """
        self.bands = bands
        self.psf = psf
        if blends is None:
            blends = {}
        self.blends = blends

    def json(self) -> str:
        """Serialize the data model to a JSON formatted string

        Returns
        -------
        result : `str`
            The result of the object converted into a JSON format
        """
        result = {
            "bands": self.bands,
            "psfShape": self.psf.shape,
            "psf": list(self.psf.flatten()),
            "blends": {bid: blend.as_dict() for bid, blend in self.blends.items()},
        }
        return json.dumps(result)

    @classmethod
    def parse_obj(cls, data: dict) -> ScarletModelData:
        """Construct a ScarletModelData from python decoded JSON object.

        Parameters
        ----------
        data:
            The result of json.load(s) on a JSON persisted ScarletModelData

        Returns
        -------
        result:
            The `ScarletModelData` that was loaded the from the input object
        """
        data_shallow_copy = dict(data)
        model_psf = (
            np.array(data_shallow_copy["psf"]).reshape(data_shallow_copy.pop("psfShape")).astype(np.float32)
        )
        data_shallow_copy["psf"] = model_psf
        data_shallow_copy["blends"] = {
            int(bid): ScarletBlendData.from_dict(blend) for bid, blend in data["blends"].items()
        }
        return cls(**data_shallow_copy)


class ComponentCube(Component):
    """Dummy component for a component cube.

    This is duck-typed to a `lsst.scarlet.lite.Component` in order to
    generate a model from the component.

    If scarlet lite ever implements a component as a data cube,
    this class can be removed.
    """

    def __init__(self, bands: tuple[Any, ...], bbox: Box, model: Image, center: tuple[int, int]):
        """Initialization

        Parameters
        ----------
        bands:
        model:
            The 3D (bands, y, x) model of the component.
        center:
            The `(y, x)` center of the component.
        bbox:
            The bounding box of the component.
        """
        super().__init__(bands, bbox)
        self._model = model
        self.center = center

    def get_model(self) -> Image:
        """Generate the model for the source

        Returns
        -------
        model:
            The model as a 3D `(band, y, x)` array.
        """
        return self._model

    def resize(self, model_box: Box) -> bool:
        """Test whether or not the component needs to be resized"""
        return False

    def update(self, it: int, input_grad: np.ndarray) -> None:
        """Implementation of unused abstract method"""

    def parameterize(self, parameterization: Callable) -> None:
        """Implementation of unused abstract method"""


class DummyObservation(Observation):
    """An observation that does not have any image data

    In order to reproduce a model in an observed seeing we make use of the
    scarlet lite `Observation` class, but since we are not fitting the model
    to data we can use empty arrays for the image, variance, and weight data,
    and zero for the `noise_rms`.

    Parameters
    ----------
    psfs:
        The array of PSF images in each band
    model_psf:
        The image of the model PSF.
    bbox:
        The bounding box of the full observation.
    dtype:
        The data type of the model that is generated.
    """

    def __init__(
        self, bands: tuple[Any], psfs: np.ndarray, model_psf: np.ndarray, bbox: Box, dtype: DTypeLike
    ):
        dummy_image = np.zeros((len(bands),) + bbox.shape, dtype=dtype)

        super().__init__(
            images=dummy_image,
            variance=dummy_image,
            weights=dummy_image,
            psfs=psfs,
            model_psf=model_psf,
            noise_rms=np.zeros((len(bands),), dtype=dtype),
            bbox=bbox,
            bands=bands,
            convolution_mode="real",
        )


def minimal_data_to_scarlet(
    bands: tuple[Any], model_psf: np.ndarray, blend_data: ScarletBlendData, dtype: DTypeLike
) -> Blend:
    """Convert the storage data model into a scarlet lite blend

    Parameters
    ----------
    bands:
        The bands from the observations, in order.
    model_psf:
        PSF in model space (usually a nyquist sampled circular Gaussian).
    blend_data:
        Persistable data for the entire blend.
    dtype:
        The data type of the model that is generated.

    Returns
    -------
    blend:
        A scarlet blend model extracted from persisted data.
    """
    model_box = Box(blend_data.shape, origin=(0, 0))
    observation = DummyObservation(
        bands=bands,
        psfs=blend_data.psf,
        model_psf=model_psf,
        bbox=model_box,
        dtype=dtype,
    )
    return data_to_scarlet(blend_data, observation)


def data_to_scarlet(blend_data: ScarletBlendData, observation: Observation) -> Blend:
    """Convert the storage data model into a scarlet lite blend

    Parameters
    ----------
    blend_data:
        Persistable data for the entire blend.
    observation:
        The observation that contains the blend.
        If `observation` is ``None`` then a `DummyObservation` containing
        no image data is initialized.

    Returns
    -------
    blend:
        A scarlet blend model extracted from persisted data.
    """
    sources = []
    for source_id, source_data in blend_data.sources.items():
        components: list[Component] = []
        for component_data in source_data.components:
            bbox = Box(component_data.shape, origin=component_data.origin)
            model = component_data.model
            component = ComponentCube(
                bands=observation.bands,
                bbox=bbox,
                model=Image(model, yx0=bbox.origin, bands=observation.bands),  # type: ignore
                center=(int(np.round(component_data.peak[0])), int(np.round(component_data.peak[0]))),
            )
            components.append(component)
        for factorized_data in source_data.factorized_components:
            bbox = Box(factorized_data.shape, origin=factorized_data.origin)
            # Add dummy values for properties only needed for
            # model fitting.
            spectrum = FixedParameter(factorized_data.spectrum)
            morph = FixedParameter(factorized_data.morph)
            # Note: since we aren't fitting a model, we don't need to
            # set the RMS of the background.
            # We set it to NaN just to be safe.
            factorized = FactorizedComponent(
                bands=observation.bands,
                spectrum=spectrum,
                morph=morph,
                peak=tuple(int(np.round(p)) for p in factorized_data.peak),  # type: ignore
                bbox=bbox,
                bg_rms=np.full((len(observation.bands),), np.nan),
            )
            components.append(factorized)

        source = Source(components=components)
        # Store identifiers for the source
        source.record_id = source_id  # type: ignore
        source.peak_id = source_data.peak_id  # type: ignore
        sources.append(source)

    return Blend(sources=sources, observation=observation)


def scarlet_to_data(blend: Blend, psf_center: tuple[int, int]) -> ScarletBlendData:
    """Convert a scarlet lite blend into a persistable data object

    Parameters
    ----------
    blend:
        The blend that is being persisted.
    psf_center:
        The center of the PSF.

    Returns
    -------
    blend_data:
        The data model for a single blend.
    """
    sources = {}
    for source in blend.sources:
        components = []
        factorized = []
        for component in source.components:
            if isinstance(component, FactorizedComponent):
                factorized_data = ScarletFactorizedComponentData(
                    origin=component.bbox.origin,  # type: ignore
                    peak=component.peak,  # type: ignore
                    spectrum=component.spectrum,
                    morph=component.morph,
                )
                factorized.append(factorized_data)
            else:
                component_data = ScarletComponentData(
                    origin=component.bbox.origin,  # type: ignore
                    peak=component.peak,  # type: ignore
                    model=component.get_model().data,
                )
                components.append(component_data)
        source_data = ScarletSourceData(
            components=components,
            factorized_components=factorized,
            peak_id=source.peak_id,  # type: ignore
        )
        sources[source.record_id] = source_data  # type: ignore

    blend_data = ScarletBlendData(
        origin=blend.bbox.origin,  # type: ignore
        shape=blend.bbox.shape,  # type: ignore
        sources=sources,
        psf_center=psf_center,
        psf=blend.observation.psfs,
    )

    return blend_data
