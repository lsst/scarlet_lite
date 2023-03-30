from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from lsst.scarlet.lite import (
    Blend,
    Box,
    Component,
    FactorizedComponent,
    FixedParameter,
    Image,
    Observation,
    Source,
)
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
    shape:
        The shape of the bounding box.
    center:
        The center of the component.
    model:
        The model for the component.
    """

    origin: tuple[int, int]
    center: tuple[float, float]
    model: np.ndarray

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
            "center": self.center,
            "model": tuple(self.model.flatten().astype(float)),
        }

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletComponentData:
        """Reconstruct `ScarletComponentData` from JSON compatible dict

        Parameters
        ----------
        data:
            Dictionary representation of the object

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
    shape:
        The shape of the component's bounding box.
    center:
        The ``(y, x)`` center of the component.
    sed:
        The SED of the component.
    morph:
        The 2D morphology of the component.
    """

    origin: tuple[int, int]
    center: tuple[float, float]
    sed: np.ndarray
    morph: np.ndarray

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result:
            The object encoded as a JSON compatible dict
        """
        return {
            "origin": self.origin,
            "shape": self.morph.shape,
            "center": self.center,
            "sed": tuple(self.sed.astype(float)),
            "morph": tuple(self.morph.flatten().astype(float)),
        }

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletFactorizedComponentData:
        """Reconstruct `ScarletFactorizedComponentData` from JSON compatible
        dict.

        Parameters
        ----------
        data : `dict`
            Dictionary representation of the object

        Returns
        -------
        result : `ScarletFactorizedComponentData`
            The reconstructed object
        """
        data_shallow_copy = dict(data)
        data_shallow_copy["origin"] = tuple(data["origin"])
        shape = tuple(data_shallow_copy.pop("shape"))
        data_shallow_copy["sed"] = np.array(data["sed"]).astype(dtype)
        data_shallow_copy["morph"] = np.array(data["morph"]).reshape(shape).astype(dtype)
        return cls(**data_shallow_copy)


@dataclass
class ScarletSourceData:
    """Data for a scarlet source

    Attributes
    ----------
    components : `list` of `ScarletComponentData`
        The components contained in the source that are not factorized.
    factorized_components : `list` of `ScarletFactorizedComponentData`
        The components contained in the source that are factorized.
    peak_id : `int`
        The peak ID of the source in it's parent's footprint peak catalog.
    """

    components: list[ScarletComponentData]
    factorized_components: list[ScarletFactorizedComponentData]
    peak_id: int

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result : `dict`
            The object encoded as a JSON compatible dict
        """
        result = {
            "components": [],
            "factorized": [],
            "peak_id": self.peak_id,
        }
        for component in self.components:
            reduced = component.asDict()
            result["components"].append(reduced)

        for component in self.factorized_components:
            reduced = component.asDict()
            result["factorized"].append(reduced)
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> "ScarletSourceData":
        """Reconstruct `ScarletSourceData` from JSON compatible
        dict.

        Parameters
        ----------
        data : `dict`
            Dictionary representation of the object

        Returns
        -------
        result : `ScarletSourceData`
            The reconstructed object
        """
        data_shallow_copy = dict(data)
        del data_shallow_copy["factorized"]
        components = []
        for component in data["components"]:
            component = ScarletComponentData.fromDict(component, dtype=dtype)
            components.append(component)
        data_shallow_copy["components"] = components

        factorized = []
        for component in data["factorized"]:
            component = ScarletFactorizedComponentData.fromDict(component, dtype=dtype)
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

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result : `dict`
            The object encoded as a JSON compatible dict
        """
        result: dict[str, Any] = {"origin": self.origin, "shape": self.shape, "psf_center": self.psf_center}
        result["sources"] = {id: source.asDict() for id, source in self.sources.items()}
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletBlendData:
        """Reconstruct `ScarletBlendData` from JSON compatible
        dict.

        Parameters
        ----------
        data:
            Dictionary representation of the object

        Returns
        -------
        result:
            The reconstructed object
        """
        data_shallow_copy = dict(data)
        data_shallow_copy["origin"] = tuple(data["origin"])
        data_shallow_copy["shape"] = tuple(data["shape"])
        data_shallow_copy["psf_center"] = tuple(data["psf_center"])
        data_shallow_copy["sources"] = {
            int(id): ScarletSourceData.fromDict(source, dtype=dtype) for id, source in data["sources"].items()
        }
        return cls(**data_shallow_copy)


class ScarletModelData:
    """A container that propagates scarlet models for an entire catalog."""

    def __init__(self, bands: list[Any], psf: list[np.ndarray], blends: dict[int, ScarletBlendData] = None):
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
            "blends": {id: blend.asDict() for id, blend in self.blends.items()},
        }
        return json.dumps(result)

    @classmethod
    def parse_obj(cls, data: dict) -> ScarletModelData:
        """Construct a ScarletModelData from python decoded JSON object.

        Parameters
        ----------
        inMemoryDataset : `Mapping`
            The result of json.load(s) on a JSON persisted ScarletModelData

        Returns
        -------
        result : `ScarletModelData`
            The `ScarletModelData` that was loaded the from the input object
        """
        data_shallow_copy = dict(data)
        model_psf = (
            np.array(data_shallow_copy["psf"]).reshape(data_shallow_copy.pop("psfShape")).astype(np.float32)
        )
        data_shallow_copy["psf"] = model_psf
        data_shallow_copy["blends"] = {
            int(id): ScarletBlendData.fromDict(blend) for id, blend in data["blends"].items()
        }
        return cls(**data_shallow_copy)


class ComponentCube(Component):
    """Dummy component for a component cube.

    This is duck-typed to a `lsst.scarlet.lite.Component` in order to
    generate a model from the component.

    If scarlet lite ever implements a component as a data cube,
    this class can be removed.
    """

    def __init__(self, bands: tuple[Any], bbox: Box, model: Image, center: tuple[int, int]):
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
    psf_model:
        The image of the model PSF.
    bbox:
        The bounding box of the full observation.
    dtype:
        The data type of the model that is generated.
    """

    def __init__(
        self, bands: list[Any], psfs: np.ndarray, model_psf: np.ndarray, bbox: Box, dtype: DTypeLike
    ):
        dummy_image = np.zeros([], dtype=dtype)

        super().__init__(
            images=dummy_image,
            variance=dummy_image,
            weights=dummy_image,
            psfs=psfs,
            model_psf=model_psf,
            noise_rms=0,
            bbox=bbox,
            bands=bands,
            convolution_mode="real",
        )


def data_to_scarlet(bands: list[Any], blend_data: ScarletBlendData, dtype=np.float32) -> Blend:
    """Convert the storage data model into a scarlet lite blend

    Parameters
    ----------
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
    sources = []
    for source_id, source_data in blend_data.sources.items():
        components = []
        for component_data in source_data.components:
            bbox = Box(component_data.shape, origin=component_data.origin)
            model = component_data.model
            component = ComponentCube(
                bands=bands,
                bbox=bbox,
                model=model,
                center=tuple(component_data.center),
            )
            components.append(component)
        for component_data in source_data.factorized_components:
            bbox = Box(component_data.shape, origin=component_data.origin)
            # Add dummy values for properties only needed for
            # model fitting.
            sed = component_data.sed
            sed = FixedParameter(sed)
            morph = FixedParameter(component_data.morph)
            # Note: since we aren't fitting a model, we don't need to
            # set the RMS of the background.
            # We set it to NaN just to be safe.
            component = FactorizedComponent(
                sed=sed,
                morph=morph,
                center=tuple(component_data.center),
                bbox=bbox,
                model_bbox=model_box,
                bg_rms=np.nan,
            )
            components.append(component)

        source = Source(components=components, dtype=dtype)
        # Store identifiers for the source
        source.recordId = source_id
        source.peak_id = source_data.peak_id
        sources.append(source)

    return Blend(sources=sources, observation=None)


def scarlet_to_data(blend: Blend, psf_center: tuple[int, int], origin: tuple[int, int]) -> ScarletBlendData:
    """Convert a scarlet lite blend into a persistable data object

    Parameters
    ----------
    blend:
        The blend that is being persisted.
    psf_center:
        The center of the PSF.
    origin:
        The lower coordinate of the entire blend.

    Returns
    -------
    blend_data:
        The data model for a single blend.
    """
    sources = {}
    for source in blend.sources:
        components = []
        for component in source.components:
            if isinstance(component, FactorizedComponent):
                component_data = ScarletFactorizedComponentData(
                    origin=component.bbox.origin,
                    center=component.center,
                    sed=component.sed,
                    morph=component.morph,
                )
            else:
                component_data = ScarletComponentData(
                    origin=component.bbox.origin,
                    center=component.center,
                    model=component.get_model(),
                )
            components.append(component_data)
        source_data = ScarletSourceData(
            components=[],
            factorized_components=components,
            peak_id=source.peak_id,
        )
        sources[source.recordId] = source_data

    blend_data = ScarletBlendData(
        origin=blend.bbox.origin,
        shape=blend.bbox.shape,
        sources=sources,
        psf_center=psf_center,
    )

    return blend_data
