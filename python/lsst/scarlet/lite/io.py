from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import numpy as np
from numpy.typing import DTypeLike

from .bbox import Box
from .blend import Blend
from .component import Component, FactorizedComponent
from .image import Image
from .observation import Observation
from .parameters import FixedParameter
from .source import Source

__all__ = [
    "ScarletComponentData",
    "ScarletFactorizedComponentData",
    "ScarletSourceData",
    "ScarletBlendData",
    "ScarletModelData",
    "ComponentCube",
]

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ScarletComponentBaseData(ABC):
    """Base data for a scarlet component"""

    component_registry: ClassVar[dict[str, type[ScarletComponentBaseData]]] = {}
    component_type: str

    @classmethod
    def register(cls) -> None:
        """Register a new component type"""
        ScarletComponentBaseData.component_registry[cls.component_type] = cls

    @abstractmethod
    def to_component(self, observation: Observation) -> Component:
        """Convert the storage data model into a scarlet Component

        Parameters
        ----------
        observation :
            The observation that the component is associated with

        Returns
        -------
        component :
            A scarlet component extracted from persisted data.
        """

    @abstractmethod
    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """

    @staticmethod
    def from_dict(data: dict, dtype: DTypeLike | None = None) -> ScarletComponentBaseData:
        """Reconstruct `ScarletComponentBaseData` from JSON compatible
        dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object
        dtype :
            Datatype of the resulting model.

        Returns
        -------
        result :
            The reconstructed object
        """
        component_type = data["component_type"]
        cls = ScarletComponentBaseData.component_registry[component_type]
        return cls.from_dict(data, dtype=dtype)

    @staticmethod
    def from_component(component: Component) -> ScarletComponentBaseData:
        """Reconstruct `ScarletComponentBaseData` from a scarlet Component.

        Parameters
        ----------
        component :
            The scarlet component to be converted.

        Returns
        -------
        result :
            The reconstructed object
        """
        if isinstance(component, FactorizedComponent):
            return ScarletFactorizedComponentData._from_component(component)
        else:
            return ScarletComponentData._from_component(component)


@dataclass(kw_only=True)
class ScarletComponentData(ScarletComponentBaseData):
    """Data for a component expressed as a 3D data cube

    This is used for scarlet component models that are not factorized,
    storing their entire model as a 3D data cube (bands, y, x).

    Attributes
    ----------
    origin :
        The lower bound of the components bounding box.
    peak :
        The peak of the component.
    model :
        The model for the component.
    """

    origin: tuple[int, int]
    peak: tuple[float, float]
    model: np.ndarray
    component_type: str = "component"

    @property
    def shape(self):
        return self.model.shape[-2:]

    def to_component(self, observation: Observation) -> ComponentCube:
        """Convert the storage data model into a scarlet Component

        Parameters
        ----------
        observation :
            The observation that the component is associated with

        Returns
        -------
        component :
            A scarlet component extracted from persisted data.
        """
        bbox = Box(self.shape, origin=self.origin)
        model = self.model
        if self.peak is None:
            peak = None
        else:
            peak = (int(np.round(self.peak[0])), int(np.round(self.peak[0])))
        assert peak is not None
        component = ComponentCube(
            bands=observation.bands,
            bbox=bbox,
            model=Image(model, yx0=bbox.origin, bands=observation.bands),  # type: ignore
            peak=peak,
        )
        return component

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        return {
            "origin": self.origin,
            "shape": self.model.shape,
            "peak": self.peak,
            "model": tuple(self.model.flatten().astype(float)),
            "component_type": "component",
        }

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike | None = None) -> ScarletComponentData:
        """Reconstruct `ScarletComponentData` from JSON compatible dict

        Parameters
        ----------
        data :
            Dictionary representation of the object
        dtype :
            Datatype of the resulting model.

        Returns
        -------
        result :
            The reconstructed object
        """
        if data["component_type"] != "component":
            raise ValueError(f"Invalid component type: {data['component_type']}")
        shape = tuple(data["shape"])

        return cls(
            origin=tuple(data["origin"]),  # type: ignore
            peak=data["peak"],
            model=np.array(data["model"]).reshape(shape).astype(dtype),
        )

    @staticmethod
    def _from_component(component: Component) -> ScarletComponentData:
        """Reconstruct `ScarletComponentData` from a scarlet Component.

        Parameters
        ----------
        component :
            The scarlet component to be converted.

        Returns
        -------
        result :
            The reconstructed object
        """
        return ScarletComponentData(
            origin=component.bbox.origin,  # type: ignore
            peak=component.peak,  # type: ignore
            model=component.get_model().data,
        )


@dataclass(kw_only=True)
class ScarletFactorizedComponentData(ScarletComponentBaseData):
    """Data for a factorized component

    Attributes
    ----------
    origin :
        The lower bound of the component's bounding box.
    peak :
        The ``(y, x)`` peak of the component.
    spectrum :
        The SED of the component.
    morph :
        The 2D morphology of the component.
    """

    component_type: str = "factorized"
    origin: tuple[int, int]
    peak: tuple[float, float]
    spectrum: np.ndarray
    morph: np.ndarray

    @property
    def shape(self):
        return self.morph.shape

    def to_component(self, observation: Observation) -> FactorizedComponent:
        """Convert the storage data model into a scarlet FactorizedComponent

        Parameters
        ----------
        observation :
            The observation that the component is associated with

        Returns
        -------
        factorized_component :
            A scarlet factorized component extracted from persisted data.
        """
        bbox = Box(self.shape, origin=self.origin)
        spectrum = self.spectrum
        morph = self.morph
        if self.peak is None:
            peak = None
        else:
            peak = (int(np.round(self.peak[0])), int(np.round(self.peak[1])))
        assert peak is not None
        # Note: since we aren't fitting a model, we don't need to
        # set the RMS of the background.
        # We set it to NaN just to be safe.
        component = FactorizedComponent(
            bands=observation.bands,
            spectrum=FixedParameter(spectrum),
            morph=FixedParameter(morph),
            peak=peak,
            bbox=bbox,
            bg_rms=np.full((len(observation.bands),), np.nan),
        )
        return component

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        return {
            "origin": tuple(int(o) for o in self.origin),
            "shape": tuple(int(s) for s in self.morph.shape),
            "peak": tuple(int(p) for p in self.peak),
            "spectrum": tuple(self.spectrum.astype(float)),
            "morph": tuple(self.morph.flatten().astype(float)),
            "component_type": "factorized",
        }

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletFactorizedComponentData:
        """Reconstruct `ScarletFactorizedComponentData` from JSON compatible
        dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object
        dtype :
            Datatype of the resulting model.

        Returns
        -------
        result :
            The reconstructed object
        """
        shape = tuple(data["shape"])

        return cls(
            origin=tuple(data["origin"]),  # type: ignore
            peak=data["peak"],
            spectrum=np.array(data["spectrum"]).astype(dtype),
            morph=np.array(data["morph"]).reshape(shape).astype(dtype),
        )

    @staticmethod
    def _from_component(component: FactorizedComponent) -> ScarletFactorizedComponentData:
        """Reconstruct `ScarletFactorizedComponentData` from a scarlet
        FactorizedComponent.

        Parameters
        ----------
        component :
            The scarlet component to be converted.

        Returns
        -------
        result :
            The reconstructed object
        """
        return ScarletFactorizedComponentData(
            origin=component.bbox.origin,  # type: ignore
            peak=component.peak,  # type: ignore
            spectrum=component.spectrum,
            morph=component.morph,
        )


# Register the component types
ScarletComponentData.register()
ScarletFactorizedComponentData.register()


@dataclass(kw_only=True)
class ScarletSourceData:
    """Data for a scarlet source

    Attributes
    ----------
    components :
        The components contained in the source that are not factorized.
    factorized_components :
        The components contained in the source that are factorized.
    peak_id :
        The peak ID of the source in it's parent's footprint peak catalog.
    """

    components: list[ScarletComponentBaseData]
    peak_id: int

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        result = {
            "components": [component.as_dict() for component in self.components],
            "peak_id": self.peak_id,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletSourceData:
        """Reconstruct `ScarletSourceData` from JSON compatible
        dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object
        dtype :
            Datatype of the resulting model.

        Returns
        -------
        result :
            The reconstructed object
        """
        # Check for legacy models
        if "factorized" in data:
            components: list[ScarletComponentBaseData] = [
                ScarletFactorizedComponentData.from_dict(component, dtype=dtype)
                for component in data["factorized"]
            ]
        else:
            components = [
                ScarletComponentBaseData.from_dict(component, dtype=dtype) for component in data["components"]
            ]
        return cls(components=components, peak_id=int(data["peak_id"]))

    @classmethod
    def from_source(cls, source: Source) -> ScarletSourceData:
        """Reconstruct `ScarletSourceData` from a scarlet Source.

        Parameters
        ----------
        source :
            The scarlet source to be converted.

        Returns
        -------
        result :
            The reconstructed object
        """
        components = [ScarletComponentBaseData.from_component(component) for component in source.components]
        return cls(components=components, peak_id=source.peak_id)  # type: ignore


@dataclass(kw_only=True)
class ScarletBlendData:
    """Data for an entire blend.

    Attributes
    ----------
    origin :
        The lower bound of the blend's bounding box.
    shape :
        The shape of the blend's bounding box.
    sources :
        Data for the sources contained in the blend,
        indexed by the source id.
    psf_center :
        The location used for the center of the PSF for
        the blend.
    psf :
        The PSF of the observation.
    bands : `list` of `str`
        The names of the bands.
        The order of the bands must be the same as the order of
        the multiband model arrays, and SEDs.
    """

    origin: tuple[int, int]
    shape: tuple[int, int]
    sources: dict[int, ScarletSourceData]
    psf_center: tuple[float, float]
    psf: np.ndarray
    bands: tuple[str]

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        result = {
            "origin": self.origin,
            "shape": self.shape,
            "psf_center": self.psf_center,
            "psf_shape": self.psf.shape,
            "psf": tuple(self.psf.flatten().astype(float)),
            "sources": {bid: source.as_dict() for bid, source in self.sources.items()},
            "bands": self.bands,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict, dtype: DTypeLike = np.float32) -> ScarletBlendData:
        """Reconstruct `ScarletBlendData` from JSON compatible
        dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object
        dtype :
            Datatype of the resulting model.

        Returns
        -------
        result :
            The reconstructed object
        """
        psf_shape = data["psf_shape"]
        return cls(
            origin=tuple(data["origin"]),  # type: ignore
            shape=tuple(data["shape"]),  # type: ignore
            psf_center=tuple(data["psf_center"]),  # type: ignore
            psf=np.array(data["psf"]).reshape(psf_shape).astype(dtype),
            sources={
                int(bid): ScarletSourceData.from_dict(source, dtype=dtype)
                for bid, source in data["sources"].items()
            },
            bands=tuple(data["bands"]),  # type: ignore
        )

    def minimal_data_to_blend(self, model_psf: np.ndarray, dtype: DTypeLike) -> Blend:
        """Convert the storage data model into a scarlet lite blend

        Parameters
        ----------
        model_psf :
            PSF in model space (usually a nyquist sampled circular Gaussian).
        dtype :
            The data type of the model that is generated.

        Returns
        -------
        blend :
            A scarlet blend model extracted from persisted data.
        """
        model_box = Box(self.shape, origin=self.origin)
        observation = Observation.empty(
            bands=self.bands,
            psfs=self.psf,
            model_psf=model_psf,
            bbox=model_box,
            dtype=dtype,
        )
        return self.to_blend(observation)

    def to_blend(self, observation: Observation) -> Blend:
        """Convert the storage data model into a scarlet lite blend

        Parameters
        ----------
        observation :
            The observation that contains the blend.
            If `observation` is ``None`` then an `Observation` containing
            no image data is initialized.

        Returns
        -------
        blend :
            A scarlet blend model extracted from persisted data.
        """
        sources = []
        for source_id, source_data in self.sources.items():
            components: list[Component] = [
                component.to_component(observation) for component in source_data.components
            ]

            source = Source(components=components)
            # Store identifiers for the source
            source.record_id = source_id  # type: ignore
            source.peak_id = source_data.peak_id  # type: ignore
            sources.append(source)

        return Blend(sources=sources, observation=observation)

    @staticmethod
    def from_blend(blend: Blend, psf_center: tuple[int, int]) -> ScarletBlendData:
        """Convert a scarlet lite blend into a persistable data object

        Parameters
        ----------
        blend :
            The blend that is being persisted.
        psf_center :
            The center of the PSF.

        Returns
        -------
        blend_data :
            The data model for a single blend.
        """
        sources = {}
        for source in blend.sources:
            sources[source.record_id] = ScarletSourceData.from_source(source)  # type: ignore

        blend_data = ScarletBlendData(
            origin=blend.bbox.origin,  # type: ignore
            shape=blend.bbox.shape,  # type: ignore
            sources=sources,
            psf_center=psf_center,
            psf=blend.observation.psfs,
            bands=blend.observation.bands,  # type: ignore
        )

        return blend_data


class ScarletModelData:
    """A container that propagates scarlet models for an entire catalog."""

    def __init__(self, psf: np.ndarray, blends: dict[int, ScarletBlendData] | None = None):
        """Initialize an instance

        Parameters
        ----------
        psf :
            The 2D array of the PSF in scarlet model space.
            This is typically a narrow Gaussian integrated over the
            pixels in the exposure.
        blends :
            Map from parent IDs in the source catalog
            to scarlet model data for each parent ID (blend).
        """
        self.psf = psf
        if blends is None:
            blends = {}
        self.blends = blends

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        result = {
            "psfShape": self.psf.shape,
            "psf": list(self.psf.flatten().astype(float)),
            "blends": {bid: blend.as_dict() for bid, blend in self.blends.items()},
        }
        return result

    def json(self) -> str:
        """Serialize the data model to a JSON formatted string

        Returns
        -------
        result : `str`
            The result of the object converted into a JSON format
        """
        result = self.as_dict()
        return json.dumps(result)

    @classmethod
    def parse_obj(cls, data: dict) -> ScarletModelData:
        """Construct a ScarletModelData from python decoded JSON object.

        Parameters
        ----------
        data :
            The result of json.load(s) on a JSON persisted ScarletModelData

        Returns
        -------
        result :
            The `ScarletModelData` that was loaded the from the input object
        """
        model_psf = np.array(data["psf"]).reshape(data["psfShape"]).astype(np.float32)
        return cls(
            psf=model_psf,
            blends={int(bid): ScarletBlendData.from_dict(blend) for bid, blend in data["blends"].items()},
        )


class ComponentCube(Component):
    """Dummy component for a component cube.

    This is duck-typed to a `lsst.scarlet.lite.Component` in order to
    generate a model from the component.

    If scarlet lite ever implements a component as a data cube,
    this class can be removed.
    """

    def __init__(self, bands: tuple[Any, ...], bbox: Box, model: Image, peak: tuple[int, int]):
        """Initialization

        Parameters
        ----------
        bands :
        model :
            The 3D (bands, y, x) model of the component.
        peak :
            The `(y, x)` peak of the component.
        bbox :
            The bounding box of the component.
        """
        super().__init__(bands, bbox)
        self._model = model
        self.peak = peak

    def get_model(self) -> Image:
        """Generate the model for the source

        Returns
        -------
        model :
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
