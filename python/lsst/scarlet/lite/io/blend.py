from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import DTypeLike

from ..bbox import Box
from ..blend import Blend
from ..observation import Observation
from .blend_base import ScarletBlendBaseData
from .migration import PRE_SCHEMA, MigrationRegistry, migration
from .source import ScarletSourceBaseData
from .utils import decode_metadata, encode_metadata, extract_from_metadata

__all__ = ["ScarletBlendData"]

CURRENT_SCHEMA = "1.0.0"
BLEND_TYPE = "blend"
MigrationRegistry.set_current(BLEND_TYPE, CURRENT_SCHEMA)
logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ScarletBlendData(ScarletBlendBaseData):
    """Data for an entire blend.

    Attributes
    ----------
    blend_type :
        The type of blend being stored.
    metadata :
        Metadata associated with the blend,
        for example the order of bands, the PSF, etc.
    origin :
        The lower bound of the blend's bounding box.
    shape :
        The shape of the blend's bounding box.
    sources :
        Data for the sources contained in the blend,
        indexed by the source id.
    version :
        The schema version of the stored data.
    """

    blend_type: str = BLEND_TYPE
    origin: tuple[int, int]
    shape: tuple[int, int]
    sources: dict[Any, ScarletSourceBaseData]
    version: str = CURRENT_SCHEMA

    def as_dict(self) -> dict:
        """Return the object encoded into a dict for JSON serialization

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict
        """
        result: dict[str, Any] = {
            "blend_type": self.blend_type,
            "origin": self.origin,
            "shape": self.shape,
            "sources": {bid: source.as_dict() for bid, source in self.sources.items()},
            "version": self.version,
        }
        if self.metadata is not None:
            result["metadata"] = encode_metadata(self.metadata)
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
        data = MigrationRegistry.migrate(BLEND_TYPE, data)
        metadata = data.get("metadata", None)

        return cls(
            origin=tuple(data["origin"]),  # type: ignore
            shape=tuple(data["shape"]),  # type: ignore
            sources={
                bid: ScarletSourceBaseData.from_dict(source, dtype=dtype)
                for bid, source in data["sources"].items()
            },
            metadata=decode_metadata(metadata),
        )

    def minimal_data_to_blend(
        self,
        model_psf: np.ndarray | None = None,
        psf: np.ndarray | None = None,
        bands: tuple[str] | None = None,
        dtype: DTypeLike = np.float32,
    ) -> Blend:
        """Convert the storage data model into a scarlet lite blend

        Parameters
        ----------
        model_psf :
            PSF in model space (usually a nyquist sampled circular Gaussian).
        psf :
            The PSF of the observation.
            If not provided, the PSF stored in the blend data is used.
        bands :
            The bands in the blend model.
            If not provided, the bands stored in the blend data are used.
        dtype :
            The data type of the model that is generated.

        Returns
        -------
        blend :
            A scarlet blend model extracted from persisted data.
        """

        _model_psf: np.ndarray = extract_from_metadata(model_psf, self.metadata, "model_psf")
        _psf: np.ndarray = extract_from_metadata(psf, self.metadata, "psf")
        _bands: tuple[str] = extract_from_metadata(bands, self.metadata, "bands")
        model_box = Box(self.shape, origin=self.origin)
        observation = Observation.empty(
            bands=_bands,
            psfs=_psf,
            model_psf=_model_psf,
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
        for source_data in self.sources.values():
            source = source_data.to_source(observation)
            sources.append(source)

        return Blend(sources=sources, observation=observation, metadata=self.metadata)

    @staticmethod
    def from_blend(blend: Blend) -> ScarletBlendData:
        """Deprecated: Convert a scarlet lite blend into a storage data model.

        Parameters
        ----------
        blend :
            The blend to convert.
        Returns
        -------
        result :
            The storage data model representing the blend.
        """
        logger.warning("ScarletBlendData.from_blend is deprecated. Use blend.to_data() instead.")
        return blend.to_data()


ScarletBlendData.register()


@migration(BLEND_TYPE, PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema blend to schema version 1.0.0

    Parameters
    ----------
    data :
        The data to migrate.

    Returns
    -------
    result :
        The migrated data.
    """
    # Support legacy models before metadata was used
    if "metadata" not in data and "psf" in data:
        data["metadata"] = {
            "psf": data["psf"],
            "psf_shape": data["psf_shape"],
            "bands": tuple(data["bands"]),
            "array" "_keys": ["psf"],
        }
    data["version"] = "1.0.0"
    return data
