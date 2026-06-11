from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from deprecated.sphinx import deprecated  # type: ignore
from numpy.typing import DTypeLike

from ..bbox import Box
from ..blend import Blend
from ..observation import Observation
from ..psf import ImagePsf, Psf
from .blend_base import ScarletBlendBaseData
from .migration import PRE_SCHEMA, MigrationRegistry, migration
from .psf import PsfBaseData
from .source import ScarletSourceBaseData
from .utils import decode_metadata, encode_metadata, extract_from_metadata

__all__ = ["ScarletBlendData"]

CURRENT_SCHEMA = "1.1.0"
BLEND_TYPE = "blend"
MigrationRegistry.set_current(BLEND_TYPE, CURRENT_SCHEMA)
logger = logging.getLogger(__name__)


def _resolve_psf(value: Psf | PsfBaseData | np.ndarray, bands: tuple) -> Psf:
    """Resolve a metadata PSF value into a `Psf`.

    Parameters
    ----------
    value :
        A `Psf` (used as-is), a PSF data object (converted with `to_psf`), or
        a legacy raw array (wrapped in an `ImagePsf`).
    bands :
        The bands used when wrapping a legacy raw array.

    Returns
    -------
    result :
        The resolved `Psf`.
    """
    if isinstance(value, Psf):
        return value
    if isinstance(value, PsfBaseData):
        return value.to_psf()
    return ImagePsf(value, bands=bands)


@dataclass(kw_only=True)
class ScarletBlendData(ScarletBlendBaseData):
    """Data for an entire blend.

    Attributes
    ----------
    blend_type :
        The type of blend being stored.
    metadata :
        Metadata associated with the blend,
        for example the order of bands.
    origin :
        The lower bound of the blend's bounding box.
    shape :
        The shape of the blend's bounding box.
    sources :
        Data for the sources contained in the blend,
        indexed by the source id.
    psf :
        The observed PSF of the blend, as a persistable data object. Optional:
        it is only set when the PSF is persisted with the blend (see
        `Blend.to_data`'s ``persist_psf``). LSST instead carries the PSF on the
        model container, so this is normally `None`.
    model_psf :
        The model-space PSF of the blend, as a persistable data object.
        Optional, with the same semantics as ``psf``.
    version :
        The schema version of the stored data.
    """

    blend_type: str = BLEND_TYPE
    origin: tuple[int, int]
    shape: tuple[int, int]
    sources: dict[Any, ScarletSourceBaseData]
    psf: PsfBaseData | None = None
    model_psf: PsfBaseData | None = None
    version: str = CURRENT_SCHEMA

    @cached_property
    def bbox(self) -> Box:
        """The bounding box of the blend"""
        return Box(self.shape, origin=self.origin)

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
        # The PSFs are first-class, so they serialize directly (each PSF data
        # object knows how) rather than being flattened into metadata.
        if self.psf is not None:
            result["psf"] = self.psf.as_dict()
        if self.model_psf is not None:
            result["model_psf"] = self.model_psf.as_dict()
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
        psf = data.get("psf", None)
        model_psf = data.get("model_psf", None)

        return cls(
            origin=tuple(data["origin"]),  # type: ignore
            shape=tuple(data["shape"]),  # type: ignore
            sources={
                bid: ScarletSourceBaseData.from_dict(source, dtype=dtype)
                for bid, source in data["sources"].items()
            },
            psf=PsfBaseData.from_dict(psf) if psf is not None else None,
            model_psf=PsfBaseData.from_dict(model_psf) if model_psf is not None else None,
            metadata=decode_metadata(metadata),
        )

    def minimal_data_to_blend(
        self,
        model_psf: np.ndarray | Psf | PsfBaseData | None = None,
        psf: np.ndarray | Psf | PsfBaseData | None = None,
        bands: tuple[str] | None = None,
        dtype: DTypeLike = np.float32,
    ) -> Blend:
        """Convert the storage data model into a scarlet lite blend

        Parameters
        ----------
        model_psf :
            PSF in model space (usually a nyquist sampled circular Gaussian).
            Accepts a `Psf` or, for backward compatibility, a raw array.
            If not provided, the blend's own ``model_psf`` (or, for legacy
            archives, the model PSF in its metadata) is used.
        psf :
            The PSF of the observation, as a `Psf` or a raw array.
            If not provided, the blend's own ``psf`` (or, for legacy archives,
            the PSF in its metadata) is used.
        bands :
            The bands in the blend model.
            If not provided, the bands stored in the blend data are used,
            falling back to the bands carried by the observed PSF.
        dtype :
            The data type of the model that is generated.

        Returns
        -------
        blend :
            A scarlet blend model extracted from persisted data.
        """
        # Only an ndarray the *caller* passes is deprecated; one that merely
        # lives in a legacy blend's metadata is fine (and silently supported).
        if isinstance(model_psf, np.ndarray):
            logger.warning(
                "Passing model_psf as a raw ndarray to `minimal_data_to_blend` is "
                "deprecated and will be unsupported after v31.0"
            )
        if isinstance(psf, np.ndarray):
            logger.warning(
                "Passing psf as a raw ndarray to `minimal_data_to_blend` is "
                "deprecated and will be unsupported after v31.0"
            )

        # Prefer an explicit argument, then the first-class blend attribute,
        # then (legacy, un-warned) metadata.
        if psf is None:
            psf = self.psf
        if model_psf is None:
            model_psf = self.model_psf
        if psf is None:
            psf = extract_from_metadata(psf, self.metadata, "psf")
        if model_psf is None:
            model_psf = extract_from_metadata(model_psf, self.metadata, "model_psf")
        # ``_bands`` is intentionally ``Any``: ``Observation.empty`` types its
        # ``bands`` loosely, and the value may come from metadata, the caller,
        # or the resolved PSF.
        _bands: Any = bands
        if _bands is None and self.metadata is not None:
            _bands = self.metadata.get("bands")

        # Resolve each PSF to a `Psf`: a PSF data object via `to_psf`, a `Psf`
        # used as-is, and a legacy raw array wrapped in an `ImagePsf`.
        psf = _resolve_psf(psf, bands=tuple(_bands) if _bands is not None else ())
        model_psf = _resolve_psf(model_psf, bands=())
        # The observation bands default to those carried by the observed PSF.
        if _bands is None:
            _bands = psf.bands

        model_box = self.bbox
        observation = Observation.empty(
            bands=_bands,
            psf=psf,
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
        for sid, source_data in self.sources.items():
            source = source_data.to_source(observation)
            # Ensure that the source id is persisted to its metadata
            if source.metadata is None:
                source.metadata = {}
            if "id" not in source.metadata:
                source.metadata["id"] = sid
            sources.append(source)

        return Blend(sources=sources, observation=observation, metadata=self.metadata)

    @staticmethod
    @deprecated(
        reason="ScarletBlendData.from_blend is deprecated. Use blend.to_data() instead.",
        version="v30.0",
        category=FutureWarning,
    )
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


@migration(BLEND_TYPE, "1.0.0")
def _to_1_1_0(data: dict) -> dict:
    """Migrate a schema version 1.0.0 blend to schema version 1.1.0.

    1.0.0 (and earlier) stored the observed PSF as a raw array hidden inside
    the free-form blend ``metadata`` dict (an ``array_keys`` entry). 1.1.0
    promotes it to the first-class ``psf`` attribute, an ``ImagePsfData`` dict
    deserialized by `ScarletBlendData.from_dict`. The migration is the one
    place where reading the legacy ``metadata`` PSF is the correct thing to do.

    Parameters
    ----------
    data :
        The data to migrate.

    Returns
    -------
    result :
        The migrated data.
    """
    metadata = data.get("metadata")
    # Only lift a PSF that was stored as a flattened array (``array_keys``);
    # the migration runs before ``decode_metadata``, so the array is still in
    # its encoded ``psf`` / ``psf_shape`` / ``psf_dtype`` form here.
    if metadata and "psf" in metadata and "psf" in metadata.get("array_keys", []):
        shape = metadata.pop("psf_shape", None)
        if shape is None:
            shape = metadata.pop("psfShape", None)
        dtype = metadata.pop("psf_dtype", "float32")
        data["psf"] = {
            "psf_type": "image",
            "bands": list(metadata.get("bands", [])),
            "padding": 3,
            "version": "1.0.0",
            "dtype": dtype,
            "shape": shape,
            "data": metadata.pop("psf"),
        }
        array_keys = [key for key in metadata["array_keys"] if key != "psf"]
        if array_keys:
            metadata["array_keys"] = array_keys
        else:
            metadata.pop("array_keys", None)
    data["version"] = "1.1.0"
    return data
