from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import DTypeLike

from .migration import PRE_SCHEMA, MigrationRegistry, migration
from .utils import PersistenceError, json_to_numpy, numpy_to_json

if TYPE_CHECKING:
    from ..psf import ImagePsf, Psf

__all__ = ["PsfBaseData", "ImagePsfData"]

CURRENT_SCHEMA = "1.0.0"
IMAGE_PSF_TYPE = "image"
MigrationRegistry.set_current(IMAGE_PSF_TYPE, CURRENT_SCHEMA)


@dataclass(kw_only=True)
class PsfBaseData(ABC):
    """Base data for a persistable scarlet `~lsst.scarlet.lite.Psf`.

    A `~lsst.scarlet.lite.Psf` is persisted by converting it to one of these
    data objects with `~lsst.scarlet.lite.Psf.to_data`; the data object owns
    the JSON serialization. The `psf_registry` accepts downstream
    registrations so a package that defines its own `~lsst.scarlet.lite.Psf`
    subclass (e.g. a spatially-varying cell-coadd PSF) round-trips through
    the same API.

    Attributes
    ----------
    psf_registry :
        A registry of all known PSF data types used for deserialization.
    psf_type :
        The discriminating tag identifying the concrete PSF data type.
    version :
        The schema version of the exact data class.
    """

    psf_registry: ClassVar[dict[str, type[PsfBaseData]]] = {}
    psf_type: str = ""
    version: str

    @classmethod
    def register(cls) -> None:
        """Register a new PSF data type for deserialization."""
        PsfBaseData.psf_registry[cls.psf_type] = cls

    @abstractmethod
    def to_psf(self) -> Psf:
        """Convert the storage data model into a scarlet `Psf`.

        Returns
        -------
        psf :
            The `~lsst.scarlet.lite.Psf` reconstructed from persisted data.
        """

    @abstractmethod
    def as_dict(self) -> dict[str, Any]:
        """Return the object encoded into a dict for JSON serialization.

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict.
        """

    @staticmethod
    def from_dict(data: dict[str, Any], dtype: DTypeLike = np.float32) -> PsfBaseData:
        """Reconstruct a `PsfBaseData` from a JSON compatible dict.

        Dispatches on the ``psf_type`` tag through `psf_registry`, so any
        registered subclass (including downstream ones) round-trips.

        Parameters
        ----------
        data :
            Dictionary representation of the object.
        dtype :
            Datatype of the reconstructed PSF array.

        Returns
        -------
        result :
            The reconstructed object.

        Raises
        ------
        PersistenceError
            If ``data`` carries an unknown (or unregistered) ``psf_type``.
        """
        psf_type = data.get("psf_type")
        if psf_type not in PsfBaseData.psf_registry:
            raise PersistenceError(f"Unknown psf_type: {psf_type!r}")
        cls = PsfBaseData.psf_registry[psf_type]
        return cls.from_dict(data, dtype=dtype)


@dataclass(kw_only=True)
class ImagePsfData(PsfBaseData):
    """Data for a spatially-constant `~lsst.scarlet.lite.ImagePsf`.

    Attributes
    ----------
    data :
        The ``(bands, height, width)`` PSF image.
    bands :
        The bands of the PSF. May be empty for a band-less (broadcast) PSF.
    padding :
        Padding used when generating the FFT for convolution.
    psf_type :
        The type of PSF being stored.
    version :
        The schema version of the stored data.
    """

    data: np.ndarray
    bands: tuple = ()
    padding: int = 3
    psf_type: str = IMAGE_PSF_TYPE
    version: str = CURRENT_SCHEMA

    def to_psf(self) -> ImagePsf:
        """Convert the storage data model into a scarlet `ImagePsf`.

        Returns
        -------
        psf :
            The reconstructed `~lsst.scarlet.lite.ImagePsf`.
        """
        from ..psf import ImagePsf

        return ImagePsf(self.data, bands=tuple(self.bands), padding=self.padding)

    def as_dict(self) -> dict[str, Any]:
        """Return the object encoded into a dict for JSON serialization.

        The array is encoded with `numpy_to_json` (storing its ``dtype``,
        ``shape`` and flattened ``data``) alongside the ``bands`` and
        ``padding`` needed to rebuild an identical PSF.

        Returns
        -------
        result :
            The object encoded as a JSON compatible dict.
        """
        result: dict[str, Any] = {
            "psf_type": self.psf_type,
            "bands": list(self.bands),
            "padding": self.padding,
            "version": self.version,
        }
        result.update(numpy_to_json(self.data))
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any], dtype: DTypeLike = np.float32) -> ImagePsfData:
        """Reconstruct a `ImagePsfData` from a JSON compatible dict.

        Parameters
        ----------
        data :
            Dictionary representation of the object.
        dtype :
            Datatype of the reconstructed PSF array.

        Returns
        -------
        result :
            The reconstructed object.
        """
        data = MigrationRegistry.migrate(IMAGE_PSF_TYPE, data)
        array = json_to_numpy(data).astype(dtype)
        return cls(
            data=array,
            bands=tuple(data.get("bands", ())),
            padding=data.get("padding", 3),
        )


ImagePsfData.register()


@migration(IMAGE_PSF_TYPE, PRE_SCHEMA)
def _to_1_0_0(data: dict) -> dict:
    """Migrate a pre-schema image PSF to schema version 1.0.0.

    There were no changes to this data model in v1.0.0 but we need to
    provide a way to migrate pre-schema data.

    Parameters
    ----------
    data :
        The data to migrate.

    Returns
    -------
    result :
        The migrated data.
    """
    data["version"] = "1.0.0"
    return data
