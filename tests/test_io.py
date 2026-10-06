# This file is part of lsst.scarlet.lite.
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

import json
import os

import numpy as np
from lsst.scarlet.lite import Blend, Box, Image, ImagePsf, Observation, io
from lsst.scarlet.lite.component import CubeComponent
from lsst.scarlet.lite.initialization import FactorizedInitialization
from lsst.scarlet.lite.io import ImagePsfData, PsfBaseData
from lsst.scarlet.lite.io.utils import PersistenceError
from lsst.scarlet.lite.operators import Monotonicity
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_almost_equal
from utils import ScarletTestCase


class TestIo(ScarletTestCase):
    def setUp(self) -> None:
        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        data = np.load(filename)
        model_psf = integrated_circular_gaussian(sigma=0.8)
        self.detect = np.sum(data["images"], axis=0)
        self.centers = np.array([data["catalog"]["y"], data["catalog"]["x"]]).T
        bands = data["filters"]
        self.observation = Observation(
            Image(data["images"], bands=bands),
            Image(data["variance"], bands=bands),
            Image(1 / data["variance"], bands=bands),
            ImagePsf(data["psfs"], bands=bands),
            ImagePsf(model_psf[None]),
            bands=bands,
        )
        monotonicity = Monotonicity((101, 101))
        init = FactorizedInitialization(self.observation, self.centers, monotonicity=monotonicity)
        self.blend = Blend(init.sources, self.observation)

    def test_json(self):
        blend = self.blend
        blend.metadata = {
            "psf": self.observation.model_psf.data,
            "bands": tuple(str(band) for band in self.observation.bands),
        }
        blend_data = blend.to_data()
        metadata = {
            "model_psf": self.observation.model_psf.data,
        }
        model_data = io.ScarletModelData(
            blends={1: blend_data},
            metadata=metadata,
        )

        # Get the json string for the model
        model_str = model_data.json()
        # Load the model string from the json
        model_dict = json.loads(model_str)
        # Load the full set of model data classes from the json string
        model_data = io.ScarletModelData.parse_obj(model_dict)
        metadata = model_data.metadata
        self.assertIsNotNone(metadata)
        # Convert the model data into scarlet models
        loaded_blend = model_data.blends[1].minimal_data_to_blend(
            model_psf=metadata["model_psf"],  # type: ignore
            dtype=blend.observation.dtype,
        )

        self.assertEqual(len(blend.sources), len(loaded_blend.sources))
        self.assertEqual(len(blend.components), len(loaded_blend.components))
        self.assertImageAlmostEqual(blend.get_model(), loaded_blend.get_model())
        self.assertBoxEqual(blend.bbox, blend_data.bbox)

        for sidx in range(len(blend.sources)):
            source1 = blend.sources[sidx]
            source2 = loaded_blend.sources[sidx]
            self.assertTupleEqual(source1.center, source2.center)
            self.assertEqual(len(source1.components), len(source2.components))
            self.assertBoxEqual(source1.bbox, source2.bbox)
            for cidx in range(len(source1.components)):
                component1 = source1.components[cidx]
                component2 = source2.components[cidx]
                self.assertEqual(component1.peak, component2.peak)
                assert_almost_equal(component1.spectrum, component2.spectrum)
                assert_almost_equal(component1.morph, component2.morph)
                self.assertBoxEqual(component1.bbox, component2.bbox)

    def test_cube_component(self):
        blend = self.blend
        for i in range(len(blend.sources)):
            blend.sources[i].metadata = {"id": f"peak-{i}"}
        component = blend.sources[-1].components[-1]
        # Replace one of the components with a Free-Form component.
        blend.sources[-1].components[-1] = CubeComponent(
            model=component.get_model(),
            peak=component.peak,
        )

        blend_data = blend.to_data()
        model_data = io.ScarletModelData(
            blends={1: blend_data},
            metadata={
                "model_psf": self.observation.model_psf.data,
                "psf": self.observation.psf.data,
                "bands": tuple(str(band) for band in self.observation.bands),
            },
        )

        # Get the json string for the model
        model_str = model_data.json()
        # Load the model string from the json
        model_dict = json.loads(model_str)
        # Load the full set of model data classes from the json string
        model_data = io.ScarletModelData.parse_obj(model_dict)
        # Convert the model data into scarlet models
        loaded_blend = model_data.blends[1].minimal_data_to_blend(
            model_psf=model_data.metadata["model_psf"],  # type: ignore
            bands=model_data.metadata["bands"],  # type: ignore
            psf=model_data.metadata["psf"],  # type: ignore
            dtype=blend.observation.dtype,
        )

        self.assertEqual(len(blend.sources), len(loaded_blend.sources))
        self.assertEqual(len(blend.components), len(loaded_blend.components))
        self.assertImageAlmostEqual(blend.get_model(), loaded_blend.get_model())

        # Check that the metadata was stored correctly
        for i in range(len(blend.sources)):
            self.assertEqual(blend.sources[i].metadata, loaded_blend.sources[i].metadata)

    def test_cube_component_to_component_preserves_peak(self):
        """``ScarletCubeComponentData.to_component`` must preserve the
        full ``(y, x)`` peak, not collapse both axes onto ``peak[0]``.

        Regression test: the implementation previously read ``peak[0]``
        twice when constructing the returned ``CubeComponent``, so any
        non-symmetric peak silently round-tripped as ``(y, y)``.
        """
        peak = (54, 105)
        n_bands, h, w = 3, 8, 10
        cube_data = io.ScarletCubeComponentData(
            origin=(50, 100),
            peak=peak,
            model=np.zeros((n_bands, h, w), dtype=np.float32),
        )
        observation = Observation.empty(
            bands=("g", "r", "i"),
            psf=ImagePsf(np.ones((n_bands, 5, 5), dtype=np.float32), bands=("g", "r", "i")),
            model_psf=ImagePsf(np.ones((1, 5, 5), dtype=np.float32)),
            bbox=Box((h, w), origin=(50, 100)),
            dtype=np.float32,
        )

        component = cube_data.to_component(observation)

        self.assertEqual(component.peak, peak)

    def test_psf_data_dict_roundtrip(self):
        """`ImagePsfData.as_dict`/`PsfBaseData.from_dict` must
        round-trip (including through JSON) and dispatch on ``psf_type``.
        """

        psf = self.observation.psf
        data = psf.to_data()
        encoded = json.loads(json.dumps(data.as_dict()))
        self.assertEqual(encoded["psf_type"], "image")

        # The base-class dispatcher reconstructs the concrete data type.
        restored = PsfBaseData.from_dict(encoded)
        self.assertIsInstance(restored, ImagePsfData)
        restored_psf = restored.to_psf()
        self.assertEqual(restored_psf.bands, psf.bands)
        assert_almost_equal(restored_psf.data, psf.data)

        # An unregistered type is a persistence error.
        with self.assertRaises(PersistenceError):
            PsfBaseData.from_dict({"psf_type": "does-not-exist"})

    def test_minimal_data_to_blend_ndarray_psf_warns(self):
        """Passing a raw ndarray PSF *argument* logs a deprecation warning
        (one per ndarray argument), but a `Psf` argument — or an ndarray that
        merely comes from legacy metadata — does not.
        """
        logger_name = "lsst.scarlet.lite.io.blend"
        bands = tuple(str(b) for b in self.observation.bands)

        # A blend whose metadata holds raw-array PSFs, as a legacy file would
        # decode to.
        legacy_blend = self.blend
        legacy_blend.metadata = {
            "psf": self.observation.psf.data,
            "model_psf": self.observation.model_psf.data,
            "bands": bands,
        }
        legacy_blend_data = legacy_blend.to_data()

        # Passing ndarray arguments warns once for each.
        with self.assertLogs(logger_name, level="WARNING") as cm:
            legacy_blend_data.minimal_data_to_blend(
                model_psf=self.observation.model_psf.data,
                psf=self.observation.psf.data,
                dtype=self.observation.dtype,
            )
        self.assertEqual(len(cm.records), 2)
        messages = "\n".join(cm.output)
        self.assertIn("Passing model_psf as a raw ndarray", messages)
        self.assertIn("Passing psf as a raw ndarray", messages)

        # Letting the PSFs come from (legacy ndarray) metadata does NOT warn —
        # only an ndarray the caller explicitly passes is deprecated.
        with self.assertNoLogs(logger_name, level="WARNING"):
            legacy_blend_data.minimal_data_to_blend(dtype=self.observation.dtype)

        # A `Psf` argument does not warn either.
        with self.assertNoLogs(logger_name, level="WARNING"):
            legacy_blend_data.minimal_data_to_blend(
                model_psf=self.observation.model_psf,
                psf=self.observation.psf,
                dtype=self.observation.dtype,
            )

    def test_persist_psf_roundtrip(self):
        """`Blend.to_data(persist_psf=True)` makes the observed and model PSFs
        first-class attributes on the blend data that round-trip through
        serialization; `minimal_data_to_blend` then uses them with no explicit
        psf argument and no deprecation warning.
        """
        blend = self.blend
        blend.metadata = {"bands": tuple(str(b) for b in self.observation.bands)}

        # Opt in to persisting the PSFs with the blend.
        blend_data = blend.to_data(persist_psf=True)
        self.assertIsInstance(blend_data.psf, io.ImagePsfData)
        self.assertIsInstance(blend_data.model_psf, io.ImagePsfData)

        model_data = io.ScarletModelData(blends={1: blend_data})

        # Round-trip through JSON; the PSFs are first-class top-level keys on
        # the blend, not entries in its metadata.
        model_dict = json.loads(model_data.json())
        blend_dict = model_dict["blends"]["1"]
        self.assertEqual(blend_dict["psf"]["psf_type"], "image")
        self.assertEqual(blend_dict["model_psf"]["psf_type"], "image")
        self.assertNotIn("psf", blend_dict.get("metadata", {}))

        model_data = io.ScarletModelData.parse_obj(model_dict)
        loaded_blend_data = model_data.blends[1]
        self.assertIsInstance(loaded_blend_data.psf, io.ImagePsfData)
        assert_almost_equal(loaded_blend_data.psf.to_psf().data, self.observation.psf.data)
        assert_almost_equal(loaded_blend_data.model_psf.to_psf().data, self.observation.model_psf.data)

        # The blend reconstructs from its own PSFs with no explicit psf args
        # and no deprecation warning.
        with self.assertNoLogs("lsst.scarlet.lite.io.blend", level="WARNING"):
            loaded_blend = loaded_blend_data.minimal_data_to_blend(
                dtype=blend.observation.dtype,
            )
        self.assertEqual(len(blend.sources), len(loaded_blend.sources))
        self.assertImageAlmostEqual(blend.get_model(), loaded_blend.get_model())

    def test_legacy_metadata_psf_migrates_to_attribute(self):
        """A legacy blend with the observed PSF buried in its metadata array
        is migrated to the first-class ``psf`` attribute (and removed from the
        metadata) by `ScarletBlendData.from_dict`.
        """
        bands = tuple(str(b) for b in self.observation.bands)

        # Build a pre-schema blend dict with the PSF stored as a raw array, as
        # old archives did.
        blend_dict = self.blend.to_data().as_dict()
        encoded_psf = io.utils.numpy_to_json(self.observation.psf.data)
        blend_dict["psf"] = encoded_psf["data"]
        blend_dict["psf_shape"] = encoded_psf["shape"]
        blend_dict["bands"] = bands
        blend_dict.pop("version", None)

        blend_data = io.ScarletBlendData.from_dict(blend_dict)

        # The PSF is now a first-class data object, not a metadata entry.
        self.assertIsInstance(blend_data.psf, io.ImagePsfData)
        self.assertEqual(tuple(blend_data.psf.bands), bands)
        assert_almost_equal(blend_data.psf.to_psf().data, self.observation.psf.data)
        self.assertNotIn("psf", blend_data.metadata)

    def test_legacy_json(self):
        blend = self.blend

        # Create legacy blend JSON data
        blend_data = blend.to_data().as_dict()
        encoded_psf = io.utils.numpy_to_json(self.observation.psf.data)
        blend_data["psf"] = encoded_psf["data"]
        blend_data["psf_shape"] = encoded_psf["shape"]
        blend_data["bands"] = tuple(str(band) for band in self.observation.bands)
        blend_data["psf_center"] = (10, 10)

        # Create legacy model data
        model_data = io.ScarletModelData(blends={}).as_dict()
        model_data["blends"][1] = blend_data
        encoded_psf = io.utils.numpy_to_json(self.observation.model_psf.data)
        model_data["psf"] = encoded_psf["data"]
        model_data["psfShape"] = encoded_psf["shape"]

        # Legacy models were pre-versioning, so delete any version key
        model_data.pop("version", None)
        blend_data.pop("version", None)
        for source in blend_data["sources"].values():
            source.pop("version", None)
            for component in source["components"]:
                component.pop("version", None)

        self.assertIsNone(model_data["metadata"])

        # Get the json string for the model
        model_str = json.dumps(model_data)
        # Load the model string from the json
        model_dict = json.loads(model_str)
        # Load the full set of model data classes from the json string
        model_data = io.ScarletModelData.parse_obj(model_dict)
        metadata = model_data.metadata
        self.assertIsNotNone(metadata)

        # Convert the model data into scarlet models
        loaded_blend = model_data.blends[1].minimal_data_to_blend(
            model_psf=metadata["model_psf"],  # type: ignore
            dtype=blend.observation.dtype,
        )

        self.assertEqual(len(blend.sources), len(loaded_blend.sources))
        self.assertEqual(len(blend.components), len(loaded_blend.components))
        self.assertImageAlmostEqual(blend.get_model(), loaded_blend.get_model())

        # Legacy models (e.g. DP1) predate the source metadata field, so the
        # source id is not stored in the source itself. Ensure that the id
        # is propagated into ``source.metadata["id"]`` on load.
        for sid, source in zip(model_data.blends[1].sources, loaded_blend.sources):
            self.assertIsNotNone(source.metadata)
            self.assertEqual(source.metadata["id"], sid)
