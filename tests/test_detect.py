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

import os

import numpy as np
from lsst.scarlet.lite import Box, Image
from lsst.scarlet.lite.detect import (
    bounds_to_bbox,
    detect_footprints,
    footprints_to_image,
    get_detect_wavelets,
    get_wavelets,
)
from lsst.scarlet.lite.detect_pybind11 import (
    Footprint,
    Peak,
    get_connected_multipeak,
    get_connected_pixels,
    get_footprints,
)
from lsst.scarlet.lite.utils import integrated_circular_gaussian
from numpy.testing import assert_array_equal
from utils import ScarletTestCase


class TestDetect(ScarletTestCase):
    def setUp(self):
        centers = (
            (17, 9),
            (27, 14),
            (41, 25),
            (10, 42),
        )
        sigmas = (1.0, 0.95, 0.9, 1.5)

        sources = []
        for sigma, center in zip(sigmas, centers):
            yx0 = center[0] - 7, center[1] - 7
            source = Image(integrated_circular_gaussian(sigma=sigma).astype(np.float32), yx0=yx0)
            sources.append(source)

        image = Image.from_box(Box((51, 51)))
        for source in sources:
            image += source
        image.data[30:32, 40] = 0.5

        self.image = image
        self.centers = centers
        self.sources = sources

        filename = os.path.join(__file__, "..", "..", "data", "hsc_cosmos_35.npz")
        filename = os.path.abspath(filename)
        self.hsc_data = np.load(filename)

    def tearDown(self):
        del self.hsc_data

    def test_connected(self):
        image = self.image.copy()

        # Check that the first 3 footprints are all connected
        # with thresholding at zero
        truth = self.sources[0] + self.sources[1] + self.sources[2]
        bbox = truth.bbox
        truth = truth.data > 0

        unchecked = np.ones(self.image.shape, dtype=bool)
        footprint = np.zeros(self.image.shape, dtype=bool)
        y, x = self.centers[0]
        get_connected_pixels(
            y,
            x,
            image.data,
            unchecked,
            footprint,
            np.array([y, y, x, x]).astype(np.int32),
            0,
        )
        assert_array_equal(footprint[bbox.slices], truth)

        # Check that only the first 2 footprints are all connected
        # with thresholding at 1e-15
        truth = self.sources[0] + self.sources[1]
        bbox = truth.bbox
        truth = truth.data > 1e-15

        unchecked = np.ones(self.image.shape, dtype=bool)
        footprint = np.zeros(self.image.shape, dtype=bool)
        y, x = self.centers[0]
        get_connected_pixels(
            y,
            x,
            image.data,
            unchecked,
            footprint,
            np.array([y, y, x, x]).astype(np.int32),
            1e-15,
        )
        assert_array_equal(footprint[bbox.slices], truth)

        # Test finding all peaks
        footprint = get_connected_multipeak(self.image.data, self.centers, 1e-15)
        truth = self.image.data > 1e-15
        truth[30:32, 40] = False
        assert_array_equal(footprint, truth)

    def _footprint_check(self, footprints):
        self.assertEqual(len(footprints), 3)

        # The first footprint has a single peak
        assert_array_equal(footprints[0].data, self.sources[3].data > 1e-15)
        self.assertEqual(len(footprints[0].peaks), 1)
        self.assertBoxEqual(footprints[0].bbox, self.sources[3].bbox)
        self.assertEqual(footprints[0].peaks[0].y, self.centers[3][0])
        self.assertEqual(footprints[0].peaks[0].x, self.centers[3][1])

        # The second footprint has two peaks
        truth = self.sources[0] + self.sources[1]
        assert_array_equal(footprints[1].data, truth.data > 1e-15)
        self.assertEqual(len(footprints[1].peaks), 2)
        self.assertBoxEqual(footprints[1].bbox, truth.bbox)
        self.assertEqual(footprints[1].peaks[0].y, self.centers[1][0])
        self.assertEqual(footprints[1].peaks[0].x, self.centers[1][1])
        self.assertEqual(footprints[1].peaks[1].y, self.centers[0][0])
        self.assertEqual(footprints[1].peaks[1].x, self.centers[0][1])

        # The third footprint has a single peak
        assert_array_equal(footprints[2].data, self.sources[2].data > 1e-15)
        self.assertEqual(len(footprints[2].peaks), 1)
        self.assertBoxEqual(footprints[2].bbox, self.sources[2].bbox)
        self.assertEqual(footprints[2].peaks[0].y, self.centers[2][0])
        self.assertEqual(footprints[2].peaks[0].x, self.centers[2][1])

        truth = 1 * self.sources[3] + 2 * (self.sources[0] + self.sources[1]) + 3 * self.sources[2]
        truth.data[truth.data < 1e-15] = 0
        fp_image = footprints_to_image(footprints, truth.bbox)
        assert_array_equal(fp_image, truth.data)

    def test_get_footprints(self):
        footprints = get_footprints(self.image.data, 1, 4, 1e-15, 1e-15, True)
        self._footprint_check(footprints)

    def _peak_check(self, peaks):
        matched_peaks = []
        for center in self.centers:
            for peak in peaks:
                if peak.y == center[0] and peak.x == center[1]:
                    matched_peaks.append(peak)
                    break
        self.assertEqual(len(matched_peaks), len(self.centers))

    def test_detect_footprints(self):
        # this method doesn't test for accurracy, since

        # There is no variance, so we set it to ones
        variance = np.ones(self.image.shape, dtype=self.image.dtype)

        footprints = detect_footprints(
            self.image.data[None, :, :],
            variance[None, :, :],
            scales=1,
            generation=2,
            origin=(0, 0),
            min_separation=1,
            min_area=4,
            peak_thresh=1e-15,
            footprint_thresh=1e-15,
            find_peaks=True,
            remove_high_freq=False,
            min_pixel_detect=1,
        )

        self.assertEqual(len(footprints), 3)
        peaks = [peak for footprint in footprints for peak in footprint.peaks]
        self._peak_check(peaks)

        footprints = detect_footprints(
            self.image.data[None, :, :],
            variance[None, :, :],
            scales=1,
            generation=1,
            min_separation=1,
            min_area=4,
            peak_thresh=1e-15,
            footprint_thresh=1e-15,
            find_peaks=True,
            remove_high_freq=True,
            min_pixel_detect=1,
        )

        self.assertEqual(len(footprints), 2)
        peaks = [peak for footprint in footprints for peak in footprint.peaks]
        self._peak_check(peaks)

    def test_bounds_to_bbox(self):
        bounds = (3, 27, 11, 52)
        truth = Box((25, 42), (3, 11))
        bbox = bounds_to_bbox(bounds)
        self.assertBoxEqual(bbox, truth)

    def test_footprint(self):
        footprint = self.sources[0].data
        footprint[footprint < 1e-15] = 0
        bounds = [
            self.sources[0].bbox.start[0],
            self.sources[0].bbox.stop[0] - 1,
            self.sources[0].bbox.start[1],
            self.sources[0].bbox.stop[1] - 1,
        ]
        print(bounds)
        peaks = [Peak(self.centers[0][0], self.centers[0][1], self.image.data[self.centers[0]])]
        footprint1 = Footprint(footprint, peaks, bounds)
        footprint = self.sources[1].data
        footprint[footprint < 1e-15] = 0
        bounds = [
            self.sources[1].bbox.start[0],
            self.sources[1].bbox.stop[0] - 1,
            self.sources[1].bbox.start[1],
            self.sources[1].bbox.stop[1] - 1,
        ]
        print(bounds)
        peaks = [Peak(self.centers[1][0], self.centers[1][1], self.image.data[self.centers[1]])]
        footprint2 = Footprint(footprint, peaks, bounds)

        truth = self.sources[0] + self.sources[1]
        truth.data[truth.data < 1e-15] = 0
        image = footprints_to_image([footprint1, footprint2], truth.bbox)
        assert_array_equal(image, truth.data)

        # Test intersection
        truth = (self.sources[0] > 1e-15) & (self.sources[1] > 1e-15)
        intersection = footprint1.intersection(footprint2)
        self.assertImageEqual(intersection, truth)

        # Test union
        truth = (self.sources[0] > 1e-15) | (self.sources[1] > 1e-15)
        union = footprint1.union(footprint2)
        self.assertImageEqual(union, truth)

    def test_get_wavelets(self):
        images = self.hsc_data["images"]
        variance = self.hsc_data["variance"]
        wavelets = get_wavelets(images, variance)

        self.assertTupleEqual(wavelets.shape, (5, 5, 58, 48))
        self.assertEqual(wavelets.dtype, np.float32)

    def test_get_detect_wavelets(self):
        images = self.hsc_data["images"]
        variance = self.hsc_data["variance"]
        wavelets = get_detect_wavelets(images, variance)

        self.assertTupleEqual(wavelets.shape, (4, 58, 48))
