# This file is part of scarlet_lite.
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

from utils import ScarletTestCase


class TestImage(ScarletTestCase):
    """Test code blocks in the scarlet lite Image docs"""

    def test_constructors(self):
        # Block 1
        import numpy as np
        from scarlet_lite import Image

        x = np.arange(12).reshape(3, 4)
        image = Image(x)
        print(image)

        # Block 2
        x = np.arange(24).reshape(2, 3, 4)
        image = Image(x, bands=("i", "z"))
        print(image)

        # Block 3
        from scarlet_lite import Box

        image = Image.from_box(Box((4, 5), (100, 120)))
        print(image)

        # Block 4
        image = Image.from_box(Box((3, 4)), bands=("r", "i"))
        print(image)
