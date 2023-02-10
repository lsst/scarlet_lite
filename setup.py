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

# This uses the code at
# https://github.com/pybind/python_example/blob/master/setup.py
# as a template to integrate pybind11

"""
Basic setuptools description.
This is not a complete definition.
* Version number is not correct.
* The shared library and include files are not installed.  This makes it
  unusable with other python packages that directly reference the C++
  interface.
"""

import glob

# Importing this automatically enables parallelized builds
import numpy.distutils.ccompiler  # noqa: F401
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


class GetEigenInclude(object):
    """Helper class to determine the peigen include path
    The purpose of this class is to postpone importing peigen
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import peigen

        return peigen.header_path


# Find the source code -- we can combine it into a single module
pybind_src = sorted(glob.glob("python/lsst/scarlet/lite/*.cc"))

ext_modules = [Pybind11Extension("lsst.scarlet.lite", pybind_src, include_dirs=[GetEigenInclude()])]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
