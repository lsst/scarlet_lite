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

import os
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys

pybind11_path = None
if "PYBIND11_DIR" in os.environ:
    pybind11_path = os.environ["PYBIND11_DIR"]
eigen_path = None
if "EIGEN_DIR" in os.environ:
    eigen_path = os.environ["EIGEN_DIR"]

packages = find_packages()


class GetPybindInclude(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        if "PYBIND11_INCLUDE" in os.environ:
            return os.environ["PYBIND11_INCLUDE"]
        if pybind11_path is not None:
            return os.path.join(os.environ["PYBIND11_DIR"], "include")
        else:
            import pybind11

            return pybind11.get_include(self.user)


class GetEigenInclude(object):
    """Helper class to determine the peigen include path
    The purpose of this class is to postpone importing peigen
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        if "EIGEN_INCLUDE" in os.environ:
            return os.environ["EIGEN_INCLUDE"]
        if eigen_path is not None:
            return os.path.join(os.environ["EIGEN_DIR"], "include")
        else:
            import peigen

            return peigen.header_path


ext_modules = [
    Extension(
        "scarlet_lite.operators_pybind11",
        ["scarlet_lite/operators_pybind11.cc"],
        include_dirs=[
            GetPybindInclude(),
            GetPybindInclude(user=True),
            GetEigenInclude(),
        ],
        language="c++",
    ),
    Extension(
        "scarlet_lite.detect_pybind11",
        ["scarlet_lite/detect_pybind11.cc"],
        include_dirs=[
            GetPybindInclude(),
            GetPybindInclude(user=True),
            GetEigenInclude(),
        ],
        language="c++",
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14"):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11"):
        return "-std=c++11"
    else:
        raise RuntimeError(
            "Unsupported compiler -- at least C++11 support " "is needed!"
        )


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }

    if sys.platform == "darwin":
        c_opts["unix"] += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


install_requires = ["numpy>=1.17", "scipy", "pydantic"]
# Only require the pybind11 and peigen packages if
# the C++ headers are not already installed
if pybind11_path is None:
    install_requires.append("pybind11>=2.2")
if eigen_path is None:
    install_requires.append("peigen>=0.0.9")

print(f"\n\ninstall requires: {install_requires} \n\n")

setup(
    name="scarlet_lite",
    packages=packages,
    extras_require={"plotting": ["astropy", "matplotlib"]},
    include_package_data=True,
    description="Blind Source Separation using proximal matrix factorization",
    author="Fred Moolekamp and Peter Melchior",
    author_email="fred.moolekamp@gmail.com",
    url="https://github.com/lsst/scarlet_lite",
    keywords=["astro", "deblending", "photometry", "nmf", "lsst"],
    ext_modules=ext_modules,
    install_requires=install_requires,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
