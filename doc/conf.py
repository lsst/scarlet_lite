"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
"""

# ruff: noqa: F403, F405

from documenteer.conf.guide import *

# Many __init__ methods in scarlet_lite are typed but lack a Parameters
# section. sphinx_autodoc_typehints injects a synthetic Parameters section
# that collides with the class docstring, producing a SEVERE parse error.
# Disable the extension until __init__ docstrings are normalized.
extensions = [e for e in extensions if e != "sphinx_autodoc_typehints"]
