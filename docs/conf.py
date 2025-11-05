"""Sphinx configuration."""

project = "Np Dist2"
author = "Konstantin Ladutenko"
copyright = "2025, Konstantin Ladutenko"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "shibuya"
