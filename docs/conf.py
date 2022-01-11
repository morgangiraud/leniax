"""Configuration file for the Sphinx documentation builder."""

import os
import sys
import sphinx

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Leniax'
copyright = '2022, The Leniax authors'  # pylint: disable=redefined-builtin
author = 'Morgan Giraud'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'recommonmark',
    'sphinx_markdown_tables'
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ['.rst', '.md']

autosummary_generate = True

master_doc = 'index'

autodoc_typehints = 'description'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_style = 'css/leniax_theme.css'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = './leniax.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

nbsphinx_codecell_lexer = 'ipython3'

nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. only:: html
    .. role:: raw-html(raw)
        :format: html
    .. nbinfo::
        :raw-html:`<a href="https://github.com/morgangiraud/leniax/blob/main/{{ docname }}"><img alt="Open On GitHub" src="https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub" style="vertical-align:text-bottom"></a>`
"""

# -- Extension configuration -------------------------------------------------

# Tell sphinx-autodoc-typehints to generate stub parameter annotations including
# types, even if the parameters aren't explicitly documented.
always_document_param_types = True