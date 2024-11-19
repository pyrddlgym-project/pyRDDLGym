# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import pyRDDLGym


# -- Project information -----------------------------------------------------

project = 'pyRDDLGym'
copyright = '2024, The pyRDDLGym Project'
author = 'The pyRDDLGym Project'

# The short X.Y version
version = '2.1.0'
# The full version, including alpha/beta/rc tags
release = '2.1.0'


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    #"sphinxcontrib.napoleon",
    "sphinx_book_theme",
    "nbsphinx",
    "sphinx_gallery.load_style",    
    "myst_nb"
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

html_theme_options = {
    "repository_url": "https://github.com/pyrddlgym-project/pyRDDLGym",
    "use_repository_button": True
}

html_title = "Documentation for the pyRDDLGym Project"
html_logo = "rddllogo.gif"

# thumbnails for nbsphinx
nbsphinx_thumbnails = {
    'notebooks/*': 'notebooks/notebook_icon.png'
}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# Sort members by type
autodoc_member_order = "bysource"

autodoc_default_options = {
    # "member-order": "bysource",
    "special-members": "special-members",
    "private-members": "private-members",
}

# -- Options for HTMLHelp output ---------------------------------------------
#
# # Output file base name for HTML help builder.
# htmlhelp_basename = 'pyRDDLGymdoc'
#
#
# # -- Options for LaTeX output ------------------------------------------------
#
# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     #
#     # 'papersize': 'letterpaper',
#
#     # The font size ('10pt', '11pt' or '12pt').
#     #
#     # 'pointsize': '10pt',
#
#     # Additional stuff for the LaTeX preamble.
#     #
#     # 'preamble': '',
#
#     # Latex figure (float) alignment
#     #
#     # 'figure_align': 'htbp',
# }
#
# # Grouping the document tree into LaTeX files. List of tuples
# # (source start file, target name, title,
# #  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#     (master_doc, 'pyRDDLGym.tex', 'pyRDDLGym Documentation',
#      'Ayal Taitler', 'manual'),
# ]
#
#
# # -- Options for manual page output ------------------------------------------
#
# # One entry per manual page. List of tuples
# # (source start file, name, description, authors, manual section).
# man_pages = [
#     (master_doc, 'pyRDDLGym', 'pyRDDLGym Documentation',
#      [author], 1)
# ]
#
#
# # -- Options for Texinfo output ----------------------------------------------
#
# # Grouping the document tree into Texinfo files. List of tuples
# # (source start file, target name, title, author,
# #  dir menu entry, description, category)
# texinfo_documents = [
#     (master_doc, 'pyRDDLGym', 'pyRDDLGym Documentation',
#      author, 'pyRDDLGym', 'One line description of project.',
#      'Miscellaneous'),
# ]