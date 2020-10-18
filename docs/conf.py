# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
""" import os
import sys
sys.path.insert(0, os.path.abspath('.'))
 """

# -- Project information -----------------------------------------------------

project = 'qsearch'
copyright = '2020, Marc Grau Davis, Ethan Smith'
author = 'Marc Grau Davis, Ethan Smith'

# The full version, including alpha/beta/rc tags
release = '2.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'recommonmark',
    'sphinx.ext.autodoc',
    'sphinx.ext.autodoc.typehints',
    'sphinx.ext.napoleon',
    'autoapi.extension',
]

from autoapi.mappers.python import objects
import inspect


@property
def constructor_docstring(self):
    docstring = ""

    constructor = self.constructor
    if constructor and constructor.docstring:
        docstring = constructor.docstring
    else:
        for child in self.children:
            if child.short_name == "__new__":
                docstring = child.docstring
                break

    return docstring.replace(inspect.getdoc(object.__init__), '')


objects.PythonClass.constructor_docstring = constructor_docstring

autoapi_python_class_content = 'both'

autoapi_type = 'python'
autoapi_dirs = ['../qsearch']

autodoc_typehints = 'description'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'
