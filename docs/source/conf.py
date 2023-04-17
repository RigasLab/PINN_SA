# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
sys.path
dirn = 'C:\\Users\\Yusuf\\Source\\Repos\\ysp15\\Oak\\Hawth2'
sys.path.insert(0, pathlib.Path(dirn).resolve().as_posix())
#sys.path.insert(0, pathlib.Path(dirn).parents[2].resolve().as_posix())
#sys.path.insert(0, pathlib.Path("././docseee").parents[2].resolve().as_posix())
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PINN-SA'
copyright = '2023, Yusuf Patel'
author = 'Yusuf Patel'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
        'sphinx.ext.duration',
        'sphinx.ext.doctest',
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx.ext.viewcode',
        'sphinx.ext.mathjax',
        'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
