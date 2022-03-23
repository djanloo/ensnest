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
import os
import sys
import mock
#import ensnest

sys.path.insert(0, os.path.abspath('..'))
print(f"current path is{sys.path}")
autodoc_mock_imports = ["numpy", 'matplotlib', 'scipy']

# -- Project information -----------------------------------------------------

project = 'ensnest'
copyright = '2021, Gianluca Becuzzi'
author = 'Gianluca Becuzzi'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

latex_engine = 'xelatex'
latex_elements = {
    'fontenc': '\\usepackage{fontenc}',
    'fontpkg': '''\
\\setmainfont{TeX Gyre Termes}
\\setsansfont{TeX Gyre Termes}
\\setmonofont{DejaVu Sans Mono}''',
    'geometry': '\\usepackage[vmargin=2.5cm, hmargin=3cm]{geometry}',
    'preamble': '''\
\\usepackage[titles]{tocloft}
\\cftsetpnumwidth {1.25cm}\\cftsetrmarg{1.5cm}
\\setlength{\\cftchapnumwidth}{0.75cm}
\\setlength{\\cftsecindent}{\\cftchapnumwidth}
\\setlength{\\cftsecnumwidth}{1.25cm}''',
    'fncychap': '\\usepackage[Bjornstrup]{fncychap}',
    'printindex': '\\footnotesize\\raggedright\\printindex',
     'extraclassoptions': 'openany,oneside'
}
latex_show_urls = 'footnote'

#added because of the raise of ones * int error
class Mock(mock.MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return Mock()
    def __mul__(self, other):
        return Mock()
    def __rmul__(self, other):
        return Mock()
    def __pow__(self, other):
        return Mock()
    def __div__(self, other):
        return Mock()
