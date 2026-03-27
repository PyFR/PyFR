import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# -- General configuration ------------------------------------------------

needs_sphinx = '5.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.viewcode',
    'sphinxcontrib.contentui',
]

autosectionlabel_prefix_document = True

# Inheritance diagram settings
graphviz_output_format = 'svg'
inheritance_graph_attrs = dict(
    center='true',
    layout='dot',
    rankdir='LR',
    ranksep='0.05',
    splines='ortho',
    ratio='compress')
inheritance_node_attrs = dict(
    color='"#333333"',
    fillcolor='"#d8e6b8"',
    fontsize='6.5',
    penwidth='0.3',
    shape='box',
    style='"rounded,filled"')
inheritance_edge_attrs = dict(
    color='"#333333"',
    penwidth='0.3')

# Modules to mock for autodoc
autodoc_mock_imports = ['h5py', 'mpi4py', 'gimmik', 'numpy', 'mako',
                        'platformdirs', 'rtree', 'pyfr.mpiutil']

source_suffix = '.rst'
root_doc = 'index'

project = 'PyFR'
copyright = '2013\u20132026, Imperial College London'
version = '3.1'
release = '3.1'

exclude_patterns = []

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_logo = '../fig/logo.svg'
html_favicon = '../fig/pyfr_favicon.png'

html_theme_options = {
    'analytics_anonymize_ip': False,
    'logo_only': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#94b24c',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_title = 'Documentation'
html_short_title = 'PyFR - Documentation'

html_static_path = ['_static']
html_css_files = ['css/custom.css']

htmlhelp_basename = 'PyFRdoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    'preamble': r'\setcounter{secnumdepth}{4}',
}

latex_documents = [
    ('index', 'PyFR.tex', 'PyFR Documentation',
     'Imperial College London', 'manual'),
]
