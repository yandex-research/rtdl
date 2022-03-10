import sys
from pathlib import Path

# Add the repository root to PYTHONPATH
rtdl_path = Path.cwd()
while not (rtdl_path.name == 'rtdl' and rtdl_path.parent.name != 'rtdl'):
    rtdl_path = rtdl_path.parent
sys.path.append(str(rtdl_path))
import rtdl  # noqa

# >>> Project information <<<
author = 'rtdl authors'
copyright = '2021, rtdl authors'
project = 'rtdl'
release = rtdl.__version__
version = rtdl.__version__

# >>> General options <<<
default_role = 'py:obj'
pygments_style = 'default'
repo_url = 'https://github.com/Yura52/rtdl'
templates_path = ['_templates']

# >>> Extensions options <<<
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    # 'sphinxcontrib.spelling',
    'sphinx_copybutton',
]

autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_inherit_docstrings = False

doctest_global_setup = '''
import numpy as np
import torch
import rtdl
import rtdl.data
from rtdl import *
from rtdl.data import *
'''

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = False

# spelling_show_suggestions = True

# >>> HTML and theme options <<<
import sphinx_material  # noqa

html_static_path = ['_static']
html_theme = 'sphinx_material'
html_css_files = ['custom.css']
html_theme_options = {
    'base_url': 'https://Yura52.github.io/rtdl',
    'color_primary': 'red',
    'globaltoc_collapse': False,
    'globaltoc_depth': 2,
    'logo_icon': '&#127968;',
    'nav_links': [],
    'nav_title': project + ' ' + version,
    'repo_name': project,
    'repo_url': repo_url,
    'repo_type': 'github',
    'master_doc': False,
    'version_dropdown': True,
    'version_json': '_static/versions.json',
}
html_sidebars = {
    '**': ['logo-text.html', 'globaltoc.html', 'searchbox.html'],
}
