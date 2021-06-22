import sys
from pathlib import Path

# Add the repository root to PYTHONPATH
rtdl_path = Path.cwd()
while not (rtdl_path.name == 'rtdl' and rtdl_path.parent.name != 'rtdl'):
    rtdl_path = rtdl_path.parent
sys.path.append(str(rtdl_path))
import rtdl  # noqa

# >>> Project information <<<
author = 'Yandex Research'
copyright = '2021, Yandex LLC'
project = 'rtdl'
release = rtdl.__version__
version = rtdl.__version__

# >>> General options <<<
default_role = 'py:obj'
pygments_style = 'default'
repo_url = 'https://github.com/yandex-research/rtdl'
templates_path = ['_templates']

# >>> Extensions options <<<
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.spelling',
]

# autoclass_content = 'both'
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
# autodoc_inherit_docstrings = False

autosummary_generate = True

doctest_global_setup = '''
import numpy as np
import torch
import rtdl
from rtdl import *
'''

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = False

spelling_show_suggestions = True

# >>> HTML and theme options <<<
import sphinx_material  # noqa

html_static_path = ['_static']
# html_favicon = 'images/favicon.ico'
# html_logo = 'images/logo.svg'
html_theme = 'sphinx_material'
html_theme_options = {
    # Full list of options: https://github.com/bashtage/sphinx-material/blob/master/sphinx_material/sphinx_material/theme.conf
    'base_url': 'https://yandex-research.github.io/rtdl',
    # Full list of colors (not all of them are available in sphinx-material, see theme.conf above):
    # https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/#primary-color
    # Nice colors: white, blue, red, deep purple (indigo in mkdocs)
    'color_primary': 'red',
    'globaltoc_collapse': False,
    'globaltoc_depth': 2,
    # search here for logo icons: https://www.compart.com/en/unicode
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
