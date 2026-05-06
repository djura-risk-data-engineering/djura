import os
import sys
import warnings

sys.path.insert(0, os.path.abspath("../../src"))

# np.bool FutureWarning originates inside sphinx_autodoc_typehints internals.
warnings.filterwarnings(
    "ignore",
    message=".*np\\.bool.*",
    category=FutureWarning,
)

project = "djura"
copyright = "2025–2026, Djura | Risk - Data - Engineering S.r.l."
author = "Djura | Risk - Data - Engineering S.r.l."
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

# sphinx-autodoc-typehints is only available on Python >=3.12
try:
    import sphinx_autodoc_typehints  # noqa: F401
    extensions.append("sphinx_autodoc_typehints")
except ImportError:
    pass

exclude_patterns = []

# Napoleon — NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc defaults
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

html_theme = "furo"
html_title = "djura"

# Suppress warnings from third-party library internals that we cannot fix.
suppress_warnings = [
    # pydantic v2 forward references (JsonValue) inspected by autodoc
    "sphinx_autodoc_typehints.forward_reference",
    # pydantic classmethod subscript issue on Python 3.12
    "sphinx_autodoc_typehints.guarded_import",
]
