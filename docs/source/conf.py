"""Sphinx configuration."""
import os
import sys 

sys.path.insert(0, os.path.abspath('../../sonicdb'))

project = "SONICDB"
author = "Daniel Kadyrov"
copyright = "2024, Daniel Kadyrov"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Support for Google and NumPy-style docstrings
    "sphinx_click",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx.ext.autosummary"
]
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,            # Include class members
    "undoc-members": True,      # Include undocumented members
    "show-inheritance": True,   # Show class inheritance
}
html_theme = "furo"  # Use the Furo theme for a modern look
html_theme_options = {
    "sidebar_hide_name": False,       # Ensure the project name is visible in the sidebar
}
html_static_path = ["_static"]  # Ensure static files like CSS are included
html_css_files = [
    "custom.css",  # Add custom CSS for styling
]
master_doc = "index"
myst_enable_extensions = [
    "colon_fence",  # For ::: fenced code blocks    "deflist",      # For definition lists
    "linkify",      # Auto-detect URLs and turn them into links]
    "substitution", # For text substitution
]
html_writer_class = "myst_parser.sphinx_.MdParserConfig"  # Output Markdown
autosummary_generate = True