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

cwd = os.getcwd()
wisdempath = os.path.dirname(cwd)
sys.path.insert(0, wisdempath)
sys.path.insert(0, os.path.abspath("./_exts"))


# -- Project information -----------------------------------------------------

project = "WISDEM"
copyright = "2019, NREL WISDEM Team"
author = "NREL WISDEM Team"

# The full version, including alpha/beta/rc tags
release = "2.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    # 'autoapi.extension',
    # 'embed-n2',
]


# Numbering figures in HTML format (always numbered in latex)
numfig = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# TODO : need to set these options correctly.
# Right now we list out all OpenMDAO methods we don't want to document.
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": False,
    "exclude-members": "__weakref__, add_constraint, add_design_var, add_discrete_input, add_discrete_output, add_input, add_objective, add_output, add_recorder, add_response, check_config, cleanup, compute, compute_jacvec_product, compute_partials, convert2units, declare_coloring, declare_partials, get_approx_coloring_fname, get_constraints, get_design_vars, get_linear_vectors, get_nonlinear_vectors, get_objectives, get_responses, initialize, is_active, list_inputs, list_outputs, reconfigure, record_iteration, resetup, run_apply_linear, run_apply_nonlinear, run_linearize, run_solve_linear, run_solve_nonlinear, set_check_partial_options, set_initial_values, set_solver_print, setup, system_iter, use_fixed_coloring, add_subsystem, approx_totals, compute_sys_graph, configure, connect, guess_nonlinear, promotes, set_order",
}

# -- Options for HTML output ----------------------------------------------

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static/custom.css"]
html_theme_options = {
    # "logo": "logo.png",
    # "logo_name": True,
    # "badge_branch": "IEAontology4all",
    # "codecov_button": True,
    # "fixed_sidebar": True,
    # "github_user": "WISDEM",
    # "github_repo": "WISDEM",
    # "sidebar_width": '220px',
    # "page_width": '75%',
    # "show_relbars": True
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = [os.path.join('source', '_static')]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {"**": ["about.html", "navigation.html", "searchbox.html"]}

# Bibtex configuration
bibtex_bibfiles = ["references.bib"]
