Utilities
---------

This module contains a collection of utilities for use across the WISDEM model set.

Differentiability
=================
Differentiable versions of several functions are provided (along with their derivatives).
These are summarized in the below table

.. currentmodule:: wisdem.commonse.utilities

.. autosummary::
    :toctree: generated

    linspace_with_deriv
    interp_with_deriv
    arc_length_deriv
    trapz_deriv
    smooth_max
    smooth_min
    smooth_abs
    CubicSplineSegment


.. currentmodule:: wisdem.commonse.xcel_wrapper
Excel Wrapper
==============

.. autosummary::
    :toctree: generated

    ExcelWrapper

Example code for using the xcel_wrapper is included in the xcel_wrapper.py file.
A path to the user's workbook will need to be specified for its use.
