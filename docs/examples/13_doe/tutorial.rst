.. _doe_tutorial-label:

13. Design of Experiments Example(s)
----------------------------------------

Many WISDEM studies involve a Design of Experiments, where inputs are parameterized across a range of values or sampled from a distribution.  At each combinatoric point, WISDEM is run either as a single evaluation or as an optimization to create a family of optimized designs.  A WISDEM user can execute a design of experiments either through WISDEM's integration with OpenMDAO or by creating a customized wrapper that controls the parameterization in an outer loop and calls WISDEM directly.  Examples of both approaches are provided.

Design of Experiments with OpenMDAO
=============================================

As an alternative to its optimization drivers, OpenMDAO provides a `DOEDriver <https://openmdao.org/newdocs/versions/latest/features/building_blocks/drivers/doe_driver.html>`_ to run a design of experiments.  WISDEM can invoke this capability through user options in the analysis options file.  To activate a design of experiments run in WISDEM, set the design variable flag to True for the input variables that should be parameterized in the run.  In the `doe_with_openmdao.py` example provided, the blade chord is the only design variable.  Next, in the ``driver`` section, the following options are available:

.. literalinclude:: ../../../examples/13_design_of_experiments/analysis_options.yaml
    :language: yaml
    :start-after: driver:


With the OpenMDAO DOEDriver, the results are stored in the recorded SQL-file, so be sure the ``recorder`` option is set to `True`.  The OpenMDAO docs can guide how to open and read the data from the file.


Design of Experiments via Python Scripting (and Multiprocessing)
=================================================================

Every study is unique and WISDEM users often prefer to customize their own wrapper that executes a design of experiments with specific variables or sampling or output analysis.  The `doe_custom.py` example one way to implement this.

.. literalinclude:: ../../../examples/13_design_of_experiments/doe_custom.py
    :language: python

In this example, the blade cone angle and shaft tilt angle are varied from zero to five degrees (steps of two) and the turbine annual energy production (AEP) is logged at each point in the parametric grid.  The WISDEM runs can be called in serial within the for loop over the input values, or the arguments themselves are stored and then sent to a parallel processing "pool" using the `multiprocessing` library.
