WISDEM Documentation
====================

.. TODO: Need to update this copy. Is it strictly correct?
The Wind-plant Integrated System Design and Engineering Model (WISDEM) includes integrated assemblies for the assessment of system behavior of wind turbines and plants.
These assemblies are template assemblies that can be tweaked for a given analysis depending on the focus of the work.
For example, any variable in these assemblies can be a design variable, an objective or part of a constraint in an multi-disciplinary optimization.
Any variable can be assigned to a distribution in order to study the behavior of the system under uncertainty or even to perform design/optimization under uncertainty.
The assemblies included here are simply the basic structure for doing a single analysis which can be extended in a multitude of directions.

.. TODO: We need to verify this list of packages, or not list them here, or provide more explanation about what each of them does
The current set of packages included in WISDEM are the NREL Cost and Scaling Model, a series of turbine systems engineering models (RotorSE, DriveSE, DriveWPACT and TowerSE) along with plant cost and energy production models (Plant_EnergySE, Turbine_CostsSE, Plant_CostsSE, Plant_FinanceSE).

Author: `NREL WISDEM Team <mailto:systems.engineering+WISDEM_Docs@nrel.gov>`_

Using WISDEM
============

.. toctree::
   :maxdepth: 2

   installation
   how_wisdem_works
   tutorials
   examples
   modules
   
   
Other Useful Docs
=================

.. toctree::
   :maxdepth: 2

   known_issues
   how_to_write_docs
   how_to_contribute_code
   
Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
