.. _known_issues:

Known issues within WISDEM
==========================

This doc page serves as a non-exhaustive record of any issues relevant to the usage of WISDEM.
Some of these items are features that would be nice to have, but are not necessarily in the pipeline of development.

General issues
--------------
The components within WISDEM do not provide analytic gradients, so any gradient-based optimization using WISDEM uses finite-differenced gradients at the top level.

Many duplicate variables and names exist within WISDEM. The naming should be consistent throughout and data not duplicated. Many variable names can be changed for clarity.

Common pitfalls
---------------
Depending on the type of design study being done, not all of the disciplines available must be included.
For example, if you are interested in the aerodynamic performance of the turbine blades, you can turn off all calculations related to the drivetrain, tower, costs, etc, to save on computational expense.

If you use an existing set of .yaml files and adapt them for your turbine analysis, make sure to update all values as needed.
For example, if using a land-based 5MW reference turbine as a starting point for an offshore 5MW turbine, you will need to change the tower foundation properties.

Improvements yet-to-be-implemented
----------------------------------
More complete documentation and comments throughout the code would be beneficial.

Portions of the codebase were written before external packages existed and the codebase could be simplified to take advantage of some built-in Scipy and OpenMDAO features now.

