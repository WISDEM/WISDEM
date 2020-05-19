.. known_issues:

Known issues with WISDEM
========================

.. TODO: expand this

There are a number of limitations and assumptions within WISDEM.
This doc page serves as a non-exhaustive record of any issues relevant to the usage of WISDEM.


Analysis limitations
--------------------

Code inefficiencies
-------------------

Common pitfalls
---------------

Improvements yet-to-be-implemented
----------------------------------

General issues
--------------
Some `compute_partials()` call depend on values from `compute()`. This is not a problem if we never have any solvers or gradient-based linesearches on the optimizer. If we use those methods, then the current implementation is not correct and would require refactoring.

Many duplicate variables and names exist within WISDEM. The naming should be consistent throughout and data not duplicated. Many variable names can be changed for clarity.

