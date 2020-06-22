Introduction
------------

CCBlade predicts aerodynamic loading of wind turbine blades using blade element momentum (BEM) theory.
CC stands for continuity and convergence.
CCBlade was developed primarily for use in gradient-based optimization applications where :math:`C^1` continuity and robust convergence are essential.

Typical BEM implementations use iterative solution methods to converge the induction factors (e.g., fixed-point iteration or Newton's method).
Some more complex implementations use numerical optimization to minimize the error in the induction factors.
These methods can be fairly robust, but all have at least some regions where the algorithm fails to converge.
A new methodology was developed that is provably convergent in every instance (see :ref:`ccblade_theory`).
This robustness is particularly important for gradient-based optimization.
To ensure :math:`C^1` continuity, lift and drag coefficients are computed using a bivariate cubic spline across angle of attack and Reynolds number.
Additionally, analytic gradients for distributed loads, thrust, torque, and power are (optionally) provided.

CCBlade is written in Python, but iteration-heavy sections are written in Fortran in order to improve performance.
The Fortran code is called from Python as an extension module using `f2py <http://https://numpy.org/doc/1.17/f2py/index.html>`_.
The module AirfoilPrep.py is also included with the source.
Although not directly used by CCBlade, the airfoil preprocessing capabilities are often useful for this application.
