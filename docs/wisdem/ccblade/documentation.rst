.. module:: wisdem.ccblade.ccblade

.. _ccblade_interfaces-label:

Module Documentation
--------------------

The main methodology is contained in :ref:`CCBlade <ccblade-class-label>`.
Airfoil data is provided by any object that implements :ref:`AirfoilInterface <airfoil-interface-label>`.
The helper class :ref:`CCAirfoil <ccairfoil-class-label>` is provided as a useful default implementation for AirfoilInterface.
If CCAirfoil is not used, the user must provide an implementation that produces :math:`C^1` continuous output (or else accept non-smooth aerodynamic calculations from CCBlade).
Some of the underlying implementation for CCBlade is written in Fortran for computational efficiency.

.. _airfoil-interface-label:

Airfoil Interface
^^^^^^^^^^^^^^^^^
The airfoil objects used in CCBlade need only implement the following evaluate() method.
Although using :ref:`CCAirfoil <ccairfoil-class-label>` for the implementation is recommended, any custom class can be used.


.. _ccairfoil-class-label:

CCAirfoil Class
^^^^^^^^^^^^^^^
CCAirfoil is a helper class used to evaluate airfoil data with a continuously differentiable bivariate spline across the angle of attack and Reynolds number.
The degree of the spline polynomials across the Reynolds number is summarized in the following table (the same applies to the angle of attack although generally, the number of points for the angle of attack is much larger).

    TABLE CAPTION:: Degree of spline across Reynolds number.

========= =====================
len(Re)    degree of spline
========= =====================
1            constant
2            linear
3            quadratic
4+           cubic
========= =====================

.. _ccblade-class-label:

CCBlade Class
^^^^^^^^^^^^^
This class provides aerodynamic analysis of wind turbine rotor blades using BEM theory.
It can compute distributed aerodynamic loads and integrated quantities such as power, thrust, and torque.
An emphasis is placed on convergence robustness and differentiable output so that it can be used with gradient-based optimization.


Polar Class
^^^^^^^^^^^
A Polar object is meant to represent the variation in lift, drag, and pitching moment coefficient with angle of attack at a fixed Reynolds number. Tools exist to read in two-dimensional (2-D) aerodynamic airfoil data (i.e., from wind tunnel data or numerical simulation), apply three-dimensional (3-D) rotation corrections for wind turbine applications, and extend the data to very large angles of attack.  Airfoil data can also be blended together to define intermediate sections between linearly lofted sections.

.. module:: wisdem.ccblade.Polar

.. autoclass:: wisdem.ccblae.Polar.Polar

.. _polar-class-label:
