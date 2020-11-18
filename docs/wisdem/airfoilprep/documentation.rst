.. module:: wisdem.airfoilprep


.. _interfaces-label:

Module Documentation
--------------------
Two classes are provided in the module: :ref:`Polar <polar-class-label>` and :ref:`Airfoil <airfoil-class-label>`. Generally, the Polar class is not needed for direct usage except for its constructor. All objects in this module are **immutable**. In other words, calling ``Airfoil.correct3D()`` creates a new modified airfoil object rather than editing the existing object.

.. _polar-class-label:

Polar Class
^^^^^^^^^^^
A Polar object is meant to represent the variation in lift, drag, and pitching moment coefficient with angle of attack at a fixed Reynolds number. Generally, the methods of this class do not need to be used directly (other than the constructor), but rather are used by the :class:`Airfoil <wisdem.airfoilprep.airfoilprep.Airfoil>` class.


.. module:: wisdem.airfoilprep.airfoilprep

.. autoclass:: wisdem.airfoilprep.airfoilprep.Polar

.. _airfoil-class-label:

Airfoil Class
^^^^^^^^^^^^^
An Airfoil object encapsulates the aerodynamic forces/moments of an airfoil as a function of angle of attack and Reynolds number.
For wind turbine analysis, this class provides capabilities to apply 3-D rotational corrections to 2-D data using the Du-Selig method :cite:`Du1998A-3-D-stall-del` for lift, and the Eggers method :cite:`Eggers-Jr2003An-assessment-o` for drag.
Airfoil data can also be extrapolated to +/-180 degrees, using Viterna’s method :cite:`Viterna1982Theoretical-and`.
This class also adds methods to read and write AeroDyn airfoil files directly.

.. Internally, Airfoil uses a two-dimensional cubic B-spline (bisplrep from FITPACK, also known as DIERCKX) fit to the lift and drag curves separately as functions of Reynolds number and angle of attack. A small amount of smoothing is used on each spline to reduce any high-frequency noise that can cause artificial multiple solutions (0.1 for lift, 0.001 for drag).




.. autoclass:: wisdem.airfoilprep.airfoilprep.Airfoil


.. bibliography:: ../../references.bib
   :filter: docname in docnames
