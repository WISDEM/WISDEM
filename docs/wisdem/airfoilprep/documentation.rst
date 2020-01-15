.. module:: wisdem.airfoilprep


.. _interfaces-label:

Module Documentation
--------------------

Two classes are provided in the module: :ref:`Polar <airfoilprep.Polar>` and :ref:`Airfoil <airfoilprep.Airfoil>`.  Generally, the Polar class is not needed for direct usage except for its constructor.  All objects in this module are **immutable**.  In other words, calling Airfoil.correct3D() creates a new modified airfoil object rather than editing the existing object.

.. only:: latex

    This PDF version of the documentation only provides an summary of the classes and methods.  Further details are found in the HTML version of this documentation, complete with hyperlinks to the source code.

.. _polar-class-label:

Polar Class
^^^^^^^^^^^
A Polar object is meant to represent the variation in lift, drag, and pitching moment coefficient with angle of attack at a fixed Reynolds number.  Generally, the methods of this class do not need to be used directly (other than the constructor), but rather are used by the :class:`Airfoil <airfoilprep.Airfoil>` class.


.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: Polar
        :members:


.. only:: html

    .. autoclass:: Polar

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~Polar.correction3D
            ~Polar.extrapolate
            ~Polar.blend
            ~Polar.unsteadyparam


.. autogenerate
    .. autosummary::
        :toctree: generated

        ~Polar.unsteadyparam
        ~Polar.blend
        ~Polar.correction3D
        ~Polar.extrapolate



.. _airfoil-class-label:

Airfoil Class
^^^^^^^^^^^^^
An Airfoil object encapsulates the aerodynamic forces/moments of an airfoil as a function of angle of attack and Reynolds number.  For wind turbine analysis, this class provides capabilities to apply 3-D rotational corrections to 2-D data using the Du-Selig method :cite:`Du1998A-3-D-stall-del` for lift, and the Eggers method :cite:`Eggers-Jr2003An-assessment-o` for drag.  Airfoil data can also be extrapolated to +/-180 degrees, using Viterna’s method :cite:`Viterna1982Theoretical-and`.  This class also adds methods to read and write AeroDyn airfoil files directly.

.. Internally, Airfoil uses a two-dimensional cubic B-spline (bisplrep from FITPACK, also known as DIERCKX) fit to the lift and and drag curves separately as functions of Reynolds number and angle of attack. A small amount of smoothing is used on each spline to reduce any high-frequency noise that can cause artificial multiple solutions (0.1 for lift, 0.001 for drag).



.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: Airfoil
        :members:


.. only:: html

    .. autoclass:: Airfoil

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~Airfoil.initFromAerodynFile
            ~Airfoil.correction3D
            ~Airfoil.extrapolate
            ~Airfoil.blend
            ~Airfoil.getPolar
            ~Airfoil.interpToCommonAlpha
            ~Airfoil.createDataGrid
            ~Airfoil.writeToAerodynFile


.. autogenerate
    .. autosummary::
        :toctree: generated

        ~Airfoil.initFromAerodynFile
        ~Airfoil.blend
        ~Airfoil.correction3D
        ~Airfoil.extrapolate
        ~Airfoil.getPolar
        ~Airfoil.writeToAerodynFile
        ~Airfoil.interpToCommonAlpha
        ~Airfoil.createDataGrid



.. role:: bib
   :class: bib

.. only:: html

    :bib:`Bibliography`



.. bibliography:: references.bib


