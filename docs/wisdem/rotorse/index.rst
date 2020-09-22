RotorSE
=======

.. only:: html

    RotorSE is an aero/structural systems engineering model of a horizontal axis wind turbine rotor.  It has been designed for use in optimization applications, and is written within the `OpenMDAO 3.x <http://openmdao.org/>`_ framework.  Aerodynamic performance is calculated by `CCBlade <http://wind.nrel.gov/designcodes/simulators/ccblade/>`_, a blade element momentum method.  Structural analysis is provided by `PreComp <http://wind.nrel.gov/designcodes/preprocessors/precomp/>`_, a classical laminate cross-section method, and the beam finite element method .  Methods exist for creating power curves, transferring loads and mass properties, estimating tip deflection, etc.  Although analytic gradients are provided for aerodynamic analyses, and some of the helper components, finite differencing is used for optimization.

    .. rubric:: Table of Contents


.. toctree::

    precomp
    bcm
    powercurve
    tutorial
    documentation
    theory

