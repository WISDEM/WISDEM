RotorSE
=======

.. only:: html

    WISDEM simulates horizontal axis wind turbine rotors with steady-state models. The combination of these models is named RotorSE. The models are all written as `OpenMDAO 3.x <http://openmdao.org/>`_ explicit components and are combined into groups. The rotor aerodynamics is solved with the blade element momentum model , the elastic properties of the composite blades are obtained running the cross sectional solver :ref:`precomp`, and the deformations are obtained running the Timoshenko beam solver `Frame3DD <http://frame3dd.sourceforge.net/>`_. A regulation trajectory is implemented and the annual energy production of the turbine is computed here. RotorSE also estimates ultimate loads, rotor blade deflections, and blade strains. Finally, RotorSE includes a detailed blade cost model :ref:`bcm`.
    
    Although analytic gradients are provided for aerodynamic analyses, and some of the helper components, finite differencing is used for optimization. 

    The pages below contain the documentation of the modules composing RotorSE, except for CCBlade that is described here :ref:`ccblade`. 

    .. rubric:: Table of Contents


.. toctree::

    precomp
    bcm
    aep
    loads_deflections_strains
    rail

