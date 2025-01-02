.. _inverse_design_tutorial-label:

16. Inverse Design Example
-----------------------------

This example walks through an inverse design of the wind turbine rotor and an inverse design of a floating spar.

Rotor Design
===============
This example shows how to redesign a wind turbine rotor so that certain user-defined outputs are matched. This process is often performed when we are trying to reproduce the design of an existing wind turbine where the numerical model has not been shared by the manufacturer or the design of a future wind turbine where we are trying to guess how this design will look like. This inverse design process is commonly required by wind farm developers and operators, who do not receive all turbine characteristics from the turbine manufacturers, but still need to model their turbines numerically.

In this example, we start from a 5MW land-based wind turbine that was developed within the Big Adaptive Rotor project (for more details refer to https://github.com/NREL/BAR_Designs) and we redesign the rotor so that the first frequencies and the blade mass match some user-defined quantities.

The focus of this example is in the file ``analysis_options_rotor.yaml``. 

The field ``general`` lists the standard output folder and naming convention.

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_rotor.yaml
    :language: python
    :end-before: # design variables

The field ``design_variables`` lists all active design variables parametrizing the blade design. The quantities parametrize aerodynamic twist and chord along blade span and structural layers. Blade structure is defined by many layers. This example is included in the regression tests of WISDEM and a toy-problem is defined where only the layer ``Shell_skin_outer`` can vary. More layers are commented. These are ``Spar_cap_ss``, ``Spar_cap_ps``, ``LE_reinf``, ``TE_reinf_ss``, ``TE_reinf_ps``, ``Shell_skin_inner``. In a real inverse design problem, make sure to uncomment them. Also, it is recommended to increase the number of spanwise points for twist, chord, and layers. A good starting point is provided below.

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_rotor.yaml
    :language: python
    :start-after: # design variables
    :end-before: # figure of merit

Twist can be parametrized at 8 stations along span, where the active design variables are the outer 6, whereas the inner 2 are locked (where the blade is cylindrical or almost cylindrical and twist is not well defined). At the 6 control points, twist can increase up to 0.087 rad (5 deg) and decrease up to 5 deg. Similarly, chord should be parametrized at 8 stations (keep only the outer 6 free) and can go up and down from 300% to 30% of the initial values.  The structural layers that are commented in example, when uncommented, could vary between 20% and 500% of the initial value. Start and end indices are chosen to ensure manufacturability of the blade (skin has to be thick at the root to accommodate bolts, spar caps and reinforcements need to start and end thin, etc).

The ``merit_figure`` of the study is LCOE. While LCOE is not always the right merit figure (the solution space is usually very flat and possibly multi-modal), in this example we use it to combine aerodynamic and structural considerations into a single metric.

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_rotor.yaml
    :language: python
    :start-after: # figure of merit
    :end-before: # constraints

The field ``constraints`` lists several constraints, the ones marked with ``flag: True`` are active. The active ones limit 

- maximum strains along the length of the spar caps
- ultimate blade tip deflection not violating minimum tower clearance, with a standard safety factor of 1.35 * 1.05 = 1.4175
- minimum margin to stall of 5 degrees (0.087 rad)
- maximum blade root flapwise moment coefficient of 0.16
- first blade flapwise frequency between 0.48 and 0.52 Hz (static natural frequency, 0 rpm, no aerodynamics). In the real world, users might have be able to infer this value from SCADA data.
- first blade edgewise frequency between 0.73 and 0.77 Hz (static natural frequency, 0 rpm, no aerodynamics). As above, in the real world, users might have be able to infer this value from SCADA data.
- first blade torsional frequency between 6.1 and 6.5 Hz
- blade mass between 34,500 and 35,500 kg. This value is often released publicly, or made available in shipping documents.

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_rotor.yaml
    :language: python
    :start-after: # constraints
    :end-before: # driver

The ``optimizer`` is the standard SciPy SLSQP, and the openmdao recorder is turned on. Note that a real optimization will need a much higher value for ``max_iter`` (100 is a good start).

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_rotor.yaml
    :language: python
    :start-after: # driver

For more details about this example, please feel free to post questions on GitHub!


Spar Design
===============
In this example we setup an optimization problem to redesign a floating spar. The design variables parametrize the keel and the freeboard of the spar, where both quantities are bound between -40 and -15 meters. In addition, the ballast is optimized and can vary between 1 kg and 10,000 kg. Lastly, the length of the mooring line can vary between 100 and 1,000 meters.

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_spar.yaml
    :language: python
    :start-after: # design variables
    :end-before: # figure of merit

In this optimization problem, we set the optimizer to minimize the difference between certain outputs and their target values. WISDEM allows to do that by setting ``inverse_design`` as the ``merit_figure`` and then by setting a user-defined list of outputs under ``inverse_design``. In this example, these outputs are:

- the mass of the platform hull
- the mass of the mooring system
- the location of the center of mass of the floating spar

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_spar.yaml
    :language: python
    :start-after: # figure of merit
    :end-before: # driver

The ``optimizer`` is the standard SciPy SLSQP. Note that a real optimization will need a much higher value for ``max_iter`` (100 is a good start).

.. literalinclude:: ../../../examples/16_inverse_design/analysis_options_spar.yaml
    :language: python
    :start-after: # driver

Once again, for more details about this example, please feel free to post questions on GitHub!

