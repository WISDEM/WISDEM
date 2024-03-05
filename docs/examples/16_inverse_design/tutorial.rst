.. _inverse_design_tutorial-label:

16. Inverse Design Example
-----------------------------

This example walks through an inverse design of the rotor and an inverse design of a floating spar.

Rotor Design
===============
The focus of this example is in the file `analysis_options_rotor.yaml`. 

The field `general` lists the standard output folder and naming convention.

The field `design_variables` lists all active design variables parametrizing the blade design. The quantities parametrize aerodynamic twist and chord along blade span and structural layers. Blade structure is defined by many layers. This example is included in the regression tests of WISDEM and a toy-problem is defined where only the layer `Shell_skin_outer` can vary. More layers are commented. These are `Spar_cap_ss`, `Spar_cap_ps`, `LE_reinf`, `TE_reinf_ss`, `TE_reinf_ps`, `Shell_skin_inner`. In a real inverse design problem, make sure to uncomment them. Also, it is recommended to increase the number of spanwise points for twist, chord, and layers. A good starting point is provided below.

Twist can be parametrized at 8 stations along span, where the active design variables are the outer 6, whereas the inner 2 are locked (where the blade is cylindrical or almost cylindrical and twist is not well defined). At the 6 control points, twist can increase up to 0.087 rad (5 deg) and decrease up to 5 deg. Similarly, chord should be parametrized at 8 stations (keep only the outer 6 free) and can go up and down from 300% to 30% of the initial values.  The structural layers that are commented in example, when uncommented, could vary between 20% and 500% of the initial value. Start and end indices are chosen to ensure manufacturability of the blade (skin has to be thick at the root to accommodate bolts, spar caps and reinforcements need to start and end thin, etc).

The `merit_figure` of the study is LCOE.

The field `constraints` lists several constraints, the ones marked with `flag: True` are active. The active ones limit 
* maximum strains along the length of the spar caps
* ultimate blade tip deflection not violating minimum tower clearance, with a standard safety factor of 1.35 * 1.05 = 1.4175
* minimum margin to stall of 5 degrees (0.087 rad)
* maximum blade root flapwise moment coefficient of 0.16
* first blade flapwise frequency between 0.48 and 0.52 Hz (static natural frequency, 0 rpm, no aerodynamics)
* first blade edgewise frequency between 0.73 and 0.77 Hz (static natural frequency, 0 rpm, no aerodynamics)
* first blade torsional frequency between 6.1 and 6.5 Hz
* blade mass between 34,500 and 35,500 kg

The `optimizer` is the standard SciPy SLSQP, and the openmdao recorder is turned on. Note that a real optimization will need a much higher value for `max_iter` (100 is a good start).



Spar Design
===============
Work in progress!