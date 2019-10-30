.. _tutorial-label:

.. currentmodule:: rotorse.rotoraero

Tutorial
--------

NOTE: this tutorial needs to be updated when the full DTU10MW model is available.  Currently using the old composite layups.

The module :mod:`rotorse.rotor_aeropower` contains methods for generating power curves and computing annual energy production (AEP) with any aerodynamic tool, any wind speed distribution (implementing :mod:`PDFBase` and :mod:`PDFBase`), any drivetrain efficiency function (implementing :mod:`DrivetrainLossesBase`), and any machine type amongst the four combinations of variable/fixed speed and variable/fixed pitch.

The module :mod:`rotorse.rotor_geometry` provides specific implementations of reference rotor designs (implementing :mod:`ReferenceBlade`).  `CCBlade <https://github.com/WISDEM/CCBlade>`_ is used for the aerodynamic analysis and `CommonSE <https://github.com/WISDEM/CommonSE>`_ provides Weibull and Rayleigh wind speed distribution.

The module :mod:`rotorse.rotor_structure` provides structural analyses including methods for managing the composite secion analysis, computing deflections, computing mass properties, etc.  The module :mod:`rotorse.rotor` provides the coupling between rotor_aeropower and rotor_sturucture for combined analysis.  

Two examples are included in this tutorial section: aerodynamic simulation and optimization of a rotor and aero/structural analysis of a rotor.



Rotor Aerodynamics
==================

.. currentmodule:: rotorse.rotor_aeropower

This example is available at :mod:`rotorse.examples.rotorse_example1` or can be viewed as an interactive Jupyter notebook _____.  The first step is to import the relevant files.

.. literalinclude:: ../src/rotorse/examples/rotorse_example1.py
    :language: python
    :start-after: # --- Import Modules
    :end-before: # ---

When setting up our Problem, a rotor design that is an implimentation of :mod:`ReferenceBlade`, is used to initialize the Group.  Two reference turbine designs are included as examples in :mod:`rotorse.rotor_geometry`, the :mod:`NREL5MW` and the :mod:`DTU10MW`.  For this tutorial, we will be working with the DTU 10 MW.

.. literalinclude:: ../src/rotorse/examples/rotorse_example1.py
    :language: python
    :start-after: # --- Init Problem
    :end-before: # ---

A number of input variablers covering the blade geometry, atmospheric conditions, and controls system must be set by the use.  While the reference blade design provides this information, it must be again set at the Problem level.  This provides flexibility for modifications by the user or by an optimizer.  The user can choose to use the default values

.. literalinclude:: ../src/rotorse/examples/rotorse_example1.py
    :language: python
    :start-after: # --- default inputs
    :end-before: # ---

Or set their own values.  First, the geometry is defined.  Spanwise blade variables such as chord and twist are definied using control points, which :class:`BladeGeometry` uses to generate the spanwise distribution using Akima splines according to :num:`Figures #chord-param-fig` and :num:`#twist-param-fig`.

.. literalinclude:: ../src/rotorse/examples/rotorse_example1b.py
    :language: python
    :start-after: # === blade grid ===
    :end-before: # ---

.. _chord-param-fig:

.. figure:: /images/chord_dtu10mw.*
    :height: 3in
    :align: left

    Chord parameterization

.. _twist-param-fig:

.. figure:: /images/theta_dtu10mw.*
    :height: 3in
    :align: center

    Twist parameterization

Atmospheric properties are defined.  The wind speed distribution parameters are determined based on the wind turbine class.

.. literalinclude:: ../src/rotorse/examples/rotorse_example1b.py
    :language: python
    :start-after: # === atmosphere ===
    :end-before: # ---

The relevant control parameters are set

.. literalinclude:: ../src/rotorse/examples/rotorse_example1b.py
    :language: python
    :start-after: # === control ===
    :end-before: # ---

Finally, a few configuation parameters are set.  The the following drivetrain types are supported: 'geared', 'single_stage', 'multi_drive', or 'pm_direct_drive'.

.. literalinclude:: ../src/rotorse/examples/rotorse_example1b.py
    :language: python
    :start-after: # === aero and structural analysis options ===
    :end-before: # ---

We can now run the analysis, print the outputs, and plot the power curve.

.. literalinclude:: ../src/rotorse/examples/rotorse_example1.py
    :language: python
    :start-after: # === run and outputs ===
    :end-before: # ---


>>> AEP = 46811339.16312428
>>> diameter = 197.51768195144518
>>> ratedConditions.V = 11.674033110109226
>>> ratedConditions.Omega = 8.887659696962098
>>> ratedConditions.pitch = 0.0
>>> ratedConditions.T = 1514792.8710181064
>>> ratedConditions.Q = 10744444.444444444

.. figure:: /images/power_curve_dtu10mw.*
    :height: 3in
    :align: center

    Power curve


Rotor Aerodynamics Optimization
===============================

This section describes a simple optimization continuing off of the same setup as the previous section.  This example is available at :mod:`rotorse.examples.rotorse_example2` or can be viewed as an interactive Jupyter notebook _____.  First, we import relevant modules and initialize the problem.

.. literalinclude:: ../src/rotorse/examples/rotorse_example2.py
    :language: python
    :start-after: # --- Import Modules
    :end-before: # ---

The optimizer must be selected and configured, in this example I choose SLSQP.

.. literalinclude:: ../src/rotorse/examples/rotorse_example2.py
    :language: python
    :start-after: # --- Optimizer
    :end-before: # ---

We now set the objective, and in this example it is normalized by the starting AEP for better convergence behavior.

.. literalinclude:: ../src/rotorse/examples/rotorse_example2.py
    :language: python
    :start-after: # --- Objective
    :end-before: # ---

The rotor chord, twist, and tip-speed ratio in Region 2 are added as design variables.

.. literalinclude:: ../src/rotorse/examples/rotorse_example2.py
    :language: python
    :start-after: # --- Design Variables
    :end-before: # ---

A recorder is added to display each iteration to the screen.

.. literalinclude:: ../src/rotorse/examples/rotorse_example2.py
    :language: python
    :start-after: # --- Recorder
    :end-before: # ---

Input variables must be set, see previous example.

.. literalinclude:: ../src/rotorse/examples/rotorse_example2.py
    :language: python
    :start-after: # --- Setup
    :end-before: # ---

Running the optimization (may take several minutes) yields a new design with a 4.83% percent increase in AEP.

.. literalinclude:: ../src/rotorse/examples/rotorse_example2.py
    :language: python
    :start-after: # --- run and outputs
    :end-before: # ---

>>> Max Chord Radius =  0.15
>>> Chord Control Points =  [4.34251278 5.86333664 4.73321059 2.69928949 1.02802714]
>>> Twist Control Points =  [14.43318815 13.24959516  9.19134204  0.44825328 -3.17382031]
>>> TSR =  10.227899944164527
>>> ----------------
>>> Objective =  -1.0483339663298121
>>> AEP =  50439020.74897289
>>> Rated Thrust = 1477743.3251388562
>>> Thrust percent change = 1.9999996705035588

.. _chord-param-opt-fig:

.. figure:: /images/chord_opt_10mw.*
    :height: 3in
    :align: left

    Optimized Chord

.. _twist-param-opt-fig:

.. figure:: /images/theta_opt_10mw.*
    :height: 3in
    :align: center

    Optimized Twist


Rotor Aero/Structures
=====================

This examples includes both aerodynamic and structural analysis.  It is available at :mod:`rotorse.examples.rotorse_example3` or can be viewed as an interactive Jupyter notebook _____.  In this case, they are not fully coupled.  The aerodynamic loads feed into the structural analysis, but there is no feedback from the structural deflections.  We first import the modules we will use and instantiate the objects.

.. literalinclude:: ../src/rotorse/examples/rotorse_example3.py
    :language: python
    :start-after: # --- Import Modules
    :end-before: # ---

Initial grids are set.  From these definitions only changes to the aerodynamic grid needs to be specified (through ``r_aero`` in the next section) and the locations along the aerodynamic and structural grids will be kept in sync.

.. literalinclude:: ../src/rotorse/examples/rotorse_example3.py
    :language: python
    :start-after: # === blade grid ===
    :end-before: # ---

Next, geometric parameters are defined.

.. literalinclude:: ../src/rotorse/examples/rotorse_example3.py
    :language: python
    :start-after: # === blade geometry ===
    :end-before: # ---

The atmospheric data also includes defining the IEC turbine and turbulence class, which are used to compute the average wind speed for the site and the survival wind speed.

.. literalinclude:: ../src/rotorse/examples/rotorse_example3.py
    :language: python
    :start-after: # === atmosphere
    :end-before: # ---

Parameters are defined for the steady-state control conditions.

.. literalinclude:: ../src/rotorse/examples/rotorse_example3.py
    :language: python
    :start-after: # === control
    :end-before: # ---

Various optional parameters for the analysis can be defined.

.. literalinclude:: ../src/rotorse/examples/rotorse_example3.py
    :language: python
    :start-after: # === aero and structural analysis options
    :end-before: # ---

A simplistic fatigue analysis can be done if damage equivalent moments are supplied.

.. literalinclude:: ../src/rotorse/examples/rotorse_example3.py
    :language: python
    :start-after: # === fatigue
    :end-before: # ---

Finally, we run the assembly and print/plot some of the outputs.  :num:`Figures #strain-spar-fig` and :num:`#strain-te-fig` show the strian distributions for the suction and pressure surface, as well as the critical strain load for buckling, in both the spar caps and trailing-edge panels.

.. literalinclude:: ../src/rotorse/examples/rotorse_example3.py
    :language: python
    :start-after: # === run and outputs
    :end-before: # ---


>>> AEP = 48113504.25433461
>>> diameter = 197.51768195144518
>>> rated_V = 11.489317285434028
>>> rated_Omega = 8.887659696962098
>>> rated_pitch = 0.0
>>> rated_T = 1448767.9705024462
>>> rated_Q = 10744444.444444444
>>> mass_one_blade = 41104.297487659685
>>> mass_all_blades = 123312.89246297906
>>> I_all_blades = [1.55254884e+08 7.41578819e+07 5.82484184e+07 0.0e+00 0.0e+00 0.0e+00]
>>> freq = [0.53916149 0.84197813 1.21966084 2.09334462 2.15778794]
>>> tip_deflection = 38.41878836125684
>>> root_bending_moment = 38539069.00864485


.. _strain-spar-fig:

.. figure:: /images/strain_spar_dtu10mw.*
    :height: 3in
    :align: left

    Strain in spar cap

.. _strain-te-fig:

.. figure:: /images/strain_te_dtu10mw.*
    :height: 3in
    :align: center

    Strain in trailing-edge panels




