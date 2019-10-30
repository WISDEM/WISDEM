.. _tutorial-label:

.. currentmodule:: wisdem.towerse.tower

Tutorial
--------

Two examples are included in this tutorial section: simulation of a land-based tower, and optimization of a land-based tower.

Land-based Tower Simulation
===========================

.. currentmodule:: wisdem.commonse.environment

This following example demonstrates how to setup and run *analysis* for a land-based tower.  First, we import the modules we want to use and setup the tower configuration.  `TowerSE` was designed to be modular, and so the specific modules you wish to use in the analysis must be specified. There are `OpenMDAO Slots <http://openmdao.org/docs/basics/variables.html#slot-variables>`_ for wind1, wind2, wave1, wave2, soil, tower1, and tower2.  The first five make use of the commonse.environment module.  We use multiple wind/wave/tower modules because we are considering two separate loading conditions in this simulation.  The slots wind1 and wind2 can be any component that inherits from :class:`WindBase`, wave1 and wave2 require components that inherit from :class:`WaveBase`, and soil uses components that inherit from :class:`SoilBase`.

In this case we are using :class:`PowerWind`, which defines a power-profile for the wind distribution.  A component, :class:`LogWind` for a logarithmic profile is also available.  We are simulating a land-based turbine so we do not load any wave modules.  The default :class:`WaveBase` module is for no wave loading.  A :class:`LinearWaves` component is available which uses linear wave theory.  A simple textbook-based soil model is provided at :class:`TowerSoil`.  For all slots, users may define any custom component as desired.

.. currentmodule:: wisdem.towerse.tower

Tor tower1 and tower2, the module uses the frame finite element code `Frame3DD <http://frame3dd.sourceforge.net/>`_.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- tower setup
    :end-before: # ---

With the tower configuration setup, we define some of the geometric parameters.  Some of the geometric parameters are seen in :num:`Figure #tower-fig`.  The tower is not restricted to 3 sections, any number of sections can be defined.  The array `z` is given in coordinates nondimensionalized by the tower height.  The array `n`, should of length len(tower.z)-1 and represents the number of finite elements to be used in each tower can.  The float `L_reinforced` is a reinforcement length used in the buckling calculations.  Yaw and tilt are needed to handle to mass/load transfer.  For offshore applications, monopile geometry can also be defined (see :class:`TowerSE`).

.. _tower-fig:

.. figure:: /images/towerse/tower.*
    :height: 4in
    :align: center

    Example of tower geometric parameterization.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- geometry
    :end-before: # ---


We now define the mass properties for the rotor-nacelle-assembly.  The center of mass locations are defined relative to the tower top in the yaw-aligned coordinate system.  Blade and hub moments of inertia should be defined about the hub center, nacelle moments of inertia are defined about the center of mass of the nacelle.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- blades
    :end-before: # ---

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- hub
    :end-before: # ---

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- nacelle
    :end-before: # ---

Environmental properties are defined below, note that the parameters that need to be defined depend on which modules were loaded.  For the power-law wind profile, the only parameter needed is the shear exponent.  For the soil, shear and modulus properties for the soil can be defined, but in this example we assume that all directions are rigid (3 translation and 3 rotation).  In addition, some geometric parameters for the wind profile's extend must be defined, the base (or no-slip location) at `z0`, and the height at which a reference velocity will be defined.


.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- wind
    :end-before: # ---


.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- soil
    :end-before: # ---

As mentioned earlier, we are allowing for two separate loading cases.  The wind speed, and rotor force/moments for those two cases are now defined.  The wind speed location corresponds to the reference height defined previously as `wind_zref`.  In this simple case, we include only thrust and torque, but in general all 3 components of force and moments can be defined in the hub-aligned coordinate system.  The assembly automatically handles translating the forces and moments defined at the rotor to the tower top.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- loading case 1
    :end-before: # ---

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- loading case 2
    :end-before: # ---

Safety factors for loading, material, consequence of failure, and buckling are defined

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- safety factors
    :end-before: # ---

A simplified fatigue analysis is available for the tower.  This requires running an aeroelastic code, like FAST, before hand and inputing the damage equivalent moments.  The locations of the moments are given on a nondimensional tower.  A safety factor, lifetime (in years), and slope of the S-N curve can be defined.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- fatigue
    :end-before: # ---

Finally, some additional parameters used for constraints can be defined.  These include the minimum allowable taper ratio of the tower (from base to top), and the minimum diameter-to-thickness ratio allowed at any section.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- constraints
    :end-before: # ---

In the folllowing specification, we used the default values for wind density and viscosity, material properties (for steel), and acceleration of gravity.  By examining, :class:`TowerSE` the user can see all possible parameters and their defaults.  We can now run the assembly and display some of the outputs.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- run
    :end-before: # ---

The resuling output is shown below.  The outputs f1 and f2 and the first two natural frequencies.  The outputs top_deflection are the deflection of the tower top in the yaw-aligned +x direction for the two loading cases.  Weldability is an array of outputs that should be <0 for feasibility.  They represent margin against the minimum diameter-to-thickness ratio.  Manufacturability represents margin against the minimum taper ratio and should be <0 for feasibility.

>>> mass (kg) = 349486.79362
>>> f1 (Hz) = 0.331531844509
>>> f2 (Hz) = 0.334804545737
>>> top_deflection1 (m) = 0.691606748192
>>> top_deflection2 (m) = 0.708610880714
>>> weldability = [-0.42450142 -0.37541806 -0.30566802]
>>> manufactuability = -0.245

The stress, bucklng, and damage loads are shown in :num:`Figure #utilization-fig`.  Each is a utilization and so should be <1 for feasibility.

.. _utilization-fig:

.. figure:: /images/towerse/utilization.*
    :width: 6in
    :align: center

    Utilization along tower for ultimate stress, shell buckling, global buckling, and fatigue damage.


Land-Based Tower Optimization
=============================

We begin with the same setup as the previous section, but now import additional modules for optimization.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- optimizer imports
    :end-before: # ---

The optimizer must first be selected and configured, in this example I use SNOPT.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- Setup Pptimizer
    :end-before: # ---

We now set the objective, and in this example it is normalized to be of order 1 for better convergence behavior.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- Objective
    :end-before: # ---

The tower diameters, thickness, and waist location are added as design variables.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- Design Variables
    :end-before: # ---

A recorder is added to display each iteration to the screen.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- recorder
    :end-before: # ---

Finally, constraints are added.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- Constraints
    :end-before: # ---

Now the optimization can be run.

.. literalinclude:: ../../../wisdem/towerse/tower.py
    :language: python
    :start-after: # --- run opt
    :end-before: # ---








