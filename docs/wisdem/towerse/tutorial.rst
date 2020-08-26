.. _tutorial-label:

.. currentmodule:: wisdem.towerse.tower

Tutorial
--------

In this example we show how to perform simulation and optimization of a land-based tower.
This example is contained in the ``WISDEM/examples/tower/example.py`` file.

Land-based Tower Simulation
===========================

.. currentmodule:: wisdem.towerse.tower

This following example demonstrates how to set up and run analysis or optimization for a land-based tower.
Some of the geometric parameters are seen in :numref:`Figure %s <tower-fig>`.
The tower is not restricted to 3 sections, any number of sections can be defined.
The array `z` is given in coordinates nondimensionalized by the tower height.
The array `n`, should of length len(tower.z)-1 and represents the number of finite elements to be used in each tower can.
Yaw and tilt are needed to handle to mass/load transfer.
For offshore applications, monopile geometry can also be defined (see :class:`TowerSE`).

.. _tower-fig:

.. figure:: /images/towerse/tower.*
    :height: 4in
    :align: center

    Example of tower geometric parameterization.

First, we import the modules we want to use and setup the tower configuration.
We set flags for if we want to perform analysis or optimization, as well as if we want plots to be shown at the end.
Next, we set the tower height, diameter, and wall thickness.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # Set analysis and optimization
    :end-before: # ---
    
We then set many analysis options for the tower, including materials, safety factors, and FEM settings.
The module uses the frame finite element code `Frame3DD <http://frame3dd.sourceforge.net/>`_ to perform the FEM analysis.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # Store analysis options in dictionary
    :end-before: # ---

Next, we instantiate the OpenMDAO problem and add a tower model to this problem.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # Instantiate OpenMDAO
    :end-before: # ---
    
Next, we have logic in the script to add the optimization problem if ``opt_flag`` is set to True.
The optimizer must first be selected and configured.
We then set the objective, in this case tower mass, and scale it so it is of order 1 for better convergence behavior.
The tower diameters and thicknesses are added as design variables.
Finally, constraints are added.
Some constraints are based on the tower geometry and others are based on the stress and buckling loads experienced in the loading cases.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # If performing optimization
    :end-before: # ---

We then call ``setup()`` on the OpenMDAO problem, which finalizes the components and groups for the tower analysis or optimization.
Once ``setup()`` has been called, we can access the problem values or modify them for a given analysis.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # Set up the OpenMDAO problem
    :end-before: # ---
    
Now that we've set up the tower problem, we set values for tower, soil, and RNA assembly properties.
For the soil, shear and modulus properties for the soil can be defined, but in this example we assume that all directions are rigid (3 translation and 3 rotation).
The center of mass locations are defined relative to the tower top in the yaw-aligned coordinate system.
Blade and hub moments of inertia should be defined about the hub center, nacelle moments of inertia are defined about the center of mass of the nacelle.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # Set geometry and turbine values
    :end-before: # ---

For the power-law wind profile, the only parameter needed is the shear exponent.
In addition, some geometric parameters for the wind profile's extend must be defined, the base (or no-slip location) at `z0`, and the height at which a reference velocity will be defined.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # cost rates
    :end-before: # ---

As mentioned earlier, we are allowing for two separate loading cases.
The wind speed, and rotor force/moments for those two cases are now defined.
The wind speed location corresponds to the reference height defined previously as `wind_zref`.
In this simple case, we include only thrust and torque, but in general all 3 components of force and moments can be defined in the hub-aligned coordinate system.
The assembly automatically handles translating the forces and moments defined at the rotor to the tower top.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # --- loading case 1
    :end-before: # ---

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # --- loading case 2
    :end-before: # ---

Finally, some additional parameters used for constraints can be defined.
These include the minimum allowable taper ratio of the tower (from base to top), and the minimum diameter-to-thickness ratio allowed at any section.
These are only meaningful in an optimization context, not for analysis.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # --- constraints
    :end-before: # ---

We can now run the model and display some of the outputs.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # run the analysis or optimization
    :end-before: # ---

Lastly, we print the results from the tower analysis or optimization.

.. literalinclude:: ../../../examples/tower/example.py
    :language: python
    :start-after: # print results from
    :end-before: # ---
    
The stress, buckling, and damage loads are shown in :numref:`Figure %s <utilization-fig>`.
Each is a utilization and so should be <1 for a feasible result.
Because the result shown here was for an optimization case, we see some of the utilization values are right at the 1.0 upper limit.

.. _utilization-fig:

.. figure:: /images/towerse/utilization.*
    :width: 6in
    :align: center

    Utilization along tower for ultimate stress, shell buckling, global buckling, and fatigue damage.








