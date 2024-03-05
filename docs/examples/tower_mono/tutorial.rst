.. _towerse_tutorial-label:

.. currentmodule:: wisdem.towerse.tower

5. Tower and Monopile Example
------------------------------

In this example we show how to perform simulation and optimization of a land-based tower and an offshore tower-monopile combination.  Both examples can be executed either by limiting the input yaml to only the necessary components or by using a python script that calls WISDEM directly.

Land-based Tower Design
===========================

The following example, demonstrates how to set up and run analysis or optimization for a land-based tower.
Some of the geometric parameters are seen in :numref:`Figure %s <tower-fig>`.
The tower is not restricted to 3 sections, any number of sections can be defined.

.. _tower-fig:

.. figure:: /images/towerse/tower.*
    :height: 4in
    :align: center

    Example of tower geometric parameterization.

Invoking with YAML files
*************************

To run just the tower analysis from the YAML input files, we just need to include the necessary elements.  First dealing with the ``geometry_option.yaml`` file, this always includes the :code:`assembly` section. Of the :code:`components`, this means just the :code:`tower` section.  Also, the :code:`materials`, :code:`environment`, and :code:`costs` section,

.. literalinclude:: ../../../examples/05_tower_monopile/nrel5mw_tower.yaml
    :language: yaml

The ``modeling_options.yaml`` file is also limited to just the sections we need.  Note that even though the :code:`monopile` options are included here, since there was no specification of a monopile in the geometry inputs, this will be ignored.  One new section is added here, a :code:`loading` section that specifies the load scenarios that are applied to the tower.  Since there is no rotor simulation to generate the loads, they must be specified by the user directly.  Note that two load cases are specified.  This creates a set of constraints for both of them.

.. literalinclude:: ../../../examples/05_tower_monopile/modeling_options.yaml
    :language: yaml

The ``analysis_options.yaml`` poses the optimization problem,

.. literalinclude:: ../../../examples/05_tower_monopile/analysis_options.yaml
    :language: yaml

The yaml files can be called directly from the command line, or via a python script that passes them to the top-level WISDEM function,

.. code-block:: bash

    $ wisdem nrel5mw_tower.yaml modeling_options.yaml analysis_options.yaml

or

.. code-block:: bash

    $ python tower_driver.py

Where the contents of ``tower_driver.py`` are,

.. literalinclude:: ../../../examples/05_tower_monopile/tower_driver.py
    :language: python
    :start-after: #!
    :end-before: # print

Calling Python Directly
*************************

The tower optimization can also be written using direct calls to WISDEM's TowerSE module.  First, we import the modules we want to use and setup the tower configuration.
We set flags for if we want to perform analysis or optimization, as well as if we want plots to be shown at the end.
Next, we set the tower height, diameter, and wall thickness.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # Tower
    :end-before: # ---

We then set many analysis options for the tower, including materials, safety factors, and FEM settings.
The module uses the frame finite element code `Frame3DD <http://frame3dd.sourceforge.net/>`_ to perform the FEM analysis.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # Store analysis options in dictionary
    :end-before: # ---

Next, we instantiate the OpenMDAO problem and add a tower model to this problem.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # Instantiate OpenMDAO
    :end-before: # ---

Next, the script proceeds to set-up the design optimization problem if :code:`opt_flag` is set to True.
In this case, the optimization driver is first be selected and configured.
We then set the objective, in this case tower mass, and scale it so it is of order 1 for better convergence behavior.
The tower diameters and thicknesses are added as design variables.
Finally, constraints are added.
Some constraints are based on the tower geometry and others are based on the stress and buckling loads experienced in the loading cases.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # If performing optimization
    :end-before: # ---

We then call :code:`setup()` on the OpenMDAO problem, which finalizes the components and groups for the tower analysis or optimization.
Once :code:`setup()` has been called, we can access the problem values or modify them for a given analysis.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # Set up the OpenMDAO problem
    :end-before: # ---

Now that we've set up the tower problem, we set values for tower, soil, and RNA assembly properties.
For the soil, shear and modulus properties for the soil can be defined, but in this example we assume that all directions are rigid (3 translation and 3 rotation).
The center of mass locations are defined relative to the tower top in the yaw-aligned coordinate system.
Blade and hub moments of inertia should be defined about the hub center, nacelle moments of inertia are defined about the center of mass of the nacelle.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # Set geometry and turbine values
    :end-before: # ---

For the power-law wind profile, the only parameter needed is the shear exponent.
In addition, some geometric parameters for the wind profile's extend must be defined, the base (or no-slip location) at `z0`, and the height at which a reference velocity will be defined.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # cost rates
    :end-before: # ---

As mentioned earlier, we are allowing for two separate loading cases.
The wind speed, and rotor force/moments for those two cases are now defined.
The wind speed location corresponds to the reference height defined previously as `wind_zref`.
In this simple case, we include only thrust and torque, but in general all 3 components of force and moments can be defined in the hub-aligned coordinate system.
The assembly automatically handles translating the forces and moments defined at the rotor to the tower top.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # --- loading case 1
    :end-before: # ---

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # --- loading case 2
    :end-before: # ---

We can now run the model and display some of the outputs.

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # run the analysis or optimization
    :end-before: # ---

Results
*******

Whether invoking from the yaml files or running via python directly, the optimization result is the same.  It should look something like this:

.. code:: console

    Optimization terminated successfully    (Exit mode 0)
                Current function value: [0.24021234]
                Iterations: 11
                Function evaluations: 12
                Gradient evaluations: 11
    Optimization Complete
    -----------------------------------

The python scripts, whether passing the yaml files or calling TowerSE directly, also print information to the screen and make a quick plot of the constraints along the tower,

.. literalinclude:: ../../../examples/05_tower_monopile/tower_direct.py
    :language: python
    :start-after: # print results from
    :end-before: # ---

This generates the terminal screen output of,

.. code:: console

    zs = [ 0. 15. 30. 45. 60. 75. 90.]
    ds = [7.96388868 6.87580097 5.78771327 4.69962557 4.42308371 4.14654186
     3.87      ]
    ts = [0.02013107 0.02013107 0.02013107 0.01793568 0.01793568 0.01793568]
    mass (kg) = [240212.34456004]
    cg (m) = [37.93106541]
    d:t constraint = [314.5266543  238.89879914]
    taper ratio constraint = [0.59011693 0.82346986]

    wind:  [11.73732]
    freq (Hz) = [0.2853175  0.30495281 1.1010711  1.39479868 2.07478329 4.33892568]
    Fore-aft mode shapes = [[  1.1546986   -2.42796755   5.84302789  -4.96078724   1.39102831]
     [-22.13651124  -0.8234945  -26.50387043  97.18894013 -46.72506397]
     [  3.11541112 -39.78452985  93.18953008 -71.6279523   16.10754095]]
    Side-side mode shapes = [[  1.11000804  -2.37064728   5.66892857  -4.71973984   1.3114505 ]
     [  7.52307725  -8.97016119  27.37999066 -38.29492352  13.3620168 ]
     [  1.70121561 -35.45954107  80.57590888 -55.60717732   9.78959389]]
    top_deflection1 (m) = [0.97689316]
    Tower base forces1 (N) = [ 1.30095663e+06 -1.51339918e-09 -7.91629763e+06]
    Tower base moments1 (Nm) = [ 4.41475791e+06  1.17246595e+08 -3.46781682e+05]
    stress1 = [0.65753555 0.72045705 0.7845276  0.77742207 0.45864489 0.30017858]
    GL buckling = [0.66188317 0.73147334 0.82483738 0.8835392  0.64351218 0.54382796]
    Shell buckling = [0.99999989 0.9999999  0.99999997 0.99999998 0.41360128 0.21722867]

    wind:  [70.]
    freq (Hz) = [0.2853767  0.30501523 1.10109606 1.39479974 2.07484605 4.33899489]
    Fore-aft mode shapes = [[  1.1547354   -2.42783958   5.84278876  -4.96078471   1.39110013]
     [-22.13257499  -0.82895388 -26.48359914  97.15514766 -46.71001964]
     [  3.11513974 -39.78378163  93.18861708 -71.62733409  16.10735889]]
    Side-side mode shapes = [[  1.11003145  -2.37051178   5.66866315  -4.71969548   1.31151264]
     [  7.5229338   -8.96852486  27.37550871 -38.28969538  13.35977773]
     [  1.70100363 -35.45896917  80.57556268 -55.60741575   9.78981861]]
    top_deflection2 (m) = [0.88335072]
    Tower base forces2 (N) = [ 1.67397530e+06 -4.65661287e-10 -7.88527891e+06]
    Tower base moments2 (Nm) = [-1.87422266e+06  1.19912567e+08  1.47301970e+05]
    stress2 = [0.64216922 0.66921096 0.69062355 0.64457187 0.35261396 0.27325483]
    GL buckling = [0.6482481  0.68801235 0.74578503 0.77196602 0.55533402 0.52692342]
    Shell buckling = [0.98272504 0.90444378 0.81799653 0.74124285 0.2767209  0.17886837]

The stress, buckling, and damage loads are shown in :numref:`Figure %s <utilization-fig>`.
Each is a utilization and so should be <1 for a feasible result.
Because the result shown here was for an optimization case, we see some of the utilization values are right at the 1.0 upper limit.

.. _utilization-fig:
.. figure:: /images/towerse/utilization.*
    :width: 6in
    :align: center

    Utilization along tower for ultimate stress, shell buckling, global buckling, and fatigue damage.


Offshore Monopile Design
===========================

Monopile design in WISDEM is modeled as an extension of the tower.  In this example, the tower from above is applied offshore with the key differences being:

- Tower base is now at 10m above the water level at a transition piece coupling with the monopile
- Monopile extends from 10m above the water level through to the sea floor, at a water depth of 30m, with the pile extending an additional 25m into the surface.
- Monopile also has three sections, with one section for the submerged pile, the middle section for the water column, and the top section above the water up until the transition piece.
- Maximum allowable diameter for the monopile is 8m

Invoking with YAML files
*************************

To run just the monopile, there is an additional :code:`monopile` section that must be added to the :code:`components` section of the yaml,

.. literalinclude:: ../../../examples/05_tower_monopile/nrel5mw_monopile.yaml
    :language: yaml
    :start-after: values: [0.027, 0.0222
    :end-before: materials:

The :code:`environment` section must also be updated with the offshore properties,

.. literalinclude:: ../../../examples/05_tower_monopile/nrel5mw_monopile.yaml
    :language: yaml
    :start-after: air_speed_sound
    :end-before: costs:

The ``modeling_options_monopile.yaml`` file already contains a section for the monopile, with entries identical to the tower section.  The input :code:`loading` scenarios are also the same. The ``analysis_options.yaml`` file, however, is different and activates the design variables and constraints associated with the monopile.  Note also that the objective function now says :code:`structural_mass`, to capture the combined mass of both the tower and monopile,

.. literalinclude:: ../../../examples/05_tower_monopile/analysis_options_monopile.yaml
    :language: yaml

The yaml files can be called directly from the command line, or via a python script that passes them to the top-level WISDEM function,

.. code-block:: bash

    $ wisdem nrel5mw_monopile.yaml modeling_options_monopile.yaml analysis_options_monopile.yaml

or

.. code-block:: bash

    $ python monopile_driver.py

Where the contents of ``monopile_driver.py`` are,

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_driver.py
    :language: python
    :start-after: #!
    :end-before: # print

Calling Python Directly
*************************

The monopile optimization script using direct calls to WISDEM's TowerSE module also resembles the tower code, with some key additions to expand the design accordingly.  First, the script setup now includes the monopile initial condition and the transition piece location between the monopile and tower.  In this script, :math:`z=0` corresponds to the mean sea level (MSL).

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # Tower
    :end-before: # ---

The modeling options only needs to add in the number of sections and control points for the monopile,

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # Store analysis options in dictionary
    :end-before: # ---

Next, we instantiate the OpenMDAO problem as before,

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # Instantiate OpenMDAO
    :end-before: # ---

The optimization problem switches the objective function to the total structural mass of the tower plus the monopile, and the monopile diameter and thickness schedule are appended to the list of design variables.

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # If performing optimization
    :end-before: # ---

We then call :code:`setup()` as before,

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # Set up the OpenMDAO problem
    :end-before: # ---

Next, the additional inputs for the monopile include its discretization of the monopile and starting depth.

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # Set geometry and turbine values
    :end-before: # ---

The offshore environment parameters, such as significant wave heights and water density, much also bet set,

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # cost rates
    :end-before: # ---

The load cases are exactly the same as in the tower-only design case,

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # --- loading case 1
    :end-before: # ---

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # --- loading case 2
    :end-before: # ---

We can now run the model and display some of the outputs, such as in :numref:`Figure %s <utilization-mono-fig>`.

.. literalinclude:: ../../../examples/05_tower_monopile/monopile_direct.py
    :language: python
    :start-after: # run the analysis or optimization
    :end-before: # ---

.. _utilization-mono-fig:
.. figure:: /images/towerse/utilization_monopile.*
    :width: 6in
    :align: center

    Utilization along tower and monopile for ultimate stress, shell buckling, global buckling, and fatigue damage.
