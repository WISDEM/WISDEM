.. _drivetrain_tutorial-label:

6. Drivetrain Model Example
---------------------------

This example optimizes the sizing of key drivetrain components, assuming their overall length and height have already been determined by a target blade tip deflection and hub height, respectively.  Only Python-scripts are supported for now, as a yaml-based input using only hub and nacelle parameters is not yet available.

Direct-Drive Design
===================

This example is for design sizing (diameter and thickness) of the shaft, nose/turret, and bedplate wall thickness of the direct-drive layout shown in :numref:`fig_layout_diagram_ex` and in :numref:`fig_layout_detail_ex`.

.. _fig_layout_diagram_ex:
.. figure::  /images/drivetrainse/layout_diagram.*
    :width: 50%
    :align: center

    Direct-drive configuration layout diagram

.. _fig_layout_detail_ex:
.. figure::  /images/drivetrainse/layout_detail.*
    :width: 100%
    :align: center

    Detailed direct-drive configuration with key user inputs and derived values.

Specifically, the design variables are,

- :math:`L_{h1}`
- :math:`L_{12}`
- :math:`D_{hub}`
- :math:`D_{lss}`
- :math:`D_{nose}`
- :math:`t_{lss}`
- :math:`t_{nose}`
- :math:`t_{bed}`

The design script starts with importing the required libraries,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Import
    :end-before: # ---

The modeling options dictionary sets the partial safety factors, the vector sizes, and specifies that a detailed generator design will *not* be run.

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Set in
    :end-before: # ---

Next, we instantiate the OpenMDAO problem and assign the drivetrain as the *model* of the problem,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Initialize
    :end-before: # ---

Next, the script proceeds to set-up the design optimization problem if :code:`opt_flag` is set to True.
In this case, the optimization driver is first be selected and configured.
We then set the objective, in this case nacelle mass, and scale it so it is of order 1 for better convergence behavior.
The drivetrain diameters and thicknesses are added as design variables, as well as the lengths that determine the shaft, nose, and bedplate lengths to reach the intended overhang distance.
Finally, a number of constraints are added, which fall into the categories of,

- **von Mises stress utilizations**: These constraints must be less than 1 and capture that the von Mises stress in the load-bearing components, multiplied by a safety factor must be less than the shear stress of the material.
- **Main bearing deflection utilizations**: Each bearing type has an associated maximum deflection, measured as an angle.  These constraints ensure that the shaft and nose/turret deflections at the bearing attachment points do not exceed those limits.
- **Minimum allowable hub diameter**: For a given blade root radius and number of blades, this is the minimum hub radius that can safely accommodate those dimensions.
- **Satisfy target overhang and hub height**: Ensure that the shaft, turret, and bedplate lengths are sufficient to meet the desired overhang and hub height
- **Allow maintenance access**: Specify the minimum height required to allow human access into the nose/turret for maintenance activities.

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # If perf
    :end-before: # ---

With the optimization problem defined, the OpenMDAO problem is *activated*

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Set up
    :end-before: # ---

Now we can specify the high level inputs that describe the turbine

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Set input
    :end-before: # ---

A number of blade properties and other parameters are needed for the hub and spinner designs,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Blade
    :end-before: # ---

Next is the layout of the drivetrain and the initial conditions of some of the design variables,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Drivetrain
    :end-before: # ---

Finally, the material properties that are used in the design are set.  Here we assume that the shaft and nose/turret are made of a steel with a slightly higher carbon content than the bedplate.  The hub is cast iron and the spinner is made of glass fiber.

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Material
    :end-before: # ---

The simulation can now be run.  If doing an optimization, we select finite differencing around the total derivatives instead of the partial derivatives.

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Run
    :end-before: # ---

All of the inputs and outputs are displayed on the screen, followed by a curated list of values relating to the optimization problem.

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_direct.py
    :language: python
    :start-after: # Display
    :end-before: # ---

The screen output should look something like the following,

.. code-block:: console

    Optimization terminated successfully    (Exit mode 0)
            Current function value: [0.22768958]
            Iterations: 31
            Function evaluations: 47
            Gradient evaluations: 31
    Optimization Complete
    -----------------------------------

    ...

    nacelle_mass: [227689.58069311]

    L_h1: [0.1]
    L_12: [1.04818685]
    L_lss: [1.14818685]
    L_nose: [3.15181315]
    L_generator: [2.15]
    L_bedplate: [4.92501211]
    H_bedplate: [4.86709905]
    hub_diameter: [7.78723633]
    lss_diameter: [2.67134366 2.67189989]
    lss_wall_thickness: [0.08664636 0.08664614]
    nose_diameter: [2.08469731 2.08525375]
    nose_wall_thickness: [0.08469731 0.08525375]
    bedplate_wall_thickness: [0.004      0.004      0.004      0.03417892]

    constr_lss_vonmises: [0.07861671 0.0784489  0.09307185 0.19116645]
    constr_bedplate_vonmises: [0.93281517 0.85272928 0.77435574 0.69658358 0.62163064 0.56099155
    0.55110369 0.41195844 0.31955169 0.3206749  0.34412109 0.17707953
    0.20135497 0.07195264 0.06577744]
    constr_mb1_defl: [0.18394075]
    constr_mb2_defl: [0.02072981]
    constr_hub_diameter: [0.58190497]
    constr_length: [1.67501211]
    constr_height: [4.86709905]
    constr_access: [[0.00000000e+00 1.35447209e-13]
    [1.56319402e-13 4.44089210e-16]]
    constr_ecc: [0.05791306]


Geared Design
====================

This example is for design sizing (diameter and thickness) of the shaft, nose/turret, and bedplate wall thickness of the direct-drive layout shown in :numref:`fig_geared_diagram_ex` and in :numref:`fig_geared_detail_ex`.


.. _fig_geared_diagram_ex:
.. figure::  /images/drivetrainse/geared_diagram.*
    :width: 75%
    :align: center

    Geared configuration layout diagram

.. _fig_geared_detail_ex:
.. figure::  /images/drivetrainse/geared_detail.*
    :width: 75%
    :align: center

    Geared configuration layout diagram

Specifically, the design variables are,

- :math:`L_{h1}`
- :math:`L_{12}`
- :math:`L_{hss}`
- :math:`D_{hub}`
- :math:`D_{lss}`
- :math:`D_{hss}`
- :math:`t_{lss}`
- :math:`t_{hss}`
- :math:`t_{web}` (bedplate I-beam dimension)
- :math:`t_{flange}` (bedplate I-beam dimension)
- :math:`w_{flange}` (bedplate I-beam dimension)

The design script is quite similar to the direct-drive version, so we will walk through it more succinctly, noting distinctions with the example above.
The design script starts with importing the required libraries,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Import
    :end-before: # ---

The only difference in the modeling options is to set the direct drive flag to :code:`False`,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Set in
    :end-before: # ---

Next, we instantiate the OpenMDAO problem and assign the drivetrain as the *model* of the problem,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Initialize
    :end-before: # ---

For the optimization problem setup, the only differences are swapping in the high speed shaft parameters instead of the nose/turret design variables and constraints.  The bedplate I-beam geometry parameters have also been added as design variables.  There also are not any constraints for maintenance accessibility because of the easier access already afforded in this layout.

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # If perf
    :end-before: # ---

With the optimization problem defined, the OpenMDAO problem is *activated*

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Set up
    :end-before: # ---

Now we can specify the high level inputs that describe the turbine.  Whereas the direct-drive example was for a 15-MW machine, this one is for a 5-MW machine with much smaller components.

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Set input
    :end-before: # ---

The blade properties and other parameters needed for the hub and spinner designs reflect the smaller machine size,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Blade
    :end-before: # ---

Next is the layout of the drivetrain and the initial conditions of some of the design variables, with additional inputs needed for the gearbox design,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Drivetrain
    :end-before: # ---

Finally, the material properties and assumptions are the same as above,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Material
    :end-before: # ---

The simulation can now be run.  If doing an optimization, the geared simulation takes longer than the direct-drive version due to the internal design iterations for the gearbox,

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Run
    :end-before: # ---

All of the inputs and outputs are displayed on the screen, followed by a curated list of values relating to the optimization problem.

.. literalinclude:: ../../../examples/06_drivetrain/drivetrain_geared.py
    :language: python
    :start-after: # Display
    :end-before: # ---

The screen output should look something like the following,

.. code-block:: console

    Optimization terminated successfully    (Exit mode 0)
                Current function value: [0.14948563]
                Iterations: 12
                Function evaluations: 12
                Gradient evaluations: 12
    Optimization Complete
    
    ...
    
    nacelle_mass: [149485.62543167]
    
    L_h1: [0.22298585]
    L_12: [0.1]
    L_lss: [0.42298585]
    L_hss: [0.52650472]
    L_generator: [2.]
    L_gearbox: [1.512]
    L_bedplate: [4.44451325]
    H_bedplate: [1.69326612]
    hub_diameter: [5.]
    lss_diameter: [0.69921744 0.70132537]
    lss_wall_thickness: [0.2878711  0.28786974]
    hss_diameter: [0.5 0.5]
    hss_wall_thickness: [0.0998473 0.0998473]
    bedplate_web_thickness: [0.09595255]
    bedplate_flange_thickness: [0.09892329]
    bedplate_flange_width: [0.1]
    
    constr_lss_vonmises: [0.98187486 0.9993906  0.99831682 0.99952247]
    constr_hss_vonmises: [0.03251358 0.03215481]
    constr_bedplate_vonmises: [1.51989918e-03 9.77245633e-03 6.01676078e-01 5.99836120e-01
     6.33223628e-01 8.63086490e-02 8.72758878e-02 4.39265181e-02
     4.30670107e-02 1.50407068e-04 2.08972837e-07 1.51990174e-03
     2.02074935e-02 6.91033399e-01 6.92826935e-01 7.50983681e-01
     1.07687995e-01 1.12023851e-01 4.39357428e-02 4.30706497e-02
     1.51237490e-04 1.59816568e-07]
    constr_mb1_defl: [0.07958005]
    constr_mb2_defl: [0.00890625]
    constr_hub_diameter: [0.09206083]
    constr_length: [-2.18136176e-10]
    constr_height: [1.69326612]
    
