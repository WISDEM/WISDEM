.. _user_custom_mass-label:

11b. User Customized Mass, Cost, Stiffness Example
----------------------------------------------------------

WISDEM takes a first-principles approach to estimation of mass and stiffness.  Cross sections and material properties are used to estimate area, mass density, moments of inertia, stiffness, etc.  While this is a step above regression-based sizing methods, and suitable level of fidelity for conceptual design, WISDEM cannot offer the level of cross sectional detail and analysis that would be present in a computer-aided design (CAD) model.  When those details are required for design iteration, or when WISDEM is used as a pre-processor for higher-fidelity aero-hydro-elastic analysis such as in `OpenFAST <https://github.com/OpenFAST/openfast>`_ via `WEIS <https://github.com/WISDEM/WEIS>`_, the user may want to override the first-principles calculation in WISDEM.

This example shows how to set user-overrides for mass, stiffness, and cost of various components or structural members in a wind turbine.  This includes the approach of bypassing the WISDEM module altogether or nudging the WISDEM calculations to arrive at the desired output value.

Another approach that should be explored is WISDEM's inverse design capability, where users adjust design variables to reach target values for component masses or other properties.  This would still use WISDEM's first-principles to compute mass values, but can be another way to match desired targets.  For the composite blade in particular, this is in fact, the recommended approach. See :ref:`inverse_design_tutorial-label`.

Bypassing WISDEM Modules
*************************
Sometimes, geometric details or design parameters of turbine components are not known, but overall elastic properties are available.  This might be the case when dealing with proprietary data, or using an aeroelastic model as a starting point.  In these scenarios, the user may want to bypass WISDEM modules as the required inputs are not available.  To do this, the ``elastic_properties`` entries in the geometry yaml file that are ordinarily used to store outputs from WISDEM are instead used as inputs.  Additionally, the ``user_elastic`` flags must be activated in the modeling yaml files while the overall flags to execute the module is set to False.

.. code-block:: yaml

   name: 5MW with OC3 semisubmersible

   assembly:
       turbine_class: I
       ...
   components:
       blade:
           ...
           elastic_properties_mb:
              six_x_six:
                  inertia_matrix:
                      mass: []
                      i_plr: []
                      i_flap: []
                      i_edge: []
                      i_cp: []
                      cm_x: []
                      cm_y: []
                  stiff_maxtrix:
                      grid: []
                      K11: []
                      K12: []
                      K13: []
                      K14: []
                      K15: []
                      K16: []
                      K21: []
                      K22: []
                      K23: []
                      K24: []
                      K25: []
                      K26: []
                      K31: []
                      K32: []
                      K33: []
                      K34: []
                      K35: []
                      K36: []
       hub:
           diameter: 7.94
           cone_angle: 0.06981317007977318
           elastic_properties_mb:
               system_mass: 69360.0
               system_inertia: [973520.0, 619970.0, 619970.0]
       nacelle:
           drivetrain:
               uptilt: 0.10471975511966002
               distance_tt_hub: 5.614100000000008
               overhang: 12.032
               gear_ratio: 1
               gearbox_efficiency: 1.0
           elastic_properties_mb:
               system_mass: 675175.0
               yaw_mass: 10000.0
               system_inertia: [9912933.0, 10862815.0, 10360761.0]
               system_inertia_tt: [9912933.0, 10862815.0, 10360761.0]
               system_center_mass: [-4.528, -0.14, 4.098]
               spring_constant: 1000.0
               damping_coefficient: 10.0
           generator:
               elastic_properties_mb:
                   system_mass: 69360.0
                   rotor_inertia: [9912933.0, 10862815.0, 10360761.0]
       tower: # user_elastic not yet implemented
           ...
       monopile: # user_elastic not yet implemented
           ...
       jacket: # user_elastic not yet implemented
           ...
       floating_platform: # user_elastic not yet implemented
           ...
           members:
               - name: spar
                 ...

The modeling yaml file would be the following,

.. code-block:: yaml

    # Generic modeling options file to run standard WISDEM case
    General:
        verbosity: False  # When set to True, the code prints to screen many infos
    WISDEM:
        RotorSE:
            flag: False
            user_elastic: True
        DriveSE:
            flag: False
            user_elastic: True
            generator:
                user_elastic: True
        TowerSE: # user_elastic not yet implemented
            flag: True
        FixedBottomSE: # user_elastic not yet implemented
            flag: True
        FloatingSE: # user_elastic not yet implemented
            flag: True

The example, ``IEA-15-240-RWT_VolturnUS-S_user_elastic.yaml``, and the associated modeling yaml file in, ``modeling_options_user_elastic.yaml``, bypass the nacelle and drivetrain modeling using summary elastic properties.

Mass
*******************

There are a number of user options available in the yaml-input file that give the user control over component mass values.  Depending on the component, there may be three pathways that the user can choose from, where the recommended approach will depend on the specific modeling problem:

- User mass overrides
- Adjustment of material properties
- Use of "outfitting factors" that estimates otherwise unresolved mass

User mass overrides
++++++++++++++++++++

Internally, WISDEM computes its mass estimate using its standard approach, then adjusts both the mass and moment of inertia to match the specified user value.

The yaml code below shows the user mass variable names (acknowledging that no turbine would have a monopile, jacket, and floating support structure).


.. code-block:: yaml

   name: 5MW with OC3 semisubmersible

   assembly:
       turbine_class: I
       ...
   components:
       blade:
           ...
       hub:
           ...
           hub_shell_mass_user: 2700.0
           spinner_mass_user: 500.0
           pitch_system_mass_user: 8300.0
       nacelle:
           drivetrain:
               ...
               mb1_mass_user: 1500.0
               mb2_mass_user: 1350.0
               bedplate_mass_user: 20000.0
               brake_mass_user: 5500.0
               converter_mass_user: 4200.0
               transformer_mass_user: 11500.0
               gearbox_mass_user: 21500.0
       tower:
           ...
           tower_mass_user: 250000.0
       monopile:
           ...
           monopile_mass_user: 250000.0
       jacket:
           ...
           jacket_mass_user: 250000.0
       floating_platform:
           ...
           members:
               - name: spar
                 ...
                 member_mass_user: 200000.0


The example file, ``nrel5mw-spar_oc3_user_mass.yaml`` uses this approach.


Adjustment of material properties
++++++++++++++++++++++++++++++++++++++++

Another approach would be to edit the material properties to arrive at the overall desired mass value.  This is applicable to the blade, drivetrain, and support structure components that rely on the material properties defined in the ``materials`` section of the geometry yaml-file.  For instance, if a user wanted to adjust the mass properties for a monopile, a new material could be created, such as "monopile-steel", that has a different density than what might be found in a material property datasheet.  Adjustment of material density would also impact moment of inertia calculations in a similar proportion.


Outfitting factors
++++++++++++++++++++++++++++++++++++++++

Some structural support components offer an "outfitting factor" to capture mass elements that are not described in the geometry-yaml parameterization or estimated separately by WISDEM.  For a wind turbine tower, this includes lights, platforms, elevators or ladders, cabling, etc.  For a monopile or floating platform member, this might include internal scantling, bulkheads, water ballast management systems, etc.  The outfitting factors acts as a multiplier on the mass calculation along the length of the component.  The outfitting factor usage is:


.. code-block:: yaml

   name: 5MW with OC3 semisubmersible

   assembly:
       ...
   components:
       ...
       tower:
           ...
           internal_structure_2d_fem:
               outfitting_factor: 1.07
               ...
       monopile:
           ...
           internal_structure_2d_fem:
               outfitting_factor: 1.07
               ...
       floating_platform:
           ...
           members:
               - name: spar
                 internal_structure:
                     outfitting_factor: 1.07
                     ...

Stiffness
***************

The recommended approach for adjusting stiffness properties of a particular component is to adjust the ``E`` and ``G`` properties of the relevant material, similar to the approach of adjusting density in :ref:`Adjustment of material properties`.


Cost
**************

The most effective means of dialing in a specific cost for most components is to adjust the mass multipliers such as the examples below.  For the larger components, the cost calculation takes a bottom-up approach, so the recommended approach to adjust cost is to adjust the individual unit costs of the materials and/or labor rates.  This more detailed approach applies to blades, towers, and offshore foundations.

.. code-block:: yaml

   name: 5MW with OC3 semisubmersible

   assembly:
       ...
   components:
       ...
   costs:
       hub_mass_cost_coeff: 3.9
       pitch_system_mass_cost_coeff: 22.1
       spinner_mass_cost_coeff: 11.1
       lss_mass_cost_coeff: 11.9
       bearing_mass_cost_coeff: 4.5
       hss_mass_cost_coeff: 6.8
       generator_mass_cost_coeff: 12.4  # Only used if not doing detailed generator modeling
       bedplate_mass_cost_coeff: 2.9
       yaw_mass_cost_coeff: 8.3
       converter_mass_cost_coeff: 18.8
       transformer_mass_cost_coeff: 18.8
       hvac_mass_cost_coeff: 124.0
       cover_mass_cost_coeff: 5.7
       elec_connec_machine_rating_cost_coeff: 41.85
       platforms_mass_cost_coeff: 17.1
       controls_machine_rating_cost_coeff: 21.15
