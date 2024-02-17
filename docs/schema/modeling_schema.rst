.. _modeling-options:

******************************
Modeling Options Inputs
******************************

The following inputs describe the options available in the ``modeling_options`` file.  This example is from the :code:`02_reference_turbines` case in the :code:`examples` directory.


.. literalinclude:: /../examples/02_reference_turbines/modeling_options.yaml
    :language: yaml



General
****************************************


:code:`verbosity` : Boolean
    Prints additional outputs to screen (and to a file log in the
    future)

    *Default* = False



RotorSE
****************************************


:code:`flag` : Boolean
    Whether or not to run RotorSE and ServoSE

    *Default* = False

:code:`n_aoa` : Integer
    Number of angles of attack in a common grid to define polars

    *Default* = 200

:code:`n_xy` : Integer
    Number of coordinate point used to define airfoils

    *Default* = 200

:code:`n_span` : Integer
    Number of spanwise stations in a common grid used to define blade
    properties

    *Default* = 30

:code:`n_pc` : Integer
    Number of wind speeds to compute the power curve

    *Default* = 20

:code:`n_pc_spline` : Integer
    Number of wind speeds to spline the power curve

    *Default* = 200

:code:`n_pitch_perf_surfaces` : Integer
    Number of pitch angles to determine the Cp-Ct-Cq-surfaces

    *Default* = 20

:code:`min_pitch_perf_surfaces` : Float
    Min pitch angle of the Cp-Ct-Cq-surfaces

    *Default* = -5.0

:code:`max_pitch_perf_surfaces` : Float
    Max pitch angle of the Cp-Ct-Cq-surfaces

    *Default* = 30.0

:code:`n_tsr_perf_surfaces` : Integer
    Number of tsr values to determine the Cp-Ct-Cq-surfaces

    *Default* = 20

:code:`min_tsr_perf_surfaces` : Float
    Min TSR of the Cp-Ct-Cq-surfaces

    *Default* = 2.0

:code:`max_tsr_perf_surfaces` : Float
    Max TSR of the Cp-Ct-Cq-surfaces

    *Default* = 12.0

:code:`n_U_perf_surfaces` : Integer
    Number of wind speeds to determine the Cp-Ct-Cq-surfaces

    *Default* = 1

:code:`regulation_reg_III` : Boolean
    Flag to derive the regulation trajectory in region III in terms of
    pitch and TSR

    *Default* = False

:code:`spar_cap_ss` : String
    Composite layer modeling the spar cap on the suction side in the
    geometry yaml. This entry is used to compute ultimate strains and
    it is linked to the design variable spar_cap_ss.

    *Default* = none

:code:`spar_cap_ps` : String
    Composite layer modeling the spar cap on the pressure side in the
    geometry yaml. This entry is used to compute ultimate strains and
    it is linked to the design variable spar_cap_ps.

    *Default* = none

:code:`te_ss` : String
    Composite layer modeling the trailing edge reinforcement on the
    suction side in the geometry yaml. This entry is used to compute
    ultimate strains and it is linked to the design variable te_ss.

    *Default* = none

:code:`te_ps` : String
    Composite layer modeling the trailing edge reinforcement on the
    pressure side in the geometry yaml. This entry is used to compute
    ultimate strains and it is linked to the design variable te_ps.

    *Default* = none

:code:`gamma_freq` : Float
    Partial safety factor for modal frequencies

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0

:code:`gust_std` : Float
    Number of standard deviations for strength of gust

    *Default* = 3.0

    *Minimum* = 0.0    *Maximum* = 5.0

:code:`root_fastener_s_f` : Float
    Safety factor for the max stress of blade root fasteners

    *Default* = 2.5

    *Minimum* = 0.1    *Maximum* = 1.e+2


DriveSE
****************************************


:code:`flag` : Boolean
    Whether or not to run RotorSE and ServoSE

    *Default* = False

:code:`model_generator` : Boolean
    Whether or not to do detailed generator modeling using tools
    formerly in GeneratorSE

    *Default* = False

:code:`gamma_f` : Float
    Partial safety factor on loads

    *Default* = 1.35

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_m` : Float
    Partial safety factor for materials

    *Default* = 1.3

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_n` : Float
    Partial safety factor for consequence of failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0




hub
########################################


:code:`hub_gamma` : Float
    Partial safety factor for hub sizing

    *Default* = 2.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`spinner_gamma` : Float
    Partial safety factor for spinner sizing

    *Default* = 1.5

    *Minimum* = 1.0    *Maximum* = 5.0




TowerSE
****************************************


:code:`flag` : Boolean
    Whether or not to run RotorSE and ServoSE

    *Default* = False

:code:`nLC` : Integer
    Number of load cases

    *Default* = 1

:code:`wind` : String from, ['PowerWind', 'LogisticWind']
    Wind scaling relationship with height

    *Default* = PowerWind

:code:`gamma_f` : Float
    Partial safety factor on loads

    *Default* = 1.35

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_m` : Float
    Partial safety factor for materials

    *Default* = 1.3

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_n` : Float
    Partial safety factor for consequence of failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_b` : Float
    Partial safety factor for buckling

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`gamma_freq` : Float
    Partial safety factor for modal frequencies

    *Default* = 1.1

    *Minimum* = 1.0    *Maximum* = 5.0

:code:`gamma_fatigue` : Float
    Partial safety factor for fatigue failure

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 5.0


:code:`buckling_length` : Float, m
    Buckling length factor in Eurocode safety check

    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 100.0




frame3dd
########################################


Set of Frame3DD options used for tower analysis

:code:`shear` : Boolean
    Inclusion of shear area for symmetric sections

    *Default* = True

:code:`geom` : Boolean
    Inclusion of shear stiffening through axial loading

    *Default* = True

:code:`nM` : Integer
    Number of tower eigenvalue modes to calculate

    *Default* = 6

    *Minimum* = 0    *Maximum* = 20


:code:`tol` : Float
    Convergence tolerance for modal eigenvalue solution

    *Default* = 1e-09

    *Minimum* = 1e-12    *Maximum* = 0.1




BOS
****************************************


:code:`flag` : Boolean
    Whether or not to run balance of station cost models (LandBOSSE or
    ORBIT)

    *Default* = False



FloatingSE
****************************************


:code:`flag` : Boolean
    Whether or not to run the floating design modules (FloatingSE)

    *Default* = False



Loading
****************************************


This is only used if not running the full WISDEM turbine Group and you need to input the mass properties, forces, and moments for a tower-only or nacelle-only analysis

:code:`mass` : Float, kilogram
    Mass at external boundary of the system.  For the tower, this
    would be the RNA mass.

    *Default* = 0.0

:code:`center_of_mass` : Array of Floats, meter
    Distance from system boundary to center of mass of the applied
    load.  For the tower, this would be the RNA center of mass in
    tower-top coordinates.

    *Default* = [0.0, 0.0, 0.0]

:code:`moment_of_inertia` : Array of Floats, kg*m^2
    Moment of inertia of external mass in coordinate system at the
    system boundary.  For the tower, this would be the RNA MoI in
    tower-top coordinates.

    *Default* = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



loads
########################################


:code:`force` : Array of Floats, Newton
    Force vector applied at system boundary

    *Default* = [0.0, 0.0, 0.0]

:code:`moment` : Array of Floats, N*m
    Force vector applied at system boundary

    *Default* = [0.0, 0.0, 0.0]

:code:`velocity` : Float, meter
    Applied wind reference velocity, if necessary

    *Default* = 0.0
