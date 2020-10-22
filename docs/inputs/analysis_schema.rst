.. _analysis-options:

******************************
Analysis Options Inputs
******************************
The following inputs describe the options available in the ``analysis_options`` file.



general
****************************************

:code:`folder_output` : String
    Name of folder to dump output files

    *Default* = output

:code:`fname_output` : String
    File prefix for output files

    *Default* = output



optimization_variables
****************************************



blade
########################################



aero_shape
========================================



twist
----------------------------------------

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`inverse` : Boolean
    Words TODO?

    *Default* = False

:code:`n_opt` : Integer
    Number of control points to use

    *Default* = 8

:code:`lower_bound` : Array of Floats, rad


    *Default* = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]

:code:`upper_bound` : Array of Floats, rad


    *Default* = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]



chord
----------------------------------------

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of control points to use

    *Default* = 8

:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float
    Upper bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 1.5



af_positions
----------------------------------------

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`af_start` : Integer
    Index of airfoil where the optimization can start shifting airfoil
    position

    *Default* = 4



structure
========================================



spar_cap_ss
----------------------------------------

:code:`name` : String
    Layer name of this design variable in the geometry yaml

    *Default* = none

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of control points to use

    *Default* = 8

:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float
    Upper bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 1.5



spar_cap_ps
----------------------------------------

:code:`name` : String
    Layer name of this design variable in the geometry yaml

    *Default* = none

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of control points to use

    *Default* = 8

:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float
    Upper bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 1.5



te_ss
----------------------------------------

:code:`name` : String
    Layer name of this design variable in the geometry yaml

    *Default* = none

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of control points to use

    *Default* = 8

:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float
    Upper bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 1.5



te_ps
----------------------------------------

:code:`name` : String
    Layer name of this design variable in the geometry yaml

    *Default* = none

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of control points to use

    *Default* = 8

:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float
    Upper bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 1.5



control
########################################



tsr
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min_gain` : Float
    Lower bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 0.5

:code:`max_gain` : Float
    Upper bound on scalar multiplier that will be applied to value at
    control points

    *Default* = 1.5



servo
========================================



pitch_control
----------------------------------------

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`omega_min` : Float


    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`omega_max` : Float


    *Default* = 0.7

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`zeta_min` : Float


    *Default* = 0.4

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`zeta_max` : Float


    *Default* = 1.5

    *Minimum* = 0.0    *Maximum* = 10.0




torque_control
----------------------------------------

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`omega_min` : Float


    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`omega_max` : Float


    *Default* = 0.7

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`zeta_min` : Float


    *Default* = 0.4

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`zeta_max` : Float


    *Default* = 1.5

    *Minimum* = 0.0    *Maximum* = 10.0




flap_control
----------------------------------------

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`omega_min` : Float


    *Default* = 0.1

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`omega_max` : Float


    *Default* = 0.7

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`zeta_min` : Float


    *Default* = 0.4

    *Minimum* = 0.0    *Maximum* = 10.0


:code:`zeta_max` : Float


    *Default* = 1.5

    *Minimum* = 0.0    *Maximum* = 10.0




tower
########################################



outer_diameter
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m


    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




layer_thickness
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0


:code:`upper_bound` : Float, m


    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0




section_height
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m


    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




constraints
****************************************



blade
########################################



strains_spar_cap_ss
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float


    *Default* = 0.004

    *Minimum* = 1e-08    *Maximum* = 0.1




strains_spar_cap_ps
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float


    *Default* = 0.004

    *Minimum* = 1e-08    *Maximum* = 0.1




tip_deflection
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`ratio` : Float


    *Default* = 0.8

    *Minimum* = 0.0    *Maximum* = 2.0




rail_transport
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`8_axle` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`4_axle` : Boolean
    Activates as a design variable or constraint

    *Default* = False



stall
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`margin` : Float


    *Default* = 0.05233

    *Minimum* = 0.0    *Maximum* = 0.5




chord
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float


    *Default* = 4.3

    *Minimum* = 0.1    *Maximum* = 20.0




frequency
========================================

:code:`flap_above_3P` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`edge_above_3P` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`flap_below_3P` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`edge_below_3P` : Boolean
    Activates as a design variable or constraint

    *Default* = False



moment_coefficient
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.15

    *Minimum* = 0.01    *Maximum* = 5.0


:code:`max` : Float


    *Default* = 0.15

    *Minimum* = 0.01    *Maximum* = 5.0




match_cl_cd
========================================

:code:`flag_cl` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`flag_cd` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`filename` : String
    file path to constraint data

    *Default* =



match_L_D
========================================

:code:`flag_L` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`flag_D` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`filename` : String
    file path to constraint data

    *Default* =



tower
########################################



height_constraint
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.01

    *Minimum* = 1e-06    *Maximum* = 10.0


:code:`upper_bound` : Float, m


    *Default* = 0.01

    *Minimum* = 1e-06    *Maximum* = 10.0




stress
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`ratio` : Float


    *Default* = 0.8

    *Minimum* = 0.0    *Maximum* = 2.0




global_buckling
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`ratio` : Float


    *Default* = 0.8

    *Minimum* = 0.0    *Maximum* = 2.0




shell_buckling
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`ratio` : Float


    *Default* = 0.8

    *Minimum* = 0.0    *Maximum* = 2.0




weldability
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`ratio` : Float


    *Default* = 0.8

    *Minimum* = 0.0    *Maximum* = 2.0




manufacturability
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`ratio` : Float


    *Default* = 0.8

    *Minimum* = 0.0    *Maximum* = 2.0




slope
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`ratio` : Float


    *Default* = 0.8

    *Minimum* = 0.0    *Maximum* = 2.0




frequency_1
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 1.0


:code:`upper_bound` : Float


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 1.0




control
########################################



flap_control
========================================

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`min` : Float


    *Default* = 0.05

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`max` : Float


    *Default* = 0.05

    *Minimum* = 0.0    *Maximum* = 1.0


:code:`merit_figure` : String from, ['LCOE', 'AEP', 'Cp', 'blade_mass', 'tower_mass', 'tower_cost', 'blade_tip_deflection', 'My_std', 'flp1_std']
    Objective function / merit figure for optimization

    *Default* = LCOE



driver
****************************************

:code:`tol` : Float
    Convergence tolerance (relative)

    *Default* = 1e-06

    *Minimum* = 1e-12    *Maximum* = 1.0


:code:`max_iter` : Integer
    Max number of optimization iterations

    *Default* = 100

    *Minimum* = 0    *Maximum* = 100000


:code:`max_function_calls` : Integer
    Max number of calls to objective function evaluation

    *Default* = 100000

    *Minimum* = 0    *Maximum* = 100000000


:code:`solver` : String from, ['SLSQP', 'CONMIN', 'COBYLA', 'SNOPT']
    Optimization driver

    *Default* = SLSQP

:code:`step_size` : Float
    Maximum step size

    *Default* = 0.001

    *Minimum* = 1e-10    *Maximum* = 100.0


:code:`form` : String from, ['central', 'forward', 'complex']
    Finite difference calculation mode

    *Default* = central



recorder
****************************************

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`file_name` : String
    OpenMDAO recorder output SQL database file

    *Default* = log_opt.sql
