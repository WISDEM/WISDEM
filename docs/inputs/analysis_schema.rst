.. _analysis-options:

******************************
Analysis Options Inputs
******************************
The following inputs describe the options available in the ``analysis_options`` file. This example is from the :code:`03_blade` case in the :code:`examples` directory.


.. literalinclude:: /../examples/03_blade/analysis_options_aerostruct.yaml
    :language: yaml

general
****************************************


:code:`folder_output` : String
    Name of folder to dump output files

    *Default* = output

:code:`fname_output` : String
    File prefix for output files

    *Default* = output



design_variables
****************************************


Sets the design variables in a design optimization and analysis



rotor_diameter
########################################


Adjust the rotor diameter by changing the blade length (all blade properties constant with respect to non-dimensional span coordinates)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`minimum` : Float, m


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0


:code:`maximum` : Float, m


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 1000.0




blade
########################################


Design variables associated with the wind turbine blades



aero_shape
========================================


Design variables associated with the blade aerodynamic shape



twist
----------------------------------------


Blade twist as a design variable by adding or subtracting radians from the initial value at spline control points along the span.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`inverse` : Boolean
    When set to True, the twist is defined inverting the 
    blade-element momentum equations to achieve a desired margin to stall, 
    which is defined among the constraints.
    :code:`flag` and :code:`inverse` cannot be simultaneously be set to True

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the twist distribution along blade span. 

    *Default* = 8

    *Minimum* = 4

:code:`lower_bound` : Array of Floats, rad
    Lowest number of radians that can be added (typically negative to
    explore smaller twist angles)

    *Default* = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]

:code:`upper_bound` : Array of Floats, rad
    Largest number of radians that can be added (typically postive to
    explore greater twist angles)

    *Default* = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

:code:`index_start` : Integer
    Integer setting the first DV of the :code:`n_opt` along span that is optimized.
    It is recommended to set :code:`index_start` to 1 
    to lock the first DV and prevent the optimizer to try to
    optimize the twist of the blade root cylinder.

    *Default* = 0

:code:`index_end` : Integer
    Integer setting the last DV of the :code:`n_opt` along span that is optimized.

    *Default* = 8


chord
----------------------------------------


Blade chord as a design variable by scaling (multiplying) the initial value at spline control points along the span.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the chord distribution along blade span.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease of the blade chord at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase of the blade chord at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    Integer setting the first DV of the :code:`n_opt` along span that is optimized.
    Setting :code:`index_start` to 1 or 2 locks the blade root diameter.

    *Default* = 0


:code:`index_end` : Integer
    Integer setting the last DV of the :code:`n_opt` along span that is optimized.
    It is recommended to lock the last point close to blade tip, setting :code:`index_end` to :code:`n_opt` minus 1. 
    The last point controls the chord length at blade tip and due to
    the imperfect tip loss models of CCBlade, it is usually a good
    idea to taper the chord manually and do not let a numerical
    optimizer control it.

    *Default* = 8


af_positions
----------------------------------------


Adjust airfoil positions along the blade span.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`af_start` : Integer
    Index of airfoil where the optimization can start shifting airfoil
    position. The airfoil at blade tip is always locked. It is advised 
    to keep the airfoils close to blade root locked.

    *Default* = 4

    *Minimum* = 1



structure
========================================


Design variables associated with the internal blade structure



spar_cap_ss
----------------------------------------


Blade suction-side spar cap thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the thickness of the spar cap on the suction side.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease of the spar cap thickness on the suction-side at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase of the spar cap thickness on the suction-side at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    Integer setting the first DV of the :code:`n_opt` along span that is optimized.
    It is recommended to set :code:`index_start` to 1 
    to lock the first DV and impose a pre-
    defined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 0

:code:`index_end` : Integer
    Integer setting the last DV of the :code:`n_opt` along span that is optimized.
    It is recommended to lock the last point close to blade tip, setting :code:`index_end` to :code:`n_opt` minus 1. 
    This imposes a predefined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 8


spar_cap_ps
----------------------------------------


Blade pressure-side spar cap thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the thickness of the spar cap on the pressure side.

    *Default* = 8

    *Minimum* = 4

:code:`max_decrease` : Float
    Maximum nondimensional decrease of the spar cap thickness on the pressure-side at each optimization location

    *Default* = 0.5

:code:`max_increase` : Float
    Maximum nondimensional increase of the spar cap thickness on the pressure-side at each optimization location

    *Default* = 1.5

:code:`index_start` : Integer
    Integer setting the first DV of the :code:`n_opt` along span that is optimized.
    It is recommended to set :code:`index_start` to 1 
    to lock the first DV and impose a pre-
    defined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 0

:code:`index_end` : Integer
    Integer setting the last DV of the :code:`n_opt` along span that is optimized.
    It is recommended to lock the last point close to blade tip, setting :code:`index_end` to :code:`n_opt` minus 1. 
    This imposes a predefined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 8


te_ss
----------------------------------------


Blade suction-side trailing edge reinforcement thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the thickness of the trailing edge reinforcement on
    the suction side. By default, the first point close to blade root
    and the last point close to blade tip are locked. This is done to
    impose a pre-defined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 8

    *Minimum* = 4

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


Blade pressure-side trailing edge reinforcement thickness as a design variable by scaling (multiplying) the initial value at spline control points along the span.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`n_opt` : Integer
    Number of equally-spaced control points of the spline
    parametrizing the thickness of the trailing edge reinforcement on
    the pressure side. By default, the first point close to blade root
    and the last point close to blade tip are locked. This is done to
    impose a pre-defined taper to small thicknesses and mimic a blade
    manufacturability constraint.

    *Default* = 8

    *Minimum* = 4

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


Design variables associated with the control of the wind turbine



tsr
========================================


Adjust the tip-speed ratio (ratio between blade tip velocity and steady hub-height wind speed)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`minimum` : Float
    Minimum allowable value

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 30.0


:code:`maximum` : Float
    Maximum allowable value

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 30.0




hub
########################################


Design variables associated with the hub



cone
========================================


Adjust the blade attachment coning angle (positive values are always away from the tower whether upwind or downwind)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 0.5235987756


:code:`upper_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 0.5235987756




hub_diameter
========================================


Adjust the rotor hub diameter

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for hub diameter

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for hub diameter

    *Default* = 30.0

    *Minimum* = 0.0    *Maximum* = 30.0




drivetrain
########################################


Design variables associated with the drivetrain



uptilt
========================================


Adjust the drive shaft tilt angle (positive values tilt away from the tower whether upwind or downwind)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 0.5235987756


:code:`upper_bound` : Float, rad
    Design variable bound

    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 0.5235987756




overhang
========================================


Adjust the x-distance, parallel to the ground or still water line, from the tower top center to the rotor apex.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




distance_tt_hub
========================================


Adjust the z-dimension height from the tower top to the rotor apex

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




distance_hub_mb
========================================


Adjust the distance along the drive staft from the hub flange to the first main bearing

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




distance_mb_mb
========================================


Adjust the distance along the drive staft from the first to the second main bearing

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




generator_length
========================================


Adjust the distance along the drive staft between the generator rotor drive shaft attachment to the stator bedplate attachment

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




gear_ratio
========================================


For geared configurations only, adjust the gear ratio of the gearbox that multiplies the shaft speed and divides the torque

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 1.0

    *Minimum* = 1.0    *Maximum* = 500.0


:code:`upper_bound` : Float


    *Default* = 150.0

    *Minimum* = 1.0    *Maximum* = 1000.0




lss_diameter
========================================


Adjust the diameter at the beginning and end of the low speed shaft (assumes a linear taper)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




hss_diameter
========================================


Adjust the diameter at the beginning and end of the high speed shaft (assumes a linear taper)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




nose_diameter
========================================


For direct-drive configurations only, adjust the diameter at the beginning and end of the nose/turret (assumes a linear taper)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Lowest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0


:code:`upper_bound` : Float, m
    Highest value allowable for design variable

    *Default* = 0.1

    *Minimum* = 0.1    *Maximum* = 30.0




lss_wall_thickness
========================================


Adjust the thickness at the beginning and end of the low speed shaft (assumes a linear taper)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




hss_wall_thickness
========================================


Adjust the thickness at the beginning and end of the high speed shaft (assumes a linear taper)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




nose_wall_thickness
========================================


For direct-drive configurations only, adjust the thickness at the beginning and end of the nose/turret (assumes a linear taper)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




bedplate_wall_thickness
========================================


For direct-drive configurations only, adjust the wall thickness along the elliptical bedplate

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




bedplate_web_thickness
========================================


For geared configurations only, adjust the I-beam web thickness of the bedplate

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




bedplate_flange_thickness
========================================


For geared configurations only, adjust the I-beam flange thickness of the bedplate

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




bedplate_flange_width
========================================


For geared configurations only, adjust the I-beam flange width of the bedplate

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.001

    *Minimum* = 0.001    *Maximum* = 3.0


:code:`upper_bound` : Float, m


    *Default* = 1.0

    *Minimum* = 0.01    *Maximum* = 5.0




tower
########################################


Design variables associated with the tower or monopile



outer_diameter
========================================


Adjust the outer diamter of the cylindrical column at nodes along the height.  Linear tapering is assumed between the nodes, creating conical frustums in each section

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




layer_thickness
========================================


Adjust the layer thickness of each section in the column

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0




section_height
========================================


Adjust the height of each conical section

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




monopile
########################################


Design variables associated with the tower or monopile



outer_diameter
========================================


Adjust the outer diamter of the cylindrical column at nodes along the height.  Linear tapering is assumed between the nodes, creating conical frustums in each section

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




layer_thickness
========================================


Adjust the layer thickness of each section in the column

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 0.01

    *Minimum* = 1e-05    *Maximum* = 1.0




section_height
========================================


Adjust the height of each conical section

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0


:code:`upper_bound` : Float, m
    Design variable bound

    *Default* = 5.0

    *Minimum* = 0.1    *Maximum* = 100.0




constraints
****************************************


Activate the constraints that are applied to a design optimization



blade
########################################


Constraints associated with the blade design



strains_spar_cap_ss
========================================


Enforce a maximum allowable strain in the suction-side spar caps

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float
    Maximum allowable strain value

    *Default* = 0.004

    *Minimum* = 1e-08    *Maximum* = 0.1




strains_spar_cap_ps
========================================


Enforce a maximum allowable strain in the pressure-side spar caps

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float
    Maximum allowable strain value

    *Default* = 0.004

    *Minimum* = 1e-08    *Maximum* = 0.1




tip_deflection
========================================


Enforce a maximum allowable blade tip deflection towards the tower expressed as a safety factor on the parked margin.  Meaning a parked distance to the tower of 30m and a constraint value here of 1.5 would mean that 30/1.5=20m of deflection is the maximum allowable

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`margin` : Float


    *Default* = 1.4175

    *Minimum* = 1.0    *Maximum* = 10.0




rail_transport
========================================


Enforce sufficient blade flexibility such that they can be transported on rail cars without exceeding maximum blade strains or derailment.  User can activate either 8-axle flatcars or 4-axle

:code:`8_axle` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`4_axle` : Boolean
    Activates as a design variable or constraint

    *Default* = False



stall
========================================


Ensuring blade angles of attacks do not approach the stall point. Margin is expressed in radians from stall.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`margin` : Float, radians


    *Default* = 0.05233

    *Minimum* = 0.0    *Maximum* = 0.5




chord
========================================


Enforcing max chord length limit at all points along blade span.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`max` : Float, meter


    *Default* = 4.3

    *Minimum* = 0.1    *Maximum* = 20.0


root_circle_diameter
========================================


Enforcing the minimum blade root circle diameter.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

frequency
========================================


Frequency separation constraint between blade fundamental frequency and blade passing (3P) frequency at rated conditions using gamma_freq margin. Can be activated for blade flap and/or edge modes.

:code:`flap_3P` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`edge_3P` : Boolean
    Activates as a design variable or constraint

    *Default* = False



moment_coefficient
========================================


(EXPERIMENTAL) Targeted blade moment coefficient (useful for managing root flap loads or inverse design approaches that is not recommendend for general use)

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


(EXPERIMENTAL) Targeted airfoil cl/cd ratio (useful for inverse design approaches that is not recommendend for general use)

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


(EXPERIMENTAL) Targeted blade moment coefficient (useful for managing root flap loads or inverse design approaches that is not recommendend for general use)

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


Constraints associated with the tower design



height_constraint
========================================


Double-sided constraint to ensure total tower height meets target hub height when adjusting section heights

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


Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



global_buckling
========================================


Enforce a global buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



shell_buckling
========================================


Enforce a shell buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



slope
========================================


Ensure that the diameter moving up the tower at any node is always equal or less than the diameter of the node preceding it

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



d_to_t
========================================


Double-sided constraint to ensure target diameter to thickness ratio for manufacturing and structural objectives

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0


:code:`upper_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0




taper
========================================


Enforcing a max allowable conical frustum taper ratio per section

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 0.5

    *Minimum* = 0.001    *Maximum* = 1.0




frequency
========================================


Frequency separation constraint between all tower modal frequencies and blade period (1P) and passing (3P) frequencies at rated conditions using gamma_freq margin.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



frequency_1
========================================


Targeted range for tower first frequency constraint.  Since first and second frequencies are generally the same for the tower, this usually governs the second frequency as well (both fore-aft and side-side first frequency)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 1.0


:code:`upper_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 1.0




monopile
########################################


Constraints associated with the monopile design



pile_depth
========================================


Ensures that the submerged suction pile depth meets a minimum value

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, m


    *Default* = 0.0

    *Minimum* = 0.0    *Maximum* = 200.0




stress
========================================


Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



global_buckling
========================================


Enforce a global buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



shell_buckling
========================================


Enforce a shell buckling limit using Eurocode checks with safety factor of gamma_f * gamma_b

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



slope
========================================


Ensure that the diameter moving up the tower at any node is always equal or less than the diameter of the node preceding it

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



d_to_t
========================================


Double-sided constraint to ensure target diameter to thickness ratio for manufacturing and structural objectives

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0


:code:`upper_bound` : Float


    *Default* = 50.0

    *Minimum* = 1.0    *Maximum* = 2000.0




taper
========================================


Enforcing a max allowable conical frustum taper ratio per section

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float


    *Default* = 0.5

    *Minimum* = 0.001    *Maximum* = 1.0




frequency
========================================


Frequency separation constraint between all tower modal frequencies and blade period (1P) and passing (3P) frequencies at rated conditions using gamma_freq margin.

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



frequency_1
========================================


Targeted range for tower first frequency constraint.  Since first and second frequencies are generally the same for the tower, this usually governs the second frequency as well (both fore-aft and side-side first frequency)

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 1.0


:code:`upper_bound` : Float, Hz


    *Default* = 0.1

    *Minimum* = 0.01    *Maximum* = 1.0




hub
########################################




hub_diameter
========================================


Ensure that the diameter of the hub is sufficient to accommodate the number of blades and blade root diameter

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



drivetrain
########################################




lss
========================================


Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



hss
========================================


Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



bedplate
========================================


Enforce a maximum allowable von Mises stress relative to the material yield stress with safety factor of gamma_f * gamma_m * gamma_n

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



mb1
========================================


Ensure that the angular deflection at this meain bearing does not exceed the maximum allowable deflection for the bearing type

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



mb2
========================================


Ensure that the angular deflection at this meain bearing does not exceed the maximum allowable deflection for the bearing type

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



length
========================================


Ensure that the bedplate length is sufficient to meet desired overhang value

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



height
========================================


Ensure that the bedplate height is sufficient to meet desired nacelle height value

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False



access
========================================


For direct-drive configurations only, ensure that the inner diameter of the nose/turret is big enough to allow human access

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`lower_bound` : Float, meter
    Minimum size to ensure human maintenance access

    *Default* = 2.0

    *Minimum* = 0.1    *Maximum* = 5.0




ecc
========================================


For direct-drive configurations only, ensure that the elliptical bedplate length is greater than its height

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`merit_figure` : String from, ['LCOE', 'AEP', 'Cp', 'blade_mass', 'tower_mass', 'tower_cost', 'monopile_mass', 'monopile_cost', 'structural_mass', 'structural_cost', 'blade_tip_deflection', 'My_std', 'flp1_std']
    Objective function / merit figure for optimization.  Choices are
    LCOE- levelized cost of energy, AEP- turbine annual energy
    production, Cp- rotor power coefficient, blade_mass, tower_mass,
    tower_cost, monopile_mass, monopile_cost, structural_mass-
    tower+monopile mass, structural_cost- tower+monopile cost,
    blade_tip_deflection- blade tip deflection distance towards tower,
    My_std- blade flap moment standard deviation, flp1_std- trailing
    flap standard deviation

    *Default* = LCOE



driver
****************************************


Specification of the optimization driver (optimization algorithm) parameters

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
    Optimization driver.  Can be one of [SLSQP, CONMIN, COBYLA, SNOPT]

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


Optimization iteration recording via OpenMDAO

:code:`flag` : Boolean
    Activates as a design variable or constraint

    *Default* = False

:code:`file_name` : String
    OpenMDAO recorder output SQL database file

    *Default* = log_opt.sql
