******************************
Drivetrain Model Introduction
******************************

The Drivetrain Systems Engineering (DrivertainSE) module is a set of models for sizing wind turbine drivetrain components as part of the larger WISDEM design and analysis tool. Wind turbine drivetrains physically connect the rotor to the tower and serve as a load-path from one to the other.  The drivetarin is also responsible for converting the aerodynamic torque of the rotor into electrical power that can be fed to the grid. Therefore, the drivetrain model interacts with the rotor and tower designs and it is important in looking at the overall design of a wind turbine to consider the coupling that exists between these three primary subsystem.  DrivetrainSE provides the capability to take in the aerodynamic loads and rotor properties and to estimate the mass properties and dimensions for all major components; the overall nacelle properties can then be used in subsequent tower design and analysis or as part of a system-level optimization of the wind turbine.  In addition, the resulting mass and dimension estimates can then be used to feed into a turbine capital cost model as well as a balance of station cost model that considers cost of assembly and installation of a wind turbine so that a full wind plant system level cost analysis could be performed.

DrivetrainSE uses a slightly different approach to the prior instances of DriveSE and HubSE, although some of the component sizing remains the same. Instead of analytical derivations of the forces and moments on the various elements and closed form expressions for sizing the components, we instead rely on Frame3DD to conduct the analysis and enable the use of optimization with stress constraints to ensure a valid design.  This proves to be an easier long-term approach to maintain correct code.

DrivetrainSE features the following capabilities:

* Upwind or downwind rotor configuration
* Direct-drive and geared
* Electromagnetic design of multiple generator technologies, including synchronous and induction generators.
* Up-tower or down-tower electronics
* Sizing of drivetrain components via structural analysis

DrivetrainSE includes sizing for the following components:

* Hub
* Spinner
* Pitch system
* Low speed shaft
* Main bearing(s)
* Gearbox (geared systems only)
* High speed shaft (geared systems only)
* Brake
* Generator
* Generator cooling
* Power electronics
* Bedplate
* Nacelle platform
* Nacelle cover
* Yaw system

Some of these components are sized with very simple, empirical- or regression-based approximations.  Others involve more detailed structural analysis, cast as utilization constraints, that are meant to be included in an optimization.
