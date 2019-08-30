Introduction
------------

Generator Systems Engineering (GeneratorSE) is a set of analytical frameworks for sizing variable speed wind turbine Generators. The tool can be used as an autonomous tool focused on generator design or integrated in the system using DriveSE, NRELs drivetrain sizing tool . 
Thus, the designer has the option to trade magnet, copper, or lamination properties and weights to achieve the optimal generator design that is also optimal for a given drivetrain system. 
Two types of generator systems : synchronous and induction machines are currently being handled by GeneratorSE. The tool includes optimisation modules for four sub-classes: 

1. Permanent-magnet synchronous generator (PMSG)
2. Electrically excited synchronous generator (EESG) 
3. Squirrel-cage induction generators (SCIG) and 
4. Doubly-fed induction generators (DFIG)

Each module is structured to perform electromagnetic, structural, and basic thermal design that are integrated to provide the optimal generator design dimensions. 
The analytical models of each generator module were created in Python within the OpenMDAO computing platform to facilitate systems design and multidisciplinary optimization. 
<<<<<<< HEAD
=======
All analytical methods were based on the magnetic circuit and equivalent circuit models as described in some previous work: citep{Upwind_study} and generator design handbooks: citep{Boldea},citep{Boldea_induction}.
Thesee methods used to evaluate the main design parameters, including the elctromagnetically active material. For the estimation of structural design, reference is made to : citep{McDonald}. 
>>>>>>> develop

GeneratorSE:
1. Provides basic design attributes in addition to key electrical performance parameters including, but not limited to, output voltage, current, resistances, inductances, and losses and also the weights and costs of materials involved in the basic design.
2. Allows for an integrated design with DriveSE and NRELs Cost and Scaling Model thereby enabling a complete drivetrain optimization of direct-drive, medium-speed, and high-speed geared systems considering the entire turbine system and balance of plant.
3. Enables drivetrain design coupled with the turbine rotor and tower for a full integrated wind turbine design or even wind plant cost of energy optimization as part of the Wind Plant Integrated Systems Design and Engineering Model (WISDEM).

As a first step, the user chooses the type of generator that needs to be optimized along with the optimization goal.
The optimization goal may be overall costs, efficiency, mass, aspect ratio or a weighted combination of these.
Each generator subset is identified by a design space that contains decision variables based on which the design is most sensitive 
and the optimal design is searched by mathematical methods. The key inputs for each design generation include the power rating, 
torque, rated speed, shear stress, specific costs (i.e., unit costs per kilogram of material), and properties of materials (e.g., material density) used in the basic design. 
The designs are generated in compliance with the user-specified constraints on generator terminal voltage and constraints imposed on the dimensions and electromagnetic performance. 
