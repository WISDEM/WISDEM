
****************************
Structural Analysis
****************************

The structural analysis in DrivetrainSE is where the largest differences with the previous DriveSE code lies.  Instead of analytical derivations of the forces and moments on the various elements and closed form expressions for sizing the components, we instead rely on Frame3DD to conduct the analysis and enable the use of optimization with stress constraints to ensure a valid design.  Separate analyses are run for the rotating and non-rotating parts of the drivetrain, with some small and large differences depending on whether a direct-drive or geared configuration is employed.

Rotating Structural Analysis
===============================


Stationary Structural Analysis
===============================
