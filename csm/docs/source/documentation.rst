Module Documentation
--------------------

To be done...

Cost and Scaling Model Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cost escalators are used to adjust for input factor prices trends for materials and labor associated with turbine manufacturing, plant development and operation. 

Turbine Capital Costs (TCC) Module
^^^^^^^^^^^^^^^^^

.. module:: csm.src.csmTurbine

The NREL Cost and Scaling Model turbine module calculates turbine component costs and masses (either onshore or offshore).  This excludes the foundation which is considered part of the balance of station even for offshore plants.

csmTurbine
""""""""""

This class calculates the the component masses and costs for a wind turbine using the NREL Cost and Scaling Model.

.. currentmodule:: csm.src.csmTurbine
.. class:: csmTurbine 
.. currentmodule:: csm.src.csmTurbine.csmTurbine 

**Methods**

.. autosummary::
    __init__
    compute
    getMass
    getCost

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        getMass
        getCost

There are several sub-modules in the overall TCC module for each of the major sub-systems including the blades, hub, nacelle and tower.

csmBlades
"""""""""

This class calculates wind turbine blade mass and cost based on the NREL Cost and Scaling Model.

.. currentmodule:: csm.src.csmBlades
.. class:: csmBlades 
.. currentmodule:: csm.src.csmBlades.csmBlades 

**Methods**

.. autosummary::
    __init__
    compute
    computeMass
    computeCost
    getMass
    getCost

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        computeMass
        computeCost
        getMass
        getCost

**Example**

The following is an example use of the individual csmBlades module.

.. literalinclude:: ..\..\..\..\models\csm\csmBlades.py
   :pyobject: example

Output should be:

Conventional blade design:
Blades    276129.6 K$   25614.4 kg

Advanced blade design:
Blades    251227.0 K$   17650.7 kg

csmHub
""""""
This class calculates wind turbine hub system mass and cost based on the NREL Cost and Scaling Model.

.. currentmodule:: csm.src.csmHub
.. class:: csmHub 
.. currentmodule:: csm.src.csmHub.csmHub 

**Methods**

.. autosummary::
    __init__
    compute
    computeMass
    computeCost
    getMass
    getCost
    getHubComponentMasses
    getHubComponentCosts

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        computeMass
        computeCost
        getMass
        getCost
        getHubComponentMasses
        getHubComponentCosts

**Example**

The following is an example use of the individual csmHub module.

.. literalinclude:: ..\..\..\..\models\csm\csmHub.py
   :pyobject: example

The output should be:

Hub System Components:
	Hub	124903.9 K$  22519.7 kg
	Pitch Mech   24379.7 K$  10313.9 kg
	Nose Cone   10509.0 K$  1810.5 kg

Hub total   379152.6 K$  34644.2 kg

csmNacelle
""""""""""
This class calculates wind turbine hub system mass and cost based on the NREL Cost and Scaling Model.

.. currentmodule:: csm.src.csmNacelle
.. class:: csmNacelle 
.. currentmodule:: csm.src.csmNacelle.csmNacelle 

**Methods**

.. autosummary::
    __init__
    compute
    computeMass
    computeCost
    getMass
    getCost
    getNacelleComponentMasses
    getNacelleComponentCosts

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        computeMass
        computeCost
        getMass
        getCost
        getNacelleComponentMasses
        getNacelleComponentCosts

**Example**

The following is an example use of the individual csmNacelle module.

.. literalinclude:: ..\..\..\..\models\csm\csmNacelle.py
   :pyobject: example

Output should be:

Nacelle Total  3249404.5229 K$  223315.6984 kg

Within the nacelle, there are additional sub-modules for the low-speed shaft, the gearbox and the generator.

LowSpdShaft
"""""""""""

This class calculates wind turbine low speed shaft mass and cost based on the NREL Cost and Scaling Model.

.. currentmodule:: csm.src.csmNacelle
.. class:: LowSpdShaft 
.. currentmodule:: csm.src.csmNacelle.LowSpdShaft 

**Methods**

.. autosummary::
    __init__
    compute
    computeMass
    computeCost
    getMass
    getCost

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        computeMass
        computeCost
        getMass
        getCost

GearBox
"""""""

This class calculates wind turbine gearbox mass and cost based on the NREL Cost and Scaling Model.

.. currentmodule:: csm.src.csmNacelle
.. class:: GearBox 
.. currentmodule:: csm.src.csmNacelle.GearBox 

**Methods**

.. autosummary::
    __init__
    compute
    computeMass
    computeCost
    getMass
    getCost

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        computeMass
        computeCost
        getMass
        getCost

Generator
"""""""""

This class calculates wind turbine generator mass and cost based on the NREL Cost and Scaling Model.

.. currentmodule:: csm.src.csmNacelle
.. class:: Generator 
.. currentmodule:: csm.src.csmNacelle.Generator 

**Methods**

.. autosummary::
    __init__
    compute
    computeMass
    computeCost
    getMass
    getCost

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        computeMass
        computeCost
        getMass
        getCost


csmTower
""""""""

This class calculates wind turbine tower mass and cost based on the NREL Cost and Scaling Model.

.. currentmodule:: csm.src.csmTower
.. class:: csmTower 
.. currentmodule:: csm.src.csmTower.csmTower 

**Methods**

.. autosummary::
    __init__
    compute
    computeMass
    computeCost
    getMass
    getCost

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        computeMass
        computeCost
        getMass
        getCost

**Example**

The following is an example use of the individual csmTower module.

.. literalinclude:: ..\..\..\..\models\csm\csmTower.py
   :pyobject: example

Output should be:

Tower Mass: 444384.157764
Tower Cost: 1009500.23594

**Examples**
The following is an example use of the overall csmTurbine module for calculating the onshore and offshore wind turbine costs using the NREL 5 MW reference turbine [1]_:

.. literalinclude:: ..\..\..\..\models\csm\csmTurbine.py
   :pyobject: example

Output should be:

Onshore configuration 5 MW turbine:
Turbine total: 5487334.85 $K  774049.19 kg

Offshore configuration 5 MW turbine:
Turbine total: 6064385.11 $K  774049.19 kg


**References**

.. [1] Jonkman, J.; Butterfield, S.; Musial, W.; Scott, G. (2009). "Definition of a 5-MW Reference Wind Turbine for Offshore System Development." NREL/TP-500-38060. Golden, CO: National Renewable Energy Laboratory, 75 pp.

Balance of Station Costs (BOS) Module
^^^^^^^^^^^^^^^^^^^

.. module:: csm.src.csmBOS

The NREL Cost and Scaling Model balance of station module calculates the installation, assembly and other non-turbine capital costs associated with the development of a wind plant (either onshore or offshore).  The foundation is considered part of the balance of station even for offshore plants.

csmBOS
""""""
.. currentmodule:: csm.src.csmBOS
.. class:: csmBOS 
.. currentmodule:: csm.src.csmBOS.csmBOS

**Methods**

.. autosummary::
    __init__
    compute
    getCost
    getDetailedCosts

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        getCost
        getDetailedCosts


csmFoundation
"""""""""""""

.. module:: csm.src.csmFoundation

.. currentmodule:: csm.src.csmFoundation
.. class:: csmFoundation 
.. currentmodule:: csm.src.csmFoundation.csmFoundation

**Methods**

.. autosummary::
    __init__
    compute
    getCost

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        getCost

**Examples**
The following is an example use of csmBOS for calculating the onshore and offshore wind plant Balance of Station Costs using the NREL 5 MW reference turbine [1]_:

.. literalinclude:: ..\..\..\..\models\csm\csmBOS.py
   :pyobject: example

The output should appear as:
BOS cost onshore: 3084560.970
BOS cost offshore: 7613488.697

The following is an example use of csmBOS for calculating the onshore and offshore wind plant Balance of Station Foundation Costs using the NREL 5 MW reference turbine [1]_:

.. literalinclude:: ..\..\..\..\models\csm\csmFoundation.py
   :pyobject: example

Onshore foundation cost:
Foundation cost: 120779.05392
Offshore foundation cost:
Foundation cost: 2125550.66079

**References**

.. [1] Jonkman, J.; Butterfield, S.; Musial, W.; Scott, G. (2009). "Definition of a 5-MW Reference Wind Turbine for Offshore System Development." NREL/TP-500-38060. Golden, CO: National Renewable Energy Laboratory, 75 pp.

Operations and Maintenance (O&M) Module
^^^^^^^^^^^^^^^^^^^

.. module:: csm.src.csmOM

The NREL Cost and Scaling Model operations and maintenance module calculates the operations and maintenance, land lease costs, levelized replacement costs for operating a wind plant (either onshore or offshore).

csmOM
""""""
.. currentmodule:: csm.src.csmOM
.. class:: csmOM
.. currentmodule:: csm.src.csmOM.csmOM

**Methods**

.. autosummary::
    __init__
    compute
    getOMCost
    getLLC
    getLRC

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        getOMCost
        getLLC
        getLRC


**Examples**
The following is an example use of csmOM for calculating the onshore and offshore wind plant Operations and Maintenance Costs:

.. literalinclude:: ..\..\..\..\models\csm\csmOM.py
   :pyobject: example

Output should be:

OM costs onshore 144053.487 LevRep 58699.223 Lease 22225.395

OM costs offshore 401819.023  LevRep 91048.387 Lease 22225.395


Annual Energy Production (AEP)
^^^^^^^^^^^^^^^

The annual energy production model uses methods as described in the references for determining a wind plant annual energy production from a small number of site and turbine design input parameters.  Within the AEP module are sub-modules for determining the wind turbine power curve and drivetrain efficiency.

.. toctree::
   :maxdepth: 2

   csmDriveEfficiency
   csmPowerCurve

csmAEP
""""""

.. module:: csm.src.csmAEP

The NREL Cost and Scaling Model AEP calculation based on a Weibull distribution of wind speed and input assumptions regarding plant losses.

.. currentmodule:: csm.src.csmAEP
.. class:: csmAEP 
.. currentmodule:: csm.src.csmAEP.csmAEP 

**Methods**

.. autosummary::
    __init__
    compute
    getAEP
    getCapacityFactor

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        getAEP
        getCapacityFactor

**Example**
The following is an example use of csmAEP to calculate the annual energy production for a fictitious plant using the NREL 5 MW Reference Turbine [1]_:

.. literalinclude:: ..\..\..\..\models\csm\csmAEP.py
   :pyobject: example

The output should appear as:
AEP:        18744306.322 MWh (for Rated Power: 5000.0)
CapFactor:      42.795 %

**References**

.. [1] Jonkman, J.; Butterfield, S.; Musial, W.; Scott, G. (2009). "Definition of a 5-MW Reference Wind Turbine for Offshore System Development." NREL/TP-500-38060. Golden, CO: National Renewable Energy Laboratory, 75 pp.


csmPowerCurve Module
^^^^^^^^^^^^^^^

.. module:: csm.src.csmPowerCurve 

This module determines a wind turbine power curve which accounts for drivetrain efficiency.  It is based on the NREL Cost and Scaling Model and requires a small number of input parameters as described below.

csmPowerCurve
"""""""""""""

The NREL Cost and Scaling Model Power Curve module.

.. currentmodule:: csm.src.csmPowerCurve
.. class:: csmPowerCurve
.. currentmodule:: csm.src.csmPowerCurve.csmPowerCurve

**Methods**

.. autosummary::
    __init__
    compute
    getRatedWindSpeed
    getRatedRotorSpeed
    getMaxEfficiency
    getPowerCurve
    idealPowerCurve

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        getRatedWindSpeed
        getRatedRotorSpeed
        getMaxEfficiency
        getPowerCurve
        idealPowerCurve
 

**Example**
The following is an example use of csmPowerCurve to the a power curve based on the NREL 5 MW Reference Turbine [1]_:

.. literalinclude:: ..\..\..\..\models\csm\csmPowerCurve.py
   :pyobject: example

The output should appear as:
Rated Speed: 11.750 mps
Rated RPM:   12.126 rpm

Along with the below plot.


**References**

.. [1] Jonkman, J.; Butterfield, S.; Musial, W.; Scott, G. (2009). "Definition of a 5-MW Reference Wind Turbine for Offshore System Development." NREL/TP-500-38060. Golden, CO: National Renewable Energy Laboratory, 75 pp.


Drivetrain Efficiency Module
^^^^^^^^^^^^^^^

.. module:: csm.src.csmDriveEfficiency 

This module describes an interface for drivetrain efficiency analysis as well as a model based on the NREL Cost and Scaling Model which implements the interface. The methods are necessary to determine the actual output power of a wind turbine to the interconnection point after drivetrain losses for the drivetrain (gearbox, generator, etc) are taken into account.  The simplified model is based on the analysis done in reference [1]_.

DrivetrainEfficiencyModel
""""""""""""""""""""""""

.. currentmodule:: csm.src.csmDriveEfficiency
.. class:: DrivetrainEfficiencyModel
.. currentmodule:: csm.src.csmDriveEfficiency.DrivetrainEfficiencyModel

**Methods**

.. autosummary::
    getMaxEfficiency
    getDrivetrainEfficiency

.. HACK
    .. autosummary::
        :toctree: generated
        getMaxEfficiency
        getDrivetrainEfficiency


csmDrivetrainModel
""""""""""""""""""

.. currentmodule:: csm.src.csmDriveEfficiency
.. class:: csmDriveEfficiency 
.. currentmodule:: csm.src.csmDriveEfficiency.csmDriveEfficiency

**Methods**

.. autosummary::
    __init__
    getMaxEfficiency
    getDrivetrainEfficiency

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        getMaxEfficiency
        getDrivetrainEfficiency


**Example**
The following is an example use of csmDriveEfficiency to determine the maximum drivetrain efficiency for a wind turbine with a 3-staged gearbox and induction generator drivetrain configuration:

.. literalinclude:: ..\..\..\..\models\csm\csmDriveEfficiency.py
   :pyobject: example

The output should appear as:
Max Efficiency:    0.902
Output power:  3422.000   Rated power:   5543.000    Efficiency:  0.894


**References**

.. [1] Maples, B.; Hand, M.; Musial, W. (2010). "Comparative Assessment of Direct Drive High Temperature Superconducting Generators in Multi-Megawatt Class Wind Turbines." NREL/TP-5000-49086. Golden, CO: National Renewable Energy Laboratory, 40 pp.


Finance
^^^^

.. module:: csm.src.csmFinance

This module uses the NREL Cost and Scaling Model to determine a wind plant project’s cost of energy and levelized cost of energy.  It takes as input all the aggregate costs and financial parameters for the project.


csmFinance
""""""""""
.. currentmodule:: csm.src.csmFinance
.. class:: csmFinance 
.. currentmodule:: csm.src.csmFinance.csmFinance

**Methods**

.. autosummary::
    __init__
    compute
    getCOE
    getLCOE

.. HACK
    .. autosummary::
        :toctree: generated
        __init__
        compute
        getCOE
        getLCOE


**Example**
The following is an example use of csmDriveEfficiency to determine the maximum drivetrain efficiency for a wind turbine with a 3-staged gearbox and induction generator drivetrain configuration:

.. literalinclude:: ..\..\..\..\models\csm\csmFinance.py
   :pyobject: example

The output should appear as:
Onshore:
LCOE: 0.052776
COE: 0.065765

Offshore:
LCOE: 0.102919
COE: 0.119694




