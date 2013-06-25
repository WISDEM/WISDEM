.. _theory:

Theory
------

The theory for the models in this software are based directly on the work described in references :cite:`Fingersh2006`, :cite:`WindPACT`, :cite:`Sunderland1993`, :cite:`Malcolm2002`, and :cite:`Maples2010`.  This section provides an overview of the development of the physical and cost models for the major hub and drivetrain components. 

The NREL Cost and Scaling Model :cite:`Fingersh2006` provides a simple cost and sizing tool to estimate wind turbine component masses and costs based on a small number of input parameters such as rotor diameter, hub height and rated power.  The model was developed over several results following the Wind Partnerships for Advanced Component Technology (WindPACT) work that occurred between roughly 2002 to 2005 :cite:`WindPACT`.  The original form of the cost model was based on an earlier model from 1993 out of the University of Sunderland (the Sunderland Model) :cite:`Sunderland1993`.  The Sunderland Model created a set of wind turbine models to estimate the mass and cost of all major wind turbine components including: blade, hub system [hub, pitch system, and nose cone], nacelle [low speed shaft, main bearings, gearbox, high speed shaft/mechanical brake, generator, variable speed electronics, electrical cabling, mainframe [bedplate, platforms and railings, base hardware, and crane], HVAC system, controls, and nacelle cover], and tower.  The Sunderland model was based on a set of semi-empirical models for each component which estimated design loads at the rotor, propagated these loads through the entire system, and used the loads to estimate the size of each component calibrated to data on actual turbines in the field during that time.  Cost estimates were then made on a per weight basis using a multiplier again based on field data or industry sources.  

To arrive at the NREL Cost and Scaling Model, the WindPACT studies began in many cases with the Sunderland model and updated the results with new coefficients or, in some cases, with entirely new cost equations based on curve fits of key design parameters (rotor diameter, etc) to the results of detailed design studies :cite:`Malcolm2002`.  In addition, the WindPACT work established estimates of costs associated with balance of station and operations and maintenance for a fictitious wind plant in North Dakota which led to an overall cost of energy model for a wind plant.  The key cost of energy equation for a wind plant is given in the NREL Cost and Scaling Model :cite:`Fingersh2006` as:

.. math:: COE = (FCR*(BOS+TCC))/AEP + (LLC + LRC + (1-tr)*OM)/AEP

where :math:`COE` in this equation is a simple estimate of a wind plant cost of energy, :math:`FCR` is the fixed charge rate for the project, :math:`BOS` are the total balance of station costs for the project, :math:`TCC` are the total turbine capital costs for the project, :math:`AEP` is the annual energy production for the project, :math:`LLC` are the annual land-lease costs, :math:`LRC` is the levelized replacement cost for major part replacement, :math:`tr` is the tax rate, and :math:`OM` are the annual operations and maintenance costs which are tax deductible. 

While the NREL Cost and Scaling Model improved the overall cost estimation for larger turbines on the order of 1 MW+, it abstracted away from the engineering analysis foundations of the original Sunderland model.  This is depicted in the below figure where it can be seen that the engineering-analysis has been replaced by a series of curve fits which relate a small number of design parameters to mass and cost estimates for major wind turbine components. 

.. _NRELCSM:

.. figure:: /images/NRELCSM.*
    :width: 5.5in

    NREL Cost and Scaling Model Key Input-Output Relationships.

The resulting NREL Cost and Scaling Model allows for a variety of interesting analyses including scaling of conventional technology from under a MW to 5 MW+, assessing impact of trends in input factors for materials and labor on wind plant cost of energy, etc.  However, it does not preserve the underlying engineering relationships of the original Sunderland model and thus loses some fidelity of assessing how design changes may impact system costs.  

The goal of the development of the following set of models for the hub and nacelle are to return to a semi-empirical physical representation of the turbine that allows one to assess the impact of design changes on dimensions, mass and mass properties while updating the Sunderland model in key areas of limitation related to outdated design criteria.  The resulting models contain two key areas:

1) A physical model for sizing each of the major hub and drivetrain components based on design loads as they are transferred through the wind turbine system (tied closely to the Sunderland Model :cite:`Sunderland1993` with some updates for WindPACT work described in :cite:`WindPACT` and :cite:`Malcolm2002` and updates for the gearbox and generator in particular based on :cite:`Maples2010`).  Using reference :cite:`Malcolm2002`, mass and baseline costs were collected from Table 6-2, p. 39 while final design costs were collected from Table B-2, p. B-2.

2) A model of the Mass Moments of Inertia for each of the components based on an assumption of homogenous distribution of homogeneous material throughout a simplified geometric structure.  These are necessary for as inputs into a model of overall mass properties for a rotor-nacelle-assembly (RNA) whose properties are fed to an engineering analysis model for the tower.

These models also require an estimate of critical dimensions of each major component.  The dimension estimates as well as some more advanced equations for Mass Moments of Inertia are based on the WindPACT design studies as described in :cite:`Malcolm2002`.  The data on critical dimensions are used in other models such as the balance of station model.

The center of mass for each component is also estimated based on the design studies as described in :cite:`Malcolm2002` and can be used to calculate the aggregate mass properties of the rotor-nacelle-assembly in other models.

Hub System Masses
^^^^^^^^^^^^^^^^^

The model begins with the Sunderland Model :cite:`Sunderland1993` for the major hub components including the hub itself and the pitch system.  The spinner / nose cone component mass is determined by the NREL Cost and Scaling Model :cite:`Fingersh2006`.  The hub design of the Sunderland model assume a hub with three-fused cylinders rather than a spherical shape as is more common of modern designs.  However, the coefficient multiplier for the hub model was modified from 31.4 to 50 in order to adapt it to data from updated turbine designs of 1 MW+ in size.

The resulting masses from the model can be compared to the estimates from the WindPACT detailed design studies :cite:`WindPACT` and the NREL Cost and Scaling Model output.  The below graphs show the mass comparisons for different size turbines of 750 kW, 1 MW, 3 MW and 5 MW for the full hub system as well as each sub-assembly.

.. _hubsysmass:

.. figure:: /images/hubsysmass.*
    :width: 6.5in

    Hub System overall mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _hubmass:

.. figure:: /images/hubmass.*
    :width: 6.5in

    Hub component mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _pitchsysmass:

.. figure:: /images/pitchsysmass.*
    :width: 6.5in

    Pitch System overall mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _spinnermass:

.. figure:: /images/spinnermass.*
    :width: 6.5in

    Spinner component mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

Limitations on data hamper validation efforts but there are also known issues with the models: notably the use of the Sunderland Model as a foundation which was developed based on technology from previous generations.  Still, there is decent agreement between the model output and both the WindPACT study data as well as the NREL Cost and Scaling Model output given the same input spectifications.  Future work will look at updates to the entire set of hub system models so that they are compatible with current technology.

Hub System Mass Moments of Inertia
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the mass moments of inertia, it is assumed that the hub system is a hollow sphere with its diameter either provided as input or determined as a fraction of the overall rotor diameter.  The thickness is assumed as a fraction of the overall rotor diameter.  The hub and nose cone mass are lumped together in a single quantity while the contribution from the pitch system is modeled as an additional mass about the hub with the hub radius as determined above.  The resulting equation is:

.. math:: I_{xx}=I_{yy}=I_{zz}=2*(hubmass + spinnermass)/5 * (r_{outer}^5 - r_{inner}^5)/(r_{outer}^3 - r_{inner}^3) + pitchmass * r_{outer}^2

where :math:`I_{xx}`, :math:`I_{yy}`, and :math:`I_{zz}` are the mass moments of inertia around the respective axis using the wind turbine yaw-aligned coordinate system, :math:`hubmass`, :math:`spinnermass`, and :math:`pitchmass` are the masses of the hub, spinner and pitch system respectively, and :math:`r_{outer}` and :math:`r_{inner}` are the inner and outer radii of the hub respectively.

The center of mass for the hub system is determined relatively to the tower top center as a function of rotor diameter.

.. math:: cm_{x} = -(0.05 * diam)

.. math:: cm_{y} = 0.0

.. math:: cm_{z} = 0.025 * diam

Where :math:`cm_{x}`,:math:`cm_{y}` and :math:`cm_{z}` are the positions of center of mass and :math:`diam` is the rotor diameter of the turbine.


Nacelle Assembly System Masses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The model begins with the Sunderland Model :cite:`Sunderland1993` for the major nacelle components including the low speed shaft, main bearings, gearbox, high speed shaft, bedplate and yaw system.  The generator component mass is determined by the NREL Cost and Scaling Model :cite:`Fingersh2006` including updates for the drivetrain as described in :cite:`Maples2010`.  Some modifications to the models were made where necessary in order to adapt it to data from updated turbine designs of 1 MW+ in size:

* The low-speed shaft model includes a mass weight factor of 1.25 which was developed as part of updates as described in :cite:`Maples2010`.
* The main bearings model includes a mass weight factor of 0.25 to bring the model into closer alignment with :cite:`Fingersh2006`
* The gearbox model includes a stage weight factor for each parallel and epicyclic stage.  The stage weight factor for the epicyclic stages has been reduced by a factor of 12.0 to align the model closer to modern gearbox sizes.
* The mechanical brake mass is created via a multiplier of 0.5 to the high speed shaft mass to make it consistent with :cite:`Fingersh2006` and :cite:`Malcolm2002`.

The models otherwise follow the references and also include the following components (after reference :cite:`Fingersh2006`): variable speed electronics, electrical connections and controls, HVAC system, mainframe platforms, base hardware and crane, and nacelle cover.

The resulting masses from the model can be compared to the estimates from the WindPACT detailed design studies :cite:`WindPACT` and the NREL Cost and Scaling Model output.  The below graphs show the mass comparisons for different size turbines of 750 kW, 1 MW, 3 MW and 5 MW for the full hub system as well as each of the modified sub-assemblies: the low-speed shaft, main bearings, gearbox, high-speed shaft and brake, bedplate and yaw system.

.. _nacellemass:

.. figure:: /images/nacellemass.*
    :width: 6.5in

    Nacelle overall system mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _lssmass:

.. figure:: /images/lssmass.*
    :width: 6.5in

    LSS component mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _bearingsmass:

.. figure:: /images/bearingsmass.*
    :width: 6.5in

    Main bearing system mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _gearboxmass:

.. figure:: /images/gearboxmass.*
    :width: 6.5in

    Gearbox mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _hssmass:

.. figure:: /images/hssmass.*
    :width: 6.5in

    HSS and brake component overall mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _bedplatemass:

.. figure:: /images/bedplatemass.*
    :width: 6.5in

    Bedplate mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

.. _yawmass:

.. figure:: /images/yawmass.*
    :width: 6.5in

    Yaw system mass comparison between the NREL Cost and Scaling Model, WindPACT detailed design studies, the NREL 5 MW reference turbine and the NREL Hub System Cost and Sizing Tool.

Limitations on data hamper validation efforts but there are also known issues with the models: notably the use of the Sunderland Model as a foundation which was developed based on technology from previous generations.  Still, there is decent agreement between the model output and both the WindPACT study data as well as the NREL Cost and Scaling Model output given the same input spectifications.  Future work will look at updates to the entire set of nacelle assembly models so that they are compatible with current technology.

Nacelle Assembly Mass Moments of Inertia
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the mass moments of inertia, it is assumed that basic geometric shapes with homogeneous distribution of homogenous materials are assumed.  Each major component has its own set of calculations.  The assumptions regarding the shape of components is based on the design studies as described in :cite:`WindPACT`.  

Low Speed Shaft:

The low speed shaft is assumed to be a hollow-cylinder along the yaw-aligned x-axis.  The resulting equations are:

.. math:: I_{xx}=lssmass * (r_{outer}^2 + r_{inner}^2) / 8

.. math:: I_{yy}=I_{zz}=lssmass * (r_{outer}^2 + r_{inner}^2 + (4/3) * lsslength^2)) / 16

where :math:`I_{xx}`, :math:`I_{yy}`, and :math:`I_{zz}` are the mass moments of inertia around the respective axis using the wind turbine yaw-aligned coordinate system, :math:`lssmass` is the mass of the low speed shaft, :math:`lsslength` is the length of the shaft, and :math:`r_{outer}` and :math:`r_{inner}` are the inner and outer radii of the hub respectively.

The center of mass for the low speed shaft is determined relatively to the tower top center as a function of rotor diameter.

.. math:: cm_{x} = -(0.035 – 0.1) * diam

.. math:: cm_{y} = 0.0

.. math:: cm_{z} = 0.025 * diam

Where :math:`cm_{x}`,:math:`cm_{y}` and :math:`cm_{z}` are the positions of center of mass and :math:`diam` is the rotor diameter of the turbine.

Main bearings:

The main bearings and their housings are assumed to be hoops along the yaw-aligned x-axis with a bearing hoop radius equal to the outer diameter of the low speed shaft and a housing hoop radius equal to a multiplier of the bearing radius of 1.5.  The resulting equations are:

.. math:: I_{xx}=(bearingmass * (r_{bearing}^2) / 4) + (housingmass * (r_{housing}^2) / 4) 

.. math:: I_{yy}=I_{zz}=I_{xx}/2

Where :math:`bearingmass` and :math:`housingmass` are the mass of an individual main bearing and its housing respectively, :math:`lsslength` is the length of the shaft, and :math:`r_{bearing}` and :math:`r_{housing}` are the radii of the bearing and its housing respectively.

The center of mass for the main bearing is determined relatively to the tower top center as a function of rotor diameter.

.. math:: cm_{x} = -(0.035 * diam)

.. math:: cm_{y} = 0.0

.. math:: cm_{z} = 0.025 * diam

The center of mass for the second bearing is determined relatively to the tower top center as a function of rotor diameter.

.. math:: cm_{x} = -(0.01 * diam)

.. math:: cm_{y} = 0.0

.. math:: cm_{z} = 0.025 * diam


Gearbox:

The gearbox is assumed to be a solid cylinder while the housing is a hollow cylinder along the yaw-aligned x-axis.  The radius of the gearbox is proportional to the housing height.  The resulting equations is:

.. math:: I_{xx}=(gearboxmass * diameter^2 / 8)+ (gearboxmass/2 * height^2 / 8) 

.. math:: I_{yy}=I_{zz}= gearboxmass * (0.5 * diameter^2 + (2/3) * length^2 + (1/4) * height^2) / 8 

where :math:`gearboxmass` is the mass of the gearbox, :math:`length` is the length of the gearbox, and :math:`diameter` and :math:`height` are the radii of the gearbox diameter and height respectively.

The center of mass for the gearbox is determined relatively to the tower top center as a function of rotor diameter.

.. math:: cm_{x} = 0.0

.. math:: cm_{y} = 0.0

.. math:: cm_{z} = 0.025 * diam

High Speed Shaft and Brake:

The high speed shaft is neglected and the brake disk is used for calculations as roughly a cylinder along the yaw-aligned x-axis.  The calculation, however, deviate for math:`I_{xx}`.  The resulting equations are:

.. math:: I_{xx}=(1/4) * length * pi * diameter^2 * gearratio * diameter^2) / 8

.. math:: I_{yy}=I_{zz}= hssmass * ((3/4) * diameter^2 + length^2) / 12

Where :math:`hssmass` is the mass of the high speed shaft and brake, :math:`lsslength` is the length of the brake, and :math:`diameter` is the diameter of the low speed shaft, and :math:`gearratio` is the high-speed to low speed shaft ratio.

The center of mass for the low speed shaft is determined relatively to the tower top center as a function of rotor diameter.

.. math:: cm_{x} = 0.5 * (0.0125 * diam)

.. math:: cm_{y} = 0.0

.. math:: cm_{z} = 0.025 * diam

Generator:

The generator housing is treated as a hollow cylinder along the yaw-aligned x-axis.  However, the rotating inertia is also approximated and this has a significant impact on the the calculations for math:`I_{xx}`.  The resulting equations are:

.. math:: I_{xx}=((4.86 * 10^-5) * diameter^5.333) + ((2/3) * generatormass * (depth^2 + width^2))/8

.. math:: I_{yy}=I_{zz}= (I_{xx}/2)/(gearratio^2) + ((1/3) * generatormass * length^2 / 12) + ((2/3) * generatormass * (depth^ 2 + width^2 + (4/3)*length^2) / 16)

Where :math:`generatormass` is the mass of the generator and housing, :math:`lsslength` is the length of the housing, :math:`width` is the width and :math:`height` is the height, :math:`gearratio` is the high-speed to low speed shaft ratio, and :math:`diam` is the overall turbine rotor diameter as before.

The center of mass for the low speed shaft is determined relatively to the tower top center as a function of rotor diameter.

.. math:: cm_{x} = 0.0125 * diam

.. math:: cm_{y} = 0.0

.. math:: cm_{z} = 0.025 * diam

Bedplate:

Finally, the bedplate is modeled as a hollow cylinder along the yaw-aligned x-axis.  The resulting equations are:

.. math:: I_{xx}=bedplatemass * (width^2 + depth^2) / 8

.. math:: I_{yy}=I_{zz}=bedplatemass * (width^2 + depth^2 + (4/3)*length^2) / 16

Where :math:`bedplatemass` is the mass of the bedplate, :math:`length` is the length of the housing, :math:`width` is the width, and :math:`depth` is the depth.

The center of mass for the low speed shaft is determined relatively to the tower top center as a function of rotor diameter.

.. math:: cm_{x} = 0.0

.. math:: cm_{y} = 0.0

.. math:: cm_{z} = 0.0122 * diam

Other components:

The other masses in the system are small compared to the major drivetrain components and are ignored for the purposes of calculating overall center of mass and mass moments of inertia for the nacelle assembly.


.. only:: html

    :bib:`Bibliography`

.. bibliography:: references.bib
    :style: unsrt
