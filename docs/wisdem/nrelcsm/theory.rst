.. _csmtheory:

Theory
------

The theory for the models in this software are based directly on the work described in references :cite:`Fingersh2006,Sunderland1993,Malcolm2002,Maples2010`.  This section provides an overview of these simple mass and cost models for the major turbine components.

The NREL Cost and Scaling Model :cite:`Fingersh2006` provides a simple cost and sizing tool to estimate wind turbine component masses and costs based on a small number of input parameters such as rotor diameter, hub height and rated power.  The model was developed over several results following the Wind Partnerships for Advanced Component Technology (WindPACT) work that occurred between roughly 2002 to 2005.  The original form of the cost model was based on an earlier model from 1993 out of the University of Sunderland (the Sunderland Model) :cite:`Sunderland1993`.  The Sunderland Model created a set of wind turbine models to estimate the mass and cost of all major wind turbine components including: blade, hub system [hub, pitch system, and nose cone], nacelle [low speed shaft, main bearings, gearbox, high speed shaft/mechanical brake, generator, variable speed electronics, electrical cabling, mainframe [bedplate, platforms and railings, base hardware, and crane], HVAC system, controls, and nacelle cover], and tower.  The Sunderland model was based on a set of semi-empirical models for each component which estimated design loads at the rotor, propagated these loads through the entire system, and used the loads to estimate the size of each component calibrated to data on actual turbines in the field during that time.  Cost estimates were then made on a per weight basis using a multiplier again based on field data or industry sources.

To arrive at the NREL Cost and Scaling Model, the WindPACT studies began in many cases with the Sunderland model and updated the results with new coefficients or, in some cases, with entirely new cost equations based on curve fits of key design parameters (rotor diameter, etc) to the results of detailed design studies :cite:`Malcolm2002`.  In addition, the WindPACT work established estimates of costs associated with balance of station and operations and maintenance for a fictitious wind plant in North Dakota which led to an overall cost of energy model for a wind plant.  The key cost of energy equation for a wind plant is given in the NREL Cost and Scaling Model :cite:`Fingersh2006` as:

.. math:: COE = (FCR*(BOS+TCC))/AEP + (LLC + LRC + (1-tr)*OM)/AEP

where :math:`COE` in this equation is a simple estimate of a wind plant cost of energy, :math:`FCR` is the fixed charge rate for the project, :math:`BOS` are the total balance of station costs for the project, :math:`TCC` are the total turbine capital costs for the project, :math:`AEP` is the annual energy production for the project, :math:`LLC` are the annual land-lease costs, :math:`LRC` is the levelized replacement cost for major part replacement, :math:`tr` is the tax rate, and :math:`OM` are the annual operations and maintenance costs which are tax deductible.

While the NREL Cost and Scaling Model improved the overall cost estimation for larger turbines on the order of 1 MW+, it abstracted away from the engineering analysis foundations of the original Sunderland model.  This is depicted in the below figure where it can be seen that the engineering-analysis has been replaced by a series of curve fits which relate a small number of design parameters to mass and cost estimates for major wind turbine components.

.. _NRELCSM:

.. figure:: /images/turbine_costsse/NRELCSM.*
   :width: 5.5in
   :align: center

   NREL Cost and Scaling Model Key Input-Output Relationships.  TODO: REFRESH GRAPHIC

The resulting NREL Cost and Scaling Model (as provided in NREL_CSM_TCC) allows for a variety of interesting analyses including scaling of conventional technology from under a MW to 5 MW+, assessing impact of trends in input factors for materials and labor on wind plant cost of energy, etc.  However, it does not preserve the underlying engineering relationships of the original Sunderland model and thus loses some fidelity of assessing how design changes may impact system costs.

The goal of the development of the second model, Turbine_CostsSE, then is to provide a set of mass-based component cost calculations.  A mass-cost model is developed for each of the major turbine components.  These use the data underlying the NREL Cost and Scaling Model to estimate relationships that can then be scaled based on economic multipliers as done in :cite:`Fingersh2006`.  Details of the models are described next.

TODO

* The equation for blade costs includes both materials and manufacturing.

Blades
~~~~~~
To obtain the blade mass in kilograms and cost in USD from the rotor diameter in meters,

.. math::
   m_{blade} &= k_m (0.5 D_{rotor})^{b}\\
   c_{blade} &= k_c m_{blade}\\
   k_m &= 0.5\\
   b   &= (see below)\\
   k_c &= 14.6

Where :math:`D_{rotor}` is the rotor diameter and :math:`b` is determined by:

* If turbine class I and blade DOES have carbon fiber spar caps, :math:`b=2.47`
* If turbine class I and blade DOES NOT have carbon fiber spar caps, :math:`b=2.54`
* If turbine class II+ and blade DOES have carbon fiber spar caps, :math:`b=2.44`
* If turbine class II+ and blade DOES NOT have carbon fiber spar caps, :math:`b=2.50`
* User override of exponent value

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/BladeMass.*
   :width: 5in
   :align: center

Hub (Shell)
~~~~~~~~~~~
To obtain the hub shell mass in kilograms and cost in USD from the blade mass in kilgograms,

.. math::
   m_{hub} &= k_m m_{blade} + b\\
   c_{hub} &= k_c m_{hub}\\
   k_m &= 2.3\\
   b   &= 1320\\
   k_c &= 3.9

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/HubMass.*
   :width: 4in
   :align: center

Pitch System
~~~~~~~~~~~~
To obtain the pitch bearing and system mass in kilograms and cost in USD from the blade mass in kilgograms,

.. math::
   m_{bearing} &= n_{blade} k_m m_{blade} + b_1\\
   m_{pitch} &= m_{bearing} (1 + h) + b_2\\
   c_{pitch} &= k_c m_{pitch}\\
   k_m &= 0.1295\\
   b_1 &= 491.31\\
   b_2 &= 555\\
   h   &= 0.328\\
   k_c &= 22.1

Where :math:`n_{blade}` is the number of blades, :math:`h` is fractional mass of the pitch bearing housing.

For variable names access to override the default values see the :ref:`csmsource`.


Spinner (Nose Cone)
~~~~~~~~~~~~~~~~~~~
To obtain the spinner (nose cone) mass in kilograms and cost in USD from the rotor diameter in meters,

.. math::
   m_{spin} &= k_m D_{rotor} + b\\
   c_{spin} &= k_c m_{spin}\\
   k_m &= 15.5\\
   b   &= -980\\
   k_c &= 11.1

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/SpinnerMass.*
   :width: 4in
   :align: center

Low Speed Shaft
~~~~~~~~~~~~~~~
To obtain the low speed shaft mass in kilograms and cost in USD from the blade mass in kilograms and the machine rating in megawatts,

.. math::
   m_{lss} &= k_m (m_{blade} P_{turbine})^{b_1} + b_2\\
   c_{lss} &= k_c m_{lss}\\
   k_m &= 13\\
   b_1 &= 0.65\\
   b_2 &= 775\\
   k_c &= 11.9

Where :math:`P_{turbine}` is the machine rating.

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/LssMass.*
   :width: 4in
   :align: center

Main Bearings
~~~~~~~~~~~~~
To obtain the main bearings mass in kilograms and cost in USD from the rotor diameter in meters,

.. math::
   m_{bearing} &= n_{bearing} k_m D_{rotor}^b\\
   c_{bearing} &= k_c m_{bearing}\\
   k_m &= 0.0001\\
   b   &= 3.5\\
   k_c &= 4.5

Where :math:`D_{rotor}` is the rotor diameter and :math:`n_{bearing}` is the number of bearings.

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/BearingMass.*
   :width: 4in
   :align: center


Gearbox
~~~~~~~
To obtain the main bearings mass in kilograms and cost in USD from the rotor torque in kilo-Newton meters,

.. math::
   m_{gearbox} &= k_m Q_{rotor}^b\\
   c_{gearbox} &= k_c m_{gearbox}\\
   k_m &= 113\\
   b   &= 0.71\\
   k_c &= 12.9

Where :math:`Q_{rotor}` is the rotor torque and is approximated by,

.. math::
   Q_{rotor} = \frac{0.5 P_{turbine} D_{rotor}}{\eta V_{tip}}

Where :math:`P_{turbine}` is the machine rating, :math:`D_{rotor}` is the rotor diameter, :math:`V_{tip}` is the max tip speed, and :math:`\eta` is the drivetrain efficiency.

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/GearboxMass.*
   :width: 4in
   :align: center

Brake
~~~~~
To obtain the brake mass in kilograms and cost in USD from the rotor torque in kilo-Newton meters (updated in 2020 by J. Keller)),

.. math::
   m_{brake} &= k_m Q_{rotor}\\
   c_{brake} &= k_c m_{brake}\\
   k_m &= 1.22\\
   k_c &= 3.6254

Where :math:`Q_{rotor}` is the rotor torque and is approximated above.

For variable names access to override the default values see the :ref:`csmsource`.

High Speed Shaft
~~~~~~~~~~~~~~~~
To obtain the high speed shaft mass in kilograms and cost in USD from the machine rating in megawatts,

.. math::
   m_{hss} &= k_m P_{turbine}\\
   c_{hss} &= k_c m_{hss}\\
   k_m &= 198.94\\
   k_c &= 6.8

Where :math:`P_{turbine}` is the machine rating.

For variable names access to override the default values see the :ref:`csmsource`.

Generator
~~~~~~~~~
To obtain the generator mass in kilograms and cost in USD from the machine rating in megawatts,

.. math::
   m_{generator} &= k_m P_{turbine} + b\\
   c_{generator} &= k_c m_{generator}\\
   k_m &= 2300\\
   b   &= 3400\\
   k_c &= 12.4

Where :math:`P_{turbine}` is the machine rating.

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/GeneratorMass.*
   :width: 4in
   :align: center

Yaw System
~~~~~~~~~~
To obtain the yaw system mass in kilograms and cost in USD from the rotor diameter in meters,

.. math::
   m_{yaw} &= k_m D_{rotor}^b\\
   c_{yaw} &= k_c m_{yaw}\\
   k_m &= 0.00135\\
   b   &= 3.314\\
   k_c &= 8.3

Where :math:`D_{rotor}` is the rotor diameter.

For variable names access to override the default values see the :ref:`csmsource`.

Hydraulic Cooling
~~~~~~~~~~~~~~~~~
To obtain the hydraulic cooling mass in kilograms and cost in USD from the machine rating in megawatts,

.. math::
   m_{hvac} &= k_m P_{turbine}\\
   c_{hvac} &= k_c m_{hvac}\\
   k_m &= 80\\
   k_c &= 124

Where :math:`P_{turbine}` is the machine rating.

For variable names access to override the default values see the :ref:`csmsource`.

Transformer
~~~~~~~~~~~
To obtain the transformer mass in kilograms and cost in USD from the machine rating in megawatts,

.. math::
   m_{transformer} &= k_m P_{rotor} + b\\
   c_{transformer} &= k_c m_{transformer}\\
   k_m &= 1915\\
   b   &= 1910\\
   k_c &= 18.8

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/TransformerMass.*
   :width: 4in
   :align: center

Cabling and Electrical Connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To obtain the cabling and electrical connections cost in USD (there is no mass calculated) from the machine rating in megawatts,

.. math::
   c_{connect} &= k_c P_{rotor}\\
   k_c &= 41850

Where :math:`P_{turbine}` is the machine rating.

For variable names access to override the default values see the :ref:`csmsource`.

Control System
~~~~~~~~~~~~~~
To obtain the control system cost in USD (there is no mass calculated) from the machine rating in megawatts,

.. math::
   c_{control} &= k_c P_{rotor}\\
   k_c &= 21150

Where :math:`P_{turbine}` is the machine rating.

For variable names access to override the default values see the :ref:`csmsource`.

Other Nacelle Equipment
~~~~~~~~~~~~~~~~~~~~~~~
To obtain the nacelle platform and service crane mass in kilograms and cost in USD from the bedplate mass in kilograms,

.. math::
   m_{platform} &= k_m m_{bedplate}\\
   c_{platform} &= k_c m_{platform}\\
   m_{crane} &= 3000\\
   c_{crane} &= 12000\\
   k_m &= 0.125\\
   k_c &= 17.1

Note that the service crane is optional with a flag set by the user.

For variable names access to override the default values see the :ref:`csmsource`.

Bedplate
~~~~~~~~
To obtain the bedplate mass in kilograms and cost in USD from the rotor diameter in meters,

.. math::
   m_{bedplate} &= D_{rotor}^b\\
   c_{bedplate} &= k_c m_{bedplate}\\
   b   &= 2.2\\
   k_c &= 2.9

Where :math:`D_{rotor}` is the rotor diameter.  The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/BedplateMass.*
   :width: 4in
   :align: center

For variable names access to override the default values see the :ref:`csmsource`.



Nacelle Cover
~~~~~~~~~~~~~
To obtain the nacelle cover mass in kilograms and cost in USD from the machine rating in megawatts,

.. math::
   m_{cover} &= k_m P_{turbine} + b\\
   c_{cover} &= k_c m_{cover}\\
   k_m &= 1.2817\\
   b   &= 428.19\\
   k_c &= 5.7

Where :math:`P_{turbine}` is the machine rating.

For variable names access to override the default values see the :ref:`csmsource`.


Tower
~~~~~
To obtain the tower mass in kilograms and cost in USD from the hub height in meters,

.. math::
   m_{tower} &= k_m L_{hub}^b\\
   c_{tower} &= k_c m_{tower}\\
   k_m &= 19.828\\
   b   &= 2.0282\\
   k_c &= 2.9

Where :math:`L_{hub}` is the hub height.

For variable names access to override the default values see the :ref:`csmsource`.

The mass scaling relationships are based on the following data,

.. figure:: /images/turbine_costsse/TowerMass.*
   :width: 4in
   :align: center

Sub-System Aggregations
~~~~~~~~~~~~~~~~~~~~~~~

There are further aggregations of the components into sub-systems, at which point additional costs and/or multipliers are included.  For the mass accounting, this includes hub system mass, rotor mass, nacelle mass, and total turbine mass,

Hub System
==========

It is assumed that the hub system is assembled and transported as a one unit, thus there are additional costs at this level of aggregation,

.. math::
   m_{hubsys} &= m_{hub} + m_{pitch} + m_{spinner}\\
   c_{hubsys} &= (1+kt_{hubsys}+kp_{hubsys}) (1+ko_{hubsys}+ka_{hubsys}) (c_{hub} + c_{pitch} + c_{spinner})

Where conceptually, :math:`kt` is a transportation multiplier, :math:`kp` is a profit multiplier, :math:`ko` is an overhead cost multiplier, and :math:`ka` is an assembly cost multiplier.  By default, :math:`kt=kp=ko=ka=0`.

For variable names access to override the default values see the :ref:`csmsource`.


Rotor System
============

The rotor mass and cost is aggregated for conceptual convenience, but it is assumed to be transported in separate pieces and assembled on-site, so there are no separate sub-system cost multipliers.

.. math::
   m_{rotor} &= n_{blade} m_{blade} + m_{hubsys}\\
   c_{rotor} &= n_{blade} c_{blade} + c_{hubsys}

For variable names access to override the default values see the :ref:`csmsource`.



Nacelle
=======

It is assumed that the nacelle and all of its sub-components are assembled and transported as a one unit, thus there are additional costs at this level of aggregation,

.. math::
   m_{nacelle} &= m_{lss} + m_{bearing} + m_{gearbox} + m_{hss} + m_{generator} +m_{bedplate} + \\
   &m_{yaw} + m_{hvac} + m_{transformer} + m_{platform} + m_{cover}\\
   c_{parts} &= c_{lss} + c_{bearing} + c_{gearbox} + c_{hss} + c_{generator} +c_{bedplate} + \\
   &c_{yaw} + c_{hvac} + c_{transformer} + c_{connect} + c_{control} + c_{platform} + c_{cover}\\
   c_{nacelle} &= (1+kt_{nacelle}+kp_{nacelle}) (1+ko_{nacelle}+ka_{nacelle}) c_{parts}

Where conceptually, :math:`kt` is a transportation multiplier, :math:`kp` is a profit multiplier, :math:`ko` is an overhead cost multiplier, and :math:`ka` is an assembly cost multiplier.  By default, :math:`kt=kp=ko=ka=0`.

For variable names access to override the default values see the :ref:`csmsource`.



Tower System
============

The tower is not aggregated with any other component, but for consistency there are allowances for additional costs incurred from transportation and assembly complexity,

.. math::
   c_{towersys} = (1+kt_{tower}+kp_{tower}) (1+ko_{tower}+ka_{tower}) c_{tower}

Where conceptually, :math:`kt` is a transportation multiplier, :math:`kp` is a profit multiplier, :math:`ko` is an overhead cost multiplier, and :math:`ka` is an assembly cost multiplier.  By default, :math:`kt=kp=ko=ka=0`.

For variable names access to override the default values see the :ref:`csmsource`.


Turbine
=======

The final turbine assembly also allows for user specification of other cost multipliers,

.. math::
   m_{turbine} &= m_{rotor} + m_{nacelle} + m_{tower}\\
   c_{turbine} &= (1+kt_{turbine}+kp_{turbine}) (1+ko_{turbine}+ka_{turbine}) (c_{rotor} + c_{nacelle} + c_{towersys})

For variable names access to override the default values see the :ref:`csmsource`.


.. bibliography:: ../../references.bib
   :filter: docname in docnames
