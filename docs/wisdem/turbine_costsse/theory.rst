.. _theory:

Theory
------

The theory for the models in this software are based directly on the work described in references :cite:`Fingersh2006`, :cite:`WindPACT`, :cite:`Sunderland1993`, :cite:`Malcolm2002`, and :cite:`Maples2010`.  This section provides an overview of the development of these simple cost models for the major turbine components. 

The NREL Cost and Scaling Model :cite:`Fingersh2006` provides a simple cost and sizing tool to estimate wind turbine component masses and costs based on a small number of input parameters such as rotor diameter, hub height and rated power.  The model was developed over several results following the Wind Partnerships for Advanced Component Technology (WindPACT) work that occurred between roughly 2002 to 2005 :cite:`WindPACT`.  The original form of the cost model was based on an earlier model from 1993 out of the University of Sunderland (the Sunderland Model) :cite:`Sunderland1993`.  The Sunderland Model created a set of wind turbine models to estimate the mass and cost of all major wind turbine components including: blade, hub system [hub, pitch system, and nose cone], nacelle [low speed shaft, main bearings, gearbox, high speed shaft/mechanical brake, generator, variable speed electronics, electrical cabling, mainframe [bedplate, platforms and railings, base hardware, and crane], HVAC system, controls, and nacelle cover], and tower.  The Sunderland model was based on a set of semi-empirical models for each component which estimated design loads at the rotor, propagated these loads through the entire system, and used the loads to estimate the size of each component calibrated to data on actual turbines in the field during that time.  Cost estimates were then made on a per weight basis using a multiplier again based on field data or industry sources.  

To arrive at the NREL Cost and Scaling Model, the WindPACT studies began in many cases with the Sunderland model and updated the results with new coefficients or, in some cases, with entirely new cost equations based on curve fits of key design parameters (rotor diameter, etc) to the results of detailed design studies :cite:`Malcolm2002`.  In addition, the WindPACT work established estimates of costs associated with balance of station and operations and maintenance for a fictitious wind plant in North Dakota which led to an overall cost of energy model for a wind plant.  The key cost of energy equation for a wind plant is given in the NREL Cost and Scaling Model :cite:`Fingersh2006` as:

.. math:: COE = (FCR*(BOS+TCC))/AEP + (LLC + LRC + (1-tr)*OM)/AEP

where :math:`COE` in this equation is a simple estimate of a wind plant cost of energy, :math:`FCR` is the fixed charge rate for the project, :math:`BOS` are the total balance of station costs for the project, :math:`TCC` are the total turbine capital costs for the project, :math:`AEP` is the annual energy production for the project, :math:`LLC` are the annual land-lease costs, :math:`LRC` is the levelized replacement cost for major part replacement, :math:`tr` is the tax rate, and :math:`OM` are the annual operations and maintenance costs which are tax deductible. 

While the NREL Cost and Scaling Model improved the overall cost estimation for larger turbines on the order of 1 MW+, it abstracted away from the engineering analysis foundations of the original Sunderland model.  This is depicted in the below figure where it can be seen that the engineering-analysis has been replaced by a series of curve fits which relate a small number of design parameters to mass and cost estimates for major wind turbine components. 

.. _NRELCSM:

.. figure:: /images/turbine_costsse/NRELCSM.*
    :width: 5.5in

    NREL Cost and Scaling Model Key Input-Output Relationships.

The resulting NREL Cost and Scaling Model (as provided in NREL_CSM_TCC) allows for a variety of interesting analyses including scaling of conventional technology from under a MW to 5 MW+, assessing impact of trends in input factors for materials and labor on wind plant cost of energy, etc.  However, it does not preserve the underlying engineering relationships of the original Sunderland model and thus loses some fidelity of assessing how design changes may impact system costs.  

The goal of the development of the second model, Turbine_CostsSE, then is to provide a set of mass-based component cost calculations.  A mass-cost model is developed for each of the major turbine components.  These use the data underlying the NREL Cost and Scaling Model to estimate relationships that can then be scaled based on economic multipliers as done in :cite:`Fingersh2006`.  Details of the models are described next.

Turbine Component Mass-Cost Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A set of models based on the NREL Cost and Scaling model data have been developed to produce relationships of mass-to-cost for all major wind turbine components :cite:`Fingersh2006`.  These in many cases supplant the NREL Cost and Scaling model cost equations for wind turbine components which are often based on a small selection of design parameters such as rotor diameter, hub height and rated power.  The set of wind turbine mass-to-cost models developed include the components: blade, hub system [hub, pitch system, and nose cone], nacelle [low speed shaft, main bearings, gearbox, high speed shaft/mechanical brake, generator, variable speed electronics, electrical cabling, mainframe [bedplate, platforms and railings, base hardware, and crane], HVAC system, controls, and nacelle cover], and tower.  In addition, a mass-to-cost model for offshore monopile foundations has been established based on the new NREL Balance of Station Model :cite:`WindPACT`.  This section will describe the mass-to-cost models each of the major wind turbine components.

Blades:

The new NREL blades mass-cost model is based on the data of the NREL Cost and Scaling Model which was acquired via the WindPACT design studies efforts :cite:`Sunderland1993`.  The data for the blade costs in particular stem from the "WindPACT Turbine Rotor Design Study" :cite:`Malcolm2002` as well as the "Cost Study for Large Wind Turbine Blades:  WindPACT Blade System Design Studies" :cite:`TPI2003`.  The equation for blade costs includes both materials and manufacturing.  The NREL Cost and Scaling Model has built in escalators to update labor and material input cost factors based on cost trends over time.  The model here is reduced to a cost model relationship dependent only on mass as is consistent with the full set of mass-to-cost models.  A graph of the relationships for mass-to-cost from the WindPACT study data based on 2002 USD is shown below.

.. _BladeCost:

.. figure:: /images/turbine_costsse/BladeCost.*
    :width: 6.5in

    Blade mass-cost relationship based on NREL Cost and Scaling Model.

Hub System:

The cost model for the hub and spinner components are already based on a mass-to-cost relationship and so no adaptation is needed.  For the pitch system, a new mass-to-cost relationship based on the WindPACT study :cite:`Malcolm2002` appendix C for bearing data.  The costs are escalated as described in the NREL Cost and Scaling Model.

.. _pitchCost:

.. figure:: /images/turbine_costsse/pitchCost.*
    :width: 6.5in

    Pitch system mass-cost relationship based on NREL Cost and Scaling Model.

The mass-cost model for the pitch system was built using the equation as presented on the above figure mutliplied by a factor of 2.28 to account for the pitch system housing as was done in the NREL Cost and Scaling Model.

Drivetrain and Nacelle:

The major components of the low-speed shaft, gearbox, yaw drive and mainframe were adapted to a mass-cost relationship based on data from the WindPACT study :cite:`Malcolm2002`.  The relationship for the main bearings was already mass-based though it had to be divided by a factor of 4 to account for the change in mass estimates from the Sunderland Model.  The relationship for the mechanical brake was an inverse mass-to-cost relationship where the mass was derived by cost by a division of 10 :cite:`WindPACT`.  This was adapted to a multiplier of 10 for the mass-cost model.  The generator cost model is as described in :cite:`Sunderland1993` but the costs were updated to the mass-cost relationship of $65/kg as described in :cite:`Malcolm2002`.  Mass-cost models for rest of the nacelle components could not be made since there is either a lack of data on individual masses and/or costs for such components.  All costs are escalated as described in the NREL Cost and Scaling Model.

.. _lssCost:

.. figure:: /images/turbine_costsse/lssCost.*
    :width: 6.5in


    LSS mass-cost relationship based on NREL Cost and Scaling Model.

.. _gearboxCost:

.. figure:: /images/turbine_costsse/gearboxCost.*
    :width: 6.5in

    Gearbox mass-cost relationship based on NREL Cost and Scaling Model.

.. _mainframeCost:

.. figure:: /images/turbine_costsse/mainframeCost.*
    :width: 6.5in

    Mainframe mass-cost relationship based on NREL Cost and Scaling Model.

.. _yawCost:

.. figure:: /images/turbine_costsse/yawCost.*
    :width: 6.5in

    Yaw system mass-cost relationship based on NREL Cost and Scaling Model.

The mass-cost models for the components above were built using the equations as presented in the above figures.

Tower:

The NREL tower mass-cost model is identical to the NREL Cost and Scaling Model tower cost model since the model was already based on a mass-to-cost relationship.

Foundation:

The new NREL foundation mass-cost model is based on the new NREL Offshore Balance of Station Model :cite:`Maples2013`.  While the model software can be used directly to calculate foundation costs for a variety of offshore configurations, it also calculates the mass of those foundations.  It desirable for a number of analyses to determine the monopile mass directly via an engineering-analysis model.  Thus, this model extracts the foundation cost model (as described in :cite:`Maples2013`) so that it can calculate the cost of a monopile foundation directly from the supplied mass of the monopile and transition pieces.  Note that this model is only valid for a monopile type of foundation.  If this model is used in conjunction with the NREL Offshore Balance of Station Model, care must be taken not to double count the foundation cost.



.. only:: html

    :bib:`Bibliography`

.. bibliography:: references.bib
    :style: unsrt
