.. _theory:

Theory
------

The theory for the models in this software are based directly on the work described in references :cite:`Fingersh2006`, :cite:`WindPACT`, :cite:`Sunderland1993`, :cite:`Malcolm2002`, and :cite:`Maples2010`.  This section provides an overview of the development of the physical and cost models for the major hub and drivetrain components. 

The NREL Cost and Scaling Model :cite:`Fingersh2006` provides a simple cost and sizing tool to estimate wind turbine component masses and costs based on a small number of input parameters such as rotor diameter, hub height and rated power.  The model was developed over several results following the Wind Partnerships for Advanced Component Technology (WindPACT) work that occurred between roughly 2002 to 2005 :cite:`WindPACT`.  The original form of the cost model was based on an earlier model from 1993 out of the University of Sunderland (the Sunderland Model) :cite:`Sunderland1993`.  The Sunderland Model created a set of wind turbine models to estimate the mass and cost of all major wind turbine components including: blade, hub system [hub, pitch system, and nose cone], nacelle [low speed shaft, main bearings, gearbox, high speed shaft/mechanical brake, generator, variable speed electronics, electrical cabling, mainframe [bedplate, platforms and railings, base hardware, and crane], HVAC system, controls, and nacelle cover], and tower.  The Sunderland model was based on a set of semi-empirical models for each component which estimated design loads at the rotor, propagated these loads through the entire system, and used the loads to estimate the size of each component calibrated to data on actual turbines in the field during that time.  Cost estimates were then made on a per weight basis using a multiplier again based on field data or industry sources.  

To arrive at the NREL Cost and Scaling Model, the WindPACT studies began in many cases with the Sunderland model and updated the results with new coefficients or, in some cases, with entirely new cost equations based on curve fits of key design parameters (rotor diameter, etc) to the results of detailed design studies :cite:`Malcolm2002`.  In addition, the WindPACT work established estimates of costs associated with balance of station and operations and maintenance for a fictitious wind plant in North Dakota which led to an overall cost of energy model for a wind plant.  The key cost of energy equation for a wind plant is given in the NREL Cost and Scaling Model :cite:`Fingersh2006` as:

.. math:: COE = (FCR*(BOS+TCC))/AEP + (LLC + LRC + (1-tr)*OM)/AEP

where :math:`COE` in this equation is a simple estimate of a wind plant cost of energy, :math:`FCR` is the fixed charge rate for the project, :math:`BOS` are the total balance of station costs for the project, :math:`TCC` are the total turbine capital costs for the project, :math:`AEP` is the annual energy production for the project, :math:`LLC` are the annual land-lease costs, :math:`LRC` is the levelized replacement cost for major part replacement, :math:`tr` is the tax rate, and :math:`OM` are the annual operations and maintenance costs which are tax deductible. 

While the NREL Cost and Scaling Model improved the overall cost estimation for larger turbines on the order of 1 MW+, it abstracted away from the engineering analysis foundations of the original Sunderland model.  This is depicted in the below figure where it can be seen that the engineering-analysis has been replaced by a series of curve fits which relate a small number of design parameters to mass and cost estimates for major wind turbine components as well as over all plant energy production and costs. 

.. _NRELCSM:

.. figure:: /images/nrelcsm/NRELCSM.*
    :width: 5.5in

    NREL Cost and Scaling Model Key Input-Output Relationships.

The resulting NREL Cost and Scaling Model allows for a variety of interesting analyses including scaling of conventional technology from under a MW to 5 MW+, assessing impact of trends in input factors for materials and labor on wind plant cost of energy, etc.  However, it does not preserve the underlying engineering relationships of the original Sunderland model and thus loses some fidelity of assessing how design changes may impact system costs.  

This model directly implements the full wind plant NREL Cost and Scaling Model as described in :cite:`Fingersh2006` with modifications for the drivetrain as described in :cite:`Maples2010`.


.. only:: html

    :bib:`Bibliography`

.. bibliography:: references.bib
    :style: unsrt
