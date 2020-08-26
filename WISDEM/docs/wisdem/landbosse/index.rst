*********
LandBOSSE
*********

Intro
#####

Capital costs associated with building land-based wind power plants
*******************************************************************

Capital costs associated with constructing a land-based wind power plant include turbine capital costs, transportation cost, and balance-fo-system cost. They are outlined in the table below:

+----------------------------+-------------------------------------------------------------------+
| Capital Cost               | Description                                                       |
+============================+===================================================================+
| Turbine capital cost (TCC) | The cost to purchase the turbine components                       |
+----------------------------+-------------------------------------------------------------------+
| Transportation cost        | The cost to transport turbine components to the construction site |
+----------------------------+-------------------------------------------------------------------+
| Balance of system (BOS)    | - Erection cost of towers, nacelles, and rotors with cranes       |
|                            | - Foundation construction cost                                    |
|                            | - Collection system installation cost                             |
|                            | - Site preparation cost                                           |
|                            | - Management costs incurred during construction                   |
|                            | - Development costs incurred before construction                  |
|                            | - Substation cost                                                 |
|                            | - Grid connection cost                                            |
+----------------------------+-------------------------------------------------------------------+

What capital costs does LandBOSSE model?
****************************************

The Land-based Balance-of-System Systems Engineering (LandBOSSE) model is a systems engineering tool that estimates the balance-of-system (BOS) costs associated with installing utility scale land-based wind plants (10, 1.5 MW turbines or larger). The methods used to develop this model and a detailed discussion of its inputs and outputs are in the following report.

Eberle, Annika, Owen Roberts, Alicia Key, Parangat Bhaskar, and Katherine Dykes. 2019. NREL’s Balance-of-System Cost Model for Land-Based Wind. Golden, CO: National Renewable Energy Laboratory. NREL/TP-6A20-72201. https://www.nrel.gov/docs/fy19osti/72201.pdf.

Documentation
#############

LandBOSSE and OpenMDAO
**********************

In WISDEM, LandBOSSE is presented as an OpenMDAO ``Group`` called ``LandBOSSE``. This group wraps an ``ExplicitComponent`` called ``LandBOSSE_API``. When using LandBOSSE in assemblies, LandBOSSE should be accessed via the ``LandBOSSE`` group.

LandBOSSE Inputs and Outputs
****************************

LandBOSSE models the construction of an entire wind plant, which itself is a project with a numerous operations. The number and diversity of inputs and outputs to and from the ``LandBOSSE`` component reflects this diversity of construction operations. Text and numeric data, both in single-value and tabular form, are the form of LandBOSSSE inputs and outputs. A full listing of every input and output is beyond the scope of this document; however, here is an overview of types of inputs and outputs and what they contain.

Inputs are grouped into three categories:

+------------------------------------------+------------------------------------------------------------------+
| Category                                 | Examples                                                         |
+==========================================+==================================================================+
| Continuous floating point numeric values | ``blade_mass``, ``tower_mass``, ``crane_breakdown_fraction``     |
+------------------------------------------+------------------------------------------------------------------+
| Discrete integer numeric values          | ``rate_of_deliveries``, ``number_of_blades``, ``num_turbines``   |
+------------------------------------------+------------------------------------------------------------------+
| Discrete Pandas dataframes               | ``crane_specs``, ``components``, ``weather_window``,             |
+------------------------------------------+------------------------------------------------------------------+

Similarly, outputs are grouped into two categories:

+------------------------------------------+-----------------------------------------------------------------------+
| Category                                 | Examples                                                              |
+==========================================+=======================================================================+
| Continuous numeric data                  | ``bos_capex_kW``, ``total_capex_kW``, ``installation_time_months``    |
+------------------------------------------+-----------------------------------------------------------------------+
| Discrete Pandas dataframes               | ``landbosse_costs_by_module_type_operation``, ``erection_components`` |
+------------------------------------------+-----------------------------------------------------------------------+

``.xlsx`` spreadsheet
*********************

Ultimately, all of these values are inputs into the ``LandBOSSE`` group. The dataframe inputs are read from ``.xlsx`` spreadsheet with the following worksheets: ``crane_specs``, ``cable_specs``, ``equip``, ``components``, ``development``, ``crew_price``, ``crew``, ``equip_price``, ``material_price``, ``rsmeans``, ``site_facility_building_area``, and ``weather_window``. These sheets are used as lookup tables for capabilities and costs of equipment and crews utilized in the BOS operations.

A file with default data is in the ``library/landbosse/ge15_public.xlsx`` file found in the WISDEM repository.

Tutorial
########

Common use case: A wind plant made of an optimized turbine
**********************************************************

A common use case of LandBOSSE within WISDEM is to model the costs of building an entire wind power plant from turbine components generated by RotorSE, DriveSE, and TowerSE. For this use case, the following table lists the needed inputs for calculating BOS costs for a turbine created by WISDEM:

+----------------+-------+-------------------------+
| Input          | Units | Description             |
+================+=======+=========================+
| hub_height     | m     | Hub height of the rotor |
+----------------+-------+-------------------------+
| blade_mass     | kg    | Mass of one blade       |
+----------------+-------+-------------------------+
| nacelle_mass   | kg    | Mass of the nacelle     |
+----------------+-------+-------------------------+
| tower_mass     | kg    | Mass of the tower       |
+----------------+-------+-------------------------+
| machine_rating | kW    | Rating of the turbine   |
+----------------+-------+-------------------------+

Troubleshooting note: During some practical construction operations, large nacelles (such as those found in high turbine ratings) are broken into multiple sections to reduce the size of the crane needed for erection. However, DriveSE models a nacelle all as one piece. If you get an error where LandBOSSE reports that a topping crane could not be found, this may mean the calculated nacelle mass exceeds the capacity of the largest crane listed in the ``crane_specs`` tab of the spreadsheet.

An example assembly that integrates calculations from RotorSE, DriveSE, TowerSE, and LandBOSSE is in the ``wisdem/assemblies/land_based.py`` file.

Theory
######

Modules
*******

Costs calculated in LandBOSSE are in eight modules. Four modules (erection, foundation, site preparation, and collection) are process based and calcualte costs based on models of physical processes that happen to accomplish the scope of work defined for each module. The other four modules (management, grid connection, substation, and development) are based on curve fits from empirical data. The operations modeled by each model are summarized in Table XX below. More details are in the technical report at https://www.nrel.gov/docs/fy19osti/72201.pdf.

Table XX: [TODO: Update citation when paper is published]

+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Module           | Type      | Summary of costs included                                                                                                                                                                                                                                                                                 |
+==================+===========+===========================================================================================================================================================================================================================================================================================================+
| Foundation       | process   | Operations specific to foundation construction, including excavating the base, installing rebar and a bolt cage, pouring concrete, constructing the pedestal, and backfilling the foundation.                                                                                                             |
+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Erection         | process   | Operations specific to erecting the tower and turbine, including removal of components from delivery trucks by offload cranes and erection of the lower tower sections onto the foundation using a base crane and the upper pieces of the tower and the components of the nacelle using a topping crane.  |
+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Development      | curve fit | Evaluation of the wind resource, acquisition of land, completion of environmental permitting, assessment of distribution costs, and marketing of power to be generated.                                                                                                                                   |
+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Management       | curve fit | Insurance, construction permits, site-specific engineering, construction of facilities for site access and construction staging, site management, and bonding, markup, and contingency.                                                                                                                   |
+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Collection       | process   | Operations specific to the construction of a collection system, which consists of cabling from the turbines to the substation (does not include power electronics or cabling already included in the turbine capital cost).                                                                               |
+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Grid Connection  | curve fit | Operations specific to grid connection (i.e., transmission and interconnection), including a land survey, clearing and grubbing the area, installation of stormwater and pollution mitigation measures, installation of conductors, and restoration of the rights of way.                                 |
+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Site Preparation | process   | Operations to prepare the wind plant site for other construction operations, including surveying and clearing areas for roads, compacting the soil, and placing rock to allow roads to support the weight of trucks, components, and cranes.                                                              |
+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Substation       | process   | Operations specific to substation construction, including a land survey; installation of stormwater and pollution mitigation measures; construction of dead-end structures, foundations, conductors, transformers, relays, controls, and breakers; and restoration of the rights of way.                  |
+------------------+-----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+