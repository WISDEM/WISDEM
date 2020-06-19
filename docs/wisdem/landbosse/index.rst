*********
LandBOSSE
*********

Intro
#####

Capital costs associated with building land-based wind power plants
*******************************************************************

Capital costs associated with constructing a land-based wind power plant include turbine capital costs, transportation cost, and balance-fo-system cost. They are outline in Table 1 below.

Table 1: The capital costs associated with building a land-based wind power plant.

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

LandBOSSE Inputs
****************

The ``LandBOSSE`` group takes input from a mix of continuous floating point values, discrete integers, and discrete dataframes. The dataframes are read from a ``.xlsx`` spreadsheet which contains the lookup tables with information about cranes, labor rates, and many more data needed to calculate balance-of-system costs.

Theory
######

Modules
*******

Costs calculated in LandBOSSE are in eight modules. Four modules (erection, foundation, site preparation, and collection) are process based and calcualte costs based on models of physical processes that happen to accomplish the scope of work defined for each module. The other four modules (management, grid connection, substation, and development) are based on curve fits from empirical data. The operations modeled by each model are summarized in Table XX below. More details are in the technical report at https://www.nrel.gov/docs/fy19osti/72201.pdf.

Table XX: (Key, et al)

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