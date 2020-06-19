***********************
LandBOSSE Documentation
***********************

What is LandBOSSE?
##################

Capital costs to install a land-based wind power plant include the following

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

The Land-based Balance-of-System Systems Engineering (LandBOSSE) model is a systems engineering tool that estimates the balance-of-system (BOS) costs associated with installing utility scale land-based wind plants (10, 1.5 MW turbines or larger). The methods used to develop this model and a detailed discussion of its inputs and outputs are in the following report.

Eberle, Annika, Owen Roberts, Alicia Key, Parangat Bhaskar, and Katherine Dykes. 2019. NREL’s Balance-of-System Cost Model for Land-Based Wind. Golden, CO: National Renewable Energy Laboratory. NREL/TP-6A20-72201. https://www.nrel.gov/docs/fy19osti/72201.pdf.

How does LandBOSSE operate in WISDEM?
#####################################

LandBOSSE + OpenMDAO Connection
*******************************

In WISDEM, LandBOSSE is presented as an OpenMDAO ``Group`` called ``LandBOSSE``. This group wraps an ``ExplicitComponent`` called ``LandBOSSE_API``. When using LandBOSSE in assemblies, LandBOSSE should be accessed via the ``LandBOSSE`` group.

LandBOSSE Inputs
****************

The ``LandBOSSE`` group takes input from a mix of continuous floating point values, discrete integers, and discrete dataframes. The dataframes are read from a ``.xlsx`` spreadsheet which contains the lookup tables with information about cranes, labor rates, and many more data needed to calculate balance-of-system costs.
