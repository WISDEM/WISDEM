The Wind-Plant Integrated System Design and Engineering Model (WISDEM) is a set of models for assessing overall wind plant cost of energy (coe).  The models use wind turbine and plant cost and energy production as well as financial models to estimate coe and other wind plant system attributes.  It is built in OpenMDAO and uses several sub-models that are also designed as OpenMDAO plugin-ins.  These sub-models can be used independently but they are required to use the overall WISDEM capability.  Please install all of the pre-requisites prior to installing WISDEM.  For additional information about the NWTC effort in systems engineering that supports WISDEM development, please visit the official [NREL systems engineering for wind energy website](http://www.nrel.gov/wind/systems_engineering/).

Authors: [NREL WISDEM Team](mailto:nrel.wisdem+wisdem@gmail.com)
K. Dykes
S. A. Ning
P. Graf
G. Scott
Y. Guo
R. King
T. Parsons
R. Damiani
P. Fleming

## Version

This software is a beta version 0.1.0.

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/WISDEM/>

## Prerequisites

General: NumPy, SciPy, OpenMDAO

Wind Plant Framework: FUSED-Wind (Framework for Unified Systems Engineering and Design of Wind Plants)

Sub-Models: CommonSE, AeroelasticSE, RotorSE, DriveSE, DriveWPACT, TowerSE, JacketSE, Turbine_CostsSE, Plant_CostsSE, Plant_EnergySE, Plant_FinanceSE

## Installation

Install WISDEM within an activated OpenMDAO environment

	$ plugin install

It is not recommended to install the software outside of OpenMDAO.

## Run Unit Tests

To check if installation was successful try to import the module

	$ python
	> import wisdem.lcoe.lcoe_csm_assembly
	> import wisdem.lcoe.lcoe_se_csm_assembly
	> import wisdem.turbinese.turbine

You may also run the unit tests.

	$ python src/test/test_WISDEM.py
	$ python src/test/test_turbine_gradients.py

For software issues please use <https://github.com/WISDEM/WISDEM/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).