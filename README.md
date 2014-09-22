WISDEM is a set of models for assessing overall wind plant cost of energy (coe).  The models use wind turbine and plant cost and energy production as well as financial models to estimate coe and other wind plant system attributes.  It is built in OpenMDAO and uses several sub-models that are also designed as OpenMDAO plugin-ins.  These sub-models can be used independently but they are also automatically installed when WISDEM is installed.

Author: [K. Dykes](mailto:katherine.dykes@nrel.gov) and [S. Andrew Ning](mailto:simeon.ning@nrel.gov)

## Detailed Documentation

For detailed documentation see <http://wisdem.github.io/WISDEM/>

## Prerequisites

General: NumPy, SciPy, OpenMDAO
Wind Plant Framework: FUSED-Wind
Sub-Models: CommonSE, AeroelasticSE, RotorSE, DriveSE, DriveWPACT, TowerSE, JacketSE, TurbineSE, Turbine_CostsSE, Plant_CostsSE, Plant_EnergySE, Plant_FinanceSE

## Installation

Install Turbine_CostsSE within an activated OpenMDAO environment

	$ plugin install

It is not recommended to install the software outside of OpenMDAO.

## Run Unit Tests

To check if installation was successful try to import the module

	$ python
	> import lcoe.lcoe_csm_assembly
	> import lcoe.lcoe_se_csm_assembly

You may also run the unit tests.

	$ python src/test/test_WISDEM.py

