# WISDEM Changelog

## 0.1.0 ([09/30/2014])

[Katherine Dykes](mailto: katherine.dykes@nrel.gov)

- initial release


## 0.1.1 ([11/03/2014])

[Katherine Dykes](mailto: katherine.dykes@nrel.gov)

[NEW]:

- added integration with JacketSE to do 5 MW turbine and lcoe analysis with jacket substructure offshore; new turbine_jacket assembly created to run turbine jacket structures and corresponding NREL5M_jacket representation in reference turbines and lcoe_se_jacket_assembly for running full LCOE analysis with jacket structure

[CHANGE]:

- new import of rna mass properties and rotor loads module from CommonSE (was in TowerSE) due to common use by TowerSE and JacketSE

[FIX]:

- TowerSE L_reinforced variable changed from float type to array type, updated 5 MW tower structure to match
