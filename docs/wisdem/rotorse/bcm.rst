.. _bcm:

-------------------------------
Blade Cost Model
-------------------------------

While WISDEM estimates the costs of some turbine components via semi-empirical relations tuned on historical data that get updated every few years, the blade cost model implemented in WISDEM adopts a bottom up approach and estimates the total costs of a blade as the sum of variable and fixed costs. The former are made of materials and direct labor costs, while the latter are the costs from overhead, building, tooling, equipment, maintenance, and capital. The model simulates the whole manufacturing process and it has been tuned to estimate the costs of blades in the range of 30 to 100 meters in length. 

The blade cost model is described in detail in the NREL technical report `https://www.nrel.gov/docs/fy19osti/73585.pdf <https://www.nrel.gov/docs/fy19osti/73585.pdf>`_.

Users should be made aware that the absolute values of the costs estimated by the cost models implemented in WISDEM certainly suffer a wide band of uncertainty, but the hope is that the models are able to capture the relative trends sufficiently well.

The blade cost model is implemented in the file wisdem/rotorse/rotor_cost.py and it is called in the file wisdem/rotorse/rotor_elasticity.py