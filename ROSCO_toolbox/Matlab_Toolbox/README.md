# Simulink Implementation of ROSCO
A simulink version of ROSCO has been created for rapid prototyping of new control ideas. The results do not exactly match results from the .DLL version of ROSCO due to the way that Simulink handles initial conditions. These differences change the wind speed estimate slightly and propogate to differences in the torque control and pitch saturation. The following modules in ROSCO have been implemented in Simulink:
  - TSR tracking torque control
  - PI gain-scheduled pitch control
  - Setpoint smoothing control
  - Extended Kalman Filter wind speed estimator
  - Pitch Saturation
  - Floating feedback control
  
The modules not currently implemented include:
  - k\omega^2 torque control
  - Individual pitch control
  - Shutdown control
  - Flap control
  
`runFAST.m` can be used to load the ROSCO parameters from a .IN file, using `load_ROSCO_params.m` and a more detailed version can be found in the `matlab-toolbox` [repository](https://github.com/dzalkind/matlab-toolbox/tree/master/Simulations).
  
