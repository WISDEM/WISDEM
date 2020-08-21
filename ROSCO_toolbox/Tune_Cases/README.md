# Tuning cases
Here are some instructions on how to use the generic tuning capabilities of the ROSCO toolbox. 

## Basic Steps
The basic steps are as follows

1. Fill out the .yaml input file to tune the controller
2. Run `tune_ROSCO.py`, or your own version of it.
3. Copy `DISCON.IN` and `Cp_Ct_Cq.txt` (or similar) to the desired folder for your wind turbine model.
4. Ensure that `PerfFileName` in `DISCON.IN` points to the proper name and location of `Cp_Ct_Cq.*.txt`
5. Ensure that `DLL_InFile` in the OpenFAST ServoDyn input file properly points to `DISCON.IN`.
6. Enjoy the results. Possibly retune your controller. 

## The .yaml File
We use a .yaml file to define the inputs to the generic controlling scripts. This .yaml file defines three different dictionaries with a number of parameters. The parameters that must be defined are listed below

1. path_params:
* `FAST_InputFile` - the name of the `*.fst` file
* `FAST_directory` - the directory for the `*.fst` file 
* `rotor_performance_filename` - the name of the existing or desired rotor performance text file (`Cp_Ct_Cq.txt` by default). If this exists, CCBlade will not be run in `tune_ROSCO.py`. If it does not exist, you CCBlade will be run. The inputs to `turbine.py` can also be changed to force CCBlade to run or not.

2. turbine_params:
* `rotor_inertia` - Rotor inertia (kg m^2). This is available in an Elastodyn .sum file. Run a turbine model for a few seconds without any controller if you need to. 
* `rated_rotor_speed` - Rated rotor speed (rad/s).
* `v_min` - Cut-in wind speed (m/s). 
* `v_rated` - Rated wind speed (m/s).
* `v_max` - Cut-out wind speed (m/s), -- Does not need to be exact (JUST ASSUME FOR NOW)
* `max_pitch_rate` - Maximum blade pitch rate (deg/s)
* `max_torque_rate` - Maximum torque rate (Nm/s)
* `rated_power` - Rated Power (W)
* `bld_edgewise_freq` - Blade edgewise first natural frequency (rad/s)

3. controller_params:
* `LoggingLevel` - Default = 1 - {0: write no debug files, 1: write standard output .dbg-file, 2: write standard output .dbg-file and complete avrSWAP-array .dbg2-file}
* `F_LPFType` - Default = 1 - {1: first-order low-pass filter, 2: second-order low-pass filter}, (rad/s) (currently filters generator speed and pitch *control signals)
* `F_NotchType` - Default = 0 - Notch on the measured generator speed {0: disable, 1: enable} 
* `IPC_ControlMode` - Default = 0 - Turn Individual Pitch Control (IPC) for fatigue load reductions (pitch contribution) {0: off, 1: 1P reductions, 2: 1P+2P *reductions}
* `VS_ControlMode` - Default = 2 -  Generator torque control mode in above rated conditions {0: constant torque, 1: constant power, 2: TSR tracking PI control}
* `PC_ControlMode` - Default = 2 - Blade pitch control mode {0: No pitch, fix to fine pitch, 1: active PI blade pitch control}
* `Y_ControlMode` - Default = 0 - Yaw control mode {0: no yaw control, 1: yaw rate control, 2: yaw-by-IPC}
* `SS_Mode` - Default = 1 - Setpoint Smoother mode {0: no setpoint smoothing, 1: introduce setpoint smoothing}
* `WE_Mode` - Default = 2 - Wind speed estimator mode {0: One-second low pass filtered hub height wind speed, 1: Immersion and Invariance Estimator, 2: Extended Kalman Filter}
* `PS_Mode` - Default = 0 - Pitch saturation mode {0: no pitch saturation, 1: peak shaving, 2: Cp-maximizing pitch saturation, 3: peak shaving and Cp-maximizing pitch saturation}
* `zeta_pc` - Pitch controller desired damping ratio (-)
* `omega_pc` - Pitch controller desired natural frequency (rad/s)
* `zeta_vs` - Torque controller desired damping ratio (-)
* `omega_vs` - Torque controller desired natural frequency (rad/s)

#### Optional Parameters
There are a few parameters that are only needed for specific controller behaviors, or to adjust behaviors to be different than the default:
1. turbine_params:
* `twr_freq` - Tower fore-aft natural frequency (rad/s), only used inf Fl_Mode = 1
* `ptfm_freq` - Platform fore-aft natural frequency (rad/s), only used inf Fl_Mode = 1
2. controller_params
* `zeta_flp` -  Flap controller desired damping ratio, only used if Flp_Mode = 1
* `omega_flp`-  Flap controller desired natural frequency (rad/s), only used if Flp_Mode = 1
* `max_pitch`-  Maximum pitch angle (rad), {default = 90 degrees}
* `min_pitch`-  Minimum pitch angle (rad), {default = 0 degrees}
* `vs_minspd`-  Minimum rotor speed (rad/s), {default = 0 rad/s}
* `ss_cornerfreq`-  First order low-pass filter cornering frequency for setpoint smoother (rad/s)
* `ss_vsgain`-  Torque controller setpoint smoother gain bias percentage [%, <= 1 ], {default = 100%}
* `ss_pcgain`-  Pitch controller setpoint smoother gain bias percentage  [%, <= 1 ], {default = 0.1%}
* `ps_percent`-  Percent peak shaving  [%, <= 1 ], {default = 80%}, only used if PS_Mode = 1 or 3
* `sd_maxpit`-  Maximum blade pitch angle to initiate shutdown (rad), {default = bld pitch at v_max}, only used if SD_Mode = 1
* `sd_cornerfreq`-  Cutoff Frequency for first order low-pass filter for blade pitch angle rad/s, {default = 0.41888 ~ time constant of 15s}, only used if SD_Mode = 1
* `flp_maxpit`-  Maximum (and minimum) flap pitch angle (rad), only used if Flp_Mode = 2

### The controller parameters
The controller flags have some default values for the ROSCO controller implementation (provided in the previous section). This, subsequently, leaves four turbine parameters that the user must decide `zeta_pc`, `omega_pc`, `zeta_vs`, and `omega_vs`. The example .yaml scripts provided offer some insight into what these values might be for different sizes and types of turbines. Generally speaking, we desire slower responses in the the variable speed (vs) torque controller. This equates to `zeta_vs = 1` and `omega_vs = 0.3` for the NREL 5MW wind turbine. These values tend to translate fairly well to larger turbines, but decreasing `omega_vs = 0.3` may be desired. For the pitch controller, the NREL 5MW turbine was tuned such that `zeta_pc = 0.7`, and `omega_pc = 0.6` in the legacy controller. As a high level rule of thumb, increasing the desired damping to be over-damped (`zeta_pc ~ 1.0`) and the natural frequency to be much slower (`omega_pc = 0.2`) seems to produce smoother wind turbine responses in large turbines with highly flexible rotors. 


## Running a tuning case
As mentioned above, a sample tuning script `tune_ROSCO.py` is provided. This script only needs the first line to be modified to point to the desired `*.yaml` file, and will produce the necessary input files for the controller. The curious user should feel free to modify and change this file. A simple addition is to include the use of `sim.py` to run a simple 1DOF simulation case to verify controller tuning - `example_06.py` shows some of this functionally. Other uses include running OpenFAST directly (see `example_08.py`), plotting system outputs (functionality pending), or visualizing results from ccblade rotor performance analysis (Cp surface is plotted currently).