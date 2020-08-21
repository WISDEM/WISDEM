# History
---------------------
## 2.1.1 - (2020-06-15)
Mostly some minor bug fixes and a function for generating rotor performance files using AeroDyn.

Major Updates
- generate_rotperf_FAST is included in the functionalities in turbine.py now. This makes it possible to call OpenFAST and use aerodyn or beamdyn (pending) to generate the - Cp, Ct, Cq tables. Slow, but useful for verification.
- Include GBEff and GenEff appropriately in turbine.rated_torque and controller.VS_Rgn2K calculations

Minor Updates
- min_pitch was not appropriately accounted for if user-defined
- print size of arrays in Cp_Ct_Cq.txt files
- minor typos and commenting cleanup
---------------------
## 2.1.0 - (2020-04-21)
Mostly major updates to the post processing scripts. Some of these may be considered API changes, but we'll treat them as feature adds for now.

A brief overview of major changes:
- `load_output` is renamed to `load_fast_out`
- `load_fast_out` now writes out a list of dictionaries containing openfast data, where each list item corresponds to an OpenFAST output case.
- All plotting functions were moved to a class `FAST_plots`
- `plot_fast_out` only receives the output from `load_fast_out` for the OpenFAST data to plot
- `plot_spectral` is included for a number of frequency-domain based plotting capabilities
- `trim_fast_out` is modified to only modify data passed in the new dictionary-based structure
- some verbosity flags have been included
---------------------
## 2.0.0 - (2020-03-04)
Admittedly poor versioning since the last release. Lots of updates...

### API Changes
* Re-org of some functionalities
    - turbine.load_from_text is now turbine.utilities 
    - utilities.write_param_file is now utilities.write_DISCON
* Include flap controller tuning methods and related inputs to DISCON.IN
* Remove unnecessary control inputs in DISCON.IN (`z_pitch_*`)

### Other changes
* Updates to floating filtering methods
* Updates to floating controller tuning methods to be more mathematically sound
* Generic flap tuning - employ reading of AeroDyn15 files with multiple distributed aerodynamic control inputs
* Test case updates and bug fixes
* Example updates to showcase all functionalities
* Updates to OpenFAST output processing and plotting scripts
* All related improvements and updates ROSCO controller itself
---------------------
## 1.0.1 - (2020-01-29)
* Major bug fixes in second order low-pass filters
* Minor bug fixes for pitch saturation and filter mode settings
* Minor updates in tuning scripts
---------------------
## 1.0.0 - (2020-01-22)
* Version 1.0 release - initial transition from DRC-Fortran with major updates and API changes


