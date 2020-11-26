# 0 ---------- (marker for docs)
import numpy as np
import openmdao.api as om
from wisdem.nrelcsm.nrel_csm_mass_2015 import nrel_csm_2015
from wisdem.nrelcsm.nrel_csm_orig import aep_csm

# 0 ---------- (marker for docs)

# 1 ---------- (marker for docs)
# OpenMDAO Problem instance
prob = om.Problem()
prob.model = nrel_csm_2015()
prob.setup()

# Initialize AEP calculator from CSM model
aep_instance = aep_csm()
# 1 ---------- (marker for docs)

# 2 ---------- (marker for docs)
# Initialize variables for NREL CSM
prob["turbine_class"] = -1  # Sets blade mass based on user input, not auto-determined
prob["blade_number"] = 3
prob["blade_has_carbon"] = False
prob["max_tip_speed"] = max_tip_speed = 90.0
prob["max_efficiency"] = max_efficiency = 0.9
prob["main_bearing_number"] = 2
prob["crane"] = True
# 2 ---------- (marker for docs)

# 3 ---------- (marker for docs)
# Initialize variables for AEP calculation
opt_tsr = 9.0  # Optimal tip speed ratio
max_Cp = 0.47  # Max (aerodynamic) power coefficient
max_Ct = 0.8  # Max thrust coefficient
max_eff = 0.95  # Drivetrain efficiency
cut_in = 4.0  # m/s
cut_out = 25.0  # m/s
altitude = 0.0  # m (assume sea level)
rho_air = 1.225  # kg/m^3
array_losses = 0.0  # Focusing on single turbine
availability = 1.0  # Assume 100% uptime
soiling_losses = 0.0  # Ignore this for now
turbine_number = 1  # Focus on single turbine
weibull_k = 2.0  # Weibull shape parameter
# 3 ---------- (marker for docs)

# 4 ---------- (marker for docs)
# Set Design of Experiment (DoE) parametric sweep bounds
machine_rating = 1e3 * np.arange(2.0, 10.1, 1.0)  # kW
rotor_diameter = np.arange(60, 201.0, 20.0)  # m
blade_mass_exp = np.arange(2.1, 2.41, 0.1)  # Relationship between blade length and mass
shear_exp = np.arange(0.1, 0.31, 0.1)
wind_speed = np.arange(5.0, 11.1, 1.0)  # m/s
# 4 ---------- (marker for docs)

# 5 ---------- (marker for docs)
# Enumerate DoE through tensor multiplication and then flatten to get vector of all of the runs
[Rating, Diameter, Bladeexp, Shear, WindV] = np.meshgrid(
    machine_rating, rotor_diameter, blade_mass_exp, shear_exp, wind_speed
)

# Shift to flattened arrays to run through each scenario easily
Rating = Rating.flatten()
Diameter = Diameter.flatten()
Bladeexp = Bladeexp.flatten()
Shear = Shear.flatten()
WindV = WindV.flatten()

# Initialize output containers
tcc = np.zeros(Rating.shape)
aep = np.zeros(Rating.shape)
# 5 ---------- (marker for docs)

# 6 ---------- (marker for docs)
# Calculation loop
npts = Rating.size
print("Running, ", npts, " points in the parametric study")
for k in range(npts):

    # Populate remaining NREL CSM inputs for this iteration
    prob["machine_rating"] = Rating[k]
    prob["rotor_diameter"] = Diameter[k]
    prob["blade_user_exp"] = Bladeexp[k]
    prob["hub_height"] = hub_height = 0.5 * Diameter[k] + 30.0

    # Compute Turbine capital cost using the NREL CSM (2015) and store the result
    prob.run_model()
    tcc[k] = float(prob["turbine_cost_kW"])

    # Compute AEP using original CSM function and store result
    aep_instance.compute(
        Rating[k],
        max_tip_speed,
        Diameter[k],
        max_Cp,
        opt_tsr,
        cut_in,
        cut_out,
        hub_height,
        altitude,
        rho_air,
        max_efficiency,
        max_Ct,
        soiling_losses,
        array_losses,
        availability,
        turbine_number,
        Shear[k],
        WindV[k],
        weibull_k,
    )
    aep[k] = aep_instance.aep.net_aep

    # Progress update to screen every 1000 iterations
    if np.mod(k, 1000) == 0:
        print("...Completed iteration, ", k)
# 6 ---------- (marker for docs)

# 7 ---------- (marker for docs)
# Write outputs to csv file for later post-processing
alldata = np.c_[Rating, Diameter, Bladeexp, Shear, WindV, tcc, aep]
header = "Rating [kW],Rotor Diam [m],Blade Mass Exp,Shear Exp,Wind Vel [m/s],TCC [USD/kW],AEP [kWh/yr]"
np.savetxt("parametric_scaling.csv", alldata, delimiter=",", header=header)
# 7 ---------- (marker for docs)
