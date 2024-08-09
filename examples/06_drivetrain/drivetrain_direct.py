#!/usr/bin/env python3
# Import needed libraries
import numpy as np
import openmdao.api as om

from wisdem.commonse.fileIO import save_data
from wisdem.drivetrainse.drivetrain import DrivetrainSE

opt_flag = True
# ---

# Set input options
opt = {}
opt["WISDEM"] = {}
opt["WISDEM"]["n_dlc"] = 1
opt["WISDEM"]["DriveSE"] = {}
opt["WISDEM"]["DriveSE"]["direct"] = True
opt["WISDEM"]["DriveSE"]["hub"] = {}
opt["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
opt["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5
opt["WISDEM"]["DriveSE"]["gamma_f"] = 1.35
opt["WISDEM"]["DriveSE"]["gamma_m"] = 1.3
opt["WISDEM"]["DriveSE"]["gamma_n"] = 1.0
opt["WISDEM"]["RotorSE"] = {}
opt["WISDEM"]["RotorSE"]["n_pc"] = 20
opt["materials"] = {}
opt["materials"]["n_mat"] = 4
opt["flags"] = {}
opt["flags"]["generator"] = False
# ---

# Initialize OpenMDAO problem
prob = om.Problem(reports=False)
prob.model = DrivetrainSE(modeling_options=opt)
# ---

# If performing optimization, set up the optimizer and problem formulation
if opt_flag:
    # Choose the optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-5

    # Add objective
    prob.model.add_objective("nacelle_mass", scaler=1e-6)

    # Add design variables, in this case the tower diameter and wall thicknesses
    prob.model.add_design_var("L_12", lower=0.1, upper=5.0)
    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0)
    prob.model.add_design_var("hub_diameter", lower=3.0, upper=15.0)
    prob.model.add_design_var("lss_diameter", lower=0.5, upper=6.0)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=5e-1, ref=1e-2)
    prob.model.add_design_var("nose_diameter", lower=0.5, upper=6.0)
    prob.model.add_design_var("nose_wall_thickness", lower=4e-3, upper=5e-1, ref=1e-2)
    prob.model.add_design_var("bedplate_wall_thickness", lower=4e-3, upper=5e-1, ref=1e-2)

    # Add constraints on the tower design
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)
    prob.model.add_constraint("constr_bedplate_vonmises", upper=1.0)
    prob.model.add_constraint("constr_mb1_defl", upper=1.0)
    prob.model.add_constraint("constr_mb2_defl", upper=1.0)
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)
    prob.model.add_constraint("constr_shaft_angle", upper=1.0)
    prob.model.add_constraint("constr_stator_deflection", upper=1.0)
    prob.model.add_constraint("constr_stator_angle", upper=1.0)
    prob.model.add_constraint("constr_hub_diameter", lower=0.0)
    prob.model.add_constraint("constr_length", lower=0.0)
    prob.model.add_constraint("constr_height", lower=0.0)
    prob.model.add_constraint("constr_access", lower=0.0)
    prob.model.add_constraint("constr_ecc", lower=0.0)
    prob.model.add_constraint("L_lss", lower=0.1)
    prob.model.add_constraint("L_nose", lower=0.1)
    # ---


# Set up the OpenMDAO problem
prob.setup()
# ---

# Set input values
prob.set_val("machine_rating", 15.0, units="MW")
prob["upwind"] = True
prob["n_blades"] = 3
prob["rotor_diameter"] = 240.0
prob["D_top"] = 6.5
prob["minimum_rpm"] = 5.0
prob["rated_rpm"] = 7.56
prob["rated_torque"] = 19947034.78543754

# Loading from rotor
prob["F_aero_hub"] = np.array([2517580.0, -27669.0, 3204.0]).reshape((3, 1))
prob["M_aero_hub"] = np.array([21030561.0, 7414045.0, 1450946.0]).reshape((3, 1))
# prob["blades_mass"] = 0.
# ---

# Blade properties and hub design options
prob["blades_cm"] = 2.46175
prob["blade_mass"] = 65252.0
prob["blades_mass"] = 3 * prob["blade_mass"]
prob["blades_I"] = np.r_[3.48453857e+08, 1.74226928e+08, 1.74226928e+08, np.zeros(3)]
prob["pitch_system.BRFM"] = 26648449.0
prob["pitch_system_scaling_factor"] = 0.75
prob["blade_root_diameter"] = 5.2
prob["flange_t2shell_t"] = 6.0
prob["flange_OD2hub_D"] = 0.6
prob["flange_ID2flange_OD"] = 0.8
prob["hub_in2out_circ"] = 1.2
prob["hub_stress_concentration"] = 3.0
prob["n_front_brackets"] = 5
prob["n_rear_brackets"] = 5
prob["clearance_hub_spinner"] = 0.5
prob["spin_hole_incr"] = 1.2
prob["spinner_gust_ws"] = 70.0
prob["hub_diameter"] = 7.94
# ---

# Drivetrain configuration and sizing inputs
prob["bear1.bearing_type"] = "CARB"
prob["bear2.bearing_type"] = "SRB"
prob["L_12"] = 1.2
prob["L_h1"] = 1.0
prob["L_generator"] = 2.15  # 2.75
prob["overhang"] = 12.0313
prob["drive_height"] = 5.614
prob["tilt"] = 6.0
prob["access_diameter"] = 2.0
prob["shaft_deflection_allowable"] = 1e-4
prob["shaft_angle_allowable"] = 1e-3
prob["stator_deflection_allowable"] = 1e-4
prob["stator_angle_allowable"] = 1e-3

myones = np.ones(2)
prob["lss_diameter"] = 3.0 * myones
prob["nose_diameter"] = 2.2 * myones
prob["lss_wall_thickness"] = 0.1 * myones
prob["nose_wall_thickness"] = 0.1 * myones
prob["bedplate_wall_thickness"] = 0.05 * np.ones(4)
prob["bear1.D_shaft"] = 2.2
prob["bear2.D_shaft"] = 2.2
# ---

# Material properties
prob["E_mat"] = np.c_[200e9 * np.ones(3), 205e9 * np.ones(3), 118e9 * np.ones(3), [4.46e10, 1.7e10, 1.67e10]].T
prob["G_mat"] = np.c_[79.3e9 * np.ones(3), 80e9 * np.ones(3), 47.6e9 * np.ones(3), [3.27e9, 3.48e9, 3.5e9]].T
prob["Xt_mat"] = np.c_[450e6 * np.ones(3), 814e6 * np.ones(3), 310e6 * np.ones(3), [6.092e8, 3.81e7, 1.529e7]].T
prob["rho_mat"] = np.r_[7800.0, 7850.0, 7200.0, 1940.0]
prob["Xy_mat"] = np.r_[345e6, 485e6, 265e6, 18.9e6]
prob["wohler_exp_mat"] = 1e1 * np.ones(4)
prob["wohler_A_mat"] = 1e1 * np.ones(4)
prob["unit_cost_mat"] = np.r_[0.7, 0.9, 0.5, 1.9]
prob["lss_material"] = prob["hss_material"] = "steel_drive"
prob["bedplate_material"] = "steel"
prob["hub_material"] = "cast_iron"
prob["spinner_material"] = "glass_uni"
prob["material_names"] = ["steel", "steel_drive", "cast_iron", "glass_uni"]
# ---

# Run the analysis or optimization
if opt_flag:
    prob.model.approx_totals()
    prob.run_driver()
else:
    prob.run_model()
save_data("drivetrain_example", prob)
# ---

# Display all inputs and outputs
#prob.model.list_inputs(units=True)
#prob.model.list_outputs(units=True)

# Print out the objective, design variables and constraints
print("nacelle_mass:", prob["nacelle_mass"])
print("")
print("L_h1:", prob["L_h1"])
print("L_12:", prob["L_12"])
print("L_lss:", prob["L_lss"])
print("L_nose:", prob["L_nose"])
print("L_generator:", prob["L_generator"])
print("L_bedplate:", prob["L_bedplate"])
print("H_bedplate:", prob["H_bedplate"])
print("hub_diameter:", prob["hub_diameter"])
print("lss_diameter:", prob["lss_diameter"])
print("lss_wall_thickness:", prob["lss_wall_thickness"])
print("nose_diameter:", prob["nose_diameter"])
print("nose_wall_thickness:", prob["nose_wall_thickness"])
print("bedplate_wall_thickness:", prob["bedplate_wall_thickness"])
print("")
print("constr_lss_vonmises:", prob["constr_lss_vonmises"].flatten())
print("constr_bedplate_vonmises:", prob["constr_bedplate_vonmises"].flatten())
print("constr_mb1_defl:", prob["constr_mb1_defl"])
print("constr_mb2_defl:", prob["constr_mb2_defl"])
print("constr_shaft_deflection:", prob["constr_shaft_deflection"])
print("constr_shaft_angle:", prob["constr_shaft_angle"])
print("constr_stator_deflection:", prob["constr_stator_deflection"])
print("constr_stator_angle:", prob["constr_stator_angle"])
print("constr_hub_diameter:", prob["constr_hub_diameter"])
print("constr_length:", prob["constr_length"])
print("constr_height:", prob["constr_height"])
print("constr_access:", prob["constr_access"])
print("constr_ecc:", prob["constr_ecc"])
# ---
