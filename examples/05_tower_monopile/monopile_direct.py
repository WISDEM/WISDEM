# Tower-Monopile analysis
# Optimization by flag
# Two load cases
import os

import numpy as np
import openmdao.api as om

from wisdem.commonse.fileIO import save_data
from wisdem.fixed_bottomse.monopile import MonopileSE

# Set analysis and optimization options and define geometry
plot_flag = False
opt_flag = True

n_control_points = 4
n_materials = 1
n_load_cases = 2
max_diam = 8.0

# Tower initial condition
hubH = 87.6
htrans = 10.0

# Monopile initial condition
pile_depth = 25.0
water_depth = 30.0
h_paramM = np.r_[pile_depth, water_depth, htrans]
d_paramM = 0.9 * max_diam * np.ones(n_control_points)
t_paramM = 0.02 * np.ones(n_control_points)
# ---

# Store analysis options in dictionary
modeling_options = {}
modeling_options["flags"] = {}
modeling_options["materials"] = {}
modeling_options["materials"]["n_mat"] = n_materials
modeling_options["WISDEM"] = {}
modeling_options["WISDEM"]["n_dlc"] = n_load_cases
modeling_options["WISDEM"]["FixedBottomSE"] = {}
modeling_options["WISDEM"]["FixedBottomSE"]["buckling_length"] = 15.0
modeling_options["WISDEM"]["FixedBottomSE"]["buckling_method"] = "dnvgl"
modeling_options["WISDEM"]["FixedBottomSE"]["n_refine"] = 3
modeling_options["flags"]["monopile"] = True
modeling_options["flags"]["tower"] = False

# Monopile foundation
modeling_options["WISDEM"]["FixedBottomSE"]["soil_springs"] = True
modeling_options["WISDEM"]["FixedBottomSE"]["gravity_foundation"] = False

# safety factors
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_f"] = 1.35
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_m"] = 1.3
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_n"] = 1.0
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_b"] = 1.1
modeling_options["WISDEM"]["FixedBottomSE"]["gamma_fatigue"] = 1.35 * 1.3 * 1.0

# Frame3DD options
modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"] = {}
modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["shear"] = True
modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["geom"] = True
modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["tol"] = 1e-7
modeling_options["WISDEM"]["FixedBottomSE"]["frame3dd"]["modal_method"] = 1
modeling_options["WISDEM"]["FixedBottomSE"]["rank_and_file"] = True

modeling_options["WISDEM"]["FixedBottomSE"]["n_height"] = n_control_points
modeling_options["WISDEM"]["FixedBottomSE"]["n_layers"] = 1
modeling_options["WISDEM"]["FixedBottomSE"]["wind"] = "PowerWind"
# ---

# Instantiate OpenMDAO problem and create a model using the FixedBottomSE group
prob = om.Problem(reports=False)
prob.model = MonopileSE(modeling_options=modeling_options)
# ---

# If performing optimization, set up the optimizer and problem formulation
if opt_flag:
    # Choose the optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["maxiter"] = 40

    # Add objective
    # prob.model.add_objective('tower_mass', ref=1e6) # Only tower
    # prob.model.add_objective("structural_mass", ref=1e6)  # Both
    prob.model.add_objective("monopile_mass", ref=1e6)  # Only monopile

    # Add design variables, in this case the tower diameter and wall thicknesses
    prob.model.add_design_var("monopile_outer_diameter_in", lower=3.87, upper=max_diam)
    prob.model.add_design_var("monopile_layer_thickness", lower=4e-3, upper=2e-1, ref=1e-2)

    # Add constraints on the tower design
    prob.model.add_constraint("post.constr_stress", upper=1.0)
    prob.model.add_constraint("post.constr_global_buckling", upper=1.0)
    prob.model.add_constraint("post.constr_shell_buckling", upper=1.0)
    prob.model.add_constraint("constr_d_to_t", lower=80.0, upper=500.0, ref=1e2)
    prob.model.add_constraint("constr_taper", lower=0.2)
    prob.model.add_constraint("slope", upper=1.0)
    prob.model.add_constraint("suctionpile_depth", lower=0.0)
    prob.model.add_constraint("f1", lower=0.13, upper=0.40, ref=0.1)
    # ---

# Set up the OpenMDAO problem
prob.setup()
# ---

# Set geometry and turbine values
prob["water_depth"] = water_depth

prob["transition_piece_mass"] = 100e3

prob["monopile_foundation_height"] = -55.0
prob["tower_foundation_height"] = 10.0
prob["monopile_height"] = h_paramM.sum()
prob["monopile_s"] = np.cumsum(np.r_[0.0, h_paramM]) / h_paramM.sum()
prob["monopile_outer_diameter_in"] = d_paramM
prob["monopile_layer_thickness"] = t_paramM.reshape((1, -1))
prob["outfitting_factor_in"] = 1.07
prob["tower_base_diameter"] = 6.0
prob["monopile_top_diameter"] = 8.0

prob["yaw"] = 0.0

# offshore specific
prob["G_soil"] = 140e6
prob["nu_soil"] = 0.4

# material properties
prob["E_mat"] = 210e9 * np.ones((n_materials, 3))
prob["G_mat"] = 79.3e9 * np.ones((n_materials, 3))
prob["rho_mat"] = [7850.0]
prob["sigma_y_mat"] = [345e6]
prob["sigma_ult_mat"] = 1.12e9 * np.ones((n_materials, 3))
prob["wohler_exp_mat"] = [4.0]
prob["wohler_A_mat"] = [4.0]

# cost rates
prob["unit_cost_mat"] = [2.0]  # USD/kg
prob["labor_cost_rate"] = 100.0 / 60.0  # USD/min
prob["painting_cost_rate"] = 30.0  # USD/m^2

# wind & wave values
prob["wind_reference_height"] = 90.0
prob["z0"] = 0.0
prob["cd_usr"] = -1.0
prob["rho_air"] = 1.225
prob["rho_water"] = 1025.0
prob["mu_air"] = 1.7934e-5
prob["mu_water"] = 1.3351e-3
prob["beta_wind"] = 0.0
prob["Hsig_wave"] = 4.52
prob["Tsig_wave"] = 9.52
if modeling_options["WISDEM"]["FixedBottomSE"]["wind"] == "PowerWind":
    prob["shearExp"] = 0.1
# ---


# two load cases.  TODO: use a case iterator

# --- loading case 1: max Thrust ---
prob["env1.Uref"] = 11.73732
prob["env2.Uref"] = 70.0
prob["turbine_F"] = np.c_[[1.28474420e06, 0.0, -1.05294614e07], [9.30198601e05, 0.0, -1.13508479e07]]
prob["turbine_M"] = np.c_[
    [4009825.86806202, 92078894.58132489, -346781.68192839], [-1704977.30124085, 65817554.0892837, 147301.97023764]
]
# ---------------

# run the analysis or optimization
prob.model.approx_totals()
if opt_flag:
    prob.run_driver()
else:
    prob.run_model()
os.makedirs("outputs", exist_ok=True)
save_data(os.path.join("outputs", "monopile_example"), prob)
# ---


# print results from the analysis or optimization
z = 0.5 * (prob["z_full"][:-1] + prob["z_full"][1:])
print("zs =", prob["z_full"])
print("ds =", prob["outer_diameter_full"])
print("ts =", prob["t_full"])
print("mass (kg) =", prob["monopile_mass"])
print("cg (m) =", prob["monopile_z_cg"])
print("d:t constraint =", prob["constr_d_to_t"])
print("taper ratio constraint =", prob["constr_taper"])
print("\nwind: ", prob["env1.Uref"], prob["env2.Uref"])
print("freq (Hz) =", prob["structural_frequencies"])
print("Fore-aft mode shapes =", prob["fore_aft_modes"])
print("Side-side mode shapes =", prob["side_side_modes"])
print("top_deflection1 (m) =", prob["monopile.top_deflection"])
print("Monopile base forces1 (N) =", prob["monopile.mudline_F"])
print("Monopile base moments1 (Nm) =", prob["monopile.mudline_M"])
print("stress1 =", prob["post.constr_stress"])
print("GL buckling =", prob["post.constr_global_buckling"])
print("Shell buckling =", prob["post.constr_shell_buckling"])

if plot_flag:
    import matplotlib.pyplot as plt

    stress = prob["post.constr_stress"]
    shellBuckle = prob["post.constr_shell_buckling"]
    globalBuckle = prob["post.constr_global_buckling"]

    plt.figure(figsize=(5.0, 3.5))
    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.plot(stress[:, 0], z, label="stress 1")
    plt.plot(stress[:, 1], z, label="stress 2")
    plt.plot(shellBuckle[:, 0], z, label="shell buckling 1")
    plt.plot(shellBuckle[:, 1], z, label="shell buckling 2")
    plt.plot(globalBuckle[:, 0], z, label="global buckling 1")
    plt.plot(globalBuckle[:, 1], z, label="global buckling 2")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    plt.xlabel("utilization")
    plt.ylabel("height along tower (m)")
    plt.tight_layout()
    plt.show()
    # ---
