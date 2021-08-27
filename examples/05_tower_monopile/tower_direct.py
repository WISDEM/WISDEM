#!/usr/bin/env python3

import os

# Tower analysis
# Optimization by flag
# Two load cases
import numpy as np
import openmdao.api as om
from wisdem.towerse.tower import TowerSE
from wisdem.commonse.fileIO import save_data

# Set analysis and optimization options and define geometry
plot_flag = False
opt_flag = True

n_control_points = 3
n_materials = 1
n_load_cases = 2

h_param = np.diff(np.linspace(0.0, 87.6, n_control_points))
d_param = np.linspace(8.0, 3.87, n_control_points)
t_param = np.linspace(0.08, 0.02, n_control_points)
max_diam = 9.0
# ---

# Store analysis options in dictionary
modeling_options = {}
modeling_options["flags"] = {}
modeling_options["materials"] = {}
modeling_options["materials"]["n_mat"] = n_materials
modeling_options["General"] = {}
modeling_options["WISDEM"] = {}
modeling_options["WISDEM"]["n_dlc"] = n_load_cases
modeling_options["WISDEM"]["TowerSE"] = {}
modeling_options["WISDEM"]["TowerSE"]["buckling_method"] = "eurocode"
modeling_options["WISDEM"]["TowerSE"]["buckling_length"] = 15.0
modeling_options["WISDEM"]["TowerSE"]["n_refine"] = 3

# safety factors
modeling_options["WISDEM"]["TowerSE"]["gamma_f"] = 1.35
modeling_options["WISDEM"]["TowerSE"]["gamma_m"] = 1.3
modeling_options["WISDEM"]["TowerSE"]["gamma_n"] = 1.0
modeling_options["WISDEM"]["TowerSE"]["gamma_b"] = 1.1
modeling_options["WISDEM"]["TowerSE"]["gamma_fatigue"] = 1.35 * 1.3 * 1.0

# Frame3DD options
modeling_options["WISDEM"]["TowerSE"]["frame3dd"] = {}
modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["shear"] = True
modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["geom"] = True
modeling_options["WISDEM"]["TowerSE"]["frame3dd"]["tol"] = 1e-9

modeling_options["WISDEM"]["TowerSE"]["n_height"] = n_control_points
modeling_options["WISDEM"]["TowerSE"]["n_layers"] = 1
modeling_options["WISDEM"]["TowerSE"]["wind"] = "PowerWind"
# ---

# Instantiate OpenMDAO problem and create a model using the TowerSE group
prob = om.Problem()
prob.model = TowerSE(modeling_options=modeling_options)
# ---

# If performing optimization, set up the optimizer and problem formulation
if opt_flag:
    # Choose the optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    # Add objective
    prob.model.add_objective("tower_mass", scaler=1e-6)

    # Add design variables, in this case the tower diameter and wall thicknesses
    prob.model.add_design_var("tower_outer_diameter_in", lower=3.87, upper=max_diam)
    prob.model.add_design_var("tower_layer_thickness", lower=4e-3, upper=2e-1)

    # Add constraints on the tower design
    prob.model.add_constraint("post.constr_stress", upper=1.0)
    prob.model.add_constraint("post.constr_global_buckling", upper=1.0)
    prob.model.add_constraint("post.constr_shell_buckling", upper=1.0)
    prob.model.add_constraint("constr_d_to_t", lower=80.0, upper=500.0)
    prob.model.add_constraint("constr_taper", lower=0.2)
    prob.model.add_constraint("slope", upper=1.0)
    prob.model.add_constraint("tower.f1", lower=0.13, upper=0.40)
    # ---


# Set up the OpenMDAO problem
prob.setup()
# ---

# Set geometry and turbine values
prob["hub_height"] = prob["tower_height"] = h_param.sum()
prob["foundation_height"] = 0.0
prob["tower_s"] = np.cumsum(np.r_[0.0, h_param]) / h_param.sum()
prob["tower_outer_diameter_in"] = d_param
prob["tower_layer_thickness"] = t_param.reshape((1, -1))
prob["outfitting_factor_in"] = 1.07
prob["yaw"] = 0.0

# material properties
prob["E_mat"] = 210e9 * np.ones((n_materials, 3))
prob["G_mat"] = 80.8e9 * np.ones((n_materials, 3))
prob["rho_mat"] = [8500.0]
prob["sigma_y_mat"] = [450e6]
prob["sigma_ult_mat"] = 600e6 * np.ones((n_materials, 3))
prob["wohler_exp_mat"] = [10.0]
prob["wohler_A_mat"] = [10.0]

# extra mass from RNA
prob["rna_mass"] = np.array([285598.8])
mIxx = 1.14930678e08
mIyy = 2.20354030e07
mIzz = 1.87597425e07
mIxy = 0.0
mIxz = 5.03710467e05
mIyz = 0.0
prob["rna_I"] = np.array([mIxx, mIyy, mIzz, mIxy, mIxz, mIyz])
prob["rna_cg"] = np.array([-1.13197635, 0.0, 0.50875268])
# ---

# cost rates
prob["unit_cost_mat"] = [2.0]  # USD/kg
prob["labor_cost_rate"] = 100.0 / 60.0  # USD/min
prob["painting_cost_rate"] = 30.0  # USD/m^2

# wind & wave values
prob["wind_reference_height"] = 90.0
prob["z0"] = 0.0
prob["cd_usr"] = -1.0
prob["rho_air"] = 1.225
prob["mu_air"] = 1.7934e-5
prob["beta_wind"] = 0.0
if modeling_options["WISDEM"]["TowerSE"]["wind"] == "PowerWind":
    prob["shearExp"] = 0.2
# ---


# two load cases.  TODO: use a case iterator

# --- loading case 1: max Thrust ---
prob["env1.Uref"] = 11.73732
Fx1 = 1284744.19620519
Fy1 = 0.0
Fz1 = -2914124.84400512
Mxx1 = 3963732.76208099
Myy1 = -2275104.79420872
Mzz1 = -346781.68192839
# ---------------

# --- loading case 2: max Wind Speed ---
prob["env2.Uref"] = 70.0
Fx2 = 930198.60063279
Fy2 = 0.0
Fz2 = -2883106.12368949
Mxx2 = -1683669.22411597
Myy2 = -2522475.34625363
Mzz2 = 147301.97023764
# ---------------
prob["tower.rna_F"] = np.c_[[Fx1, Fy1, Fz1], [Fx2, Fy2, Fz2]]
prob["tower.rna_M"] = np.c_[[Mxx1, Myy1, Mzz1], [Mxx2, Myy2, Mzz2]]

# run the analysis or optimization
prob.model.approx_totals()
if opt_flag:
    prob.run_driver()
else:
    prob.run_model()
os.makedirs("outputs", exist_ok=True)
save_data(os.path.join("outputs", "tower_example"), prob)
# ---

# print results from the analysis or optimization
z = 0.5 * (prob["z_full"][:-1] + prob["z_full"][1:])
print("zs =", prob["z_full"])
print("ds =", prob["d_full"])
print("ts =", prob["t_full"])
print("mass (kg) =", prob["tower_mass"])
print("cg (m) =", prob["tower_center_of_mass"])
print("d:t constraint =", prob["constr_d_to_t"])
print("taper ratio constraint =", prob["constr_taper"])
print("\nwind: ", prob["env1.Uref"])
print("freq (Hz) =", prob["tower.structural_frequencies"])
print("Fore-aft mode shapes =", prob["tower.fore_aft_modes"])
print("Side-side mode shapes =", prob["tower.side_side_modes"])
print("top_deflection (m) =", prob["tower.top_deflection"])
print("Tower base forces (N) =", prob["tower.turbine_F"])
print("Tower base moments (Nm) =", prob["tower.turbine_M"])
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
