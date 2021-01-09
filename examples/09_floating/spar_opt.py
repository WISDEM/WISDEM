# TODO: Code commenting and RST parallel

# Optimization of OC3 spar (by flag)
import numpy as np
import openmdao.api as om
from wisdem.commonse import fileIO
from wisdem.floatingse import FloatingSE

plot_flag = False
opt_flag = False

npts = 10
min_diam = 7.0
max_diam = 20.0

opt = {}
opt["floating"] = {}
opt["WISDEM"] = {}
opt["WISDEM"]["FloatingSE"] = {}
opt["floating"]["members"] = {}
opt["floating"]["members"]["n_members"] = 1
opt["floating"]["members"]["n_height"] = [npts]
opt["floating"]["members"]["n_bulkheads"] = [4]
opt["floating"]["members"]["n_layers"] = [1]
opt["floating"]["members"]["n_ballasts"] = [2]
opt["floating"]["members"]["n_axial_joints"] = [1]
opt["floating"]["tower"] = {}
opt["floating"]["tower"]["n_height"] = [npts]
opt["floating"]["tower"]["n_bulkheads"] = [0]
opt["floating"]["tower"]["n_layers"] = [1]
opt["floating"]["tower"]["n_ballasts"] = [0]
opt["floating"]["tower"]["n_axial_joints"] = [0]
opt["WISDEM"]["FloatingSE"]["frame3dd"] = {}
opt["WISDEM"]["FloatingSE"]["frame3dd"]["shear"] = True
opt["WISDEM"]["FloatingSE"]["frame3dd"]["geom"] = True
opt["WISDEM"]["FloatingSE"]["frame3dd"]["modal"] = True
opt["WISDEM"]["FloatingSE"]["frame3dd"]["tol"] = 1e-6
opt["WISDEM"]["FloatingSE"]["gamma_f"] = 1.35  # Safety factor on loads
opt["WISDEM"]["FloatingSE"]["gamma_m"] = 1.3  # Safety factor on materials
opt["WISDEM"]["FloatingSE"]["gamma_n"] = 1.0  # Safety factor on consequence of failure
opt["WISDEM"]["FloatingSE"]["gamma_b"] = 1.1  # Safety factor on buckling
opt["WISDEM"]["FloatingSE"]["gamma_fatigue"] = 1.755  # Not used
opt["WISDEM"]["FloatingSE"]["run_modal"] = True  # Not used
opt["mooring"] = {}
opt["mooring"]["n_attach"] = 3
opt["mooring"]["n_anchors"] = 3
opt["materials"] = {}
opt["materials"]["n_mat"] = 2

prob = om.Problem()
prob.model = FloatingSE(modeling_options=opt)


# Setup up optimization problem
if opt_flag:
    prob.driver = om.ScipyOptimizeDriver()  # pyOptSparseDriver() #
    prob.driver.options["optimizer"] = "SLSQP"  #'SNOPT' #
    prob.driver.options["tol"] = 1e-4
    prob.driver.options["maxiter"] = 400

    # --- Objective ---
    prob.model.add_objective("structural_mass", scaler=1e-6)
    # ----------------------

    # --- Design Variables ---
    # Mooring system
    # prob.model.add_design_var('line_diameter', lower=1e-3, upper=0.35)
    # prob.model.add_design_var('line_length', lower=1.0, upper=10e4)
    # prob.model.add_design_var('anchor_radius', lower=1.0, upper=10e4)

    # Column geometry
    prob.model.add_design_var("member0.ballast_volume", lower=10.0, upper=2000.0, indices=[0])
    prob.model.add_design_var("member0.joint1", lower=-200, upper=-25.0, indices=[-1])
    prob.model.add_design_var("member0.joint2", lower=1.0, upper=30.0, indices=[-1])
    prob.model.add_design_var("member0.outer_diameter_in", lower=min_diam, upper=max_diam)
    prob.model.add_design_var("member0.layer_thickness", lower=2e-3, upper=8e-1)

    # prob.model.add_design_var('member0.stiffener_web_height', lower=1e-2, upper=1.0)
    # prob.model.add_design_var('member0.stiffener_web_thickness', lower=1e-3, upper=1e-1)
    # prob.model.add_design_var('member0.stiffener_flange_width', lower=1e-2, upper=1.0)
    # prob.model.add_design_var('member0.stiffener_flange_thickness', lower=1e-3, upper=1e-1)
    # prob.model.add_design_var('member0.stiffener_spacing', lower=2.0, upper=10.0)

    # --- Constraints ---
    # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
    # prob.model.add_constraint('member0.draft_margin', upper=1.0)
    # prob.model.add_constraint('member0.wave_height_freeboard_ratio', upper=1.0)
    # prob.model.add_constraint('wave_height_fairlead_ratio', upper=1.0)

    # Ensure that the radius doesn't change dramatically over a section
    prob.model.add_constraint("member0.constr_taper", lower=0.2)
    prob.model.add_constraint("member0.constr_d_to_t", lower=80.0)

    # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
    # prob.model.add_constraint('axial_unity', lower=0.0, upper=1.0)

    # Ensure there is sufficient mooring line length, MAP doesn't give an error about this
    # prob.model.add_constraint('mooring_length_max', upper=1.0)

    # API Bulletin 2U constraints
    # prob.model.add_constraint('member0.flange_spacing_ratio', upper=1.0)
    # prob.model.add_constraint('member0.stiffener_radius_ratio', upper=0.5)
    # prob.model.add_constraint('member0.flange_compactness', lower=1.0)
    # prob.model.add_constraint('member0.web_compactness', lower=1.0)
    # prob.model.add_constraint('member0.axial_local_api', upper=1.0)
    # prob.model.add_constraint('member0.axial_general_api', upper=1.0)
    # prob.model.add_constraint('member0.external_local_api', upper=1.0)
    # prob.model.add_constraint('member0.external_general_api', upper=1.0)

    # Eurocode constraints
    prob.model.add_constraint("main_stress", upper=1.0)
    prob.model.add_constraint("main_shell_buckling", upper=1.0)
    prob.model.add_constraint("main_global_buckling", upper=1.0)
    # prob.model.add_constraint('tower_stress', upper=1.0)
    # prob.model.add_constraint('tower_shell_buckling', upper=1.0)
    # prob.model.add_constraint('tower_global_buckling', upper=1.0)

    # Achieving non-zero variable ballast height means the semi can be balanced with margin as conditions change
    prob.model.add_constraint("variable_ballast_height_ratio", lower=0.0, upper=1.0)
    prob.model.add_constraint("variable_ballast_mass", lower=0.0)

    # Metacentric height should be positive for static stability
    prob.model.add_constraint("metacentric_height", lower=1.0)

    # Surge restoring force should be greater than wave-wind forces (ratio < 1)
    # prob.model.add_constraint('offset_force_ratio', upper=1.0)

    # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
    # prob.model.add_constraint('heel_moment_ratio', upper=1.0)

prob.setup()

# Material properties
prob["rho_mat"] = np.array([7850.0, 5000.0])  # Steel, ballast slurry [kg/m^3]
prob["E_mat"] = 200e9 * np.ones((2, 3))  # Young's modulus [N/m^2]
prob["G_mat"] = 79.3e9 * np.ones((2, 3))  # Shear modulus [N/m^2]
prob["sigma_y_mat"] = 3.45e8 * np.ones(2)  # Elastic yield stress [N/m^2]
prob["unit_cost_mat"] = np.array([2.0, 1.0])
prob["material_names"] = ["steel", "slurry"]

# Mass and cost scaling factors
prob["labor_cost_rate"] = 1.0  # Cost factor for labor time [$/min]
prob["painting_cost_rate"] = 14.4  # Cost factor for column surface finishing [$/m^2]

# Column geometry
h = np.array([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
prob["member0.outfitting_factor_in"] = 1.05  # Fraction of additional outfitting mass for each column
prob["member0.grid_axial_joints"] = [0.384615]  # Fairlead at 70m
prob["member0.ballast_grid"] = np.array([[0, 0.37692308], [0, 0.89230769]])
prob["member0.ballast_volume"] = [np.pi * 4.7 ** 2 * 10, 0.0]
prob["member0.s"] = np.linspace(0, 1, npts)
prob["member0.outer_diameter_in"] = 10 * np.ones(npts)
prob["member0.layer_thickness"] = 0.05 * np.ones((1, npts))
prob["member0.layer_materials"] = ["steel"]
prob["member0.ballast_materials"] = ["slurry", "seawater"]
prob["member0.joint1"] = np.array([0.0, 0.0, 10.0 - h.sum()])
prob["member0.joint2"] = np.array([0.0, 0.0, 10.0])  # Freeboard=10
prob["member0.transition_flag"] = [False, True]
prob["member0.bulkhead_thickness"] = 0.05 * np.ones(4)
prob["member0.bulkhead_grid"] = np.array([0.0, 0.37692308, 0.89230769, 1.0])
prob["member0.ring_stiffener_web_height"] = 0.10
prob["member0.ring_stiffener_web_thickness"] = 0.04
prob["member0.ring_stiffener_flange_width"] = 0.10
prob["member0.ring_stiffener_flange_thickness"] = 0.02
prob["member0.ring_stiffener_spacing"] = 2.15

# Mooring parameters
prob["line_diameter"] = 0.09  # Diameter of mooring line/chain [m]
prob["line_length"] = 300 + 902.2  # Unstretched mooring line length
prob["line_mass_density_coeff"] = 19.9e3
prob["line_stiffness_coeff"] = 8.54e10
prob["line_breaking_load_coeff"] = 818125253.0
prob["line_cost_rate_coeff"] = 3.415e4
prob["fairlead_radius"] = 10.0
prob["fairlead"] = 70.0
prob["anchor_radius"] = 853.87
prob["anchor_cost"] = 1e5

# Mooring constraints
prob["max_surge_fraction"] = 0.1  # Max surge/sway offset [m]
prob["survival_heel"] = 10.0  # Max heel (pitching) angle [deg]
prob["operational_heel"] = 5.0  # Max heel (pitching) angle [deg]

# Set environment to that used in OC3 testing campaign
# prob["rho_air"] = 1.226  # Density of air [kg/m^3]
# prob["mu_air"] = 1.78e-5  # Viscosity of air [kg/m/s]
prob["rho_water"] = 1025.0  # Density of water [kg/m^3]
# prob["mu_water"] = 1.08e-3  # Viscosity of water [kg/m/s]
prob["water_depth"] = 320.0  # Distance to sea floor [m]
# prob["Hsig_wave"] = 10.8  # Significant wave height [m]
# prob["Tsig_wave"] = 9.8  # Wave period [s]
# prob["shearExp"] = 0.11  # Shear exponent in wind power law
# prob["cm"] = 2.0  # Added mass coefficient
# prob["Uc"] = 0.0  # Mean current speed
# prob["yaw"] = 0.0  # Turbine yaw angle
# prob["beta_wind"] = prob["beta_wave"] = 0.0
# prob["cd_usr"] = -1.0  # Compute drag coefficient
# prob["Uref"] = 11.0
# prob["zref"] = 119.0

# Porperties of turbine tower
nTower = prob.model.options["modeling_options"]["floating"]["tower"]["n_height"][0]
prob["hub_height"] = 85.0
prob["tower.height"] = 85.0 - prob["member0.joint2"][-1]
prob["tower.s"] = np.linspace(0.0, 1.0, nTower)
prob["tower.outer_diameter_in"] = np.linspace(6.5, 3.87, nTower)
prob["tower.layer_thickness"] = np.linspace(0.027, 0.019, nTower).reshape((1, nTower))
prob["tower.layer_materials"] = ["steel"]
prob["tower.outfitting_factor"] = 1.07

prob["transition_node"] = prob["member0.joint2"]

# Properties of rotor-nacelle-assembly (RNA)
prob["rna_mass"] = 350e3
prob["rna_I"] = 1e5 * np.array([1149.307, 220.354, 187.597, 0, 5.037, 0])
prob["rna_cg"] = np.array([-1.132, 0, 0.509])
prob["rna_F"] = np.array([1284744.196, 0, -112400.5527])
prob["rna_M"] = np.array([3963732.762, 896380.8464, -346781.682])

# Use FD and run optimization
if opt_flag:
    prob.model.approx_totals(form="central", step=1e-4)
    prob.run_driver()
    fileIO.save_data("spar_out", prob)
else:
    prob.run_model()

# Visualize with mayavi, which can be difficult to install
if plot_flag:
    import wisdem.floatingse.visualize as viz

    vizobj = viz.Visualize(prob)
    vizobj.draw_spar()
