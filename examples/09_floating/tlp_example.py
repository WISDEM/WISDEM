# TODO: Code commenting and RST parallel

import numpy as np
import openmdao.api as om

from wisdem.commonse import fileIO
from wisdem.floatingse import FloatingSE

plot_flag = False  # True
opt_flag = False

npts = 5

opt = {}
opt["flags"] = {}
opt["flags"]["floating"] = opt["flags"]["offshore"] = opt["flags"]["tower"] = True
opt["flags"]["tower"] = False
opt["floating"] = {}
opt["floating"]["members"] = {}
opt["floating"]["members"]["n_members"] = 4
opt["floating"]["members"]["n_height"] = [npts, 3, 3, 3]
opt["floating"]["members"]["n_bulkheads"] = [4, 2, 2, 2]
opt["floating"]["members"]["n_layers"] = [1, 1, 1, 1]
opt["floating"]["members"]["n_ballasts"] = [2, 0, 0, 0]
opt["floating"]["members"]["n_axial_joints"] = [0, 0, 0, 0]
opt["floating"]["members"]["outer_shape"] = 4 * ["circular"]
opt["WISDEM"] = {}
opt["WISDEM"]["n_dlc"] = 1
opt["WISDEM"]["TowerSE"] = {}
opt["WISDEM"]["TowerSE"]["n_height"] = npts
opt["WISDEM"]["TowerSE"]["n_layers"] = 1
opt["WISDEM"]["TowerSE"]["n_refine"] = 1
opt["WISDEM"]["FloatingSE"] = {}
opt["WISDEM"]["FloatingSE"]["frame3dd"] = {}
opt["WISDEM"]["FloatingSE"]["frame3dd"]["shear"] = False
opt["WISDEM"]["FloatingSE"]["frame3dd"]["geom"] = False
opt["WISDEM"]["FloatingSE"]["frame3dd"]["modal_method"] = 2
opt["WISDEM"]["FloatingSE"]["frame3dd"]["tol"] = 1e-6
opt["WISDEM"]["FloatingSE"]["gamma_f"] = 1.35  # Safety factor on loads
opt["WISDEM"]["FloatingSE"]["gamma_m"] = 1.3  # Safety factor on materials
opt["WISDEM"]["FloatingSE"]["gamma_n"] = 1.0  # Safety factor on consequence of failure
opt["WISDEM"]["FloatingSE"]["gamma_b"] = 1.1  # Safety factor on buckling
opt["WISDEM"]["FloatingSE"]["gamma_fatigue"] = 1.755  # Not used
opt["WISDEM"]["FloatingSE"]["rank_and_file"] = True
opt["mooring"] = {}
opt["mooring"]["n_attach"] = 3
opt["mooring"]["n_anchors"] = 3
opt["mooring"]["line_anchor"] = ["custom"] * 3
opt["mooring"]["line_material"] = ["custom"] * 3
opt["materials"] = {}
opt["materials"]["n_mat"] = 2

prob = om.Problem(reports=False)
prob.model = FloatingSE(modeling_options=opt)
prob.setup()

# Material properties
prob["rho_mat"] = np.array([7850.0, 5000.0])  # Steel, ballast slurry [kg/m^3]
prob["E_mat"] = 200e9 * np.ones((2, 3))  # Young's modulus [N/m^2]
prob["G_mat"] = 79.3e9 * np.ones((2, 3))  # Shear modulus [N/m^2]
prob["sigma_y_mat"] = 3.45e8 * np.ones(2)  # Elastic yield stress [N/m^2]
prob["sigma_ult_mat"] = 5e8 * np.ones((2, 3))
prob["wohler_exp_mat"] = 4.0 * np.ones(2)
prob["wohler_A_mat"] = 7.5e8 * np.ones(2)
prob["unit_cost_mat"] = np.array([2.0, 1.0])
prob["material_names"] = ["steel", "slurry"]

# Mass and cost scaling factors
prob["labor_cost_rate"] = 1.0  # Cost factor for labor time [$/min]
prob["painting_cost_rate"] = 14.4  # Cost factor for column surface finishing [$/m^2]

# Main geometry
h = np.array([10.0, 20.0, 10.0, 8.0])
prob["member0.outfitting_factor_in"] = 1.05  # Fraction of additional outfitting mass for each column
prob["member0.grid_axial_joints"] = []
prob["member0.ballast_grid"] = np.array([[0, 0.25], [0, 0.5]])
prob["member0.ballast_volume"] = [np.pi * 7**2 * 5, 0.0]
prob["member0.s_in"] = np.cumsum(np.r_[0, h]) / h.sum()
prob["member0.outer_diameter_in"] = 14 * np.ones(npts)
prob["member0.layer_thickness"] = 0.05 * np.ones((1, npts))
prob["member0.layer_materials"] = ["steel"]
prob["member0.ballast_materials"] = ["slurry", "seawater"]
prob["member0:joint1"] = np.array([0.0, 0.0, 8.0 - h.sum()])
prob["member0:joint2"] = np.array([0.0, 0.0, 8.0])  # Freeboard=10
prob["member0.bulkhead_thickness"] = 0.05 * np.ones(4)  # Locations of internal bulkheads
prob["member0.bulkhead_grid"] = np.array([0.0, 0.25, 0.5, 1.0])
prob["member0.ring_stiffener_web_height"] = 0.10
prob["member0.ring_stiffener_web_thickness"] = 0.04
prob["member0.ring_stiffener_flange_width"] = 0.10
prob["member0.ring_stiffener_flange_thickness"] = 0.02
prob["member0.ring_stiffener_spacing"] = 0.044791667  # non-dimensional ring stiffener spacing

# Now do the legs
angs = np.linspace(0, 2 * np.pi, 1 + opt["mooring"]["n_attach"])
for k in range(1, 4):
    prob["member" + str(k) + ".outfitting_factor_in"] = 1.05  # Fraction of additional outfitting mass for each column
    prob["member" + str(k) + ".grid_axial_joints"] = []
    prob["member" + str(k) + ".s_in"] = np.array([0.0, 0.5, 1.0])
    prob["member" + str(k) + ".outer_diameter_in"] = 5 * np.ones(3)
    prob["member" + str(k) + ".ca_usr_grid"] = 2.0*np.ones(3)  # Added mass coefficient
    prob["member" + str(k) + ".cd_usr_grid"] = -1.0*np.ones(3)  # drag coefficient
    prob["member" + str(k) + ".layer_thickness"] = 0.05 * np.ones((1, 3))
    prob["member" + str(k) + ".layer_materials"] = ["steel"]
    prob["member" + str(k) + ".ballast_materials"] = []
    prob["member" + str(k) + ":joint1"] = np.array([30.0 * np.cos(angs[k - 1]), 30.0 * np.sin(angs[k - 1]), -40.0])
    prob["member" + str(k) + ":joint2"] = np.array([0.0, 0.0, -40.0])  # Freeboard=10
    prob["member" + str(k) + ".bulkhead_thickness"] = 0.05 * np.ones(2)  # Locations of internal bulkheads
    prob["member" + str(k) + ".bulkhead_grid"] = np.array([0.0, 1.0])
    prob["member" + str(k) + ".ring_stiffener_web_height"] = 0.10
    prob["member" + str(k) + ".ring_stiffener_web_thickness"] = 0.04
    prob["member" + str(k) + ".ring_stiffener_flange_width"] = 0.10
    prob["member" + str(k) + ".ring_stiffener_flange_thickness"] = 0.02
    prob["member" + str(k) + ".ring_stiffener_spacing"] = 0.06666667  # non-dimensional ring stiffener spacing

# Mooring parameters: Nylon
prob["line_diameter"] = 0.5  # Diameter of mooring line/chain [m]
prob["line_length"] = 250.0  # Unstretched mooring line length
prob["line_mass_density_coeff"] = 0.6476e3
prob["line_stiffness_coeff"] = 1.18e8
prob["line_breaking_load_coeff"] = 139357e3
prob["line_cost_rate_coeff"] = 3.415e4
prob["fairlead_radius"] = 30.0
prob["fairlead"] = 40.0
prob["anchor_radius"] = 50.0
prob["anchor_cost"] = 1e5

# Mooring constraints
prob["max_surge_fraction"] = 0.1  # Max surge/sway offset [m]
prob["survival_heel"] = 10.0  # Max heel (pitching) angle [deg]
prob["operational_heel"] = 5.0  # Max heel (pitching) angle [deg]

# Set environment to that used in OC3 testing campaign
prob["rho_air"] = 1.226  # Density of air [kg/m^3]
prob["mu_air"] = 1.78e-5  # Viscosity of air [kg/m/s]
prob["rho_water"] = 1025.0  # Density of water [kg/m^3]
prob["mu_water"] = 1.08e-3  # Viscosity of water [kg/m/s]
prob["water_depth"] = 320.0  # Distance to sea floor [m]
prob["Hsig_wave"] = 10.8  # Significant wave height [m]
prob["Tsig_wave"] = 9.8  # Wave period [s]
prob["shearExp"] = 0.11  # Shear exponent in wind power law
prob["Uc"] = 0.0  # Mean current speed
prob["beta_wind"] = prob["beta_wave"] = 0.0
prob["env.Uref"] = 11.0
prob["wind_reference_height"] = 119.0

prob["transition_node"] = prob["member0:joint2"]

# Properties of rotor-nacelle-assembly (RNA)
prob["turbine_mass"] = 350e3
prob["turbine_F"] = np.array([1284744.196, 0, -112400.5527])
prob["turbine_M"] = np.array([3963732.762, 896380.8464, -346781.682])

# Use FD and run optimization
prob.run_model()
prob.model.list_outputs(units=True)

# Visualize with mayavi, which can be difficult to install
if plot_flag:
    import wisdem.floatingse.visualize as viz

    vizobj = viz.Visualize(prob)
    vizobj.draw_spar()
