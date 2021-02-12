# TODO: Code commenting and RST parallel

# Simple mooring optimization
# Optimization by flag
import numpy as np
import openmdao.api as om
from wisdem.floatingse.map_mooring import MapMooring

plot_flag = False
opt_flag = True

# Analysis options
opt = {}
opt["n_attach"] = 3
opt["n_anchors"] = 3

# OpenMDAO initialization
prob = om.Problem()
prob.model.add_subsystem("moor", MapMooring(options=opt, gamma=1.35), promotes=["*"])

# Setup up optimization problem
if opt_flag:
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    # --- Objective ---
    prob.model.add_objective("mooring_cost", scaler=1e-6)
    # ----------------------

    prob.model.add_design_var("line_diameter", lower=1e-3, upper=2.0)
    prob.model.add_design_var("line_length", lower=1.0, upper=10e4)
    prob.model.add_design_var("anchor_radius", lower=1.0, upper=10e4)

    # --- Constraints ---
    # Make sure chain doesn't break during extreme events
    prob.model.add_constraint("constr_axial_load", upper=1.0)
    prob.model.add_constraint("max_surge_restoring_force", lower=1e6)  # N
    prob.model.add_constraint("constr_mooring_length", upper=1.0)
    # ----------------------

prob.setup()

# Environment
prob["rho_water"] = 1025.0  # kg/m^3
prob["water_depth"] = 300.0  # m

# "Vessel" geometry
prob["fairlead"] = 20.0  # m below surface
prob["fairlead_radius"] = 5.0  # m from (0,0)

# Mooring design variables initial conditions
prob["anchor_radius"] = 900.0  # m
prob["anchor_cost"] = 1e4  # m
prob["line_length"] = 0.5 * np.sqrt(prob["anchor_radius"] ** 2 + prob["water_depth"] ** 2)
prob["line_diameter"] = 1.0  # m chain half-diameter
prob["line_mass_density_coeff"] = 19.9e3
prob["line_stiffness_coeff"] = 8.54e10
prob["line_breaking_load_coeff"] = 818125253.0
prob["line_cost_rate_coeff"] = 3.415e4

# User inputs (could be design variables)
prob["max_surge_fraction"] = 0.1
prob["operational_heel"] = 6.0  # deg
prob["survival_heel"] = 10.0  # deg

# Use FD and run optimization
prob.model.approx_totals()
prob.run_driver()

# Simple screen outputs
print("Design Variables:")
print("Mooring diameter:", prob["line_diameter"])
print("Mooring line length:", prob["line_length"])
print("Anchor distance:", prob["anchor_radius"])
print("")
print("Constraints")
print("Axial tension utilization:", prob["constr_axial_load"])
print("Line length max:", prob["constr_mooring_length"])
print("Force at max offset:", prob["max_surge_restoring_force"])
print("Force at max heel:", prob["operational_heel_restoring_force"][:3, :])

# Plot mooring system if requested
if plot_flag:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    nlines = int(opt["n_anchors"])
    data = prob["mooring_plot_matrix"]
    fig = plt.figure()
    ax = Axes3D(fig)
    for k in range(nlines):
        ax.plot(data[k, :, 0], data[k, :, 1], data[k, :, 2], "b-")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    plt.show()
