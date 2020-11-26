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
opt["gamma_f"] = 1.35

# OpenMDAO initialization
prob = om.Problem()
prob.model.add_subsystem("moor", MapMooring(modeling_options=opt), promotes=["*"])

# Setup up optimization problem
if opt_flag:
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    # --- Objective ---
    prob.model.add_objective("mooring_cost", scaler=1e-6)
    # ----------------------

    prob.model.add_design_var("mooring_diameter", lower=1e-3, upper=0.35)
    prob.model.add_design_var("mooring_line_length", lower=1.0, upper=10e4)
    prob.model.add_design_var("anchor_radius", lower=1.0, upper=10e4)

    # --- Constraints ---
    # Make sure chain doesn't break during extreme events
    prob.model.add_constraint("axial_unity", upper=1.0)
    prob.model.add_constraint("max_offset_restoring_force", lower=1e6)  # N
    prob.model.add_constraint("mooring_length_max", upper=10e4)
    # ----------------------

prob.setup()

# Environment
prob["rho_water"] = 1025.0  # kg/m^3
prob["water_depth"] = 300.0  # m

# "Vessel" geometry
prob["fairlead"] = 20.0  # m below surface
prob["fairlead_radius"] = 5.0  # m from (0,0)

# Mooring design variables initial conditions
prob["mooring_diameter"] = 0.2  # m chain half-diameter
prob["anchor_radius"] = 800.0  # m
prob["mooring_line_length"] = 0.9 * np.sqrt(prob["anchor_radius"] ** 2 + prob["water_depth"] ** 2)

# User inputs (could be design variables)
prob["number_of_mooring_connections"] = 3
prob["mooring_lines_per_connection"] = 1
prob["mooring_type"] = "CHAIN"
prob["anchor_type"] = "DRAGEMBEDMENT"
prob["max_offset"] = 0.1 * prob["water_depth"]  # m
prob["operational_heel"] = 6.0  # deg
prob["max_survival_heel"] = 10.0  # deg

# Cost rates
prob["mooring_cost_factor"] = 1.0

# Use FD and run optimization
prob.model.approx_totals()
prob.run_driver()

# Simple screen outputs
print("Design Variables:")
print("Mooring diameter:", prob["mooring_diameter"])
print("Mooring line length:", prob["mooring_line_length"])
print("Anchor distance:", prob["anchor_radius"])
print("")
print("Constraints")
print("Axial tension utilization:", prob["axial_unity"])
print("Force at max offset:", prob["max_offset_restoring_force"])
print("Force at max heel:", prob["operational_heel_restoring_force"][:3, :])

# Plot mooring system if requested
if plot_flag:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    nlines = int(prob["number_of_mooring_connections"] * prob["mooring_lines_per_connection"])
    data = prob["mooring_plot_matrix"]
    fig = plt.figure()
    ax = Axes3D(fig)
    for k in range(nlines):
        ax.plot(data[k, :, 0], data[k, :, 1], data[k, :, 2], "b-")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    plt.show()
