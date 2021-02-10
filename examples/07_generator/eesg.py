import numpy as np
import openmdao.api as om
import wisdem.commonse.fileIO as fio
from wisdem.drivetrainse.generator import Generator

opt_flag = False
n_pc = 20

# Example optimization of a generator for costs on a 5 MW reference turbine
prob = om.Problem()
prob.model = Generator(design="eesg", n_pc=n_pc)

if opt_flag:
    # add optimizer and set-up problem (using user defined input on objective function)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    # Specificiency target efficiency(%)
    Eta_Target = 0.930

    eps = 1e-6

    # Set up design variables and bounds for a SCIG designed for a 5MW turbine
    # Design variables
    prob.model.add_design_var("r_s", lower=0.5, upper=9.0)
    prob.model.add_design_var("len_s", lower=0.5, upper=2.5)
    prob.model.add_design_var("h_s", lower=0.06, upper=0.15)
    prob.model.add_design_var("tau_p", lower=0.04, upper=0.2)
    prob.model.add_design_var("N_f", lower=10, upper=300, ref=1e2)
    prob.model.add_design_var("I_f", lower=1, upper=500, ref=1e2)
    prob.model.add_design_var("n_r", lower=5.0, upper=15.0)
    prob.model.add_design_var("h_yr", lower=0.01, upper=0.25)
    prob.model.add_design_var("h_ys", lower=0.01, upper=0.25)
    prob.model.add_design_var("b_arm", lower=0.1, upper=1.5)
    prob.model.add_design_var("d_r", lower=0.1, upper=1.5)
    prob.model.add_design_var("t_wr", lower=0.001, upper=0.2)
    prob.model.add_design_var("n_s", lower=5.0, upper=15.0)
    prob.model.add_design_var("b_st", lower=0.1, upper=1.5)
    prob.model.add_design_var("d_s", lower=0.1, upper=1.5)
    prob.model.add_design_var("t_ws", lower=0.001, upper=0.2)

    # Constraints
    prob.model.add_constraint("B_gfm", lower=0.617031, upper=1.057768)
    prob.model.add_constraint("B_pc", upper=2.0)
    prob.model.add_constraint("E_s", lower=500.0, upper=5000.0, ref=1e3)
    prob.model.add_constraint("J_f", upper=6.0)
    prob.model.add_constraint("n_brushes", upper=6)
    prob.model.add_constraint("Power_ratio", upper=2 - eps)
    prob.model.add_constraint("B_symax", upper=2.0 - eps)
    prob.model.add_constraint("B_rymax", upper=2.0 - eps)
    prob.model.add_constraint("B_tmax", upper=2.0 - eps)
    prob.model.add_constraint("B_g", lower=0.7, upper=1.2)
    prob.model.add_constraint("con_uas", lower=0.0 + eps)
    prob.model.add_constraint("con_zas", lower=0.0 + eps)
    prob.model.add_constraint("con_yas", lower=0.0 + eps)
    prob.model.add_constraint("con_uar", lower=0.0 + eps)
    prob.model.add_constraint("con_yar", lower=0.0 + eps)
    prob.model.add_constraint("con_TC2r", lower=0.0 + eps)
    prob.model.add_constraint("con_TC2s", lower=0.0 + eps)
    prob.model.add_constraint("con_bst", lower=0.0 - eps)
    prob.model.add_constraint("A_1", upper=60000.0 - eps, ref=1e5, indices=[-1])
    prob.model.add_constraint("J_s", upper=6.0)
    prob.model.add_constraint("A_Cuscalc", lower=5.0, upper=300, ref=1e2)
    prob.model.add_constraint("K_rad", lower=0.2 + eps, upper=0.27)
    prob.model.add_constraint("Slot_aspect_ratio", lower=4.0, upper=10.0)
    prob.model.add_constraint("generator_efficiency", lower=Eta_Target, indices=[-1])
    prob.model.add_constraint("con_zar", lower=0.0 + eps)
    prob.model.add_constraint("con_br", lower=0.0 + eps)

    Objective_function = "generator_cost"
    prob.model.add_objective(Objective_function, scaler=1e-5)


prob.setup()

# Specify Target machine parameters
prob["machine_rating"] = 5000000.0
prob["rated_torque"] = 4.143289e6
prob["shaft_rpm"] = np.linspace(2, 12.1, n_pc)  # 8.68                # rpm 9.6
prob["sigma"] = 48.373e3
# Initial design variables
# prob['r_s']     = 3.2
prob["rad_ag"] = 3.2
prob["len_s"] = 1.4
prob["h_s"] = 0.060
prob["tau_p"] = 0.170
prob["I_f"] = 69
prob["N_f"] = 100
prob["h_ys"] = 0.130
prob["h_yr"] = 0.120
prob["n_s"] = 5
prob["b_st"] = 0.470
prob["n_r"] = 5
prob["b_r"] = 0.480
prob["d_r"] = 0.510
prob["d_s"] = 0.400
prob["t_wr"] = 0.140
prob["t_ws"] = 0.070
prob["D_shaft"] = 2 * 0.43  # 10MW: 0.523950817,#5MW: 0.43, #3MW:0.363882632 #1.5MW: 0.2775  0.75MW: 0.17625
prob["q1"] = 2
# ---------------------------------------------------
prob["E"] = 2e11
prob["G"] = 79.3e9

# Specific costs
prob["C_Cu"] = 4.786  # Unit cost of Copper $/kg
prob["C_Fe"] = 0.556  # Unit cost of Iron $/kg
prob["C_Fes"] = 0.50139  # specific cost of Structural_mass $/kg
prob["C_PM"] = 95.0

# Material properties
prob["rho_Fe"] = 7700.0  # Steel density Kg/m3
prob["rho_Fes"] = 7850  # structural Steel density Kg/m3
prob["rho_Copper"] = 8900.0  # copper density Kg/m3
prob["rho_PM"] = 7450.0  # typical density Kg/m3 of neodymium magnets (added 2019 09 18) - for pmsg_[disc|arms]

# Run optimization
if opt_flag:
    prob.model.approx_totals()
    prob.run_driver()
else:
    prob.run_model()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
# fio.save_data('EESG', prob, npz_file=False, mat_file=False, xls_file=True)
