import numpy as np
import openmdao.api as om
import wisdem.commonse.fileIO as fio
from wisdem.drivetrainse.generator import Generator

opt_flag = False
n_pc = 20

# Example optimization of a generator for costs on a 5 MW reference turbine
prob = om.Problem()
prob.model = Generator(design="pmsg_arms", n_pc=n_pc)

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
    prob.model.add_design_var("h_s", lower=0.04, upper=0.1)
    prob.model.add_design_var("tau_p", lower=0.04, upper=0.1)
    prob.model.add_design_var("h_m", lower=0.005, upper=0.1)
    prob.model.add_design_var("n_r", lower=5.0, upper=15.0)
    prob.model.add_design_var("h_yr", lower=0.045, upper=0.25)
    prob.model.add_design_var("h_ys", lower=0.045, upper=0.25)
    prob.model.add_design_var("n_s", lower=5.0, upper=15.0)
    prob.model.add_design_var("b_st", lower=0.1, upper=1.5)
    prob.model.add_design_var("d_s", lower=0.1, upper=1.5)
    prob.model.add_design_var("t_ws", lower=0.001, upper=0.2)

    prob.model.add_constraint("con_Bsmax", lower=0.0 + eps)
    prob.model.add_constraint("E_p", lower=500.0, upper=5000.0, ref=1e3)
    prob.model.add_design_var("b_arm", lower=0.1, upper=1.5)
    prob.model.add_design_var("d_r", lower=0.1, upper=1.5)
    prob.model.add_design_var("t_wr", lower=0.001, upper=0.2)

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
    prob.model.add_constraint("A_Cuscalc", lower=5.0, upper=300, ref=1e3)
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
prob["shaft_rpm"] = np.linspace(2, 12.1, n_pc)
prob["sigma"] = 48.373e3
# prob['r_s']     = 3.26
prob["rad_ag"] = 3.26
prob["len_s"] = 1.60
prob["h_s"] = 0.070
prob["tau_p"] = 0.080
prob["h_m"] = 0.009
prob["h_ys"] = 0.075
prob["h_yr"] = 0.075
prob["n_s"] = 5.0
prob["b_st"] = 0.480
prob["n_r"] = 5.0
prob["b_r"] = 0.530
prob["d_r"] = 0.700
prob["d_s"] = 0.350
prob["t_wr"] = 0.06
prob["t_ws"] = 0.06
prob["D_shaft"] = (
    2 * 0.43
)  # 0.523950817  #0.43  #0.523950817 #0.17625 #0.2775 #0.363882632 ##0.35 #0.523950817 #0.43 #523950817 #0.43 #0.523950817 #0.523950817 #0.17625 #0.2775 #0.363882632 #0.43 #0.523950817 #0.43
prob["q1"] = 1
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
# fio.save_data('PMSG_ARMS', prob, npz_file=False, mat_file=False, xls_file=True)
