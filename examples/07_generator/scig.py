import numpy as np
import openmdao.api as om
import wisdem.commonse.fileIO as fio
from wisdem.drivetrainse.generator import Generator

opt_flag = False
n_pc = 20

# Example optimization of a generator for costs on a 5 MW reference turbine
prob = om.Problem()
prob.model = Generator(design="scig", n_pc=n_pc)

if opt_flag:
    # add optimizer and set-up problem (using user defined input on objective function)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"

    # Specificiency target efficiency(%)
    Eta_Target = 0.930

    eps = 1e-6

    # Set up design variables and bounds for a SCIG designed for a 5MW turbine
    # Design variables
    prob.model.add_design_var("r_s", lower=0.2, upper=1.0)
    prob.model.add_design_var("len_s", lower=0.4, upper=2.0)
    prob.model.add_design_var("h_s", lower=0.04, upper=0.1)
    prob.model.add_design_var("B_symax", lower=1.0, upper=2.0 - eps)
    prob.model.add_design_var("I_0", lower=5.0, upper=200.0, ref=1e2)

    # Constraints
    prob.model.add_constraint("generator_efficiency", lower=Eta_Target, indices=[-1])
    prob.model.add_constraint("E_p", lower=500.0 + eps, upper=5000.0 - eps, ref=1e3)
    prob.model.add_constraint("TCr", lower=0.0 + eps)
    prob.model.add_constraint("TCs", lower=0.0 + eps)
    prob.model.add_constraint("B_g", lower=0.7, upper=1.2)
    prob.model.add_constraint("B_trmax", upper=2.0 - eps)
    prob.model.add_constraint("B_tsmax", upper=2.0 - eps)
    prob.model.add_constraint("A_1", upper=60000.0 - eps, ref=1e5, indices=[-1])
    prob.model.add_constraint("J_s", upper=6.0)
    prob.model.add_constraint("J_r", upper=6.0)
    prob.model.add_constraint("Slot_aspect_ratio1", lower=4.0, upper=10.0)
    prob.model.add_constraint("B_rymax", upper=2.0 - eps)
    prob.model.add_constraint("K_rad_L", lower=0.0)
    prob.model.add_constraint("K_rad_U", upper=0.0)
    prob.model.add_constraint("D_ratio_L", lower=0.0)
    prob.model.add_constraint("D_ratio_U", upper=0.0)

    Objective_function = "generator_cost"
    prob.model.add_objective(Objective_function, scaler=1e-5)


prob.setup()

# Specify Target machine parameters

prob["machine_rating"] = 5000000.0
prob["shaft_rpm"] = np.linspace(200, 1200.0, n_pc)
prob["cofi"] = 0.9
prob["y_tau_p"] = 12.0 / 15.0
prob["sigma"] = 21.5e3
# prob['r_s']     = 0.55 #0.484689156353 #0.55 #meter
prob["rad_ag"] = 0.55  # 0.484689156353 #0.55 #meter
prob["len_s"] = 1.30  # 1.27480124244 #1.3 #meter
prob["h_s"] = 0.090  # 0.098331868116 # 0.090 #meter
prob["I_0"] = 140  # 139.995232826 #140  #Ampere
prob["B_symax"] = 1.4  # 1.86140258387 #1.4 #Tesla
prob["q1"] = 6
prob["h_0"] = 0.05

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
# fio.save_data('SCIG', prob, npz_file=False, mat_file=False, xls_file=True)
