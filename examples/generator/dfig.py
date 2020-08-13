import openmdao.api as om
import numpy as np
from wisdem.drivetrainse.generator import Generator
import wisdem.commonse.fileIO as fio

opt_flag = False

#Example optimization of a generator for costs on a 5 MW reference turbine
prob=om.Problem()
prob.model = Generator(design='dfig', topLevelFlag=True)

if opt_flag:
    # add optimizer and set-up problem (using user defined input on objective function)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    # Specificiency target efficiency(%)
    Eta_Target = 93.0

    eps = 1e-6

    # Design variables
    prob.model.add_design_var('r_s',     lower=0.2,  upper=1.0)
    prob.model.add_design_var('len_s',   lower=0.4,  upper=2.0)
    prob.model.add_design_var('h_s',     lower=0.04, upper=0.1)
    prob.model.add_design_var('B_symax', lower=1.0,  upper=2.0-eps)
    prob.model.add_design_var('I_0',          lower=5.,   upper=100.0)
    prob.model.add_design_var('S_Nmax',       lower=-0.3, upper=-0.1)

    # Constraints
    prob.model.add_constraint('Overall_eff',        lower=Eta_Target)
    prob.model.add_constraint('E_p',                lower=500.0+eps, upper=5000.0-eps)
    prob.model.add_constraint('TCr',                 lower=0.0+eps)
    prob.model.add_constraint('TCs',                 lower=0.0+eps)
    prob.model.add_constraint('B_g',                lower=0.7,       upper=1.2)
    prob.model.add_constraint('B_trmax',                             upper=2.0-eps)
    prob.model.add_constraint('B_tsmax',                             upper=2.0-eps)
    prob.model.add_constraint('A_1',                                 upper=60000.0-eps)
    prob.model.add_constraint('J_s',                                 upper=6.0)
    prob.model.add_constraint('J_r',                                 upper=6.0)
    prob.model.add_constraint('Slot_aspect_ratio1', lower=4.0,       upper=10.0)
    prob.model.add_constraint('K_rad',        lower=0.2,  upper=1.5)
    prob.model.add_constraint('D_ratio',      lower=1.37, upper=1.4)
    prob.model.add_constraint('Current_ratio',lower=0.1,  upper=0.3)

    Objective_function = 'Costs'
    prob.model.add_objective(Objective_function, scaler=1e-5)


prob.setup()

# Specify Target machine parameters

prob['machine_rating'] = 5000000.0
prob['n_nom']              = 1200.0
prob['Gearbox_efficiency'] = 0.955
prob['cofi'] = 0.9
prob['y_tau_p'] = 12./15.
prob['sigma'] = 21.5e3
prob['rad_ag']  = 0.61 #0.493167295965 #0.61 #meter
prob['len_s']   = 0.49 #1.06173588215 #0.49 #meter
prob['h_s']     = 0.08 #0.1 # 0.08 #meter
prob['I_0']     = 40.0 # 40.0191207049 #40.0 #Ampere
prob['B_symax'] = 1.3 #1.59611292026 #1.3 #Tesla
prob['S_Nmax']  = -0.2 #-0.3 #-0.2
prob['k_fillr']  = 0.55
prob['q1']      = 5
prob['h_0']     = 0.1
#---------------------------------------------------

# Specific costs
prob['C_Cu']         = 4.786         # Unit cost of Copper $/kg
prob['C_Fe']         = 0.556         # Unit cost of Iron $/kg
prob['C_Fes']        = 0.50139       # specific cost of Structural_mass $/kg
prob['C_PM']         =   95.0

#Material properties
prob['rho_Fe']       = 7700.0        # Steel density Kg/m3
prob['rho_Fes']      = 7850          # structural Steel density Kg/m3
prob['rho_Copper']   = 8900.0        # copper density Kg/m3
prob['rho_PM']       = 7450.0        # typical density Kg/m3 of neodymium magnets (added 2019 09 18) - for pmsg_[disc|arms]

#Run optimization
if opt_flag:
    prob.model.approx_totals()
    prob.run_driver()
else:
    prob.run_model()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
# fio.save_data('DFIG', prob, npz_file=False, mat_file=False, xls_file=True)
