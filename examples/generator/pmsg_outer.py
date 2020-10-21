import openmdao.api as om
import numpy as np
from wisdem.drivetrainse.generator import Generator
import wisdem.commonse.fileIO as fio

opt_flag = False
n_pc = 20

#Example optimization of a generator for costs on a 5 MW reference turbine
prob=om.Problem()
prob.model = Generator(design='pmsg_outer', n_pc=n_pc)

if opt_flag:
    # add optimizer and set-up problem (using user defined input on objective function)
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    # Specificiency target efficiency(%)
    Eta_Target = 93.0

    eps = 1e-6

    # Set up design variables and bounds for a SCIG designed for a 5MW turbine
    # Design variables
    prob.model.add_design_var('r_g', lower=3.0, upper=5 ) 
    prob.model.add_design_var('len_s', lower=1.5, upper=3.5 )  
    prob.model.add_design_var('h_s', lower=0.1, upper=1.00 )  
    prob.model.add_design_var('p', lower=50.0, upper=100)
    prob.model.add_design_var('h_m', lower=0.005, upper=0.2 )  
    prob.model.add_design_var('h_yr', lower=0.035, upper=0.22 )
    prob.model.add_design_var('h_ys', lower=0.035, upper=0.22 )
    prob.model.add_design_var('B_tmax', lower=1, upper=2.0 ) 
    prob.model.add_design_var('t_r', lower=0.05, upper=0.3 ) 
    prob.model.add_design_var('t_s', lower=0.05, upper=0.3 )  
    prob.model.add_design_var('h_ss', lower=0.04, upper=0.2)
    prob.model.add_design_var('h_sr', lower=0.04, upper=0.2)

    # Constraints
    prob.model.add_constraint('B_symax',lower=0.0,upper=2.0)
    prob.model.add_constraint('B_rymax',lower=0.0,upper=2.0)
    prob.model.add_constraint('b_t',    lower=0.01)
    prob.model.add_constraint('B_g', lower=0.7,upper=1.3)
    #prob.model.add_constraint('E_p',    lower=500, upper=10000)
    prob.model.add_constraint('A_Cuscalc',lower=5.0,upper=500)
    prob.model.add_constraint('K_rad',    lower=0.15,upper=0.3)
    prob.model.add_constraint('Slot_aspect_ratio',lower=4.0, upper=10.0)
    prob.model.add_constraint('gen_eff',lower=93.)
    prob.model.add_constraint('A_1',upper=95000.0)
    prob.model.add_constraint('T_e', lower= 10.26812e6,upper=10.3e6)
    prob.model.add_constraint('J_actual',lower=3,upper=6)    
    prob.model.add_constraint('con_uar',lower = 1e-2)
    prob.model.add_constraint('con_yar', lower = 1e-2)
    prob.model.add_constraint('con_uas', lower = 1e-2)
    prob.model.add_constraint('con_yas', lower = 1e-2)   

    Objective_function = 'Costs'
    prob.model.add_objective(Objective_function, scaler=1e-5)


prob.setup()

# Specify Target machine parameters

prob['B_r']            = 1.2
prob['E']              = 2e11
prob['G']              = 79.3e9
prob['P_Fe0e']         = 1.0
prob['P_Fe0h']         = 4.0
prob['S_N']            = -0.002
prob['alpha_p']        = 0.5*np.pi*0.7
prob['b_r_tau_r']      = 0.45
prob['b_ro']           = 0.004
prob['b_s_tau_s']      = 0.45
prob['b_so']           = 0.004
prob['cofi']           = 0.85
prob['freq']           = 60
prob['h_i']            = 0.001
prob['h_sy0']          = 0.0
prob['h_w']            = 0.005
prob['k_fes']          = 0.9
prob['k_fillr']        = 0.7
prob['k_fills']        = 0.65
prob['k_s']            = 0.2
prob['m']     = 3
prob['mu_0']           = np.pi*4e-7
prob['mu_r']           = 1.06
prob['phi']            = np.deg2rad(90)
prob['q1']    = 6
prob['q2']    = 4
prob['ratio_mw2pp']    = 0.7
prob['resist_Cu']      = 1.8e-8*1.4
prob['y_tau_p']        = 1.0
prob['y_tau_pr']       = 10. / 12

prob['machine_rating'] = 10.321e6
prob['rated_torque']   = 10.25e6       #rev 1 9.94718e6
prob['P_mech']         = 10.71947704e6 #rev 1 9.94718e6
prob['shaft_rpm']      = np.linspace(2,10,n_pc)            #8.68                # rpm 9.6
prob['rad_ag']         = 4.0           # rev 1  4.92
prob['len_s']          = 1.7           # rev 2.3
prob['h_s']            = 0.7            # rev 1 0.3
prob['p']              = 70            #100.0    # rev 1 160
prob['h_m']            = 0.005         # rev 1 0.034
prob['h_ys']           = 0.04          # rev 1 0.045
prob['h_yr']           = 0.06          # rev 1 0.045
prob['b']              = 2.
prob['c']              = 5.0
prob['B_tmax']         = 1.9
prob['E_p']            = 3300/np.sqrt(3)
prob['D_nose']         = 2*1.1             # Nose outer radius
prob['D_shaft']        = 2*1.34            # Shaft outer radius =(2+0.25*2+0.3*2)*0.5
prob['t_r']            = 0.05          # Rotor disc thickness
prob['h_sr']           = 0.04          # Rotor cylinder thickness
prob['t_s']            = 0.053         # Stator disc thickness
prob['h_ss']           = 0.04          # Stator cylinder thickness
prob['y_sh']           = 0.0005*0      # Shaft deflection
prob['theta_sh']       = 0.00026*0     # Slope at shaft end
prob['y_bd']           = 0.0005*0      # deflection at bedplate
prob['theta_bd']       = 0.00026*0      # Slope at bedplate end
prob['u_allow_pcent']  = 8.5            # % radial deflection
prob['y_allow_pcent']  = 1.0            # % axial deflection
prob['z_allow_deg']    = 0.05           # torsional twist
prob['sigma']          = 60.0e3         # Shear stress
prob['B_r']            = 1.279
prob['ratio_mw2pp']    = 0.8
prob['h_0']            = 5e-3
prob['h_w']            = 4e-3
prob['k_fes']          = 0.8
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
# fio.save_data('PMSG_OUTER', prob, npz_file=False, mat_file=False, xls_file=True)
