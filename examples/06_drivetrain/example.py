from wisdem.drivetrainse.drivetrain import DrivetrainSE
import openmdao.api as om
import numpy as np

npts = 12

opt = {}
opt['drivetrainse'] = {}
opt['drivetrainse']['direct'] = True
opt['drivetrainse']['hub'] = {}
opt['drivetrainse']['hub']['hub_gamma'] = 2.0
opt['drivetrainse']['hub']['spinner_gamma'] = 1.5
opt['drivetrainse']['gamma_f'] = 1.35
opt['drivetrainse']['gamma_m'] = 1.3
opt['drivetrainse']['gamma_n'] = 1.0
opt['servose'] = {}
opt['servose']['n_pc'] = 20
opt['materials'] = {}
opt['materials']['n_mat'] = 4
opt['GeneratorSE'] = {}
opt['GeneratorSE']['type'] = 'pmsg_outer'
opt['flags'] = {}
opt['flags']['generator'] = True

prob = om.Problem()
prob.model = DrivetrainSE(modeling_options=opt, n_dlcs=1)
prob.setup()

prob['upwind'] = True
prob['n_blades'] = 3
prob['rotor_diameter'] = 240.0
prob.set_val('machine_rating', 15.0, units='MW')
prob['D_top'] = 6.5

prob['F_hub'] = np.array([2517580., -27669., 3204.]).reshape((3,1))
prob['M_hub'] = np.array([21030561., 7414045., 1450946.]).reshape((3,1))

prob['E_mat'] = np.c_[200e9*np.ones(3), 205e9*np.ones(3), 118e9*np.ones(3), [4.46E10, 1.7E10, 1.67E10]].T
prob['G_mat'] = np.c_[79.3e9*np.ones(3), 80e9*np.ones(3), 47.6e9*np.ones(3), [3.27E9, 3.48E9, 3.5E9]].T
prob['Xt_mat'] = np.c_[450e6*np.ones(3), 814e6*np.ones(3), 310e6*np.ones(3), [6.092E8, 3.81E7, 1.529E7]].T
prob['rho_mat'] = np.r_[7800.0, 7850.0, 7200.0, 1940.0]
prob['sigma_y_mat'] = np.r_[345e6, 485e6, 265e6, 18.9e6]
prob['unit_cost_mat'] = np.r_[0.7, 0.9, 0.5, 1.9]
prob['lss_material'] = prob['hss_material'] = 'steel_drive'
prob['bedplate_material'] = 'steel'
prob['hub_material'] = 'cast_iron'
prob['spinner_material'] = 'glass_uni'
prob['material_names'] = ['steel','steel_drive','cast_iron','glass_uni']

prob['blade_mass']                  = 65252.
prob['blades_mass']                 = 3*prob['blade_mass']
prob['pitch_system.BRFM']           = 26648449.
prob['pitch_system_scaling_factor'] = 0.75

prob['blade_root_diameter']         = 5.2
prob['flange_t2shell_t']            = 6.
prob['flange_OD2hub_D']             = 0.7
prob['flange_ID2flange_OD']         = 0.9
prob['hub_stress_concentration']    = 3.0

prob['n_front_brackets']            = 8
prob['n_rear_brackets']             = 8
prob['clearance_hub_spinner']       = 0.5
prob['spin_hole_incr']              = 1.2
prob['spinner_gust_ws']             = 70.

prob['hub_diameter']                = 7.94
prob['minimum_rpm']                 = 5.0
prob['rated_rpm']                   = 7.56
prob['blades_I']                    = np.r_[4.12747714e+08, 1.97149973e+08, 1.54854398e+08, np.zeros(3)]

prob['bear1.bearing_type'] = 'CARB'
prob['bear2.bearing_type'] = 'SRB'

prob['L_12'] = 1.2
prob['L_h1'] = 1.0
prob['L_generator'] = 2.15 #2.75
prob['overhang'] = 12.0313
prob['drive_height'] = 5.614
prob['tilt'] = 6.0
prob['access_diameter'] = 0.8001

myones = np.ones(5)
prob['lss_diameter'] = 3.0*myones
prob['nose_diameter'] = 2.2*myones
prob['lss_wall_thickness'] = 0.1*myones
prob['nose_wall_thickness'] = 0.1*myones
prob['bedplate_wall_thickness'] = 0.05*np.ones(npts)
prob['bear1.D_shaft'] = 2.2
prob['bear2.D_shaft'] = 2.2
prob['generator.D_shaft'] = 3.0
prob['generator.D_nose'] = 2.2

prob['rated_torque']             = prob['M_hub'][0]
prob['generator.P_mech']         = 15354206.45251639

prob['generator.rho_Fe'] = 7700.0
prob['generator.rho_Fes'] = 7850.0
prob['generator.rho_Copper'] = 8900.0
prob['generator.rho_PM'] = 7450.0
prob['generator.B_r'] = 1.279
prob['generator.P_Fe0e'] = 1.0
prob['generator.P_Fe0h'] = 4.0
prob['generator.S_N'] = -0.002
prob['generator.alpha_p'] = 1.0995574287564276 #0.5*np.pi*0.7
prob['generator.b_r_tau_r'] = 0.45
prob['generator.b_ro'] = 0.004
prob['generator.b_s_tau_s'] = 0.45
prob['generator.b_so'] = 0.004
prob['generator.cofi'] = 0.9
prob['generator.freq'] = 60.0
prob['generator.h_i'] = 0.004
prob['generator.h_sy0'] = 0.0
prob['generator.h_w'] = 0.005
prob['generator.k_fes'] = 0.8
prob['generator.k_fillr'] = 0.55
prob['generator.k_fills'] = 0.65
prob['generator.k_s'] = 0.2
prob['generator.m'] = 3
prob['generator.mu_0'] = 1.2566370614359173e-06 #np.pi*4e-7
prob['generator.mu_r'] = 1.06
prob['generator.phi'] = 1.5707963267948966 # 90 deg
prob['generator.q1'] = 5
prob['generator.q2'] = 4
prob['generator.ratio_mw2pp'] = 0.8
prob['generator.resist_Cu'] = 2.52e-8 #1.8e-8*1.4
prob['generator.y_tau_pr'] = 0.8333333 #10. / 12
prob['generator.y_tau_p'] = 0.8 #12./15.
prob['generator.rad_ag'] = 5.12623359
prob['generator.len_s'] = 2.23961662
prob['generator.tau_p'] = 0.1610453779
prob['generator.tau_s'] = 0.1339360726
prob['generator.h_s'] = 0.3769449149
prob['generator.b_s'] = 0.05121796888
prob['generator.b_t'] = 0.08159225451
prob['generator.h_t'] = 0.3859449149
prob['generator.h_ys'] = 0.03618114528
prob['generator.h_yr'] = 0.0361759511
prob['generator.h_m'] = 0.0995385122
prob['generator.b_m'] = 0.1288363023
prob['generator.B_g'] = 1.38963289
prob['generator.B_symax'] = 1.63455514
prob['generator.B_rymax'] = 1.63478983
prob['generator.p'] = 100
prob['generator.R_s'] = 0.02457052
prob['generator.L_s'] = 0.01138752
prob['generator.S'] = 240
prob['generator.t_r'] = 0.0607657125
prob['generator.h_sr'] = 0.04677116325
prob['generator.t_s'] = 0.06065213554
prob['generator.h_ss'] = 0.04650825863
prob['generator.E_p'] = 1905.2558883257652
prob['generator.b'] = 2.0
prob['generator.c'] = 5.0
prob['generator.N_c'] = 2
prob['generator.B_tmax'] = 1.9
prob['generator.u_allow_pcent'] = 8.5
prob['generator.y_allow_pcent'] = 1.0
prob['generator.z_allow_deg'] = 0.05
prob['generator.sigma'] = 60e3
prob['generator.h_0'] = 0.05
prob['generator.C_Cu'] = 4.786
prob['generator.C_Fe'] = 0.556
prob['generator.C_Fes'] = 0.50139
prob['generator.C_PM'] = 50.0

prob.run_model()
prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
