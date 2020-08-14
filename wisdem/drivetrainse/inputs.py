import numpy as np


def drivese_inputs(wt_opt):


    wt_opt['drivese.L_12'] = 2.0
    wt_opt['drivese.L_h1'] = 1.0
    wt_opt['drivese.L_2n'] = 1.5
    wt_opt['drivese.L_grs'] = 1.1
    wt_opt['drivese.L_gsn'] = 1.1
    wt_opt['drivese.L_bedplate'] = 5.0
    wt_opt['drivese.H_bedplate'] = 4.875
    wt_opt['drivese.access_diameter'] = 0.9

    npts = 15
    myones = np.ones(5)
    wt_opt['drivese.lss_diameter'] = 3.3*myones
    wt_opt['drivese.nose_diameter'] = 2.2*myones
    wt_opt['drivese.lss_wall_thickness'] = 0.45*myones
    wt_opt['drivese.nose_wall_thickness'] = 0.1*myones
    wt_opt['drivese.bedplate_wall_thickness'] = 0.06*np.ones(npts)


    wt_opt['drivese.E'] = 210e9
    wt_opt['drivese.G'] = 80.8e9
    wt_opt['drivese.v'] = 0.3
    wt_opt['drivese.rho'] = 7850.
    wt_opt['drivese.sigma_y'] = 250e6
    wt_opt['drivese.gamma_f'] = 1.35
    wt_opt['drivese.gamma_m'] = 1.3
    wt_opt['drivese.gamma_n'] = 1.0

    wt_opt['drivese.pitch_system.scaling_factor']   = 0.54
    wt_opt['drivese.pitch_system.rho']              = 7850.
    wt_opt['drivese.pitch_system.Xy']               = 371.e+6

    wt_opt['drivese.flange_t2shell_t']              = 4.
    wt_opt['drivese.flange_OD2hub_D']               = 0.5
    wt_opt['drivese.flange_ID2flange_OD']           = 0.8
    wt_opt['drivese.hub_shell.rho']                 = 7200.
    wt_opt['drivese.in2out_circ']                   = 1.2 
    wt_opt['drivese.hub_shell.Xy']                  = 200.e+6
    wt_opt['drivese.stress_concentration']          = 2.5
    wt_opt['drivese.hub_shell.gamma']               = 2.0
    wt_opt['drivese.hub_shell.metal_cost']          = 3.00

    wt_opt['drivese.n_front_brackets']              = 3
    wt_opt['drivese.n_rear_brackets']               = 3
    wt_opt['drivese.clearance_hub_spinner']         = 0.5
    wt_opt['drivese.spin_hole_incr']                = 1.2
    wt_opt['drivese.spinner.gamma']                 = 1.5
    wt_opt['drivese.spinner.composite_Xt']          = 60.e6
    wt_opt['drivese.spinner.composite_SF']          = 1.5
    wt_opt['drivese.spinner.composite_rho']         = 1600.
    wt_opt['drivese.spinner.Xy']                    = 225.e+6
    wt_opt['drivese.spinner.metal_SF']              = 1.5
    wt_opt['drivese.spinner.metal_rho']             = 7850.
    wt_opt['drivese.spinner.composite_cost']        = 7.00
    wt_opt['drivese.spinner.metal_cost']            = 3.00

    wt_opt['drivese.generator.n_nom']          = 10            #8.68                # rpm 9.6
    wt_opt['drivese.generator.r_g']            = 4.0           # rev 1  4.92
    wt_opt['drivese.generator.len_s']          = 1.7           # rev 2.3
    wt_opt['drivese.generator.h_s']            = 0.7            # rev 1 0.3
    wt_opt['drivese.generator.p']              = 70            #100.0    # rev 1 160
    wt_opt['drivese.generator.h_m']            = 0.005         # rev 1 0.034
    wt_opt['drivese.generator.h_ys']           = 0.04          # rev 1 0.045
    wt_opt['drivese.generator.h_yr']           = 0.06          # rev 1 0.045
    wt_opt['drivese.generator.b']              = 2.
    wt_opt['drivese.generator.c']              = 5.0
    wt_opt['drivese.generator.B_tmax']         = 1.9
    wt_opt['drivese.generator.E_p']            = 3300/np.sqrt(3)
    wt_opt['drivese.generator.D_nose']         = 2*1.1             # Nose outer radius
    wt_opt['drivese.generator.D_shaft']        = 2*1.34            # Shaft outer radius =(2+0.25*2+0.3*2)*0.5
    wt_opt['drivese.generator.t_r']            = 0.05          # Rotor disc thickness
    wt_opt['drivese.generator.h_sr']           = 0.04          # Rotor cylinder thickness
    wt_opt['drivese.generator.t_s']            = 0.053         # Stator disc thickness
    wt_opt['drivese.generator.h_ss']           = 0.04          # Stator cylinder thickness
    wt_opt['drivese.generator.u_allow_pcent']  = 8.5            # % radial deflection
    wt_opt['drivese.generator.y_allow_pcent']  = 1.0            # % axial deflection
    wt_opt['drivese.generator.z_allow_deg']    = 0.05           # torsional twist
    wt_opt['drivese.generator.sigma']          = 60.0e3         # Shear stress
    wt_opt['drivese.generator.B_r']            = 1.279
    wt_opt['drivese.generator.ratio_mw2pp']    = 0.8
    wt_opt['drivese.generator.h_0']            = 5e-3
    wt_opt['drivese.generator.h_w']            = 4e-3
    wt_opt['drivese.generator.k_fes']          = 0.8
    wt_opt['drivese.generator.C_Cu']         = 4.786         # Unit cost of Copper $/kg
    wt_opt['drivese.generator.C_Fe']         = 0.556         # Unit cost of Iron $/kg
    wt_opt['drivese.generator.C_Fes']        = 0.50139       # specific cost of Structural_mass $/kg
    wt_opt['drivese.generator.C_PM']         =   95.0
    wt_opt['drivese.generator.rho_Fe']       = 7700.0        # Steel density Kg/m3
    wt_opt['drivese.generator.rho_Fes']      = 7850          # structural Steel density Kg/m3
    wt_opt['drivese.generator.rho_Copper']   = 8900.0        # copper density Kg/m3
    wt_opt['drivese.generator.rho_PM']       = 7450.0        # typical density Kg/m3 of neodymium magnets



    return wt_opt