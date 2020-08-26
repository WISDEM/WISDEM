# scigSE.py
# 2019 09 12

import sys
import numpy as np
from math import pi, sin, cos, radians, sqrt, log
from ddse_utils import carterFactor

#--------------------------------

class SCIG(object):
	
    def __init__(self):

        super(SCIG, self).__init__()
        self.debug = False

        #self.bearing_position = bearing_position
        
    #------------------------
    
    def compute(self, rad_ag, len_s, h_s, h_r, machine_rating, n_nom, Gearbox_efficiency, I_0, 
                rho_Fe, rho_Copper, B_symax, shaft_cm, shaft_length, debug=False):
        
        if debug or self.debug:
            sys.stderr.write('AG {:9.2f} LS {:9.2f}  HS {:9.2f}  HR {:9.2f}  MR {:7.1f}  NN {:9.2f}  GE {:9.2f}  I0 {:9.2f}\n'.format(rad_ag, 
                             len_s, h_s, h_r, machine_rating, n_nom, Gearbox_efficiency, I_0))
            sys.stderr.write('rF {:9.2f} rC {:9.2f}  Bs {:9.2f}  SC {:9.2f}  SC {:9.2f}  SC {:9.2f}  SL {:9.2f}\n'.format(rho_Fe, 
                             rho_Copper, B_symax, shaft_cm[0], shaft_cm[1], shaft_cm[2], shaft_length))
        
        # Assign values to universal constants
        g1    = 9.81              # m/s^2 acceleration due to gravity
        sigma = 21.5e3            # shear stress (psi) what material?
        mu_0  = pi*4e-7           # permeability of free space in m * kg / (s**2 * A**2)
        cofi  = 0.9               # power factor
        h_w   = 0.005             # wedge height
        m     = 3                 # Number of phases
        resist_Cu = 1.8e-8 * 1.4  # Copper resistivity
        
        #Assign values to design constants
        b_so      = 0.004                     # Stator slot opening width
        b_ro      = 0.004                     # Rotor  slot opening width
        q1        = 6                         # Stator slots per pole per phase
        q2        = 4                         # Rotor  slots per pole per phase
        b_s_tau_s = 0.45                      # Stator Slot width/Slot pitch ratio
        b_r_tau_r = 0.45                      # Rotor Slot width/Slot pitch ratio
        y_tau_p   = 12./15                    # Coil span/pole pitch
        
        p         = 3                         # number of pole pairs
        freq      = 60                        # frequency in Hz
        k_fillr   = 0.7                       # Rotor  slot fill factor
        k_fills   = None                      # Stator slot fill factor - to be assigned later
        P_Fe0h    = 4                         # specific hysteresis losses W / kg @ 1.5 T @50 Hz
        P_Fe0e    = 1                         # specific eddy losses W / kg @ 1.5 T @50 Hz
        
        S_N       = -0.002                    # Slip
        n_1       = n_nom / (1 - S_N)         # actual rotor speed (rpm)
        
        # Calculating winding factor 
        
        k_y1      = sin(pi/2 * y_tau_p)       # winding chording factor
        k_q1      = sin(pi/6) / (q1 * sin(pi/(6 * q1))) # zone factor
        k_wd      = k_y1 * k_q1               # winding factor                
        
        # Calculating air gap length
        ag_dia     = 2 * rad_ag               # air gap diameter
        ag_len     = (0.1 + 0.012 * machine_rating**(1./3)) * 0.001 # air gap length in m
        K_rad      = len_s / ag_dia             # Aspect ratio
        K_rad_LL   = 0.5                      # lower limit on aspect ratio
        K_rad_UL   = 1.5                      # upper limit on aspect ratio
        rad_r        = rad_ag - ag_len        # rotor radius
                                              
        tau_p      = pi * ag_dia / (2 * p)    # pole pitch
        S          = 2 * p * q1 * m           # Stator slots                                             
        N_slots_pp = S / (m * p * 2)          # Number of stator slots per pole per phase
        tau_s      = tau_p / (m * q1)         # Stator slot pitch
                                              
        b_s        = b_s_tau_s * tau_s        # Stator slot width
        b_t        = tau_s - b_s              # Stator tooth width
                                              
        Q_r        = 2 * p * m * q2           # Rotor slots
        tau_r      = pi * (ag_dia - 2 * ag_len) / Q_r # Rotor slot pitch
        b_r        = b_r_tau_r * tau_r        # Rotor slot width
        b_tr       = tau_r - b_r              # Rotor tooth width
        tau_r_min  = pi * (ag_dia - 2 * (ag_len + h_r)) / Q_r
        b_trmin    = tau_r_min - b_r_tau_r * tau_r_min # minumum rotor tooth width
        
        
        # Calculating equivalent slot openings
        mu_rs       = 0.005
        mu_rr       = 0.005
        W_s         = (b_s / mu_rs) * 1e-3  # Stator, in m
        W_r         = (b_r / mu_rr) * 1e-3  # Rotor,  in m
        
        Slot_aspect_ratio1 = h_s / b_s      # Stator slot aspect ratio
        Slot_aspect_ratio2 = h_r / b_r      # Rotor slot aspect ratio
	                             
        # Calculating Carter factor for stator,rotor and effective air gap length
        '''
        gamma_s = (2 * W_s / ag_len)**2 / (5 + 2 * W_s / ag_len)
        K_Cs    = tau_s / (tau_s - ag_len * gamma_s * 0.5) # page 3-13 Boldea Induction machines Chapter 3
        gamma_r = (2 * W_r / ag_len)**2 / (5 + 2 * W_r / ag_len)
        K_Cr    = tau_r / (tau_r - ag_len * gamma_r * 0.5) # page 3-13 Boldea Induction machines Chapter 3
        '''
        
        K_Cs    = carterFactor(ag_len, W_s, tau_s)
        K_Cr    = carterFactor(ag_len, W_r, tau_r)
        K_C     = K_Cs * K_Cr
        g_eff   = K_C * ag_len
                  
        om_m    = 2 * pi * n_nom / 60     # mechanical frequency
        om_e    = p * om_m                # electrical frequency
        f       = n_nom * p / 60          # generator output freq
        K_s     = 0.3                     # saturation factor for Iron
        n_c     = 2                       # number of conductors per coil
        a1      = 2                       # number of parallel paths
        
        # Calculating stator winding turns
        N_s    = np.round(2 * p * N_slots_pp * n_c / a1)
        
        # Calculating Peak flux densities
        B_g1    = mu_0 * 3 * N_s * I_0 * sqrt(2) * k_y1 * k_q1 / (pi * p * g_eff * (1 + K_s))
        B_g     = B_g1 * K_C
        B_rymax = B_symax
        
        # calculating back iron thickness
        h_ys = B_g * tau_p / (B_symax * pi)
        h_yr = h_ys
        
        d_se    = ag_dia + 2 * (h_ys + h_s + h_w)  # stator outer diameter
        D_ratio = d_se / ag_dia                    # Diameter ratio
        
        # limits for Diameter ratio depending on pole pair
        if (2 * p == 2):
            D_ratio_LL = 1.65
            D_ratio_UL = 1.69
        elif (2 * p == 4):
            D_ratio_LL = 1.46
            D_ratio_UL = 1.49
        elif (2 * p == 6):
            D_ratio_LL = 1.37
            D_ratio_UL = 1.4
        elif (2 * p == 8):
            D_ratio_LL = 1.27
            D_ratio_UL = 1.3
        else:
            D_ratio_LL = 1.2
            D_ratio_UL = 1.24
            
        # Stator slot fill factor
        if ag_dia > 2:
            k_fills = 0.65
        else:
            k_fills = 0.4
        
        # Stator winding length and cross-section
        l_fs      = 2 * (0.015 + y_tau_p * tau_p / 2 / cos(radians(40))) + pi * h_s # end connection
        l_Cus     = 2 * N_s * (l_fs + len_s) / a1                                     # shortpitch
        A_s       = b_s *        (h_s - h_w)                                        # Slot area
        A_scalc   = b_s * 1000 * (h_s - h_w) * 1000                                 # Conductor cross-section (mm^2)
        A_Cus     = A_s     * q1 * p * k_fills / N_s                                # Conductor cross-section (m^2)        
        A_Cuscalc = A_scalc * q1 * p * k_fills / N_s
        
        # Stator winding resistance
        R_s          = l_Cus * resist_Cu / A_Cus
        
        # Calculating no-load voltage
        om_s        = n_nom * 2 * pi / 60                     # rated angular frequency            
        P_e         = machine_rating / (1 - S_N)              # Electrical power
        E_p         = om_s * N_s * k_wd * rad_ag * len_s * B_g1 * sqrt(2)
        
        S_GN        = (1.0 - S_N) * machine_rating # same as P_e?
        T_e         = p * S_GN / (2 * pi * freq * (1 - S_N))
        I_srated    = machine_rating / (3 * E_p * cofi)
        
        #Rotor design
        diff        = h_r - h_w
        A_bar       = b_r * diff                                # bar cross section
        Beta_skin   = sqrt(pi * mu_0 * freq / 2 / resist_Cu)    # coefficient for skin effect correction
        k_rm        = Beta_skin * h_r                           # coefficient for skin effect correction
        J_b         = 6e+06                                     # Bar current density
        K_i         = 0.864
        I_b         = 2 * m * N_s * k_wd * I_srated / Q_r       # bar current
        
        # Calculating bar resistance
        
        R_rb        = resist_Cu * k_rm * len_s / A_bar
        I_er        = I_b / (2 * sin(pi * p / Q_r))             # End ring current
        J_er        = 0.8 * J_b                                 # End ring current density
        A_er        = I_er / J_er                               # End ring cross-section
        b           = h_r                                       # End ring dimension
        a           = A_er / b                                  # End ring dimension
        D_er        = (rad_ag * 2 - 2 * ag_len) - 0.003         # End ring diameter
        l_er        = pi * (D_er - b) / Q_r                     # End ring segment length
        if debug:
            sys.stderr.write('l_er {:.4f} A_er {:.4f} D_er {:.4f}\n'.format(l_er[0], A_er[0], D_er[0]))
        
        # Calculating end ring resistance
        R_re = resist_Cu * l_er / (2 * A_er * (sin(pi * p / Q_r))**2)
        
        # Calculating equivalent rotor resistance
        if debug:
            sys.stderr.write('R_rb {:.3e} R_re {:.3e} k_wd {:.4f} N_s {} Q_r {}\n'.format(R_rb, R_re, k_wd, N_s, Q_r))
        R_R = (R_rb + R_re) * 4 * m * (k_wd * N_s)**2 / Q_r
        
        # Calculating Rotor and Stator teeth flux density
        B_trmax = B_g * tau_r / b_trmin
        B_tsmax = B_g * tau_s / b_t
        
        # Calculating Equivalent core lengths
        l_r  = len_s + 4     * ag_len   # for axial cooling
        l_se = len_s + (2/3) * ag_len
        K_fe = 0.95              # Iron factor
        L_e = l_se * K_fe        # radial cooling
        
        # Calculating leakage inductance in  stator
        if debug:
            sys.stderr.write('b_s {:.3e} b_so {:.3e}\n'.format(b_s[0], b_so[0]))
        L_ssigmas  = 2 * mu_0 * len_s * N_s**2 / p / q1 * ((h_s - h_w) / (3 * b_s) + h_w / b_so)                        # slot        leakage inductance
        L_ssigmaew = 2 * mu_0 * len_s * N_s**2 / p / q1 * 0.34 * q1 * (l_fs - 0.64 * tau_p * y_tau_p) / len_s           # end winding leakage inductance
        L_ssigmag  = 2 * mu_0 * len_s * N_s**2 / p / q1 * (5 * (ag_len * K_C / b_so) / (5 + 4 * (ag_len * K_C / b_so))) # tooth tip   leakage inductance
        L_s        = L_ssigmas + L_ssigmaew + L_ssigmag                                                                 # stator      leakage inductance
        L_sm       = 6 * mu_0 * len_s * tau_p * (k_wd * N_s)**2 / (pi**2 * p * g_eff * (1 + K_s))
        
        # Calculating leakage inductance in  rotor
        lambda_ei = 2.3 * D_er / (4 * Q_r * len_s * (sin(pi * p / Q_r)**2)) * log(4.7 * ag_dia / (a + 2 * b))
        lambda_b  = h_r / (3 * b_r) + h_w / b_ro
        L_i       = pi * ag_dia / Q_r
        
        L_rsl = mu_0 * len_s * ((h_r - h_w) / (3 * b_r) + h_w / b_ro)       # slot        leakage inductance
        L_rel = mu_0 * (len_s * lambda_b + 2 * lambda_ei * L_i)             # end winding leakage inductance
        L_rtl = mu_0 * len_s * (0.9 * tau_r * 0.09 / g_eff)                 # tooth tip   leakage inductance
        L_rsigma = (L_rsl + L_rtl + L_rel) * 4 * m * (k_wd * N_s)**2 / Q_r  # rotor       leakage inductance
        
        # Calculating rotor current
        if debug:
            sys.stderr.write('S_N {} P_e {:.1f} m {} R_R {:.4f} = {:.1f}\n'.format(S_N, P_e, m, R_R, -S_N * P_e / m / R_R))
        I_r = sqrt( -S_N * P_e / m / R_R)
        
        I_sm = E_p / (2 * pi * freq * L_sm)
        # Calculating stator currents and specific current loading
        I_s = sqrt((I_r**2 + I_sm**2))
        
        A_1 = 2 * m * N_s * I_s / (pi * 2 * rad_ag)
        
        # Calculating masses of the electromagnetically active materials
        
        V_Cuss = m * l_Cus * A_Cus                                                         # Volume of copper in stator
        V_Cusr = Q_r * len_s * A_bar + pi * (D_er * A_er - A_er * b)                         # Volume of copper in rotor
        V_Fest = len_s * pi * ((rad_ag + h_s)**2 - rad_ag**2) - 2 * m * q1 * p * b_s * h_s * len_s   # Volume of iron in stator teeth
        V_Fesy = len_s * pi * ((rad_ag + h_s + h_ys)**2 - (rad_ag + h_s)**2)                       # Volume of iron in stator yoke
        rad_r = rad_ag - ag_len                                                               # rotor radius
        
        V_Fert = pi * len_s * (rad_r**2 - (rad_r - h_r)**2) - 2 * m * q2 * p * b_r * h_r * len_s # Volume of iron in rotor teeth
        V_Fery = pi * len_s * ((rad_r - h_r)**2 - (rad_r - h_r - h_yr)**2)                     # Volume of iron in rotor yoke
        Copper = (V_Cuss + V_Cusr) * rho_Copper            # Mass of Copper
        M_Fest = V_Fest * rho_Fe                           # Mass of stator teeth
        M_Fesy = V_Fesy * rho_Fe                           # Mass of stator yoke
        M_Fert = V_Fert * rho_Fe                           # Mass of rotor tooth
        M_Fery = V_Fery * rho_Fe                           # Mass of rotor yoke
        Iron = M_Fest + M_Fesy + M_Fert + M_Fery
        
        Active_mass = Copper + Iron
        L_tot = len_s
        Structural_mass = 0.0001 * Active_mass**2 + 0.8841 * Active_mass - 132.5
        Mass = Active_mass + Structural_mass
        
        # Calculating Losses and efficiency
        
        # 1. Copper losses
        
        K_R = 1.2                        # skin effect correction coefficient
        P_Cuss = m * I_s**2 * R_s * K_R  # Copper loss - stator
        P_Cusr = m * I_r**2 * R_R        # Copper loss - rotor
        P_Cusnom = P_Cuss + P_Cusr       # Copper loss - total
        
        # Iron Losses ( from Hysteresis and eddy currents)          
        P_Hyys = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))                 # Hysteresis losses in stator yoke
        P_Ftys = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2)            # Eddy       losses in stator yoke
        P_Hyd  = M_Fest * (B_tsmax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))                 # Hysteresis losses in stator tooth
        P_Ftd  = M_Fest * (B_tsmax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2)            # Eddy       losses in stator tooth
        P_Hyyr = M_Fery * (B_rymax / 1.5)**2 * (P_Fe0h * abs(S_N) * om_e / (2 * pi * 60))      # Hysteresis losses in rotor yoke
        P_Ftyr = M_Fery * (B_rymax / 1.5)**2 * (P_Fe0e * (abs(S_N) * om_e / (2 * pi * 60))**2) # Eddy       losses in rotor yoke
        P_Hydr = M_Fert * (B_trmax / 1.5)**2 * (P_Fe0h * abs(S_N) * om_e / (2 * pi * 60))      # Hysteresis losses in rotor tooth
        P_Ftdr = M_Fert * (B_trmax / 1.5)**2 * (P_Fe0e * (abs(S_N) * om_e / (2 * pi * 60))**2) # Eddy       losses in rotor tooth
        
        # Calculating Additional losses
        P_add = 0.5 * machine_rating / 100
        P_Fesnom = P_Hyys + P_Ftys + P_Hyd + P_Ftd + P_Hyyr + P_Ftyr + P_Hydr + P_Ftdr
        Losses = P_Cusnom + P_Fesnom + P_add
        gen_eff = (P_e - Losses) * 100 / P_e
        Overall_eff = gen_eff * Gearbox_efficiency
        
        # Calculating current densities in the stator and rotor
        J_s = I_s / A_Cuscalc
        J_r = I_r / A_bar / 1e6
        
        # Calculating Tangential stress constraints
        TC1 = T_e / (2 * pi * sigma)
        TC2 = rad_ag**2 * len_s
        
        # Calculating mass moments of inertia and center of mass
        
        I = np.zeros(3)
        r_out = d_se * 0.5
        I[0]   = 0.50 * Mass * r_out**2
        I[1]   = 0.25 * Mass * r_out**2 + Mass * len_s**2 / 12
        I[2]   = I[1]
        cm = np.zeros(3)
        cm[0]  = shaft_cm[0] + shaft_length/2. + len_s/2.
        cm[1]  = shaft_cm[1]
        cm[2]  = shaft_cm[2]

        return B_tsmax, B_trmax, B_rymax, B_g, B_g1, q1, N_s, S, h_ys, b_s, b_t, D_ratio, D_ratio_UL, \
            D_ratio_LL, A_Cuscalc, Slot_aspect_ratio1, h_yr, tau_p, p, Q_r, b_r, b_trmin, b_tr, \
            rad_r, S_N, A_bar, Slot_aspect_ratio2, E_p, f, I_s, A_1, J_s, J_r, R_s, R_R, L_s, \
            L_sm, Mass, K_rad, K_rad_UL, K_rad_LL, Losses, gen_eff, Copper, Iron, Structural_mass, \
            TC1, TC2, Overall_eff, cm, I                  

'''        
        outputs = {}
        outputs['B_tsmax']            = B_tsmax
        outputs['B_trmax']            = B_trmax
        outputs['B_rymax']            = B_rymax
        outputs['B_g']                = B_g
        outputs['B_g1']               = B_g1
        outputs['q1']                 = q1
        outputs['N_s']                = N_s
        outputs['S']                  = S
        outputs['h_ys']               = h_ys
        outputs['b_s']                = b_s
        outputs['b_t']                = b_t
        outputs['D_ratio']            = D_ratio
        outputs['D_ratio_UL']         = D_ratio_UL
        outputs['D_ratio_LL']         = D_ratio_LL
        outputs['A_Cuscalc']          = A_Cuscalc
        outputs['Slot_aspect_ratio1'] = Slot_aspect_ratio1
        outputs['h_yr']               = h_yr
        outputs['tau_p']              = tau_p
        outputs['p']                  = p
        outputs['Q_r']                = Q_r
        outputs['b_r']                = b_r
        outputs['b_trmin']            = b_trmin
        outputs['b_tr']               = b_tr
        outputs['rad_r']              = rad_r
        outputs['S_N']                = S_N
        outputs['A_bar']              = A_bar
        outputs['Slot_aspect_ratio2'] = Slot_aspect_ratio2
        outputs['E_p']                = E_p
        outputs['f']                  = f
        outputs['I_s']                = I_s
        outputs['A_1']                = A_1
        outputs['J_s']                = J_s
        outputs['J_r']                = J_r
        outputs['R_s']                = R_s
        outputs['R_R']                = R_R
        outputs['L_s']                = L_s
        outputs['L_sm']               = L_sm
        outputs['Mass']               = Mass
        outputs['K_rad']              = K_rad
        outputs['K_rad_UL']           = K_rad_UL
        outputs['K_rad_LL']           = K_rad_LL
        outputs['Losses']             = Losses
        outputs['gen_eff']            = gen_eff
        outputs['Copper']             = Copper
        outputs['Iron']               = Iron
        outputs['Structural_mass']    = Structural_mass
        outputs['TC1']                = TC1
        outputs['TC2']                = TC2
        outputs['Overall_eff']        = Overall_eff
        outputs['cm']                 = cm
        outputs['I']                  = I
'''
 
#%%-----------------

        
if __name__ == '__main__':

    scig = SCIG()
    
    r_s                = 0.55    # meter
                                 
    len_s                = 1.30    # meter
    h_s                = 0.090   # meter
    h_r                = 0.050   # meter
    machine_rating     = 5000000.0
    n_nom              = 1200.0
    Gearbox_efficiency = 0.955        
    I_0                = 140     # Ampere
    rho_Fe             = 7700.0  # Steel density Kg/m3
    rho_Copper         = 8900.0  # copper density Kg/m3       
    B_symax            = 1.4     # Tesla
    shaft_cm           = np.array([0.0, 0.0, 0.0])
    shaft_length       = 2.0
        
    rad_ag = r_s
        
    outputs = scig.compute(rad_ag, len_s, h_s, h_r, machine_rating, n_nom, Gearbox_efficiency, I_0, 
                rho_Fe, rho_Copper, B_symax, shaft_cm, shaft_length)


      
