"""DFIG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.api import Group, Problem, ExplicitComponent,ExecComp,IndepVarComp,ScipyOptimizeDriver
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan


class DFIG(ExplicitComponent):
    """ Estimates overall mass, dimensions and Efficiency of DFIG generator. """
    
    def setup(self):
        # DFIG design inputs
        #self.add_input('r_s', val=0.0, units='m', desc='airgap radius r_s')
        self.add_input('rad_ag', val=0.0, units='m', desc='airgap radius')
        self.add_input('l_s', val=0.0, units='m', desc='Stator core length l_s')
        self.add_input('h_s', val=0.0, units='m', desc='Stator slot height')
        self.add_input('h_r', val=0.0, units='m', desc='Rotor slot height')
        
        self.add_input('machine_rating', val=0.0, units='W',   desc='Machine rating')
        self.add_input('n_nom',          val=0.0, units='rpm', desc='rated speed')
        
        self.add_input('B_symax', val=0.0,            desc='Peak Stator Yoke flux density B_symax')
        self.add_input('S_Nmax',  val=0.0, units='T', desc='Max rated Slip ')
        self.add_input('I_0',     val=0.0, units='A', desc='no-load excitation current')
        
        self.add_input('shaft_cm',           val= np.zeros(3), units='m', desc='Main Shaft CM')
        self.add_input('shaft_length',       val=0.0,          units='m', desc='main shaft length')
        self.add_input('Gearbox_efficiency', val=0.0,                     desc='Gearbox efficiency')
        
        # Material properties
        self.add_input('rho_Fe',     val=0.0, units='kg*m**-3', desc='Magnetic Steel density ')
        self.add_input('rho_Copper', val=0.0, units='kg*m**-3', desc='Copper density ')
        
        # DFIG generator design output
        # Magnetic loading
        self.add_output('B_g',     val=0.0, desc='Peak air gap flux density B_g')
        self.add_output('B_g1',    val=0.0, desc='air gap flux density fundamental ')
        self.add_output('B_rymax', val=0.0, desc='maximum flux density in rotor yoke')
        self.add_output('B_tsmax', val=0.0, desc='maximum tooth flux density in stator')
        self.add_output('B_trmax', val=0.0, desc='maximum tooth flux density in rotor')
        
        #Stator design
        self.add_output('N_s',                val=0.0, desc='Stator turns')
        self.add_output('S',                  val=0.0, desc='Stator slots')
        self.add_output('h_ys',               val=0.0, desc='Stator Yoke height')
        self.add_output('b_s',                val=0.0, desc='stator slot width')
        self.add_output('b_t',                val=0.0, desc='stator tooth width')
        self.add_output('A_Cuscalc',          val=0.0, desc='Stator Conductor cross-section mm^2')
        self.add_output('Slot_aspect_ratio1', val=0.0, desc='Stator slot aspect ratio')
        self.add_output('D_ratio',            val=0.0, desc='Stator diameter ratio')
        self.add_output('q1',                 val=0.0, desc='Slots per pole per phase')
        
        #Rotor design
        self.add_output('N_r',                val=0.0, desc='Rotor turns')
        self.add_output('Q_r',                val=0.0, desc='Rotor slots')
        self.add_output('h_yr',               val=0.0, desc='rotor yoke height')
        self.add_output('b_r',                val=0.0, desc='rotor slot width')
        self.add_output('b_tr',               val=0.0, desc='rotor tooth width')
        self.add_output('A_Curcalc',          val=0.0, desc='Rotor Conductor cross-section mm^2')
        self.add_output('Slot_aspect_ratio2', val=0.0, desc='Rotor slot aspect ratio')
        self.add_output('b_trmin',            val=0.0, desc='minimum tooth width')
        self.add_output('tau_p',              val=0.0, desc='Pole pitch')
        self.add_output('p',                  val=0.0, desc='No of pole pairs')
        
        # Electrical performance
        self.add_output('E_p',  val=0.0, desc='Stator phase voltage')
        self.add_output('f',    val=0.0, desc='Generator output frequency')
        self.add_output('I_s',  val=0.0, desc='Generator output phase current')
        self.add_output('A_1' , val=0.0, desc='Specific current loading')
        self.add_output('J_s',  val=0.0, desc='Stator winding Current density')
        self.add_output('J_r',  val=0.0, desc='Rotor winding Current density')
        self.add_output('R_s',  val=0.0, desc='Stator resistance')
        self.add_output('R_R',  val=0.0, desc='Rotor resistance')
        self.add_output('L_r',  val=0.0, desc='Rotor inductance')
        self.add_output('L_s',  val=0.0, desc='Stator synchronising inductance')
        self.add_output('L_sm', val=0.0, desc='mutual inductance')
        
        # Objective functions
        self.add_output('Mass',    val=0.0, desc='Actual mass')
        self.add_output('K_rad',   val=0.0, desc='Stack length ratio')
        self.add_output('Losses',  val=0.0, desc='Total loss')
        self.add_output('gen_eff', val=0.0, desc='Generator efficiency')
        
        # Mass Outputs
        self.add_output('Copper',          val=0.0, units='kg', desc='Copper Mass')
        self.add_output('Iron',            val=0.0, units='kg', desc='Electrical Steel Mass')
        self.add_output('Structural_mass', val=0.0, units='kg', desc='Structural Mass')
        
        # Structural performance
        self.add_output('TC1', val=0.0, desc='Torque constraint-stator')
        self.add_output('TC2', val=0.0, desc='Torque constraint-rotor')
        
        # Other parameters
        self.add_output('Current_ratio', val=0.0,         desc='Rotor current ratio')
        self.add_output('Overall_eff',   val=0.0,         desc='Overall drivetrain efficiency')
        self.add_output('I',             val=np.zeros(3), desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('cm',            val=np.zeros(3), desc='COM [x,y,z]')
        
        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs):
        # Unpack inputs
        rad_ag               = inputs['rad_ag']
        l_s                  = inputs['l_s']
        h_s                  = inputs['h_s']
        h_r                  = inputs['h_r']
        I_0                  = inputs['I_0']
        
        machine_rating       = inputs['machine_rating']
        n_nom                = inputs['n_nom']
        Gearbox_efficiency   = inputs['Gearbox_efficiency']
        
        rho_Fe               = inputs['rho_Fe']
        rho_Copper           = inputs['rho_Copper']
        
        B_symax              = inputs['B_symax']
        S_Nmax               = inputs['S_Nmax']
        
        shaft_cm             = inputs['shaft_cm']
        shaft_length         = inputs['shaft_length']
        
        #Assign values to universal constants
        g1          = 9.81             # m/s^2 acceleration due to gravity
        sigma       = 21.5e3           # shear stress
        mu_0        = pi * 4e-7        # permeability of free space in m * kg / (s**2 * A**2)
        cofi        = 0.9              # power factor
        h_w         = 0.005            # wedge height
        m           = 3                # Number of phases
        resist_Cu   = 1.8e-8 * 1.4     # copper resisitivity
        h_sy0       = 0
        
        #Assign values to design constants
        b_so = 0.004                    # Stator slot opening width
        b_ro = 0.004                    # Rotor  slot opening width
        q1 = 5                          # Stator slots per pole per phase
        q2 = q1 - 1                     # Rotor  slots per pole per phase
        b_s_tau_s = 0.45                # Stator slot-width / slot-pitch ratio
        b_r_tau_r = 0.45                # Rotor  slot-width / slot-pitch ratio
        y_tau_p  = 12. / 15             # Stator coil span to pole pitch    
        y_tau_pr = 10. / 12             # Rotor  coil span to pole pitch
        
        p = 3                           # pole pairs
        freq = 60                       # grid frequency in Hz
        k_fillr = 0.55                  # Rotor Slot fill factor
        k_fills = 0.65                  # Stator Slot fill factor (will be set later)
        P_Fe0h = 4                      # specific hysteresis losses W / kg @ 1.5 T 
        P_Fe0e = 1                      # specific eddy losses W / kg @ 1.5 T 
        
        K_rs     = 1 / ( -1 * S_Nmax)              # Winding turns ratio between rotor and Stator        
        I_SN   = machine_rating / (sqrt(3) * 3000) # Rated current
        I_SN_r = I_SN / K_rs                       # Stator rated current reduced to rotor
        
        # Calculating winding factor for stator and rotor    
        
        k_y1 = sin(pi/2 * y_tau_p)                              # winding chording factor
        k_q1 = sin(pi/6) / (q1 * sin(pi/(6 * q1)))              # winding zone factor
        k_y2 = sin(pi/2 * y_tau_pr)                             # winding chording factor
        k_q2 = sin(pi/6) / (q2 * sin(pi/(6 * q2)))              # winding zone factor
        k_wd1 = k_y1 * k_q1                                     # Stator winding factor
        k_wd2 = k_q2 * k_y2                                     # Rotor winding factor
        
        ag_dia = 2 * rad_ag                                     # air gap diameter
        ag_len = (0.1 + 0.012 * machine_rating**(1./3)) * 0.001 # air gap length in m
        K_rad = l_s / ag_dia                                    # Aspect ratio
        rad_r = rad_ag - ag_len                                 # rotor radius  (was r_r)
        tau_p = pi * ag_dia / (2 * p)                           # pole pitch
        
        S = 2 * p * q1 * m                                      # Stator slots
        N_slots_pp = S / (m * p * 2)                            # Number of stator slots per pole per phase
        n  = S / 2 * p / q1                                     # no of slots per pole per phase
        tau_s = tau_p / (m * q1)                                # Stator slot pitch
        b_s = b_s_tau_s * tau_s                                 # Stator slot width
        b_t = tau_s - b_s                                       # Stator tooth width
        
        Q_r = 2 * p * m * q2                                    # Rotor slots
        tau_r = pi * (ag_dia - 2 * ag_len) / Q_r                # Rotor slot pitch
        b_r = b_r_tau_r * tau_r                                 # Rotor slot width
        b_tr = tau_r - b_r                                      # Rotor tooth width
        
        # Calculating equivalent slot openings
        
        mu_rs = 0.005
        mu_rr = 0.005
        W_s   = (b_s / mu_rs) * 1e-3  # Stator, in m
        W_r   = (b_r / mu_rr) * 1e-3  # Rotor,  in m
        
        Slot_aspect_ratio1 = h_s / b_s
        Slot_aspect_ratio2 = h_r / b_r
        
        # Calculating Carter factor for stator,rotor and effective air gap length
        
        gamma_s = (2 * W_s / ag_len)**2 / (5 + 2 * W_s / ag_len)
        K_Cs    = tau_s / (tau_s - ag_len * gamma_s * 0.5)  #page 3 - 13
        gamma_r = (2 * W_r / ag_len)**2 / (5 + 2 * W_r / ag_len)
        K_Cr    = tau_r / (tau_r - ag_len * gamma_r * 0.5)  #page 3 - 13
        K_C     = K_Cs * K_Cr
        g_eff   = K_C * ag_len
        
        om_m = 2 * pi * n_nom / 60                      # mechanical frequency
        om_e = p * om_m                                 # electrical frequency
        f = n_nom * p / 60                              # generator output freq
        K_s = 0.3                                       # saturation factor for Iron
        n_c1 = 2                                        # number of conductors per coil
        a1 = 2                                          # number of parallel paths
        N_s = np.round(2 * p * N_slots_pp * n_c1 / a1)  # Stator winding turns per phase        
        N_r = np.round(N_s * k_wd1 * K_rs / k_wd2)      # Rotor winding turns per phase
        n_c2 = N_r / (Q_r / m)                          # rotor turns per coil
        
        # Calculating peak flux densities and back iron thickness
        
        B_g1 = mu_0 * 3 * N_r * I_0 * sqrt(2) * k_y2 * k_q2 / (pi * p * g_eff * (1 + K_s))
        B_g = B_g1 * K_C
        h_ys = B_g * tau_p / (B_symax * pi)
        B_rymax = B_symax
        h_yr = h_ys
        B_tsmax = B_g * tau_s / b_t 
        
        d_se = ag_dia + 2 * (h_ys + h_s + h_w)           # stator outer diameter
        D_ratio = d_se / ag_dia                          # Diameter ratio
        
        # Stator slot fill factor
        if ag_dia > 2:
            k_fills = 0.65
        else:
            k_fills = 0.4
            
        # Stator winding calculation
        
        # End connection length for stator winding coils
        
        l_fs = 2 * (0.015 + y_tau_p * tau_p / (2 * cos(radians(40)))) + pi * h_s   # added radians() 2019 09 11
        
        l_Cus = 2 * N_s * (l_fs + l_s) / a1             # Length of Stator winding 
        
        # Conductor cross-section
        A_s       = b_s *        (h_s - h_w)
        A_scalc   = b_s * 1000 * (h_s - h_w) * 1000
        A_Cus     = A_s     * q1 * p * k_fills / N_s
        A_Cuscalc = A_scalc * q1 * p * k_fills / N_s
        
        # Stator winding resistance
        
        R_s = l_Cus * resist_Cu / A_Cus
        tau_r_min = pi * (ag_dia - 2 * (ag_len + h_r)) / Q_r
        
        # Peak magnetic loading on the rotor tooth
        
        b_trmin = tau_r_min - b_r_tau_r * tau_r_min
        B_trmax = B_g * tau_r / b_trmin
        
        # Calculating leakage inductance in  stator
        
        K_01 = 1 - 0.033 * (W_s**2 / ag_len / tau_s)
        sigma_ds = 0.0042
        
        L_ssigmas  = (2 * mu_0 * l_s * n_c1**2 * S / m / a1**2) * ((h_s - h_w) / (3 * b_s) + h_w / b_so)                # slot leakage inductance
        L_ssigmaew = (2 * mu_0 * l_s * n_c1**2 * S / m / a1**2) * 0.34 * q1 * (l_fs - 0.64 * tau_p * y_tau_p) / l_s     # end winding leakage inductance
        L_ssigmag  = (2 * mu_0 * l_s * n_c1**2 * S / m / a1**2) * (0.9 * tau_s * q1 * k_wd1 * K_01 * sigma_ds / g_eff)  # tooth tip leakage inductance
        L_ssigma   = (L_ssigmas + L_ssigmaew + L_ssigmag)                                                               # stator leakage inductance
        L_sm       = 6 * mu_0 * l_s * tau_p * (k_wd1 * N_s)**2 / (pi**2 * (p) * g_eff * (1 + K_s))                      
        L_s        = (L_ssigmas + L_ssigmaew + L_ssigmag)                                                               # stator  inductance
        
        # Calculating leakage inductance in  rotor
        
        K_02 = 1 - 0.033 * (W_r**2 / ag_len / tau_r)
        sigma_dr = 0.0062
        
        l_fr = (0.015 + y_tau_pr * tau_r / 2 / cos(radians(40))) + pi * h_r                                   # Rotor end connection length
        L_rsl = (mu_0 * l_s * (2 * n_c2)**2 * Q_r / m) * ((h_r - h_w) / (3 * b_r) + h_w / b_ro)               # slot leakage inductance
        L_rel = (mu_0 * l_s * (2 * n_c2)**2 * Q_r / m) * 0.34 * q2 * (l_fr - 0.64 * tau_r * y_tau_pr) / l_s   # end winding leakage inductance
        L_rtl = (mu_0 * l_s * (2 * n_c2)**2 * Q_r / m) * (0.9 * tau_s * q2 * k_wd2 * K_02 * sigma_dr / g_eff) # tooth tip leakage inductance
        L_r = (L_rsl + L_rtl + L_rel) / K_rs**2                                                               # rotor leakage inductance
        sigma1 = 1 - (L_sm**2 / L_s / L_r)
        
        #Rotor Field winding
        
        # conductor cross-section
        diff = h_r - h_w
        A_Cur = k_fillr * p*q2 * b_r * diff / N_r
        A_Curcalc = A_Cur * 1e6
        
        L_cur = 2 * N_r * (l_fr + l_s)     # rotor winding length
        Resist_r = resist_Cu * L_cur / A_Cur    # Rotor resistance
        R_R = Resist_r / K_rs**2                # Equivalent rotor resistance reduced to stator
        
        om_s = n_nom * 2*pi / 60                     # synchronous speed in rad / s
        P_e = machine_rating / (1 - S_Nmax)          # Air gap power
        
        # Calculating No-load voltage
        E_p      = om_s * N_s * k_wd1 * rad_ag * l_s * B_g1 * sqrt(2)
        I_r      = P_e / m / E_p                        # rotor active current        
        I_sm     = E_p / (2 * pi * freq * (L_s + L_sm)) # stator reactive current
        I_s      = sqrt(I_r**2 + I_sm**2)               # Stator current
        I_srated = machine_rating / 3 / K_rs / E_p      # Rated current
        
        # Calculating winding current densities and specific current loading
        
        J_s = I_s / A_Cuscalc
        J_r = I_r / A_Curcalc
        A_1 = 2 * m*N_s * I_s / (pi * 2 * rad_ag)
        Current_ratio = I_0 / I_srated           # Ratio of magnetization current to rated current
        
        # Calculating masses of the electromagnetically active materials
        
        V_Cuss = m * l_Cus * A_Cus
        V_Cusr = m * L_cur * A_Cur
        V_Fest = l_s * pi * ((rad_ag + h_s)**2 - rad_ag**2) - (2 * m*q1 * p*b_s * h_s * l_s)
        V_Fesy = l_s * pi * ((rad_ag + h_s + h_ys)**2    - (rad_ag + h_s)**2)
        V_Fert = l_s * pi * (rad_r**2 - (rad_r - h_r)**2) - 2 * m*q2 * p*b_r * h_r * l_s
        V_Fery = l_s * pi * ((rad_r - h_r)**2           - (rad_r - h_r - h_yr)**2)
        Copper = (V_Cuss + V_Cusr) * rho_Copper
        M_Fest = V_Fest * rho_Fe
        M_Fesy = V_Fesy * rho_Fe
        M_Fert = V_Fert * rho_Fe
        M_Fery = V_Fery * rho_Fe
        Iron = M_Fest + M_Fesy + M_Fert + M_Fery
        M_gen = (Copper) + (Iron)
        
        #K_gen = Cu * C_Cu + (Iron) * C_Fe #%M_pm * K_pm
        
        L_tot = l_s
        Structural_mass = 0.0002 * M_gen**2 + 0.6457 * M_gen + 645.24
        Mass = M_gen + Structural_mass
        
        # Calculating Losses and efficiency 
        # 1. Copper losses
        
        K_R = 1.2 # skin effect correction coefficient
        P_Cuss = m * I_s**2 * R_s * K_R            # Copper loss - stator
        P_Cusr = m * I_r**2 * R_R                  # Copper loss - rotor
        P_Cusnom = P_Cuss + P_Cusr                 # Copper loss - total
        
        # Iron Losses ( from Hysteresis and eddy currents)      
        P_Hyys = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))                    # Hysteresis losses in stator yoke
        P_Ftys = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2)               # Eddy losses in stator yoke
        P_Hyd  = M_Fest * (B_tsmax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))                    # Hysteresis losses in stator teeth
        P_Ftd  = M_Fest * (B_tsmax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2)               # Eddy losses in stator teeth
        P_Hyyr = M_Fery * (B_rymax / 1.5)**2 * (P_Fe0h * abs(S_Nmax) * om_e / (2 * pi * 60))      # Hysteresis losses in rotor yoke
        P_Ftyr = M_Fery * (B_rymax / 1.5)**2 * (P_Fe0e * (abs(S_Nmax) * om_e / (2 * pi * 60))**2) # Eddy losses in rotor yoke
        P_Hydr = M_Fert * (B_trmax / 1.5)**2 * (P_Fe0h * abs(S_Nmax) * om_e / (2 * pi * 60))      # Hysteresis losses in rotor teeth
        P_Ftdr = M_Fert * (B_trmax / 1.5)**2 * (P_Fe0e * (abs(S_Nmax) * om_e / (2 * pi * 60))**2) # Eddy losses in rotor teeth
        P_add = 0.5 * machine_rating / 100                                                        # additional losses
        P_Fesnom = P_Hyys + P_Ftys + P_Hyd + P_Ftd + P_Hyyr + P_Ftyr + P_Hydr + P_Ftdr            # Total iron loss
        delta_v = 1                                                                               # allowable brush voltage drop
        p_b = 3 * delta_v * I_r                                                                   # Brush loss
        
        Losses = P_Cusnom + P_Fesnom + p_b + P_add
        gen_eff = (P_e - Losses) * 100 / P_e
        Overall_eff = gen_eff * Gearbox_efficiency
        
        # Calculating stator winding current density
        J_s = I_s / A_Cuscalc
        
        # Calculating  electromagnetic torque
        T_e = p *(machine_rating * 1.01) / (2 * pi * freq * (1 - S_Nmax))
        
        # Calculating for tangential stress constraints
        
        TC1 = T_e / (2 * pi * sigma)
        TC2 = rad_ag**2 * l_s
        
        # Calculating mass moments of inertia and center of mass
        I = np.zeros(3)
        r_out = d_se * 0.5
        I[0]   = (0.5 * Mass * r_out**2)
        I[1]   = (0.25 * Mass * r_out**2 + Mass * l_s**2 / 12) 
        I[2]   = I[1]
        cm = np.zeros(3)
        cm[0]  = shaft_cm[0] + shaft_length/2. + l_s/2.
        cm[1]  = shaft_cm[1]
        cm[2]  = shaft_cm[2]

        outputs['B_g']                = B_g
        outputs['B_g1']               = B_g1
        outputs['B_rymax']            = B_rymax
        outputs['B_tsmax']            = B_tsmax
        outputs['B_trmax']            = B_trmax

        outputs['q1']                 = q1
        outputs['N_s']                = N_s
        outputs['S']                  = S
        outputs['h_ys']               = h_ys
        outputs['b_s']                = b_s
        outputs['b_t']                = b_t

        outputs['D_ratio']            = D_ratio
        outputs['A_Cuscalc']          = A_Cuscalc
        outputs['Slot_aspect_ratio1'] = Slot_aspect_ratio1
        outputs['h_yr']               = h_yr

        outputs['tau_p']              = tau_p
        outputs['p']                  = p
        outputs['Q_r']                = Q_r
        outputs['N_r']                = N_r
        outputs['b_r']                = b_r
        outputs['b_trmin']            = b_trmin

        outputs['b_tr']               = b_tr
        outputs['A_Curcalc']          = A_Curcalc
        outputs['Slot_aspect_ratio2'] = Slot_aspect_ratio2
        outputs['E_p']                = E_p
        outputs['f']                  = f
        
        outputs['I_s']                = I_s
        outputs['A_1']                = A_1
        outputs['J_s']                = J_s
        outputs['J_r']                = J_r
        outputs['R_s']                = R_s
        outputs['R_R']                = R_R

        outputs['L_r']                = L_r
        outputs['L_s']                = L_s
        outputs['L_sm']               = L_sm
        outputs['Mass']               = Mass
        outputs['K_rad']              = K_rad
        outputs['Losses']             = Losses

        outputs['gen_eff']            = gen_eff
        outputs['Copper']             = Copper
        outputs['Iron']               = Iron
        outputs['Structural_mass']    = Structural_mass
        outputs['TC1']                = TC1
        outputs['TC2']                = TC2

        outputs['Current_ratio']      = Current_ratio
        outputs['Overall_eff']        = Overall_eff
        outputs['cm']                 = cm
        outputs['I']                  =  I
        
        
