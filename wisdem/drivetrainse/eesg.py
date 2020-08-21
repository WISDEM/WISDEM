"""EESG.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

from openmdao.api import Group, Problem, ExplicitComponent,ExecComp,IndepVarComp,ScipyOptimizeDriver
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan
import sys, os

class EESG(ExplicitComponent):
    """ Estimates overall mass dimensions and Efficiency of Electrically Excited Synchronous generator. """
    
    def setup(self):
        # EESG generator design inputs
        #self.add_input('r_s', val=0.0, units ='m', desc='airgap radius r_s')
        self.add_input('rad_ag', val=0.0, units ='m', desc='airgap radius')
        self.add_input('l_s', val=0.0, units ='m', desc='Stator core length l_s')
        self.add_input('h_s', val=0.0, units ='m', desc='Yoke height h_s')
        self.add_input('tau_p',val=0.0, units ='m', desc='Pole pitch self.tau_p')
        
        self.add_input('machine_rating',val=0.0, units ='W', desc='Machine rating')
        self.add_input('n_nom',val=0.0, units ='rpm', desc='rated speed')
        self.add_input('Torque',val=0.0, units ='N*m', desc='Rated torque ')
        self.add_input('I_f',val=0.0000,units='A',desc='Excitation current')
        self.add_input('N_f',val=0.0,units='A',desc='field turns')
        self.add_input('h_ys',val=0.0, units ='m', desc='Yoke height')
        self.add_input('h_yr',val=0.0, units ='m', desc='rotor yoke height')
        
        # structural design variables
        self.add_input('n_s' ,val=0.0, desc='number of stator arms n_s')
        self.add_input('b_st' , val=0.0, units ='m', desc='arm width b_st')
        self.add_input('d_s',val=0.0,units ='m', desc='arm depth d_s')
        self.add_input('t_ws' ,val=0.0,units ='m', desc='arm depth thickness self.t_wr')
        self.add_input('n_r' ,val=0.0, desc='number of arms n')
        self.add_input('b_r' ,val=0.0,units ='m', desc='arm width b_r')
        self.add_input('d_r' ,val=0.0, units ='m', desc='arm depth d_r')
        self.add_input('t_wr' ,val=0.0, units ='m', desc='arm depth thickness self.t_wr')
        self.add_input('R_o',val=0.0, units ='m',desc='Shaft radius')
        
        # EESG generator design outputs
        
        # Magnetic loading
        self.add_output('B_symax' ,val=0.0, desc='Peak Stator Yoke flux density B_ymax')
        self.add_output('B_tmax',val=0.0, desc='Peak Teeth flux density')
        self.add_output('B_rymax',val=0.0, desc='Peak Rotor yoke flux density')
        self.add_output('B_gfm',val=0.0, desc='Average air gap flux density B_g')
        self.add_output('B_g' ,val=0.0, desc='Peak air gap flux density B_g')
        self.add_output('B_pc',val=0.0, desc='Pole core flux density')
        
        # Stator design
        self.add_output('N_s' ,val=0.0, desc='Number of turns in the stator winding')
        self.add_output('b_s',val=0.0, desc='slot width')
        self.add_output('b_t',val=0.0, desc='tooth width')
        self.add_output('A_Cuscalc',val=0.0, desc='Conductor cross-section mm^2')
        self.add_output('S',val=0.0, desc='Stator slots')
        
        # # Output parameters : Rotor design
        self.add_output('h_p',val=0.0, desc='Pole height')
        self.add_output('b_p',val=0.0, desc='Pole width')
        self.add_output('p',val=0.0, desc='No of pole pairs')
        self.add_output('n_brushes',val=0.0, desc='number of brushes')
        self.add_output('A_Curcalc',val=0.0, desc='Rotor Conductor cross-section')
        
        # Output parameters : Electrical performance
        self.add_output('E_s',val=0.0, desc='Stator phase voltage')
        self.add_output('f',val=0.0, desc='Generator output frequency')
        self.add_output('I_s',val=0.0, desc='Generator output phase current')
        self.add_output('R_s',val=0.0, desc='Stator resistance')
        self.add_output('R_r',val=0.0, desc='Rotor resistance')
        self.add_output('L_m',val=0.0, desc='Stator synchronising inductance')
        self.add_output('J_s',val=0.0, desc='Stator Current density')
        self.add_output('J_f',val=0.0, desc='rotor Current density')
        self.add_output('A_1',val=0.0, desc='Specific current loading')
        self.add_output('Load_mmf_ratio',val=0.0, desc='mmf_ratio')
        
        # Objective functions and output
        self.add_output('Mass',val=0.0, desc='Actual mass')
        self.add_output('K_rad',val=0.0, desc='K_rad')
        self.add_output('Losses',val=0.0, desc='Total loss')
        self.add_output('gen_eff',val=0.0, desc='Generator efficiency')
        
        # Structural performance
        self.add_output('u_Ar',val=0.0, desc='Rotor radial deflection')
        self.add_output('y_Ar',val=0.0, desc='Rotor axial deflection')
        self.add_output('z_A_r',val=0.0, desc='Rotor circumferential deflection')
        self.add_output('u_As',val=0.0, desc='Stator radial deflection')
        self.add_output('y_As',val=0.0, desc='Stator axial deflection')
        self.add_output('z_A_s',val=0.0, desc='Stator circumferential deflection')  
        self.add_output('u_all_r',val=0.0, desc='Allowable radial rotor')
        self.add_output('u_all_s',val=0.0, desc='Allowable radial stator')
        self.add_output('y_all',val=0.0, desc='Allowable axial')
        self.add_output('z_all_s',val=0.0, desc='Allowable circum stator')
        self.add_output('z_all_r',val=0.0, desc='Allowable circum rotor')
        self.add_output('b_all_s',val=0.0, desc='Allowable arm')
        self.add_output('b_all_r',val=0.0, desc='Allowable arm dimensions')
        self.add_output('TC1',val=0.0, desc='Torque constraint')
        self.add_output('TC2',val=0.0, desc='Torque constraint-rotor')
        self.add_output('TC3',val=0.0, desc='Torque constraint-stator')
        
        # Material properties
        self.add_input('rho_Fes',val=0.0,units='kg*m**-3', desc='Structural Steel density ')
        self.add_input('rho_Fe',val=0.0,units='kg*m**-3', desc='Magnetic Steel density ')
        self.add_input('rho_Copper',val=0.0,units='kg*m**-3', desc='Copper density ')
        
        # Mass Outputs
        self.add_output('Copper', val=0.0, units='kg', desc='Copper Mass')
        self.add_output('Iron', val=0.0, units='kg', desc='Electrical Steel Mass')
        self.add_output('Structural_mass', val=0.0, units='kg', desc='Structural Mass')
        
        # Other parameters
        self.add_output('Power_ratio',val=0.0, desc='Power_ratio')
        self.add_output('Slot_aspect_ratio',val=0.0,desc='Stator slot aspect ratio')
        self.add_output('R_out',val=0.0, desc='Outer radius')
        
        # inputs/outputs for interface with drivese
        self.add_input('shaft_cm',val= np.zeros(3), units='m', desc='Main Shaft CM')
        self.add_input('shaft_length',val=0.0, units='m', desc='main shaft length')
        self.add_output('I',val=np.zeros(3),desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('cm', val=np.zeros(3),desc='COM [x,y,z]')

        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        
    def compute(self, inputs, outputs):
        # Unpack outputs
        rad_ag               = inputs['rad_ag']
        l_s               = inputs['l_s']
        h_s               = inputs['h_s']
        tau_p             = inputs['tau_p']
        N_f               = inputs['N_f']
        I_f               = inputs['I_f']
        h_ys              = inputs['h_ys']
        h_yr              = inputs['h_yr']
        machine_rating    = inputs['machine_rating']
        n_nom             = inputs['n_nom']
        Torque            = inputs['Torque']
        
        b_st              = inputs['b_st']
        d_s               = inputs['d_s']
        t_ws              = inputs['t_ws']
        n_r               = inputs['n_r']
        n_s               = inputs['n_s']
        b_r               = inputs['b_r']
        d_r               = inputs['d_r']
        t_wr              = inputs['t_wr']
        
        R_o               = inputs['R_o']
        rho_Fe            = inputs['rho_Fe']
        rho_Copper        = inputs['rho_Copper']
        rho_Fes           = inputs['rho_Fes']
        shaft_cm          = inputs['shaft_cm']
        shaft_length      = inputs['shaft_length']
        
        # Assign values to universal constants
        g1     = 9.81                # m / s^2 acceleration due to gravity
        E      = 2e11                # N / m^2 young's modulus
        sigma  = 48.373e3            # shear stress
        mu_0   = pi * 4e-7           # permeability of free space
        phi    = radians(90)
        
        # Assign values to design constants
        h_w       = 0.005
        b_so      = 0.004              # Stator slot opening
        m         = 3                  # number of phases
        q1        = 2                  # no of stator slots per pole per phase
        b_s_tau_s = 0.45               # ratio of slot width to slot pitch
        k_sfil    = 0.65               # Slot fill factor (not used)
        P_Fe0h    = 4                  # specific hysteresis losses W / kg @ 1.5 T @50 Hz
        P_Fe0e    = 1                  # specific eddy losses W / kg @ 1.5 T @50 Hz
        rho_Cu    = 1.8e-8 * 1.4       # resisitivity of copper
        k_fes     = 0.9                # iron fill factor (not used)
        y_tau_p   = 1                  # coil span / pole pitch fullpitch
        k_fillr   = 0.7                # rotor slot fill factor
        k_s       = 0.2                # magnetic saturation factor for iron
        T         = Torque
        cos_phi   = 0.85               # power factor
        
        # back iron thickness for rotor and stator
        t_s = h_ys
        t   = h_yr
        
        # Aspect ratio
        K_rad = l_s / (2 * rad_ag)
        
        ###################################################### Electromagnetic design#############################################
        
        alpha_p = pi / 2 * .7 # (not used)
        dia = 2 * rad_ag             # air gap diameter
        
        # air gap length and minimum values
        g = 0.001 * dia       
        if(g < 0.005):
            g = 0.005
            
        r_r = rad_ag - g                             # rotor radius
        d_se = dia + 2 * h_s + 2 * h_ys           # stator outer diameter (not used)
        p = np.round(pi * dia / (2 * tau_p))      # number of pole pairs
        S = 2 * p*q1 * m                          # number of slots of stator phase winding
        N_conductors = S * 2
        N_s = N_conductors / 2/3                  # Stator turns per phase
        alpha = 180 / S/p                         # electrical angle (not used)
        
        tau_s = pi * dia / S                      # slot pitch
        h_ps = 0.1 * tau_p                        # height of pole shoe
        b_pc = 0.4 * tau_p                        # width of pole core
        h_pc = 0.6 * tau_p                        # height of pole core
        h_p = 0.7 * tau_p                         # pole height
        b_p = h_p
        b_s = tau_s * b_s_tau_s                   # slot width
        Slot_aspect_ratio = h_s / b_s
        b_t = tau_s - b_s                         # tooth width
        
        # Calculating carter factor and effective air gap
        g_a = g
        K_C1 = (tau_s + 10 * g_a) / (tau_s - b_s + 10 * g_a)  # salient pole rotor
        g_1 = K_C1 * g
        
        # calculating angular frequency
        om_m = 2 * pi * n_nom / 60
        om_e = 60
        f    =  n_nom * p / 60
        
        # Slot fill factor according to air gap radius
        
        if (2 * rad_ag>2):
            K_fills = 0.65
        else:
            K_fills = 0.4
            
        # Calculating Stator winding factor    
        
        k_y1 = sin(y_tau_p * pi / 2)                        # chording factor
        k_q1 = sin(pi / 6) / q1 / sin(pi / 6 / q1)          # winding zone factor
        k_wd = k_y1 * k_q1
        
        # Calculating stator winding conductor length, cross-section and resistance
        
        shortpitch = 0
        l_Cus = 2 * N_s * (2 * (tau_p - shortpitch / m/q1) + l_s)   # length of winding
        A_s       = b_s        * (h_s - h_w)
        A_scalc   = b_s * 1000 * (h_s - h_w) * 1000            # cross section in mm^2
        A_Cus     = A_s     * q1 * p * K_fills / N_s
        A_Cuscalc = A_scalc * q1 * p * K_fills / N_s
        R_s = l_Cus * rho_Cu / A_Cus
        
        # field winding design, conductor lenght, cross-section and resistance
        
        N_f       = np.round(N_f)            # rounding the field winding turns to the nearest integer
        I_srated  = machine_rating / (sqrt(3) * 5000 * cos_phi)
        l_pole    = l_s - 0.05 + 0.120  # 50mm smaller than stator and 120mm longer to accommodate end stack
        K_fe      = 0.95                    
        l_pfe     = l_pole * K_fe
        l_Cur     = 4 * p*N_f * (l_pfe + b_pc + pi / 4 * (pi * (r_r - h_pc - h_ps) / p - b_pc))
        A_Cur     = k_fillr * h_pc        * 0.5 / N_f * (pi * (r_r - h_pc - h_ps)        / p - b_pc)
        A_Curcalc = k_fillr * h_pc * 1000 * 0.5 / N_f * (pi * (r_r - h_pc - h_ps) * 1000 / p - b_pc * 1000)
        Slot_Area = A_Cur * 2 * N_f / k_fillr # (not used)
        R_r = rho_Cu * l_Cur / A_Cur
        
        # field winding current density
        
        J_f = I_f / A_Curcalc
        
        # calculating air flux density
        
        B_gfm = mu_0 * N_f * I_f / (g_1 * (1 + k_s))  # No load air gap flux density
        
        B_g = B_gfm * 4*sin(0.5 * b_p * pi / tau_p) / pi  # fundamental component
        B_symax = tau_p * B_g / pi / h_ys                 # stator yoke flux density
        L_fg = 2 * mu_0 * p*l_s * 4*N_f**2 * ((h_ps / (tau_p - b_p)) + (h_pc / (3 * pi * (r_r - h_pc - h_ps) / p - b_pc))) #  (not used)
        
        # calculating no-load voltage and stator current
        
        E_s = 2 * N_s * l_s * rad_ag * k_wd * om_m * B_g / sqrt(2) # no-load voltage
        I_s = (E_s - (E_s**2 - 4 * R_s * machine_rating / m)**0.5) / (2 * R_s)
        
        # Calculating stator winding current density and specific current loading
        
        A_1 = 6 * N_s * I_s / (pi * dia)
        J_s = I_s / A_Cuscalc
        
        # Calculating magnetic loading in other parts of the machine
        
        delta_m = 0  # Initialising load angle
        
        # peak flux density in pole core, rotor yoke and stator teeth
        
        B_pc = (1 / b_pc) * ((2 * tau_p / pi) * B_g * cos(delta_m) + (2 * mu_0 * I_f * N_f * ((2 * h_ps / (tau_p - b_p)) + (h_pc / (tau_p - b_pc)))))
        B_rymax = 0.5 * b_pc * B_pc / h_yr
        B_tmax = (B_gfm + B_g) * tau_s * 0.5 / b_t
        
        # Calculating leakage inductances in the stator
        
        L_ssigmas = 2 * mu_0 * l_s * N_s**2 / p / q1 * ((h_s - h_w) / (3 * b_s) + h_w / b_so)  # slot leakage inductance
        L_ssigmaew =    mu_0 * 1.2 * N_s**2 / p * 1.2 * (2 / 3 * tau_p + 0.01)                 # end winding leakage inductance
        L_ssigmag = 2 * mu_0 * l_s * N_s**2 / p / q1 * (5 * (g / b_so) / (5 + 4 * (g / b_so))) # tooth tip leakage inductance
        L_ssigma = (L_ssigmas + L_ssigmaew + L_ssigmag)  # stator leakage inductance
        
        # Calculating effective air gap
        
        '''
        What is the source of this function that combines 1st and 13th powers? Very suspicious...
        Inputs appear to be in the range of 0.45 to 2.2, so outputs are 180 to 178000
        
        def airGapFn(B, fact):
            val = 400 * B + 7 * B**13
            ans = val * fact
            sys.stderr.write('aGF: B {} val {} ans {}\n'.format(B, val, ans))
            return val
        
        At_t = h_s * airGapFn(B_tmax, h_s)
        At_sy = tau_p * 0.5 * airGapFn(B_symax, tau_p/2)
        At_pc = (h_pc + h_ps) * airGapFn(B_pc, h_pc + h_ps)
        At_ry = tau_p * 0.5 * airGapFn(B_rymax, tau_p/2)
        '''
        At_g  = g_1 * B_gfm / mu_0
        At_t  = h_s           * (400 * B_tmax  + 7 * B_tmax**13)
        At_sy = tau_p * 0.5   * (400 * B_symax + 7 * B_symax**13)
        At_pc = (h_pc + h_ps) * (400 * B_pc    + 7 * B_pc**13)
        At_ry = tau_p * 0.5   * (400 * B_rymax + 7 * B_rymax**13)
        g_eff = (At_g + At_t + At_sy + At_pc + At_ry) * g_1 / At_g
        
        L_m = 6 * k_wd**2 * N_s**2 * mu_0 * rad_ag * l_s / pi / g_eff / p**2
        B_r1 = (mu_0 * I_f * N_f * 4 * sin(0.5 * (b_p / tau_p) * pi)) / g_eff / pi # (not used)
        
        # Calculating direct axis and quadrature axes inductances
        L_dm = (b_p / tau_p +(1 / pi) * sin(pi * b_p / tau_p)) * L_m
        L_qm = (b_p / tau_p -(1 / pi) * sin(pi * b_p / tau_p) + 2 / (3 * pi) * cos(b_p * pi / 2 * tau_p)) * L_m
        
        # Calculating actual load angle
        
        delta_m = (atan(om_e * L_qm * I_s / E_s))
        L_d = L_dm + L_ssigma # (not used)
        L_q = L_qm + L_ssigma # (not used)
        I_sd = I_s * sin(delta_m)
        I_sq = I_s * cos(delta_m)
        
        # induced voltage
        
        E_p = om_e * L_dm * I_sd + sqrt(E_s**2 - (om_e * L_qm * I_sq)**2) # (not used)
        # M_sf = mu_0 * 8*rad_ag * l_s * k_wd * N_s * N_f * sin(0.5 * b_p / tau_p * pi) / (p * g_eff * pi)
        # I_f1 = sqrt(2) * (E_p) / (om_e * M_sf)
        # I_f2 = (E_p / E_s) * B_g * g_eff * pi / (4 * N_f * mu_0 * sin(pi * b_p / 2/tau_p))
        # phi_max_stator = k_wd * N_s * pi * rad_ag * l_s * 2*mu_0 * N_f * I_f * 4*sin(0.5 * b_p / tau_p / pi) / (p * pi * g_eff * pi)
        # M_sf = mu_0 * 8*rad_ag * l_s * k_wd * N_s * N_f * sin(0.5 * b_p / tau_p / pi) / (p * g_eff * pi)
        
        L_tot = l_s + 2 * tau_p
        
        # Excitation power
        V_fn = 500
        Power_excitation = V_fn * 2*I_f   # total rated power in excitation winding
        Power_ratio = Power_excitation * 100 / machine_rating
        
        # Calculating Electromagnetically Active mass
        L_tot = l_s + 2 * tau_p # (not used)
        V_Cuss = m * l_Cus * A_Cus                                                     # volume of copper in stator
        V_Cusr = l_Cur * A_Cur                                                         # volume of copper in rotor
        V_Fest = (l_s * pi * ((rad_ag + h_s)**2 - rad_ag**2) - 2 * m*q1 * p*b_s * h_s * l_s) # volume of iron in stator tooth
        V_Fesy = l_s * pi * ((rad_ag + h_s + h_ys)**2 - (rad_ag + h_s)**2)                   # volume of iron in stator yoke
        V_Fert = 2 * p*l_pfe * (h_pc * b_pc + b_p * h_ps)                              # volume of iron in rotor pole
        V_Fery = l_pfe * pi * ((r_r - h_ps - h_pc)**2 - (r_r - h_ps - h_pc - h_yr)**2) # volume of iron in rotor yoke
        
        Copper = (V_Cuss + V_Cusr) * rho_Copper
        M_Fest = V_Fest * rho_Fe
        M_Fesy = V_Fesy * rho_Fe
        M_Fert = V_Fert * rho_Fe
        M_Fery = V_Fery * rho_Fe
        Iron = M_Fest + M_Fesy + M_Fert + M_Fery
        
        I_snom = machine_rating / (3 * E_s * cos_phi)
        
        ## Optional## Calculating mmf ratio
        F_1no_load = 3 * 2**0.5 * N_s * k_wd * I_s / (pi * p) # (not used)
        Nf_If_no_load = N_f * I_f
        F_1_rated = (3 * 2**0.5 * N_s * k_wd * I_srated) / (pi * p)
        Nf_If_rated = 2 * Nf_If_no_load
        Load_mmf_ratio = Nf_If_rated / F_1_rated
        
        ## Calculating losses
        #1. Copper losses
        K_R = 1.2
        P_Cuss = m * I_snom**2 * R_s * K_R
        P_Cusr = I_f**2 * R_r
        P_Cusnom_total = P_Cuss + P_Cusr
        
        #2. Iron losses ( Hysteresis and Eddy currents)
        P_Hyys = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))      # Hysteresis losses in stator yoke
        P_Ftys = M_Fesy * (B_symax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2) # Eddy losses in stator yoke
        P_Fesynom = P_Hyys + P_Ftys
        P_Hyd = M_Fest * (B_tmax / 1.5)**2 * (P_Fe0h * om_e / (2 * pi * 60))        # Hysteresis losses in stator teeth
        P_Ftd = M_Fest * (B_tmax / 1.5)**2 * (P_Fe0e * (om_e / (2 * pi * 60))**2)   # Eddy losses in stator teeth
        P_Festnom = P_Hyd + P_Ftd
        
        # brushes
        delta_v = 1
        n_brushes = (I_f * 2 / 120)
        
        if (n_brushes<0.5):
            n_brushes = 1
        else:
            n_brushes = np.round(n_brushes)
            
        #3. brush losses
        
        p_b = 2 * delta_v * (I_f)
        Losses = P_Cusnom_total + P_Festnom + P_Fesynom + p_b
        gen_eff = machine_rating * 100 / (Losses + machine_rating)
        
        ################################################## Structural  Design ########################################################
        
        ## Structural deflection calculations
        
        # rotor structure
        
        q3          = B_g**2 / 2/mu_0           # normal component of Maxwell's stress
        #l           = l_s                       # l - stator core length - now using l_s everywhere
        l_b         = 2 * tau_p                 # end winding length # (not used)
        l_e         = l_s + 2 * 0.001 * rad_ag     # equivalent core length # (not used)
        a_r         = (b_r * d_r) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr))  # cross-sectional area of rotor armms
        A_r         = l_s * t                   # cross-sectional area of rotor cylinder
        N_r         = np.round(n_r)
        theta_r     = pi / N_r                  # half angle between spokes
        I_r         = l_s * t**3 / 12             # second moment of area of rotor cylinder
        I_arm_axi_r = ((b_r * d_r**3) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr)**3)) / 12  # second moment of area of rotor arm
        I_arm_tor_r = ((d_r * b_r**3) - ((d_r - 2 * t_wr) * (b_r - 2 * t_wr)**3)) / 12  # second moment of area of rotot arm w.r.t torsion
        R           = r_r - h_ps - h_pc - 0.5 * h_yr
        R_1         = R - h_yr * 0.5            # inner radius of rotor cylinder
        k_1         = sqrt(I_r / A_r)           # radius of gyration
        m1          = (k_1 / R)**2 
        c           = R / 500 # (not used)
        
        u_all_r     = R / 10000 # allowable radial deflection 
        b_all_r     = 2 * pi * R_o / N_r  # allowable circumferential arm dimension 
        
        # Calculating radial deflection of rotor structure according to Mc Donald's
        Numer = R**3 * ((0.25 * (sin(theta_r) - (theta_r * cos(theta_r))) / (sin(theta_r))**2) - (0.5 / sin(theta_r)) + (0.5 / theta_r))
        Pov   = ((theta_r / (sin(theta_r))**2) + 1 / tan(theta_r)) * ((0.25 * R / A_r) + (0.25 * R**3 / I_r))
        Qov   = R**3 / (2 * I_r * theta_r * (m1 + 1))
        Lov   = (R_1 - R_o) / a_r
        Denom = I_r * (Pov - Qov + Lov) # radial deflection % rotor
        u_Ar  = (q3 * R**2 / E / h_yr) * (1 + Numer / Denom)
        
        # Calculating axial deflection of rotor structure
        
        w_r         = rho_Fes * g1 * sin(phi) * a_r * N_r
        mass_st_lam = rho_Fe * 2*pi * (R + 0.5 * h_yr) * l_s * h_yr                             # mass of rotor yoke steel
        W           = g1 * sin(phi) * (mass_st_lam + (V_Cusr * rho_Copper) + M_Fert) / N_r  # weight of rotor cylinder
        l_ir        = R                                      # length of rotor arm beam at which rotor cylinder acts
        l_iir       = R_1
        
        y_Ar        = (W * l_ir**3 / 12 / E / I_arm_axi_r) + (w_r * l_iir**4 / 24 / E / I_arm_axi_r)  # axial deflection
        
        # Calculating torsional deflection of rotor structure
        
        z_all_r     = radians(0.05 * R)  # allowable torsional deflection
        z_A_r       = (2 * pi * (R - 0.5 * h_yr) * l_s / N_r) * sigma * (l_ir - 0.5 * h_yr)**3 / 3 / E / I_arm_tor_r # circumferential deflection
        
        # STATOR structure
        
        A_st        = l_s * t_s
        a_s         = (b_st * d_s) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws))
        N_st        = np.round(n_s)
        theta_s     = pi / N_st 
        I_st        = l_s * t_s**3 / 12
        I_arm_axi_s = ((b_st * d_s**3) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws)**3)) / 12  # second moment of area of stator arm
        I_arm_tor_s = ((d_s * b_st**3) - ((d_s - 2 * t_ws) * (b_st - 2 * t_ws)**3)) / 12  # second moment of area of rotot arm w.r.t torsion
        R_st        = rad_ag + h_s + h_ys * 0.5
        R_1s        = R_st - h_ys * 0.5
        k_2         = sqrt(I_st / A_st)
        m2          = (k_2 / R_st)**2
        
        # allowable deflections
        
        b_all_s   = 2 * pi * R_o / N_st
        u_all_s   = R_st / 10000
        y_all     = 2 * l_s / 100         # allowable axial     deflection
        z_all_s   = radians(0.05 * R_st)  # allowable torsional deflection
        
        # Calculating radial deflection according to McDonald's
        
        Numers = R_st**3 * ((0.25 * (sin(theta_s) - (theta_s * cos(theta_s))) / (sin(theta_s))**2) - (0.5 / sin(theta_s)) + (0.5 / theta_s))
        Povs   = ((theta_s / (sin(theta_s))**2) + 1 / tan(theta_s)) * ((0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st))
        Qovs   = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
        Lovs   = (R_1s - R_o) * 0.5 / a_s
        Denoms = I_st * (Povs - Qovs + Lovs)
        R_out  = (R / 0.995 + h_s + h_ys)
        u_As   = (q3 * R_st**2 / E / t_s) * (1 + Numers / Denoms)
        
        # Calculating axial deflection according to McDonald
        
        l_is          = R_st - R_o
        l_iis         = l_is
        l_iiis        = l_is                                                  # length of rotor arm beam at which self-weight acts
        mass_st_lam_s = M_Fest + pi * l_s * rho_Fe * ((R_st + 0.5 * h_ys)**2 - (R_st - 0.5 * h_ys)**2)
        W_is          = g1 * sin(phi) * (rho_Fes * l_s * d_s**2 * 0.5) # weight of rotor cylinder                              
        W_iis         = g1 * sin(phi) * (V_Cuss * rho_Copper + mass_st_lam_s) / 2/N_st
        w_s           = rho_Fes * g1 * sin(phi) * a_s * N_st
        
        X_comp1 = W_is  * l_is**3   / 12 / E / I_arm_axi_s
        X_comp2 = W_iis * l_iis**4  / 24 / E / I_arm_axi_s
        X_comp3 = w_s   * l_iiis**4 / 24 / E / I_arm_axi_s
        
        y_As    = X_comp1 + X_comp2 + X_comp3  # axial deflection
        
        # Calculating torsional deflection
        
        z_A_s  = 2 * pi * (R_st + 0.5 * t_s) * l_s / (2 * N_st) * sigma * (l_is + 0.5 * t_s)**3 / 3 / E / I_arm_tor_s 
        
        # tangential stress constraints
        
        TC1 = T / (2 * pi * sigma)
        TC2 = R**2 * l_s
        TC3 = R_st**2 * l_s
        
        mass_stru_steel  = 2 * (N_st * (R_1s - R_o) * a_s * rho_Fes)
        
        # Calculating inactive mass and total mass
        
        Structural_mass = mass_stru_steel + (N_r * (R_1 - R_o) * a_r * rho_Fes)
        
        Mass = Copper + Iron + Structural_mass
        
        I = np.zeros(3)
        # Calculating mass moments of inertia and center of mass
        I[0]   = (0.50 * Mass*R_out**2)
        I[1]   = (0.25 * Mass*R_out**2 + Mass * l_s**2 / 12) 
        I[2]   = I[1]
        cm = np.zeros(3)
        cm[0]  = shaft_cm[0] + shaft_length / 2. + l_s / 2
        cm[1]  = shaft_cm[1]
        cm[2]  = shaft_cm[2]

        outputs['B_symax']           = B_symax
        outputs['B_tmax']            = B_tmax
        outputs['B_rymax']           = B_rymax
        outputs['B_gfm']             = B_gfm
        outputs['B_g']               = B_g
        outputs['B_pc']              = B_pc
        outputs['N_s']               = N_s
        outputs['b_s']               = b_s

        outputs['b_t']               = b_t
        outputs['A_Cuscalc']         = A_Cuscalc
        outputs['A_Curcalc']         = A_Curcalc
        outputs['b_p']               = b_p
        outputs['h_p']               = h_p
        outputs['p']                 = p
        outputs['E_s']               = E_s
        outputs['f']                 = f

        outputs['I_s']               = I_s
        outputs['R_s']               = R_s
        outputs['L_m']               = L_m
        outputs['A_1']               = A_1
        outputs['J_s']               = J_s
        outputs['R_r']               = R_r
        outputs['Losses']            = Losses

        outputs['Load_mmf_ratio']    = Load_mmf_ratio
        outputs['Power_ratio']       = Power_ratio
        outputs['n_brushes']         = n_brushes
        outputs['J_f']               = J_f
        outputs['K_rad']             = K_rad
        outputs['gen_eff']           = gen_eff
        outputs['S']                 = S

        outputs['Slot_aspect_ratio'] = Slot_aspect_ratio
        outputs['Copper']            = Copper
        outputs['Iron']              = Iron
        outputs['u_Ar']              = u_Ar
        outputs['y_Ar']              = y_Ar

        outputs['z_A_r']             = z_A_r
        outputs['u_As']              = u_As
        outputs['y_As']              = y_As
        outputs['z_A_s']             = z_A_s
        outputs['u_all_r']           = u_all_r
        outputs['u_all_s']           = u_all_s

        outputs['y_all']             = y_all
        outputs['z_all_s']           = z_all_s
        outputs['z_all_r']           = z_all_r
        outputs['b_all_s']           = b_all_s
        outputs['b_all_r']           = b_all_r
        outputs['TC1']               = TC1

        outputs['TC2']               = TC2
        outputs['TC3']               = TC3
        outputs['R_out']             = R_out
        outputs['Structural_mass']   = Structural_mass
        outputs['Mass']              = Mass
        outputs['cm']                = cm
        outputs['I']                 = I
        
        
