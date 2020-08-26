"""scig.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved."""

from openmdao.api import Group, Problem, ExplicitComponent,ExecComp,IndepVarComp,ScipyOptimizeDriver
import numpy as np
from math import pi, cos, sqrt, radians, sin, exp, log10, log, tan, atan


class SCIG(ExplicitComponent):
    
    """ Estimates overall mass dimensions and Efficiency of Squirrel cage Induction generator. """
    
    def setup(self):
        
        # SCIG generator design inputs
        self.add_input('r_s', val=0.0, units ='m', desc='airgap radius r_s')
        self.add_input('l_s', val=0.0, units ='m', desc='Stator core length l_s')
        self.add_input('h_s', val=0.0, units ='m', desc='Stator slot height')
        self.add_input('h_r',val=0.0, units ='m', desc='Rotor slot height')
        
        self.add_input('machine_rating',val=0.0, units ='W', desc='Machine rating')
        self.add_input('n_nom',val=0.0, units ='rpm', desc='rated speed')
        
        self.add_input('B_symax',val=0.0, desc='Peak Stator Yoke flux density B_symax')
        self.add_input('I_0', val=0.0, units='A', desc='no-load excitation current')
        
        self.add_input('shaft_cm',val= np.array([0.0, 0.0, 0.0]), units='m', desc='Main Shaft CM')
        self.add_input('shaft_length',val=0.0, units='m', desc='main shaft length')
        self.add_input('Gearbox_efficiency',val=0.0, desc='Gearbox efficiency')
        
        # Material properties
        self.add_input('rho_Fe',val=0.0,units='kg*m**-3', desc='Magnetic Steel density ')
        self.add_input('rho_Copper',val=0.0,units='kg*m**-3', desc='Copper density ')
        
        # DFIG generator design output
        # Magnetic loading
        self.add_output('B_g',val=0.0, desc='Peak air gap flux density B_g')
        self.add_output('B_g1',val=0.0,  desc='air gap flux density fundamental ')
        self.add_output('B_rymax',val=0.0,  desc='maximum flux density in rotor yoke')
        self.add_output('B_tsmax',val=0.0,  desc='maximum tooth flux density in stator')
        self.add_output('B_trmax',val=0.0, desc='maximum tooth flux density in rotor')
        
        #Stator design
        self.add_output('q1',val=0.0, desc='Slots per pole per phase')
        self.add_output('N_s',val=0.0,desc='Stator turns')
        self.add_output('S',val=0.0, desc='Stator slots')
        self.add_output('h_ys',val=0.0, desc='Stator Yoke height')
        self.add_output('b_s',val=0.0,desc='stator slot width')
        self.add_output('b_t',val=0.0, desc='stator tooth width')
        self.add_output('D_ratio',val=0.0,desc='Stator diameter ratio')
        self.add_output('A_Cuscalc',val=0.0, desc='Stator Conductor cross-section mm^2')
        self.add_output('Slot_aspect_ratio1',val=0.0, desc='Stator slot apsect ratio')
        
        #Rotor design
        self.add_output('h_yr',val=0.0, desc=' rotor yoke height')
        self.add_output('tau_p',val=0.0, desc='Pole pitch')
        self.add_output('p',val=0.0, desc='No of pole pairs')
        self.add_output('Q_r',val=0.0, desc='Rotor slots')
        self.add_output('b_r',val=0.0, desc='rotor slot width')
        self.add_output('b_trmin',val=0.0,desc='minimum tooth width')
        self.add_output('b_tr',val=0.0, desc='rotor tooth width')
        self.add_output('A_bar',val=0.0, desc='Rotor Conductor cross-section mm^2')
        self.add_output('Slot_aspect_ratio2',val=0.0, desc='Rotor slot apsect ratio')
        self.add_output('r_r',val=0.0,desc='rotor radius')
        
        # Electrical performance
        self.add_output('E_p',val=0.0, desc='Stator phase voltage')
        self.add_output('f',val=0.0, desc='Generator output frequency')
        self.add_output('I_s',val=0.0, desc='Generator output phase current')
        self.add_output('A_1' ,val=0.0, desc='Specific current loading')
        self.add_output('J_s',val=0.0, desc='Stator winding Current density')
        self.add_output('J_r',val=0.0, desc='Rotor winding Current density')
        self.add_output('R_s',val=0.0, desc='Stator resistance')
        self.add_output('R_R',val=0.0, desc='Rotor resistance')
        self.add_output('L_s',val=0.0, desc='Stator synchronising inductance')
        self.add_output('L_sm',val=0.0, desc='mutual inductance')
        
        # Structural performance
        self.add_output('TC1',val=0.0, desc='Torque constraint -stator')
        self.add_output('TC2',val=0.0, desc='Torque constraint-rotor')
        
        # Mass Outputs
        self.add_output('Copper', val=0.0, units='kg', desc='Copper Mass')
        self.add_output('Iron', val=0.0, units='kg', desc='Electrical Steel Mass')
        self.add_output('Structural_mass', val=0.0, units='kg', desc='Structural Mass')
        
        # Objective functions
        self.add_output('Mass',val=0.0, desc='Actual mass')
        self.add_output('K_rad',val=0.0, desc='Stack length ratio')
        self.add_output('Losses',val=0.0, desc='Total loss')
        self.add_output('gen_eff',val=0.0, desc='Generator efficiency')
        
        # Other parameters
        self.add_output('S_N',val=0.0,desc='Slip')
        self.add_output('D_ratio_UL',val=0.0, desc='Dia ratio upper limit')
        self.add_output('D_ratio_LL',val=0.0, desc='Dia ratio Lower limit')
        self.add_output('K_rad_UL',val=0.0, desc='Aspect ratio upper limit')
        self.add_output('K_rad_LL',val=0.0, desc='Aspect ratio Lower limit')
        
        self.add_output('Overall_eff',val=0.0, desc='Overall drivetrain efficiency')
        self.add_output('I',val=np.array([0.0, 0.0, 0.0]),desc='Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass')
        self.add_output('cm', val=np.array([0.0, 0.0, 0.0]),desc='COM [x,y,z]')
                
        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)
        
    def compute(self, inputs, outputs):
        #Create internal variables based on inputs
        r_s                  = inputs['r_s']
        l_s                  = inputs['l_s']
        h_s                  = inputs['h_s']
        h_r                  = inputs['h_r']
        machine_rating       = inputs['machine_rating']
        n_nom                = inputs['n_nom']
        Gearbox_efficiency   = inputs['Gearbox_efficiency']
        I_0                  = inputs['I_0']
        rho_Fe               = inputs['rho_Fe']
        rho_Copper           = inputs['rho_Copper']
        B_symax              = inputs['B_symax']
        shaft_cm     = inputs['shaft_cm']
        shaft_length = inputs['shaft_length']
        
        #Assign values to universal constants
        g1    = 9.81    # m/s^2 acceleration due to gravity
        sigma = 21.5e3  # shear stress 
        mu_0  = pi*4e-7 # permeability of free space
        cofi  = 0.9     # power factor
        h_w   = 0.005   # wedge height
        m     = 3       # Number of phases
        
        #Assign values to design constants
        q1        = 6                         # Number of slots per pole per phase
        b_s_tau_s = 0.45                      # Stator Slot width/Slot pitch ratio
        b_r_tau_r = 0.45                      # Rotor Slot width/Slot pitch ratio
        S_N       = -0.002                    # Slip
        y_tau_p   = 12./15                    # Coil span/pole pitch
        freq      = 60                  # frequency in Hz
        
        k_y1      = sin(pi*0.5*y_tau_p)       # winding Chording factor
        k_q1      = sin(pi/6)/q1/sin(pi/6/q1) # # zone factor
        k_wd      = k_y1*k_q1                 # Calcuating winding factor
        P_Fe0h    = 4                         #specific hysteresis losses W/kg @ 1.5 T @50 Hz
        P_Fe0e    = 1                         #specific hysteresis losses W/kg @ 1.5 T @50 Hz
        rho_Cu    = 1.8*10**(-8)*1.4          # Copper resistivity
        
        n_1       = n_nom/(1-S_N)             # actual rotor speed
        
        dia       = 2*r_s                       # air gap diameter
        p         = 3                    # number of pole pairs
        K_rad     = l_s/dia              # Aspect ratio
        K_rad_LL   = 0.5                 # lower limit on aspect ratio
        K_rad_UL   = 1.5                 # upper limit on aspect ratio
        
        # Calculating air gap length
        g          =(0.1+0.012*(machine_rating)**(1./3))*1e-3
        r_r        =r_s-g                         #rotor radius
        
        tau_p      =pi*dia/2/p              # Calculating pole pitch
        S          =2*m*p*q1                      # Calculating Stator slots
        
        tau_s      =tau_p/m/q1              # Stator slot pitch
        N_slots_pp =S/(m*p*2)              # Slots per pole per phase
        
        b_s        =b_s_tau_s*tau_s               # Calculating stator slot width
        b_so       =0.004;              #Stator slot opening wdth
        b_ro       =0.004;              #Rotor slot opening wdth
        b_t        =tau_s-b_s                     #tooth width
        
        q2         =4                         # rotor slots per pole per phase
        Q_r        =2*p*m*q2              # Calculating Rotor slots
        tau_r      =pi*(dia-2*g)/Q_r          # rotot slot pitch
        b_r        =b_r_tau_r*tau_r          # Rotor slot width
        b_tr       =tau_r-b_r              # rotor tooth width
        tau_r_min  =pi*(dia-2*(g+h_r))/Q_r
        b_trmin    =tau_r_min-b_r_tau_r*tau_r_min # minumum rotor tooth width
        
        
        # Calculating equivalent slot openings
        mu_rs       =0.005
        mu_rr       =0.005
        W_s         =(b_s/mu_rs)*1e-3   # Stator
        W_r         =(b_r/mu_rr)*1e-3   # Rotor
        
        Slot_aspect_ratio1 =h_s/b_s  # Stator slot aspect ratio
        Slot_aspect_ratio2 =h_r/b_r  # Rotor slot aspect ratio
        
        # Calculating Carter factor for stator,rotor and effective air gap length
        gamma_s = (2*W_s/g)**2/(5+2*W_s/g)
        K_Cs    =(tau_s)/(tau_s-g*gamma_s*0.5) #page 3-13 Boldea Induction machines Chapter 3
        gamma_r = (2*W_r/g)**2/(5+2*W_r/g)
        K_Cr    =(tau_r)/(tau_r-g*gamma_r*0.5) #page 3-13 Boldea Boldea Induction machines Chapter 3
        K_C     =K_Cs*K_Cr
        g_eff   =K_C*g
        
        om_m    =2*pi*n_nom/60               # mechanical frequency
        om_e    =p*om_m                   # electrical frequency
        f       =n_nom*p/60
        K_s     =0.3                     # saturation factor for Iron
        n_c     =2                       #number of conductors per coil
        a1      =2                       # number of parallel paths
        
        # Calculating stator winding turns
        N_s    = np.round(2*p*N_slots_pp*n_c/a1)
        
        # Calculating Peak flux densities
        B_g1   =mu_0*3*N_s*I_0*sqrt(2)*k_y1*k_q1/(pi*p*g_eff*(1+K_s))
        B_g    =B_g1*K_C
        B_rymax=B_symax
        
        # calculating back iron thickness
        h_ys= B_g*tau_p/(B_symax*pi)
        h_yr= h_ys
        
        d_se        =dia+2*(h_ys+h_s+h_w)  # stator outer diameter
        D_ratio=d_se/dia                        # Diameter ratio
        
        # limits for Diameter ratio depending on pole pair
        if (2*p==2):
            D_ratio_LL =1.65
            D_ratio_UL =1.69
        elif (2*p==4):
            D_ratio_LL =1.46
            D_ratio_UL =1.49
        elif (2*p==6):
            D_ratio_LL =1.37
            D_ratio_UL =1.4
        elif (2*p==8):
            D_ratio_LL =1.27
            D_ratio_UL =1.3
        else:
            D_ratio_LL =1.2
            D_ratio_UL =1.24
            
        # Stator slot fill factor
        if (2*r_s>2):
            K_fills=0.65
        else:
            K_fills=0.4
        
        # Stator winding length and cross-section
        l_fs=2*(0.015+y_tau_p*tau_p/2/cos(40*pi/180))+pi*(h_s) # end connection
        l_Cus = 2*N_s*(l_fs+l_s)/a1                               #shortpitch
        A_s = b_s*(h_s-h_w)                                                                         #Slot area
        A_scalc=b_s*1000*(h_s*1000-h_w*1000)                               #Conductor cross-section (mm^2)
        A_Cus = A_s*q1*p*K_fills/N_s                                             #Conductor cross-section (m^2)
        
        A_Cuscalc = A_scalc*q1*p*K_fills/N_s
        
        # Stator winding resistance
        R_s          =l_Cus*rho_Cu/A_Cus
        
        # Calculating no-load voltage
        om_s        =(n_nom)*2*pi/60                                                                 # rated angular frequency            
        P_e         =machine_rating/(1-S_N)                                         # Electrical power
        E_p    =om_s*N_s*k_wd*r_s*l_s*B_g1*sqrt(2)
        
        S_GN=(machine_rating-S_N*machine_rating)
        T_e         =p *(S_GN)/(2*pi*freq*(1-S_N))
        I_srated    =machine_rating/3/E_p/cofi
        
        #Rotor design
        k_fillr     = 0.7                                                                        #    Rotor slot fill factor
        diff        = h_r-h_w
        A_bar      = b_r*diff                                            # bar cross section
        Beta_skin   = sqrt(pi*mu_0*freq/2/rho_Cu)           #co-efficient for skin effect correction
        k_rm        = Beta_skin*h_r                    #co-efficient for skin effect correction
        J_b         = 6e+06                                                                    # Bar current density
        K_i         = 0.864
        I_b         = 2*m*N_s*k_wd*I_srated/Q_r    # bar current
        
        # Calculating bar resistance
        
        R_rb        =rho_Cu*k_rm*(l_s)/(A_bar)
        I_er        =I_b/(2*sin(pi*p/Q_r))                # End ring current
        J_er        = 0.8*J_b                                                                # End ring current density
        A_er        =I_er/J_er                                                            # End ring cross-section
        b           =h_r                                                                # End ring dimension
        a           =A_er/b                                                                 # End ring dimension
        D_er=(r_s*2-2*g)-0.003                                                    # End ring diameter
        l_er=pi*(D_er-b)/Q_r                                                        #  End ring segment length
        
        # Calculating end ring resistance
        R_re=rho_Cu*l_er/(2*A_er*(sin(pi*p/Q_r))**2)
        
        # Calculating equivalent rotor resistance
        R_R=(R_rb+R_re)*4*m*(k_wd*N_s)**2/Q_r
        
        # Calculating Rotor and Stator teeth flux density
        B_trmax = B_g*tau_r/b_trmin
        B_tsmax=tau_s*B_g/b_t
        
        # Calculating Equivalent core lengths
        l_r=l_s+(4)*g  # for axial cooling
        l_se =l_s+(2/3)*g
        K_fe=0.95           # Iron factor
        L_e=l_se *K_fe   # radial cooling
        
        # Calculating leakage inductance in  stator
        L_ssigmas=(2*mu_0*l_s*N_s**2/p/q1)*((h_s-h_w)/(3*b_s)+h_w/b_so)  #slot leakage inductance
        L_ssigmaew=(2*mu_0*l_s*N_s**2/p/q1)*0.34*q1*(l_fs-0.64*tau_p*y_tau_p)/l_s #end winding leakage inductance
        L_ssigmag=2*mu_0*l_s*N_s**2/p/q1*(5*(g*K_C/b_so)/(5+4*(g*K_C/b_so))) # tooth tip leakage inductance
        L_s=(L_ssigmas+L_ssigmaew+L_ssigmag)  # stator leakage inductance
        L_sm =6*mu_0*l_s*tau_p*(k_wd*N_s)**2/(pi**2*(p)*g_eff*(1+K_s))
        
        lambda_ei=2.3*D_er/(4*Q_r*l_s*(sin(pi*p/Q_r)**2))*log(4.7*dia/(a+2*b))
        lambda_b=h_r/3/b_r+h_w/b_ro
        L_i=pi*dia/Q_r
        
        # Calculating leakage inductance in  rotor
        L_rsl=(mu_0*l_s)*((h_r-h_w)/(3*b_r)+h_w/b_ro)  #slot leakage inductance
        L_rel=mu_0*(l_s*lambda_b+2*lambda_ei*L_i)                  #end winding leakage inductance
        L_rtl=(mu_0*l_s)*(0.9*tau_r*0.09/g_eff) # tooth tip leakage inductance
        L_rsigma=(L_rsl+L_rtl+L_rel)*4*m*(k_wd*N_s)**2/Q_r  # rotor leakage inductance
        
        # Calculating rotor current
        I_r=sqrt(-S_N*P_e/m/R_R)
        
        I_sm=E_p/(2*pi*freq*L_sm)
        # Calculating stator currents and specific current loading
        I_s=sqrt((I_r**2+I_sm**2))
        
        A_1=2*m*N_s*I_s/pi/(2*r_s)
        
        # Calculating masses of the electromagnetically active materials
        
        V_Cuss=m*l_Cus*A_Cus                                                                                # Volume of copper in stator
        V_Cusr=(Q_r*l_s*A_bar+pi*(D_er*A_er-A_er*b))    # Volume of copper in rotor
        V_Fest=(l_s*pi*((r_s+h_s)**2-r_s**2)-2*m*q1*p*b_s*h_s*l_s) # Volume of iron in stator teeth
        V_Fesy=l_s*pi*((r_s+h_s+h_ys)**2-(r_s+h_s)**2)                # Volume of iron in stator yoke
        r_r=r_s-g                                                                                            # rotor radius
        
        V_Fert=pi*l_s*(r_r**2-(r_r-h_r)**2)-2*m*q2*p*b_r*h_r*l_s # Volume of iron in rotor teeth
        V_Fery=l_s*pi*((r_r-h_r)**2-(r_r-h_r-h_yr)**2)                                         # Volume of iron in rotor yoke
        Copper=(V_Cuss+V_Cusr)*rho_Copper                                # Mass of Copper
        M_Fest=V_Fest*rho_Fe                                                                        # Mass of stator teeth
        M_Fesy=V_Fesy*rho_Fe                                                                        # Mass of stator yoke
        M_Fert=V_Fert*rho_Fe                                                                        # Mass of rotor tooth
        M_Fery=V_Fery*rho_Fe                                                                        # Mass of rotor yoke
        Iron=M_Fest+M_Fesy+M_Fert+M_Fery
        
        Active_mass=(Copper+Iron)
        L_tot=l_s
        Structural_mass=0.0001*Active_mass**2+0.8841*Active_mass-132.5
        Mass=Active_mass+Structural_mass
        
        # Calculating Losses and efficiency
        
        # 1. Copper losses
        
        K_R=1.2 # skin effect correction coefficient
        P_Cuss=m*I_s**2*R_s*K_R  # Copper loss-stator
        P_Cusr=m*I_r**2*R_R                     # Copper loss-rotor
        P_Cusnom=P_Cuss+P_Cusr                         # Copper loss-total
        
        # Iron Losses ( from Hysteresis and eddy currents)          
        P_Hyys=M_Fesy*(B_symax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))         # Hysteresis losses in stator yoke
        P_Ftys=M_Fesy*(B_symax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)    # Eddy losses in stator yoke
        P_Hyd=M_Fest*(B_tsmax/1.5)**2*(P_Fe0h*om_e/(2*pi*60))                    # Hysteresis losses in stator tooth
        P_Ftd=M_Fest*(B_tsmax/1.5)**2*(P_Fe0e*(om_e/(2*pi*60))**2)     # Eddy losses in stator tooth
        P_Hyyr=M_Fery*(B_rymax/1.5)**2*(P_Fe0h*abs(S_N)*om_e/(2*pi*60)) # Hysteresis losses in rotor yoke
        P_Ftyr=M_Fery*(B_rymax/1.5)**2*(P_Fe0e*(abs(S_N)*om_e/(2*pi*60))**2) # Eddy losses in rotor yoke
        P_Hydr=M_Fert*(B_trmax/1.5)**2*(P_Fe0h*abs(S_N)*om_e/(2*pi*60))            # Hysteresis losses in rotor tooth
        P_Ftdr=M_Fert*(B_trmax/1.5)**2*(P_Fe0e*(abs(S_N)*om_e/(2*pi*60))**2) # Eddy losses in rotor tooth
        
        # Calculating Additional losses
        P_add=0.5*machine_rating/100
        P_Fesnom=P_Hyys+P_Ftys+P_Hyd+P_Ftd+P_Hyyr+P_Ftyr+P_Hydr+P_Ftdr
        Losses=P_Cusnom+P_Fesnom+P_add
        gen_eff=(P_e-Losses)*100/P_e
        Overall_eff=gen_eff*Gearbox_efficiency
        
        # Calculating current densities in the stator and rotor
        J_s=I_s/A_Cuscalc
        J_r=I_r/(A_bar)/1e6
        
        # Calculating Tangential stress constraints
        TC1=T_e/(2*pi*sigma)
        TC2=r_s**2*l_s
        
        # Calculating mass moments of inertia and center of mass
        
        I = np.array([0.0, 0.0, 0.0])
        r_out=d_se*0.5
        I[0]   = (0.5*Mass*r_out**2)
        I[1]   = (0.25*Mass*r_out**2+(1/12)*Mass*l_s**2) 
        I[2]   = I[1]
        cm = np.array([0.0, 0.0, 0.0])
        cm[0]  = shaft_cm[0] + shaft_length/2. + l_s/2.
        cm[1]  = shaft_cm[1]
        cm[2]  = shaft_cm[2]
        
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
        outputs['r_r']                = r_r
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

