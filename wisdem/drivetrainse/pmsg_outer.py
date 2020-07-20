"""PMSG_Outer_rotor.py

Created by Latha Sethuraman
Copyright (c) NREL. All rights reserved.
Electromagnetic design based on conventional magnetic circuit laws
Structural design based on Pippard """

import numpy as np
import openmdao.api as om
from wisdem.commonse import gravity
import pandas as pd

#Assign values to universal constants
B_r        = 1.279      # Tesla remnant flux density
E          = 2e11       # N/m^2 young's modulus
ratio      = 0.8        # ratio of magnet width to pole pitch(bm/self.tau_p)
mu_0       = np.pi*4e-7 # permeability of free space
mu_r       = 1.06       # relative permeability
cofi       = 0.85       # power factor

#Assign values to design constants
h_0        = 0.005 # Slot opening height
h_1        = 0.004 # Slot wedge height
m          = 3     # no of phases
#b_s_tau_s = 0.45   # slot width to slot pitch ratio
k_sfil     = 0.65  # Slot fill factor
P_Fe0h     = 4	   # specific hysteresis losses W/kg @ 1.5 T 
P_Fe0e     = 1	   # specific hysteresis losses W/kg @ 1.5 T
k_fes      = 0.8   # Iron fill factor

#Assign values to universal constants
E          = 2e11           # Young's modulus
phi        = 90*2*np.pi/360 # tilt angle (rotor tilt -90 degrees during transportation)
v          = 0.3            # Poisson's ratio
G          = 79.3e9



def array_seq(q1,b,c,Total_number):
    #ONES  = b
    #ZEROS = b-c
    Seq	  = np.array([1,0,0,1,0])
    diff  = Total_number*5/6
    G     = np.prod(Seq.shape)
    return Seq, diff,G
	
def winding_factor(Sin, b, c, p, m):
    S = int(Sin)
    
    #Step 1 Writing q1 as a fraction
    q1=b/c
    
    # Step 2: Writing a binary sequence of b-c zeros and b ones
    Total_number=int(S/b)
    L = array_seq(q1,b,c,Total_number)
    
    # STep 3 : Repeat binary sequence Q_s/b times
    New_seq=np.tile(L[0],Total_number)
    Actual_seq1=(pd.DataFrame(New_seq[:,None].T))
    Winding_sequence=['A','C1','B','A1','C','B1']
    
    New_seq2=np.tile(Winding_sequence,int(L[1]))
    Actual_seq2= pd.DataFrame(New_seq2[:,None].T)
    Seq_f= pd.concat([Actual_seq1, Actual_seq2],ignore_index=True)
    Seq_f.reset_index(drop=True)
    
    Slots=S
    R=S if S%2 ==0 else S+1
        
    Windings_arrange=(pd.DataFrame(index=Seq_f.index,columns=Seq_f.columns[1:R])).fillna(0)
    counter=1

    #Step #4 Arranging winding in Slots
    for i in range(0,len(New_seq)):
        if Seq_f.loc[0,i]==1:
            Windings_arrange.loc[0,counter]=Seq_f.loc[1,i]
            counter=counter+1
            
    Windings_arrange.loc[1,1]='C1'
    
    for k in range(1,R):
        if Windings_arrange.loc[0,k]=='A':
            Windings_arrange.loc[1,k+1]='A1'
        elif Windings_arrange.loc[0,k]=='B':
            Windings_arrange.loc[1,k+1]='B1'
        elif Windings_arrange.loc[0,k]=='C':
            Windings_arrange.loc[1,k+1]='C1'
        elif Windings_arrange.loc[0,k]=='A1':
            Windings_arrange.loc[1,k+1]='A'
        elif Windings_arrange.loc[0,k]=='B1':
            Windings_arrange.loc[1,k+1]='B'
        elif Windings_arrange.loc[0,k]=='C1':
            Windings_arrange.loc[1,k+1]='C'
	
    Phase_A=np.zeros((1000,1),dtype=float)
    counter_A=0
    #Windings_arrange.to_excel('test.xlsx')
    # Winding vector, W_A for Phase A
    for l in range(1,R):
        if Windings_arrange.loc[0,l]=='A' and Windings_arrange.loc[1,l]=='A':
            Phase_A[counter_A,0]=l
            Phase_A[counter_A+1,0]=l
            counter_A=counter_A+2
        elif Windings_arrange.loc[0,l]== 'A1' and Windings_arrange.loc[1,l]=='A1':
            Phase_A[counter_A,0]=-1*l
            Phase_A[counter_A+1,0]=-1*l
            counter_A=counter_A+2
        elif Windings_arrange.loc[0,l]=='A' or Windings_arrange.loc[1,l]=='A':
            Phase_A[counter_A,0]=l
            counter_A=counter_A+1
        elif Windings_arrange.loc[0,l]=='A1' or Windings_arrange.loc[1,l]=='A1':
            Phase_A[counter_A,0]=-1*l
            counter_A=counter_A+1
    
    W_A=(np.trim_zeros(Phase_A)).T
    # Calculate winding factor
    K_w=0
    
    for r in range(0,int(2*(S)/3)):
        Gamma=2*np.pi*p*np.abs(W_A[0,r])/S
        K_w+=np.sign(W_A[0,r])*(np.exp(Gamma*1j))

    K_w       = np.abs(K_w)/(2*S/3)
    CPMR      = np.lcm(S,int(2*p))
    N_cog_s   = CPMR/S
    N_cog_p   = CPMR/p
    N_cog_t   = CPMR*0.5/p
    A         = np.lcm(S,int(2*p))
    b_p_tau_p = 2*1*p/S-0
    b_t_tau_s = (2)*S*0.5/p-2
        
    return K_w


def shell_constant(R,t,l,x,v):
    
    Lambda     = (3*(1-v**2)/(R**2*t**2))**0.25
    D          = E*t**3/(12*(1-v**2))
    C_14       = (np.sinh(Lambda*l))**2+ (np.sin(Lambda*l))**2
    C_11       = (np.sinh(Lambda*l))**2- (np.sin(Lambda*l))**2
    F_2        = np.cosh(Lambda*x)*np.sin(Lambda*x) + np.sinh (Lambda*x)* np.cos(Lambda*x)
    C_13       = np.cosh(Lambda*l)*np.sinh(Lambda*l) - np.cos(Lambda*l)* np.sin(Lambda*l)
    F_1        = np.cosh(Lambda*x)*np.cos(Lambda*x)
    F_4        = np.cosh(Lambda*x)*np.sin(Lambda*x)-np.sinh(Lambda*x)*np.cos(Lambda*x)
    
    return D,Lambda,C_14,C_11,F_2,C_13,F_1,F_4
        
def plate_constant(a,b,v,r_o,t):
    
    D          = E*t**3/(12*(1-v**2))
    C_2        = 0.25*(1-(b/a)**2*(1+2*np.log(a/b)))
    C_3        = 0.25*(b/a)*(((b/a)**2+1)*np.log(a/b)+(b/a)**2 -1)
    C_5        = 0.5*(1-(b/a)**2)
    C_6        = 0.25*(b/a)*((b/a)**2-1+2*np.log(a/b))
    C_8        = 0.5*(1+v+(1-v)*(b/a)**2)
    C_9        = (b/a)*(0.5*(1+v)*np.log(a/b)+0.25*(1-v)*(1-(b/a)**2))
    L_11       = (1/64)*(1+4*(r_o/a)**2-5*(r_o/a)**4-4*(r_o/a)**2*(2+(r_o/a)**2)*np.log(a/r_o))
    L_17       = 0.25*(1-0.25*(1-v)*((1-(r_o/a)**4)-(r_o/a)**2*(1+(1+v)*np.log(a/r_o))))
            
    return D,C_2,C_3,C_5,C_6,C_8,C_9,L_11,L_17


class PMSG_active(om.ExplicitComponent):
    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        self.add_input('r_g',0.0, units ='m', desc='airgap radius ')
        self.add_input('l_s',0.0, units ='m', desc='Stator core length ')
        self.add_input('h_s',0.0, units ='m', desc='Yoke height ')
        self.add_input('P_av_v', units ='W', desc='Machine rating')
        self.add_input('P_mech', units ='W', desc='Shaft mechanical power')
        self.add_input('N_nom', 0.0, units = 'rpm', desc='rated speed')
        self.add_input('T_rated',0.0, units = 'N*m', desc='Rated torque ')
        self.add_input('h_m',0.0, units ='m',desc='magnet height')
        self.add_input('h_ys',0.0, units ='m', desc='Yoke height')
        self.add_input('h_yr',0.0, units ='m', desc='rotor yoke height')
        self.add_input('J_s',0.0, units ='A/(mm*mm)', desc='Stator winding current density')
        self.add_input('N_c',0.0, desc='Number of turns per coil')
        self.add_input('b',0.0, desc='Slot pole combination')
        self.add_input('c',0.0, desc='Slot pole combination')
        self.add_input('p',0.0, desc='Pole pairs ')
        self.add_input('Sigma',0.0,units='N/m**2',desc='Shear Stress')
        self.add_input('E_p',0.0, units ='V', desc='Stator phase voltage')
        
        # Material properties
        self.add_input('rho_Fes',0.0, units='kg/(m**3)',desc='Structural Steel density')
        self.add_input('rho_Fe',0.0, units='kg/(m**3)', desc='Electrical Steel density ')
        self.add_input('rho_Copper',0.0,units='kg/(m**3)', desc='Copper density')
        self.add_input('rho_PM',0.0,units ='kg/(m**3)', desc='Magnet density ')
        self.add_input('resist_Cu',0.0,units ='ohm*m', desc='Copper resistivity ')
        
        # PMSG_structrual inputs
        self.add_input('Structural_mass_rotor',0.0,units='kg',desc='Structural Mass')
        self.add_input('Structural_mass_stator',0.0,units='kg',desc='Structural Mass')
        self.add_input('h_sr',0.0,units='m',desc='Structural Mass')
        
        # Magnetic loading
        self.add_input('B_tmax',0.0, desc='Peak Teeth flux density')
        self.add_output('B_symax',0.0,desc='Peak Stator Yoke flux density B_ymax')
        self.add_output('B_rymax',0.0, desc='Peak Rotor yoke flux density')
        self.add_output('B_smax',0.0,desc='Peak Stator flux density')
        self.add_output('B_pm1', 0.0,desc='Fundamental component of peak air gap flux density')
        self.add_output('B_g', 0.0, desc='Peak air gap flux density ')
        self.add_output('tau_p',0.0, units ='m',desc='Pole pitch')
        self.add_output('q',0.0, units ='N/m**2',desc='Normal stress')
        self.add_output('g',0.0,units='m',desc='Air gap length')
        
        #Stator design
        self.add_output('N_s', 0.0, desc='Number of turns in the stator winding')
        self.add_output('b_s',0.0, units ='m', desc='slot width')
        self.add_output('b_t',0.0, units = 'm',desc='tooth width')
        self.add_output('h_t',0.0, units = 'm', desc ='tooth height')
        self.add_output('A_Cuscalc',0.0, units ='mm**2', desc='Conductor cross-section')
        self.add_output('tau_s', 0.0, units='m',desc='Slot pitch ')
        
        #Rotor magnet dimension
        self.add_output('b_m',0.0, units ='m',desc='magnet width')
        
        # Electrical performance
        self.add_output('f',0.0, units ='Hz', desc='Generator output frequency')
        self.add_output('I_s', 0.0, units ='A', desc='Generator output phase current')
        self.add_output('R_s',0.0, units ='ohm',desc='Stator resistance')
        self.add_output('L_s',0.0,desc='Stator synchronising inductance')
        self.add_output('A_1',0.0, units='A/m', desc='Electrical loading')
        self.add_output('J_actual',0.0, units ='A/m**2', desc='Current density')
        self.add_output('T_e',0.0,units='N*m',desc='Electromagnetic torque')
        
        # Objective functions
        self.add_output('Mass',0.0, units='kg',desc='Actual mass')
        self.add_output('K_rad', desc='Aspect ratio')
        self.add_output('Losses',desc='Total power losses')
        self.add_output('gen_eff', desc='Generator efficiency')
       
        # Other parameters
        self.add_output('R_out',0.0,units='m', desc='Outer radius')
        self.add_output('S',0.0,desc='Stator slots')
        self.add_output('Slot_aspect_ratio',0.0,desc='Slot aspect ratio')
        
        # Mass Outputs
        self.add_output('mass_PM',0.0,units = 'kg', desc='Magnet mass')
        self.add_output('Copper',0.0, units ='kg', desc='Copper Mass')
        self.add_output('Iron',0.0, units = 'kg', desc='Electrical Steel Mass')
        self.add_output('Mass_tooth_stator',0.0, units = 'kg', desc='Teeth and copper mass')
        self.add_output('Mass_yoke_rotor',0.0, units = 'kg', desc='yoke mass')
        self.add_output('Mass_yoke_stator',0.0, units = 'kg', desc='yoke mass')
        self.add_output('Structural_mass',units='kg',desc='Total Structural Mass')

        self.add_output('I',np.zeros(3),units='kg*m**2',desc='Structural Mass')
        
        
    def compute(self, inputs, outputs):
        # Unpack inputs
        r_g    = float(inputs['r_g'])
        l_s    = float(inputs['l_s'])
        p      = float(inputs['p'])
        b      = float(inputs['b'])
        c      = float(inputs['c'])
        h_m    = float(inputs['h_m'])
        h_ys   = float(inputs['h_ys'])
        h_yr   = float(inputs['h_yr'])
        h_s    = float(inputs['h_s'])
        B_tmax = float(inputs['B_tmax'])
        E_p    = float(inputs['E_p'])
        P_mech = float(inputs['P_mech'])
        P_av_v = float(inputs['P_av_v'])
        
        ######################## Electromagnetic design ###################################
        outputs['K_rad'] = l_s/(2* r_g) # Aspect ratio
        
        # Calculating air gap length
        dia		 = 2* r_g            # air gap diameter
        g                = 0.001*dia         # air gap length
        r_s		 = r_g-g             #Stator outer radius
        b_so		 = 2*g               # Slot opening
        tau_p            = (np.pi*dia/(2*p)) # pole pitch
        outputs['g']     = g
        outputs['tau_p'] = tau_p
        
        # Calculating winding factor
        Slot_pole    = b/c
        S            = Slot_pole*2*p*m
        testval      = S/(m*np.gcd(int(S),int(p)))
        outputs['S'] = S
    
        if testval.is_integer():
            k_w		     = winding_factor(int(S),b,c,int(p),m)
            b_m              = ratio*tau_p  # magnet width
            alpha_p	     = np.pi/2*ratio
            tau_s            = np.pi*(dia-2*g)/S
            outputs['tau_s'] = tau_s
            outputs['b_m']   = b_m
            
            # Calculating Carter factor for statorand effective air gap length
            gamma	 = 4/np.pi*(b_so/2/(g+h_m/mu_r)*np.arctan(b_so/2/(g+ h_m/mu_r))-np.log(np.sqrt(1+(b_so/2/(g+h_m/mu_r))**2)))
            k_C		 = tau_s/(tau_s-gamma*(g+h_m/mu_r))   # carter coefficient
            g_eff	 = k_C*(g+ h_m/mu_r)
            
            # angular frequency in radians
            om_m	 = 2*np.pi*inputs['N_nom']/60
            om_e	 = p*om_m
            outputs['f'] = om_e/2/np.pi # outout frequency
            
            # Calculating magnetic loading
            B_pm1              =  B_r*h_m/mu_r/(g_eff)
            B_g                =  B_r*h_m/(mu_r*g_eff)*(4/np.pi)*np.sin(alpha_p)
            B_symax            =  B_pm1* b_m/(2*h_ys)*k_fes
            B_rymax            =  B_pm1*b_m*k_fes/(2*h_yr)
            b_t                =  B_pm1*tau_s/B_tmax
            N_c                =  2    # Number of turns per coil
            outputs['B_pm1']   =  B_pm1
            outputs['B_g']     =  B_g
            outputs['B_symax'] =  B_symax
            outputs['B_rymax'] =  B_rymax
            outputs['b_t']     =  b_t
            outputs['q']       = (B_g)**2/2/mu_0
            
            # Stator winding length ,cross-section and resistance
            l_Cus = (2*(l_s+np.pi/4*(tau_s+b_t)))  # length of a turn
            
            # Calculating no-load voltage induced in the stator
            N_s            = np.rint(E_p/(np.sqrt(2)*l_s*r_s*k_w*om_m*B_g))
            Z              = (P_av_v/(m*E_p))
            outputs['N_s'] = N_s
            
            # Calculating leakage inductance in  stator
            V_1            = E_p/1.1
            I_n		   = P_av_v/3/cofi/V_1
            J_s            = 6.0
            A_Cuscalc      = I_n/J_s
            A_slot         = 2*N_c*A_Cuscalc*(10**-6)/k_sfil
            tau_s_new      = np.pi*(dia-2*g-2*h_1-2*h_0)/S
            b_s2	   = tau_s_new-b_t # Slot top width
            b_s1	   = np.sqrt(b_s2**2-4*np.pi*A_slot/S)
            b_s            = (b_s1+b_s2)*0.5
            N_coil         = 2*S
            P_s            = mu_0*(h_s/3/b_s +h_1*2/(b_s2+b_so)+h_0/b_so)    #Slot permeance function
            L_ssigmas      = S/3*4*N_c**2*l_s*P_s  #slot leakage inductance
            L_ssigmaew     = N_coil*N_c**2*mu_0*tau_s*np.log((0.25*np.pi*tau_s**2)/(0.5*h_s*b_s))     #end winding leakage inductance
            L_aa           = 2*np.pi/3*(N_c**2*mu_0*l_s*r_s/g_eff)
            L_ab           = 0.0
            L_m            = L_aa
            L_ssigma       = (L_ssigmas+L_ssigmaew)
            L_s            = L_m+L_ssigma
            G              = E_p**4-1/9*(P_av_v*om_e*L_s)**2
            outputs['A_Cuscalc']         = A_Cuscalc
            outputs['b_s']               = b_s
            outputs['L_s']               = L_s
            outputs['Slot_aspect_ratio'] = h_s/b_s

            # Calculating stator current and electrical loading
            I_s                 = np.sqrt(2*((E_p)**2-G**0.5)/(om_e*L_s)**2)
            outputs['I_s']      = I_s
            outputs['A_1']      = 6*I_s*N_s/np.pi/dia
            outputs['J_actual'] = I_s/(A_Cuscalc*2**0.5)
            L_Cus               = N_s*l_Cus
            outputs['R_s']      = inputs['resist_Cu']*(N_s)*l_Cus/(A_Cuscalc*(10**-6))
            outputs['B_smax']   = np.sqrt(2)*I_s*mu_0/g_eff

            # Calculating Electromagnetically active mass
            wedge_area         = (b_s*0.5-b_so*0.5)*(2*h_0+h_1)
            V_Cus 	       = m*L_Cus*(A_Cuscalc*(10**-6))     # copper volume
            outputs['h_t']     = (h_s+h_1+h_0)
            V_Fest	       = l_s*S*(b_t*(h_s+h_1+h_0)+wedge_area)# volume of iron in stator tooth
            V_Fesy	       = l_s*np.pi*((r_g-g-h_s-h_1-h_0)**2-(r_g-g-h_s-h_1-h_0-h_ys)**2) # volume of iron in stator yoke
            V_Fery	       = l_s*np.pi*((r_g+h_m+h_yr)**2-(r_g+h_m)**2)
            outputs['Copper']  = V_Cus*inputs['rho_Copper']
            M_Fest	       = V_Fest*inputs['rho_Fe']    # Mass of stator tooth
            M_Fesy	       = V_Fesy*inputs['rho_Fe']    # Mass of stator yoke
            M_Fery	       = V_Fery*inputs['rho_Fe']    # Mass of rotor yoke
            outputs['Iron']    = M_Fest+M_Fesy+M_Fery
            outputs['mass_PM'] = 2*np.pi*(r_g+ h_m)*l_s*h_m*ratio*inputs['rho_PM']
            
            # Calculating Losses
            ##1. Copper Losses
            K_R       = 1.0   # Skin effect correction co-efficient
            P_Cu      = m*(I_s/2**0.5)**2*outputs['R_s']*K_R
            
            # Iron Losses ( from Hysteresis and eddy currents)
            P_Hyys    = M_Fesy*(B_symax/1.5)**2*(P_Fe0h*om_e/(2*np.pi*60)) # Hysteresis losses in stator yoke
            P_Ftys    = M_Fesy*((B_symax/1.5)**2)*(P_Fe0e*(om_e/(2*np.pi*60))**2) # Eddy losses in stator yoke
            P_Fesynom = P_Hyys+P_Ftys
            P_Hyd     = M_Fest*(B_tmax/1.5)**2*(P_Fe0h*om_e/(2*np.pi*60))  # Hysteresis losses in stator teeth
            P_Ftd     = M_Fest*(B_tmax/1.5)**2*(P_Fe0e*(om_e/(2*np.pi*60))**2) # Eddy losses in stator teeth
            P_Festnom = P_Hyd+P_Ftd
                
            # Iron Losses ( from Hysteresis and eddy currents)
            P_Hyyr    = M_Fery*(B_rymax/1.5)**2*(P_Fe0h*om_e/(2*np.pi*60)) # Hysteresis losses in stator yoke
            P_Ftyr    = M_Fery*((B_rymax/1.5)**2)*(P_Fe0e*(om_e/(2*np.pi*60))**2) # Eddy losses in stator yoke
            P_Ferynom = P_Hyyr+P_Ftyr

            # additional stray losses due to leakage flux
            P_ad           = 0.2*(P_Hyys + P_Ftys + P_Hyd + P_Ftd +P_Hyyr+P_Ftyr )
            pFtm           = 300 # specific magnet loss
            P_Ftm          = pFtm*2*p*b_m*l_s
            Losses         = P_Cu+P_Festnom+P_Fesynom+P_ad+P_Ftm+P_Ferynom 
            gen_eff        = (P_mech-Losses)/(P_mech)*100
            I_snom	   = gen_eff*(P_mech/m/E_p/cofi) #rated current
            I_qnom	   = gen_eff*P_mech/(m* E_p)
            X_snom	   = om_e*(L_m+L_ssigma)
            outputs['T_e'] = np.pi*r_g**2*l_s*2*inputs['Sigma']
            Stator         = M_Fesy+M_Fest+outputs['Copper'] #modified mass_stru_steel
            Rotor          = M_Fery+outputs['mass_PM']  #modified (N_r*(R_1-self.R_o)*a_r*self.rho_Fes))
            
            outputs['Mass_tooth_stator'] = M_Fest+outputs['Copper']
            outputs['Mass_yoke_rotor']   = M_Fery
            outputs['Mass_yoke_stator']  = M_Fesy
            outputs['Structural_mass']   = inputs['Structural_mass_rotor']+inputs['Structural_mass_stator']
            outputs['Mass']              = Stator+Rotor+outputs['Structural_mass']
            outputs['R_out']             = (dia+2*h_m+2*h_yr+2*inputs['h_sr'])*0.5
            outputs['Losses']            = Losses
            outputs['gen_eff']           = gen_eff
        else:
            pass

            			

class PMSG_rotor_inactive(om.ExplicitComponent):
    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        self.add_input('R_sh',0.0, units ='m', desc='airgap radius ')
        self.add_input('y_sh', units ='W', desc='Machine rating')
        self.add_input('theta_sh', 0.0, units = 'rad', desc='slope')
        self.add_input('T_rated',0.0, units = 'N*m', desc='Rated torque ')
        self.add_input('r_g',0.0, units ='m', desc='air gap radius')
        self.add_input('h_m',0.0, units ='m', desc='Magnet height ')
        self.add_input('l_s',0.0, units ='m', desc='core length')
        self.add_input('q',0.0, units ='N/m**2', desc='Normal Stress')
        self.add_input('h_yr',0.0, units ='m', desc='Rotor yoke height ')
        self.add_input('u_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('y_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
        
        # structural design variables
        self.add_input('t_r',0.0, units ='m', desc='Rotor disc thickness')
        self.add_input('h_sr',0.0, units ='m', desc='Yoke height ')
        self.add_input('K_rad', desc='Aspect ratio')
        
        # Material properties
        self.add_input('rho_Fes',0.0, units='kg/(m**3)',desc='Structural Steel density')
        self.add_input('rho_Fe',0.0, units='kg/(m**3)', desc='Electrical Steel density ')
        self.add_input('mass_PM',0.0,units ='kg', desc='Magnet density ')
        
        self.add_output('Structural_mass_rotor',0.0, units='kg', desc='Rotor mass (kg)')
        self.add_output('u_ar',0.0,units='m', desc='Radial deformation')
        self.add_output('y_ar',0.0, units ='m', desc='Axial deformation')
        self.add_output('twist_r', 0.0, units ='deg', desc='torsional twist')
        self.add_output('TC1r',0.0, desc='Torque constraint1-rotor')
        self.add_output('TC2r',0.0, desc='Torque constraint2-rotor')
        self.add_output('TC_test_r',0.0, desc='Torque constraint flag')
        self.add_output('u_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_output('y_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_input('Sigma',0.0,units='N/m**2',desc='Shear stress')
        
        
    def compute(self, inputs, outputs):
        # Unpack inputs
        r_g  = float(inputs['r_g'])
        l_s  = float(inputs['l_s'])
        h_m  = float(inputs['h_m'])
        h_yr = float(inputs['h_yr'])
        h_sr = float(inputs['h_sr'])
        t_r  = float(inputs['t_r'])
        R_sh = float(inputs['R_sh'])
        y_sh = float(inputs['y_sh'])
        q    = float(inputs['q'])
        rho_Fes = float(inputs['rho_Fes'])
        
        # Radial deformation of rotor
        R               = r_g+h_m
        L_r             = l_s+t_r+0.125
        constants_x_0   = shell_constant(R,t_r,L_r,0,v)
        constants_x_L   = shell_constant(R,t_r,L_r,L_r,v)
        f_d_denom1      = R/(E*((R)**2-(R_sh)**2))*((1-v)*R**2+(1+v)*(R_sh)**2)
        f_d_denom2      = t_r/(2*constants_x_0[0]*(constants_x_0[1])**3)*(constants_x_0[2]/(2*constants_x_0[3])*constants_x_0[4]-constants_x_0[5]/constants_x_0[3]*constants_x_0[6]-0.5*constants_x_0[7])
        f               = q*(R)**2*t_r/(E*(h_yr+h_sr)*(f_d_denom1+f_d_denom2))
        u_d             = f/(constants_x_L[0]*(constants_x_L[1])**3)*((constants_x_L[2]/(2*constants_x_L[3])*constants_x_L[4] -constants_x_L[5]/constants_x_L[3]*constants_x_L[6]-0.5*constants_x_L[7]))+y_sh
 
        outputs['u_ar'] = (q*(R)**2)/(E*(h_yr+h_sr))-u_d
        outputs['u_ar'] = np.abs(outputs['u_ar'] + y_sh)
        outputs['u_allowable_r'] =2*r_g/1000*inputs['u_allow_pcent']/100
        
        # axial deformation of rotor
        W_back_iron =  plate_constant(R+h_sr+h_yr,R_sh,v,0.5*h_yr+R,t_r)
        W_ssteel    =  plate_constant(R+h_sr+h_yr,R_sh,v,h_yr+R+h_sr*0.5,t_r)
        W_mag       =  plate_constant(R+h_sr+h_yr,R_sh,v,h_yr+R-0.5*h_m,t_r)
        W_ir        =  inputs['rho_Fe']*gravity*np.sin(phi)*(L_r-t_r)*h_yr
        y_ai1r      = -W_ir*(0.5*h_yr+R)**4/(R_sh*W_back_iron[0])*(W_back_iron[1]*W_back_iron[4]/W_back_iron[3]-W_back_iron[2])
        W_sr        =  rho_Fes*gravity*np.sin(phi)*(L_r-t_r)*h_sr
        y_ai2r      = -W_sr*(h_sr*0.5+h_yr+R)**4/(R_sh*W_ssteel[0])*(W_ssteel[1]*W_ssteel[4]/W_ssteel[3]-W_ssteel[2])
        W_m         =  np.sin(phi)*inputs['mass_PM']/(2*np.pi*(R-h_m*0.5))
        y_ai3r      = -W_m*(R-h_m)**4/(R_sh*W_mag[0])*(W_mag[1]*W_mag[4]/W_mag[3]-W_mag[2])
        w_disc_r    = rho_Fes*gravity*np.sin(phi)*t_r
        a_ii        = R+h_sr+h_yr
        r_oii       = R_sh
        M_rb        = -w_disc_r*a_ii**2/W_ssteel[5]*(W_ssteel[6]*0.5/(a_ii*R_sh)*(a_ii**2-r_oii**2)-W_ssteel[8])
        Q_b         =  w_disc_r*0.5/R_sh*(a_ii**2-r_oii**2)
        y_aiir      =  M_rb*a_ii**2/W_ssteel[0]*W_ssteel[1]+Q_b*a_ii**3/W_ssteel[0]*W_ssteel[2]-w_disc_r*a_ii**4/W_ssteel[0]*W_ssteel[7]
        I           = np.pi*0.25*(R**4-(R_sh)**4)
        F_ecc       = q*2*np.pi*inputs['K_rad']*r_g**3
        M_ar        = F_ecc*L_r*0.5
        outputs['y_ar'] = np.abs(y_ai1r+y_ai2r+y_ai3r)+y_aiir+(R+h_yr+h_sr)*inputs['theta_sh']+M_ar*L_r**2*0/(2*E*I)
        outputs['y_allowable_r'] = L_r/100*inputs['y_allow_pcent']
        
        # Torsional deformation of rotor
        J_dr            = 0.5*np.pi*((R+h_yr+h_sr)**4-R_sh**4)
        J_cylr          = 0.5*np.pi*((R+h_yr+h_sr)**4-R**4)
        outputs['twist_r'] = 180/np.pi*inputs['T_rated']/G*(t_r/J_dr+(L_r-t_r)/J_cylr)
        outputs['Structural_mass_rotor'] = rho_Fes*np.pi*(((R+h_yr+h_sr)**2-(R_sh)**2)*t_r+\
                                           ((R+h_yr+h_sr)**2-(R+h_yr)**2)*l_s)
        outputs['TC1r']  = (R+(h_yr+h_sr))**2*L_r
        outputs['TC2r']  = inputs['T_rated']/(2*np.pi*inputs['Sigma'])
        outputs['TC_test_r'] = 1 if outputs['TC1r']>outputs['TC2r'] else 0
        
        
        
        
class PMSG_stator_inactive(om.ExplicitComponent):
    """ Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator. """
    
    def setup(self):
    
        self.add_input('R_no',0.0, units ='m', desc='Nose outer radius ')
        self.add_input('y_bd', units ='W', desc='Deflection of the bedplate')
        self.add_input('theta_bd', 0.0, units = 'm', desc='Slope at the bedplate')
        self.add_input('T_rated',0.0, units = 'N*m', desc='Rated torque ')
        self.add_input('r_g',0.0, units ='m', desc='air gap radius')
        self.add_input('g',0.0, units ='m', desc='air gap length')
        self.add_input('h_t',0.0, units ='m', desc='tooth height')
        self.add_input('l_s',0.0, units ='m', desc='core length')
        self.add_input('q',0.0, units ='N/m**2', desc='Normal stress')
        self.add_input('h_ys',0.0, units ='m', desc='Stator yoke height ')
        
        # structural design variables
        self.add_input('t_s',0.0, units ='m', desc='Stator disc thickness')
        self.add_input('h_ss',0.0, units ='m', desc='Stator yoke height ')
        self.add_input('K_rad', desc='Aspect ratio')
        self.add_input('u_allow_pcent',0.0,desc='Radial deflection as a percentage of air gap diameter')
        self.add_input('y_allow_pcent',0.0,desc='Axial deflection as a percentage of air gap diameter')
        self.add_input('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
        
        # Material properties
        self.add_input('rho_Fes',0.0, units='kg/(m**3)',desc='Structural Steel density')
        self.add_input('rho_Fe',0.0, units='kg/(m**3)', desc='Electrical Steel density ')
        self.add_input('Mass_tooth_stator',0.0,units ='kg', desc='Stator teeth mass ')
        self.add_input('Copper',0.0,units ='kg', desc='Copper mass ')

        self.add_output('Structural_mass_stator',0.0, units='kg', desc='Stator mass (kg)')
        self.add_output('u_as',0.0,units='m', desc='Radial deformation')
        self.add_output('y_as',0.0, units ='m', desc='Axial deformation')
        self.add_output('twist_s', 0.0, units ='deg', desc='Stator torsional twist')
        self.add_input('Sigma',0.0,units='N/m**2',desc='Shear stress')
        self.add_output('TC1s',0.0, desc='Torque constraint1-stator')
        self.add_output('TC2s',0.0, desc='Torque constraint2-stator')
        self.add_output('TC_test_s',0.0, desc='Torque constraint flag')
        self.add_output('u_allowable_s',0.0,units='m',desc='Allowable Radial deflection as a percentage of air gap diameter')
        self.add_output('y_allowable_s',0.0,units='m',desc='Allowable Axial deflection as a percentage of air gap diameter')
        
        
    def compute(self, inputs, outputs):
        # Unpack inputs
        r_g  = float(inputs['r_g'])
        l_s  = float(inputs['l_s'])
        t_s  = float(inputs['t_s'])
        h_ys = float(inputs['h_ys'])
        h_ss = float(inputs['h_ss'])
        h_t  = float(inputs['h_t'])
        R_no = float(inputs['R_no'])
        y_bd = float(inputs['y_bd'])
        q    = float(inputs['q'])
        rho_Fes = float(inputs['rho_Fes'])
       
        # Radial deformation of Stator
        L_s             = l_s+t_s+0.1
        R_s             = r_g-inputs['g']-h_t-h_ys-h_ss
        constants_x_0   = shell_constant(R_s,t_s,L_s,0,v)
        constants_x_L   = shell_constant(R_s,t_s,L_s,L_s,v)
        f_d_denom1      = R_s/(E*((R_s)**2-(R_no)**2))*((1-v)*R_s**2+(1+v)*(R_no)**2)
        f_d_denom2      = t_s/(2*constants_x_0[0]*(constants_x_0[1])**3)*(constants_x_0[2]/(2*constants_x_0[3])*constants_x_0[4]-constants_x_0[5]/constants_x_0[3]*constants_x_0[6]-0.5*constants_x_0[7])
        f               = q*(R_s)**2*t_s/(E*(h_ys+h_ss)*(f_d_denom1+f_d_denom2))
        outputs['u_as'] = (q*(R_s)**2)/(E*(h_ys+h_ss))-f*0/(constants_x_L[0]*(constants_x_L[1])**3)*((constants_x_L[2]/(2*constants_x_L[3])*constants_x_L[4] -constants_x_L[5]/constants_x_L[3]*constants_x_L[6]-1/2*constants_x_L[7]))+y_bd
        outputs['u_as'] = np.abs(outputs['u_as'] + y_bd)
        outputs['u_allowable_s'] =2*r_g/1000*inputs['u_allow_pcent']/100
        
        # axial deformation of stator
        W_back_iron =  plate_constant(R_s+h_ss+h_ys+h_t,R_no,v,0.5*h_ys+h_ss+R_s,t_s)
        W_ssteel    =  plate_constant(R_s+h_ss+h_ys+h_t,R_no,v,R_s+h_ss*0.5,t_s)
        W_active    =  plate_constant(R_s+h_ss+h_ys+h_t,R_no,v,R_s+h_ss+h_ys+h_t*0.5,t_s)
        W_is        =  inputs['rho_Fe']*gravity*np.sin(phi)*(L_s-t_s)*h_ys
        y_ai1s      = -W_is*(0.5*h_ys+R_s)**4/(R_no*W_back_iron[0])*(W_back_iron[1]*W_back_iron[4]/W_back_iron[3]-W_back_iron[2])
        W_ss        =  rho_Fes*gravity*np.sin(phi)*(L_s-t_s)*h_ss
        y_ai2s      = -W_ss*(h_ss*0.5+h_ys+R_s)**4/(R_no*W_ssteel[0])*(W_ssteel[1]*W_ssteel[4]/W_ssteel[3]-W_ssteel[2])
        W_cu        =  np.sin(phi)*inputs['Mass_tooth_stator']/(2*np.pi*(R_s+h_ss+h_ys+h_t*0.5))
        y_ai3s      = -W_cu*(R_s+h_ss+h_ys+h_t*0.5)**4/(R_no*W_active[0])*(W_active[1]*W_active[4]/W_active[3]-W_active[2])
        w_disc_s    = rho_Fes*gravity*np.sin(phi)*t_s
        a_ii        = R_s+h_ss+h_ys+h_t
        r_oii       = R_no
        M_rb        = -w_disc_s*a_ii**2/W_ssteel[5]*(W_ssteel[6]*0.5/(a_ii*R_no)*(a_ii**2-r_oii**2)-W_ssteel[8])
        Q_b         =  w_disc_s*0.5/R_no*(a_ii**2-r_oii**2)
        y_aiis      =  M_rb*a_ii**2/W_ssteel[0]*W_ssteel[1]+Q_b*a_ii**3/W_ssteel[0]*W_ssteel[2]-w_disc_s*a_ii**4/W_ssteel[0]*W_ssteel[7]
        I           = np.pi*0.25*(R_s**4-(R_no)**4)
        F_ecc       = q*2*np.pi*inputs['K_rad']*r_g**2
        M_as        = F_ecc*L_s*0.5
        
        outputs['y_as'] = np.abs(y_ai1s+y_ai2s+y_ai3s+y_aiis+(R_s+h_ys+h_ss+h_t)*inputs['theta_bd'])+M_as*L_s**2*0/(2*E*I)
        outputs['y_allowable_s'] = L_s*inputs['y_allow_pcent']/100
        
        # Torsional deformation of stator
        J_ds            = 0.5*np.pi*((R_s+h_ys+h_ss+h_t)**4-R_no**4)
        J_cyls          = 0.5*np.pi*((R_s+h_ys+h_ss+h_t)**4-R_s**4)
        outputs['twist_s']= 180.0/np.pi*inputs['T_rated']/G*(t_s/J_ds+(L_s-t_s)/J_cyls)
        
        outputs['Structural_mass_stator'] = rho_Fes*(np.pi*((R_s+h_ys+h_ss+h_t)**2-(R_no)**2)*t_s+\
                                            np.pi*((R_s+h_ss)**2-R_s**2)*l_s)
        outputs['TC1s']  = (R_s+h_ys+h_ss+h_t)**2*L_s
        outputs['TC2s']  = inputs['T_rated']/(2*np.pi*inputs['Sigma'])
        outputs['TC_test_s'] = 1 if outputs['TC1s']>outputs['TC2s'] else 0


class PMSG_Constraints(om.ExplicitComponent):
    
    
    def setup(self):
        
        self.add_output('con_uar', val=0.0)
        self.add_output('con_yar', val=0.0)
        self.add_output('con_uas', val=0.0)
        self.add_output('con_yas', val=0.0)
        

        self.add_input('u_ar',0.0,units='m', desc='Radial deformation')
        self.add_input('y_ar',0.0, units ='m', desc='Axial deformation')
          
       
        self.add_input('u_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_input('y_allowable_r',0.0,units='m',desc='Allowable Radial deflection')
        self.add_input('z_allowable_r',0.0,units='m',desc='Allowable Circumferential deflection')
        
        self.add_input('u_as',0.0,units='m', desc='Radial deformation')
        self.add_input('y_as',0.0, units ='m', desc='Axial deformation')
        self.add_input('u_allowable_s',0.0,units='m',desc='Allowable Radial deflection as a percentage of air gap diameter')
        self.add_input('y_allowable_s',0.0,units='m',desc='Allowable Axial deflection as a percentage of air gap diameter')
        self.add_input('z_allowable_s',0.0,units='m',desc='Allowable Circumferential deflection')
        
        
    def compute(self, inputs, outputs):
    
        outputs['con_uar'] = 1e3*(inputs['u_allowable_r'] - inputs['u_ar'])
        outputs['con_yar'] = 1e3*(inputs['y_allowable_r'] - inputs['y_ar'])
        outputs['con_uas'] = 1e3*(inputs['u_allowable_s'] - inputs['u_as'])
        outputs['con_yas'] = 1e3*(inputs['y_allowable_s'] - inputs['y_as'])
        
        

####################################################Cost Analysis#######################################################################
class PMSG_Cost(om.ExplicitComponent):
    """ Provides a material cost estimate for a PMSG _arms generator. Manufacturing costs are excluded"""
    # Inputs
    # Specific cost of material by type
    
    def setup(self):
        
        self.add_input('C_Cu',0.0, units='USD/kg', desc='Specific cost of copper')
        self.add_input('C_Fe',0.0, units='USD/kg', desc='Specific cost of magnetic steel/iron')
        self.add_input('C_Fes',0.0, units='USD/kg', desc='Specific cost of structural steel')
        self.add_input('C_PM',0.0, units='USD/kg' , desc='Specific cost of Magnet')
        
        # Mass of each material type
        self.add_input('Copper',0.0, units ='kg',desc='Copper mass')
        self.add_input('Iron',0.0, units = 'kg', desc='Iron mass')
        self.add_input('mass_PM', 0.0, units ='kg',desc='Magnet mass')
        
        self.add_input('Structural_mass',0.0, units='kg', desc='Structural mass')
        # Outputs
        self.add_output('Costs',0.0,units ='USD', desc='Total cost')
        
    def compute(self, inputs, outputs):
    
        # Material cost as a function of material mass and specific cost of material
        
        K_gen=inputs['Copper']*inputs['C_Cu']+inputs['Iron']*inputs['C_Fe']+inputs['C_PM']*inputs['mass_PM']
        
        Cost_str=inputs['C_Fes']*inputs['Structural_mass']
        
        outputs['Costs']=K_gen +Cost_str
		
  


class PMSG_Outer_rotor_Opt(om.Group):
    
    def setup(self):
        
        ivc = om.IndepVarComp()
        ivc.add_discrete_output('Eta_target',0.0,desc='Target drivetrain efficiency')
        ivc.add_output('P_av_v', 0.0,units='W',desc='Rated Power')
        ivc.add_output('T_rated', 0.0,units='N*m',desc='Torque')
        ivc.add_output('N_nom', 0.0,units='rpm',desc='rated speed')
        ivc.add_output('r_g', 0.0,units='m',desc='Air gap radius')
        ivc.add_output('l_s', 0.0,units='m',desc='Core length')
        ivc.add_output('h_s', 0.0,units='m',desc='Slot height')
        ivc.add_output('p', 0.0,desc='Pole pairs')
        ivc.add_output('h_m', 0.0,units='m',desc='Magnet height' )
        ivc.add_output('h_yr', 0.0,units='m',desc='Rotor yoke height'  )
        ivc.add_output('h_ys', 0.0,units='m',desc='Stator yoke height')
        ivc.add_output('B_tmax',0.0,desc='Teeth flux density')
        ivc.add_output('t_r', 0.0,units='m',desc='Rotor disc thickness')
        ivc.add_output('t_s', 0.0,units='m',desc='Stator disc thickness' )
        ivc.add_output('h_ss',0.0,units='m',desc='Stator rim thickness')
        ivc.add_output('h_sr', 0.0,units='m',desc='Rotor rim thickness')    
        ivc.add_output('rho_Fe', 0.0,units='kg/m**3',desc='Electrical steel density')
        ivc.add_output('rho_Fes', 0.0,units='kg/m**3',desc='Structural steel density')
        ivc.add_output('u_allow_pcent', 0.0,desc='% radial deflection')
        ivc.add_output('y_allow_pcent', 0.0,desc='% axial deflection')
        ivc.add_output('Sigma', 0.0,units='N/m**2',desc='shear stress')
        ivc.add_output('z_allow_deg',0.0,units='deg',desc='Allowable torsional twist')
                
        self.add_subsystem('ivc',ivc, promotes =['*'])
        self.add_subsystem('PMSG_active',PMSG_active(), promotes =['*'])
        self.add_subsystem('PMSG_rotor_inactive',PMSG_rotor_inactive(), promotes =['*'])
        self.add_subsystem('PMSG_stator_inactive',PMSG_stator_inactive(), promotes =['*'])
        self.add_subsystem('PMSG_Cost',PMSG_Cost(), promotes =['*'])
        self.add_subsystem('PMSG_Constraints',PMSG_Constraints(), promotes =['*'])
        
       
		
if __name__ == '__main__':

    prob = om.Problem()
    
 
    prob.model = PMSG_Outer_rotor_Opt()
    
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'CONMIN' #'COBYLA'
    prob.driver.opt_settings['IPRINT'] = 4
    prob.driver.opt_settings['ITRM'] = 3
    prob.driver.opt_settings['ITMAX'] = 1000
    # prob.driver.opt_settings['DELFUN'] = 1e-3
    # prob.driver.opt_settings['DABFUN'] = 1e-3
    prob.driver.opt_settings['IFILE'] = 'CONMIN_PMSG_disc.out'
    # prob.root.deriv_options['type']='fd'
   
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
    
    # structural constraints
    prob.model.add_constraint('con_uar',lower = 1e-2)
    prob.model.add_constraint('con_yar', lower = 1e-2)
    prob.model.add_constraint('con_uas', lower = 1e-2)
    prob.model.add_constraint('con_yas',lower = 1e-2)   
    
    prob.model.add_objective('Mass',scaler=1e-6)
    
    prob.model.add_design_var('r_g', lower=3.0, upper=5 ) 
    prob.model.add_design_var('l_s', lower=1.5, upper=3.5 )  
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
    
    prob.setup()
    
    # --- Design Variables ---

    #Initial design variables for a PMSG designed for a 15MW turbine
    prob['P_av_v']        =   10.321e6
    prob['T_rated']       =   10.25e6         #rev 1 9.94718e6
    prob['P_mech']        =   10.71947704e6   #rev 1 9.94718e6
    prob['N_nom']         =   10              #8.68                # rpm 9.6
    prob['r_g']           =   4.0             # rev 1  4.92
    prob['l_s']           =   1.7             # rev 2.3
    prob['h_s']           =  0.7              # rev 1 0.3
    prob['p']             =   70              #100.0    # rev 1 160
    prob['h_m']           =   0.005           # rev 1 0.034
    prob['h_ys']          =   0.04            # rev 1 0.045
    prob['h_yr']          =   0.06            # rev 1 0.045
    prob['b']             =   2.
    prob['c']             =5.0
    prob['B_tmax']        =   1.9
    prob['E_p']           =   3300/np.sqrt(3)

    # Specific costs
    prob['C_Cu']          =   4.786
    prob['C_Fe']          =   0.556
    prob['C_Fes']         =   0.50139
    prob['C_PM']          =   95.0

    #Material properties
    prob['rho_Fe']        =   7700.0           #Steel density
    prob['rho_Fes']       =   7850.0           #Steel density
    prob['rho_Copper']    =   8900.0           # Kg/m3 copper density
    prob['rho_PM']        =   7400.0           # magnet density
    prob['resist_Cu']     =   1.8*10**(-8)*1.4 # Copper resisitivty

    #Support structure parameters
    prob['R_no']          = 1.1		# Nose outer radius
    prob['R_sh']          = 1.34	# Shaft outer radius =(2+0.25*2+0.3*2)*0.5
    prob['t_r']           =   0.05 	# Rotor disc thickness
    prob['h_sr']          =   0.04      # Rotor cylinder thickness

    prob['t_s']           =   0.053 	# Stator disc thickness
    prob['h_ss']          =   0.04      # Stator cylinder thickness
    prob['y_sh']          =   0.0005*0	# Shaft deflection
    prob['theta_sh']      =   0.00026*0 # Slope at shaft end

    prob['y_bd']          =   0.0005*0  # deflection at bedplate
    prob['theta_bd']      =  0.00026*0  # Slope at bedplate end
    prob['u_allow_pcent'] =  8.5        # % radial deflection
    prob['y_allow_pcent'] =  1.0        # % axial deflection
    prob['z_allow_deg']   =  0.05       # torsional twist
    prob['Sigma']         =  60.0e3     # Shear stress
  
    prob.model.approx_totals(method='fd')
    #prob.run_driver()
    prob.run_model()
    
    #prob.model.list_outputs(values = True, hierarchical=True)
    raw_data = {'Parameters': ['Rating','Air gap diameter','Overall Outer diameter', 'Stator length', 'l/d ratio',\
                'Slot_aspect_ratio','Pole pitch','Slot pitch', 'Stator slot height','Stator slotwidth','Stator tooth width','Stator tooth height',\
                'Stator yoke height', 'Rotor yoke height', 'Magnet height', 'Magnet width', 'Peak air gap flux density fundamental',\
                'Peak stator yoke flux density','Peak rotor yoke flux density','Flux density above magnet','Maximum Stator flux density',\
                'Maximum tooth flux density','Pole pairs', 'Generator output frequency', 'Generator output phase voltage',\
                'Generator Output phase current', 'Stator resistance','Synchronous inductance', 'Stator slots','Stator turns',\
                'Conductor cross-section','Stator Current density ','Specific current loading','Generator Efficiency ','Iron mass',\
                'Magnet mass','Copper mass','Structural Mass','**************************','Rotor disc thickness','Rotor rim thickness',\
                'Stator disc thickness','Stator rim thickness','Rotor radial deflection','Rotor axial deflection','Rotor twist','Torsional Constraint',\
                'Stator radial deflection','Stator axial deflection','Stator twist','Torsional Constraint','**************************','Total Mass','Total Material Cost'],
    'Values': [prob['P_mech']/1000000,2*prob['r_g'],prob['R_out']*2,prob['l_s'],prob['K_rad'],prob['Slot_aspect_ratio'],\
                prob['tau_p']*1000,prob['tau_s']*1000,prob['h_s']*1000,prob['b_s']*1000,prob['b_t']*1000,prob['h_t']*1000,prob['h_ys']*1000,\
                prob['h_yr']*1000,prob['h_m']*1000,prob['b_m']*1000,prob['B_g'],prob['B_symax'],prob['B_rymax'],prob['B_pm1'],\
                prob['B_smax'],prob['B_tmax'],prob['p'],prob['f'],prob['E_p'],prob['I_s'],prob['R_s'],prob['L_s'],prob['S'],\
                prob['N_s'],prob['A_Cuscalc'],prob['J_actual'],prob['A_1']/1000,prob['gen_eff'],prob['Iron']/1000,\
                prob['mass_PM']/1000,prob['Copper']/1000,prob['Structural_mass']/1000,'************************',prob['t_r']*1000,\
                prob['h_sr']*1000,prob['t_s']*1000,prob['h_ss']*1000,prob['u_ar']*1000,prob['y_ar']*1000,prob['twist_r'],prob['TC_test_r']\
                ,prob['u_as']*1000,prob['y_as']*1000,prob['twist_s'],prob['TC_test_s'],'**************************',prob['Mass']/1000,prob['Costs']/1000],
    'Limit': ['','','','','(0.15-0.3)','(4-10)','','','','','','','','','','','<2','<2','<2',prob['B_g'],'<2','<2','','','500<E_p<10000','','','','','','',\
    '3-6','85','>=95.4','','','','','************************','','','','',prob['u_allowable_r']*1000,prob['y_allowable_r']*1000,\
    prob['z_allow_deg'],'',prob['u_allowable_s']*1000,prob['y_allowable_s']*1000,prob['z_allow_deg'],'','**************************','',''],
    'Units':['MW','m','m','m','','','mm','mm','mm','mm','mm','mm','mm','mm','mm','mm','T','T','T','T','T','T','-','Hz','V','A','ohm/phase','p.u','slots','turns','mm^2','A/mm^2',\
            'kA/m','%','tons','tons','tons','tons','************','mm','mm','mm','mm','mm','mm','deg','','mm','mm','deg','','************','tons','k$']}
    
    df=pd.DataFrame(raw_data, columns=['Parameters','Values','Limit','Units'])
    print(df)
    df.to_excel('PMSG_Revised_new'+str(prob['P_mech'][0]/1e6)+'_Outer_MW_VOltage.xlsx')
	
	
