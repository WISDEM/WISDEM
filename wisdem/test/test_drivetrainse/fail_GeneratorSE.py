
"""
test_GeneratorSE.py

Created by Latha Sethuraman on 2017-03-09.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np



# from drivese.drive_smooth import YawSystemSmooth, BedplateSmooth
import generatorse.PMSG_arms, generatorse.PMSG_disc, generatorse.DFIG,generatorse.SCIG,generatorse.EESG
from generatorse.PMSG_arms import PMSG_arms_Opt
from generatorse.EESG import EESG_Opt
from generatorse.PMSG_disc import PMSG_disc_Opt
from generatorse.DFIG import DFIG_Opt
from generatorse.SCIG import SCIG_Opt


class Test_PMSG_arms(unittest.TestCase):

    def setUp(self):

        self.PMSG_arms = PMSG_arms_Opt('CONMINdriver','PMSG_Cost.Costs',0)
 
        
        # Initial design variables for a DD PMSG designed for a 5MW turbine
        self.PMSG_arms.Eta_target = 93
        self.PMSG_arms.P_rated=5.0e6
        self.PMSG_arms.T_rated=4.143289e6
        self.PMSG_arms.N=12.1
        self.PMSG_arms.PMSG_r_s= 3.26
        self.PMSG_arms.PMSG_l_s= 1.6
        self.PMSG_arms.PMSG_h_s = 0.07
        self.PMSG_arms.PMSG_tau_p = 0.08
        self.PMSG_arms.PMSG_h_m = 0.009
        self.PMSG_arms.PMSG_h_ys = 0.075
        self.PMSG_arms.PMSG_h_yr = 0.075
        self.PMSG_arms.PMSG_n_s = 5
        self.PMSG_arms.PMSG_b_st = 0.480
        self.PMSG_arms.PMSG_n_r =5
        self.PMSG_arms.PMSG_b_r = 0.530
        self.PMSG_arms.PMSG_d_r = 0.7
        self.PMSG_arms.PMSG_d_s= 0.35
        self.PMSG_arms.PMSG_t_wr =0.06
        self.PMSG_arms.PMSG_t_ws =0.06
        self.PMSG_arms.PMSG_R_o =0.43
        
        # Specific costs
        self.PMSG_arms.C_Cu   =4.786
        self.PMSG_arms.C_Fe	= 0.556
        self.PMSG_arms.C_Fes =0.50139
        self.PMSG_arms.C_PM  =95
        
        #Material properties
        self.PMSG_arms.rho_Fe = 7700                 #Steel density
        self.PMSG_arms.rho_Fes = 7850                 #Steel density
        self.PMSG_arms.rho_Copper =8900                  # Kg/m3 copper density
        self.PMSG_arms.rho_PM =7450
        

       
    def test_functionality(self):
        
        self.PMSG_arms.run()
        
        self.assertEqual(round(self.PMSG_arms.Mass/1000,2), 101.05)


class Test_PMSG_disc(unittest.TestCase):
	
	def setUp(self):
		
				self.PMSG_disc=PMSG_disc_Opt('CONMINdriver','PMSG_Cost.Costs',0)
				
				
				
				# Initial design variables for a DD PMSG designed for a 1.5MW turbine
				self.PMSG_disc.Eta_target = 93
				self.PMSG_disc.P_rated=5.0e6
				self.PMSG_disc.T_rated=4143289.841
				self.PMSG_disc.N=12.1
				self.PMSG_disc.PMSG_r_s=3.49
				self.PMSG_disc.PMSG_l_s= 1.5
				self.PMSG_disc.PMSG_h_s = 0.06
				self.PMSG_disc.PMSG_tau_p = 0.07
				self.PMSG_disc.PMSG_h_m = 0.0105
				self.PMSG_disc.PMSG_h_ys = 0.085
				self.PMSG_disc.PMSG_h_yr = 0.055
				self.PMSG_disc.PMSG_n_s = 5
				self.PMSG_disc.PMSG_b_st = 0.46
				self.PMSG_disc.PMSG_t_d = 0.105
				self.PMSG_disc.PMSG_d_s= 0.350
				self.PMSG_disc.PMSG_t_ws =0.15
				
				self.PMSG_disc.PMSG_R_o =0.43
				
				# Specific costs
				self.PMSG_disc.C_Cu   =4.786
				self.PMSG_disc.C_Fe	= 0.556
				self.PMSG_disc.C_Fes =0.50139
				self.PMSG_disc.C_PM  =95
				
				#Material properties
				self.PMSG_disc.rho_Fe = 7700                 #Steel density
				self.PMSG_disc.rho_Fes = 7850                 #Steel density
				self.PMSG_disc.rho_Copper =8900                  # Kg/m3 copper density
				self.PMSG_disc.rho_PM =7450
				
				def test_functionality(self):
					
					self.PMSG_disc.run()
					self.assertEqual(round(self.PMSG_disc.Mass/1000,1), 122.5)

class Test_EESG(unittest.TestCase):

    def setUp(self):

        self.EESG = EESG_Opt('CONMINdriver','EESG_Cost.Costs',0)
        
        # Initial design variables for a DD EESG designed for a 5MW turbine
        self.EESG.Eta_target= 93
        self.EESG.P_rated=5.0e6
        self.EESG.T_rated=4.143289e6
        self.EESG.N_rated=12.1
        self.EESG.EESG_r_s=3.2
        self.EESG.EESG_l_s= 1.4
        self.EESG.EESG_h_s = 0.060
        self.EESG.EESG_tau_p = 0.17
        self.EESG.EESG_I_f = 69
        self.EESG.EESG_N_f = 100
        self.EESG.EESG_h_ys = 0.130
        self.EESG.EESG_h_yr = 0.12
        self.EESG.EESG_n_s = 5
        self.EESG.EESG_b_st = 0.47
        self.EESG.EESG_n_r =5
        self.EESG.EESG_b_r = 0.48
        self.EESG.EESG_d_r = 0.51
        self.EESG.EESG_d_s= 0.4
        self.EESG.EESG_t_wr =0.14
        self.EESG.EESG_t_ws =0.07
        self.EESG.EESG_R_o =0.43
        
        # Costs
        self.EESG.C_Cu   =4.786
        self.EESG.C_Fe	= 0.556
        self.EESG.C_Fes =0.50139
        
        #Material properties
        
        self.EESG.rho_Fe = 7700                 #Steel density
        self.EESG.rho_Fes = 7850                 #Steel density
        
        self.EESG.rho_Copper =8900                  # Kg/m3 copper density

	
	
    def test_functionality(self):
        
        self.EESG.run()
        
        self.assertEqual(round(self.EESG.Mass/1000,1), 147.9)


class Test_SCIG(unittest.TestCase):

    def setUp(self):

        self.SCIG = SCIG_Opt('CONMINdriver','SCIG_Cost.Costs',0)
        
        #Initial design variables for a SCIG designed for a 5MW turbine
        self.SCIG.SCIG_r_s= 0.55 #meter
        self.SCIG.SCIG_l_s= 1.3 #meter
        self.SCIG.SCIG_h_s = 0.09 #meter
        self.SCIG.SCIG_h_r = 0.050 #meter
        self.SCIG.SCIG_I_0 = 140   #Ampere
        self.SCIG.SCIG_B_symax = 1.4  #Tesla
        self.SCIG.Eta_target=93
        self.SCIG.SCIG_P_rated=5e6
        self.SCIG.Gearbox_efficiency=0.955
        self.SCIG.SCIG_N_rated=1200
        
        # Specific costs
        self.SCIG.C_Cu   =4.786                  # Unit cost of Copper $/kg
        self.SCIG.C_Fe	= 0.556                    # Unit cost of Iron $/kg
        self.SCIG.C_Fes =0.50139                   # specific cost of structure
        
        #Material properties
        self.SCIG.rho_Fe = 7700                 #Steel density
        self.SCIG.rho_Copper =8900                  # Kg/m3 copper density
	
	
    def test_functionality(self):
        
        self.SCIG.run()
        
        self.assertEqual(round(self.SCIG.Mass/1000,1), 23.7)
        
        
class Test_DFIG(unittest.TestCase):

    def setUp(self):

        self.DFIG = DFIG_Opt('CONMINdriver','DFIG_Cost.Costs',0)
        
        #Initial design variables for a DFIG designed for a 5MW turbine
        
        self.DFIG.Eta_target=93
        self.DFIG.DFIG_P_rated=5e6
        self.DFIG.Gearbox_efficiency=0.955
        self.DFIG.DFIG_r_s= 0.61 #meter
        self.DFIG.DFIG_l_s= 0.49 #meter
        self.DFIG.DFIG_h_s = 0.08 #meter
        self.DFIG.DFIG_h_r = 0.1 #meter
        self.DFIG.DFIG_I_0 = 40 #Ampere
        self.DFIG.DFIG_B_symax = 1.3 #Tesla
        self.DFIG.DFIG_S_Nmax = -0.2  #Tesla
        self.DFIG.DFIG_N_rated=1200
        
        # Specific costs
        self.DFIG.C_Cu   =4.786                  # Unit cost of Copper $/kg
        self.DFIG.C_Fe	= 0.556                    # Unit cost of Iron $/kg
        self.DFIG.C_Fes =0.50139                   # specific cost of structure
        
        #Material properties
        self.DFIG.rho_Fe = 7700                 #Steel density
        self.DFIG.rho_Copper =8900                  # Kg/m3 copper density
	
	
    def test_functionality(self):
        
        self.DFIG.run()
        
        self.assertEqual(round(self.DFIG.Mass/1000,1), 25.2)



if __name__ == "__main__":
    unittest.main()
    