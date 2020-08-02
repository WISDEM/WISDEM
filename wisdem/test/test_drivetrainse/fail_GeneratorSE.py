import numpy as np
import numpy.testing as npt
import unittest
import wisdem.drivetrainse.generator_models as gm

class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.inputs['Overall_eff']    = 93
        self.inputs['machine_rating'] = 5e6
        self.inputs['T_rated']        = 4.143289e6

        self.inputs['rho_Fe']         = 7700.0
        self.inputs['rho_Fes']        = 7850.0
        self.inputs['rho_Copper']     = 8900.0
        self.inputs['rho_PM']         = 7450.0

        self.inputs['B_r']            = 1.2
        self.inputs['E']              = 2e11
        self.inputs['G']              = 79.3e9
        self.inputs['P_Fe0e']         = 1.0
        self.inputs['P_Fe0h']         = 4.0
        self.inputs['S_N']            = -0.002
        self.inputs['alpha_p']        = 0.5*np.pi*0.7
        self.inputs['b_r_tau_r']      = 0.45
        self.inputs['b_ro']           = 0.004
        self.inputs['b_s_tau_s']      = 0.45
        self.inputs['b_so']           = 0.004
        self.inputs['cofi']           = 0.85
        self.inputs['freq']           = 60
        self.inputs['h_i']            = 0.001
        self.inputs['h_sy0']          = 0.0
        self.inputs['h_w']            = 0.005
        self.inputs['k_fes']          = 0.9
        self.inputs['k_fillr']        = 0.7
        self.inputs['k_fills']        = 0.65
        self.inputs['k_s']            = 0.2
        self.discrete_inputs['m']     = 3
        self.inputs['mu_0']           = np.pi*4e-7
        self.inputs['mu_r']           = 1.06
        self.inputs['p']              = 3.0
        self.inputs['phi']            = np.deg2rad(90)
        self.discrete_inputs['q1']    = 6
        self.discrete_inputs['q2']    = 4
        self.inputs['ratio_mw2pp']    = 0.7
        self.inputs['resist_Cu']      = 1.8e-8*1.4
        self.inputs['sigma']          = 40e3
        self.inputs['v']              = 0.3
        self.inputs['y_tau_p']        = 1.0
        self.inputs['y_tau_pr']       = 10. / 12
        
    def testPMSG_Arms(self):

        myobj = gm.PMSG_Arms()

        self.inputs['n_nom']   = 12.1
        self.inputs['sigma']   = 48.373e3
        self.inputs['rad_ag']  = 3.26
        self.inputs['len_s']   = 1.60
        self.inputs['h_s']     = 0.070
        self.inputs['tau_p']   = 0.080
        self.inputs['h_m']     = 0.009
        self.inputs['h_ys']    = 0.075
        self.inputs['h_yr']    = 0.075
        self.inputs['n_s']     = 5.0
        self.inputs['b_st']    = 0.480
        self.inputs['n_r']     = 5.0
        self.inputs['b_arm']   = 0.530
        self.inputs['d_r']     = 0.700
        self.inputs['d_s']     = 0.350
        self.inputs['t_wr']    = 0.06
        self.inputs['t_ws']    = 0.06
        self.inputs['D_shaft'] = 2*0.43
        self.inputs['q1']      = 1

        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs['generator_mass']*1e-3, 101.05, 2)

	
    def testPMSG_disc(self):
		
        myobj = gm.PMSG_Disc()

        self.inputs['n_nom']   = 12.1
        self.inputs['sigma']   = 48.373e3
        self.inputs['rad_ag']  = 3.49
        self.inputs['len_s']   = 1.5
        self.inputs['h_s']     = 0.06
        self.inputs['tau_p']   = 0.07
        self.inputs['h_m']     = 0.0105
        self.inputs['h_ys']    = 0.085
        self.inputs['h_yr']    = 0.055
        self.inputs['n_s']     = 5.0
        self.inputs['b_st']    = 0.460
        self.inputs['t_d']     = 0.105
        self.inputs['d_s']     = 0.350
        self.inputs['t_ws']    = 0.150
        self.inputs['D_shaft'] = 2*0.43
        self.inputs['q1']      = 1
        
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs['generator_mass']*1e-3, 122.5, 1)


    def testEESG(self):

        myobj = gm.EESG()

        self.inputs['n_nom']   = 12.1
        self.inputs['sigma']   = 48.373e3
        self.inputs['rad_ag']  = 3.2
        self.inputs['len_s']   = 1.4
        self.inputs['h_s']     = 0.060
        self.inputs['tau_p']   = 0.170
        self.inputs['I_f']     = 69
        self.inputs['N_f']     = 100
        self.inputs['h_ys']    = 0.130
        self.inputs['h_yr']    = 0.120
        self.inputs['n_s']     = 5
        self.inputs['b_st']    = 0.470
        self.inputs['n_r']     = 5
        self.inputs['b_arm']   = 0.480
        self.inputs['d_r']     = 0.510
        self.inputs['d_s']     = 0.400
        self.inputs['t_wr']    = 0.140
        self.inputs['t_ws']    = 0.070
        self.inputs['D_shaft'] = 2*0.43
        self.inputs['q1']      = 2
        
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs['generator_mass']*1e-3, 147.9, 1)


    def testSCIG(self):

        myobj = gm.SCIG()
        
        self.inputs['n_nom']              = 1200.0
        self.inputs['Gearbox_efficiency'] = 0.955
        self.inputs['cofi']               = 0.9
        self.inputs['y_tau_p']            = 12./15.
        self.inputs['sigma']              = 21.5e3
        self.inputs['rad_ag']             = 0.55
        self.inputs['len_s']              = 1.30
        self.inputs['h_s']                = 0.090
        self.inputs['I_0']                = 140 
        self.inputs['B_symax']            = 1.4
        self.inputs['q1']                 = 6
        self.inputs['h_0']                = 0.05
        
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs['generator_mass']*1e-3, 23.7, 1)
        

    def testDFIG(self):

        myobj = gm.DFIG()

        self.inputs['n_nom']              = 1200.0
        self.inputs['Gearbox_efficiency'] = 0.955
        self.inputs['cofi']               = 0.9
        self.inputs['y_tau_p']            = 12./15.
        self.inputs['sigma']              = 21.5e3
        self.inputs['rad_ag']             = 0.61
        self.inputs['len_s']              = 0.49
        self.inputs['h_s']                = 0.08
        self.inputs['I_0']                = 40.0
        self.inputs['B_symax']            = 1.3
        self.inputs['S_Nmax']             = -0.2
        self.inputs['k_fillr']            = 0.55
        self.inputs['q1']                 = 5
        self.inputs['h_0']                = 0.1
        
        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs['generator_mass']*1e-3, 25.2, 1)
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGenerators))
    return suite


if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())

    
