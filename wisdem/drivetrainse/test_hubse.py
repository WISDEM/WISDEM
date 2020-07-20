import unittest
import os
from wisdem.drivetrainse.hub import Hub_System
import openmdao.api as om


class TestRegression(unittest.TestCase):
    
    def testHubSE(self):
        
        hub_prob = om.Problem(model=Hub_System())
        hub_prob.setup()
        
        hub_prob['pitch_system.blade_mass']         = 17000.
        hub_prob['pitch_system.BRFM']               = 1.e+6
        hub_prob['pitch_system.n_blades']           = 3
        hub_prob['pitch_system.scaling_factor']     = 0.54
        hub_prob['pitch_system.rho']                = 7850.
        hub_prob['pitch_system.Xy']                 = 371.e+6

        hub_prob['hub_shell.blade_root_diameter']   = 4.5
        hub_prob['hub_shell.n_blades']              = 3
        hub_prob['hub_shell.flange_t2shell_t']      = 4.
        hub_prob['hub_shell.flange_OD2hub_D']       = 0.5
        hub_prob['hub_shell.flange_ID2flange_OD']   = 0.8
        hub_prob['hub_shell.rho']                   = 7200.
        hub_prob['hub_shell.in2out_circ']           = 1.2 
        hub_prob['hub_shell.max_torque']            = 199200777.51
        hub_prob['hub_shell.Xy']                    = 200.e+6
        hub_prob['hub_shell.stress_concentration']  = 2.5
        hub_prob['hub_shell.reserve_factor']        = 2.0
        hub_prob['hub_shell.metal_cost']            = 3.00

        hub_prob['spinner.n_front_brackets']        = 3
        hub_prob['spinner.n_rear_brackets']         = 3
        hub_prob['spinner.n_blades']                = 3
        hub_prob['spinner.blade_root_diameter']     = hub_prob['hub_shell.blade_root_diameter']
        hub_prob['spinner.clearance_hub_spinner']   = 0.5
        hub_prob['spinner.spin_hole_incr']          = 1.2
        hub_prob['spinner.gust_ws']                 = 70
        hub_prob['spinner.load_scaling']            = 1.5
        hub_prob['spinner.composite_Xt']            = 60.e6
        hub_prob['spinner.composite_SF']            = 1.5
        hub_prob['spinner.composite_rho']           = 1600.
        hub_prob['spinner.Xy']                      = 225.e+6
        hub_prob['spinner.metal_SF']                = 1.5
        hub_prob['spinner.metal_rho']               = 7850.
        hub_prob['spinner.composite_cost']          = 7.00
        hub_prob['spinner.metal_cost']              = 3.00

        hub_prob.run_model()

        self.assertAlmostEqual(hub_prob['pitch_system.pitch_mass'][0],6202.76603773585)
        self.assertAlmostEqual(hub_prob['hub_shell.hub_mass'][0],     79417.52737564275)
        self.assertAlmostEqual(hub_prob['hub_shell.hub_diameter'][0], 6.235382907247958)
        self.assertAlmostEqual(hub_prob['hub_shell.hub_cost'][0],     238252.58212692826)
        self.assertAlmostEqual(hub_prob['hub_shell.hub_cm'][0],       2.8681137824204033)
        self.assertAlmostEqual(hub_prob['hub_shell.hub_I'][0],        514625.57739416)
        self.assertAlmostEqual(hub_prob['spinner.spinner_mass'][0],   1704.965737284796)
        self.assertAlmostEqual(hub_prob['spinner.spinner_diameter'][0], 7.235382907247958)
        self.assertAlmostEqual(hub_prob['spinner.spinner_cost'][0],   9395.05078685929)
        self.assertAlmostEqual(hub_prob['spinner.spinner_cm'][0],     3.117691453623979)
        self.assertAlmostEqual(hub_prob['spinner.spinner_I'][0],      14876.04367239)
        self.assertAlmostEqual(hub_prob['adder.hub_system_mass'][0],  87325.2591506634)
        self.assertAlmostEqual(hub_prob['adder.hub_system_cost'][0],  247647.63291378756)
        self.assertAlmostEqual(hub_prob['adder.hub_system_cm'][0],    2.8729866151188928)
        self.assertAlmostEqual(hub_prob['adder.hub_system_I'][0],     589792.50695335)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
