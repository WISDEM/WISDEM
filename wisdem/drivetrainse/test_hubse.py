import unittest
import os
from wisdem.drivetrainse.hubse_openmdao import Hub_System
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

        hub_prob['hub_shell.blade_root_diameter']   = 4.
        hub_prob['hub_shell.n_blades']              = 3
        hub_prob['hub_shell.flange_t2shell_t']      = 4.
        hub_prob['hub_shell.flange_OD2hub_D']       = 0.5
        hub_prob['hub_shell.flange_ID2flange_OD']   = 0.8
        hub_prob['hub_shell.rho']                   = 7200.
        hub_prob['hub_shell.in2out_circ']           = 1.2 
        hub_prob['hub_shell.max_torque']            = 30.e+6
        hub_prob['hub_shell.Xy']                    = 200.e+6
        hub_prob['hub_shell.stress_concentration']  = 2.5
        hub_prob['hub_shell.reserve_factor']        = 2.0
        hub_prob['hub_shell.metal_cost']            = 3.00

        hub_prob['spinner.n_front_brackets']        = 3
        hub_prob['spinner.n_rear_brackets']         = 3
        hub_prob['spinner.n_blades']                = 3
        hub_prob['spinner.blade_root_diameter']     = 4.
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

        self.assertAlmostEqual(hub_prob['pitch_system.mass'][0],        6202.76603773585)
        self.assertAlmostEqual(hub_prob['hub_shell.total_mass'][0],     11810.857858231573)
        self.assertAlmostEqual(hub_prob['hub_shell.outer_diameter'][0], 5.542562584220408)
        self.assertAlmostEqual(hub_prob['hub_shell.cost'][0],           35432.57357469472)
        self.assertAlmostEqual(hub_prob['hub_shell.cm'][0],             2.5438705751839086)
        self.assertAlmostEqual(hub_prob['spinner.diameter'][0],         6.542562584220408)
        self.assertAlmostEqual(hub_prob['spinner.total_mass'][0],       1496.307988376076)
        self.assertAlmostEqual(hub_prob['spinner.cost'][0],             8177.635093264784)
        self.assertAlmostEqual(hub_prob['spinner.cm'][0],               3.271281292110204)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
