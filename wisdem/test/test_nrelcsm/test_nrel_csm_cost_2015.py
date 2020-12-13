import unittest

import numpy as np
from openmdao.api import Problem
from wisdem.nrelcsm.nrel_csm_cost_2015 import Turbine_CostsSE_2015


class TestNewAssembly(unittest.TestCase):
    def setUp(self):
        turbine = Turbine_CostsSE_2015(verbosity=False)
        self.prob = Problem(turbine)
        self.prob.setup()

        self.prob["blade_mass"] = 17650.67  # inline with the windpact estimates
        self.prob["hub_mass"] = 31644.5
        self.prob["pitch_system_mass"] = 17004.0
        self.prob["spinner_mass"] = 1810.5
        self.prob["lss_mass"] = 31257.3
        # bearingsMass'] = 9731.41
        self.prob["main_bearing_mass"] = 9731.41 / 2
        self.prob["gearbox_mass"] = 30237.60
        self.prob["hss_mass"] = 1492.45
        self.prob["generator_mass"] = 16699.85
        self.prob["bedplate_mass"] = 93090.6
        self.prob["yaw_mass"] = 11878.24
        self.prob["tower_mass"] = 434559.0
        self.prob["converter_mass"] = 1000.0
        self.prob["hvac_mass"] = 1000.0
        self.prob["cover_mass"] = 1000.0
        self.prob["platforms_mass"] = 1000.0
        self.prob["transformer_mass"] = 1000.0

        # other inputs
        self.prob["machine_rating"] = 5000.0
        self.prob["blade_number"] = 3
        self.prob["crane"] = True
        self.prob["main_bearing_number"] = 2

    def test1(self):

        self.prob.run_model()

        self.assertEqual(np.round(self.prob["rotor_cost"], 2), 1292397.85)
        self.assertEqual(np.round(self.prob["generator_cost"], 2), 207078.14)
        self.assertEqual(np.round(self.prob["transformer_cost"], 2), 18800.00)
        self.assertEqual(np.round(self.prob["turbine_cost"], 2), 4404316.13)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestNewAssembly))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
