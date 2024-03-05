import os
import unittest

import openmdao.api as om
# from wisdem.rotorse.rotor_cost import StandaloneRotorCost, initialize_omdao_prob
# from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
# from wisdem.glue_code.gc_PoseOptimization import PoseOptimization
from wisdem import run_wisdem




class TestRC(unittest.TestCase):
    def testBladeJointSizerBAR_USC(self):

        wisdem_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        example_dir = os.path.join(wisdem_dir, "examples", "03_blade")  # get path example 03_blade
        fname_modeling_options = os.path.join(example_dir, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir, "BAR_USC.yaml")
        fname_analysis_options = os.path.join(example_dir, "analysis_options_no_opt.yaml")
        accuracy = 0
        wt_opt, _, _ = run_wisdem(fname_wt_input, fname_modeling_options,
                                                                fname_analysis_options)

        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.blade_mass"][0], 51002.33629166093, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.L_transition_joint"][0], -0.004017126893942291, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.t_reinf_ratio_joint"][0], 0.9899571827651443, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.w_reinf_ratio_joint"][0], 1.2087641190997038, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.n_joint_bolt"][0], 38.05368523091661, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.joint_mass"][0], 430.94139294963134, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.joint_material_cost"][0], 1857.5411987710845, places=accuracy)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRC))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
