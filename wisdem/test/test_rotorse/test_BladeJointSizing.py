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
        example_dir = os.path.join(wisdem_dir, "examples", "18_segmented_blade")  # get path example 03_blade
        fname_modeling_options = os.path.join(example_dir, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir, "BAR_USC.yaml")
        fname_analysis_options = os.path.join(example_dir, "analysis_options_no_opt.yaml")
        accuracy = 0
        wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options,
                                                                fname_analysis_options)
        # TODO prevent model from creating output files. Also, maybe there's a way to just run the parts of the code being tested?
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.blade_mass"][0], 50063.19790792579, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.L_transition_joint"][0], 0.2427557944762739, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.t_sc_ratio_joint"][0], 2.3761676787299004, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.n_joint_bolt"][0], 36.53783575753797, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.joint_mass"][0], 449.6091546753399, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.joint_material_cost"][0], 1783.5469763758085, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_start_nd_bjs"][2, 20], 0.10727560035555678, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_start_nd_bjs"][3, 20], 0.5584724343531979, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_end_nd_bjs"][2, 20], 0.44034799099288907, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_end_nd_bjs"][3, 20], 0.8915448249905302, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_offset_y_bjs"][2, 20], 0.11525697296129567, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_offset_y_bjs"][3, 20], 0.11525697296129567, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_width_bjs"][2, 20], 1.97304313090705, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_width_bjs"][3, 20], 1.97304313090705, places=accuracy)


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
