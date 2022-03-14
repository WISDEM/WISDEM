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
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.blade_mass"][0], 50097.73914846, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.L_transition_joint"][0], 0.23860868, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.t_sc_ratio_joint"][0], 2.32068153, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.n_joint_bolt"][0], 38.24367252, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.joint_mass"][0], 470.5980646, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.joint_material_cost"][0], 1866.81518121, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_start_nd_bjs"][2, 20], 0.10059220034742442, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_start_nd_bjs"][3, 20], 0.5586590919049519, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_end_nd_bjs"][2, 20], 0.4399273214865673, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_end_nd_bjs"][3, 20], 0.8979942130440948, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_offset_y_bjs"][2, 20], 0.13924107064259483, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_offset_y_bjs"][3, 20], 0.13924107064259483, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_width_bjs"][2, 20], 2.06515832, places=accuracy)
        self.assertAlmostEqual(wt_opt["rotorse.rs.bjs.layer_width_bjs"][3, 20], 2.06515832, places=accuracy)


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
