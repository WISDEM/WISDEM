import os
import unittest

import wisdem.glue_code.gc_LoadInputs as gcl

test_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))),
    "examples",
    "02_reference_turbines",
)


class TestLoadInputs(unittest.TestCase):
    def setUp(self):
        fname_wt_input = os.path.join(test_dir, "nrel5mw.yaml")
        fname_modeling_options = os.path.join(test_dir, "modeling_options_nrel5.yaml")
        fname_analysis_options = os.path.join(test_dir, "analysis_options.yaml")

        self.myobj = gcl.WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_analysis_options)

    def testRunFlags(self):
        self.myobj.wt_init["airfoils"] = {}
        self.myobj.wt_init["components"]["blade"] = {}
        self.myobj.wt_init["components"].pop("tower")
        self.myobj.set_run_flags()

        self.assertTrue(self.myobj.modeling_options["flags"]["airfoils"])
        self.assertTrue(self.myobj.modeling_options["flags"]["blade"])
        self.assertFalse(self.myobj.modeling_options["flags"]["tower"])

    def testOptFlags(self):
        self.myobj.analysis_options["design_variables"]["test1"] = {}
        self.myobj.analysis_options["design_variables"]["test1"]["flag"] = False
        self.myobj.analysis_options["design_variables"]["test1"]["test2"] = {}
        self.myobj.analysis_options["design_variables"]["test1"]["test2"]["flag"] = False
        self.myobj.analysis_options["design_variables"]["test1"]["test2"]["test3"] = {}
        self.myobj.analysis_options["design_variables"]["test1"]["test2"]["test3"]["flag"] = False

        self.myobj.set_opt_flags()
        self.assertFalse(self.myobj.analysis_options["opt_flag"])

        self.myobj.analysis_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"] = 500
        self.myobj.analysis_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"] = 600
        self.myobj.analysis_options["design_variables"]["blade"]["n_opt_struct"] = [700, 800]
        self.myobj.set_opt_flags()
        self.assertEqual(
            self.myobj.analysis_options["design_variables"]["blade"]["aero_shape"]["twist"]["n_opt"],
            self.myobj.modeling_options["WISDEM"]["RotorSE"]["n_span"],
        )
        self.assertEqual(
            self.myobj.analysis_options["design_variables"]["blade"]["aero_shape"]["chord"]["n_opt"],
            self.myobj.modeling_options["WISDEM"]["RotorSE"]["n_span"],
        )
        self.assertEqual(
            self.myobj.analysis_options["design_variables"]["blade"]["n_opt_struct"][0],
            self.myobj.modeling_options["WISDEM"]["RotorSE"]["n_span"],
        )
        self.assertEqual(
            self.myobj.analysis_options["design_variables"]["blade"]["n_opt_struct"][1],
            self.myobj.modeling_options["WISDEM"]["RotorSE"]["n_span"],
        )


if __name__ == "__main__":
    unittest.main()
