import os
import unittest

from wisdem.inputs import load_yaml, write_yaml
from wisdem.glue_code.runWISDEM import run_wisdem

floating_dir = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    + os.sep
    + "examples"
    + os.sep
    + "09_floating"
    + os.sep
)

inverse_design_dir = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    + os.sep
    + "examples"
    + os.sep
    + "16_inverse_design"
    + os.sep
)

fname_wt_input = floating_dir + "nrel5mw-spar_oc3.yaml"
fname_modeling_options = floating_dir + "modeling_options.yaml"
fname_analysis_options = inverse_design_dir + "analysis_options.yaml"


class TestInverseDesign(unittest.TestCase):
    def test_error_message(self):
        analysis_options = load_yaml(fname_analysis_options)

        analysis_options["inverse_design"]["val0"]["name"] = "fake_name"

        new_fname_analysis_options = f"{fname_analysis_options.split('/')[-1].split('.')[0]}_{0}.yaml"
        write_yaml(analysis_options, new_fname_analysis_options)

        with self.assertRaises(NameError) as raises_msg:
            wt_opt, modeling_options, opt_options = run_wisdem(
                fname_wt_input, fname_modeling_options, new_fname_analysis_options
            )

        exception = raises_msg.exception

        msg = "NLoptDriver: Tried to set read-only option 'equality_constraints'."

        self.assertIn("Attempted to connect from 'fake_name'", exception.args[0])


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestInverseDesign))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
