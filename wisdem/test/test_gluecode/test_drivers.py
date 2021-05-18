import os
import unittest

from wisdem.inputs import load_yaml, write_yaml
from wisdem.glue_code.runWISDEM import run_wisdem

test_dir = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    + os.sep
    + "examples"
    + os.sep
    + "05_tower_monopile"
    + os.sep
)

fname_wt_input = test_dir + "nrel5mw_tower.yaml"
fname_modeling_options = test_dir + "modeling_options.yaml"
fname_analysis_options = test_dir + "analysis_options.yaml"


class TestRegression(unittest.TestCase):
    def test_GA(self):
        analysis_options = load_yaml(fname_analysis_options)

        solver = "GA"
        analysis_options["driver"]["optimization"]["solver"] = solver
        analysis_options["driver"]["optimization"]["max_gen"] = 1

        new_fname_analysis_options = f"{fname_analysis_options.split('/')[-1].split('.')[0]}_{solver}.yaml"
        write_yaml(analysis_options, new_fname_analysis_options)

        wt_opt, modeling_options, opt_options = run_wisdem(
            fname_wt_input, fname_modeling_options, new_fname_analysis_options
        )

    def test_nelder(self):
        analysis_options = load_yaml(fname_analysis_options)

        solver = "Nelder-Mead"
        analysis_options["driver"]["optimization"]["solver"] = solver
        analysis_options["driver"]["optimization"]["adaptive"] = True
        analysis_options["driver"]["optimization"]["max_iter"] = 1

        new_fname_analysis_options = f"{fname_analysis_options.split('/')[-1].split('.')[0]}_{solver}.yaml"
        write_yaml(analysis_options, new_fname_analysis_options)

        wt_opt, modeling_options, opt_options = run_wisdem(
            fname_wt_input, fname_modeling_options, new_fname_analysis_options
        )

    def test_cobyla(self):
        analysis_options = load_yaml(fname_analysis_options)

        solver = "COBYLA"
        analysis_options["driver"]["optimization"]["solver"] = solver
        analysis_options["driver"]["optimization"]["rhobeg"] = 0.5
        analysis_options["driver"]["optimization"]["max_iter"] = 1

        new_fname_analysis_options = f"{fname_analysis_options.split('/')[-1].split('.')[0]}_{solver}.yaml"
        write_yaml(analysis_options, new_fname_analysis_options)

        wt_opt, modeling_options, opt_options = run_wisdem(
            fname_wt_input, fname_modeling_options, new_fname_analysis_options
        )

    def test_ld_slsqp(self):
        analysis_options = load_yaml(fname_analysis_options)

        solver = "LD_SLSQP"
        analysis_options["driver"]["optimization"]["solver"] = solver
        analysis_options["driver"]["optimization"]["max_iter"] = 1

        new_fname_analysis_options = f"{fname_analysis_options.split('/')[-1].split('.')[0]}_{solver}.yaml"
        write_yaml(analysis_options, new_fname_analysis_options)

        wt_opt, modeling_options, opt_options = run_wisdem(
            fname_wt_input, fname_modeling_options, new_fname_analysis_options
        )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRegression))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
