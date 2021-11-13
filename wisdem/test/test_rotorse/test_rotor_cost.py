import os
import unittest

import openmdao.api as om
from wisdem.rotorse.rotor_cost import StandaloneRotorCost, initialize_omdao_prob
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from wisdem.glue_code.gc_PoseOptimization import PoseOptimization

wisdem_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
example_dir = os.path.join(wisdem_dir, "examples", "02_reference_turbines")  # get path example 03_blade
fname_opt_options = os.path.join(example_dir, "analysis_options.yaml")
accuracy = 0


class TestRC(unittest.TestCase):
    def testRotorCostIEA3p4(self):

        fname_modeling_options = os.path.join(example_dir, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir, "IEA-3p4-130-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        wt_opt = om.Problem(model=StandaloneRotorCost(modeling_options=modeling_options, opt_options=opt_options))
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 942.2580090792371, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 125.47743556393156, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 4667.410277624349, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 7200.344422290056, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 69743.70423837316, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 31792.709454484186, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 638.5561660415175, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 102174.96985889885, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 3334.488593496729, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 9970.668209900952, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 544.3309391994371, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 3582.2994780221684, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 9537.812836345256, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 7223.287108846724, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 34192.88716581127, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 136367.85702471013, places=accuracy)

    def testRotorCostIEA10(self):

        fname_modeling_options = os.path.join(example_dir, "modeling_options_iea10.yaml")
        fname_wt_input = os.path.join(example_dir, "IEA-10-198-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        wt_opt = om.Problem(model=StandaloneRotorCost(modeling_options=modeling_options, opt_options=opt_options))
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 2104.669926420341, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 224.7722030790102, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 7621.331066101973, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 12913.891638670682, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 195975.7612745464, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 71148.57041818449, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 2582.596384597327, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 269706.9280773282, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 7975.896398389559, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 34012.403410785664, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 1960.741347506985, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 10985.232722089915, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 21344.571125455346, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 21348.39972589469, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 97627.24473012217, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 367334.17280745035, places=accuracy)

    def testRotorCostIEA15(self):

        fname_modeling_options = os.path.join(example_dir, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir, "IEA-15-240-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        wt_opt = om.Problem(model=StandaloneRotorCost(modeling_options=modeling_options, opt_options=opt_options))
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 3145.072189096041, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 275.33521169175623, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 8885.377394584766, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 15046.051961996029, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 372330.1181735587, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 106256.80891726053, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 3786.9785901227265, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 482373.90568094194, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 25537.757130096536, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 32290.706286056855, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 2371.9214744259534, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 18227.921627118856, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 31877.042675178156, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 36001.15960953355, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 146306.5088024099, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 628680.4144833519, places=accuracy)


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
