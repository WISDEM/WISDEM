import os
import unittest

import openmdao.api as om
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from wisdem.glue_code.gc_PoseOptimization import PoseOptimization
from wisdem.rotorse.rotor_cost import StandaloneRotorCost, initialize_omdao_prob

wisdem_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
example_dir =  os.path.join(wisdem_dir, "examples", "02_reference_turbines")  # get path example 03_blade
fname_opt_options = os.path.join(example_dir ,"analysis_options.yaml")
accuracy = 0

class TestRC(unittest.TestCase):

    def testRotorCostIEA3p4(self):

        fname_modeling_options = os.path.join(example_dir, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir, "IEA-3p4-130-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        wt_opt = om.Problem(model=StandaloneRotorCost(modeling_options=modeling_options ,opt_options=opt_options))
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 942.2559441138253, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 125.4774355655832, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 4667.410277624349, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 7200.344422290056, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 69743.70423837316, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 31792.702225211335, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 640.5836160844093, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 102176.9900796689, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 3334.4885934967465, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 9970.66820990104, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 544.3309391994405, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 3582.2994780221934, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 9537.8106675634, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 7223.320836086661, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 34192.91872426948, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 136369.9088039384, places=accuracy)

    def testRotorCostIEA10(self):

        fname_modeling_options = os.path.join(example_dir, "modeling_options_iea10.yaml")
        fname_wt_input = os.path.join(example_dir, "IEA-10-198-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        wt_opt = om.Problem(model=StandaloneRotorCost(modeling_options=modeling_options ,opt_options=opt_options))
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
        wt_opt = om.Problem(model=StandaloneRotorCost(modeling_options=modeling_options ,opt_options=opt_options))
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 3144.546512161421, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 275.3052204776816, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 8885.377394584766, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 15040.127644116868, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 360963.0848169348, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 106238.93978148817, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 3690.088667977554, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 470892.11326640047, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 25454.939245455862, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 32316.52632162093, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 2373.102687509132, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 18200.343134652652, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 31871.68193444645, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 35768.36501439226, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 145984.9583380773, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 616877.0716044777, places=accuracy)

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
