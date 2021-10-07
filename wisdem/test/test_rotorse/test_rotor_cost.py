import os
import unittest

import openmdao.api as om
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from wisdem.glue_code.gc_PoseOptimization import PoseOptimization
from wisdem.rotorse.rotor_cost import StandaloneRotorCost, initialize_omdao_prob

wisdem_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
example_dir =  os.path.join(wisdem_dir, "examples", "02_reference_turbines")  # get path example 03_blade
fname_opt_options = os.path.join(example_dir ,"analysis_options.yaml")

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


        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 942.25594411, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 125.4774355655832, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 4667.410277624349, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 7200.344422290056, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 69743.70423837316, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 31792.702225211335, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 640.5836160844093, places=1)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 102176.9900796689, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 3334.4885934967465, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 9973.06061433774, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 544.3309392, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 3582.6822627320657, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 9537.8106675634, places=1)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 7223.847351785511, places=1)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 34196.2204291149, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 136373.2105087838, places=1)

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

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 2104.669926420341, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 224.7722030790102, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 7621.331066101973, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 12913.891638670682, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 195975.7612745464, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 71148.57041818449, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 2582.596384597327, places=1)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 269706.9280773282, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 7975.896398389559, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 34037.02892730745, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 1960.741347506985, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 10989.1728047334, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 21344.571125455346, places=1)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 21353.81926150578, places=1)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 97661.22986489852, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 367368.15794222674, places=1)

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

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 3144.546512161421, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 275.3052204776816, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 8885.377394584766, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 15040.127644116868, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 360963.0848169348, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 106238.93978148817, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 3690.088667977554, places=1)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 470892.11326640047, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 25454.939245455862, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 32368.053976900293, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 2373.102687509132, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 18208.58755949735, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 31871.68193444645, places=1)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 35779.70512019253, places=1)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 146056.0705240016, places=1)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 616948.183790402, places=1)

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
