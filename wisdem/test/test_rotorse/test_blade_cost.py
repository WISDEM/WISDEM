import os
import unittest

import openmdao.api as om
from wisdem.rotorse.blade_cost import StandaloneBladeCost, initialize_omdao_prob
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from wisdem.glue_code.gc_PoseOptimization import PoseOptimization

wisdem_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
example_dir = os.path.join(wisdem_dir, "examples", "02_reference_turbines")  # get path example 03_blade
fname_opt_options = os.path.join(example_dir, "analysis_options.yaml")
accuracy = 0


class TestBC(unittest.TestCase):
    def testBladeCostIEA3p4(self):

        fname_modeling_options = os.path.join(example_dir, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir, "IEA-3p4-130-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        wt_opt = om.Problem(model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options))
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 948.2286340184439, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 135.40494719056008, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 4667.410277624349, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 6893.931399813539, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 69437.29121589664, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 31995.094813170865, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 646.9942936858887, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 102079.3803227534, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 3521.6661356306527, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 9764.710077639173, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 571.4754551067743, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 3656.7906128026575, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 9598.52844395126, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 7347.4353623435945, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 34460.606087474116, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 136539.98641022752, places=accuracy)

    def testBladeCostIEA10(self):

        fname_modeling_options = os.path.join(example_dir, "modeling_options_iea10.yaml")
        fname_wt_input = os.path.join(example_dir, "IEA-10-198-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        wt_opt = om.Problem(model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options))
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 1958.635714717289, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 219.90827279432972, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 7621.331066101973, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 12200.172713827997, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 195262.04234970373, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 66185.46152958255, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 2235.047250747109, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 263682.5511300334, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 7953.279695998157, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 31598.877474314024, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 1911.5957835608222, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 10531.047214562495, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 19855.638458874764, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 20571.16360397285, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 92421.6022312831, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 356104.1533613165, places=accuracy)

    def testBladeCostIEA15(self):

        fname_modeling_options = os.path.join(example_dir, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir, "IEA-15-240-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        wt_opt = om.Problem(model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options))
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 3129.28115302, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 274.60008727, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 8885.39094536, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 15045.2932311, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 361486.20116216, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 105721.3493804, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 3625.71214243, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 470833.26268499, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 25172.62082018, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 32212.17519621, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 2367.0756174, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 18063.48710034, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 31716.40481412, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 35561.41412394, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 145093.17767219, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 615925.8906317754, places=accuracy)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBC))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
