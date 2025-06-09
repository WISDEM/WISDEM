import os
import unittest

import openmdao.api as om

from wisdem.rotorse.blade_cost import StandaloneBladeCost, initialize_omdao_prob
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython
from wisdem.glue_code.gc_PoseOptimization import PoseOptimization

wisdem_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
example_dir02 = os.path.join(wisdem_dir, "examples", "02_reference_turbines")  # get path example 02_reference_turbines
example_dir03 = os.path.join(wisdem_dir, "examples", "03_blade")  # get path example 03_blade
fname_opt_options = os.path.join(example_dir02, "analysis_options.yaml")
accuracy = 0


class TestBC(unittest.TestCase):
    def testBladeCostIEA3p4(self):
        fname_modeling_options = os.path.join(example_dir02, "modeling_options_iea3p4.yaml")
        fname_wt_input = os.path.join(example_dir02, "IEA-3p4-130-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["blade"] = True
        modeling_options["flags"]["drivetrain"] = False
        modeling_options["flags"]["tower"] = False
        modeling_options["user_elastic"]["blade"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 974.9306430717324, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 129.49541374812043, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 4667.408050425351, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 7352.03816427098, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 73061.84564924994, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 57592.662087911005, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 677.3692887284064, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 131331.87702588935, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 3422.2615152395774, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 10698.091234061443, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 615.2126134243939, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 3818.8543396549344, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 17277.7986263733, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 8206.50453359744, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 44038.72286235109, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 175370.59988824045, places=accuracy)

    def testBladeCostIEA10(self):
        fname_modeling_options = os.path.join(example_dir02, "modeling_options_iea10.yaml")
        fname_wt_input = os.path.join(example_dir02, "IEA-10-198-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["blade"] = True
        modeling_options["flags"]["drivetrain"] = False
        modeling_options["flags"]["tower"] = False
        modeling_options["user_elastic"]["blade"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 1986.758487494103, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 232.67157996262412, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 7621.331066101973, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 12909.592687574073, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 200625.38632374577, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 117582.98961141727, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 3044.743043652825, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 321253.1189788159, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 8925.576654949757, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 37051.75561203497, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 2857.5017727684367, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 12927.513687227623, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 35274.89688342518, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 25628.589867774248, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 122665.83447818022, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 443918.9534569961, places=accuracy)

    def testBladeCostIEA15(self):
        fname_modeling_options = os.path.join(example_dir02, "modeling_options_iea15.yaml")
        fname_wt_input = os.path.join(example_dir02, "IEA-15-240-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["blade"] = True
        modeling_options["flags"]["drivetrain"] = False
        modeling_options["flags"]["tower"] = False
        modeling_options["user_elastic"]["blade"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 3224.7079294956966, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 286.74832776122395, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 8885.378951968672, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 15449.199713513908, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 373248.92899153935, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 190723.83470951696, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 4097.0520851797555, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 568069.8157862361, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 26972.511608125664, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 40421.47113134616, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 3217.0541251523264, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 21116.904974448436, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 57217.15041285509, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 42338.17386322677, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 191283.26611515446, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 759353.0819013906, places=accuracy)

    def testBladeCostBAR_USC(self):
        fname_modeling_options = os.path.join(example_dir03, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir03, "BAR_USC.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["blade"] = True
        modeling_options["flags"]["drivetrain"] = False
        modeling_options["flags"]["tower"] = False
        modeling_options["user_elastic"]["blade"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt.run_model()

        # Seems like this used to be a jointed blade (with groups, rc_in and rc_out) but now is not?
        self.assertAlmostEqual(wt_opt["rc_in.total_labor_hours"][0], 1927.7917436423027, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_non_gating_ct"][0], 216.48813756723342, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_metallic_parts_cost"][0], 6482.081656404547, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_consumable_cost_w_waste"][0], 9544.750659016909, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_blade_mat_cost_w_waste"][0], 291145.97155238746, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_labor"][0], 114083.18409255406, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_utility"][0], 2070.1239297046773, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.blade_variable_cost"][0], 407299.27957464615, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_equipment"][0], 10789.006912069297, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_tooling"][0], 21493.114049038515, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_building"][0], 1194.2378082155187, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_maintenance_cost"][0], 9187.586382532503, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_labor_overhead"][0], 34224.955227766215, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.cost_capital"][0], 21104.295122252643, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.blade_fixed_cost"][0], 97993.1955018747, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_blade_cost"][0], 505292.4750765208, places=accuracy)
        
        self.assertAlmostEqual(wt_opt["rc_out.total_labor_hours"][0], 376.3788724669887, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_non_gating_ct"][0], 74.27405292931572, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_metallic_parts_cost"][0], 3428.7668145723205, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_consumable_cost_w_waste"][0], 3047.517386308622, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_blade_mat_cost_w_waste"][0], 30311.35853288354, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_labor"][0], 22191.541222021126, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_utility"][0], 170.5320820429768, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.blade_variable_cost"][0], 52673.43183694764, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_equipment"][0], 1498.4932691136132, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_tooling"][0], 2548.2868240772104, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_building"][0], 182.44777955979856, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_maintenance_cost"][0], 1226.060534969557, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_labor_overhead"][0], 6657.462366606337, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.cost_capital"][0], 2837.864625496365, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.blade_fixed_cost"][0], 14950.61539982288, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_blade_cost"][0], 67624.04723677052, places=accuracy)
        self.assertAlmostEqual(wt_opt["total_bc.total_blade_cost"][0], 591516.5223132913, places=accuracy)

if __name__ == '__main__':
    unittest.main()
