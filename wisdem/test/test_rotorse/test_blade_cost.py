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

        fname_modeling_options = os.path.join(example_dir02, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir02, "IEA-3p4-130-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["nacelle"] = False
        modeling_options["flags"]["tower"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 944.1382619536412, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 126.16553815344044, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 4667.410277624349, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 7200.344422290056, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 69743.70423837316, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 55750.58279955469, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 639.1295835528967, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 126133.41662148076, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 3356.5273806578602, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 9970.668209900952, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 544.9659865519155, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 3591.8770497095948, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 16725.174839866406, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 7758.261680842697, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 41947.475147529425, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 168080.8917690102, places=accuracy)

    def testBladeCostIEA10(self):

        fname_modeling_options = os.path.join(example_dir02, "modeling_options_iea10.yaml")
        fname_wt_input = os.path.join(example_dir02, "IEA-10-198-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["nacelle"] = False
        modeling_options["flags"]["tower"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 2130.5884620752204, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 234.2572952143954, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 7621.331066101973, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 12913.891638670682, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 195975.7612745464, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 126076.27789720206, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 2594.8151841626523, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 324646.8543559111, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 8702.329785471115, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 34012.403410785664, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 1974.2734007848132, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 11292.044540855932, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 37822.88336916062, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 23001.894329633295, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 116805.82883669145, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 441452.68319260253, places=accuracy)

    def testBladeCostIEA15(self):

        fname_modeling_options = os.path.join(example_dir02, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir02, "IEA-15-240-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["nacelle"] = False
        modeling_options["flags"]["tower"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 3133.7875857844165, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 276.2981847459007, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 8885.387764565836, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 15045.283018454009, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 361485.95863826084, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 185284.01067270082, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 3628.3329952322097, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 550398.3023061939, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 25795.210670398752, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 32212.11155993386, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 2369.9813509855767, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 18315.99973893161, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 55585.20320181025, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 37665.27934867489, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 171943.78587073492, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 722342.0881769288, places=accuracy)

    def testBladeCostBAR_USC(self):

        fname_modeling_options = os.path.join(example_dir03, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir03, "BAR_USC.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["nacelle"] = False
        modeling_options["flags"]["tower"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc_in.total_labor_hours"][0], 1854.9370312841427, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_non_gating_ct"][0], 206.20994524870673, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_metallic_parts_cost"][0], 6482.0815222645215, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_consumable_cost_w_waste"][0], 9195.272976087883, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_blade_mat_cost_w_waste"][0], 281022.3956005813, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_labor"][0], 109716.38161241644, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_utility"][0], 1889.570499707424, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.blade_variable_cost"][0], 392628.3477127052, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_equipment"][0], 10183.740562358538, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_tooling"][0], 17476.586422417342, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_building"][0], 867.2481476986652, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_maintenance_cost"][0], 7910.447829768591, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_labor_overhead"][0], 32914.91448372493, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.cost_capital"][0], 18882.029771341622, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.blade_fixed_cost"][0], 88234.96721730969, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_blade_cost"][0], 480863.3149300149, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_labor_hours"][0], 373.5411645718723, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_non_gating_ct"][0], 74.06265954079467, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_metallic_parts_cost"][0], 3428.7667570837384, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_consumable_cost_w_waste"][0], 3045.423658004067, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_blade_mat_cost_w_waste"][0], 30481.925123449975, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_labor"][0], 22023.441358254942, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_utility"][0], 169.66659621941906, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.blade_variable_cost"][0], 52675.03307792433, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_equipment"][0], 1415.270445312764, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_tooling"][0], 2551.7103294623803, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_building"][0], 180.10100291044174, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_maintenance_cost"][0], 1190.5030343316168, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_labor_overhead"][0], 6607.032407476482, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.cost_capital"][0], 2783.3786202360243, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.blade_fixed_cost"][0], 14727.995839729709, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_blade_cost"][0], 67403.02891765404, places=accuracy)
        self.assertAlmostEqual(wt_opt["total_bc.total_blade_cost"][0], 566866.343847669, places=accuracy)


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
