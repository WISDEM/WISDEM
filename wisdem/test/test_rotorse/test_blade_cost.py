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
        modeling_options["flags"]["blade"] = True
        modeling_options["flags"]["nacelle"] = False
        modeling_options["flags"]["tower"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 974.9322040457341, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 129.49749983873951, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 4667.408050425351, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 7352.174602266414, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 73111.34271089677, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 57592.75149280874, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 677.5268457218377, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 131381.62104942737, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 3422.282699631233, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 10699.403671715654, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 615.4214804607724, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 3819.3234438799245, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 17277.82544784262, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 8208.092919296532, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 44042.34966282674, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 175423.97071225411, places=accuracy)

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
        modeling_options["flags"]["nacelle"] = False
        modeling_options["flags"]["tower"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 2120.114976619193, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 232.7325246708642, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 7621.310200125126, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 12911.790488639614, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 203757.42484377683, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 125472.08278820755, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 3061.525355375679, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 332291.03298736, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 8926.999731580396, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 37096.29238681671, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 2863.937297439419, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 12942.931431450135, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 37641.624836462266, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 25877.288327010752, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 125349.07401075968, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 457640.10699811974, places=accuracy)

    def testBladeCostIEA15(self):
        fname_modeling_options = os.path.join(example_dir02, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir02, "IEA-15-240-RWT.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["direct"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["blade"] = True
        modeling_options["flags"]["nacelle"] = False
        modeling_options["flags"]["tower"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc.total_labor_hours"][0], 3225.970423321292, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_non_gating_ct"][0], 286.8691726009284, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_metallic_parts_cost"][0], 8885.378951968672, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_consumable_cost_w_waste"][0], 15452.72272332480, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_mat_cost_w_waste"][0], 373394.4858711899, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_labor"][0], 190799.0839263441, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_utility"][0], 4106.865948259457, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_variable_cost"][0], 568300.4357457935, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_equipment"][0], 26995.11322495800, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_tooling"][0], 40526.55505925335, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_cost_building"][0], 3228.385949075586, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_maintenance_cost"][0], 21156.35723835444, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_labor_overhead"][0], 57239.72517790324, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.cost_capital"][0], 42403.61902650416, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.blade_fixed_cost"][0], 191549.7556760488, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc.total_blade_cost"][0], 759850.1914218423, places=accuracy)

    def testBladeCostBAR_USC(self):
        fname_modeling_options = os.path.join(example_dir03, "modeling_options.yaml")
        fname_wt_input = os.path.join(example_dir03, "BAR_USC.yaml")
        wt_initial = WindTurbineOntologyPython(fname_wt_input, fname_modeling_options, fname_opt_options)
        wt_init, modeling_options, opt_options = wt_initial.get_input_data()
        modeling_options["WISDEM"]["RotorSE"]["flag"] = False
        modeling_options["WISDEM"]["DriveSE"]["flag"] = False
        modeling_options["WISDEM"]["TowerSE"]["flag"] = False
        modeling_options["flags"]["blade"] = True
        modeling_options["flags"]["nacelle"] = False
        modeling_options["flags"]["tower"] = False
        wt_opt = om.Problem(
            reports=False, model=StandaloneBladeCost(modeling_options=modeling_options, opt_options=opt_options)
        )
        wt_opt.setup(derivatives=False)
        myopt = PoseOptimization(wt_init, modeling_options, opt_options)
        wt_opt = myopt.set_initial(wt_opt, wt_init)
        wt_opt = initialize_omdao_prob(wt_opt, modeling_options, wt_init, opt_options)
        wt_opt.run_model()

        self.assertAlmostEqual(wt_opt["rc_in.total_labor_hours"][0], 1942.8819544723106, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_non_gating_ct"][0], 216.583706126828, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_metallic_parts_cost"][0], 6482.081656404547, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_consumable_cost_w_waste"][0], 9548.189259560568, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_blade_mat_cost_w_waste"][0], 291634.8402696372, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_labor"][0], 114976.391089742, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_utility"][0], 2096.5115602004776, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.blade_variable_cost"][0], 408707.7429195797, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_equipment"][0], 10794.9355969486, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_tooling"][0], 21552.342278459513, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_cost_building"][0], 1196.308056402001, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_maintenance_cost"][0], 9201.918671015363, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_labor_overhead"][0], 34492.9173269226, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.cost_capital"][0], 21153.372097916483, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.blade_fixed_cost"][0], 98391.79402766455, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_in.total_blade_cost"][0], 507099.5369472442, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_labor_hours"][0], 380.9886664587877, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_non_gating_ct"][0], 74.27226272287318, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_metallic_parts_cost"][0], 3428.7668145723205, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_consumable_cost_w_waste"][0], 3047.4391050025547, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_blade_mat_cost_w_waste"][0], 30310.936957848793, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_labor"][0], 22464.241165095576, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_utility"][0], 170.5875847656905, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.blade_variable_cost"][0], 52945.76570771007, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_equipment"][0], 1498.482616267636, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_tooling"][0], 2548.178298648845, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_cost_building"][0], 182.427366960888, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_maintenance_cost"][0], 1226.0144146439354, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_labor_overhead"][0], 6739.272349528673, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.cost_capital"][0], 2843.7086692285984, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.blade_fixed_cost"][0], 15038.083715278575, places=accuracy)
        self.assertAlmostEqual(wt_opt["rc_out.total_blade_cost"][0], 67983.84942298864, places=accuracy)
        self.assertAlmostEqual(wt_opt["total_bc.total_blade_cost"][0], 593683.3863702328, places=accuracy)

if __name__ == '__main__':
    unittest.main()
