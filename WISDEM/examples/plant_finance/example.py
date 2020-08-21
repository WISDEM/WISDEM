import openmdao.api as om
from wisdem.plant_financese.plant_finance import PlantFinance


prob = om.Problem()
prob.model = PlantFinance()
prob.setup()

prob["machine_rating"] = 2.32 * 1.0e003  # kW
prob["tcc_per_kW"] = 1093  # USD/kW
prob["turbine_number"] = 87.0
prob["opex_per_kW"] = 43.56  # USD/kW/yr Source: 70 $/kW/yr, updated from report
prob["fixed_charge_rate"] = 0.079216644  # 7.9 % confirmed from report
prob["bos_per_kW"] = 517.0  # USD/kW from appendix of report
prob["wake_loss_factor"] = 0.15  # confirmed from report
prob["turbine_aep"] = 9915.95 * 1.0e003  # confirmed from report

prob.run_driver()

prob.model.list_inputs(units=True)
prob.model.list_outputs(units=True)
