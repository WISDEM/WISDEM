import unittest

import numpy as np
import numpy.testing as npt
from openmdao.api import Group, Problem

import wisdem.plant_financese.plant_finance as pf


class TestPlantFinance(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.discrete_inputs = {}
        self.outputs = {}

        self.inputs["machine_rating"] = 1000.0
        self.inputs["tcc_per_kW"] = 1.2e3
        self.inputs["offset_tcc_per_kW"] = 0.0
        self.inputs["bos_per_kW"] = 7.7e3
        self.inputs["opex_per_kW"] = 7e2
        self.inputs["fixed_charge_rate"] = 0.12
        self.inputs["turbine_aep"] = 1.6e7
        self.inputs["plant_aep_in"] = 1.6e7 * 50
        self.inputs["wake_loss_factor"] = 0.15

        self.discrete_inputs["turbine_number"] = 50

        self.inputs["electricity_price"] = 0.04
        self.inputs["reserve_margin_price"] = 120
        self.inputs["capacity_credit"] = 1.0
        self.inputs["benchmark_price"] = 0.071

        self.mypfin = pf.PlantFinance(verbosity=True)

    def testRun(self):
        # Plant AEP way
        self.mypfin.compute(self.inputs, self.outputs, self.discrete_inputs, {})

        C = 0.12 * (1.2e3 + 7.7e3) + 7e2
        E = (1.6e7 * 50.0) / (1e3 * 50)
        V = (1.6e7 * 50.0) / (1e3 * 50) * 0.04 + 120.0 * 1.0
        lcoe = C / E
        plcoe = C / V * 0.071
        self.assertEqual(self.outputs["lcoe"], lcoe)
        self.assertEqual(self.outputs["plcoe"], plcoe)

        # Wake loss way
        self.inputs["plant_aep_in"] = 0.0
        self.mypfin.compute(self.inputs, self.outputs, self.discrete_inputs, {})

        C = 0.12 * (1.2e3 + 7.7e3) + 7e2
        E = (1.6e7 * 50.0 * (1 - 0.15)) / (1e3 * 50)
        V = (1.6e7 * 50.0 * (1 - 0.15)) / (1e3 * 50) * 0.04 + 120.0 * 1.0
        lcoe = C / E
        plcoe = C / V * 0.071
        self.assertEqual(self.outputs["lcoe"], lcoe)
        self.assertEqual(self.outputs["plcoe"], plcoe)

    def testDerivatives(self):
        prob = Problem(reports=False)
        root = prob.model = Group()
        root.add_subsystem("pf", pf.PlantFinance(), promotes=["*"])
        prob.setup()
        for k in self.inputs.keys():
            prob[k] = self.inputs[k]
        for k in self.discrete_inputs.keys():
            prob[k] = self.discrete_inputs[k]
        # prob.check_partials()


if __name__ == "__main__":
    unittest.main()
