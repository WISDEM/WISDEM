"""
test_Plant_FinanceSE_gradients.py

Created by Katherine Dykes on 2014-01-07.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
from commonse.utilities import check_gradient_unit_test
from plant_financese.basic_finance.basic_finance import fin_cst_component, fin_cst_assembly
from plant_financese.nrel_csm_fin.nrel_csm_fin import fin_csm_component, fin_csm_assembly

# Basic Finance Tests

class Test_fin_cst_assembly(unittest.TestCase):

    def setUp(self):

        self.fin = fin_cst_assembly()

        self.fin.turbine_cost = 6087803.555 / 50
        self.fin.turbine_number = 50
        preventative_maintenance_cost = 401819.023
        land_lease_cost = 22225.395
        corrective_maintenance_cost = 91048.387
        self.fin.avg_annual_opex = preventative_maintenance_cost + corrective_maintenance_cost + land_lease_cost
        self.fin.bos_costs = 7668775.3
        self.fin.net_aep = 15756299.843

    def test_functionality(self):
        
        self.fin.run()
        
        self.assertEqual(round(self.fin.coe,4), 0.1307)


class Test_fin_cst_component(unittest.TestCase):

    def setUp(self):

        self.fin = fin_cst_component()

        self.fin.turbine_cost = 6087803.555 / 50
        self.fin.turbine_number = 50
        preventative_maintenance_cost = 401819.023
        land_lease_cost = 22225.395
        corrective_maintenance_cost = 91048.387
        self.fin.avg_annual_opex = preventative_maintenance_cost + corrective_maintenance_cost + land_lease_cost
        self.fin.bos_costs = 7668775.3
        self.fin.net_aep = 15756299.843

    def test_functionality(self):
        
        self.fin.run()
        
        self.assertEqual(round(self.fin.coe,4), 0.1307)

    def test_gradient(self):

        check_gradient_unit_test(self, self.fin)


# NREL CSM Finance Tests

class Test_fin_csm_assembly(unittest.TestCase):

    def setUp(self):

        self.fin = fin_csm_assembly()

        self.fin.turbine_cost = 6087803.555 / 50
        self.fin.turbine_number = 50
        preventative_opex = 401819.023
        lease_opex = 22225.395
        corrective_opex = 91048.387
        self.fin.avg_annual_opex = preventative_opex + corrective_opex + lease_opex
        self.fin.bos_costs = 7668775.3
        self.fin.net_aep = 15756299.843

    def test_functionality(self):
        
        self.fin.run()
        
        self.assertEqual(round(self.fin.coe,4), 0.1307)
        self.assertEqual(round(self.fin.lcoe,4), 0.1231)

class Test_fin_csm_component(unittest.TestCase):

    def setUp(self):

        self.fin = fin_csm_component()

        self.fin.turbine_cost = 6087803.555 / 50
        self.fin.turbine_number = 50
        preventative_opex = 401819.023
        lease_opex = 22225.395
        corrective_opex = 91048.387
        self.fin.avg_annual_opex = preventative_opex + corrective_opex + lease_opex
        self.fin.bos_costs = 7668775.3
        self.fin.net_aep = 15756299.843

    def test_functionality(self):
        
        self.fin.run()
        
        self.assertEqual(round(self.fin.coe,4), 0.1307)
        self.assertEqual(round(self.fin.lcoe,4), 0.1231)

    def test_gradient(self):

        check_gradient_unit_test(self, self.fin)

if __name__ == "__main__":
    unittest.main()
