from unittest import TestCase
import pandas as pd
from wisdem.landbosse.model import ErectionCost
import os
from wisdem.landbosse.excelio import XlsxReader
from wisdem.landbosse.tests.model.test_filename_functions import wisdem.landbosse_test_input_dir
import logging
import sys


log = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
# out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setFormatter(logging.Formatter('%(message)s'))
out_hdlr.setLevel(logging.DEBUG)
log.addHandler(out_hdlr)
log.setLevel(logging.DEBUG)

class TestErectionCost(TestCase):
    def setUp(self):
        print('<><>><><><><><><><><> Begin load of ErectionCost test data <><>><><><><><><><><>')

        self.erection_cost_input_dict = None
        project = pd.Series(
            {
                'Turbine rating MW': 1.5,
                'Hub height m': 80,
                'Rotor diameter m': 77,
                'Number of breakdowns per 100 MW': 4,
                'Number of sections per tower': 3,
                'Number of RNA picks (at hub height)': 4,
                'Number of turbines': 100,
                'Breakpoint between base and topping (percent)': 0.75,
                'Fuel cost USD per gal': 1,
                'Rate of deliveries (turbines per week)': 10,
                'Wind shear exponent': 0.2,
                'Tower type': 'steel',
                'Foundation depth m': 3,
                'Rated Thrust (N)': 100,
                'Bearing Pressure (n/m2)': 100,
                'Critical Velocity (m/s)': 20,
                'Line Frequency (Hz)': 60,
                'Project ID': 'foo',
                'Turbine spacing (times rotor diameter)': 5,
                'Row spacing (times rotor diameter)': 5,
                'Number of turbines per day': 2,
                'Flag for user-defined home run trench length (0 = no; 1 = yes)': 1,
                'Combined Homerun Trench Length to Substation (km)': 67,
                'Total project construction time (months)': 9,
                '50-year Gust Velocity (m/s)': 60,
                'Road length adder (m)': 5000,
                'Percent of roads that will be constructed': 0.33,
                'Road Quality (0-1)': 0.6,
                'Line Frequency (Hz)': 60,
                'Row spacing (times rotor diameter)': 10,
                'Flag for user-defined home run trench length (0 = no; 1 = yes)': 0,
                'Combined Homerun Trench Length to Substation (km)': 50,
                'Distance to interconnect (miles)': 50,
                'Interconnect Voltage (kV)': 137,
                'New Switchyard (y/n)': 'y',
                'Non-Erection Wind Delay Critical Speed (m/s)': 15,
                'Non-Erection Wind Delay Critical Height (m)': 10,
                'Road width (ft)': 20,
                'Road thickness (in)': 8,
                'Crane width (m)': 12.2,
                'Number of highway permits': 10,
                'Number of access roads': 2,
                'Overtime multiplier': 1.4,
                'Allow same flag': 'n',
                'Markup contingency': 0.03,
                'Markup warranty management': 0.0002,
                'Markup sales and use tax': 0,
                'Markup overhead': 0.05,
                'Markup profit margin': 0.05
            }
        )

        xlsx_reader = XlsxReader()
        input_xlsx = os.path.join(landbosse_test_input_dir(), 'erection_cost_tests.xlsx')
        self.master_input_dict = xlsx_reader.create_master_input_dictionary(input_xlsx=input_xlsx, project_parameters=project)

        print('<><>><><><><><><><><> End load of ErectionCost test data <><>><><><><><><><><>')

    def key_value_logging_helper(self, _dict):
        for key, value in _dict.items():
            if isinstance(value, dict):
                print(f'{key}:')
                self.key_value_logging_helper(value)
            else:
                print(key, ':\n' if isinstance(value, pd.DataFrame) else ':', value)

    def test_ErectionCostModule_Black_Box(self):
        print('>>>>>>>>>>>>>>>>>>>>> Begin ErectionCost Module black box test <<<<<<<<<<<<<<<<<<<')
        print('===================== Inputs =====================')
        self.key_value_logging_helper(self.master_input_dict)
        print('===================== Outputs =====================')
        erection_cost_output_dict = dict()

        ec = ErectionCost(input_dict=self.master_input_dict, output_dict=erection_cost_output_dict, log=log, project_name='foo')
        ec.run_module()
        self.key_value_logging_helper(erection_cost_output_dict)
        print('>>>>>>>>>>>>>>>>>>>>> End ErectionCost Module black box test <<<<<<<<<<<<<<<<<<<')
        self.assertTrue(True)
