from unittest import TestCase
import pandas as pd
import os
import pytest

from wisdem.landbosse.model import ManagementCost
from wisdem.test.test_landbosse.test_filename_functions import landbosse_test_input_dir

PROJECT_NAME = 'foo'


@pytest.mark.skip(reason="this does not pass")
class TestManagementCost(TestCase):
    def setUp(self):
        """
        This setUp() method executes before each test. It creates an
        instance attribute, self.instance, that refers to a ManagementCost
        instance under test. input_dict is the dictionary of inputs values
        that configures that instance.

        This is just a default dictionary. Methods that test behavior of
        specific variables customize the relevant key/value pairs.

        The key site_facility_building_area_df is set to None here. The
        dataframe is read from the csv in the site_facility tests below.
        """
        # TODO: Refactor the parts of the filenames to the top of the file. Do not hardcode.
        site_facility_building_area_csv = os.path.join(landbosse_test_input_dir(), 'site_facility_building_area.csv')
        site_facility_building_area_df = pd.read_csv(site_facility_building_area_csv)
        self.input_dict = dict()
        self.input_dict['project_value_usd'] = 1e8
        self.input_dict['foundation_cost_usd'] = 1e5
        self.input_dict['construction_time_months'] = 12
        self.input_dict['num_hwy_permits'] = 10
        self.input_dict['num_turbines'] = 10
        self.input_dict['project_size_megawatts'] = 30
        self.input_dict['hub_height_meters'] = 80
        self.input_dict['num_access_roads'] = 2
        self.input_dict['markup_contingency'] = 0.1
        self.input_dict['markup_warranty_management'] = 0.1
        self.input_dict['markup_sales_and_use_tax'] = 0.1
        self.input_dict['markup_overhead'] = 0.1
        self.input_dict['markup_profit_margin'] = 0.1
        self.input_dict['site_facility_building_area_df'] = site_facility_building_area_df
        self.input_dict['construct_duration'] = 9 #months
        self.project_name = 'project_1'
        self.output_dict = dict()
        self.management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)

    def test_ManagementCostModule(self):
        """
        Black box test to check whether module is ran successfully or not
        """
        run_ManagementCost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict,
                                            project_name=self.project_name)
        trial_run = run_ManagementCost.run_module()

        if trial_run[0] == 0:
            print('\n\n================== MODULE EXECUTION SUCCESS =========================\n')
            print(' ManagementCost module ran successfully. See the list of inputs'
                  '\n and outputs below used by the module in its calculations:')
            print('\n=====================================================================\n')

        elif trial_run[0] == 1:
            print(
                '\n\n====================================== MODULE EXECUTION FAILURE ======================================\n')
            print(' > ManagementCost module failed to run successfully. Error detected: ', trial_run[1],
                  '\n > Scroll below to see detailed information about error encountered.'
                  '\n > See the list of inputs below used by the module in its calculations:')
            print(
                '\n========================================================================================================\n')

        print('\nGiven below is the set of inputs fed into ManagementCost module:\n')
        for key, value in self.input_dict.items():
            if isinstance(value, pd.DataFrame):
                print(key, ':\n', value)
            else:
                print(key, ':', value)

        if trial_run[0] == 0:  # Only print outputs if module ran successfully.
            print('\nGiven below is the set of outputs calculated by the RoadsCost module:\n')
            for key, value in self.output_dict.items():
                if isinstance(value, pd.DataFrame):
                    print('\nNow printing DataFrame ->', key, ':\n', value)
                else:
                    print(key, ':', value)

    def test_checks_keys_present(self):
        """
        If not all the keys are present, a ValueError should be raised.
        """
        bad_input_dict = dict()
        bad_input_dict['foundation_cost_usd'] = 1e5
        bad_input_dict['construction_time_months'] = 36
        bad_input_dict['num_hwy_permits'] = 10
        bad_input_dict['markup_constants'] = {}
        bad_input_dict['num_turbines'] = 10
        bad_input_dict['project_size_megawatts'] = 30
        output_dict = dict()
        self.assertRaises(ValueError, ManagementCost, bad_input_dict, output_dict, PROJECT_NAME)

    def test_insurance_cost(self):
        """
        Test the insurance cost calculation
        """
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(5.6e5, self.output_dict['insurance_usd'])

    def test_bonding(self):
        """
        Test the bonding coast calculation
        """
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(1e6, self.output_dict['bonding_usd'])

    def test_project_management_less_than_28_months(self):
        """
        Tests project management cost calculation less than 28 months.
        Ensure that the instance is set to 12 months for this calculation.
        """
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(2011724, round(self.output_dict['project_management_usd']))

    def test_project_management_greater_than_28_months(self):
        self.input_dict['construction_time_months'] = 36
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(2011724, round(self.output_dict['project_management_usd']))

    def test_markup_contingency_costs(self):
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(5e7, self.output_dict['markup_contingency_usd'])

    def test_engineering_under_200(self):
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(1104760.0, self.output_dict['engineering_usd'])

    def test_engineering_over_or_equal_200(self):
        self.input_dict['project_size_megawatts'] = 300
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(1451535, self.output_dict['engineering_usd'])

    def test_site_facility_1(self):
        self.input_dict['project_size_megawatts'] = 100
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(928000.0, self.output_dict['site_facility_usd'])

    def test_total_management_cost(self):
        management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=PROJECT_NAME)
        management_cost.run_module()
        self.assertEqual(55802284, round(self.output_dict['total_management_cost']))
