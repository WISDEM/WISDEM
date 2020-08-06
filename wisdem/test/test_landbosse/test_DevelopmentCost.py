from unittest import TestCase
import pandas as pd
from wisdem.landbosse.model import DevelopmentCost


class TestDevelopmentCost(TestCase):
    def setUp(self):
        """
         This setUp() method executes before each test. It creates an
         instance attribute, self.instance, that refers to a DevelopmentCost
         instance under test. input_dict is the dictionary of inputs values
         that configures that instance.
         This is just a default dictionary. Methods that test behavior of
         specific variables customize the relevant key/value pairs.
         The key site_facility_building_area_df is set to None here. The
         dataframe is read from the csv in the site_facility tests below.
        """

        self.input_dict = dict()
        self.input_dict['development_materials_cost_usd'] = 10000
        self.input_dict['development_labor_cost_usd'] = 10000
        self.input_dict['development_equipment_cost_usd'] = 10000
        self.input_dict['development_mobilization_cost_usd'] = 10000
        self.input_dict['development_other_cost_usd'] = 10000
        self.project_name = 'project 1'
        self.output_dict = dict()


    def test_DevelopmentCostModule(self):
        """
        Black box test to check whether module is ran successfully or not
        """
        run_DevelopmentCost = DevelopmentCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        trial_run = run_DevelopmentCost.run_module()


        if trial_run[0] == 0 :
            print('\n\n================== MODULE EXECUTION SUCCESS =========================\n')
            print(' DevelopmentCost module ran successfully. See the list of inputs'
                  '\n and outputs below used by the module in its calculations:')
            print( '\n=====================================================================\n')

        elif trial_run[0] == 1 :
            print('\n\n====================================== MODULE EXECUTION FAILURE ======================================\n')
            print(' > DevelopmentCost module failed to run successfully. Error detected: ',trial_run[1] ,
                  '\n > Scroll below to see detailed information about error encountered.'
                  '\n > See the list of inputs below used by the module in its calculations:')
            print('\n========================================================================================================\n')


        print('\nGiven below is the set of inputs fed into DevelopmentCost module:\n')
        for key, value in self.input_dict.items():
            if isinstance(value, pd.DataFrame):
                print(key, ':\n', value)
            else:
                print(key, ':', value)

        if trial_run[0] == 0:  # Only print outputs if module ran successfully.
            print('\nGiven below is the set of outputs calculated by the DevelopmentCost module:\n')
            for key, value in self.output_dict.items():
                if isinstance(value, pd.DataFrame):
                    print('\nNow printing DataFrame ->', key, ':\n', value)
                else:
                    print(key, ':', value)