from unittest import TestCase
import pandas as pd
from wisdem.landbosse.model import GridConnectionCost

class TestGridConnectionCost(TestCase):

    def setUp(self):

        self.input_dict = dict()
        self.input_dict['distance_to_interconnect_mi'] = 5
        self.input_dict['interconnect_voltage_kV'] = 130
        self.input_dict['new_switchyard'] = True
        self.input_dict['turbine_rating_MW'] = 1.5
        self.input_dict['num_turbines'] = 1
        self.input_dict['rotor_diameter_m'] = 75
        self.project_name = 'Project_1'
        self.output_dict = dict()

    def test_GridConnectionCost(self):
        """
        Black box test to check whether module is ran successfully or not
        """
        run_TransmissionCost = GridConnectionCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        trial_run = run_TransmissionCost.run_module()


        if trial_run[0] == 0 :
            print('\n\n================== MODULE EXECUTION SUCCESS =========================\n')
            print(' GridConnectionCost module ran successfully. See the list of inputs'
                  '\n and outputs below used by the module in its calculations:')
            print( '\n=====================================================================\n')

        elif trial_run[0] == 1 :
            print('\n\n================== MODULE EXECUTION FAILURE ==================\n')
            print(' GridConnectionCost module failed to run successfully. See the list'
                  '\n of inputs below used by the module in its calculations:')
            print('\n================================================================\n')


        print('\nGiven below is the set of inputs fed into GridConnectionCost module:\n')
        for key, value in self.input_dict.items():
            print(key, ':', value)

        if trial_run[0] == 0:  # Only print outputs if module ran successfully.
            print('\nGiven below is the set of outputs calculated by the GridConnectionCost module:\n')
            for key, value in self.output_dict.items():
                if isinstance(value, pd.DataFrame):
                    print('\nNow printing DataFrame ->', key, ':\n', value)
                else:
                    print(key, ':', value)
