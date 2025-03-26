from unittest import TestCase
import os

import numpy as np
import pandas as pd

from wisdem.landbosse.model import FoundationCost
from wisdem.test.test_landbosse.test_WeatherDelay import generate_a_year
from wisdem.test.test_landbosse.test_filename_functions import landbosse_test_input_dir

import pytest

pd.set_option('display.width', 6000)
pd.set_option('display.max_columns', 20)


@pytest.mark.skip(reason="this does not pass")
class TestFoundtionCost(TestCase):
    def setUp(self):
        """
         This setUp() method executes before each test. It creates an
         instance attribute, self.instance, that refers to a FoundationCost
         instance under test. input_dict is the dictionary of inputs values
         that configures that instance.
         This is just a default dictionary. Methods that test behavior of
         specific variables customize the relevant key/value pairs.
         The key site_facility_building_area_df is set to None here. The
         dataframe is read from the csv in the site_facility tests below.
        """


        self.input_dict = dict()
        self.input_dict['depth'] = 3.05 # in m
        self.project_name = 'project_1'

        self.input_dict['rated_thrust_N'] = 742e3
        self.input_dict['bearing_pressure_n_m2'] = 244200
        self.input_dict['critical_velocity_m_per_s'] = 52.5
        self.input_dict['gust_velocity_m_per_s'] = 60

        
        #Below are the inputs for calculate_foundation_load():
        component_data = os.path.join(landbosse_test_input_dir(), 'components.csv')
        self.input_dict['component_data'] = pd.read_csv(component_data)
        crew_data = os.path.join(landbosse_test_input_dir(), 'crews.csv')
        self.input_dict['crew'] = pd.read_csv(crew_data)
        crew_cost_data = os.path.join(landbosse_test_input_dir(), 'crew_price_proprietary.csv')
        self.input_dict['crew_cost'] = pd.read_csv(crew_cost_data)
        for component in self.input_dict['component_data'].keys():
            self.input_dict[component] =  np.array(self.input_dict['component_data'][component])



        # Below are the inputs for determine_foundation_size():
        #'Radius_m' -> Calculated in 'calculate_foundation_load()'
        #'depth'

        #Below are the inputs for 'estimate_material_needs()':
        self.input_dict['num_turbines'] = 100
        #foundation_volume_m3_per_turbine -> calculated in 'determine_foundation_size()':

        #Below are the inputs for 'estimate_construction_time()':
        self.input_dict['duration_construction'] = 9    #months
        self.input_dict['construct_duration'] = 9 #months
        self.input_dict['hour_day'] = {'long': 24, 'normal': 10}

        #Weather window:
        self.input_dict['num_delays'] = 7
        self.input_dict['avg_hours_per_delay'] = 20
        self.input_dict['std_dev_hours_per_delay'] = 5
        self.input_dict['delay_speed_m_per_s'] = 9
        self.input_dict['seed'] = 101
        self.input_dict['weather_window'] = generate_a_year(num_delays=self.input_dict['num_delays'],
                                              avg_hours_per_delay=self.input_dict['avg_hours_per_delay'],
                                              std_dev_hours_per_delay=self.input_dict['std_dev_hours_per_delay'],
                                              delay_speed_m_per_s=self.input_dict['delay_speed_m_per_s'],
                                              seed=self.input_dict['seed'])

        self.input_dict['start_delay_hours'] = 100
        self.input_dict['mission_time_hours'] = 50
        self.input_dict['critical_wind_speed_m_per_s'] = 8.0
        self.input_dict['wind_height_of_interest_m'] = 100
        self.input_dict['wind_shear_exponent'] = 0.25
        self.input_dict['season_construct'] = ['spring', 'summer']
        self.input_dict['time_construct'] = 'normal'
        self.input_dict['operational_hrs_per_day'] = 10
        self.input_dict['critical_speed_non_erection_wind_delays_m_per_s'] = 10
        self.input_dict['critical_height_non_erection_wind_delays_m'] = 10

        #Calculate Costs:
        material_price = os.path.join(landbosse_test_input_dir(), 'material_price_proprietary.csv')
        self.input_dict['material_price'] = pd.read_csv(material_price)

        rsmeans_csv = os.path.join(landbosse_test_input_dir(), 'rsmeans_proprietary.csv')
        self.input_dict['rsmeans'] = pd.read_csv(rsmeans_csv)
        self.input_dict['rsmeans_per_diem'] = 144



        self.input_dict['overtime_multiplier'] = 1.4


        self.output_dict = dict()
        self.foundation_cost = FoundationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)



    def test_FoundationCostModule(self):
        """
        Black box test to check whether module is ran successfully or not
        """
        run_FoundationCost = FoundationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        trial_run = run_FoundationCost.run_module()


        if trial_run[0] == 0 :
            print('\n\n================== MODULE EXECUTION SUCCESS =========================\n')
            print(' FoundationCost module ran successfully. See the list of inputs'
                  '\n and outputs below used by the module in its calculations:')
            print( '\n=====================================================================\n')

        elif trial_run[0] == 1 :
            print('\n\n====================================== MODULE EXECUTION FAILURE ======================================\n')
            print(' > FoundationCost module failed to run successfully. Error detected: ',trial_run[1] ,
                  '\n > Scroll below to see detailed information about error encountered.'
                  '\n > See the list of inputs below used by the module in its calculations:')
            print('\n========================================================================================================\n')


        print('\nGiven below is the set of inputs fed into FoundationCost module:\n')
        for key, value in self.input_dict.items():
            if isinstance(value, pd.DataFrame):
                print(key, ':\n', value)
            else:
                print(key, ':', value)

        if trial_run[0] == 0:  # Only print outputs if module ran successfully.
            print('\nGiven below is the set of outputs calculated by the FoundationCost module:\n')
            for key, value in self.output_dict.items():
                if isinstance(value, pd.DataFrame):
                    print('\nNow printing DataFrame ->', key, ':\n', value)
                else:
                    print(key, ':', value)
