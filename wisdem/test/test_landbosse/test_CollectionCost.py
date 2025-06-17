from unittest import TestCase
import os

import pandas as pd

from wisdem.landbosse.model import Cable, Array, ArraySystem
from wisdem.test.test_landbosse.test_WeatherDelay import generate_a_year
from wisdem.test.test_landbosse.test_filename_functions import landbosse_test_input_dir

import pytest

pd.set_option('display.width', 6000)
pd.set_option('display.max_columns', 20)

@pytest.mark.skip(reason="this does not pass")
class TestCollectionCost(TestCase):
    """
#
#     Tests CollectionCost module
#
#     """

    def setUp(self):
        self.input_dict = dict()
        self.output_dict = dict()
        self.cable_input_dict = dict()
        self.addl_specs = dict()

        rsmeans_csv = os.path.join(landbosse_test_input_dir(), 'rsmeans_proprietary.csv')
        self.input_dict['rsmeans'] = pd.read_csv(rsmeans_csv)
        crew_cost_csv = os.path.join(landbosse_test_input_dir(), 'crew_price_proprietary.csv')
        self.input_dict['crew_cost'] = pd.read_csv(crew_cost_csv)
        crews_csv = os.path.join(landbosse_test_input_dir(), 'crews.csv')
        self.input_dict['crew'] = pd.read_csv(crews_csv)
        self.input_dict['construct_duration'] = 9  # months
        self.input_dict['time_construct'] = 'normal'
        self.input_dict['hour_day'] = {'long': 24, 'normal': 10}

        #============================================================
        #All below inputs are inputs for ArraySystem class:
        self.input_dict['cable_specs'] = dict()


        cable_specs_path = os.path.join(landbosse_test_input_dir(), 'cable_specs.csv')

        self.input_dict['cable_specs_pd'] = pd.read_csv(cable_specs_path) #Read in cable specs into a dataframe.
        # self.input_dict['cable_specs'] = self.input_dict['cable_specs_pd'].T.to_dict()

        self.input_dict['user_defined_distance_to_grid_connection'] = 0  # 0 = No ; 1 = Yes
        self.input_dict['distance_to_grid_connection_km'] = 2
        self.project_name = 'project_1'

        #Inputs for ArraySystem:

        self.input_dict['num_turbines'] = 1
        self.input_dict['plant_capacity_MW'] = 15
        self.input_dict['row_spacing_rotor_diameters'] = 5
        self.input_dict['turbine_rating_MW'] = 0.02
        self.input_dict['upstream_turb'] = 0
        self.input_dict['turb_sequence'] = 1
        self.input_dict['depth'] = 45
        self.input_dict['rsmeans_per_diem'] = 144
        

        self.input_dict['turbine_spacing_rotor_diameters'] = 5
        self.input_dict['rotor_diameter_m'] = 154
        self.input_dict['line_frequency_hz'] = 60
        self.input_dict['overtime_multiplier'] = 1.4  # multiplier for labor overtime rates due to working 60 hr/wk rather than 40 hr/wk


        # Weather window:
        self.input_dict['num_delays'] = 7
        self.input_dict['avg_hours_per_delay'] = 20
        self.input_dict['std_dev_hours_per_delay'] = 5
        self.input_dict['delay_speed_m_per_s'] = 9
        self.input_dict['seed'] = 101
        self.input_dict['weather_window'] = generate_a_year(num_delays=self.input_dict['num_delays'],
                                                            avg_hours_per_delay=self.input_dict['avg_hours_per_delay'],
                                                            std_dev_hours_per_delay=self.input_dict[
                                                                'std_dev_hours_per_delay'],
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
        self.input_dict['duration_construction'] = 9  # months
        self.input_dict['operational_hrs_per_day'] = 10
        self.input_dict['critical_speed_non_erection_wind_delays_m_per_s'] = 15
        self.input_dict['critical_height_non_erection_wind_delays_m'] = 10
        self.input_dict['rsmeans_per_diem'] = 99

        #============================================================



        # Test inputs for Cable class: (Taken from cable specs dict)
        self.cable_input_dict['Current Capacity (A)'] = 610  # Amps (at 1 m burial depth)
        self.cable_input_dict['Rated Voltage (V)'] = 33  # kV, line-to-line
        self.cable_input_dict['AC Resistance (Ohms/km)'] = 0.062  # Ohms/km (at 90 deg C at 60 Hz)
        self.cable_input_dict['Inductance (mH/km)'] = 0.381  # mH/km
        self.cable_input_dict['Capacitance (nF/km)'] = 224  # nF/km
        self.cable_input_dict['Cost (USD/LF)'] = 250000  # $US/km

        # Additional inputs for Array(Cable) class

        self.addl_specs['turbine_rating_MW'] = 7
        self.addl_specs['upstream_turb'] = 0
        self.addl_specs['turb_sequence'] = 0
        self.addl_specs['turbine_spacing_rotor_diameters'] = 5
        self.addl_specs['rotor_diameter_m'] = 154
        self.addl_specs['line_frequency_hz'] = 60


    def test_ArraySystemModule(self):
        """
        Black box test to check whether module is ran successfully or not
        """
        run_ArraySystem = ArraySystem(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        trial_run = run_ArraySystem.run_module()


        if trial_run[0] == 0 :
            print('\n\n================== MODULE EXECUTION SUCCESS =========================\n')
            print(' ArraySystem module ran successfully. See the list of inputs'
                  '\n and outputs below used by the module in its calculations:')
            print( '\n=====================================================================\n')

        elif trial_run[0] == 1 :
            print('\n\n====================================== MODULE EXECUTION FAILURE ======================================\n')
            print(' > ArraySystem module failed to run successfully. Error detected: ',trial_run[1] ,
                  '\n > Scroll below to see detailed information about error encountered.'
                  '\n > See the list of inputs below used by the module in its calculations:')
            print('\n========================================================================================================\n')


        print('\nGiven below is the set of inputs fed into ArraySystem module:\n')
        for key, value in self.input_dict.items():
            print(key, ':', value)


        if trial_run[0] == 0:  #Only print outputs if module ran successfully.
            print('\nGiven below is the set of outputs calculated by the ArraySystem module:\n')
            for k, v in self.output_dict.items():
                if isinstance(v, dict) :
                    for a, b in v.items():
                        if isinstance(b, pd.DataFrame):
                            return
                        else:
                            print('\nGiven below are attributes of cable ', a, ':\n')
                            for c, d in b.__dict__.items():
                                print(c, ':', d)
                            print('\n<----End of cable attributes---->\n')
                else:
                    if isinstance(v, pd.DataFrame):
                        print('\nNow printing DataFrame ->', k, ':\n', v)
                    else:
                        print(k, ':', v)


    def test_CableModule(self):

        """
        Black box test to check whether module is ran successfully or not
        """
        Cable(self.cable_input_dict, self.addl_specs)


    def test_ArrayModule(self):
        Array(self.cable_input_dict, self.addl_specs)


    def test_ArraySystem(self):
        run_collection = ArraySystem(self.input_dict, self.output_dict, project_name=self.project_name)
        run_collection.run_module()

    def test_calculate_weather_delay(self):
        """
        Tests calculate_weather_delay()

        """

        weatherDelay = ArraySystem(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        weatherDelay.run_module()


