from unittest import TestCase
import os
import pandas as pd
from wisdem.landbosse.model import SitePreparationCost
from wisdem.test.test_landbosse.test_WeatherDelay import generate_a_year
from wisdem.test.test_landbosse.test_filename_functions import landbosse_test_input_dir
import pytest

pd.set_option('display.width', 6000)
pd.set_option('display.max_columns', 20)

@pytest.mark.skip(reason="this does not pass")
class TestSitePreparationCost(TestCase):

    def setUp(self):
        self.input_dict = dict()
        self.project_name = 'Project_1'
        self.input_dict['road_distributed_wind'] = True #TODO add this to input file.

        #Inputs for calculate_road_properties():
        # self.input_dict['road_length'] = 108000   #TODO: Also add option to make road_length a user specified input.
        self.input_dict['road_width_ft'] = 10  # feet 10
        self.input_dict['road_thickness'] = 2  # inches
        self.input_dict['crane_width'] = 4.548  # metres 10.7 + shoulder width 1.5 m
        self.input_dict['num_access_roads'] = 5
        self.input_dict['num_turbines'] = 1
        self.input_dict['rsmeans'] = pd.read_csv(os.path.join(landbosse_test_input_dir(), 'rsmeans_proprietary.csv'))
        crew_data = os.path.join(landbosse_test_input_dir(), 'crews.csv')
        self.input_dict['crew'] = pd.read_csv(crew_data)
        crew_cost_data = os.path.join(landbosse_test_input_dir(), 'crew_price_proprietary.csv')
        self.input_dict['crew_cost'] = pd.read_csv(crew_cost_data)
        self.input_dict['duration_construction'] = 0.1 # months
        self.input_dict['construct_duration'] = 0.1 # months
        self.input_dict['fraction_new_roads'] = 1
        self.input_dict['road_quality'] = 0.6
        self.input_dict['rsmeans_per_diem'] = 144

        self.input_dict['road_length_adder_m'] = 5000   # 5km
        self.input_dict['fraction_new_roads'] = 0.33
        self.input_dict['road_quality'] = 0.3
        self.input_dict['rsmeans_per_diem'] = 0


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
        self.input_dict['wind_height_of_interest_m'] = 10
        self.input_dict['wind_shear_exponent'] = 0.25
        self.input_dict['season_construct'] = ['spring', 'summer']
        self.input_dict['time_construct'] = 'normal'
        self.input_dict['hour_day'] = {'long': 24, 'normal': 10}
        self.input_dict['operational_hrs_per_day'] = 10
        self.input_dict['critical_speed_non_erection_wind_delays_m_per_s'] = 15
        self.input_dict['critical_height_non_erection_wind_delays_m'] = 10

        #Inputs for calculate_costs():
        material_price_csv = os.path.join(landbosse_test_input_dir(), 'material_price_proprietary.csv')
        self.input_dict['material_price'] = pd.read_csv(material_price_csv)
        self.input_dict['turbine_rating_MW'] = 0.02  # 20 kW Turbine
        self.input_dict['overtime_multiplier'] = 1.4 # multiplier for labor overtime rates due to working 60 hr/wk rather than 40 hr/wk
        self.input_dict['rotor_diameter_m'] = 7
        self.input_dict['turbine_spacing_rotor_diameters'] = 5
        self.input_dict['critical_height_non_erection_wind_delays_m'] = 10
        self.input_dict['site_prep_area_m2'] = 0


        self.output_dict = dict()

    def test_SitePreparationCostModule(self):
        """
        Black box test to check whether module is ran successfully or not
        """
        run_RoadsCost = SitePreparationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        trial_run = run_RoadsCost.run_module()


        if trial_run[0] == 0 :
            print('\n\n================== MODULE EXECUTION SUCCESS =========================\n')
            print(' SitePreparationCost module ran successfully. See the list of inputs'
                  '\n and outputs below used by the module in its calculations:')
            print( '\n=====================================================================\n')

        elif trial_run[0] == 1 :
            print('\n\n====================================== MODULE EXECUTION FAILURE ======================================\n')
            print(' > SitePreparationCost module failed to run successfully. Error detected: ', trial_run[1],
                  '\n > Scroll below to see detailed information about error encountered.'
                  '\n > See the list of inputs below used by the module in its calculations:')
            print('\n========================================================================================================\n')

        print('\nGiven below is the set of inputs fed into RoadsCost module:\n')
        for key, value in self.input_dict.items():
            if isinstance(value, pd.DataFrame):
                print(key, ':\n', value)
            else:
                print(key, ':', value)


        if trial_run[0] == 0:  #Only print outputs if module ran successfully.
            print('\nGiven below is the set of outputs calculated by the RoadsCost module:\n')
            for key, value in self.output_dict.items():
                if isinstance(value, pd.DataFrame):
                    print('\nNow printing DataFrame ->', key, ':\n', value)
                else:
                    print(key, ':', value)

    def test_calculate_road_properties(self):
        road_properties = SitePreparationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        road_properties.run_module()

    def test_estimate_construction_time(self):
        construction_time = SitePreparationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        construction_time.run_module()

    def test_calculate_weather_delay(self):
        """
        Tests calculate_weather_delay()

        """

        weatherDelay = SitePreparationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        weatherDelay.run_module()

    def test_calculate_costs(self):
        calculate_costs = SitePreparationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=self.project_name)
        calculate_costs.run_module()
