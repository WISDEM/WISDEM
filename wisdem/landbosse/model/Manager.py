import traceback
import math

from wisdem.landbosse.model.ManagementCost import ManagementCost
from wisdem.landbosse.model.FoundationCost import FoundationCost
from wisdem.landbosse.model.SubstationCost import SubstationCost
from wisdem.landbosse.model.TransportCost import TransportCost
from wisdem.landbosse.model.TurbineCost import TurbineCost
from wisdem.landbosse.model.GridConnectionCost import GridConnectionCost
from wisdem.landbosse.model.SitePreparationCost import SitePreparationCost
from wisdem.landbosse.model.CollectionCost import ArraySystem
from wisdem.landbosse.model.ErectionCost import ErectionCost
from wisdem.landbosse.model.DevelopmentCost import DevelopmentCost

import pandas as pd

class Manager:
    """
    The Manager class distributes input and output dictionaries among
    the various modules. It maintains the hierarchical dictionary
    structure.
    """

    def __init__(self, input_dict, output_dict):
        """
        This initializer sets up the instance variables of:

        self.cost_modules: A list of cost module instances. Each of the
            instances must implement the method input_output.

        self.input_dict: A placeholder for the inputs dictionary

        self.output_dict: A placeholder for the output dictionary
        """
        self.input_dict = input_dict
        self.output_dict = output_dict

    def execute_landbosse(self, project_name):
        try:
            # Create weather window that will be used for all tasks (window for entire project; selected to restrict to seasons and hours specified)
            weather_data_user_input = self.input_dict['weather_window']
            season_construct = self.input_dict['season_construct']
            time_construct = self.input_dict['time_construct']
            daily_operational_hours = self.input_dict['hour_day'][time_construct]

            # Filtered window. Restrict to the seasons and hours specified.
            filtered_weather_window = weather_data_user_input.loc[(weather_data_user_input['Season'].isin(season_construct)) & (weather_data_user_input['Time window'] == time_construct)]
            filtered_weather_window = filtered_weather_window[0:(math.ceil(self.input_dict['construct_duration'] * 30 * daily_operational_hours))]

            # Rename weather data to specify types
            self.input_dict['weather_window'] = filtered_weather_window
            self.input_dict['weather_data_user_input'] = weather_data_user_input

            foundation_cost = FoundationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=project_name)
            foundation_cost.run_module()

            roads_cost = SitePreparationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=project_name)
            roads_cost.run_module()

            substation_cost = SubstationCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=project_name)
            substation_cost.run_module()

            transport_cost = TransportCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=project_name)
            transport_cost.run_module()

            transdist_cost = GridConnectionCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=project_name)
            transdist_cost.run_module()

            collection_cost = ArraySystem(input_dict=self.input_dict, output_dict=self.output_dict, project_name=project_name)
            collection_cost.run_module()

            development_cost = DevelopmentCost(input_dict=self.input_dict, output_dict=self.output_dict,
                                          project_name=project_name)
            development_cost.run_module()

            turbine_cost = TurbineCost(
                input_dict=self.input_dict,
                output_dict=self.output_dict,
                project_name=project_name,
            )
            turbine_cost.run_module()

            erection_cost_output_dict = dict()
            erection_cost = ErectionCost(
                input_dict=self.input_dict,
                output_dict=self.output_dict,
                project_name=project_name
            )
            erection_cost.run_module()
            self.output_dict['erection_cost'] = erection_cost_output_dict



            self.output_dict['actual_construction_months'] = self.output_dict['siteprep_construction_months'] + \
                                                             max(self.output_dict['erection_construction_months'],
                                                             self.output_dict['foundation_construction_months'],
                                                             self.output_dict['collection_construction_months']) + 1

            if self.output_dict['actual_construction_months'] < self.input_dict['construct_duration']:
                road_cost = self.output_dict['total_road_cost']
                index = road_cost['Type of cost'] == 'Other'
                other = road_cost[index]
                amount_shorter_than_input_construction_time = (self.input_dict['construct_duration'] - self.output_dict['siteprep_construction_months'])
                road_cost.loc[index, 'Cost USD'] = other['Cost USD'] - amount_shorter_than_input_construction_time * 55500
                self.output_dict['total_road_cost'] = road_cost

            total_costs = pd.concat((self.output_dict['total_collection_cost'],
                                      self.output_dict['total_road_cost'],
                                      self.output_dict['total_transdist_cost'],
                                      self.output_dict['total_substation_cost'],
                                      self.output_dict['total_transport_cost'],
                                      self.output_dict['total_foundation_cost'], 
                                      self.output_dict['total_erection_cost'], 
                                      self.output_dict['total_development_cost'], 
                                      ), sort=True)
            self.input_dict['project_value_usd'] = float(total_costs['Cost USD'].sum())
            self.input_dict['foundation_cost_usd'] = self.output_dict['total_foundation_cost']['Cost USD'].sum()

            management_cost = ManagementCost(input_dict=self.input_dict, output_dict=self.output_dict, project_name=project_name)
            management_cost.run_module()

            return 0
        except Exception:
            traceback.print_exc()
            return 1  # module did not run successfully
