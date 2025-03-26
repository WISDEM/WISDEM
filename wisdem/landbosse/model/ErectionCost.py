import pandas as pd
import numpy as np
from math import ceil

from wisdem.landbosse.model.CostModule import CostModule
from wisdem.landbosse.model.WeatherDelay import WeatherDelay

import traceback

# constants
km_per_m = 0.001
hr_per_min = 1 / 60
m_per_ft = 0.3048


class Point(object):
    def __init__(self, x, y):
        if type(x) == type(pd.Series(dtype=np.float64)):
            self.x = float(x.values[0])
            self.y = float(y.values[0])
        elif type(x) == type(np.array([])):
            self.x = float(x[0])
            self.y = float(y[0])
        elif type(x) == type(int(0)):
            self.x = float(x)
            self.y = float(y)
        elif type(x) == type(float(0.0)):
            self.x = x
            self.y = y
        else:
            raise ValueError(type(x))


def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def point_in_polygon(pt, poly):
    result = False
    maxx = float(np.r_[pt.x, np.array([m.x for m in poly])].max())
    for i in range(len(poly) - 1):
        if intersect(poly[i], poly[i + 1], pt, Point(1.1 * maxx, pt.y)):
            result = not result
    if intersect(poly[-1], poly[0], pt, Point(1.1 * maxx, pt.y)):
        result = not result
    return result


class ErectionCost(CostModule):
    """
        ErectionCost.py
        Created by Annika Eberle and Owen Roberts on Mar. 16, 2018
        Created by Alicia Key and Parangat Bhaskar on 01 June 2019

        Calculates the costs for erecting the tower and rotor nacelle assembly for land-based wind projects
        (items in brackets are not yet implemented)

        [Get terrain complexity]
        [Get site complexity]
        Get number of turbines
        Get duration of construction
        Get rate of deliveries
        Get daily hours of operation
        Get turbine rating
        Get component specifications
        [Get crane availability]

        Get price data
            Get labor mobilization_prices by crew type
            Get labor prices by crew type
            Get equipment mobilization prices by equipment type
            Get fuel prices
            Get equipment prices by equipment type

        Calculate operational time for lifting components

        Estimate potential time delays due to weather

        Calculate required labor and equip for erection (see equip_labor_by_type method below)
            Calculate number of workers by crew type
            Calculate man hours by crew type
            Calculate number of equipment by equip type
            Calculate equipment hours by equip type

        Calculate erection costs by type (see methods below)
            Calculate mobilization costs as function of number of workers by crew type, number of equipment by equipment type, labor_mobilization_prices, and equip_mobilization_prices
            Calculate labor costs as function of man_hours and labor prices by crew type
            Calculate fuel costs as function of equipment hours by equipment type and fuel prices by equipment type
            Calculate equipment costs as function of equipment hours by equipment type and equipment prices by equipment type

        Sum erection costs over all types to get total costs

        Find the least cost option

        Return total erection costs

        Keys in the input dictionary are the following:

        construct_duration
            (int) duration of construction (in months)

        rate_of_deliveries
            (int) rate of deliveries (number of turbines per week)

        weather_window
            (pd.DataFrame) window of weather data for project of interest.

        wind_shear_exponent
    -        (float) The exponent of the power law wind shear calculation

        overtime_multiplier:
            (float) multiplier for overtime work (working 60 hr/wk vs 40 hr/wk)

        allow_same_flag
            (bool) boolean flag to indicate whether choosing same base and
            topping crane is allowed.

        operational_construction_time
            (int) Number of hours per day when construction can happen.

        time_construct
            (int) 'normal' (10 hours per day) or 'long' (24 hours per day)

        project_data
            (dict) dictionary of pd.DataFrame for each of the csv files loaded
            for the project.

        In turn, the project_data dictionary contains key value pairs of the
        following:

        crane_specs:
            (pd.DateFrame) Specs about the cranes for the cost calculations.

        equip
            (pd.DataFrame) Equipment needed for various tasks

        crew
            (pd.DataFrame) Crew configurations needed for various tasks

        components
            (pd.DataFrame) components to build a wind turbine

        project
            (pd.DataFrame) The project of the project to calculate.

        equip_price
            (pd.DatFrame) Prices to operate various pieces of equipment.

        crew_price
            (pd.DataFrame) THe prices for various crews

        material_price
            (pd.DatFrame) Prices for various materials used during erection.

        rsmeans
            (p.DataFrame) RSMeans data
    """

    def __init__(self, input_dict, output_dict, project_name):
        """
        Parameters
        ----------
        input_dict : dict
            The input dictionary with key value pairs described in the
            class documentation

        output_dict : dict
            The output dictionary with key value pairs as found on the
            output documentation.
        """
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.project_name = project_name

        # These instance attributes are for diagnostics inside this class
        # only.
        self._possible_crane_cost = None
        self._number_of_equip = None

    def run_module(self):
        """
        Runs the ErectionCost model and populates the IO dictionaries with
        calculated values.

        Returns
        -------
        int
            0 if the module ran successfully. 1 if the module did not run
            successfully
        """
        try:
            self.calculate_costs()
            self.outputs_for_detailed_tab()
            self.output_dict["erection_module_type_operation"] = self.outputs_for_costs_by_module_type_operation(
                input_df=self.output_dict["total_erection_cost"], project_id=self.project_name, total_or_turbine=True
            )
            return 0, 0  # Module ran successfully
        except Exception as error:
            traceback.print_exc()
            print(f"Fail {self.project_name} ErectionCost")
            return 1, error  # Module did not run successfully

    def outputs_for_detailed_tab(self):
        """
        Creates a list of dictionaries which can be used on their own or
        used to make a dataframe.

        Must be called after self.run_module()

        Returns
        -------
        list(dict)
            A list of dicts, with each dict representing a row of the data.
        """
        result = []

        for _, row in self._number_of_equip.iterrows():
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "_number_of_equip: Operation-Crane name-Boom system-Number of equipment",
                    "value": f'{row["Operation"]}-{row["Crane name"]}-{row["Boom system"]}-{row["Number of equipment"]}',
                    "last_number": row["Number of equipment"],
                }
            )

        for _, row in self.output_dict["erection_selected_detailed_data"].iterrows():
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": f"erection_selected_detailed_data: Operation-Crane name-Boom system-Operational construct days over time construct days",
                    "value": f'{row["Operation"]}-{row["Crane name"]}-{row["Boom system"]}-{row["Operational construct days over time construct days"]}',
                    "last_number": row["Operational construct days over time construct days"],
                }
            )

        for row in self.output_dict["component_name_topvbase"].itertuples():
            dashed_row = "{} - {}".format(row[1], row[2])
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "component_name_topvbase: Operation - Top or Base",
                    "value": dashed_row,
                }
            )

        for row in self.output_dict["crane_choice"].itertuples():
            dashed_row = "{} - {} - {}".format(row[1], row[2], row[3])
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "crane_choice: Crew name - Boom system - Operation",
                    "value": dashed_row,
                }
            )

        for _, row in self.output_dict["crane_data_output"].iterrows():
            dashed_row = "{} - {} - {}".format(row.iloc[0], row.iloc[1], row.iloc[2])
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "crane_data_output: crane_boom_operation_concat - variable - value",
                    "value": dashed_row,
                    "last_number": row.iloc[2],
                }
            )

        for _, row in self.output_dict["crane_cost_details"].iterrows():
            dashed_row = "{} - {} - {}".format(row.iloc[0], row.iloc[1], row.iloc[2])
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "crane_cost_details: Operation ID - Type of cost - Cost",
                    "value": dashed_row,
                    "last_number": row.iloc[2],
                }
            )

        for _, row in self.output_dict["total_erection_cost"].iterrows():
            dashed_row = "{} - {} - {}".format(row.iloc[0], row.iloc[1], row.iloc[2])
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "total_erection_cost: Phase of construction - Type of cost - Cost USD",
                    "value": dashed_row,
                    "last_number": row.iloc[2],
                }
            )

        for _, row in self.output_dict["erection_selected_detailed_data"].iterrows():
            value = row["Labor cost USD without management"]
            operation = row["Operation"]
            result.append(
                {
                    "unit": "usd",
                    "type": "dataframe",
                    "variable_df_key_col_name": "erection_selected_detailed_data: crew cost without management",
                    "value": value,
                    "non_numeric_value": operation,
                }
            )

        for _, row in self.output_dict["erection_selected_detailed_data"].iterrows():
            value = row["Mobilization cost USD"]
            crane_boom_operation_concat = row["crane_boom_operation_concat"]
            result.append(
                {
                    "unit": "usd",
                    "type": "dataframe",
                    "variable_df_key_col_name": "erection_selected_detailed_data: mobilization",
                    "value": value,
                    "non_numeric_value": crane_boom_operation_concat,
                }
            )

        for _, row in self.output_dict["erection_selected_detailed_data"].iterrows():
            value = row["Wind multiplier"]
            operation = row["Operation"]
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": f"erection_selected_detailed_data: wind multiplier",
                    "value": value,
                    "non_numeric_value": operation,
                }
            )

        result.append(
            {
                "unit": "usd",
                "type": "variable",
                "variable_df_key_col_name": "total_cost_summed_erection",
                "value": float(self.output_dict["total_cost_summed_erection"]),
            }
        )

        for _, row in self.output_dict["management_crews_cost"].iterrows():
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "management_crews_cost: {}".format(" - ".join(row.index)),
                    "value": " - ".join(list(str(x) for x in row)[1:]),
                }
            )

        result.append(
            {
                "unit": "hours",
                "type": "variable",
                "variable_df_key_col_name": "number of hours in weather window",
                "value": len(self.input_dict["weather_window"]),
            }
        )

        result.append(
            {
                "unit": "none",
                "type": "variable",
                "variable_df_key_col_name": "time_weighted_weather_multiplier",
                "value": self.output_dict["time_weighted_weather_multiplier"],
            }
        )

        result.append(
            {
                "unit": "months",
                "type": "variable",
                "variable_df_key_col_name": "erection_construction_months",
                "value": self.output_dict["erection_construction_months"],
            }
        )

        result.append(
            {
                "unit": "usd",
                "type": "variable",
                "variable_df_key_col_name": "labor_cost_management",
                "value": self.output_dict["labor_cost_management"],
            }
        )

        result.append(
            {
                "unit": "usd",
                "type": "variable",
                "variable_df_key_col_name": "labor_cost_non_management",
                "value": self.output_dict["labor_cost_non_management"],
            }
        )

        result.append(
            {
                "unit": "usd",
                "type": "variable",
                "variable_df_key_col_name": "labor_cost_total",
                "value": self.output_dict["labor_cost_total"],
            }
        )

        module = type(self).__name__
        for _dict in result:
            _dict["project_id_with_serial"] = self.project_name
            _dict["module"] = module
        self.output_dict["erection_cost_csv"] = result

        return result

    def calculate_erection_operation_time(self):
        """
        Calculates operation time required for each type of equipment included in project data.

        self.output_dict['possible_cranes'] = possible_cranes
        self.output_dict['erection_operation_time'] = erection_operation_time_dict

        self.input_dict keys
        ---------------------
        construct_duration : int
            int duration of construction (in months)

        operational_construction_time : int
            Number of hours each day that are available for construction hours.

        self.output_dict keys
        ---------------------
        self.output_dict['possible_cranes'] : possible_cranes (with geometry)
        self.output_dict['erection_operation_time'] : Operation time for each crane.

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            Dataframe of possible_cranes (with geometry) and operational time for cranes
        """
        project_data = self.input_dict["project_data"]
        construct_duration = self.input_dict["construct_duration"]  # Total project construction time (months)
        operational_construction_time = self.input_dict["operational_construction_time"]  # Hours, 10 or 24

        # Why multiply by hardcoded 1/3?
        erection_construction_time = 1 / 3 * construct_duration
        breakpoint_between_base_and_topping_percent = self.input_dict["breakpoint_between_base_and_topping_percent"]
        hub_height_m = self.input_dict["hub_height_meters"]
        rotor_diameter_m = self.input_dict["rotor_diameter_m"]
        num_turbines = float(self.input_dict["num_turbines"])
        turbine_spacing_rotor_diameters = self.input_dict["turbine_spacing_rotor_diameters"]

        # for components in component list determine if base or topping
        project_data["components"]["Operation"] = project_data["components"]["Lift height m"] > (
            float(hub_height_m * breakpoint_between_base_and_topping_percent)
        )
        boolean_dictionary = {True: "Top", False: "Base"}
        project_data["components"]["Operation"] = project_data["components"]["Operation"].map(boolean_dictionary)

        # For output to a csv file
        self.output_dict["component_name_topvbase"] = project_data["components"][["Component", "Operation"]]

        # create groups for operations
        top_v_base = project_data["components"].groupby("Operation")

        # group crane data by boom system and crane name to get distinct cranes
        crane_grouped = project_data["crane_specs"].groupby(
            ["Equipment name", "Equipment ID", "Crane name", "Boom system", "Crane capacity tonne"]
        )

        # Calculate the crane lift polygons
        crane_poly = self.calculate_crane_lift_polygons(crane_grouped=crane_grouped)

        # loop through operation type (topping vs. base)
        component_max_speed = pd.DataFrame()
        for name_operation, component_group in top_v_base:
            lift_max_wind_speed = self.calculate_component_lift_max_wind_speed(
                component_group=component_group,
                crane_poly=crane_poly,
                component_max_speed=component_max_speed,
                operation=name_operation,
            )
            crane_poly = lift_max_wind_speed["crane_poly"]
            component_max_speed = lift_max_wind_speed["component_max_speed"]

        # Sorting can help for some operations, but isn't strictly necessary, so it can be turned
        # off when not debugging
        # component_max_speed = component_max_speed.sort_values(by=['Crane name', 'Boom system', 'Component'])

        # join crane polygon to crane specs
        crane_component = pd.merge(crane_poly, component_max_speed, on=["Crane name", "Boom system"], sort=True)

        # select only cranes that could lift the component
        possible_cranes = (
            crane_component.where(crane_component["crane_bool"] == True).dropna(thresh=1).reset_index(drop=True)
        )

        # calculate travel time per cycle
        turbine_spacing = float(turbine_spacing_rotor_diameters * rotor_diameter_m * km_per_m)
        possible_cranes["Travel time hr"] = (
            turbine_spacing / possible_cranes["Speed of travel km per hr"] * num_turbines
        )

        # calculate erection time
        possible_cranes["Operation time hr"] = (
            (possible_cranes["Lift height m"] / possible_cranes["Hoist speed m per min"] * hr_per_min)
            + (possible_cranes["Cycle time installation hrs"])
        ) * num_turbines

        # Modify the breakdown time column to reflect all the crane breakdowns
        # the entire project
        crane_breakdown_fraction = self.input_dict["crane_breakdown_fraction"]
        num_turbines_needing_breakdowns = ceil(num_turbines * crane_breakdown_fraction)
        breakdown_time_all_turbines_hrs = possible_cranes["Breakdown time hr"] * num_turbines_needing_breakdowns

        # Combine the breakdown time with the setup time to get total setup + breakdown
        # time and store it as setup time.
        setup_time_all_turbines_hrs = possible_cranes["Setup time hr"] * num_turbines
        possible_cranes["Setup time hr"] = setup_time_all_turbines_hrs + breakdown_time_all_turbines_hrs

        # check that crane can lift all components within a group (base vs top)
        # crane_lift_entire_group_for_operation = crane_component.groupby(by=["Crane name", "Boom system", "Operation"])[
        #     "crane_bool"
        # ].all()
        # if it can't then we need to remove it.
        # otherwise we end up with an option for a crane to perform an operation without lifting all of the corresponding components
        # testcranenew = possible_cranes.merge(
        #     crane_lift_entire_group_for_operation, on=["Crane name", "Boom system", "Operation"], sort=True
        # )
        # possible_cranes = testcranenew.loc[testcranenew["crane_bool_y"]]

        erection_time = possible_cranes.groupby(
            ["Crane name", "Equipment name", "Crane capacity tonne", "Crew type ID", "Boom system", "Operation"]
        )["Operation time hr"].sum()
        travel_time = possible_cranes.groupby(
            ["Crane name", "Equipment name", "Crane capacity tonne", "Crew type ID", "Boom system", "Operation"]
        )["Travel time hr"].max()
        setup_time = possible_cranes.groupby(
            ["Crane name", "Equipment name", "Crane capacity tonne", "Crew type ID", "Boom system", "Operation"]
        )["Setup time hr"].max()
        rental_time_without_weather = erection_time + travel_time + setup_time

        operation_time = rental_time_without_weather.reset_index()
        operation_time = operation_time.rename(columns={0: "Operation time all turbines hrs"})
        operation_time["Operational construct days"] = (
            operation_time["Operation time all turbines hrs"] / operational_construction_time
        )

        # if more than one crew needed to complete within construction duration then assume that all construction happens
        # within that window and use that time frame for weather delays; if not, use the number of days calculated
        operation_time["time_construct_bool"] = (
            operation_time["Operational construct days"] > erection_construction_time * 30
        )
        boolean_dictionary = {True: erection_construction_time * 30, False: np.nan}
        operation_time["time_construct_bool"] = operation_time["time_construct_bool"].map(boolean_dictionary)
        operation_time["Time construct days"] = operation_time[
            ["time_construct_bool", "Operational construct days"]
        ].min(axis=1)

        for operation, component_group in top_v_base:
            unique_component_crane = possible_cranes.loc[possible_cranes["Operation"] == operation][
                "Component"
            ].unique()
            for component in component_group["Component"]:
                if component not in unique_component_crane:
                    raise Exception(
                        "Error: Unable to find installation crane for {} operation and {} component".format(
                            operation, component
                        )
                    )

        erection_operation_time_dict = dict()
        erection_operation_time_dict["possible_cranes"] = possible_cranes
        erection_operation_time_dict["operation_time"] = operation_time

        self.output_dict["possible_cranes"] = possible_cranes
        self.output_dict["erection_operation_time"] = erection_operation_time_dict

        return possible_cranes, operation_time

    def calculate_offload_operation_time(self):
        """
        Calculates time for the offload operation.

        self.input_dict keys
        --------------------
        project_data : dict
            dict of data frames for each of the csv files loaded for the project

        operational_construction_time : int
            operational hours of construction

        rate_of_deliveries : int
            rate of deliveries of turbines ready for erection.

        self.output_dict key
        --------------------
        possible_cranes : pd.DataFrame
            Dataframe of cranes possibly available for the operation

        operation_time : int
            Integer of number of hours per day construction can proceed.
        """
        project_data = self.input_dict["project_data"]
        operational_construction_time = self.input_dict["operational_construction_time"]
        rate_of_deliveries = self.input_dict["rate_of_deliveries"]
        rotor_diameter_m = self.input_dict["rotor_diameter_m"]
        num_turbines = float(self.input_dict["num_turbines"])
        turbine_spacing_rotor_diameters = self.input_dict["turbine_spacing_rotor_diameters"]

        offload_cranes = project_data["crane_specs"].where(
            project_data["crane_specs"]["Equipment name"] == "Offload crane"
        )

        # group crane data by boom system and crane name to get distinct cranes
        crane_grouped = offload_cranes.groupby(
            ["Equipment name", "Equipment ID", "Crane name", "Boom system", "Crane capacity tonne"]
        )

        crane_poly = self.calculate_crane_lift_polygons(crane_grouped=crane_grouped)
        component_group = project_data["components"]
        component_max_speed = pd.DataFrame()
        lift_max_wind_speed = self.calculate_component_lift_max_wind_speed(
            component_group=component_group,
            crane_poly=crane_poly,
            component_max_speed=component_max_speed,
            operation="offload",
        )
        component_max_speed = lift_max_wind_speed["component_max_speed"]
        crane_poly = lift_max_wind_speed["crane_poly"]

        if len(crane_poly) != 0:
            # join crane polygon to crane specs
            crane_component = pd.merge(crane_poly, component_max_speed, on=["Crane name", "Boom system"], sort=True)

            # select only cranes that could lift the component
            possible_cranes = (
                crane_component.where(crane_component["crane_bool"] == True).dropna(thresh=1).reset_index(drop=True)
            )

            # calculate travel time per cycle
            turbine_spacing = float(turbine_spacing_rotor_diameters * rotor_diameter_m * km_per_m)
            turbine_num = float(self.input_dict["num_turbines"])
            possible_cranes["Travel time hr"] = (
                turbine_spacing / possible_cranes["Speed of travel km per hr"] * num_turbines
            )

            # calculate erection time
            possible_cranes["Operation time hr"] = (
                (possible_cranes["Lift height m"] / possible_cranes["Hoist speed m per min"] * hr_per_min)
                + (possible_cranes["Offload cycle time hrs"])
            ) * turbine_num

            # store setup time
            possible_cranes["Setup time hr"] = possible_cranes["Setup time hr"] * turbine_num

            erection_time = possible_cranes.groupby(
                ["Crane name", "Equipment name", "Crane capacity tonne", "Crew type ID", "Boom system"]
            )["Operation time hr"].sum()
            travel_time = possible_cranes.groupby(
                ["Crane name", "Equipment name", "Crane capacity tonne", "Crew type ID", "Boom system"]
            )["Travel time hr"].max()
            setup_time = possible_cranes.groupby(
                ["Crane name", "Equipment name", "Crane capacity tonne", "Crew type ID", "Boom system"]
            )["Setup time hr"].max()
            rental_time_without_weather = erection_time + travel_time + setup_time

            operation_time = rental_time_without_weather.reset_index()
            operation_time = operation_time.rename(columns={0: "Operation time all turbines hrs"})
            operation_time["Operational construct days"] = (
                operation_time["Operation time all turbines hrs"] / operational_construction_time
            )

            # if more than one crew needed to complete within construction duration
            # then assume that all construction happens within that window and use
            # that timeframe for weather delays; if not, use the number of days calculated
            operation_time["time_construct_bool"] = turbine_num / operation_time[
                "Operational construct days"
            ] * 6 > float(rate_of_deliveries)
            boolean_dictionary = {True: (float(turbine_num) / (float(rate_of_deliveries) / 6)), False: np.nan}
            operation_time["time_construct_bool"] = operation_time["time_construct_bool"].map(boolean_dictionary)
            operation_time["Time construct days"] = operation_time[
                ["time_construct_bool", "Operational construct days"]
            ].max(axis=1)

            possible_cranes["Operation"] = "Offload"
            operation_time["Operation"] = "Offload"
        else:
            possible_cranes = []
            operation_time = []

        # print(possible_cranes[['Crane name', 'Component', 'Operation time hr']])
        unique_components = project_data["components"]["Component"].unique()
        unique_component_crane = possible_cranes["Component"].unique()
        for component in unique_components:
            if component not in unique_component_crane:
                raise Exception("Error: Unable to find offload crane for {}".format(component))

        return possible_cranes, operation_time

    def calculate_crane_lift_polygons(self, crane_grouped):
        """
        Here we associate polygons with each crane. However, these polygons are not shapes
        for the lift. Rather, they define functions f(x), where x is a crane lift load and
        f(x) is the height to which that load can be lifted. To find out whether the crane
        can lift a particular load, one just needs to check whether a point x (lift mass in
        tonnes) and y (lift height in m) lies within the crane's polygon.

        Parameters
        ----------
        crane_grouped : pandas.core.groupby.generic.DataFrameGroupBy
            The aggregation of the cranes to compute the lift polygons for. The columns
            in the aggregation are assume to be 'Equipment name', 'Crane name', 'Boom system',
            'Crane capacity tonne'

        Returns
        -------
        pd.DataFrame
            A dataframe of the cranes and their lifting polygons.
        """
        crane_poly = pd.concat(self.iterate_crane_lift_polygons(crane_grouped))
        return crane_poly

    def iterate_crane_lift_polygons(self, crane_grouped):
        for (equipment_name, equipment_id, crane_name, boom_system, crane_capacity_tonne), crane in crane_grouped:
            crane = crane.reset_index(drop=True)
            x = crane["Max capacity tonne"]
            y = crane["Hub height m"]
            wind_speed = min(crane["Max wind speed m per s"])
            hoist_speed = min(crane["Hoist speed m per min"])
            travel_speed = min(crane["Speed of travel km per hr"])
            setup_time = max(crane["Setup time hr"])
            breakdown_time = max(crane["Breakdown time hr"])
            crew_type = crane.loc[
                0, "Crew type ID"
            ]  # For every crane/boom combo the crew is the same, so we can just take first crew.
            polygon = [
                Point(0, 0),
                Point(0, max(y)),
                Point(min(x), max(y)),
                Point(max(x), min(y)),
                Point(max(x), 0),
            ]
            df = pd.DataFrame(
                [
                    [
                        equipment_name,
                        equipment_id,
                        crane_name,
                        boom_system,
                        crane_capacity_tonne,
                        wind_speed,
                        setup_time,
                        breakdown_time,
                        hoist_speed,
                        travel_speed,
                        crew_type,
                        polygon,
                    ]
                ],
                columns=[
                    "Equipment name",
                    "Equipment ID",
                    "Crane name",
                    "Boom system",
                    "Crane capacity tonne",
                    "Max wind speed m per s",
                    "Setup time hr",
                    "Breakdown time hr",
                    "Hoist speed m per min",
                    "Speed of travel km per hr",
                    "Crew type ID",
                    "Crane poly",
                ],
            )
            yield df

    def calculate_component_lift_max_wind_speed(self, *, component_group, crane_poly, component_max_speed, operation):
        """
        First, using the height and mass of the component being lifted, this method determines
        if a component can be lifted to the necessary height by each crane.

        Also, creates a dataframe that has the maximum wind speeds to lift particular components,
        given the component data and crane lift data given in the arguments.

        For the maximum wind speed calculations, we use these equations to calculation vmax,
        which is the maximum permissible wind speed:

        vmax = max_TAB * sqrt(1.2 * mh / aw), where
        mh = hoist load
        aw = area exposed to wind = surface area * coeff drag
        1.2 = constant in m^2 / t
        vmax_tab = maximum load speed per load chart
        (source: pg. 33 of Liebherr)

        See the source code for this method on how this calculation is used.

        Parameters
        ----------
        component_group : pd.DataFrame
            Dataframe with component data.

        crane_poly : pd.DataFrame
            Data about cranes doing the lifting. The polygons are specifications of
            functions that define lift height f(x) as a function of component mass x.

        component_max_speed : pd.DataFrame
            The dataframe into which maximum wind speeds for lifting each component
            will be accumulated. For the first call into this method, pass in an
            empty dataframe created with pd.DataFrame

        operation : str
            The name of the operation ("base", "top" or "offload") that the cranes
            are performing for this calculation. If the operation is "Offload"
            the 'Mass tonne' is divided by two when making the lift polygons.
            This created the assumption that there are always 2 offload cranes during
            offload operations. (See the calculate_crane_lift_polygons() method
            above for more about calculating the lift polygons.

        Returns
        -------
        dict
            Returns a dict of pd.DataFrame values. The key "component_max_speed" is the
            the dataframe of max component speeds. The key "crane_poly" is a COPY of the
            crane_poly dataframe passed as a parameter to this function and with a column
            of "Crane bool {operation}" attached.
        """
        for idx, crane in crane_poly.iterrows():
            polygon = crane["Crane poly"]

            # calculate polygon for crane capacity and check if component can be lifted by each crane without wind loading
            for component in component_group["Component"]:
                # get weight and height of component in each component group
                component_only = component_group.where(component_group["Component"] == component).dropna(thresh=1)

                # See docstring for "operation" parameter above about mass calculations for offloading
                if operation == "offload":
                    point = Point(
                        component_only["Mass tonne"] / 2,
                        (component_only["Section height m"] + component_only["Offload hook height m"]),
                    )
                else:
                    point = Point(
                        component_only["Mass tonne"],
                        (component_only["Lift height m"] + component_only["Offload hook height m"]),
                    )
                crane["Lift boolean {component}".format(component=component)] = point_in_polygon(point, polygon)

            # Transform the "Lift boolean" indexes in the series to a list of booleans
            # that signify if the crane can lift a component.
            bool_list = []
            for component in component_group["Component"]:
                if crane["Lift boolean {component}".format(component=component)] is False:
                    crane_bool = False
                else:
                    crane_bool = True
                bool_list.append(crane_bool)

            # mh is an effective mass (it should be the mass of the entire component for both offload and other cranes, not just 1/2 that's used above for determining whether the part can be lifted)
            mh = component_group["Mass tonne"]
            aw = component_group["Surface area sq m"] * component_group["Coeff drag"]
            vmax_tab = crane["Max wind speed m per s"]
            vmax_calc = vmax_tab * np.sqrt(1.2 * mh / aw)

            # if vmax_calc is less than vmax_tab then vmax_calc, otherwise vmax_tab (based on pg. 33 of Liebherr)
            component_group_new = pd.DataFrame(
                component_group,
                columns=list(component_group.columns.values) + ["vmax", "Crane name", "Boom system", "crane_bool"],
            )
            component_group_new["vmax"] = np.minimum(vmax_tab, vmax_calc)
            component_group_new["Crane name"] = crane["Crane name"]
            component_group_new["Boom system"] = crane["Boom system"]
            component_group_new["crane_bool"] = bool_list

            component_max_speed = pd.concat((component_max_speed, component_group_new), sort=True)

        crane_poly_new = crane_poly.copy()
        crane_poly_new["Crane bool {}".format(operation)] = min(bool_list)

        result = {"component_max_speed": component_max_speed, "crane_poly": crane_poly_new}

        return result

    def calculate_wind_delay_by_component(self):
        """
        Calculates wind delay for each component in the project.

        Returns
        -------
        pd.DataFrame
            crane specifications and component properties joined with wind delays for each case.
        """

        # Get necessary values from input_dict
        crane_specs = self.output_dict["crane_specs_withoffload"]
        weather_window = self.input_dict["weather_window"]

        # calculate wind delay for each component and crane combination
        crane_specs = crane_specs.reset_index()
        crane_specs["Wind delay percent"] = np.nan

        # pull global inputs for weather delay from input_dict
        weather_data_keys = {"wind_shear_exponent", "weather_window"}

        # specify collection-specific weather delay inputs
        weather_delay_global_inputs = {i: self.input_dict[i] for i in self.input_dict if i in weather_data_keys}

        # Iterate over every crane + boom combination
        for i, row in crane_specs.iterrows():
            # assume we don't know when the operation occurs
            operation_window = len(weather_window.index)  # operation window = entire construction weather window
            operation_start = 0  # start time is at beginning of construction weather window

            # extract critical wind speed
            critical_wind_operation = row["vmax"]

            # extract height of interest (differs for offload cranes)
            if (row["Crane bool offload"] == 1) is True:
                height_interest = row["Section height m"] + row["Offload hook height m"]
            else:
                height_interest = row["Lift height m"] + row["Offload hook height m"]

            # compute weather delay
            weather_delay_input_dict = weather_delay_global_inputs
            weather_delay_output_dict = dict()
            weather_delay_input_dict["start_delay_hours"] = operation_start
            weather_delay_input_dict["critical_wind_speed_m_per_s"] = critical_wind_operation
            weather_delay_input_dict["wind_height_of_interest_m"] = height_interest
            weather_delay_input_dict["mission_time_hours"] = operation_window

            WeatherDelay(weather_delay_input_dict, weather_delay_output_dict)
            wind_delay = np.array(weather_delay_output_dict["wind_delays"])

            # if greater than 4 hour delay, then shut down for full day (10 hours)
            wind_delay[(wind_delay > 4)] = 10
            wind_delay_time = float(wind_delay.sum())

            # store weather delay for operation, component, crane, and boom combination
            crane_specs.loc[i, "Wind delay percent"] = wind_delay_time / len(weather_window)

        self.output_dict["enhanced_crane_specs"] = crane_specs
        return crane_specs

    def aggregate_erection_costs(self):
        """
        Aggregates labor, equipment, mobilization and fuel costs for erection.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Two dataframes: First, utilizing the separate cranes for base and topping.
            Second, utilizing same crane for base and topping,
            Third is crew cost.
        """
        join_wind_operation = self.output_dict["join_wind_operation"]
        overtime_multiplier = self.input_dict["overtime_multiplier"]
        time_construct = self.input_dict["time_construct"]
        project_data = self.input_dict["project_data"]
        hour_day = self.input_dict["hour_day"]

        # TODO: consider removing equipment name and crane capacity from crane_specs tab (I believe these data are unused here and they get overwritten later with equip information from equip tab)
        join_wind_operation = join_wind_operation.drop(columns=["Equipment name", "Crane capacity tonne"])

        possible_crane_cost_with_equip = pd.merge(
            join_wind_operation, project_data["equip"], on=["Equipment ID", "Operation"], sort=True
        )

        equip_crane_cost = pd.merge(
            possible_crane_cost_with_equip,
            project_data["equip_price"],
            on=["Equipment name", "Crane capacity tonne"],
            sort=True,
        )

        equip_crane_cost["Equipment rental cost USD"] = (
            equip_crane_cost["Total time per op with weather"]
            * equip_crane_cost["Equipment price USD per hour"]
            * equip_crane_cost["Number of equipment"]
        )

        # Drop duplicates from equip crane cost, if any
        # equip_crane_cost.drop_duplicates(subset=['Equipment ID', 'Operation', 'Crane name', 'Boom system'], inplace=True)

        equipment_cost_to_merge = equip_crane_cost[
            [
                "Crane name",
                "Boom system",
                "Equipment ID",
                "Operation",
                "Equipment price USD per hour",
                "Number of equipment",
                "Equipment rental cost USD",
                "Fuel consumption gal per day",
            ]
        ]
        equipment_cost_to_merge = (
            equipment_cost_to_merge.groupby(["Crane name", "Boom system", "Equipment ID", "Operation"])
            .sum()
            .reset_index()
        )

        possible_crane_cost = pd.merge(
            join_wind_operation,
            equipment_cost_to_merge,
            on=["Crane name", "Boom system", "Equipment ID", "Operation"],
            sort=True,
        )

        # Remove any duplicates from crew data.
        crew_deduped = project_data["crew"].drop_duplicates(
            subset=["Crew type ID", "Operation", "Crew name", "Labor type ID"], keep="first"
        )

        # Merge crew and price data for non-management crews only (base, topping, and offload only)
        crew_cost = pd.merge(crew_deduped, project_data["crew_price"], on=["Labor type ID"], sort=True)
        self.output_dict["crew_cost"] = crew_cost
        non_management_crew_cost = crew_cost.loc[crew_cost["Operation"].isin(["Base", "Top", "Offload"])]

        # calculate crew costs
        non_overtime_hours_per_week = 40
        working_days_per_week = 6
        hours_per_week = working_days_per_week * hour_day[time_construct]
        overtime_percentage = (hours_per_week - non_overtime_hours_per_week) / hours_per_week
        normal_labor_rate = non_overtime_hours_per_week / hours_per_week
        crew_cost["Hourly rate for all workers"] = (
            non_management_crew_cost["Hourly rate USD per hour"] * non_management_crew_cost["Number of workers"]
        ) * (normal_labor_rate + overtime_percentage * overtime_multiplier)
        crew_cost["Per diem all workers"] = (
            non_management_crew_cost["Per diem USD per day"] * non_management_crew_cost["Number of workers"]
        )

        # Crew cost group is getting two more people in it.
        # Note to future self, enforce constraints on dataframes.
        #
        # Before crew_price sheet is used, sort based on labor cost. Then drop rows with duplicated job titles.
        # Intent is to keep the most expensive labor row.

        # group crew costs by crew type and operation
        crew_cost_grouped = crew_cost.groupby(["Crew type ID", "Operation"]).sum(numeric_only=True).reset_index()

        # merge crane data with grouped crew costs

        possible_crane_cost = pd.merge(
            possible_crane_cost, crew_cost_grouped, on=["Crew type ID", "Operation"], sort=True
        )

        # calculate labor costs
        labor_day_operation = round(possible_crane_cost["Total time per op with weather"] / hour_day[time_construct])
        possible_crane_cost["Subtotal for hourly labor (non-management) USD"] = (
            possible_crane_cost["Total time per op with weather"] * possible_crane_cost["Hourly rate for all workers"]
        )
        possible_crane_cost["Subtotal for per diem labor (non-management) USD"] = (
            labor_day_operation * possible_crane_cost["Per diem all workers"]
        )
        possible_crane_cost["Labor cost USD without management"] = (
            possible_crane_cost["Subtotal for hourly labor (non-management) USD"]
            + possible_crane_cost["Subtotal for per diem labor (non-management) USD"]
        )

        # calculate fuel costs
        possible_crane_cost["Fuel cost USD"] = (
            possible_crane_cost["Fuel consumption gal per day"]
            * float(self.input_dict["fuel_cost_usd_per_gal"])
            * labor_day_operation
        )

        # calculate costs if top and base cranes are the same
        base_cranes = possible_crane_cost[possible_crane_cost["Operation"] == "Base"]
        top_cranes = possible_crane_cost[possible_crane_cost["Operation"] == "Top"]
        crane_topbase_bool = top_cranes["Crane name"].isin(base_cranes["Crane name"])
        boom_topbase_bool = top_cranes["Boom system"].isin(base_cranes["Boom system"])
        possible_crane_topbase = top_cranes[boom_topbase_bool & crane_topbase_bool]
        same_topbase_crane_list = possible_crane_topbase[["Crane name", "Boom system"]]
        possible_crane_topbase = same_topbase_crane_list.merge(
            possible_crane_cost, on=["Crane name", "Boom system"], sort=True
        )
        possible_crane_topbase_sum = (
            possible_crane_topbase.groupby(["Crane name", "Boom system"])[
                [
                    "Labor cost USD without management",
                    "Subtotal for hourly labor (non-management) USD",
                    "Subtotal for per diem labor (non-management) USD",
                    "Equipment rental cost USD",
                    "Fuel cost USD",
                ]
            ]
            .sum()
            .reset_index()
        )

        # Store the possible cranes for the top and base for future diagnostics.
        self._possible_crane_cost = possible_crane_cost.copy()

        # group crane spec data for mobilization
        mobilization_costs = (
            project_data["crane_specs"]
            .groupby(["Crane name", "Boom system"])["Mobilization cost USD"]
            .max()
            .reset_index()
        )

        mobilization_costs["Mobilization cost USD"] = (
            mobilization_costs["Mobilization cost USD"] * 2
        )  # for mobilization and demobilizaton

        # join top and base crane data with mobilization data
        topbase_same_crane_cost = pd.merge(
            possible_crane_topbase_sum, mobilization_costs, on=["Crane name", "Boom system"], sort=True
        )

        # compute total project cost for erection
        topbase_same_crane_cost["Total cost USD"] = (
            topbase_same_crane_cost["Labor cost USD without management"]
            + topbase_same_crane_cost["Equipment rental cost USD"]
            + topbase_same_crane_cost["Fuel cost USD"]
            + topbase_same_crane_cost["Mobilization cost USD"]
        )

        # adds operation label for same crane used for base and topping (this way columns are consistent for same and separate basetop)
        topbase_same_crane_cost["Operation"] = "Base + Top"

        # calculate costs if top and base use separate cranes
        separate_topbase = (
            possible_crane_cost.groupby(["Operation", "Crane name", "Boom system"])[
                [
                    "Labor cost USD without management",
                    "Subtotal for hourly labor (non-management) USD",
                    "Subtotal for per diem labor (non-management) USD",
                    "Equipment rental cost USD",
                    "Fuel cost USD",
                ]
            ]
            .sum()
            .reset_index()
        )

        # join mobilization data to separate top base crane costs
        separate_topbase_crane_cost = pd.merge(
            separate_topbase, mobilization_costs, on=["Crane name", "Boom system"], sort=True
        )

        # compute total project cost for erection
        separate_topbase_crane_cost["Total cost USD"] = (
            separate_topbase_crane_cost["Labor cost USD without management"]
            + separate_topbase_crane_cost["Equipment rental cost USD"]
            + separate_topbase_crane_cost["Fuel cost USD"]
            + separate_topbase_crane_cost["Mobilization cost USD"]
        )

        return separate_topbase_crane_cost, topbase_same_crane_cost, crew_cost

    def find_minimum_cost_cranes(self):
        """
        Finds the minimum cost crane(s) based on the aggregated labor, equipment,
        mobilization and fuel costs for erection.

        self.output_dict keys used as inputs
        ------------------------------------
        separate_basetop : pd.DataFrame
            data frame with aggregated labor, equipment, mobilization, and fuel costs for utilizing
            separate cranes for base and topping.
        same_basetop : pd.DataFrame
            data frame with aggregated labor, equipment, mobilization, and fuel costs for utilizing the
            same crane for base and topping

        allow_same_flag : boolean
            flag to indicate whether choosing same base and topping crane is allowed
        """
        allow_same_flag = self.input_dict["allow_same_flag"]
        separate_basetop = self.output_dict["separate_basetop"]
        same_basetop = self.output_dict["same_basetop"]

        self.output_dict["separate_basetop"] = separate_basetop

        total_separate_cost = pd.DataFrame()
        for operation in separate_basetop["Operation"].unique():
            # find minimum cost option for separate base and topping cranes
            min_val = min(separate_basetop["Total cost USD"].where(separate_basetop["Operation"] == operation).dropna())

            # find the crane that corresponds to the minimum cost for each operation
            crane = separate_basetop[separate_basetop["Total cost USD"] == min_val]
            cost = crane.groupby("Operation").min()
            total_separate_cost = pd.concat((total_separate_cost, cost), sort=True)

        # reset index for separate crane costs
        total_separate_cost = total_separate_cost.reset_index()

        # duplicate offload records because assuming two offload cranes are on site
        total_separate_cost = pd.concat(
            (total_separate_cost, total_separate_cost.loc[total_separate_cost["Operation"] == "Offload"]), sort=True
        )

        # sum costs for separate cranes to get total for all cranes
        cost_chosen_separate = total_separate_cost["Total cost USD"].sum()

        if allow_same_flag is True:
            # get the minimum cost for using the same crane for all operations
            cost_chosen_same = min(same_basetop["Total cost USD"])

            # check if separate or same crane option is cheaper and choose crane cost
            if cost_chosen_separate < cost_chosen_same:
                cost_chosen = total_separate_cost.groupby(
                    by=["Boom system", "Crane name", "Operation"]
                ).sum()  # added crane name and operation to groupby
            else:
                cost_chosen = same_basetop.where(same_basetop["Total cost USD"] == cost_chosen_same).dropna()
        else:
            cost_chosen = total_separate_cost.groupby(
                by=["Boom system", "Crane name", "Operation"]
            ).sum()  # added crane name and operation to groupby

        return cost_chosen

    def calculate_management_crews_cost(self, erection_cost):
        """
        Calculates management costs for erection, based on rate of turbine deliveries.

        Parameters
        ----------
        erection_cost : pd.DataFrame
            Other erection costs from which to calculate duration of management
            operations.

        Returns
        -------
        pd.DataFrame, pd.DataFrame, float
            The first dataframe are management costs by each role on each team.
            The second dataframe are management costs summed over aggregations of
            teams. The float is the sum of all the management costs for erection.
        """

        num_turbines = self.input_dict["num_turbines"]
        overtime_multiplier = self.input_dict["overtime_multiplier"]
        hour_day = self.input_dict["hour_day"]
        time_construct = self.input_dict["time_construct"]
        deliveries_per_week = float(self.input_dict["rate_of_deliveries"])
        duration_days = erection_cost["Time construct days"].sum()
        duration_hours = round(duration_days * hour_day[time_construct])

        # Merge crew and price data for management crews only.
        crew_cost = self.output_dict["crew_cost"]
        management_crews = crew_cost.loc[crew_cost["Operation"].isin(["Management", "Mechanical completion"])].copy()

        # increase management crews by project size
        management_crews.loc[management_crews["Crew name"] == "Management - project size", "Number of workers"] = round(
            management_crews.loc[management_crews["Crew name"] == "Management - project size", "Number of workers"]
            * np.ceil(num_turbines / 100)
        )

        # increase management crews by rate of construction (scale if greater than 10/wk)
        management_crews.loc[management_crews["Crew name"] == "Management - rate construction", "Number of workers"] = (
            round(
                management_crews.loc[
                    management_crews["Crew name"] == "Management - rate construction", "Number of workers"
                ]
                * np.ceil(deliveries_per_week / 10)
            )
        )
        management_crews.loc[management_crews["Crew name"] == "Mechanical completion", "Number of workers"] = round(
            management_crews.loc[management_crews["Crew name"] == "Mechanical completion", "Number of workers"]
            * np.ceil(deliveries_per_week / 10)
        )

        management_crews["Hourly rate for all workers"] = (
            management_crews["Hourly rate USD per hour"] * management_crews["Number of workers"]
        )

        management_crews["Per diem all workers"] = (
            management_crews["Per diem USD per day"] * management_crews["Number of workers"]
        )

        # Now calculate management costs
        management_crews["per_diem_costs"] = management_crews["Per diem all workers"] * duration_days
        hours_per_week = 6 * hour_day[time_construct]
        overtime_percentage = (hours_per_week - 40) / hours_per_week
        normal_labor_rate = 40 / hours_per_week
        management_crews["hourly_costs"] = management_crews["Hourly rate for all workers"] * (
            duration_hours * (normal_labor_rate + overtime_percentage * overtime_multiplier)
        )
        management_crews["crew_level_total_costs"] = (
            management_crews["per_diem_costs"] + management_crews["hourly_costs"]
        )

        # Aggregate and sum
        management_crew_cost_grouped = (
            management_crews.groupby(["Crew type ID", "Operation", "Crew name"]).sum(numeric_only=True).reset_index()
        )

        # Total management cost
        total_management_cost = management_crews["crew_level_total_costs"].sum()

        # erection_construction_months is the duration of erection time in
        # units of months.
        days_per_month = 30
        self.output_dict["erection_construction_months"] = duration_days / days_per_month

        return management_crews, management_crew_cost_grouped, total_management_cost

    def calculate_costs(self):
        """
        Calculates BOS costs for erection including selecting cranes that can lift
        components, incorporating wind delays and finding the least cost crane options
        for erection.
        """
        [crane_specs, operation_time] = self.calculate_erection_operation_time()
        crane_specs = crane_specs.infer_objects()  # cast cols of Booleans to bool dtype

        self.output_dict["crane_specs"] = crane_specs
        self.output_dict["operation_time"] = operation_time

        [offload_specs, offload_time] = self.calculate_offload_operation_time()
        offload_specs = offload_specs.infer_objects()  # cast cols of Booleans to bool dtype

        self.output_dict["offload_specs"] = offload_specs
        self.output_dict["offload_time"] = offload_time

        # append data for offloading
        if len(offload_specs) != 0:
            crane_specs_withoffload = pd.concat((crane_specs, offload_specs), sort=True)
            operation_time_withoffload = pd.concat((operation_time, offload_time), sort=True)
        else:
            raise Exception("ErectionCost calculate_costs(): offload_specs empty")

        self.output_dict["crane_specs_withoffload"] = crane_specs_withoffload
        self.output_dict["operation_time_withoffload"] = operation_time_withoffload
        crane_specs_with_weather = self.calculate_wind_delay_by_component()
        self.output_dict["cranes_wind_delay_withoffload"] = crane_specs_with_weather

        average_wind_delay = (
            crane_specs_with_weather.groupby(["Crane name", "Boom system", "Operation", "Equipment ID"])[
                "Wind delay percent"
            ]
            .mean()
            .reset_index()
        )

        join_wind_operation = pd.merge(
            operation_time_withoffload, average_wind_delay, on=["Crane name", "Boom system", "Operation"], sort=True
        )

        join_wind_operation["Wind multiplier"] = 1 / (1 - join_wind_operation["Wind delay percent"])

        # 'Total time per op with weather' units are hours
        join_wind_operation["Total time per op with weather"] = (
            join_wind_operation["Operation time all turbines hrs"] * join_wind_operation["Wind multiplier"]
        )

        self.output_dict["join_wind_operation"] = join_wind_operation

        [separate_basetop, same_basetop, crew_cost] = self.aggregate_erection_costs()

        self.output_dict["separate_basetop"] = separate_basetop
        self.output_dict["same_basetop"] = same_basetop
        self.output_dict["crew_cost"] = crew_cost

        erection_cost = self.find_minimum_cost_cranes()

        selected_time = join_wind_operation[
            [
                "Crane name",
                "Boom system",
                "Operation",
                "Operation time all turbines hrs",
                "Total time per op with weather",
                "Wind multiplier",
                "Operational construct days",
                "Time construct days",
            ]
        ]
        erection_cost = erection_cost.reset_index()
        selected_detailed_data = erection_cost.merge(
            selected_time, on=["Crane name", "Boom system", "Operation"], sort=True
        )
        selected_detailed_data["Total time per turbine"] = (
            selected_detailed_data["Total time per op with weather"] / self.input_dict["num_turbines"]
        )

        management_crews_cost, management_crews_cost_grouped, total_management_cost = (
            self.calculate_management_crews_cost(selected_detailed_data)
        )

        crane_choice = selected_detailed_data[["Crane name", "Boom system", "Operation"]].drop_duplicates()

        selected_detailed_data["crane_boom_operation_concat"] = selected_detailed_data[
            ["Crane name", "Boom system", "Operation"]
        ].apply(lambda x: "-".join(x), axis=1)
        crane_data_output = selected_detailed_data.drop(["Crane name", "Boom system", "Operation"], axis=1)
        crane_data_output = crane_data_output.melt(id_vars=["crane_boom_operation_concat"])

        crane_cost_details = crane_data_output.where(crane_data_output["variable"].str.contains("cost")).dropna()
        crane_cost_details = crane_cost_details.rename(
            index=str,
            columns={"crane_boom_operation_concat": "Operation ID", "variable": "Type of cost", "value": "Cost"},
        )

        subtotal_per_diem_labor_management_USD = management_crews_cost["per_diem_costs"].sum()
        subtotal_hourly_labor_management_USD = management_crews_cost["hourly_costs"].sum()

        self.output_dict["labor_cost_management"] = (
            subtotal_per_diem_labor_management_USD + subtotal_hourly_labor_management_USD
        )
        self.output_dict["labor_cost_non_management"] = selected_detailed_data[
            "Labor cost USD without management"
        ].sum()
        self.output_dict["labor_cost_total"] = (
            self.output_dict["labor_cost_management"] + self.output_dict["labor_cost_non_management"]
        )

        total_erection_cost = pd.DataFrame(
            [
                ["Erection", "Equipment rental", selected_detailed_data["Equipment rental cost USD"].sum()],
                ["Erection", "Fuel", selected_detailed_data["Fuel cost USD"].sum()],
                ["Erection", "Labor", self.output_dict["labor_cost_total"]],
                ["Erection", "Mobilization", selected_detailed_data["Mobilization cost USD"].sum()],
                ["Erection", "Other", 0],
                ["Erection", "Materials", 0],
            ],
            columns=["Phase of construction", "Type of cost", "Cost USD"],
        )

        total_cost_summed_erection = total_erection_cost.sum(numeric_only=True).iloc[0]

        erection_wind_mult = selected_detailed_data["Wind multiplier"]
        erection_wind_mult = erection_wind_mult.reset_index(drop=True).mean()

        self.output_dict["total_erection_cost"] = total_erection_cost
        self.output_dict["erection_wind_mult"] = erection_wind_mult
        self.output_dict["crane_choice"] = crane_choice
        self.output_dict["crane_data_output"] = crane_data_output
        self.output_dict["crane_cost_details"] = crane_cost_details
        self.output_dict["total_cost_summed_erection"] = total_cost_summed_erection

        # Put some diagnostic data on selected_detailed_data. This is the number of crews needed
        # To complete the construction withing the construction duration.

        # Operational days over time construct days isn't necessarily a proxy for a particular value.
        # But in case it is useful for something, here it is.

        selected_detailed_data["Operational construct days over time construct days"] = np.ceil(
            selected_detailed_data["Operational construct days"] / selected_detailed_data["Time construct days"]
        )

        total_time_construct_days = (selected_detailed_data["Time construct days"]).sum()
        self.output_dict["time_weighted_weather_multiplier"] = (
            selected_detailed_data["Wind multiplier"]
            * (selected_detailed_data["Time construct days"])
            / total_time_construct_days
        ).sum()

        # Now get the number of equipment diagnostic data ready. This is held on an instance
        # attribute because it isn't meant to be used outside of the class.
        self._number_of_equip = selected_detailed_data.merge(
            self._possible_crane_cost, on=["Crane name", "Boom system", "Operation"], how="inner", sort=True
        )
        self._number_of_equip = self._number_of_equip[["Operation", "Crane name", "Boom system", "Number of equipment"]]

        # Management crews data
        self.output_dict["management_crews_cost"] = management_crews_cost
        self.output_dict["management_crews_cost_grouped"] = management_crews_cost_grouped
        self.output_dict["erection_selected_detailed_data"] = selected_detailed_data
