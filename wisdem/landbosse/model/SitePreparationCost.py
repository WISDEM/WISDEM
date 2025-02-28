import numpy as np
import math
from wisdem.landbosse.model.WeatherDelay import WeatherDelay as WD
import traceback
from wisdem.landbosse.model.CostModule import CostModule
import pandas as pd


class SitePreparationCost(CostModule):
    """
    **SitePreparationCost.py**

    - Created by Annika Eberle and Owen Roberts on Apr. 3, 2018

    - Refactored by Parangat Bhaskar and Alicia Key on Jun 3, 2019



    **Calculates cost of constructing roads for land-based wind projects:**

    - Get terrain complexity

    - Get turbine spacing

    - Get road width

    - Get number of turbines

    - Get turbine rating

    - Get duration of construction*  #todo: add to process diagram

    - Get road length

    - Get weather data

    - Get road thickness

    - Calculate volume of road based on road thickness, road width, and road length

    - Calculate road labor and equipment costs by operation and type using RSMeans data

    - Calculate man hours and equipment hours for compaction of soil based on road length, road thickness, soil type, road width, and equipment size

    - Calculate man hours and equipment hours for mass material movement based on land cover, terrain complexity, and road length

    - Calculate man hours and equipment hours for rock placement based on equipment size, distance to quarry, and volume of road

    - Calculate man hours and equipment hours for compaction of rock based on road length, road thickness, and rock type

    - Calculate man hours and equipment hours for final grading based on road length

    - Calculate quantity of materials based on volume of materials

    - Calculate material costs by type

    - Calculate material costs using quantity of materials by material type and material prices by material type

    - Sum road costs over all operations and material types to get total costs by type of cost (e.g., material vs. equipment)

    - Return total road costs by type of cost


    **Keys in the input dictionary are the following:**

    - road_length

    - road_width

    - road_thickness

    - crane_width

    - num_access_roads  #TODO: Add to excel inputs sheet

    - num_turbines

    - rsmeans (dataframe)

    - duration_construction

    - wind_delays

    - wind_delay_time

    - material_price (dataframe)

    - rsmeans_per_diem

    - rotor_diameter_m

    - turbine_spacing_rotor_diameters


    **Keys in the output dictionary are the following:**

    - road_length_m

    - road_volume_m3

    - depth_to_subgrade_m

    - crane_path_widthm

    - road_thickess_m

    - road_width_m

    - road_width_m

    - material_volume_cubic_yards

    - road_construction_time

    - topsoil_volume

    - embankment_volume_crane

    - embankment_volume_road

    - rough_grading_area

    - material_needs (dataframe)

    - operation_data (dataframe)

    - total_road_cost



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

        # Road quality and fraction of roads that are
        self.fraction_new_roads = self.input_dict["fraction_new_roads"]
        self.road_quality = self.input_dict["road_quality"]

        # Conversion factors. Making this data private (hidden from outside of this class):
        self._meters_per_foot = 0.3
        self._meters_per_inch = 0.025
        self._cubic_yards_per_cubic_meter = 1.30795
        self._square_feet_per_square_meter = 10.7639

        # cubic meters for crane pad and maintenance ring for each turbine
        # (from old BOS model - AI - Access Roads & Site Imp. tab cell J33)
        self._crane_pad_volume = 125

        # conversion factor for converting packed cubic yards to loose cubic yards
        # material volume is about 1.4 times greater than road volume due to compaction
        self._yards_loose_per_yards_packed = 1.39

        # calculated road properties
        self._lift_depth_m = 0.2

    def calculate_road_properties(self, road_properties_input, road_properties_output):
        """
        Calculates the volume of road materials need based on length, width, and thickness of roads

        Parameters
        ----------

        int num_turbines
            number of turbines in wind farm

        unitless turbine_spacing_rotor_diameters
            Immediate spacing between two turbines as a function of rotor diameter

        int rotor_diameter_m
            Rotor diameter

        float crane_width
            Crane width in meters

        float road_thickness
            Road thickness in inches

        float road_width_ft
            Road width in feet



        Returns
        ----------

        road_length_m
            Calculated road length (in m)

        road_volume_m3
            Calculated road volume (in m^3)

        material_volume_cubic_yards
            Material volume required (in cubic yards) based on road volume


        """

        if road_properties_input["road_distributed_wind"] == True or road_properties_input["turbine_rating_MW"] < 0.1:
            road_properties_output["road_volume"] = (
                road_properties_input["road_length_adder_m"]
                * (road_properties_input["road_width_ft"] * self._meters_per_foot)
                * (road_properties_input["road_thickness"] * self._meters_per_inch)
            )  # units of cubic meters
        else:
            road_properties_output["road_length_m"] = (
                (road_properties_input["num_turbines"] - 1)
                * road_properties_input["turbine_spacing_rotor_diameters"]
                * road_properties_input["rotor_diameter_m"]
            ) + road_properties_input["road_length_adder_m"]
            road_properties_output["road_width_m"] = road_properties_input["road_width_ft"] * self._meters_per_foot
            road_properties_output["road_volume"] = (
                road_properties_output["road_length_m"]
                * (road_properties_input["road_width_ft"] * self._meters_per_foot)
                * (road_properties_input["road_thickness"] * self._meters_per_inch)
            )  # units of cubic meters

        road_properties_output["road_volume_m3"] = (
            road_properties_output["road_volume"] + self._crane_pad_volume * road_properties_input["num_turbines"]
        )  # in cubic meters

        road_properties_output["depth_to_subgrade_m"] = 0.1

        road_properties_output["crane_path_width_m"] = road_properties_input["crane_width"] + 1.5

        # todo: replace with actual crane path width from erection module
        road_properties_output["road_thickness_m"] = road_properties_input["road_thickness"] * self._meters_per_inch
        road_properties_output["road_width_m"] = road_properties_input["road_width_ft"] * self._meters_per_foot
        road_properties_output["material_volume_cubic_yards"] = (
            road_properties_output["road_volume_m3"]
            * self._cubic_yards_per_cubic_meter
            * self._yards_loose_per_yards_packed
        )  # todo: output_dict material volume

        return road_properties_output

    def estimate_construction_time(self, estimate_construction_time_input, estimate_construction_time_output):
        """
        Estimates construction time of roads for entire project.

        Parameters
        ----------
        float crane_path_width_m
            Width of crane path (in m)

        float road_length_m
            Road length (in m)

        float depth_to_subgrade_m
            Depth to subgarde (in m)

        float road_volume
            Road volume (in m^3)

        float road_thickness_m
            Road thickness (in m)



        Returns
        ----------

        pd.DataFrame operation_data
            Dataframe which conatains following outputs:

            -  Number of days required for construction

            - Number of crews required to complete roads construction in specified construction time

            - Cost of labor and equipment rental prior to weather delays

        """
        throughput_operations = estimate_construction_time_input["rsmeans"]

        # TODO: Figure out where 'construct_duration' gets read in.
        estimate_construction_time_output["road_construction_time"] = (
            estimate_construction_time_input["construct_duration"] * 1 / 5
        )  # assumes road construction occurs for 1/5 of project time

        # Main switch between small DW wind and (utility scale + distributed wind)
        # select operations for roads module that have data
        if estimate_construction_time_input["turbine_rating_MW"] >= 0.1:
            operation_data = throughput_operations.where(throughput_operations["Module"] == "Roads").dropna(thresh=4)
        else:
            operation_data = throughput_operations.where(throughput_operations["Module"] == "Small DW Roads").dropna(
                thresh=4
            )
            operation_data = operation_data.dropna(subset=["Units"])

        # create list of unique material units for operations
        list_units = operation_data["Units"].unique()

        if (
            estimate_construction_time_input["road_distributed_wind"] == True
            and estimate_construction_time_input["turbine_rating_MW"] >= 0.1
        ):
            estimate_construction_time_output["topsoil_volume"] = (
                estimate_construction_time_input["site_prep_area_m2"]
                * (estimate_construction_time_output["depth_to_subgrade_m"])
                * self._cubic_yards_per_cubic_meter
            )  # units: cubic yards
            estimate_construction_time_output["embankment_volume_crane"] = estimate_construction_time_output[
                "topsoil_volume"
            ]  # units: cubic yards
            estimate_construction_time_output["embankment_volume_road"] = estimate_construction_time_output[
                "topsoil_volume"
            ]  # units: cubic yards
            estimate_construction_time_output["rough_grading_area"] = (
                estimate_construction_time_input["site_prep_area_m2"] * 10.76391
            ) / 100000  # where 10.76391 sq ft = 1 sq m

        elif (
            estimate_construction_time_input["road_distributed_wind"] == True
            and estimate_construction_time_input["turbine_rating_MW"] < 0.1
        ):
            estimate_construction_time_output["topsoil_volume"] = (
                estimate_construction_time_input["road_length_adder_m"]
                * (estimate_construction_time_input["road_width_ft"] * 0.3048)
                * (estimate_construction_time_input["road_thickness"] * 0.0254)
                * 1.308
            )  # Units: CY (where 1 m3 = 1.308 CY)

        else:
            estimate_construction_time_output["topsoil_volume"] = (
                (estimate_construction_time_output["crane_path_width_m"])
                * estimate_construction_time_output["road_length_m"]
                * (estimate_construction_time_output["depth_to_subgrade_m"])
                * self._cubic_yards_per_cubic_meter
            )  # units: cubic yards
            estimate_construction_time_output["embankment_volume_crane"] = (
                (estimate_construction_time_output["crane_path_width_m"])
                * estimate_construction_time_output["road_length_m"]
                * (estimate_construction_time_output["depth_to_subgrade_m"])
                * self._cubic_yards_per_cubic_meter
            )  # units: cubic yards
            estimate_construction_time_output["embankment_volume_road"] = (
                estimate_construction_time_output["road_volume"]
                * self._cubic_yards_per_cubic_meter
                * math.ceil(estimate_construction_time_output["road_thickness_m"] / self._lift_depth_m)
            )  # units: cubic yards road
            estimate_construction_time_output["rough_grading_area"] = (
                estimate_construction_time_output["road_length_m"]
                * estimate_construction_time_output["road_width_m"]
                * self._square_feet_per_square_meter
                * math.ceil(estimate_construction_time_output["road_thickness_m"] / self._lift_depth_m)
                / 100000
            )  # Units: Each (100,000 square feet)

        if estimate_construction_time_input["turbine_rating_MW"] >= 0.1:
            material_quantity_dict = {
                "cubic yard": estimate_construction_time_output["topsoil_volume"],
                "embankment cubic yards crane": estimate_construction_time_output["embankment_volume_crane"],
                "embankment cubic yards road": estimate_construction_time_output["embankment_volume_road"],
                "loose cubic yard": estimate_construction_time_output["material_volume_cubic_yards"],
                "Each (100000 square feet)": estimate_construction_time_output["rough_grading_area"],
            }
        else:  # small DW
            material_quantity_dict = {
                "cubic yard": estimate_construction_time_output["topsoil_volume"],
                "embankment cubic yards crane": estimate_construction_time_output["topsoil_volume"],
                "loose cubic yard": estimate_construction_time_output["topsoil_volume"],
                "embankment cubic yards road": estimate_construction_time_output["topsoil_volume"],
            }

        material_needs = pd.DataFrame(
            data=[[unit, material_quantity_dict[unit]] for unit in list_units],
            columns=["Units", "Quantity of material"],
        )

        estimate_construction_time_output["material_needs"] = material_needs

        # join material needs with operational data to compute costs
        operation_data = pd.merge(operation_data, material_needs, on=["Units"], sort=True).dropna(thresh=3)
        operation_data = operation_data.where((operation_data["Daily output"]).isnull() == False).dropna(thresh=4)

        # calculate operational parameters and estimate costs without weather delays
        operation_data["Number of days"] = operation_data["Quantity of material"] / operation_data["Daily output"]
        operation_data["Number of crews"] = np.ceil(
            (operation_data["Number of days"] / 30.0) / estimate_construction_time_output["road_construction_time"]
        )
        operation_data["Cost USD without weather delays"] = (
            operation_data["Quantity of material"] * operation_data["Rate USD per unit"]
        )

        # if more than one crew needed to complete within construction duration then assume that all construction happens
        # within that window and use that time frame for weather delays; if not, use the number of days calculated
        operation_data["time_construct_bool"] = (
            operation_data["Number of days"] > estimate_construction_time_output["road_construction_time"] * 30.0
        )
        boolean_dictionary = {True: estimate_construction_time_output["road_construction_time"] * 30.0, False: np.nan}
        operation_data["time_construct_bool"] = operation_data["time_construct_bool"].map(boolean_dictionary)
        operation_data["Time construct days"] = operation_data[["time_construct_bool", "Number of days"]].min(axis=1)
        num_days = operation_data["Time construct days"].max()

        # pull out management data
        if estimate_construction_time_input["turbine_rating_MW"] >= 0.1:
            crew_cost = self.input_dict["crew_cost"]
            crew = self.input_dict["crew"][self.input_dict["crew"]["Crew type ID"].str.contains("M0")]
            management_crew = pd.merge(crew_cost, crew, on=["Labor type ID"], sort=True)
            management_crew = management_crew.assign(
                per_diem_total=management_crew["Per diem USD per day"] * management_crew["Number of workers"] * num_days
            )
            management_crew = management_crew.assign(
                hourly_costs_total=management_crew["Hourly rate USD per hour"]
                * self.input_dict["hour_day"][self.input_dict["time_construct"]]
                * num_days
            )
            management_crew = management_crew.assign(
                total_crew_cost_before_wind_delay=management_crew["per_diem_total"]
                + management_crew["hourly_costs_total"]
            )
            self.output_dict["management_crew"] = management_crew

            self.output_dict["managament_crew_cost_before_wind_delay"] = management_crew[
                "total_crew_cost_before_wind_delay"
            ].sum()

        estimate_construction_time_output["operation_data"] = operation_data

        return operation_data

    def calculate_weather_delay(self, weather_delay_input_data, weather_delay_output_data):
        """
        Calculates wind delay for roads.

        """

        # construct WeatherDelay module
        WD(weather_delay_input_data, weather_delay_output_data)

        # compute weather delay
        wind_delay = pd.DataFrame(weather_delay_output_data["wind_delays"])

        # if greater than 4 hour delay, then shut down for full day (10 hours)
        wind_delay[(wind_delay > 4)] = 10
        weather_delay_output_data["wind_delay_time"] = float(wind_delay.sum().iloc[0])

        return weather_delay_output_data

    def new_and_existing_total_road_cost(self, new_road_cost_by_type):
        """
        Calculates total road cost, with contribution from new and existing roads. Refer to eq. 3.3.10 in Technical Report.

        Returns
        -------
        float new_and_existing_total_road_cost_USD_df
            Sum total cost in USD of new and existing roads.
        """

        f_new = (
            self.fraction_new_roads
        )  # fraction of new roads that will be built (default assumes 33% new). TODO: Change to user input.
        r_q = (
            self.road_quality
        )  # is a non-dimensional representation of road quality. Default assumes 0.6, which is representative of average road conditions. TODO: Change to user input.
        cost_existing_roads = (
            0.5 * new_road_cost_by_type
        )  # cost of installing existing roads for the entire project. We currently assume default case to equal 50 % of cost of building new roads.
        new_and_existing_total_road_cost = (f_new * new_road_cost_by_type) + ((1 - f_new) * r_q * cost_existing_roads)
        return new_and_existing_total_road_cost

    def calculate_costs(self, calculate_cost_input_dict, calculate_cost_output_dict):
        """
        Function to calculate total labor, equipment, material, mobilization, and anyother associated costs after
        factoring in weather delays.


        Parameters
        ----------
        pd.Dataframe RSMeans
            Dataframe containing labor and equipment rental costs

        pd.DataFrame operation_data
            DataFrame containing estimates for total roads construction time and cost


        Returns
        ----------

        pd.DataFrame total_road_cost
            Dataframe containing following calculated outputs (after weather delay considerations):

            - Total labor cost

            - Totoal material cost

            - Total equipment rental cost

            - Total mobilization cost

            - Any other related costs


        """
        rsmeans = calculate_cost_input_dict["rsmeans"]

        material_name = rsmeans["Material type ID"].where(rsmeans["Module"] == "Roads").dropna().unique()

        material_vol = pd.DataFrame(
            [[material_name[0], calculate_cost_output_dict["material_volume_cubic_yards"], "Loose cubic yard"]],
            columns=["Material type ID", "Quantity of material", "Units"],
        )

        material_data = pd.merge(
            material_vol, calculate_cost_input_dict["material_price"], on=["Material type ID"], sort=True
        )
        material_data["Cost USD"] = material_data["Quantity of material"] * pd.to_numeric(
            material_data["Material price USD per unit"]
        )

        # New roads material cost:
        material_cost_of_new_roads = material_data["Quantity of material"].iloc[0] * pd.to_numeric(
            material_data["Material price USD per unit"].iloc[0]
        )

        # New + old roads material cost:
        if calculate_cost_input_dict["turbine_rating_MW"] >= 0.1:
            material_cost_of_old_and_new_roads = self.new_and_existing_total_road_cost(material_cost_of_new_roads)
            material_costs = pd.DataFrame(
                [["Materials", float(material_cost_of_old_and_new_roads), "Roads"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )
        else:
            material_cost_of_old_and_new_roads = material_cost_of_new_roads
            material_costs = pd.DataFrame(
                [["Materials", float(material_cost_of_old_and_new_roads), "Small DW Roads"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        operation_data = self.estimate_construction_time(calculate_cost_input_dict, calculate_cost_output_dict)

        wind_delay_fraction = (
            calculate_cost_output_dict["wind_delay_time"] / calculate_cost_input_dict["operational_hrs_per_day"]
        ) / operation_data["Time construct days"].max(skipna=True)
        # check if wind_delay_fraction is greater than 1, which would mean weather delays are longer than they can possibily be for the input data
        if wind_delay_fraction > 1:
            raise ValueError("{}: Error: Wind delay greater than 100%".format(type(self).__name__))
        calculate_cost_output_dict["wind_multiplier"] = 1.0 / (1.0 - wind_delay_fraction)

        per_diem = (
            operation_data["Number of workers"]
            * operation_data["Number of crews"]
            * (operation_data["Time construct days"] + np.ceil(operation_data["Time construct days"] / 7.0))
            * calculate_cost_input_dict["rsmeans_per_diem"]
        )
        labor_per_diem = per_diem.dropna()

        # calculate_cost_output_dict['Total per diem (USD)'] = per_diem.sum()
        labor_equip_data = pd.merge(
            operation_data[["Operation ID", "Units", "Quantity of material"]],
            rsmeans,
            on=["Units", "Operation ID"],
            sort=True,
        )
        # Calculating labor costs:
        if calculate_cost_input_dict["turbine_rating_MW"] >= 0.1:
            labor_equip_data["Calculated per diem"] = per_diem
            labor_data = labor_equip_data[labor_equip_data["Type of cost"] == "Labor"].copy()
        else:
            labor_equip_data["Calculated per diem"] = 0.0
            # labor_data = labor_equip_data[labor_equip_data['Type of cost'] == 'Labor'].copy()
            labor_data = labor_equip_data[labor_equip_data["Module"] == "Small DW Roads"].copy()

        overtime_multiplier = calculate_cost_input_dict["overtime_multiplier"]
        wind_multiplier = calculate_cost_output_dict["wind_multiplier"]

        labor_data["Cost USD"] = (
            (labor_data["Quantity of material"] * labor_data["Rate USD per unit"] * overtime_multiplier)
            + labor_per_diem
        ) * wind_multiplier

        if calculate_cost_input_dict["road_distributed_wind"] and calculate_cost_input_dict["turbine_rating_MW"] >= 0.1:

            labor_for_new_roads_cost_usd = (
                labor_data["Cost USD"].sum() + calculate_cost_output_dict["managament_crew_cost_before_wind_delay"]
            )

            labor_for_new_and_old_roads_cost_usd = self.new_and_existing_total_road_cost(labor_for_new_roads_cost_usd)
            labor_costs = pd.DataFrame(
                [["Labor", float(labor_for_new_and_old_roads_cost_usd), "Roads"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        elif (
            calculate_cost_input_dict["road_distributed_wind"] and calculate_cost_input_dict["turbine_rating_MW"] < 0.1
        ):  # small DW

            labor_for_new_roads_cost_usd = labor_data["Cost USD"].sum()
            labor_for_new_and_old_roads_cost_usd = self.new_and_existing_total_road_cost(labor_for_new_roads_cost_usd)

            labor_costs = pd.DataFrame(
                [["Labor", float(labor_for_new_and_old_roads_cost_usd), "Small DW Roads"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        else:
            labor_for_new_roads_cost_usd = (
                labor_data["Cost USD"].sum() + calculate_cost_output_dict["managament_crew_cost_before_wind_delay"]
            )

            labor_for_new_and_old_roads_cost_usd = self.new_and_existing_total_road_cost(labor_for_new_roads_cost_usd)

            labor_costs = pd.DataFrame(
                [["Labor", float(labor_for_new_and_old_roads_cost_usd), "Roads"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        # Filter out equipment costs from rsmeans tab:
        if calculate_cost_input_dict["turbine_rating_MW"] >= 0.1:
            equipment_data = labor_equip_data[labor_equip_data["Type of cost"] == "Equipment rental"].copy()
        else:
            equipment_data = labor_equip_data[labor_equip_data["Module"] == "Small DW Roads"].copy()
            equipment_data = equipment_data[equipment_data["Type of cost"] == "Equipment rental"].copy()

        equipment_data["Cost USD"] = (
            equipment_data["Quantity of material"] * equipment_data["Rate USD per unit"]
        ) * calculate_cost_output_dict["wind_multiplier"]

        # if rental cost is < half day minimum:
        if equipment_data["Cost USD"].sum() < 460:
            equip_for_new_roads_cost_usd = 460
        else:
            equip_for_new_roads_cost_usd = equipment_data["Cost USD"].sum()

        if calculate_cost_input_dict["turbine_rating_MW"] >= 0.1:
            equip_for_new_and_old_roads_cost_usd = self.new_and_existing_total_road_cost(equip_for_new_roads_cost_usd)
            equipment_costs = pd.DataFrame(
                [["Equipment rental", float(equip_for_new_and_old_roads_cost_usd), "Roads"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )
        else:
            equip_for_new_and_old_roads_cost_usd = self.new_and_existing_total_road_cost(equip_for_new_roads_cost_usd)
            equipment_costs = pd.DataFrame(
                [["Equipment rental", float(equip_for_new_and_old_roads_cost_usd), "Small DW Roads"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        # add costs for other operations not included in process data for utility mode (e.g., fencing, access roads)
        if calculate_cost_input_dict["turbine_rating_MW"] > 0.1:
            num_turbines = calculate_cost_input_dict["num_turbines"]
            rotor_diameter_m = calculate_cost_input_dict["rotor_diameter_m"]
            construct_duration = calculate_cost_input_dict["construct_duration"]
            num_access_roads = calculate_cost_input_dict["num_access_roads"]

            cost_new_roads_adder = (
                (float(num_turbines) * 17639)
                + (float(num_turbines) * float(rotor_diameter_m) * 24.8)
                + (float(construct_duration) * 55500)
                + float(num_access_roads) * 3800
            )
            cost_adder = self.new_and_existing_total_road_cost(cost_new_roads_adder)
            additional_costs = pd.DataFrame(
                [["Other", float(cost_adder), "Roads"]], columns=["Type of cost", "Cost USD", "Phase of construction"]
            )

        else:  # No 'Other' cost in distributed wind mode:
            cost_new_roads_adder = 0
            cost_adder = self.new_and_existing_total_road_cost(cost_new_roads_adder)
            additional_costs = pd.DataFrame(
                [["Other", float(cost_adder), "Small DW Roads"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        # Create empty road cost (showing cost breakdown by type) dataframe:
        road_cost_columns = ["Type of cost", "Cost USD", "Phase of construction"]

        # Filter out equipment costs from rsmeans tab:
        equipment_data = labor_equip_data[labor_equip_data["Type of cost"] == "Equipment rental"].copy()
        equipment_data["Cost USD"] = (
            equipment_data["Quantity of material"] * equipment_data["Rate USD per unit"]
        ) * calculate_cost_output_dict[
            "wind_multiplier"
        ]  # TODO: Annika can you confirm if this is correct.

        equip_for_new_roads_cost_usd = equipment_data["Cost USD"].sum()
        equip_for_new_and_old_roads_cost_usd = self.new_and_existing_total_road_cost(equip_for_new_roads_cost_usd)
        equipment_costs = pd.DataFrame(
            [["Equipment rental", float(equip_for_new_and_old_roads_cost_usd), "Roads"]],
            columns=road_cost_columns,
        )

        # add costs for other operations not included in process data (e.g., fencing, access roads)
        #
        # Assume the following things to compute the cost adder:
        #
        # The crews only work 6 days per week, so a number of days equal to the
        # number of weeks worked is added to convert working days to calendar
        # days.
        #
        # All crews work concurrently, so the total site preparation time is the
        # the max of "Time construct days"

        max_time_construct_days = operation_data["Time construct days"].max()
        num_turbines = float(calculate_cost_input_dict["num_turbines"])
        rotor_diameter_m = float(calculate_cost_input_dict["rotor_diameter_m"])
        num_access_roads = float(calculate_cost_input_dict["num_access_roads"])
        calendar_construct_days = max_time_construct_days + np.ceil(
            max_time_construct_days / 6
        )  # assumes working only 6 days per week
        siteprep_construction_months = calendar_construct_days / 30.0
        cost_new_roads_adder = (
            num_turbines * 17639
            + num_turbines * rotor_diameter_m * 24.8
            + calculate_cost_input_dict["construct_duration"] * 55500
            + num_access_roads * 3800
        )
        cost_adder = self.new_and_existing_total_road_cost(cost_new_roads_adder)
        additional_costs = pd.DataFrame([["Other", cost_adder, "Roads"]], columns=road_cost_columns)

        road_cost = pd.concat((material_costs, equipment_costs, labor_costs, additional_costs))

        # set mobilization cost equal to 5% of total road cost for utility scale model and function of
        # of turbine size for distributed wind:
        if calculate_cost_input_dict["num_turbines"] > 10:
            mobilization_costs_new_roads = road_cost["Cost USD"].sum() * 0.05
            mobilization_costs_new_plus_old_roads = self.new_and_existing_total_road_cost(mobilization_costs_new_roads)
            mobilization_costs = pd.DataFrame(
                [["Mobilization", mobilization_costs_new_plus_old_roads, "Roads"]],
                columns=road_cost_columns,
            )
        else:
            mobilization_costs_new_roads = road_cost["Cost USD"].sum() * self.mobilization_cost_multiplier(
                calculate_cost_input_dict["turbine_rating_MW"]
            )
            mobilization_costs_new_plus_old_roads = self.new_and_existing_total_road_cost(mobilization_costs_new_roads)

            if calculate_cost_input_dict["turbine_rating_MW"] >= 0.1:
                mobilization_costs = pd.DataFrame(
                    [["Mobilization", mobilization_costs_new_plus_old_roads, "Roads"]],
                    columns=road_cost_columns,
                )
            else:
                mobilization_costs = pd.DataFrame(
                    [["Mobilization", mobilization_costs_new_plus_old_roads, "Small DW Roads"]],
                    columns=road_cost_columns,
                )

        road_cost = pd.concat((road_cost, mobilization_costs))
        total_road_cost = road_cost
        calculate_cost_output_dict["total_road_cost"] = total_road_cost
        calculate_cost_output_dict["siteprep_construction_months"] = siteprep_construction_months
        return total_road_cost

    def outputs_for_detailed_tab(self, input_dict, output_dict):
        """
        Creates a list of dictionaries which can be used on their own or
        used to make a dataframe.


        Returns
        -------
        list(dict)
            A list of dicts, with each dict representing a row of the data.
        """
        result = []
        module = type(self).__name__

        # Note that some values are cast with float() so that XlsxWriter
        # (the library depended on by XlsxGenerator) can output them as
        # numbers. XlsxWriter, interestingly, cannot handle np.float32()
        # types.

        result.append(
            {
                "unit": "m^3",
                "type": "variable",
                "variable_df_key_col_name": "Total road volume",
                "value": float(self.output_dict["road_volume_m3"]),
            }
        )

        result.append(
            {
                "unit": "m",
                "type": "variable",
                "variable_df_key_col_name": "Depth to subgrade",
                "value": self.output_dict["depth_to_subgrade_m"],
            }
        )

        result.append(
            {
                "unit": "ft",
                "type": "variable",
                "variable_df_key_col_name": "Crane path width",
                "value": self.output_dict["crane_path_width_m"],  # TODO: Rename variable to: crane_path_width_ft
            }
        )

        if not input_dict["road_distributed_wind"]:
            result.append(
                {
                    "unit": "m",
                    "type": "variable",
                    "variable_df_key_col_name": "Road length",
                    "value": float(self.output_dict["road_length_m"]),
                }
            )

        result.append(
            {
                "unit": "m",
                "type": "variable",
                "variable_df_key_col_name": "Road width",
                "value": self.output_dict["road_width_m"],
            }
        )

        result.append(
            {
                "unit": "m",
                "type": "variable",
                "variable_df_key_col_name": "Road thickness",
                "value": self.output_dict["road_thickness_m"],
            }
        )

        result.append(
            {
                "unit": "cubic yards",
                "type": "variable",
                "variable_df_key_col_name": "Material volume",
                "value": float(self.output_dict["material_volume_cubic_yards"]),
            }
        )

        result.append(
            {
                "unit": "cubic yards",
                "type": "variable",
                "variable_df_key_col_name": "Topsoil volume",
                "value": float(self.output_dict["topsoil_volume"]),
            }
        )

        if input_dict["turbine_rating_MW"] >= 0.1:
            result.append(
                {
                    "unit": "cubic yards",
                    "type": "variable",
                    "variable_df_key_col_name": "Embankment volume crane",
                    "value": float(self.output_dict["embankment_volume_crane"]),
                }
            )

            result.append(
                {
                    "unit": "cubic yards",
                    "type": "variable",
                    "variable_df_key_col_name": "Embankment volume road",
                    "value": float(self.output_dict["embankment_volume_road"]),
                }
            )

            result.append(
                {
                    "unit": "ft^2",
                    "type": "variable",
                    "variable_df_key_col_name": "Rough grading area",
                    "value": float(self.output_dict["rough_grading_area"]),
                }
            )

        for row in self.output_dict["total_road_cost"].itertuples():
            dashed_row = "{} <--> {} <--> {}".format(row[1], row[3], math.ceil(row[2]))
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "Type of Cost <--> Phase of Construction <--> Cost in USD ",
                    "value": dashed_row,
                    "last_number": row[2],
                }
            )

        for _dict in result:
            _dict["project_id_with_serial"] = self.project_name
            _dict["module"] = module

        self.output_dict["roads_cost_csv"] = result
        return result

    def outputs_for_module_type_operation(self, input_dict, output_dict):
        """
        Outputs dictionaries that are rows for the
        costs_by_module_type_operation

        Returns
        -------
        list
            List of dicts, with each dict representing a row for
            the output.
        """
        result = []
        module = type(self).__name__
        num_turbines = self.input_dict["num_turbines"]

        costs_by_module_type_operation = output_dict["total_road_cost"]
        for _, row in costs_by_module_type_operation.iterrows():
            _dict = dict()
            row = row.to_dict()
            _dict["operation_id"] = row["Phase of construction"]
            _dict["type_of_cost"] = row["Type of cost"]
            _dict["cost"] = row["Cost USD"]
            result.append(_dict)

        for _dict in result:
            _dict["project_id_with_serial"] = self.project_name
            _dict["module"] = module
            _dict["total_or_turbine"] = "total"
            _dict["num_turbines"] = num_turbines

        output_dict["roads_cost_module_type_operation"] = result
        return result

    def run_module(self):
        """
        Runs the SitePreparation module and populates the IO dictionaries with calculated values.

        """
        try:
            self.calculate_road_properties(self.input_dict, self.output_dict)
            operation_data = self.estimate_construction_time(self.input_dict, self.output_dict)

            # pull only global inputs for weather delay from input_dict
            weather_data_keys = ("wind_shear_exponent", "weather_window")

            # specify roads-specific weather delay inputs
            self.weather_input_dict = dict(
                [(i, self.input_dict[i]) for i in self.input_dict if i in set(weather_data_keys)]
            )
            self.weather_input_dict["start_delay_hours"] = (
                0  # assume zero start for when road construction begins (start at beginning of construction time)
            )
            self.weather_input_dict["critical_wind_speed_m_per_s"] = self.input_dict[
                "critical_speed_non_erection_wind_delays_m_per_s"
            ]
            self.weather_input_dict["wind_height_of_interest_m"] = self.input_dict[
                "critical_height_non_erection_wind_delays_m"
            ]

            # compute and specify weather delay mission time for roads
            duration_construction = operation_data["Time construct days"].max(skipna=True)
            operational_hrs_per_day = self.input_dict["hour_day"][self.input_dict["time_construct"]]
            mission_time_hrs = duration_construction * operational_hrs_per_day
            self.weather_input_dict["mission_time_hours"] = mission_time_hrs

            self.calculate_weather_delay(self.weather_input_dict, self.output_dict)
            self.calculate_costs(self.input_dict, self.output_dict)
            self.outputs_for_detailed_tab(self.input_dict, self.output_dict)
            # self.outputs_for_module_type_operation(self.input_dict, self.output_dict)
            self.output_dict["siteprep_module_type_operation"] = self.outputs_for_costs_by_module_type_operation(
                input_df=self.output_dict["total_road_cost"], project_id=self.project_name, total_or_turbine=True
            )
            return 0, 0  # module ran successfully
        except Exception as error:
            traceback.print_exc()
            print(f"Fail {self.project_name} SitePreparationCost")
            return 1, error  # module did not run successfully
