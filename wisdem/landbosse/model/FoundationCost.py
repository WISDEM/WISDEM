import math
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

from wisdem.landbosse.model.CostModule import CostModule
from wisdem.landbosse.model.WeatherDelay import WeatherDelay as WD


class FoundationCost(CostModule):
    """
    **FoundationCost.py**

    - Created by Annika Eberle and Owen Roberts on Apr. 3, 2018

    - Refactored by Parangat Bhaskar and Alicia Key on June 3, 2019

    \nCalculates the costs of constructing foundations for land-based wind projects *(items in brackets are not yet implemented)*:

    * Get number of turbines
    * Get duration of construction
    * Get daily hours of operation*  # todo: add to process diagram
    * Get season of construction*  # todo: add to process diagram
    * [Get region]
    * Get rotor diameter
    * Get hub height
    * Get turbine rating
    * Get buoyant foundation design flag
    * [Get seismic zone]
    * Get tower technology type
    * Get hourly weather data
    * [Get specific seasonal delays]
    * [Get long-term, site-specific climate data]
    * Get price data
    * Get labor rates
    * Get material prices for steel and concrete
    * [Use region to determine weather data]


    \n\nGiven below is the set of calculations carried out in this module:

    * Calculate the foundation loads using the rotor diameter, hub height, and turbine rating

    * Determine the foundation size based on the foundation loads, buoyant foundation design flag, and type of tower technology

    * Estimate the amount of material needed for foundation construction based on foundation size and number of turbines

    * Estimate the amount of time required to construct foundation based on foundation size, hours of operation, duration of construction, and number of turbines

    * Estimate the additional amount of time for weather delays (currently only assessing wind delays) based on hourly weather data, construction time, hours of operation, and season of construction

    * Estimate the amount of labor required for foundation construction based on foundation size, construction time, and weather delay
        * Calculate number of workers by crew type
        * Calculate man hours by crew type

    * Estimate the amount of equipment needed for foundation construction based on foundation size, construction time, and weather delay
        * Calculate number of equipment by equip type
        * Calculate equipment hours by equip type

    - Calculate the total foundation cost based on amount of equipment, amount of labor, amount of material, and price data.


    **Keys in the input dictionary are the following:**

    depth
        (int) depth of foundation [in m]


    component_data
        (pd.DataFrame) data frame with wind turbine component data

    def __init__(self, input_dict, output_dict, project_name):
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.project_name = project_name


    num_turbines
        (int) total number of turbines in wind farm

    duration_construction
        (int) estimated construction time in months

    num_delays
        (int) Number of delay events

    avg_hours_per_delay
        (float) Average hours per delay event

    std_dev_hours_per_delay
        (float) Standard deviation from average hours per delay event

    delay_speed_m_per_s
        (float) wind speed above which weather delays kick in

    start_delay_hours
        (int)

    mission_time_hours
        (int)

    gust_wind_speed_m_per_s
        (float)

    wind_height_of_interest_m
        (int)

    wind_shear_exponent
        (float)

    season_construct
        list of seasons (like ['spring', 'summer']) for the construction.

    time_construct
        list of time windows for constructions. Use ['normal'] for a
        0800 to 1800 schedule 10 hour schedule. Use ['long'] for an
        overnight 1800 to 2359, 0000 to 0759 overnight schedule. Use
        ['normal', 'long'] for a 24-hour schedule.

    operational_hrs_per_day
        (float)


    material_price
        (pd.DataFrame) dataframe containing foundation cost related material prices

    rsmeans
        (pd.DataFrame) TODO: Formal definition for rsmeans?


    **Keys in the output dictionary are the following:**

    F_dead_kN_per_turbine
        (float) foundation dead load [in kN]

    F_horiz_kN_per_turbine
        (float) total lateral load [kN]

    M_tot_kN_m_per_turbine
        (float) Moment [kN.m]

    Radius_o_m
        (float) foundation radius based on overturning moment [in m]

    Radius_g_m
        (float) foundation radius based on gapping [in m]

    Radius_b_m
        (float) foundation radius based on bearing pressure [in m]

    Radius_m
        (float) largest foundation radius based on all three foundation design criteria: moment, gapping, bearing [in m]

    foundation_volume_concrete_m3_per_turbine
        (float) volume of a round, raft foundation [in m^3]

    steel_mass_short_ton
        (float) short tons of reinforcing steel

    material_needs_per_turbine
        (pd.DataFrame) table containing material needs info for -> Steel - rebar, Concrete 5000 psi, Excavated dirt, Backfill.

    operation_data
        (pd.DataFrame) TODO: What's the best one line definition for this?


    **TODO: Weather delay set of outputs -> ask Alicia for formal definitions of these keys.**

    total_foundation_cost
        (pd.DataFrame) summary of foundation costs (in USD) broken down into 4 main categories:
        1. Equipment Rental
        2. Labor
        3. Materials
        4. Mobilization
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

        # Constants used in FoundationCost class. Making this data private (hidden from outside of this class): #TODO: Change private variables to protected.
        self._kg_per_tonne = 1000
        self._cubicm_per_cubicft = 0.0283168
        self._steel_density = 9490  # kg / m^3
        self._cubicyd_per_cubicm = 1.30795
        self._ton_per_tonne = 0.907185

    def calculate_foundation_load(self, foundation_load_input_data, foundation_load_output_data):
        """

        Function to calculate foundation load.

        Parameters
        -------
        Int Section height m

        Surface area sq (in m^2)

        Coeff drag (installed)

        Lever arm m (in m)

        Multplier drag rotor

        Multiplier tower drag

        Mass tonne



        Returns
        -------
        Dead load [in N] -> F_dead_kN_per_turbine

        Lateral load [in N] -> F_horiz_kN_per_turbine

        Moment [N.m] -> M_tot_kN_m_per_turbine

        Foundation radius based on overturning moment [in m] -> Radius_o_m

        Foundation radius based on slipping [in m] -> Radius_s_m

        Foundation radius based on gapping [in m] -> Radius_g_m

        Foundation radius based on bearing pressure [in m] -> Radius_b_m

        Largest foundation radius based on all three foundation design criteria (moment, gapping, bearing [in m]) -> Radius_m

        Raises
        ------
        ValueError
            Raises a value error if r_bearing is calculated to be a negative value.
        """
        # set exposure constants
        a = 9.5
        z_g = 274.32

        # get section height
        z = foundation_load_input_data["Section height m"]

        # get cross-sectional area
        a_f = foundation_load_input_data["Surface area sq m"]

        # get coefficient of drag
        c_d = foundation_load_input_data["Coeff drag (installed)"]

        # get lever arm
        l = foundation_load_input_data["Lever arm m"]

        # get multipliers for tower and rotor
        multiplier_rotor = foundation_load_input_data["Multplier drag rotor"]
        multiplier_tower = foundation_load_input_data["Multiplier tower drag"]

        # calculate wind pressure
        k_z = 2.01 * (z / z_g) ** (2 / a)  # exposure factor
        k_d = 0.95  # wind directionality factor
        k_zt = 1  # topographic factor
        v = foundation_load_input_data["gust_velocity_m_per_s"]
        wind_pressure = 0.613 * k_z * k_zt * k_d * v**2

        # calculate wind loads on each tower component
        g = 0.85  # gust factor
        c_f = 0.6  # coefficient of force
        f_t = (wind_pressure * g * c_f * a_f) * multiplier_tower

        # calculate drag rotor
        rho = 1.225  # air density in kg/m^3
        f_r = (0.5 * rho * c_d * a_f * v**2) * multiplier_rotor

        f = f_t + f_r

        # calculate dead load in N
        g = 9.8  # m / s ^ 2
        f_dead = (
            sum(foundation_load_input_data["Mass tonne"]) * g * self._kg_per_tonne / 1.15
        )  # scaling factor to adjust dead load for uplift

        # calculate moment from each component at base of tower
        m_overturn = f * l

        # get total lateral load (N) and moment (N * m)
        f_lat = f.sum()  # todo: add f_lat (drag force) to output csv
        m_overturn = m_overturn.sum()

        # compare to moment from rated thrust
        rated_thrust = foundation_load_input_data["rated_thrust_N"]
        m_thrust = rated_thrust * max(l)
        m_tot = max(m_thrust, m_overturn)

        # compare lateral load to rated thrust
        f_horiz = max(f_lat, rated_thrust)

        # calculate foundation radius based on overturning moment
        vol_fraction_fill = 0.55
        vol_fraction_concrete = 1 - vol_fraction_fill
        safety_overturn = 1.5
        unit_weight_fill = 17.3e3  # in N / m^3
        unit_weight_concrete = 23.6e3  # in N / m^3
        bearing_pressure = foundation_load_input_data["bearing_pressure_n_m2"]
        p = [
            (
                np.pi
                * foundation_load_input_data["depth"]
                * (vol_fraction_fill * unit_weight_fill + vol_fraction_concrete * unit_weight_concrete)
            ),
            0,
            f_dead,
            -(safety_overturn * (m_tot + f_horiz * foundation_load_input_data["depth"])),
        ]
        r_overturn = np.roots(p)
        r_overturn = np.real(r_overturn[np.isreal(r_overturn)])[0]

        # calculate foundation radius based on slipping
        safety_slipping = 1.5
        friction_angle_soil = 25
        tangent_slip_angle = math.tan((friction_angle_soil * math.pi) / 180)
        slipping_force_with_sf = safety_slipping * f_lat
        # first check if slipping is already satisfied by dead weight
        if slipping_force_with_sf < (f_dead * tangent_slip_angle):
            r_slipping = 0
        else:
            # Calculate foundation radius based on slipping:
            r_slipping = (
                ((slipping_force_with_sf / tangent_slip_angle) - f_dead)
                / (
                    (vol_fraction_fill * unit_weight_fill + vol_fraction_concrete * unit_weight_concrete)
                    * math.pi
                    * foundation_load_input_data["depth"]
                )
            ) ** 0.5

        r_test_gapping = max(r_overturn, r_slipping)

        # calculate foundation radius based on gapping
        # check if gapping constrain is already satisfied - r / 3 < e
        foundation_vol = np.pi * r_test_gapping**2 * foundation_load_input_data["depth"]
        v_1 = (
            foundation_vol * (vol_fraction_fill * unit_weight_fill + vol_fraction_concrete * unit_weight_concrete)
            + f_dead
        )
        e = m_tot / v_1
        if (r_test_gapping / 3) < e:
            r_gapping = 0
        else:

            def r_g(x):
                foundation_vol = np.pi * x**2 * foundation_load_input_data["depth"]
                v_1 = (
                    foundation_vol
                    * (vol_fraction_fill * unit_weight_fill + vol_fraction_concrete * unit_weight_concrete)
                    + f_dead
                )
                e = m_tot / v_1
                return e * 3 - x

            result = root_scalar(r_g, method="brentq", bracket=[0.5 * r_overturn, 50], xtol=1e-4, maxiter=50)
            r_gapping = result.root
            if not result.converged:
                raise ValueError(
                    f"Warning {self.project_name} calculate_foundation_load r_gapping solve failed, {result.flag}"
                )

        r_test_bearing = max(r_test_gapping, r_gapping)

        # calculate foundation radius based on bearing pressure
        def r_b(x):
            foundation_vol = np.pi * r_test_bearing**2 * foundation_load_input_data["depth"]
            v_1 = (
                foundation_vol * (vol_fraction_fill * unit_weight_fill + vol_fraction_concrete * unit_weight_concrete)
                + f_dead
            )
            e = m_tot / v_1
            a_eff = v_1 / bearing_pressure
            return 2 * (x**2 - e * (x**2 - e**2) ** 0.5) - a_eff

        # Get minimum radius
        foundation_vol = np.pi * r_test_bearing**2 * foundation_load_input_data["depth"]
        v_1 = (
            foundation_vol * (vol_fraction_fill * unit_weight_fill + vol_fraction_concrete * unit_weight_concrete)
            + f_dead
        )
        min_r = m_tot / v_1

        result = root_scalar(r_b, method="brentq", bracket=[min_r + 1e-3, 50], xtol=1e-10, maxiter=50)
        r_bearing = result.root

        if not result.converged:
            raise ValueError(
                f"Warning {self.project_name} calculate_foundation_load r_bearing solve failed, {result.flag}"
            )

        # pick the largest foundation radius based on all 4 foundation design criteria: moment, gapping, bearing, slipping
        r_choosen = max(r_bearing, r_overturn, r_slipping, r_gapping)

        foundation_load_output_data["F_dead_kN_per_turbine"] = f_dead / 1e3
        foundation_load_output_data["F_horiz_kN_per_turbine"] = f_lat / 1e3
        foundation_load_output_data["M_tot_kN_m_per_turbine"] = m_tot / 1e3
        foundation_load_output_data["Radius_o_m"] = r_overturn
        foundation_load_output_data["Radius_s_m"] = r_slipping
        foundation_load_output_data["Radius_g_m"] = r_gapping
        foundation_load_output_data["Radius_b_m"] = r_bearing
        foundation_load_output_data["Radius_m"] = r_choosen

        return foundation_load_output_data

    def determine_foundation_size(self, foundation_size_input_data, foundation_size_output_data):
        """
        Function to calculate the volume of a round, raft foundation. Assumes foundation made of concrete with 1 m thickness.

        Parameters
        -------
        Largest foundation radius based on all three foundation design criteria: moment, gapping, bearing [in m] -> Radius_m [in m]

        depth of foundation [in m] -> depth


        Returns
        -------
        Foundation volume [in m^3] -> foundation_volume_concrete_m3_per_turbine

        """
        # TODO: still updating/fine-tuning foundation size equations for small DW (Parangat - Feb 27, 2020)
        r = float(foundation_size_output_data["Radius_m"])
        if foundation_size_input_data["turbine_rating_MW"] < 0.1:
            foundation_size_output_data["excavated_volume_m3"] = r * r * foundation_size_input_data["depth"] * np.pi
            foundation_size_output_data["foundation_volume_concrete_m3_per_turbine"] = (
                foundation_size_output_data["excavated_volume_m3"] * 0.45
            )
        else:
            foundation_size_output_data["excavated_volume_m3"] = (
                np.pi * (r + 0.5) ** 2 * foundation_size_input_data["depth"]
            )

            # only compute the portion of the foundation that is composed of concrete (45% concrete; other portion is
            # backfill); TODO: Add to sphinx -> (volume excavated = pi*(r_pick + .5m)^2 this assumes vertical sides which
            #  does not reflect reality as OSHA requires benched sides over 3’)
            foundation_size_output_data["foundation_volume_concrete_m3_per_turbine"] = (
                np.pi * r**2 * foundation_size_input_data["depth"] * 0.45
            )

        return foundation_size_output_data

    def estimate_material_needs_per_turbine(
        self, material_needs_per_turbine_input_data, material_needs_per_turbine_output_data
    ):
        """
        Function to estimate amount of material based on foundation size and number of turbines.


        Parameters
        -------
        Foundation concrete volume [in m^3] -> foundation_volume_concrete_m3_per_turbine


        Returns
        -------

        (Returns pd.DataFrame) material_needs_per_turbine


        """

        steel_mass_short_ton_per_turbine = (
            material_needs_per_turbine_output_data["foundation_volume_concrete_m3_per_turbine"]
            * 0.012
            * self._steel_density
            / self._kg_per_tonne
        )
        concrete_volume_cubic_yards_per_turbine = (
            material_needs_per_turbine_output_data["foundation_volume_concrete_m3_per_turbine"]
            * 0.985
            * self._cubicyd_per_cubicm
        )

        # Assign values to output dictionary:
        material_needs_per_turbine_output_data["material_needs_per_turbine"] = pd.DataFrame(
            [
                ["Steel - rebar", steel_mass_short_ton_per_turbine, "ton (short)"],
                ["Concrete 5000 psi", concrete_volume_cubic_yards_per_turbine, "cubic yards"],
                [
                    "Excavated dirt",
                    material_needs_per_turbine_output_data["excavated_volume_m3"] * self._cubicyd_per_cubicm,
                    "cubic_yards",
                ],
                [
                    "Backfill",
                    material_needs_per_turbine_output_data["excavated_volume_m3"] * self._cubicyd_per_cubicm,
                    "cubic_yards",
                ],
            ],
            columns=["Material type ID", "Quantity of material", "Units"],
        )

        material_needs_per_turbine_output_data["steel_mass_short_ton_per_turbine"] = steel_mass_short_ton_per_turbine
        return material_needs_per_turbine_output_data["material_needs_per_turbine"]

    def estimate_construction_time(self, construction_time_input_data, construction_time_output_data):
        """
        Function to estimate construction time on per turbine basis. TODO: What's a better definition of this function. It's task is to return a pd.DataFrame (operation_data).

        Parameters
        -------
        duration_construction

        pd.DataFrame
            rsmeans

        pd.DataFrame
            material_needs_per_turbine



        Returns
        -------

        (pd.DataFrame) operation_data

        """

        foundation_construction_time = construction_time_input_data["construct_duration"] * 1 / 3
        # throughput_operations = construction_time_input_data['throughput_operations']
        throughput_operations = construction_time_input_data["rsmeans"]
        material_needs_per_turbine = construction_time_output_data["material_needs_per_turbine"]
        quantity_materials_entire_farm = (
            material_needs_per_turbine["Quantity of material"] * construction_time_input_data["num_turbines"]
        )

        # Calculations for estimate construction time will be on entire wind farm basis:
        construction_time_output_data["material_needs_entire_farm"] = material_needs_per_turbine.copy()
        material_needs_entire_farm = construction_time_output_data["material_needs_entire_farm"]
        material_needs_entire_farm["Quantity of material"] = quantity_materials_entire_farm
        if construction_time_input_data["turbine_rating_MW"] <= 0.1:
            operation_data = throughput_operations.where(
                throughput_operations["Module"] == "Small DW Foundations"
            ).dropna(thresh=4)
        else:
            operation_data = throughput_operations.where(throughput_operations["Module"] == "Foundations").dropna(
                thresh=4
            )

        # operation data for entire wind farm:
        operation_data = pd.merge(
            material_needs_entire_farm, operation_data, on=["Material type ID"], how="outer", sort=True
        )
        operation_data["Number of days"] = operation_data["Quantity of material"] / operation_data["Daily output"]
        operation_data["Number of crews"] = np.ceil(
            (operation_data["Number of days"] / 30) / foundation_construction_time
        )

        alpha = operation_data[operation_data["Type of cost"] == "Labor"]
        operation_data_id_days_crews_workers = alpha[
            ["Operation ID", "Number of days", "Number of crews", "Number of workers"]
        ]

        # if more than one crew needed to complete within construction duration then assume that all construction happens
        # within that window and use that timeframe for weather delays; if not, use the number of days calculated
        operation_data["time_construct_bool"] = operation_data["Number of days"] > foundation_construction_time * 30
        boolean_dictionary = {True: foundation_construction_time * 30, False: np.nan}
        operation_data["time_construct_bool"] = operation_data["time_construct_bool"].map(boolean_dictionary)
        operation_data["Time construct days"] = operation_data[["time_construct_bool", "Number of days"]].min(axis=1)
        num_days = operation_data["Time construct days"].max()

        construction_time_output_data["operation_data_id_days_crews_workers"] = operation_data_id_days_crews_workers
        construction_time_output_data["operation_data_entire_farm"] = operation_data

        # pull out management data #TODO: Add this cost to Labor cost next
        if construction_time_input_data["turbine_rating_MW"] > 0.1:
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
        else:
            self.output_dict["managament_crew_cost_before_wind_delay"] = 0

        return construction_time_output_data["operation_data_entire_farm"]

    def calculate_weather_delay(self, weather_delay_input_data, weather_delay_output_data):
        """
        Function to calculate wind delay for foundations.

        Keys in weather_delay_input_data
        --------------------------------
        weather_window

        duration_construction

        start_delay

        critical_wind_speed

        operational_hrs_per_day

        height_interest

        wind_shear_exponent
        """

        # construct WeatherDelay module
        WD(weather_delay_input_data, weather_delay_output_data)

        # compute weather delay
        wind_delay = pd.DataFrame(weather_delay_output_data["wind_delays"])

        # if greater than 4 hour delay, then shut down for full day (10 hours)
        wind_delay[(wind_delay > 4)] = 10
        weather_delay_output_data["wind_delay_time"] = float(wind_delay.sum().iloc[0])

        return weather_delay_output_data

    def calculate_costs(self, calculate_costs_input_dict, calculate_costs_output_dict):
        """
        Function to calculate the total foundation cost.

        Keys in input dictionary
        ------------------------
        pd.DataFrame
            material_needs_per_turbine

        pd.DataFrame
            material_price

        pd.DataFrame
            operation_data

        wind_delay_time

        operational_hrs_per_day

        wind_multiplier

        pd.DataFrame
            rsmeans


        Returns
        -------

        (pd.DataFrame) total_foundation_cost


        """

        material_vol_entire_farm = calculate_costs_output_dict["material_needs_entire_farm"]
        material_price = calculate_costs_input_dict["material_price"]

        material_data_entire_farm = pd.merge(
            material_vol_entire_farm, material_price, on=["Material type ID"], sort=True
        )
        material_data_entire_farm["Cost USD"] = material_data_entire_farm["Quantity of material"] * pd.to_numeric(
            material_data_entire_farm["Material price USD per unit"]
        )  # material data on a total wind farm basis

        operation_data = calculate_costs_output_dict["operation_data_entire_farm"]

        wind_delay = calculate_costs_output_dict["wind_delay_time"]

        wind_delay_fraction = (wind_delay / calculate_costs_input_dict["operational_hrs_per_day"]) / operation_data[
            "Time construct days"
        ].max(skipna=True)
        # check if wind_delay_fraction is greater than 1, which would mean weather delays are longer than they can possibily be for the input data
        if wind_delay_fraction > 1:
            raise ValueError("{}: Error: Wind delay greater than 100%".format(type(self).__name__))
        wind_multiplier = 1 / (1 - wind_delay_fraction)
        calculate_costs_output_dict["wind_multiplier"] = wind_multiplier

        rsmeans = calculate_costs_input_dict["rsmeans"]
        if calculate_costs_input_dict["turbine_rating_MW"] > 0.1:
            rsmeans = rsmeans.where(rsmeans["Module"] == "Foundations").dropna(thresh=4)
        else:
            rsmeans = rsmeans.where(rsmeans["Module"] == "Small DW Foundations").dropna(thresh=4)

        labor_equip_data = pd.merge(material_vol_entire_farm, rsmeans, on=["Material type ID"], sort=True)

        # Create foundation cost dataframe
        foundation_cost_columns = ["Type of cost", "Cost USD", "Phase of construction"]

        # Calculate per diem
        per_diem = (
            operation_data["Number of workers"]
            * operation_data["Number of crews"]
            * (operation_data["Time construct days"] + np.ceil(operation_data["Time construct days"] / 7))
            * calculate_costs_input_dict["rsmeans_per_diem"]
        )
        where_are_na_ns = np.isnan(per_diem)
        per_diem[where_are_na_ns] = 0
        labor_equip_data["Cost USD"] = (
            labor_equip_data["Quantity of material"]
            * labor_equip_data["Rate USD per unit"]
            * calculate_costs_input_dict["overtime_multiplier"]
            + per_diem
            + calculate_costs_output_dict["managament_crew_cost_before_wind_delay"]
        ) * wind_multiplier
        self.output_dict["labor_equip_data"] = labor_equip_data

        # EQUIPMENT COST
        # Create equipment costs row to be appended to foundation_cost
        equipment_dataframe = labor_equip_data[labor_equip_data["Type of cost"].str.match("Equipment rental")]

        equipment_cost_usd_without_delay = (
            equipment_dataframe["Quantity of material"]
            * equipment_dataframe["Rate USD per unit"]
            * calculate_costs_input_dict["overtime_multiplier"]
            + per_diem
        )
        equipment_cost_usd_with_weather_delays = equipment_cost_usd_without_delay.sum() * wind_multiplier
        equipment_costs = pd.DataFrame(
            [["Equipment rental", equipment_cost_usd_with_weather_delays, "Foundation"]],
            columns=foundation_cost_columns,
        )

        # LABOR COST
        # Create labor costs row to be appended to foundation_cost
        labor_dataframe = labor_equip_data[labor_equip_data["Type of cost"].str.match("Labor")]
        labor_cost_usd_without_management = (
            labor_dataframe["Quantity of material"]
            * labor_dataframe["Rate USD per unit"]
            * calculate_costs_input_dict["overtime_multiplier"]
            + per_diem
        )
        labor_cost_usd_with_management = (
            labor_cost_usd_without_management.sum()
            + calculate_costs_output_dict["managament_crew_cost_before_wind_delay"]
        )
        labor_cost_usd_with_management_plus_weather_delays = labor_cost_usd_with_management * wind_multiplier
        labor_costs = pd.DataFrame(
            [["Labor", labor_cost_usd_with_management_plus_weather_delays, "Foundation"]],
            columns=foundation_cost_columns,
        )

        # MATERIAL COST
        material_cost_dataframe = pd.DataFrame(columns=["Operation ID", "Type of cost", "Cost USD"])
        material_cost_dataframe["Operation ID"] = material_data_entire_farm["Material type ID"]
        material_cost_dataframe["Type of cost"] = "Materials"
        material_cost_dataframe["Cost USD"] = material_data_entire_farm["Cost USD"]
        material_costs_sum = material_cost_dataframe["Cost USD"].sum()
        material_costs = pd.DataFrame(
            [["Materials", material_costs_sum, "Foundation"]], columns=foundation_cost_columns
        )

        # Append all cost items to foundation_cost
        foundation_cost = pd.concat((equipment_costs, labor_costs, material_costs))

        # Calculate mobilization cost as percentage of total foundation cost and add to foundation_cost
        # Assumed 5% of total foundation cost and add to foundation_cost for utility scale plant
        # A function of turbine size for distributed wind (< 10 turbines)
        if calculate_costs_input_dict["num_turbines"] > 10:
            mobilization_cost = foundation_cost["Cost USD"].sum() * 0.05
        else:
            if calculate_costs_input_dict["turbine_rating_MW"] < 0.1:
                # Zero since mobilization cost of equipment is included in the equipment rental cost
                mobilization_cost = 0
            else:
                # There is mobilization cost for 0-10 turbines 100+ kW in rating.
                num_turbines = calculate_costs_input_dict["num_turbines"]
                rating = calculate_costs_input_dict["turbine_rating_MW"]
                mobilization_multipler = self.mobilization_cost_multiplier(rating)
                mobilization_cost = foundation_cost["Cost USD"].sum() / num_turbines * mobilization_multipler

        mob_cost = pd.DataFrame(
            [["Mobilization", mobilization_cost, "Foundation"]],
            columns=["Type of cost", "Cost USD", "Phase of construction"],
        )

        foundation_cost = pd.concat((foundation_cost, mob_cost))

        # todo: we add a separate tab in the output file for costs (all costs will be the same format but it's a different format than other data)
        # columns in cost tab would include project_id, module, operation_id, type_of_cost, total_or_per_turbine, cost_usd
        # an example row for this type of output would be "project1, FoundationCost, Rebar installation, Labor, total, 2.371127e+06"
        # total_foundation_cost = foundation_cost.groupby(by=['Type of cost']).sum().reset_index()
        # total_foundation_cost['Phase of construction'] = 'Foundations'
        # total_cost_summed_foundation = total_foundation_cost.sum(numeric_only=True)[0] # todo: add total_cost_summed_foundation to output dict

        total_foundation_cost = foundation_cost
        calculate_costs_output_dict["total_foundation_cost"] = total_foundation_cost

        self.output_dict["labor_equip_data"] = labor_equip_data

        return total_foundation_cost

    def outputs_for_detailed_tab(self, input_dict, output_dict):
        """
        Creates a list of dictionaries which can be used on their own or
        used to make a dataframe.

        Must be called after self.run_module()

        Returns
        -------
        list(dict)
            A list of dicts, with each dict representing a row of the data.
        """

        # Note that some values are cast with float() so that XlsxWriter
        # (the library depended on by XlsxGenerator) can output them as
        # numbers. XlsxWriter, interestingly, cannot handle np.float32()
        # types.

        result = []
        module = type(self).__name__
        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "wind_multiplier",
                "value": float(self.output_dict["wind_multiplier"]),
            }
        )
        result.append(
            {
                "unit": "kN",
                "type": "variable",
                "variable_df_key_col_name": "F_dead",
                "value": float(self.output_dict["F_dead_kN_per_turbine"]),
            }
        )
        result.append(
            {
                "unit": "kN",
                "type": "variable",
                "variable_df_key_col_name": "F_horiz",
                "value": float(self.output_dict["F_horiz_kN_per_turbine"]),
            }
        )
        result.append(
            {
                "unit": "kN_m",
                "type": "variable",
                "variable_df_key_col_name": "M_tot_kN",
                "value": float(self.output_dict["M_tot_kN_m_per_turbine"]),
            }
        )
        result.append(
            {
                "unit": "m",
                "type": "variable",
                "variable_df_key_col_name": "Radius_o",
                "value": float(self.output_dict["Radius_o_m"]),
            }
        )
        result.append(
            {
                "unit": "m",
                "type": "variable",
                "variable_df_key_col_name": "Radius_g",
                "value": float(self.output_dict["Radius_g_m"]),
            }
        )
        result.append(
            {
                "unit": "m",
                "type": "variable",
                "variable_df_key_col_name": "Radius_b",
                "value": float(self.output_dict["Radius_b_m"]),
            }
        )
        result.append(
            {
                "unit": "m",
                "type": "variable",
                "variable_df_key_col_name": "Radius",
                "value": float(self.output_dict["Radius_m"]),
            }
        )
        result.append(
            {
                "unit": "short_ton",
                "type": "variable",
                "variable_df_key_col_name": "steel_mass_short_ton_per_turbine",
                "value": self.output_dict["steel_mass_short_ton_per_turbine"],
            }
        )
        # foundation_volume_concrete_m3_per_turbine
        result.append(
            {
                "unit": "m^3",
                "type": "variable",
                "variable_df_key_col_name": "foundation_volume_concrete_m3_per_turbine",
                "value": self.output_dict["foundation_volume_concrete_m3_per_turbine"],
            }
        )

        for row in self.output_dict["operation_data_id_days_crews_workers"].itertuples():
            dashed_row = "{}-{}-{}-{}".format(row[1], math.ceil(row[2]), row[3], row[4])
            result.append(
                {
                    "unit": "",
                    "type": "dataframe",
                    "variable_df_key_col_name": "operation_data: Operation ID-Number of days-Number of crews-Number of workers",
                    "value": dashed_row,
                }
            )

        for row in self.output_dict["material_needs_per_turbine"].itertuples():
            # This must be formatted in Python
            dashed_row = "{}-{}-{:.2e}".format(row[0], row[1], row[2])
            result.append(
                {
                    "unit": row[3],
                    "type": "dataframe",
                    "variable_df_key_col_name": "material_needs_per_turbine: {}".format(
                        "-".join(self.output_dict["material_needs_per_turbine"].columns[:-1])
                    ),
                    "value": dashed_row,
                }
            )

        for row in self.output_dict["total_foundation_cost"].itertuples():
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

        self.output_dict["foundation_cost_csv"] = result
        return result

    def outputs_for_module_type_operation(self, input_dict, output_dict):
        result = []
        module = type(self).__name__

        costs_by_module_type_operation = self.output_dict["labor_equip_data"]
        for _, row in costs_by_module_type_operation.iterrows():
            _dict = dict()
            row = row.to_dict()
            _dict["operation_id"] = row["Operation ID"]
            _dict["type_of_cost"] = row["Type of cost"]
            _dict["cost"] = row["Cost USD"]
            result.append(_dict)

        for _dict in result:
            _dict["project_id_with_serial"] = self.project_name
            _dict["module"] = module
            _dict["total_or_turbine"] = "total"

        self.output_dict["foundation_module_type_operation"] = result
        return result

    def run_module(self):
        """
        Runs the FoundationCost module and populates the IO dictionaries with calculated values.

        Parameters
        ----------
        <None>

        Returns
        -------
        tuple
            First element of tuple contains a 0 or 1. 0 means no errors happened and
            1 means an error happened and the module failed to run. The second element
            either returns a 0 if the module ran successfully, or it returns the error
            raised that caused the failure.

        """
        try:
            self.calculate_foundation_load(self.input_dict, self.output_dict)  # Returns foundation load
            self.determine_foundation_size(self.input_dict, self.output_dict)  # Returns foundation volume
            self.estimate_material_needs_per_turbine(self.input_dict, self.output_dict)  # Returns material volume
            operation_data = self.estimate_construction_time(
                self.input_dict, self.output_dict
            )  # Estimates construction time

            # pull only global inputs for weather delay from input_dict
            weather_data_keys = ("wind_shear_exponent", "weather_window")

            # specify foundation-specific weather delay inputs
            self.weather_input_dict = dict(
                [(i, self.input_dict[i]) for i in self.input_dict if i in set(weather_data_keys)]
            )
            self.weather_input_dict["start_delay_hours"] = (
                0  # assume zero start for when foundation construction begins (start at beginning of construction time)
            )
            self.weather_input_dict["critical_wind_speed_m_per_s"] = self.input_dict[
                "critical_speed_non_erection_wind_delays_m_per_s"
            ]
            self.weather_input_dict["wind_height_of_interest_m"] = self.input_dict[
                "critical_height_non_erection_wind_delays_m"
            ]

            # duration_construction is in units of days
            # duration_construction_months is in units of months
            days_per_month = 30
            duration_construction = operation_data["Time construct days"].max(skipna=True)
            duration_construction_months = duration_construction / days_per_month
            self.output_dict["foundation_construction_months"] = duration_construction_months

            # compute and specify weather delay mission time for roads
            operational_hrs_per_day = self.input_dict["hour_day"][self.input_dict["time_construct"]]
            mission_time_hrs = duration_construction * operational_hrs_per_day
            self.weather_input_dict["mission_time_hours"] = mission_time_hrs
            self.input_dict["operational_hrs_per_day"] = operational_hrs_per_day

            self.calculate_weather_delay(self.weather_input_dict, self.output_dict)
            self.calculate_costs(self.input_dict, self.output_dict)
            self.outputs_for_detailed_tab(self.input_dict, self.output_dict)
            # self.output_dict['labor_equip_data']
            # self.output_dict['foundation_module_type_operation'] = self.outputs_for_module_type_operation(self.input_dict, self.output_dict)
            self.output_dict["foundation_module_type_operation"] = self.outputs_for_costs_by_module_type_operation(
                input_df=self.output_dict["total_foundation_cost"], project_id=self.project_name, total_or_turbine=True
            )
            return 0, 0  # module ran successfully
        except Exception as error:
            traceback.print_exc()
            print(f"Fail {self.project_name} FoundationCost")
            return 1, error  # module did not run successfully
