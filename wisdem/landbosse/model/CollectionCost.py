"""
**CollectionCost.py**
- Created by Matt Shields for Offshore BOS
- Refactored by Parangat Bhaskar for LandBOSSE

NREL - 05/31/2019

This module consists of two classes:

- The first class in this module is the parent class Cable, with a sublass Array that inherits from Cable

- The second class is the ArraySystem class that instantiates the Array class and determines the wind farm layout and calculates total collection system cost
"""

import math
import traceback

import numpy as np
import pandas as pd
from wisdem.landbosse.model.CostModule import CostModule
from wisdem.landbosse.model.WeatherDelay import WeatherDelay as WD


class Cable:
    """

    Create an instance of Cable (either array or export)

        Parameters
        ---------
        cable_specs : dict
            Dictionary containing cable specifications
        line_frequency_hz : int
            Additional user inputs

        Returns
        -------
        current_capacity : float
            Cable current rating at 1m burial depth, Amps
        rated_voltage : float
            Cable rated voltage, kV
        ac_resistance : float
            Cable resistance for AC current, Ohms/km
        inductance : float
            Cable inductance, mH/km
        capacitance : float
            Cable capacitance, nF/km
        cost : int
            Cable cost, $US/km
        char_impedance : float
            Characteristic impedance of equivalent cable circuit, Ohms
        power_factor : float
            Power factor of AC current in cable (nondim)
        cable_power : float
            Maximum 3-phase power dissipated in cable, MW

    """

    def __init__(self, cable_specs, addl_specs):
        """
        Parameters
        ----------
        cable_specs : dict
            The input dictionary with key value pairs described in the
            class documentation

        addl_specs : dict
            The output dictionary with key value pairs as found on the
            output documentation.

        """

        self.current_capacity = cable_specs["Current Capacity (A)"]
        self.rated_voltage = cable_specs["Rated Voltage (V)"]
        self.ac_resistance = cable_specs["AC Resistance (Ohms/km)"]
        self.inductance = cable_specs["Inductance (mH/km)"]
        self.capacitance = cable_specs["Capacitance (nF/km)"]
        self.cost = cable_specs["Cost (USD/LF)"]
        self.line_frequency_hz = addl_specs["line_frequency_hz"]

        # Calc additional cable specs
        self.calc_char_impedance(self.line_frequency_hz)
        self.calc_power_factor()
        self.calc_cable_power()

    def calc_char_impedance(self, line_frequency_hz):
        """
        Calculate characteristic impedance of cable, Ohms

        Parameters
        ----------
        line_frequency_hz : int
            Frequency of AC current, Hz
        """
        conductance = 1 / self.ac_resistance

        num = complex(self.ac_resistance, 2 * math.pi * line_frequency_hz * self.inductance)
        den = complex(conductance, 2 * math.pi * line_frequency_hz * self.capacitance)
        self.char_impedance = np.sqrt(num / den)

    def calc_power_factor(self):
        """
        Calculate power factor
        """

        phase_angle = math.atan(np.imag(self.char_impedance) / np.real(self.char_impedance))
        self.power_factor = math.cos(phase_angle)

    def calc_cable_power(self):
        """
        Calculate maximum power transfer through 3-phase cable, MW
        """

        # TODO: Verify eqn is correct
        self.cable_power = np.sqrt(3) * self.rated_voltage * self.current_capacity * self.power_factor / 1000


class Array(Cable):
    """Array cable base class"""

    def __init__(self, cable_specs, addl_inputs):
        """
        Creates an instance of Array cable.
        (May be multiple instances of different capacity cables in a string)

        Parameters
        ----------
        cable_specs : dict
            Dictionary containing following cable specifications:

            - turbine_rating_MW

            - upstream_turb

            - turbine_spacing_rotor_diameters

            - rotor_diameter_m

        addl_inputs : dict

            - Any additional user inputs

        Returns
        -------
        self.max_turb_per_cable : float
            Maximum number of turbines (at turbine_rating_MW) an individual cable
            can support
        self.num_turb_per_cable : float
            Number of turbines each cable in a string actually supports.
        self.turb_sequence : float
            Ordering of cable in string, starting with smallest cable at 0
        self.downstream_connection : int
            Additional cable length requried to connect between different sized
            cables (for first cable in string only)
        self.array_cable_len : float
            Length of individual cable in a string, km
        """

        super().__init__(cable_specs, addl_inputs)
        self.line_frequency_hz = addl_inputs["line_frequency_hz"]
        self.calc_max_turb_per_cable(addl_inputs)
        self.calc_num_turb_per_cable(addl_inputs)
        self.calc_array_cable_len(addl_inputs)

    def calc_max_turb_per_cable(self, addl_inputs):
        """
        Calculate the number of turbines that each cable can support

        Parameters
        ----------
        turbine_rating_MW : int
            Nameplate capacity of individual turbines
        """

        turbine_rating_MW = addl_inputs["turbine_rating_MW"]

        self.max_turb_per_cable = np.floor(self.cable_power / turbine_rating_MW)

    def calc_num_turb_per_cable(self, addl_inputs):
        """
        Calculates actual number of turbines per cable, accounting for upstream
        turbines.

        Parameters
        ----------
        upstream_turb : int
            Number of turbines on upstream cables in string
        """

        upstream_turb = addl_inputs["upstream_turb"]
        self.turb_sequence = addl_inputs["turb_sequence"]

        self.num_turb_per_cable = self.max_turb_per_cable - upstream_turb

        if upstream_turb == 0:
            self.downstream_connection = -1
        else:
            self.downstream_connection = 0

    def calc_array_cable_len(self, addl_inputs):
        """
        Calculate array cable length per string, km

        Parameters
        ----------
        turbine_spacing_rotor_diameters : int
            Spacing between turbines in string, # of rotor diameters
        rotor_diameter_m : int or float
            Rotor diameter, m
        """

        turbine_spacing_rotor_diameters = addl_inputs["turbine_spacing_rotor_diameters"]
        rotor_diameter_m = addl_inputs["rotor_diameter_m"]

        self.calc_turb_section_len(turbine_spacing_rotor_diameters, rotor_diameter_m)

        self.array_cable_len = (self.num_turb_per_cable + self.downstream_connection) * self.turb_section_length

    #    @staticmethod
    def calc_turb_section_len(self, turbine_spacing_rotor_diameters, rotor_diameter_m):
        """
        Calculate array cable section length between two turbines. Also, section length == trench length. Which means
        trench_length = cable_length for that section.

        Parameters
        ----------
        turbine_spacing_rotor_diameters : int
            Spacing between turbines in string, # of rotor diameters
        rotor_diameter_m : int or float
            Rotor diameter, m

        Returns
        -------
        turb_connect_len : int
            Length of array cable between two turbines, km
        """

        self.turb_section_length = (turbine_spacing_rotor_diameters * rotor_diameter_m) / 1000

        return self.turb_section_length


class ArraySystem(CostModule):
    """


    \nThis module:

    * Calculates cable length to substation

    * Calculates number of strings in a subarray

    * Calculated number of strings

    * Calculates total cable length for each cable type

    * Calculates total trench length

    * Calculates total collection system cost based on amount of material, amount of labor, price data, cable length, and trench length.



    **Keys in the input dictionary are the following:**

    * Given below are attributes that define each cable type:
        * conductor_size
            (int)   cross-sectional diameter of cable [in mm]



    """

    def __init__(self, input_dict, output_dict, project_name):

        self.input_dict = input_dict
        self.output_dict = output_dict
        self.project_name = project_name
        self.output_dict["total_cable_len_km"] = 0
        self._km_to_LF = 0.0003048  # Units: [km/LF] Conversion factor for converting from km to linear foot.
        self._total_cable_cost = 0
        self._total_turbine_counter = 0
        self.turbines_on_cable = []
        self._cable_length_km = dict()
        self.check_terminal = 0

    def calc_num_strings(self):
        """
        Calculate number of full and partial strings to support full plant
        capacity.

        Parameters
        ----------
        available cables : dict
            Dictionary of cable types
        plant_capacity : int | float
            Total capcity of wind plant (MW)
        turbine_capacity : int | float
            Nameplate capacity of individual turbines (MW)

        Returns
        -------
        self.output_dict['total_turb_per_string'] : float
            Number of turbines on each string
        self.output_dict['num_full_strings'] : float
            Number of complete strings in array
        turb_per_partial_string : float
            Number of turbines in the partial string (if applicable)
        self.output_dict['num_partial_strings'] : float
            Number of partial strings (if applicable, 0 or 1)
        perc_full_string : list
            Percentage of maximum number of turbines per cable type on
            partial string
        self.output_dict['num_turb_per_cable'] : list
            Number of turbines on each cable type in string
        """

        # Calculate total number of individual turbines in wind plant
        self.output_dict["total_turb"] = self.input_dict["num_turbines"]

        # Calculate the number of turbines on each cable type in a string
        self.output_dict["num_turb_per_cable"] = [cable.num_turb_per_cable for cable in self.cables.values()]

        # Calculate the total number of turbines per string
        self.output_dict["total_turb_per_string"] = sum(self.output_dict["num_turb_per_cable"])

        # Calculate number of full strings and any remainder required to
        # support the total number of turbines
        self.output_dict["num_full_strings"] = np.floor(
            self.output_dict["total_turb"] / self.output_dict["total_turb_per_string"]
        )
        self.output_dict["num_leftover_turb"] = (
            self.output_dict["total_turb"] % self.output_dict["total_turb_per_string"]
        )

        # Calculate number of turbines on a remaining partial string

        # Note: self.output_dict['turb_per_partial_string'] is only set if
        # calc_num_turb_partial_strings()
        # is called, which isn't always the case, as seen in the if...else construct below
        #
        # This means that self.output_dict['turb_per_partial_string'] cannot
        # be used an output value for the details output.

        if self.output_dict["num_leftover_turb"] > 0:
            self.output_dict["num_partial_strings"] = 1
            self.output_dict["perc_partial_string"] = self.calc_num_turb_partial_strings(
                self.output_dict["num_leftover_turb"], self.output_dict["num_turb_per_cable"]
            )
        else:
            self.output_dict["num_partial_strings"] = 0
            self.output_dict["perc_partial_string"] = np.zeros(len(self.output_dict["num_turb_per_cable"]))

        return (
            self.output_dict["total_turb_per_string"],
            self.output_dict["num_full_strings"],
            self.output_dict["num_partial_strings"],
            self.output_dict["perc_partial_string"],
            self.output_dict["num_turb_per_cable"],
        )

    def calc_num_turb_partial_strings(self, num_leftover_turb, num_turb_per_cable):
        """
        If a partial string exists, calculate the percentage of turbines on
        each cable relative to a full string

        Parameters
        ----------
        self.output_dict['num_leftover_turb'] : float
            Number of turbines in partial string
        self.output_dict['num_turb_per_cable'] : list
            List of number of turbines per cable type on a full string

        Returns
        -------
        np.array
            Array of percent of turbines per cable type on partial string
            relative to full string
        """

        num_remaining = num_leftover_turb
        turb_per_partial_string = []

        # Loop through each cable type in the string. Determine how many
        # turbines are required for each cable type on the partial string
        for max_turb in num_turb_per_cable:
            if num_remaining > 0:
                turb_per_partial_string.append(min(num_remaining, max_turb))
            else:
                turb_per_partial_string.append(0.0)
            num_remaining -= max_turb

        perc_partial_string = np.divide(turb_per_partial_string, num_turb_per_cable)

        # Check to make sure there aren't any zeros in num_turbines_per_cable, which is used as the denominator
        # in the division above (this happens when not all of the cable types in the input sheet need to be used).
        # If there is a zero, then print a warning and change NaN to 0 in perc_partial_string.
        if 0.0 in num_turb_per_cable:
            print(
                f"Warning: {self.project_name} CollectionCost module generates number of turbines per string that "
                f"includes a zero entry. Please confirm that there not all cable types need to be used for the number of turbines that are being run."
                f' num_turbines={self.input_dict["num_turbines"]} rating_MW={self.input_dict["turbine_rating_MW"]}'
                f" num_turb_per_cable: {num_turb_per_cable}"
            )
            perc_partial_string = np.nan_to_num(perc_partial_string)

        self.output_dict["turb_per_partial_string"] = turb_per_partial_string

        return perc_partial_string

    # TODO: change length_to_substation calculation as a user defined input?
    @staticmethod
    def calc_cable_len_to_substation(
        distance_to_grid, turbine_spacing_rotor_diameters, row_spacing_rotor_diameters, num_strings
    ):
        """
        Calculate the distance for the largest cable run to substation
        Assumes substation is in the center of the layout, 1 row spacing in
        front of first row

        Parameters
        ----------
        turbine_spacing_rotor_diameters : int or float
            Spacing between turbines in a row, # of rotor diameters
        row_spacing_rotor_diameters : int or float
            Spacing between rows in wind plant, # of rotor diameters
        num_strings : int
            Total number of strings

        Returns
        -------
        len_to_substation : int or float
            Total length of largest array cable required to connect each string
            to substation, km
        """

        string_to_substation_length = []

        if num_strings > 1:
            # Define spacing terms for even or odd number of strings
            #   Even number: substation centered between middle two strings
            #   Odd number : substation centered on middle string
            if (num_strings % 2) == 0:
                n_max = int(num_strings / 2)
                turb_space_scaling = 0.5
                range_strings = range(1, n_max + 1)

            else:
                n_max = int((num_strings - 1) / 2)
                turb_space_scaling = 1
                range_strings = range(n_max + 1)

            # Calculate hypotenuse length of each string to substation
            for idx in range_strings:
                if idx == 0:
                    c = 1
                else:
                    c = 2
                string_to_substation_length.append(
                    c
                    * np.sqrt(
                        row_spacing_rotor_diameters ** 2
                        + (turb_space_scaling * idx * turbine_spacing_rotor_diameters) ** 2
                    )
                )

        else:
            string_to_substation_length.append(distance_to_grid)

        # Sum up total length to substation
        len_to_substation = np.sum(string_to_substation_length)

        return len_to_substation

    # TODO: Add parameter info in docstrings
    @staticmethod
    def calc_total_cable_length(
        total_turbines,
        count,
        check_terminal,
        turbines_per_cable,
        cable,
        cable_specs,
        num_full_strings,
        num_partial_strings,
        len_to_substation,
        perc_partial_string,
    ):
        """
        Calculate total length of each cable type, km

        Parameters
        ----------
        cable : object
            Instance of individual cable type
        cable_specs : dict
            Dictionary containing cable specifications
        self.output_dict['num_full_strings'] : float
            Number of complete strings in array
        self.output_dict['num_partial_strings'] : float
            Number of partial strings (if applicable, 0 or 1)
        len_to_substation : int or float
            Total length of largest array cable required to connect each string
            to substation, km
        self.output_dict['perc_partial_string'] : list
            List of percent of turbines per cable type on partial string
            relative to full string

        Returns
        -------
        total_cable_len : int or float
            Total length of individual cable type
        """

        # If terminal cable has already been accounted for, skip any
        # calculations for other cables.
        if (cable.turb_sequence - 1) > check_terminal:
            cable.array_cable_len = 0
            cable.total_length = 0
            cable.num_turb_per_cable = 0
            return 0, 0

        # If num full strings < = 1, find which cable the final turbine
        # is on, and calculate total cable length (including the len to
        # substation) using that cable.

        # This 'elif' is essentially a switch for distributed wind:
        elif num_full_strings < 1 and num_partial_strings >= 0:

            # if number of turbines is less than total string capacity,
            # find the terminal cable and find total cable len
            # up till that cable.

            # If total turbines in project are less than cumulative turbines
            # up till and including that cable.

            terminal_string = cable.turb_sequence - 1  # Flag this cable as it is
            # also the actual terminal cable

            if (cable.turb_sequence - 1) == 0:  # That is, if cable # 1 can hold
                # more turbines than specified by user, it is the terminal cable

                cable.num_turb_per_cable = total_turbines
                cable.array_cable_len = (
                    cable.num_turb_per_cable + cable.downstream_connection
                ) * cable.turb_section_length

                total_cable_len = (
                    (num_full_strings * cable.array_cable_len) + (num_partial_strings * cable.array_cable_len)
                ) + len_to_substation

            else:

                cable.num_turb_per_cable = total_turbines - turbines_per_cable[(count - 1)]
                cable.array_cable_len = (
                    cable.num_turb_per_cable + cable.downstream_connection
                ) * cable.turb_section_length

                total_cable_len = (
                    (num_full_strings * cable.array_cable_len) + (num_partial_strings * cable.array_cable_len)
                ) + len_to_substation

            return total_cable_len, terminal_string

        else:  # Switch for utility scale landbosse
            if cable.turb_sequence == len(cable_specs):

                # Only add len_to_substation to the final cable in the string
                total_cable_len = (
                    num_full_strings * cable.array_cable_len
                    + num_partial_strings * (cable.array_cable_len * perc_partial_string)
                    + len_to_substation
                )
            else:
                total_cable_len = num_full_strings * cable.array_cable_len + num_partial_strings * (
                    cable.array_cable_len * perc_partial_string
                )

        # here 9999 == flag to announce that the terminal cable has NOT been reached
        # and to continue calculations for each cable
        return total_cable_len, 9999

    def create_ArraySystem(self):
        # data used in parent classes:
        self.addl_specs = dict()
        self.addl_specs["turbine_rating_MW"] = self.input_dict["turbine_rating_MW"]
        self.addl_specs["upstream_turb"] = 0
        self.addl_specs["turb_sequence"] = 1
        self.addl_specs["turbine_spacing_rotor_diameters"] = self.input_dict["turbine_spacing_rotor_diameters"]
        self.addl_specs["rotor_diameter_m"] = self.input_dict["rotor_diameter_m"]
        self.addl_specs["line_frequency_hz"] = self.input_dict["line_frequency_hz"]

        system = {
            "upstream_turb": self.addl_specs["upstream_turb"],
            "turb_sequence": self.addl_specs["turb_sequence"],
            "turbine_rating_MW": self.addl_specs["turbine_rating_MW"],
            "turbine_spacing_rotor_diameters": self.addl_specs["turbine_spacing_rotor_diameters"],
            "rotor_diameter_m": self.addl_specs["rotor_diameter_m"],
        }

        # Loops through all user defined array cable types, composing them
        # in ArraySystem

        self.cables = {}
        self.input_dict["cable_specs"] = self.input_dict["cable_specs_pd"].T.to_dict()
        n = 0  # to keep tab of number of cables input by user.
        while n < len(self.input_dict["cable_specs"]):
            specs = self.input_dict["cable_specs"][n]
            # Create instance of each cable and assign to ArraySystem.cables
            cable = Array(specs, self.addl_specs)
            n += 1

            # self.cables[name] stores value which is a new instantiation of object of type Array.
            self.cables[specs["Array Cable"]] = cable
            self.output_dict["cables"] = self.cables

            # Update number of upstream cables on the string
            self.addl_specs["upstream_turb"] += cable.num_turb_per_cable
            self.addl_specs["turb_sequence"] += 1

        # Calculate number of required strings to support plant capacity
        (
            self.output_dict["turb_per_string"],
            self.output_dict["num_full_strings"],
            self.output_dict["num_partial_strings"],
            self.output_dict["perc_partial_string"],
            self.output_dict["num_turb_per_cable"],
        ) = self.calc_num_strings()

        # Calculate total length of cable run to substation
        self.output_dict["num_strings"] = self.output_dict["num_full_strings"] + self.output_dict["num_partial_strings"]

        if self.input_dict["user_defined_distance_to_grid_connection"] == 0:  # where (0 = No) and (1 = Yes)

            # This only gets used if number of strings is <= 1 :
            distributed_wind_distance_to_grid = (
                self.input_dict["turbine_spacing_rotor_diameters"] * self.input_dict["rotor_diameter_m"]
            ) / 1000
            self.output_dict["distance_to_grid_connection_km"] = self.calc_cable_len_to_substation(
                distributed_wind_distance_to_grid,
                self.input_dict["turbine_spacing_rotor_diameters"],
                self.input_dict["row_spacing_rotor_diameters"],
                self.output_dict["num_strings"],
            )
        else:
            self.output_dict["distance_to_grid_connection_km"] = self.input_dict["distance_to_grid_connection_km"]

        self.output_dict["cable_len_to_grid_connection_km"] = self.output_dict[
            "distance_to_grid_connection_km"
        ]  # assumes 3 conductors and fiber and neutral

        cable_sequence = 0
        # Make a list of how many turbines per cable
        for _, (name, cable) in enumerate(self.cables.items()):
            if cable_sequence == 0:
                self.turbines_on_cable.append(cable.num_turb_per_cable)
            else:
                self.turbines_on_cable.append(cable.num_turb_per_cable + self.turbines_on_cable[(cable_sequence - 1)])
            # turbines_on_cable[cable_sequence] += cable.num_turb_per_cable
            cable_sequence += 1
        self.__turbines_on_cable = self.turbines_on_cable

        # Calculate total length of each cable type, and total cost that calculated length of cable:
        count = 0
        for idx, (name, cable) in enumerate(self.cables.items()):
            total_cable_len, self.check_terminal = self.calc_total_cable_length(
                self.output_dict["total_turb"],
                count,
                self.check_terminal,
                self.__turbines_on_cable,
                cable,
                self.input_dict["cable_specs"],
                self.output_dict["num_full_strings"],
                self.output_dict["num_partial_strings"],
                self.output_dict["distance_to_grid_connection_km"],
                self.output_dict["perc_partial_string"][idx],
            )
            count += 1
            # self._total_turbine_counter = turbine_tally
            self._cable_length_km[name] = total_cable_len
            cable.total_length = total_cable_len
            self.output_dict["total_cable_len_km"] += total_cable_len
            cable.total_cost = (total_cable_len / self._km_to_LF) * cable.cost
            self._total_cable_cost += cable.total_cost  # Keep running tally of total cable cost used in wind farm.

        # Repopulate the turbines per cable sequence to make sure it reflects any changes that happened since
        # the first time this sequence was populated.
        self.output_dict["num_turb_per_cable"] = [cable.num_turb_per_cable for cable in self.cables.values()]
        self.output_dict["total_turb_per_string"] = sum(self.output_dict["num_turb_per_cable"])

    def calculate_trench_properties(self, trench_properties_input, trench_properties_output):
        """
        Calculates the length of trench needed based on cable length and width of mulcher.
        """

        # units of cubic meters
        trench_properties_output["trench_length_km"] = trench_properties_output["total_cable_len_km"]

    def calculate_weather_delay(self, weather_delay_input_data, weather_delay_output_data):
        """Calculates wind delays for roads"""
        # construct WeatherDelay module
        WD(weather_delay_input_data, weather_delay_output_data)

        # compute weather delay
        wind_delay = pd.DataFrame(weather_delay_output_data["wind_delays"])

        # if greater than 4 hour delay, then shut down for full day (10 hours)
        wind_delay[(wind_delay > 4)] = 10
        weather_delay_output_data["wind_delay_time"] = float(wind_delay.sum())

        return weather_delay_output_data

    def estimate_construction_time(self, construction_time_input_data, construction_time_output_data):
        """
        Function to estimate construction time on per turbine basis.

        Parameters
        -------
        duration_construction

        pd.DataFrame
            rsmeans

        pd.DataFrame
            trench_length_km



        Returns
        -------

        (pd.DataFrame) operation_data

        """

        collection_construction_time = (
            construction_time_input_data["construct_duration"] * 1 / 3
        )  # assumes collection construction occurs for one-third of project duration

        throughput_operations = construction_time_input_data["rsmeans"]
        trench_length_km = construction_time_output_data["trench_length_km"]
        if construction_time_input_data["turbine_rating_MW"] >= 0.1:
            operation_data = throughput_operations.where(throughput_operations["Module"] == "Collection").dropna(
                thresh=4
            )
            # from rsmeans data, only read in Collection related data and filter out the rest:
            cable_trenching = throughput_operations[throughput_operations.Module == "Collection"]
        else:  # switch for small DW
            operation_data = throughput_operations.where(
                throughput_operations["Module"] == "Small DW Collection"
            ).dropna(thresh=4)
            # from rsmeans data, only read in Collection related data and filter out the rest:
            cable_trenching = throughput_operations[throughput_operations.Module == "Small DW Collection"]
        # operation_data = pd.merge()

        # from rsmeans data, only read in Collection related data and filter out the rest:
        cable_trenching = throughput_operations[throughput_operations.Module == "Collection"]

        # Storing data with labor related inputs:
        trenching_labor = cable_trenching[cable_trenching.values == "Labor"]
        trenching_labor_usd_per_hr = trenching_labor["Rate USD per unit"].sum()

        construction_time_output_data["trenching_labor_usd_per_hr"] = trenching_labor_usd_per_hr
        trenching_labor_daily_output = trenching_labor["Daily output"].values[
            0
        ]  # Units:  LF/day  -> where LF = Linear Foot
        trenching_labor_num_workers = trenching_labor["Number of workers"].sum()

        # Storing data with equipment related inputs:
        trenching_equipment = cable_trenching[cable_trenching.values == "Equipment"]
        trenching_cable_equipment_usd_per_hr = trenching_equipment["Rate USD per unit"].sum()
        construction_time_output_data["trenching_cable_equipment_usd_per_hr"] = trenching_cable_equipment_usd_per_hr
        trenching_equipment_daily_output = trenching_equipment["Daily output"].values[
            0
        ]  # Units:  LF/day  -> where LF = Linear Foot
        construction_time_output_data["trenching_labor_daily_output"] = trenching_labor_daily_output
        construction_time_output_data["trenching_equipment_daily_output"] = trenching_equipment_daily_output

        operation_data["Number of days taken by single crew"] = (
            trench_length_km / self._km_to_LF
        ) / trenching_labor_daily_output
        operation_data["Number of crews"] = np.ceil(
            (operation_data["Number of days taken by single crew"] / 30) / collection_construction_time
        )
        operation_data["Cost USD without weather delays"] = (
            (trench_length_km / self._km_to_LF) / trenching_labor_daily_output
        ) * (operation_data["Rate USD per unit"] * construction_time_input_data["operational_hrs_per_day"])
        alpha = operation_data[operation_data["Type of cost"] == "Collection"]
        operation_data_id_days_crews_workers = alpha[
            ["Operation ID", "Number of days taken by single crew", "Number of crews", "Number of workers"]
        ]

        alpha = operation_data[operation_data["Type of cost"] == "Labor"]
        operation_data_id_days_crews_workers = alpha[
            ["Operation ID", "Number of days taken by single crew", "Number of crews", "Number of workers"]
        ]

        # if more than one crew needed to complete within construction duration then assume that all construction
        # happens within that window and use that timeframe for weather delays;
        # if not, use the number of days calculated
        operation_data["time_construct_bool"] = (
            operation_data["Number of days taken by single crew"] > collection_construction_time * 30
        )
        boolean_dictionary = {True: collection_construction_time * 30, False: np.NAN}
        operation_data["time_construct_bool"] = operation_data["time_construct_bool"].map(boolean_dictionary)
        operation_data["Time construct days"] = operation_data[
            ["time_construct_bool", "Number of days taken by single crew"]
        ].min(axis=1)
        num_days = operation_data["Time construct days"].max()

        # No 'management crew' in small DW
        if construction_time_input_data["turbine_rating_MW"] >= 0.1:
            # pull out management data
            crew_cost = self.input_dict["crew_cost"]
            crew = self.input_dict["crew"][self.input_dict["crew"]["Crew type ID"].str.contains("M0")]
            management_crew = pd.merge(crew_cost, crew, on=["Labor type ID"])
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
            self.output_dict["managament_crew_cost_before_wind_delay"] = 0.0

        construction_time_output_data["operation_data_id_days_crews_workers"] = operation_data_id_days_crews_workers
        construction_time_output_data["operation_data_entire_farm"] = operation_data

        return construction_time_output_data["operation_data_entire_farm"]

    def calculate_costs(self, calculate_costs_input_dict, calculate_costs_output_dict):

        # read in rsmeans data:
        # rsmeans = calculate_costs_input_dict['rsmeans']
        operation_data = calculate_costs_output_dict["operation_data_entire_farm"]

        per_diem = (
            operation_data["Number of workers"]
            * operation_data["Number of crews"]
            * (operation_data["Time construct days"] + np.ceil(operation_data["Time construct days"] / 7))
            * calculate_costs_input_dict["rsmeans_per_diem"]
        )
        per_diem = per_diem.dropna()

        calculate_costs_output_dict["time_construct_days"] = (
            calculate_costs_output_dict["trench_length_km"] / self._km_to_LF
        ) / calculate_costs_output_dict["trenching_labor_daily_output"]
        wind_delay_fraction = (
            calculate_costs_output_dict["wind_delay_time"] / calculate_costs_input_dict["operational_hrs_per_day"]
        ) / calculate_costs_output_dict["time_construct_days"]
        # check if wind_delay_fraction is greater than 1, which would mean weather delays are longer than they can possibily be for the input data
        if wind_delay_fraction > 1:
            raise ValueError("{}: Error: Wind delay greater than 100%".format(type(self).__name__))
        calculate_costs_output_dict["wind_multiplier"] = 1 / (1 - wind_delay_fraction)

        # Calculating trenching cost:
        calculate_costs_output_dict["Days taken for trenching (equipment)"] = (
            calculate_costs_output_dict["trench_length_km"] / self._km_to_LF
        ) / calculate_costs_output_dict["trenching_equipment_daily_output"]
        calculate_costs_output_dict["Equipment cost of trenching per day {usd/day)"] = (
            calculate_costs_output_dict["trenching_cable_equipment_usd_per_hr"]
            * calculate_costs_input_dict["operational_hrs_per_day"]
        )
        calculate_costs_output_dict["Equipment Cost USD without weather delays"] = (
            calculate_costs_output_dict["Days taken for trenching (equipment)"]
            * calculate_costs_output_dict["Equipment cost of trenching per day {usd/day)"]
        )
        calculate_costs_output_dict["Equipment Cost USD with weather delays"] = (
            calculate_costs_output_dict["Equipment Cost USD without weather delays"]
            * calculate_costs_output_dict["wind_multiplier"]
        )

        if calculate_costs_input_dict["turbine_rating_MW"] >= 0.1:
            trenching_equipment_rental_cost_df = pd.DataFrame(
                [
                    [
                        "Equipment rental",
                        calculate_costs_output_dict["Equipment Cost USD with weather delays"],
                        "Collection",
                    ]
                ],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        # switch for small DW
        else:
            if calculate_costs_output_dict["Equipment Cost USD with weather delays"] < 137:
                calculate_costs_output_dict["Equipment Cost USD with weather delays"] = 137  # cost of renting for a day
                trenching_equipment_rental_cost_df = pd.DataFrame(
                    [
                        [
                            "Equipment rental",
                            calculate_costs_output_dict["Equipment Cost USD with weather delays"],
                            "Collection",
                        ]
                    ],
                    columns=["Type of cost", "Cost USD", "Phase of construction"],
                )
            else:
                trenching_equipment_rental_cost_df = pd.DataFrame(
                    [
                        [
                            "Equipment rental",
                            calculate_costs_output_dict["Equipment Cost USD with weather delays"],
                            "Small DW Collection",
                        ]
                    ],
                    columns=["Type of cost", "Cost USD", "Phase of construction"],
                )

        # Calculating labor cost:
        calculate_costs_output_dict["Days taken for trenching (labor)"] = (
            calculate_costs_output_dict["trench_length_km"] / self._km_to_LF
        ) / calculate_costs_output_dict["trenching_labor_daily_output"]
        calculate_costs_output_dict["Labor cost of trenching per day (usd/day)"] = (
            calculate_costs_output_dict["trenching_labor_usd_per_hr"]
            * calculate_costs_input_dict["operational_hrs_per_day"]
            * calculate_costs_input_dict["overtime_multiplier"]
        )
        calculate_costs_output_dict["Total per diem costs (USD)"] = per_diem.sum()
        calculate_costs_output_dict["Labor Cost USD without weather delays"] = (
            calculate_costs_output_dict["Days taken for trenching (labor)"]
            * calculate_costs_output_dict["Labor cost of trenching per day (usd/day)"]
        ) + (
            calculate_costs_output_dict["Total per diem costs (USD)"]
            + calculate_costs_output_dict["managament_crew_cost_before_wind_delay"]
        )
        calculate_costs_output_dict["Labor Cost USD with weather delays"] = (
            calculate_costs_output_dict["Labor Cost USD without weather delays"]
            * calculate_costs_output_dict["wind_multiplier"]
        )

        if calculate_costs_input_dict["turbine_rating_MW"] >= 0.1:
            trenching_labor_cost_df = pd.DataFrame(
                [["Labor", calculate_costs_output_dict["Labor Cost USD with weather delays"], "Collection"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        # switch for small DW
        else:
            trenching_labor_cost_df = pd.DataFrame(
                [["Labor", calculate_costs_output_dict["Labor Cost USD with weather delays"], "Small DW Collection"]],
                columns=["Type of cost", "Cost USD", "Phase of construction"],
            )

        # Calculate cable cost:
        cable_cost_usd_per_LF_df = pd.DataFrame(
            [["Materials", self._total_cable_cost, "Collection"]],
            columns=["Type of cost", "Cost USD", "Phase of construction"],
        )

        # Combine all calculated cost items into the 'collection_cost' dataframe:
        collection_cost = pd.DataFrame([], columns=["Type of cost", "Cost USD", "Phase of construction"])
        collection_cost = pd.concat(
            (collection_cost, trenching_equipment_rental_cost_df, trenching_labor_cost_df, cable_cost_usd_per_LF_df)
        )

        # Calculate Mobilization Cost and add to collection_cost dataframe.
        # For utility scale plants, mobilization is assumed to be 5% of the sum of labor, equipment, and material costs.
        # For distributed mode, mobilization is a calculated % that is a function of turbine size.
        if calculate_costs_input_dict["num_turbines"] > 10:
            calculate_costs_output_dict["mob_cost"] = collection_cost["Cost USD"].sum() * 0.05
        else:
            if calculate_costs_input_dict["turbine_rating_MW"] >= 0.1:
                calculate_costs_output_dict["mob_cost"] = collection_cost[
                    "Cost USD"
                ].sum() * self.mobilization_cost_multiplier(calculate_costs_input_dict["turbine_rating_MW"])

            # switch for small DW
            else:  # mobilization cost included in equipment rental cost
                calculate_costs_output_dict["mob_cost"] = 0.0

        mobilization_cost = pd.DataFrame(
            [["Mobilization", calculate_costs_output_dict["mob_cost"], "Collection"]],
            columns=["Type of cost", "Cost USD", "Phase of construction"],
        )
        collection_cost = pd.concat((collection_cost, mobilization_cost))

        calculate_costs_output_dict["total_collection_cost"] = collection_cost

        return collection_cost

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
        module = "Collection Cost"
        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "Total Number of Turbines",
                "value": float(self.output_dict["total_turb"]),
            }
        )

        result.append(
            {
                "unit": "km",
                "type": "variable",
                "variable_df_key_col_name": "Total trench length",
                "value": float(self.output_dict["trench_length_km"]),
            }
        )

        result.append(
            {
                "unit": "km",
                "type": "variable",
                "variable_df_key_col_name": "Total cable length",
                "value": float(self.output_dict["total_cable_len_km"]),
            }
        )

        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "Number of Turbines Per String in Full String",
                "value": float(self.output_dict["total_turb_per_string"]),
            }
        )
        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "Number of Full Strings",
                "value": float(self.output_dict["num_full_strings"]),
            }
        )
        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "Number of Turbines in Partial String",
                "value": float(self.output_dict["num_leftover_turb"]),
            }
        )
        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "Number of Partial Strings",
                "value": float(self.output_dict["num_partial_strings"]),
            }
        )
        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "Total number of strings full + partial",
                "value": float(self.output_dict["num_full_strings"] + self.output_dict["num_partial_strings"]),
            }
        )
        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "Trench Length to Substation (km)",
                "value": float(self.output_dict["distance_to_grid_connection_km"]),
            }
        )
        result.append(
            {
                "unit": "",
                "type": "variable",
                "variable_df_key_col_name": "Cable Length to Substation (km)",
                "value": float(self.output_dict["cable_len_to_grid_connection_km"]),
            }
        )

        cables = ""
        n = 1  # to keep tab of number of cables input by user.
        for cable, specs in self.output_dict["cables"].items():
            if n == len(self.output_dict["cables"]):
                cables += str(cable)
            else:
                cables += str(cable) + "  ,  "

            for variable, value in specs.__dict__.items():
                if variable == "array_cable_len":
                    result.append(
                        {
                            "unit": "km",
                            "type": "variable",
                            "variable_df_key_col_name": "Array cable length for cable  " + cable,
                            "value": float(value),
                        }
                    )
                elif variable == "total_length":
                    result.append(
                        {
                            "unit": "km",
                            "type": "variable",
                            "variable_df_key_col_name": "Total cable length for cable  " + cable,
                            "value": float(value),
                        }
                    )

                elif variable == "total_cost":
                    result.append(
                        {
                            "unit": "usd",
                            "type": "variable",
                            "variable_df_key_col_name": "Total cable cost for cable  " + cable,
                            "value": float(value),
                        }
                    )
            n += 1

        result.append(
            {
                "unit": "",
                "type": "list",
                "variable_df_key_col_name": "Number of turbines per cable type in full strings [" + cables + "]",
                "value": str(self.output_dict["num_turb_per_cable"]),
            }
        )

        if self.input_dict["turbine_rating_MW"] > 0.1:
            for row in self.output_dict["management_crew"].itertuples():
                dashed_row = " <--> ".join(str(x) for x in list(row))
                result.append(
                    {
                        "unit": "",
                        "type": "dataframe",
                        "variable_df_key_col_name": "Labor type ID <--> Hourly rate USD per hour <--> Per diem USD per day <--> Operation <--> Crew type <--> Crew name <--> Number of workers <--> Per Diem Total <--> Hourly costs total <--> Crew total cost ",
                        "value": dashed_row,
                    }
                )

        result.append(
            {
                "unit": "",
                "type": "list",
                "variable_df_key_col_name": "Percent length of cable in partial string [" + cables + "]",
                "value": str(self.output_dict["perc_partial_string"]),
            }
        )

        for row in self.output_dict["total_collection_cost"].itertuples():
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

        self.output_dict["collection_cost_csv"] = result
        return result

    def run_module(self):
        """
        Runs the CollectionCost module and populates the IO dictionaries with calculated values.

        """

        try:
            self.create_ArraySystem()
            self.calculate_trench_properties(self.input_dict, self.output_dict)
            operation_data = self.estimate_construction_time(self.input_dict, self.output_dict)

            # pull only global inputs for weather delay from input_dict
            weather_data_keys = ("wind_shear_exponent", "weather_window")

            # specify collection-specific weather delay inputs
            self.weather_input_dict = dict(
                [(i, self.input_dict[i]) for i in self.input_dict if i in set(weather_data_keys)]
            )
            self.weather_input_dict[
                "start_delay_hours"
            ] = 0  # assume zero start for when collection construction begins (start at beginning of construction time)
            self.weather_input_dict["critical_wind_speed_m_per_s"] = self.input_dict[
                "critical_speed_non_erection_wind_delays_m_per_s"
            ]
            self.weather_input_dict["wind_height_of_interest_m"] = self.input_dict[
                "critical_height_non_erection_wind_delays_m"
            ]

            # Compute the duration of the construction for electrical collection
            duration_construction = operation_data["Time construct days"].max(skipna=True)
            days_per_month = 30
            duration_construction_months = duration_construction / days_per_month
            self.output_dict["collection_construction_months"] = duration_construction_months

            # compute and specify weather delay mission time for roads
            operational_hrs_per_day = self.input_dict["hour_day"][self.input_dict["time_construct"]]
            mission_time_hrs = duration_construction * operational_hrs_per_day
            self.weather_input_dict["mission_time_hours"] = int(mission_time_hrs)

            self.calculate_weather_delay(self.weather_input_dict, self.output_dict)
            self.calculate_costs(self.input_dict, self.output_dict)
            self.outputs_for_detailed_tab(self.input_dict, self.output_dict)
            self.output_dict["collection_cost_module_type_operation"] = self.outputs_for_costs_by_module_type_operation(
                input_df=self.output_dict["total_collection_cost"], project_id=self.project_name, total_or_turbine=True
            )
            return 0, 0  # module ran successfully
        except Exception as error:
            traceback.print_exc()
            print(f"Fail {self.project_name} CollectionCost")
            return 1, error  # module did not run successfully
