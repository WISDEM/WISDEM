import warnings
from math import ceil

import numpy as np
import openmdao.api as om

from wisdem.landbosse.model.Manager import Manager
from wisdem.landbosse.model.DefaultMasterInputDict import DefaultMasterInputDict
from wisdem.landbosse.landbosse_omdao.OpenMDAODataframeCache import OpenMDAODataframeCache
from wisdem.landbosse.landbosse_omdao.WeatherWindowCSVReader import read_weather_window

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    import pandas as pd


use_default_component_data = -1.0


class LandBOSSE(om.Group):
    def setup(self):
        # Add a tower section height variable. The default value of 30 m is for transportable tower sections.
        self.set_input_defaults("tower_section_length_m", 30.0, units="m")
        self.set_input_defaults("blade_drag_coefficient", use_default_component_data)  # Unitless
        self.set_input_defaults("blade_lever_arm", use_default_component_data, units="m")
        self.set_input_defaults("blade_install_cycle_time", use_default_component_data, units="h")
        self.set_input_defaults("blade_offload_hook_height", use_default_component_data, units="m")
        self.set_input_defaults("blade_offload_cycle_time", use_default_component_data, units="h")
        self.set_input_defaults("blade_drag_multiplier", use_default_component_data)  # Unitless
        self.set_input_defaults("blade_surface_area", use_default_component_data, units="m**2")

        self.set_input_defaults("turbine_spacing_rotor_diameters", 4)
        self.set_input_defaults("row_spacing_rotor_diameters", 10)
        self.set_input_defaults("commissioning_pct", 0.01)
        self.set_input_defaults("decommissioning_pct", 0.15)
        self.set_input_defaults("trench_len_to_substation_km", 50.0, units="km")
        self.set_input_defaults("interconnect_voltage_kV", 130.0, units="kV")

        self.set_input_defaults("foundation_height", 0.0, units="m")
        self.set_input_defaults("blade_mass", 8000.0, units="kg")
        self.set_input_defaults("hub_mass", 15.4e3, units="kg")
        self.set_input_defaults("nacelle_mass", 50e3, units="kg")
        self.set_input_defaults("tower_mass", 240e3, units="kg")
        self.set_input_defaults("turbine_rating_MW", 1500.0, units="kW")

        self.add_subsystem("landbosse", LandBOSSE_API(), promotes=["*"])


class LandBOSSE_API(om.ExplicitComponent):
    def setup(self):
        # Clear the cache
        OpenMDAODataframeCache._cache = {}

        self.setup_inputs()
        self.setup_outputs()
        self.setup_discrete_outputs()
        self.setup_discrete_inputs_that_are_not_dataframes()
        self.setup_discrete_inputs_that_are_dataframes()

    def setup_inputs(self):
        """
        This method sets up the inputs.
        """
        self.add_input("blade_drag_coefficient", use_default_component_data)  # Unitless
        self.add_input("blade_lever_arm", use_default_component_data, units="m")
        self.add_input("blade_install_cycle_time", use_default_component_data, units="h")
        self.add_input("blade_offload_hook_height", use_default_component_data, units="m")
        self.add_input("blade_offload_cycle_time", use_default_component_data, units="h")
        self.add_input("blade_drag_multiplier", use_default_component_data)  # Unitless
        self.add_input("blade_surface_area", use_default_component_data, units="m**2")

        # Even though LandBOSSE doesn't use foundation height, TowerSE does,
        # and foundation height can be used with hub height to calculate
        # tower height.

        self.add_input("foundation_height", 0.0, units="m")

        self.add_input("tower_section_length_m", 30.0, units="m")
        self.add_input("nacelle_mass", 0.0, units="kg")
        self.add_input("tower_mass", 0.0, units="kg")

        # A discrete input below, number_of_blades, gives the number of blades
        # on the rotor.
        #
        # The total mass of the rotor nacelle assembly (RNA) is the following
        # sum:
        #
        # (blade_mass * number_of_blades) + nac_mass + hub_mass

        self.add_input("blade_mass", use_default_component_data, units="kg", desc="The mass of one rotor blade.")

        self.add_input("hub_mass", use_default_component_data, units="kg", desc="Mass of the rotor hub")

        self.add_input(
            "crane_breakdown_fraction",
            val=0.0,
            desc="0 means the crane is never broken down. 1 means it is broken down every turbine.",
        )

        self.add_input("construct_duration", val=9, desc="Total project construction time (months)")
        self.add_input("hub_height_meters", val=80, units="m", desc="Hub height m")
        self.add_input("rotor_diameter_m", val=77, units="m", desc="Rotor diameter m")
        self.add_input("wind_shear_exponent", val=0.2, desc="Wind shear exponent")
        self.add_input("turbine_rating_MW", val=1.5, units="MW", desc="Turbine rating MW")
        self.add_input("fuel_cost_usd_per_gal", val=1.5, desc="Fuel cost USD/gal")

        self.add_input(
            "breakpoint_between_base_and_topping_percent", val=0.8, desc="Breakpoint between base and topping (percent)"
        )

        # Could not place units in turbine_spacing_rotor_diameters
        self.add_input("turbine_spacing_rotor_diameters", desc="Turbine spacing (times rotor diameter)", val=4)

        self.add_input("depth", units="m", desc="Foundation depth m", val=2.36)
        self.add_input("rated_thrust_N", units="N", desc="Rated Thrust (N)", val=5.89e5)

        # Can't set units
        self.add_input("bearing_pressure_n_m2", desc="Bearing Pressure (n/m2)", val=191521)

        self.add_input("gust_velocity_m_per_s", units="m/s", desc="50-year Gust Velocity (m/s)", val=59.5)
        self.add_input("road_length_adder_m", units="m", desc="Road length adder (m)", val=5000)

        # Can't set units
        self.add_input("fraction_new_roads", desc="Percent of roads that will be constructed (0.0 - 1.0)", val=0.33)

        self.add_input("road_quality", desc="Road Quality (0-1)", val=0.6)
        self.add_input("line_frequency_hz", units="Hz", desc="Line Frequency (Hz)", val=60)

        # Can't set units
        self.add_input("row_spacing_rotor_diameters", desc="Row spacing (times rotor diameter)", val=10)

        self.add_input(
            "trench_len_to_substation_km", units="km", desc="Combined Homerun Trench Length to Substation (km)", val=50
        )
        self.add_input("distance_to_interconnect_mi", units="mi", desc="Distance to interconnect (miles)", val=5)
        self.add_input("interconnect_voltage_kV", units="kV", desc="Interconnect Voltage (kV)", val=130)
        self.add_input(
            "critical_speed_non_erection_wind_delays_m_per_s",
            units="m/s",
            desc="Non-Erection Wind Delay Critical Speed (m/s)",
            val=15,
        )
        self.add_input(
            "critical_height_non_erection_wind_delays_m",
            units="m",
            desc="Non-Erection Wind Delay Critical Height (m)",
            val=10,
        )
        self.add_discrete_input("road_distributed_winnd", val=False)
        self.add_input("road_width_ft", units="ft", desc="Road width (ft)", val=20)
        self.add_input("road_thickness", desc="Road thickness (in)", val=8)
        self.add_input("crane_width", units="m", desc="Crane width (m)", val=12.2)
        self.add_input("overtime_multiplier", desc="Overtime multiplier", val=1.4)
        self.add_input("markup_contingency", desc="Markup contingency", val=0.03)
        self.add_input("markup_warranty_management", desc="Markup warranty management", val=0.0002)
        self.add_input("markup_sales_and_use_tax", desc="Markup sales and use tax", val=0)
        self.add_input("markup_overhead", desc="Markup overhead", val=0.05)
        self.add_input("markup_profit_margin", desc="Markup profit margin", val=0.05)
        self.add_input("Mass tonne", val=(1.0,), desc="", units="t")
        self.add_input(
            "development_labor_cost_usd", val=1e6, desc="The cost of labor in the development phase", units="USD"
        )
        # Disabled due to Pandas conflict right now.
        self.add_input("labor_cost_multiplier", val=1.0, desc="Labor cost multiplier")

        self.add_input("commissioning_pct", 0.01)
        self.add_input("decommissioning_pct", 0.15)

    def setup_discrete_inputs_that_are_not_dataframes(self):
        """
        This method sets up the discrete inputs that aren't dataframes.
        """
        self.add_discrete_input("num_turbines", val=100, desc="Number of turbines in project")

        # Since 3 blades are so common on rotors, that is a reasonable default
        # value that will not need to be checked during component list
        # assembly.

        self.add_discrete_input("number_of_blades", val=3, desc="Number of blades on the rotor")

        self.add_discrete_input(
            "user_defined_home_run_trench", val=0, desc="Flag for user-defined home run trench length (0 = no; 1 = yes)"
        )

        self.add_discrete_input(
            "allow_same_flag",
            val=False,
            desc="Allow same crane for base and topping (True or False)",
        )

        self.add_discrete_input(
            "hour_day",
            desc="Dictionary of normal and long hours for construction in a day in the form of {'long': 24, 'normal': 10}",
            val={"long": 24, "normal": 10},
        )

        self.add_discrete_input(
            "time_construct",
            desc="One of the keys in the hour_day dictionary to specify how many hours per day construction happens.",
            val="normal",
        )

        self.add_discrete_input(
            "user_defined_distance_to_grid_connection",
            desc="Flag for user-defined home run trench length (True or False)",
            val=False,
        )

        # Could not place units in rate_of_deliveries
        self.add_discrete_input("rate_of_deliveries", val=10, desc="Rate of deliveries (turbines per week)")

        self.add_discrete_input("new_switchyard", desc="New Switchyard (True or False)", val=True)
        self.add_discrete_input("num_hwy_permits", desc="Number of highway permits", val=10)
        self.add_discrete_input("num_access_roads", desc="Number of access roads", val=2)

    def setup_discrete_inputs_that_are_dataframes(self):
        """
        This sets up the default inputs that are dataframes. They are separate
        because they hold the project data and the way we need to hold their
        data is different. They have defaults loaded at the top of the file
        which can be overridden outside by setting the properties listed
        below.
        """
        # Read in default sheets for project data
        default_project_data = OpenMDAODataframeCache.read_all_sheets_from_xlsx("ge15_public")

        self.add_discrete_input(
            "site_facility_building_area_df",
            val=default_project_data["site_facility_building_area"],
            desc="site_facility_building_area DataFrame",
        )

        self.add_discrete_input(
            "components",
            val=default_project_data["components"],
            desc="Dataframe of components for tower, blade, nacelle",
        )

        self.add_discrete_input(
            "crane_specs", val=default_project_data["crane_specs"], desc="Dataframe of specifications of cranes"
        )

        self.add_discrete_input(
            "weather_window",
            val=read_weather_window(default_project_data["weather_window"]),
            desc="Dataframe of wind toolkit data",
        )

        self.add_discrete_input("crew", val=default_project_data["crew"], desc="Dataframe of crew configurations")

        self.add_discrete_input(
            "crew_price",
            val=default_project_data["crew_price"],
            desc="Dataframe of costs per hour for each type of worker.",
        )

        self.add_discrete_input(
            "equip", val=default_project_data["equip"], desc="Collections of equipment to perform erection operations."
        )

        self.add_discrete_input(
            "equip_price", val=default_project_data["equip_price"], desc="Prices for various type of equipment."
        )

        self.add_discrete_input("rsmeans", val=default_project_data["rsmeans"], desc="RSMeans price data")

        self.add_discrete_input(
            "cable_specs", val=default_project_data["cable_specs"], desc="cable specs for collection system"
        )

        self.add_discrete_input(
            "material_price",
            val=default_project_data["material_price"],
            desc="Prices of materials for foundations and roads",
        )

        self.add_discrete_input("project_data", val=default_project_data, desc="Dictionary of all dataframes of data")

    def setup_outputs(self):
        """
        This method sets up the continuous outputs. This is where total costs
        and installation times go.

        To see how cost totals are calculated see, the compute_total_bos_costs
        method below.
        """
        self.add_output(
            "bos_capex", 0.0, units="USD", desc="Total BOS CAPEX not including commissioning or decommissioning."
        )
        self.add_output(
            "bos_capex_kW",
            0.0,
            units="USD/kW",
            desc="Total BOS CAPEX per kW not including commissioning or decommissioning.",
        )
        self.add_output(
            "total_capex", 0.0, units="USD", desc="Total BOS CAPEX including commissioning and decommissioning."
        )
        self.add_output(
            "total_capex_kW",
            0.0,
            units="USD/kW",
            desc="Total BOS CAPEX per kW including commissioning and decommissioning.",
        )
        self.add_output("installation_capex", 0.0, units="USD", desc="Total foundation and erection installation cost.")
        self.add_output(
            "installation_capex_kW", 0.0, units="USD", desc="Total foundation and erection installation cost per kW."
        )
        self.add_output("installation_time_months", 0.0, desc="Total balance of system installation time (months).")

    def setup_discrete_outputs(self):
        """
        This method sets up discrete outputs.
        """
        self.add_discrete_output(
            "landbosse_costs_by_module_type_operation", desc="The costs by module, type and operation", val=None
        )

        self.add_discrete_output(
            "landbosse_details_by_module",
            desc="The details from the run of LandBOSSE. This includes some costs, but mostly other things",
            val=None,
        )

        self.add_discrete_output("erection_crane_choice", desc="The crane choices for erection.", val=None)

        self.add_discrete_output(
            "erection_component_name_topvbase",
            desc="List of components and whether they are a topping or base operation",
            val=None,
        )

        self.add_discrete_output(
            "erection_components", desc="List of components with their values modified from the defaults.", val=None
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        This runs the ErectionCost module using the inputs and outputs into and
        out of this module.

        Note: inputs, discrete_inputs are not dictionaries. They do support
        [] notation. inputs is of class 'openmdao.vectors.default_vector.DefaultVector'
        discrete_inputs is of class openmdao.core.component._DictValues. Other than
        [] brackets, they do not behave like dictionaries. See the following
        documentation for details.

        http://openmdao.org/twodocs/versions/latest/_srcdocs/packages/vectors/default_vector.html
        https://mdolab.github.io/OpenAeroStruct/_modules/openmdao/core/component.html

        Parameters
        ----------
        inputs : openmdao.vectors.default_vector.DefaultVector
            A dictionary-like object with NumPy arrays that hold float
            inputs. Note that since these are NumPy arrays, they
            need indexing to pull out simple float64 values.

        outputs : openmdao.vectors.default_vector.DefaultVector
            A dictionary-like object to store outputs.

        discrete_inputs : openmdao.core.component._DictValues
            A dictionary-like with the non-numeric inputs (like
            pandas.DataFrame)

        discrete_outputs : openmdao.core.component._DictValues
            A dictionary-like for non-numeric outputs (like
            pandas.DataFrame)
        """

        # Put the inputs together and run all the modules
        master_output_dict = dict()
        master_input_dict = self.prepare_master_input_dictionary(inputs, discrete_inputs)
        manager = Manager(master_input_dict, master_output_dict)
        result = manager.execute_landbosse("WISDEM")

        # Check if everything executed correctly
        if result != 0:
            raise Exception("LandBOSSE didn't execute correctly")

        # Gather the cost and detail outputs

        costs_by_module_type_operation = self.gather_costs_from_master_output_dict(master_output_dict)
        discrete_outputs["landbosse_costs_by_module_type_operation"] = costs_by_module_type_operation

        details = self.gather_details_from_master_output_dict(master_output_dict)
        discrete_outputs["landbosse_details_by_module"] = details

        # This is where we have access to the modified components, so put those
        # in the outputs of the component
        discrete_outputs["erection_components"] = master_input_dict["components"]

        # Now get specific outputs. These have been refactored to methods that work
        # with each module so as to keep this method as compact as possible.
        self.gather_specific_erection_outputs(master_output_dict, outputs, discrete_outputs)

        # Compute the total BOS costs
        self.compute_total_bos_costs(costs_by_module_type_operation, master_output_dict, inputs, outputs)

    def prepare_master_input_dictionary(self, inputs, discrete_inputs):
        """
        This prepares a master input dictionary by applying all the necessary
        modifications to the inputs.

        Parameters
        ----------
        inputs : openmdao.vectors.default_vector.DefaultVector
            A dictionary-like object with NumPy arrays that hold float
            inputs. Note that since these are NumPy arrays, they
            need indexing to pull out simple float64 values.

        discrete_inputs : openmdao.core.component._DictValues
            A dictionary-like with the non-numeric inputs (like
            pandas.DataFrame)

        Returns
        -------
        dict
            The prepared master input to go to the Manager.
        """
        inputs_dict = {key: inputs[key][0] for key in inputs.keys()}
        discrete_inputs_dict = {key: value for key, value in discrete_inputs.items()}
        incomplete_input_dict = {**inputs_dict, **discrete_inputs_dict}

        # Modify the default component data if needed and copy it into the
        # appropriate values of the input dictionary.
        modified_components = self.modify_component_lists(inputs, discrete_inputs)
        incomplete_input_dict["project_data"]["components"] = modified_components
        incomplete_input_dict["components"] = modified_components

        # FoundationCost needs to have all the component data split into separate
        # NumPy arrays.
        incomplete_input_dict["component_data"] = modified_components
        for component in incomplete_input_dict["component_data"].keys():
            incomplete_input_dict[component] = np.array(incomplete_input_dict["component_data"][component])

        # These are aliases because parts of the code call the same thing by
        # difference names.
        incomplete_input_dict["crew_cost"] = discrete_inputs["crew_price"]
        incomplete_input_dict["cable_specs_pd"] = discrete_inputs["cable_specs"]

        # read in RSMeans per diem:
        crew_cost = discrete_inputs["crew_price"]
        crew_cost = crew_cost.set_index("Labor type ID", drop=False)
        incomplete_input_dict["rsmeans_per_diem"] = crew_cost.loc["RSMeans", "Per diem USD per day"]

        # Calculate project size in megawatts
        incomplete_input_dict["project_size_megawatts"] = float(
            discrete_inputs["num_turbines"] * inputs["turbine_rating_MW"][0]
        )

        # Needed to avoid distributed wind keys
        incomplete_input_dict["road_distributed_wind"] = False

        defaults = DefaultMasterInputDict()
        master_input_dict = defaults.populate_input_dict(incomplete_input_dict)

        return master_input_dict

    def gather_costs_from_master_output_dict(self, master_output_dict):
        """
        This method extract all the cost_by_module_type_operation lists for
        output in an Excel file.

        It finds values for the keys ending in '_module_type_operation'. It
        then concatenates them together so they can be easily written to
        a .csv or .xlsx

        On every row, it includes the:
            Rotor diameter m
            Turbine rating MW
            Number of turbines

        This enables easy mapping of new columns if need be. The columns have
        spaces in the names so that they can be easily written to a user-friendly
        output.

        Parameters
        ----------
        runs_dict : dict
            Values are the names of the projects. Keys are the lists of
            dictionaries that are lines for the .csv

        Returns
        -------
        list
            List of dicts to write to the .csv.
        """
        line_items = []

        # Gather the lists of costs
        cost_lists = [value for key, value in master_output_dict.items() if key.endswith("_module_type_operation")]

        # Flatten the list of lists that is the result of the gathering
        for cost_list in cost_lists:
            line_items.extend(cost_list)

        # Filter out the keys needed and rename them to meaningful values
        final_costs = []
        for line_item in line_items:
            item = {
                "Module": line_item["module"],
                "Type of cost": line_item["type_of_cost"],
                "Cost / kW": line_item["usd_per_kw_per_project"],
                "Cost / project": line_item["cost_per_project"],
                "Cost / turbine": line_item["cost_per_turbine"],
                "Number of turbines": line_item["num_turbines"],
                "Rotor diameter (m)": line_item["rotor_diameter_m"],
                "Turbine rating (MW)": line_item["turbine_rating_MW"],
                "Project ID with serial": line_item["project_id_with_serial"],
            }
            final_costs.append(item)

        return final_costs

    def gather_details_from_master_output_dict(self, master_output_dict):
        """
        This extracts the detail lists from all the modules to output
        the detailed non-cost data from the model run.

        Parameters
        ----------
        master_output_dict : dict
            The master output dict with the finished module output in it.

        Returns
        -------
        list
            List of dicts with detailed data.
        """
        line_items = []

        # Gather the lists of costs
        details_lists = [value for key, value in master_output_dict.items() if key.endswith("_csv")]

        # Flatten the list of lists
        for details_list in details_lists:
            line_items.extend(details_list)

        return line_items

    def gather_specific_erection_outputs(self, master_output_dict, outputs, discrete_outputs):
        """
        This method gathers specific outputs from the ErectionCost module and places
        them on the outputs.

        The method does not return anything. Rather, it places the outputs directly
        on the continuous of discrete outputs.

        Parameters
        ----------
        master_output_dict: dict
            The master output dictionary out of LandBOSSE

        outputs : openmdao.vectors.default_vector.DefaultVector
            A dictionary-like object to store outputs.

        discrete_outputs : openmdao.core.component._DictValues
            A dictionary-like for non-numeric outputs (like
            pandas.DataFrame)
        """
        discrete_outputs["erection_crane_choice"] = master_output_dict["crane_choice"]
        discrete_outputs["erection_component_name_topvbase"] = master_output_dict["component_name_topvbase"]

    def compute_total_bos_costs(self, costs_by_module_type_operation, master_output_dict, inputs, outputs):
        """
        This computes the total BOS costs from the master output dictionary
        and places them on the necessary outputs.

        Parameters
        ----------
        costs_by_module_type_operation: List[Dict[str, Any]]
            The lists of costs by module, type and operation.

        master_output_dict: Dict[str, Any]
            The master output dictionary from the run. Used to obtain the
            construction time,

        outputs : openmdao.vectors.default_vector.DefaultVector
            The outputs in which to place the results of the computations
        """
        bos_per_kw = 0.0
        bos_per_project = 0.0
        installation_per_project = 0.0
        installation_per_kW = 0.0

        for row in costs_by_module_type_operation:
            bos_per_kw += row["Cost / kW"]
            bos_per_project += row["Cost / project"]
            if row["Module"] in ["ErectionCost", "FoundationCost"]:
                installation_per_project += row["Cost / project"]
                installation_per_kW += row["Cost / kW"]

        commissioning_pct = inputs["commissioning_pct"]
        decommissioning_pct = inputs["decommissioning_pct"]

        commissioning_per_project = bos_per_project * commissioning_pct
        decomissioning_per_project = bos_per_project * decommissioning_pct
        commissioning_per_kW = bos_per_kw * commissioning_pct
        decomissioning_per_kW = bos_per_kw * decommissioning_pct

        outputs["total_capex_kW"] = bos_per_kw + commissioning_per_kW + decomissioning_per_kW
        outputs["total_capex"] = bos_per_project + commissioning_per_project + decomissioning_per_project
        outputs["bos_capex"] = bos_per_project
        outputs["bos_capex_kW"] = bos_per_kw
        outputs["installation_capex"] = installation_per_project
        outputs["installation_capex_kW"] = installation_per_kW

        actual_construction_months = master_output_dict["actual_construction_months"]
        outputs["installation_time_months"] = actual_construction_months

    def modify_component_lists(self, inputs, discrete_inputs):
        """
        This method modifies the previously loaded default component lists with
        data about blades, tower sections, if they have been provided as input
        to the component.

        It only modifies the project component data if default data for the proper
        inputs have been overridden.

        The default blade data is assumed to be the first component that begins
        with the word "Blade"

        This should take mass from the tower in WISDEM. Ideally, this should have
        an input for transportable tower 4.3, large diameter steel tower LDST 6.2m, or
        unconstrained key stone tower. Or give warnings about the boundaries
        that we assume.

        Parameters
        ----------
        inputs : openmdao.vectors.default_vector.DefaultVector
            A dictionary-like object with NumPy arrays that hold float
            inputs. Note that since these are NumPy arrays, they
            need indexing to pull out simple float64 values.

        discrete_inputs : openmdao.core.component._DictValues
            A dictionary-like with the non-numeric inputs (like
            pandas.DataFrame)

        Returns
        -------
        pd.DataFrame
            The dataframe with the modified components.
        """
        input_components = discrete_inputs["components"]

        # This list is a sequence of pd.Series instances that have the
        # specifications of each component.
        output_components_list = []

        # Need to convert kg to tonnes
        kg_per_tonne = 1000

        # Get the hub height
        hub_height_meters = float(inputs["hub_height_meters"][0])

        # Make the nacelle. This does not include the hub or blades.
        nacelle_mass_kg = float(inputs["nacelle_mass"][0])
        nacelle = input_components[input_components["Component"].str.startswith("Nacelle")].iloc[0].copy()
        if inputs["nacelle_mass"] != use_default_component_data:
            nacelle["Mass tonne"] = nacelle_mass_kg / kg_per_tonne
            nacelle["Component"] = "Nacelle"
        nacelle["Lift height m"] = nacelle["Lever arm m"] = hub_height_meters
        output_components_list.append(nacelle)

        # Make the hub
        hub_mass_kg = float(inputs["hub_mass"][0])
        hub = input_components[input_components["Component"].str.startswith("Hub")].iloc[0].copy()
        hub["Lift height m"] = hub["Lever arm m"] = hub_height_meters
        if hub_mass_kg != use_default_component_data:
            hub["Mass tonne"] = hub_mass_kg / kg_per_tonne
        output_components_list.append(hub)

        # Make blades
        blade = input_components[input_components["Component"].str.startswith("Blade")].iloc[0].copy()

        # There is always a hub height, so use that as the lift height
        blade["Lift height m"] = blade["Lever arm m"] = hub_height_meters

        if float(inputs["blade_drag_coefficient"][0]) != use_default_component_data:
            blade["Coeff drag"] = float(inputs["blade_drag_coefficient"][0])

        if float(inputs["blade_lever_arm"][0]) != use_default_component_data:
            blade["Lever arm m"] = float(inputs["blade_lever_arm"][0])

        if float(inputs["blade_install_cycle_time"][0]) != use_default_component_data:
            blade["Cycle time installation hrs"] = float(inputs["blade_install_cycle_time"][0])

        if float(inputs["blade_offload_hook_height"][0]) != use_default_component_data:
            blade["Offload hook height m"] = hub_height_meters

        if float(inputs["blade_offload_cycle_time"][0]) != use_default_component_data:
            blade["Offload cycle time hrs"] = inputs["blade_offload_cycle_time"][0]

        if float(inputs["blade_drag_multiplier"][0]) != use_default_component_data:
            blade["Multiplier drag rotor"] = inputs["blade_drag_multiplier"][0]

        if float(inputs["blade_mass"][0]) != use_default_component_data:
            blade["Mass tonne"] = float(inputs["blade_mass"][0]) / kg_per_tonne

        if float(inputs["blade_surface_area"][0]) != use_default_component_data:
            blade["Surface area sq m"] = float(inputs["blade_surface_area"][0])

        # Assume that number_of_blades always has a reasonable value. It's
        # default count when the discrete input is declared of 3 is always
        # reasonable unless overridden by another input.
        number_of_blades = discrete_inputs["number_of_blades"]
        for i in range(number_of_blades):
            component = f"Blade {i + 1}"
            blade_i = blade.copy()
            blade_i["Component"] = component
            output_components_list.append(blade_i)

        # Make tower sections
        tower_mass_tonnes = float(inputs["tower_mass"][0]) / kg_per_tonne
        tower_height_m = hub_height_meters - float(inputs["foundation_height"][0])
        default_tower_section = input_components[input_components["Component"].str.startswith("Tower")].iloc[0]
        tower_sections = self.make_tower_sections(tower_mass_tonnes, tower_height_m, default_tower_section)
        output_components_list.extend(tower_sections)

        # Make the output component dataframe and return it.
        output_components = pd.DataFrame(output_components_list)
        return output_components

    @staticmethod
    def make_tower_sections(tower_mass_tonnes, tower_height_m, default_tower_section):
        """
        This makes tower sections for a transportable tower.

        Approximations:

        - Weight is distributed uniformly among the sections

        - The number of sections is either the maximum allowed by mass or
          the maximum allowed by height, to maintain transportability.

        For each tower section, calculate:
            - lift height
            - lever arm
            - surface area

        The rest of values should remain at their defaults.

        Note: Tower sections are constrained in maximum diameter to 4.5 m.
            However, their surface area is calculated with a 1.3 m radius
            to agree more closely with empirical data. Also, tower sections
            are approximated as cylinders.

        Parameters
        ----------
        tower_mass_tonnes: float
            The total tower mass in tonnes

        tower_height_m: float
            The total height of the tower in meters.

        default_tower_section: pd.Series
            There are a number of values that are kept constant in creating
            the tower sections. This series holds the values.

        Returns
        -------
        List[pd.Series]
            A list of series to be appended onto an output component list.
            It is not a dataframe, because it is faster to append to a list
            and make a dataframe once.
        """
        tower_radius = 1.3

        number_of_sections = max(ceil(tower_height_m / 30), ceil(tower_mass_tonnes / 80))

        tower_section_height_m = tower_height_m / number_of_sections

        tower_section_mass = tower_mass_tonnes / number_of_sections

        tower_section_surface_area_m2 = np.pi * tower_section_height_m * (tower_radius**2)

        sections = []
        for i in range(number_of_sections):
            lift_height_m = (i * tower_section_height_m) + tower_section_height_m
            lever_arm = (i * tower_section_height_m) + (0.5 * tower_section_height_m)
            name = f"Tower {i + 1}"

            section = default_tower_section.copy()
            section["Component"] = name
            section["Mass tonne"] = tower_section_mass
            section["Lift height m"] = lift_height_m
            section["Surface area sq m"] = tower_section_surface_area_m2
            section["Section height m"] = tower_section_height_m
            section["Lever arm m"] = lever_arm

            sections.append(section)

        return sections
