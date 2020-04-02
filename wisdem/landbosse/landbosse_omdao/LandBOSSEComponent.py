from functools import reduce

import openmdao.api as om
import numpy as np

from ..model.Manager import Manager
from ..model.DefaultMasterInputDict import DefaultMasterInputDict


class LandBOSSEComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('top_level_flag', default=True)

    def setup(self):
        # if self.options['top_level_flag']:
        #     shared_indeps = om.IndepVarComp()
        #     shared_indeps.add_output('hub_height', val=0.0, units='m')
        #     self.add_subsystem('indeps', shared_indeps, promotes=['*'])

        self.add_discrete_input('default_project_data_filename',
                                val=None,
                                desc='The input project data filename that has default project data.')

        self.setup_inputs()
        self.setup_discrete_outputs()
        self.setup_discrete_inputs_that_are_not_dataframes()
        self.setup_discrete_inputs_that_are_dataframes()

    def setup_inputs(self):
        """
        This method sets up the inputs.
        """
        self.add_input('crane_breakdown_fraction', val=0.0,
                       desc='0 means the crane is never broken down. 1 means it is broken down every turbine.')

        self.add_input('construct_duration', val=9, desc='Total project construction time (months)')
        self.add_input('hub_height_meters', val=80, units='m', desc='Hub height m')
        self.add_input('rotor_diameter_m', val=77, units='m', desc='Rotor diameter m')
        self.add_input('wind_shear_exponent', val=0.2, desc='Wind shear exponent')
        self.add_input('turbine_rating_MW', val=1.5, units='MW', desc='Turbine rating MW')
        self.add_input('num_turbines', val=100, desc='Number of turbines in project')
        self.add_input('fuel_cost_usd_per_gal', val=1.0, desc='Fuel cost USD/gal')

        self.add_input('breakpoint_between_base_and_topping_percent', val=70,
                       desc='Breakpoint between base and topping (percent)')

        # Could not place units in rate_of_deliveries
        self.add_input('rate_of_deliveries', val=10, desc='Rate of deliveries (turbines per week)')

        # Could not place units in turbine_spacing_rotor_diameters
        # indeps.add_output('turbine_spacing_rotor_diameters', units='rotor diameters', desc='Turbine spacing (times rotor diameter)', val=4)
        self.add_input('turbine_spacing_rotor_diameters', desc='Turbine spacing (times rotor diameter)', val=4)

        self.add_input('depth', units='m', desc='Foundation depth m', val=2.36)
        self.add_input('rated_thrust_N', units='N', desc='Rated Thrust (N)', val=5.89e5)

        # Can't set units
        # indeps.add_output('bearing_pressure_n_m2', units='n/m2', desc='Bearing Pressure (n/m2)', val=191521)
        self.add_input('bearing_pressure_n_m2', desc='Bearing Pressure (n/m2)', val=191521)

        self.add_input('gust_velocity_m_per_s', units='m/s', desc='50-year Gust Velocity (m/s)', val=59.5)
        self.add_input('road_length_adder_m', units='m', desc='Road length adder (m)', val=5000)

        # Can't set units
        self.add_input('fraction_new_roads',
                       desc='Percent of roads that will be constructed (0.0 - 1.0)', val=0.33)

        self.add_input('road_quality', desc='Road Quality (0-1)', val=0.6)
        self.add_input('line_frequency_hz', units='Hz', desc='Line Frequency (Hz)', val=60)

        # Can't set units
        self.add_input('row_spacing_rotor_diameters',
                       desc='Row spacing (times rotor diameter)', val=4)

        self.add_input(
            'user_defined_distance_to_grid_connection',
            desc='Flag for user-defined home run trench length (True or False)',
            val=False
        )

        self.add_input('trench_len_to_substation_km', units='km',
                       desc='Combined Homerun Trench Length to Substation (km)', val=50)
        self.add_input('distance_to_interconnect_mi', units='mi', desc='Distance to interconnect (miles)', val=5)
        self.add_input('interconnect_voltage_kV', units='kV', desc='Interconnect Voltage (kV)', val=130)
        self.add_input('new_switchyard', desc='New Switchyard (True or False)', val=True)
        self.add_input('critical_speed_non_erection_wind_delays_m_per_s', units='m/s',
                       desc='Non-Erection Wind Delay Critical Speed (m/s)', val=15)
        self.add_input('critical_height_non_erection_wind_delays_m', units='m',
                       desc='Non-Erection Wind Delay Critical Height (m)', val=10)
        self.add_input('road_width_ft', units='ft', desc='Road width (ft)', val=20)
        self.add_input('road_thickness', desc='Road thickness (in)', val=8)
        self.add_input('crane_width', units='m', desc='Crane width (m)', val=12.2)
        self.add_input('num_hwy_permits', desc='Number of highway permits', val=10)
        self.add_input('num_access_roads', desc='Number of access roads', val=2)
        self.add_input('overtime_multiplier', desc='Overtime multiplier', val=1.4)
        self.add_input('markup_contingency', desc='Markup contingency', val=0.03)
        self.add_input('markup_warranty_management', desc='Markup warranty management', val=0.0002)
        self.add_input('markup_sales_and_use_tax', desc='Markup sales and use tax', val=0)
        self.add_input('markup_overhead', desc='Markup overhead', val=0.05)
        self.add_input('markup_profit_margin', desc='Markup profit margin', val=0.05)
        self.add_input('Mass tonne', val=(1.,), desc='', units='t')
        self.add_input('development_labor_cost_usd', val=1e6, desc='The cost of labor in the development phase',
                       units='USD')

    def setup_discrete_inputs_that_are_not_dataframes(self):
        """
        This method sets up the discrete inputs that aren't dataframes.
        The dataframes need to be handled differently because the way
        they will get their default data is different.
        """
        self.add_discrete_input('user_defined_home_run_trench', val=1,
                                desc='Flag for user-defined home run trench length (0 = no; 1 = yes)')

        self.add_discrete_input(
            'allow_same_flag',
            val=False,
            desc='Allow same crane for base and topping (True or False)',
        )

        self.add_discrete_input(
            'hour_day',
            desc="Dictionary of normal and long hours for construction in a day in the form of {'long': 24, 'normal': 10}",
            val={'long': 24, 'normal': 10}
        )

        self.add_discrete_input(
            'time_construct',
            desc='One of the keys in the hour_day dictionary to specify how many hours per day construction happens.',
            val='normal'
        )

    def setup_discrete_inputs_that_are_dataframes(self):
        """
        This sets up the default inputs that are dataframes. They are separate
        because the way they handle their default data is different.

        To get around the need to specify dataframes in the problem inputs,
        these methods just populate needed dataframes (which may or may not
        have been modified) as defaults. These inputs probably shouldn't be
        promoted but we will worry about that later.
        """
        self.add_discrete_input('site_facility_building_area_df', val=None,
                                desc='site_facility_building_area DataFrame')
        self.add_discrete_input('components', val=None, desc='Dataframe of components for tower, blade, nacelle')
        self.add_discrete_input('crane_specs', val=None, desc='Dataframe of specifications of cranes')
        self.add_discrete_input('weather_window', val=None, desc='Dataframe of wind toolkit data')
        self.add_discrete_input('crew', val=None, desc='Dataframe of crew configurations')
        self.add_discrete_input('crew_price', val=None, desc='Dataframe of costs per hour for each type of worker.')
        self.add_discrete_input('equip', val=None, desc='Collections of equipment to perform erection operations.')
        self.add_discrete_input('equip_price', val=None, desc='Prices for various type of equipment.')
        self.add_discrete_input('rsmeans', val=None, desc='RSMeans price data')
        self.add_discrete_input('cable_specs', val=None, desc='cable specs for collection system')

        self.add_discrete_input(
            'material_price',
            desc='Prices of materials for foundations and roads',
            val=None
        )

        # This is a dictionary with values that are mostly the dataframes above
        # This is somewhat redundant, but it is what many parts of the code rely
        # on.
        self.add_discrete_input('project_data', val=None, desc='Dictionary of all dataframes of data')

    def setup_outputs(self):
        """
        This method sets up the outputs.

        But wait--all outputs are discrete! So this method is simply blank.
        """
        pass

    def setup_discrete_outputs(self):
        """
        This method sets up discrete outputs.
        """
        self.add_discrete_output(
            'landbosse_costs_by_module_type_operation',
            desc='The costs by module, type and operation',
            val=None
        )

        self.add_discrete_output(
            'landbosse_details_by_module',
            desc='The details from the run of LandBOSSE. This includes some costs, but mostly other things',
            val=None
        )

        # OUTPUTS, SPECIFIC

        self.add_discrete_output(
            'erection_crane_choice',
            desc='The crane choices for erection.',
            val=None
        )

        self.add_discrete_output(
            'erection_component_name_topvbase',
            desc='List of components and whether they are a topping or base operation',
            val=None
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
        result = manager.execute_landbosse('WISDEM')

        # Check if everything executed correctly
        if result != 0:
            raise Exception("LandBOSSE didn't execute correctly")

        # Gather the cost and detail outputs

        costs_by_module_type_operation = self.gather_costs_from_master_output_dict(master_output_dict)
        discrete_outputs['landbosse_costs_by_module_type_operation'] = costs_by_module_type_operation

        details = self.gather_details_from_master_output_dict(master_output_dict)
        discrete_outputs['landbosse_details_by_module'] = details

        # Now get specific outputs. These have been refactored to methods that work
        # with each module so as to keep this method as compact as possible.
        self.gather_specific_erection_outputs(master_output_dict, outputs, discrete_outputs)

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

        # FoundationCost needs to have all the component data split into separate
        # NumPy arrays.
        incomplete_input_dict['component_data'] = discrete_inputs['components']
        for component in incomplete_input_dict['component_data'].keys():
            incomplete_input_dict[component] = np.array(incomplete_input_dict['component_data'][component])

        # These are aliases because parts of the code call the same thing by
        # difference names.
        incomplete_input_dict['crew_cost'] = discrete_inputs['crew_price']
        incomplete_input_dict['cable_specs_pd'] = discrete_inputs['cable_specs']

        # read in RSMeans per diem:
        crew_cost = discrete_inputs['crew_price']
        crew_cost = crew_cost.set_index("Labor type ID", drop=False)
        incomplete_input_dict['rsmeans_per_diem'] = crew_cost.loc['RSMeans', 'Per diem USD per day']

        # Calculate project size in megawatts
        incomplete_input_dict['project_size_megawatts'] = float(inputs['num_turbines'] * inputs['turbine_rating_MW'])

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
        cost_lists = [value for key, value in master_output_dict.items() if key.endswith('_module_type_operation')]

        # Flatten the list of lists that is the result of the gathering
        for cost_list in cost_lists:
            line_items.extend(cost_list)

        # Filter out the keys needed and rename them to meaningful values
        final_costs = []
        for line_item in line_items:
            item = {
                'Module': line_item['module'],
                'Type of cost': line_item['type_of_cost'],
                'Cost / kW': line_item['usd_per_kw_per_project'],
                'Cost / project': line_item['cost_per_project'],
                'Cost / turbine': line_item['cost_per_turbine'],
                'Number of turbines': line_item['num_turbines'],
                'Rotor diameter (m)': line_item['rotor_diameter_m'],
                'Turbine rating (MW)': line_item['turbine_rating_MW'],
                'Project ID with serial': line_item['project_id_with_serial']
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
        details_lists = [value for key, value in master_output_dict.items() if key.endswith('_csv')]

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
        discrete_outputs['erection_crane_choice'] = master_output_dict['crane_choice']
        discrete_outputs['erection_component_name_topvbase'] = master_output_dict['component_name_topvbase']
