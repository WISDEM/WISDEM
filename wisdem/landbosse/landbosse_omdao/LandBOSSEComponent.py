import openmdao.api as om


class LandBOSSEComponent(om.ExplicitComponent):
    """
    This is a superclass for all the components that wrap LandBOSSE
    cost modules. It holds functionality used for the other components
    that wrap LandBOSSE cost modules.

    This component should not be instantiated directly.
    """

    def initialize(self):
        self.options.declare('top_level_flag', default=True)

    def setup(self):
        # if self.options['top_level_flag']:
        #     shared_indeps = om.IndepVarComp()
        #     shared_indeps.add_output('hub_height', val=0.0, units='m')
        #     self.add_subsystem('indeps', shared_indeps, promotes=['*'])

        self.add_input('project_value_usd', val=1, units='USD', desc='Project value in USD')
        self.add_input('construct_duration', val=9, desc='Total project construction time (months)')
        self.add_input('hub_height_meters', val=80, units='m', desc='Hub height m')
        self.add_input('rotor_diameter_m', val=77, units='m', desc='Rotor diameter m')
        self.add_input('wind_shear_exponent', val=0.2, desc='Wind shear exponent')
        self.add_input('turbine_rating_MW', val=1.5, units='MW', desc='Turbine rating MW')
        self.add_input('num_turbines', val=100, desc='Number of turbines in project')
        self.add_input('fuel_cost_usd_per_gal', val=1.0, desc='Fuel cost USD/gal')
        self.add_input('foundation_cost_usd', val=1, units='USD', desc='Foundation cost, USD')

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

        self.add_input('trench_len_to_substation_km', units='km', desc='Combined Homerun Trench Length to Substation (km)', val=50)
        self.add_input('distance_to_interconnect_mi', units='mi', desc='Distance to interconnect (miles)', val=5)
        self.add_input('interconnect_voltage_kV', units='kV', desc='Interconnect Voltage (kV)', val=130)
        self.add_input('new_switchyard', desc='New Switchyard (True or False)', val=True)
        self.add_input('critical_speed_non_erection_wind_delays_m_per_s', units='m/s', desc='Non-Erection Wind Delay Critical Speed (m/s)', val=15)
        self.add_input('critical_height_non_erection_wind_delays_m', units='m', desc='Non-Erection Wind Delay Critical Height (m)', val=10)
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

        self.add_discrete_input('site_facility_building_area_df', val=None, desc='site_facility_building_area DataFrame')
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

        self.add_discrete_input(
            'material_price',
            desc='Prices of materials for foundations and roads',
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

        # Create real dictionaries to pass to the module
        inputs_dict = {key: inputs[key][0] for key in inputs.keys()}
        discrete_inputs_dict = {key: value for key, value in discrete_inputs.items()}
        master_inputs_dict = {**inputs_dict, **discrete_inputs_dict}
        master_outputs_dict = dict()

    def print_verbose_module_type_operation(self, module_name, module_type_operation):
        """
        This method prints the module_type_operation costs list for
        verbose output.

        Parameters
        ----------
        module_name : str
            The name of the LandBOSSE module being logged.

        module_type_operation : list
            The list of dictionaries of costs to print.
        """
        print('################################################')
        print(f'LandBOSSE {module_name} costs by module, type and operation')
        for row in module_type_operation:
            operation_id = row['operation_id']
            type_of_cost = row['type_of_cost']
            cost_per_turbine = round(row['cost_per_turbine'], 2)
            cost_per_project = round(row['cost_per_project'], 2)
            usd_per_kw = round(row['usd_per_kw_per_project'], 2)
            print(f'{operation_id}\t{type_of_cost}\t${cost_per_turbine}/turbine\t${cost_per_project}/project\t${usd_per_kw}/kW')
        print('################################################')

    def print_verbose_details(self, module_name, details):
        """
        This method prints the verbose details output of a module.

        Parameters
        ----------
        module_name : str
            The name of the cost module reporting thee details

        details : list[dict]
            The list of dictionaries that contain the details to be
            printed.
        """
        print('################################################')
        print(f'LandBOSSE {module_name} detailed outputs')
        for row in details:
            unit = row['unit']
            name = row['variable_df_key_col_name']
            value = row['value']
            print(f'{name}\t{unit}\t{value}')
        print('################################################')

