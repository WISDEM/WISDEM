import openmdao.api as om

from .DummyComponent import DummyComponent
from .ManagementCostComponent import ManagementCostComponent
from .ErectionCostComponent import ErectionCostComopnent
from .FoundationCostComponent import FoundationCostComponent
from .CollectionCostComponent import CollectionCostComponent
from .SitePreparationComponent import SitePreparationCostComponent


class LandBOSSEGroup(om.Group):
    def initialize(self):
        self.options.declare('top_level_flag', default=True)

    def setup(self):
        # if self.options['top_level_flag']:
        #     shared_indeps = om.IndepVarComp()
        #     shared_indeps.add_output('hub_height', val=0.0, units='m')
        #     self.add_subsystem('indeps', shared_indeps, promotes=['*'])

        # Numeric inputs
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

        indeps.add_output('project_value_usd', val=1, units='USD', desc='Project value in USD')
        indeps.add_output('construct_duration', val=9, desc='Total project construction time (months)')
        indeps.add_output('hub_height_meters', val=80, units='m', desc='Hub height m')
        indeps.add_output('rotor_diameter_m', val=77, units='m', desc='Rotor diameter m')
        indeps.add_output('wind_shear_exponent', val=0.2, desc='Wind shear exponent')
        indeps.add_output('turbine_rating_MW', val=1.5, units='MW', desc='Turbine rating MW')
        indeps.add_output('num_turbines', val=100, desc='Number of turbines in project')
        indeps.add_output('fuel_cost_usd_per_gal', val=1.0, desc='Fuel cost USD/gal')
        indeps.add_output('foundation_cost_usd', val=1, units='USD', desc='Foundation cost, USD')

        indeps.add_output('breakpoint_between_base_and_topping_percent',
                          val=70,
                          desc='Breakpoint between base and topping (percent)')

        # Could not place units in rate_of_deliveries
        indeps.add_output('rate_of_deliveries', val=10, desc='Rate of deliveries (turbines per week)')

        # Could not place units in turbine_spacing_rotor_diameters
        # indeps.add_output('turbine_spacing_rotor_diameters', units='rotor diameters', desc='Turbine spacing (times rotor diameter)', val=4)
        indeps.add_output('turbine_spacing_rotor_diameters', desc='Turbine spacing (times rotor diameter)', val=4)

        indeps.add_output('depth', units='m', desc='Foundation depth m', val=2.36)
        indeps.add_output('rated_thrust_N', units='N', desc='Rated Thrust (N)', val=5.89e5)

        # Can't set units
        # indeps.add_output('bearing_pressure_n_m2', units='n/m2', desc='Bearing Pressure (n/m2)', val=191521)
        indeps.add_output('bearing_pressure_n_m2', desc='Bearing Pressure (n/m2)', val=191521)

        indeps.add_output('gust_velocity_m_per_s', units='m/s', desc='50-year Gust Velocity (m/s)', val=59.5)
        indeps.add_output('road_length_adder_m', units='m', desc='Road length adder (m)', val=5000)

        # Can't set units
        # indeps.add_output('fraction_new_roads', units='fraction', desc='Percent of roads that will be constructed (0.0 - 1.0)', val=0.33)
        indeps.add_output('fraction_new_roads',
                          desc='Percent of roads that will be constructed (0.0 - 1.0)', val=0.33)

        indeps.add_output('road_quality', desc='Road Quality (0-1)', val=0.6)
        indeps.add_output('line_frequency_hz', units='Hz', desc='Line Frequency (Hz)', val=60)

        # Can't set units
        indeps.add_output('row_spacing_rotor_diameters',
                          desc='Row spacing (times rotor diameter)', val=4)

        indeps.add_output(
            'user_defined_distance_to_grid_connection',
            desc='Flag for user-defined home run trench length (True or False)',
            val=False
        )

        indeps.add_output('trench_len_to_substation_km', units='km', desc='Combined Homerun Trench Length to Substation (km)', val=50)
        indeps.add_output('distance_to_interconnect_mi', units='mi', desc='Distance to interconnect (miles)', val=5)
        indeps.add_output('interconnect_voltage_kV', units='kV', desc='Interconnect Voltage (kV)', val=130)

        indeps.add_output('new_switchyard', desc='New Switchyard (True or False)', val=True)

        indeps.add_output('critical_speed_non_erection_wind_delays_m_per_s', units='m/s', desc='Non-Erection Wind Delay Critical Speed (m/s)', val=15)
        indeps.add_output('critical_height_non_erection_wind_delays_m', units='m', desc='Non-Erection Wind Delay Critical Height (m)', val=10)
        indeps.add_output('road_width_ft', units='ft', desc='Road width (ft)', val=20)

        # Can't add units
        indeps.add_output('road_thickness', desc='Road thickness (in)', val=8)

        indeps.add_output('crane_width', units='m', desc='Crane width (m)', val=12.2)
        indeps.add_output('num_hwy_permits', desc='Number of highway permits', val=10)
        indeps.add_output('num_access_roads', desc='Number of access roads', val=2)
        indeps.add_output('overtime_multiplier', desc='Overtime multiplier', val=1.4)

        # Dropping the column 'Override total management cost for distributed (0 does not override)'

        indeps.add_output('markup_contingency', desc='Markup contingency', val=0.03)
        indeps.add_output('markup_warranty_management', desc='Markup warranty management', val=0.0002)
        indeps.add_output('markup_sales_and_use_tax', desc='Markup sales and use tax', val=0)
        indeps.add_output('markup_overhead', desc='Markup overhead', val=0.05)
        indeps.add_output('markup_profit_margin', desc='Markup profit margin', val=0.05)

        # Numeric inputs, NumPy arrays
        indeps.add_output('Mass tonne', val=(1.,), desc='', units='t')

        # Discrete inputs, including dataframes
        indeps.add_discrete_output('site_facility_building_area_df', val=None, desc='site_facility_building_area DataFrame')
        indeps.add_discrete_output('components', val=None, desc='Dataframe of components for tower, blade, nacelle')
        indeps.add_discrete_output('crane_specs', val=None, desc='Dataframe of specifications of cranes')
        indeps.add_discrete_output('weather_window', val=None, desc='Dataframe of wind toolkit data')
        indeps.add_discrete_output('crew', val=None, desc='Dataframe of crew configurations')
        indeps.add_discrete_output('crew_price', val=None, desc='Dataframe of costs per hour for each type of worker.')
        indeps.add_discrete_output('equip', val=None, desc='Collections of equipment to perform erection operations.')
        indeps.add_discrete_output('equip_price', val=None, desc='Prices for various type of equipment.')
        indeps.add_discrete_output('rsmeans', val=None, desc='RSMeans price data')
        indeps.add_discrete_output('cable_specs', val=None, desc='cable specs for collection system')

        indeps.add_discrete_output(
            'allow_same_flag',
            val=False,
            desc='Allow same crane for base and topping (True or False)',
        )

        indeps.add_discrete_output(
            'hour_day',
            desc="Dictionary of normal and long hours for construction in a day in the form of {'long': 24, 'normal': 10}",
            val={'long': 24, 'normal': 10}
        )

        indeps.add_discrete_output(
            'time_construct',
            desc='One of the keys in the hour_day dictionary to specify how many hours per day construction happens.',
            val='normal'
        )

        indeps.add_discrete_output(
            'material_price',
            desc='Prices of materials for foundations and roads',
            val=None
        )

        self.add_subsystem('management_cost', ManagementCostComponent(), promotes=['*'])
        self.add_subsystem('erection_cost', ErectionCostComopnent(), promotes=['*'])
        self.add_subsystem('foundation_cost', FoundationCostComponent(), promotes=['*'])
        # self.add_subsystem('collection_cost', CollectionCostComponent(), promotes=['*'])
        self.add_subsystem('site_preparation_cost', SitePreparationCostComponent(), promotes=['*'])

# Calculate this input instead
# self.add_input('project_size_megawatts', units='MW', desc='(Number of turbines) * (Turbine rating MW)', value=)

