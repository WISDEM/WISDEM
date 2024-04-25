import math
import traceback

class ManagementCost:
    """
    This class models management costs of a wind plant. Its inputs are
    configured with a dictionary with the key value pairs being the
    inputs into the model.

    See the input_output documentation below for a description of the key
    names for this inputs values dictionary.

    The INPUT keys are the following:

    project_value_usd
        (float) Sum of all other BOS costs (e.g., roads, foundations, erection)

    foundation_cost_usd
        (float) Foundation costs for the project

    construct_duration
        (float) Project duration (in months)

    num_hwy_permits
        (int) Number of highway permits needed for the project

    markup_constants
        (dict) Markup and contingency costs that can be set by user
        (see the markup_contingency method for key names to use in this dictionary)

    num_turbines
        (int) Number of turbines for project

    project_size_megawatts
        (float) Total power output of project in megawatts

    hub_height_meters
        (float) Hub height for all turbines in meters

    project_size_megawatts

        (float) Total power output of project in megawatts

    num_access_roads
        (int) Number of access roads into project site

    site_facility_building_area_df
        (pd.DataFrame) Building areas dataframe. This should be loaded
        from a .csv file on the filesystem.

    The OUTPUT keys are the following

    insurance
        (float) The cost of insurance for the project. USD. Eqn. 3.4.1

    construction_permitting
        (float) The cost of the construction permitting. USD.

    project_management
        (float) The cost of the project management. USD.

    bonding
        (float) The cost of binding for the project. USD.

    engineering_usd
        (float) Site-specific engineering costs for foundations and collection. USD.

    site_security_usd
        (float) estimate cost of site security. USD.

    site_facility
        (float) Site facility costs. USD.

    markup_contingency
        (float) Markup contingency costs. USD.

    total_cost
        (float) Total cost of everything else
    """

    def __init__(self, input_dict, output_dict, project_name):
        """
        This method runs all cost calculations in the model based on the
        inputs key value pairs in the dictionary that is the only
        argument.

        Parameters
        ----------
        input_dict : dict
            Dictionary with the inputs key / value pairs

        output_dict : dict
            Dictionary with output key / value pairs.
        """
        self.in_distributed_mode = 'override_total_management_cost' in input_dict
        self.validate_inputs(input_dict)
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.project_name = project_name

    def validate_inputs(self, input_dict):
        """
        This method checks a dictionary to make sure it has keys for all
        necessary values needed for calculations in an instance of this
        class. It is made to validate input_dict dictionaries.

        Returns
        -------
        None
            This method returns nothing if the validation passes. If it does
            not pass an exception is raised.

        Raises
        ------
        ValueError
            If one of the keys is missing, this method raises a ValueError
        """
        if not self.in_distributed_mode:
            required_keys = {
                'project_value_usd',
                'foundation_cost_usd',
                'construct_duration',
                'num_hwy_permits',
                'num_turbines',
                'project_size_megawatts',
                'hub_height_meters',
                'num_access_roads',
                'markup_contingency',
                'markup_warranty_management',
                'markup_sales_and_use_tax',
                'markup_overhead',
                'markup_profit_margin',
                'site_facility_building_area_df'
            }
            found_keys = set(input_dict.keys())
            if len(required_keys - found_keys) > 0:
                err_msg = '{}: did not find all required keys in inputs dictionary. Missing keys are {}'
                raise ValueError(err_msg.format(type(self).__name__, required_keys - found_keys))

    def insurance(self):
        """
        Calculate insurance costs based on project value, builder size, and project size;
        equation based on empirical data from industry.

        Eqn. 3.4.1

        Includes:

        Builder's risk

        General liability

        Umbrella policy

        Professional liability

        It uses only the self.project_value attribute in calculations.

        :math:`C_I = 0.0056 * V_p`

        Returns
        -------
        float
            Insurance costs in USD
        """
        insurance_cost = 0.0056 * self.input_dict['project_value_usd']
        return insurance_cost

    def construction_permitting(self):
        """
        Calculate construction permitting costs based empirical data from industry.
        Includes building and highway permits

        Returns
        -------
        float
            Construction permitting cost in USD
        """
        building_permits = 0.02 * self.input_dict['foundation_cost_usd']
        highway_permits = 20000 * self.input_dict['num_hwy_permits']
        construction_permitting_cost = building_permits + highway_permits
        return construction_permitting_cost

    def bonding(self):
        """
        Calculate bonding costs based on project size; equation based on empirical data from industry.

        Returns
        -------
        float
            Bonding cost in USD
        """
        # Calculate bonding costs based on project size
        performance_bond_cost = 0.01 * self.input_dict['project_value_usd']
        return performance_bond_cost

    def project_management(self):
        """
        Calculate project management costs based on project size; equation based on empirical data from industry.
        Includes:

        Project manager and assistant project manager for site

        Site managers for civil, electrical, and erection

        QA/QC management

        QA/QC inspections for civil, structural, electrical, and mechanical

        Administrative support for the site

        Health and safety supervisors

        Environmental supervisors

        Office equipment & materials

        Site radios, communication, and vehicles

        Management team per diem and travel

        Legal and public relations

        Returns
        -------
        float
            Project management cost
        """
        # todo: add relationship to site-specific interface with public infrastructure
        if self.output_dict['actual_construction_months'] < 28:
            project_management_cost = (53.333 * self.output_dict['actual_construction_months'] ** 2 -
                                       3442 * self.output_dict['actual_construction_months'] +
                                       209542) * (self.output_dict['actual_construction_months'] + 2)
        else:
            project_management_cost = (self.output_dict['actual_construction_months'] + 2) * 155000
        return project_management_cost

    def markup_contingency(self):
        """
        Calculate mark-up and contingency costs based on project value. Includes:

        Markup contingency

        Markup warranty management

        Sales and use tax

        Markup overhead

        Markup profit margin

        Returns
        -------
        float
            Mark up and contingency costs.
        """
        # Calculate mark-up and contingency costs based on project value
        markup_contingency_cost = (self.input_dict['markup_contingency']
                                   + self.input_dict['markup_warranty_management']
                                   + self.input_dict['markup_sales_and_use_tax']
                                   + self.input_dict['markup_overhead']
                                   + self.input_dict['markup_profit_margin']) * self.input_dict['project_value_usd']
        return markup_contingency_cost

    def engineering_foundations_collection_sys(self):
        """
        Calculate site-specific engineering costs for foundations and collection system
        based on empirical data from industry. Includes met masts and power performance.

        Returns
        -------
        float
            site-specific engineering costs
        """
        # development engineering costs for foundations and collection system
        if self.input_dict['project_size_megawatts'] < 200:
            development_engineering_cost = 7188.5 * self.input_dict['num_turbines'] + round(3.4893 * math.log(self.input_dict['num_turbines']) - 7.3049, 0) * 16800 + 165675
        else:
            development_engineering_cost = 7188.5 * self.input_dict['num_turbines'] + round(3.4893 * math.log(self.input_dict['num_turbines']) - 7.3049, 0) * 16800 + 327250

        # engineering costs for met masts
        # TODO: Projects less than 30 MW for met masts
        if 30 <= self.input_dict['project_size_megawatts'] <= 100:
            num_perm_met_mast = 2
            num_temp_met_mast = 2
        elif 100 < self.input_dict['project_size_megawatts'] <= 300:
            num_perm_met_mast = 2
            num_temp_met_mast = 4
        elif self.input_dict['project_size_megawatts'] > 300:
            num_perm_met_mast = round(self.input_dict['project_size_megawatts'] / 100)
            num_temp_met_mast = round(self.input_dict['project_size_megawatts'] / 100) * 2
        else:
            num_perm_met_mast = 1
            num_temp_met_mast = 1

        if self.input_dict['hub_height_meters'] < 90:
            multiplier_perm = 232600
            multiplier_temp = 92600
        else:
            multiplier_perm = 290000
            multiplier_temp = 116800

        met_mast_cost = (num_perm_met_mast * multiplier_perm) + (num_temp_met_mast * multiplier_temp) + 200000

        engineering_cost = development_engineering_cost + met_mast_cost

        return engineering_cost

    def site_facility(self):
        """
        Uses empirical data to estimate cost of site facilities and security, including


        Site facilities:

        Building design and construction

        Drilling and installing a water well, including piping

        Electric power for a water well

        Septic tank and drain field


        Site security:

        Constructing and reinstating the compound

        Constructing and reinstating the batch plant site

        Setting up and removing the site offices for the contractor, turbine supplier, and owner

        Restroom facilities

        Electrical and telephone hook-up

        Monthly office costs

        Signage for project information, safety and directions

        Cattle guards and gates

        Number of access roads



        In main.py, a csv is loaded into a Pandas dataframe. The columns of the
        dataframe must be:

        Size Min (MW)
            Minimum power output for a plant that needs a certain size of
            building.

        Size Max (MW)
            Maximum power output of a plant that need a certain size of
            building.

        Building Area (sq. ft.)
            The area of the building needed to provide O & M support to plants
            with power output between "Size Min (MW)" and "Size Max (MW)".

        Returns
        -------
        float
            Building area in square feet
        """
        df = self.input_dict['site_facility_building_area_df']
        project_size_megawatts = self.input_dict['project_size_megawatts']
        row = df[(df['Size Max (MW)'] > project_size_megawatts) & (df['Size Min (MW)'] <= project_size_megawatts)]
        building_area_sq_ft = float(row['Building area (sq. ft.)'].iloc[0])

        construction_building_cost = building_area_sq_ft * 125 + 176125

        ps = self.input_dict['project_size_megawatts']
        ct = self.output_dict['actual_construction_months']
        nt = self.input_dict['num_turbines']
        if nt < 30:
            nr = 1
            acs = 30000
        elif nt < 100:
            nr = round(0.05 * nt)
            acs = 240000
        else:
            nr = round(0.05 * nt)
            acs = 390000
        compound_security_cost = 9825 * nr + 29850 * ct + acs + 60 * ps + 62400

        site_facility_cost = construction_building_cost + compound_security_cost
        return site_facility_cost


    def total_management_cost(self):
        """
        Calculates the total cost of returned by the rest of the methods.
        This must be called after all the other methods in the module.

        This method uses the outputs for all the other individual costs as
        set in self.output_dict. It does not call the other methods directly.

        Returns
        -------
        float
            Total management cost as summed from the outputs of all
            other methods.
        """
        total = 0
        total += self.output_dict['insurance_usd']
        total += self.output_dict['construction_permitting_usd']
        total += self.output_dict['bonding_usd']
        total += self.output_dict['project_management_usd']
        total += self.output_dict['markup_contingency_usd']
        total += self.output_dict['engineering_usd']
        total += self.output_dict['site_facility_usd']
        return total

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
        if self.in_distributed_mode:
            row = {
                'project_id_with_serial': self.project_name,
                'module': type(self).__name__,
                'type': 'variable',
                'variable_df_key_col_name': 'total_management_cost',
                'unit': 'usd',
                'value': self.output_dict['total_management_cost']
            }
            result.append(row)
        else:
            management_cost_keys = [
                'insurance_usd',
                'construction_permitting_usd',
                'bonding_usd',
                'project_management_usd',
                'markup_contingency_usd',
                'engineering_usd',
                'site_facility_usd'
            ]

            for key in management_cost_keys:
                value = self.output_dict[key]
                row = {
                    'project_id_with_serial': self.project_name,
                    'module': type(self).__name__,
                    'type': 'variable',
                    'variable_df_key_col_name': key,
                    'unit': 'usd',
                    'value': value
                }
                result.append(row)

        return result

    def outputs_for_module_type_operation(self):
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
        turbine_rating_MW = self.input_dict['turbine_rating_MW']
        num_turbines = self.input_dict['num_turbines']
        project_size_kw = num_turbines * turbine_rating_MW * 1000

        if self.in_distributed_mode:
            result.append({
                'type_of_cost': 'total_management_cost',
                'raw_cost': self.output_dict['total_management_cost']
            })

        else:
            result.append({
                'type_of_cost': 'insurance',
                'raw_cost': self.output_dict['insurance_usd']
            })
            result.append({
                'type_of_cost': 'Construction Permitting',
                'raw_cost': self.output_dict['construction_permitting_usd']
            })
            result.append({
                'type_of_cost': 'Project Management',
                'raw_cost': self.output_dict['project_management_usd']
            })
            result.append({
                'type_of_cost': 'Bonding',
                'raw_cost': self.output_dict['bonding_usd']
            })
            result.append({
                'type_of_cost': 'Markup Contingency',
                'raw_cost': self.output_dict['markup_contingency_usd']
            })
            result.append({
                'type_of_cost': 'Engineering Foundation and Collections System (includes met mast)',
                'raw_cost': self.output_dict['engineering_usd']
            })
            result.append({
                'type_of_cost': 'Site Facility',
                'raw_cost': self.output_dict['site_facility_usd']
            })

        for _dict in result:
            _dict['turbine_rating_MW'] = self.input_dict['turbine_rating_MW']
            _dict['num_turbines'] = self.input_dict['num_turbines']
            _dict['rotor_diameter_m'] = self.input_dict['rotor_diameter_m']
            _dict['project_id_with_serial'] = self.project_name
            _dict['operation_id'] = 'Management'
            _dict['module'] = module
            _dict['raw_cost_total_or_per_turbine'] = 'total'
            _dict['cost_per_turbine'] = _dict['raw_cost'] / num_turbines
            _dict['cost_per_project'] = _dict['raw_cost']
            _dict['usd_per_kw_per_project'] = _dict['raw_cost'] / project_size_kw

        return result

    def run_module(self):
        """
        Runs all the calculation methods in order.

        Parameters
        ----------
        project_name : str
            The name of the project for which this calculation is being done.

        Returns
        -------
        tuple
            First element of tuple contains a 0 or 1. 0 means no errors happened and
            1 means an error happened and the module failed to run. The second element
            depends on the error condition. If the condition is 1, then the second
            element is the error raised that caused the failure. If the condition is
            0 then the second element is 0 as well.
        """
        try:
            if self.in_distributed_mode:
                self.output_dict['insurance_usd'] = 0
                self.output_dict['construction_permitting_usd'] = 0
                self.output_dict['project_management_usd'] = 0
                self.output_dict['bonding_usd'] = 0
                self.output_dict['markup_contingency_usd'] = 0
                self.output_dict['engineering_usd'] = 0
                self.output_dict['site_facility_usd'] = 0
                self.output_dict['total_management_cost'] = self.input_dict['override_total_management_cost']

            else:
                self.output_dict['insurance_usd'] = self.insurance()
                self.output_dict['construction_permitting_usd'] = self.construction_permitting()
                self.output_dict['project_management_usd'] = self.project_management()
                self.output_dict['bonding_usd'] = self.bonding()
                self.output_dict['markup_contingency_usd'] = self.markup_contingency()
                self.output_dict['engineering_usd'] = self.engineering_foundations_collection_sys()
                self.output_dict['site_facility_usd'] = self.site_facility()
                self.output_dict['total_management_cost'] = self.total_management_cost()
            self.output_dict['management_cost_csv'] = self.outputs_for_detailed_tab()
            self.output_dict['mangement_module_type_operation'] = self.outputs_for_module_type_operation()
            return 0, 0    # module ran successfully
        except Exception as error:
            traceback.print_exc()
            print(f"Fail {self.project_name} ManagementCost")
            return 1, error  # module did not run successfully
