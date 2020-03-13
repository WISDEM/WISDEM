import openmdao.api as om


class LandBOSSEBaseComponent(om.ExplicitComponent):
    """
    This is a superclass for all the components that wrap LandBOSSE
    cost modules. It holds functionality used for the other components
    that wrap LandBOSSE cost modules.

    This component should not be instantiated directly.
    """

    def initialize(self):
        """
        There is one option for this component: verbosity. If it is
        set to true, the component will print the summary of costs
        with print() after it finishes calculating them.
        """
        self.options.declare('verbosity', default=True)

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

