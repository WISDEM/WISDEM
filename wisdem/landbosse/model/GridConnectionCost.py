import traceback
import pandas as pd
import math


from .CostModule import CostModule


class GridConnectionCost(CostModule):
    """
    TransDistCost.py
     - Created by Annika Eberle and Owen Roberts on Dec. 17, 2018
     - Refactored by Parangat Bhaskar and Alicia Key on June 3, 2019

    Calculates the costs associated with transmission and distribution for land-based wind projects (module is currently based on curve fit of empirical data)

    * Get distance to interconnection
    * Get interconnection voltage
    * Get toggle for new switchyard
    * Return total transmission and distribution costs

    \n\n**Keys in the input dictionary are the following:**

    distance_to_interconnect_mi
        (float) distance to interconnection [in miles]

    \n\n**Keys in the output dictionary are the following:**

    trans_dist_usd
        (float) total transmission and distribution costs [in USD]

    """

    def __init__(self, input_dict, output_dict , project_name):
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

    def calculate_costs(self, calculate_costs_input_dict, calculate_costs_output_dict):
        """
        Function to calculate total costs for transmission and distribution.

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

        if calculate_costs_input_dict['distance_to_interconnect_mi'] == 0:
            calculate_costs_output_dict['trans_dist_usd'] = 0
        else:
            if calculate_costs_input_dict['new_switchyard'] is True:
                calculate_costs_output_dict['interconnect_adder_USD'] = 18115 * self.input_dict['interconnect_voltage_kV'] + 165944
            else:
                calculate_costs_output_dict['interconnect_adder_USD'] = 0
            calculate_costs_output_dict['trans_dist_usd'] = ((1176 * self.input_dict['interconnect_voltage_kV'] + 218257) * (calculate_costs_input_dict['distance_to_interconnect_mi'] ** (-0.1063)) * calculate_costs_input_dict['distance_to_interconnect_mi']) + calculate_costs_output_dict['interconnect_adder_USD']


        calculate_costs_output_dict['trans_dist_usd_df'] = pd.DataFrame([['Other', calculate_costs_output_dict['trans_dist_usd'], 'Transmission and Distribution']],
                     columns=['Type of cost', 'Cost USD', 'Phase of construction'])

        calculate_costs_output_dict['total_transdist_cost'] = calculate_costs_output_dict['trans_dist_usd_df']

        return calculate_costs_output_dict['trans_dist_usd_df']


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
        result = []
        module = type(self).__name__

        for row in self.output_dict['trans_dist_usd_df'].itertuples():
            dashed_row = '{} <--> {} <--> {}'.format(row[1], row[3], math.ceil(row[2]))
            result.append({
                'unit': '',
                'type': 'dataframe',
                'variable_df_key_col_name': 'Type of Cost <--> Phase of Construction <--> Cost in USD ',
                'value': dashed_row,
                'last_number': row[2]
            })

        for _dict in result:
            _dict['project_id_with_serial'] = self.project_name
            _dict['module'] = module

        self.output_dict['trans_dist_cost_csv'] = result
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

        costs_by_module_type_operation = output_dict['trans_dist_usd_df']
        for _, row in costs_by_module_type_operation.iterrows():
            _dict = dict()
            row = row.to_dict()
            _dict['operation_id'] = row['Phase of construction']
            _dict['type_of_cost'] = row['Type of cost']
            _dict['cost'] = row['Cost USD']
            result.append(_dict)

        for _dict in result:
            _dict['project_id_with_serial'] = self.project_name
            _dict['module'] = module
            _dict['total_or_turbine'] = 'total'

        output_dict['trans_dist_cost_module_type_operation'] = result
        return result

    def run_module(self):
        """
        Runs the TransDistCost module and populates the IO dictionaries with calculated values.

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
            self.calculate_costs(self.input_dict, self.output_dict)
            self.outputs_for_detailed_tab(self.input_dict, self.output_dict)
            self.output_dict['trans_dist_cost_module_type_operation'] = \
                self.outputs_for_costs_by_module_type_operation(input_df=self.output_dict['trans_dist_usd_df'],
                                                                project_id=self.project_name,
                                                                total_or_turbine=True)
            return 0, 0 # module ran successfully
        except Exception as error:
            traceback.print_exc()
            print(f"Fail {self.project_name} GridConnectionCost")
            return 1, error # module did not run successfully

