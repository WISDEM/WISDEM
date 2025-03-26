from wisdem.landbosse.model.CostModule import CostModule

import traceback
import numpy as np
import pandas as pd

class TurbineCost(CostModule):
    """
    **TurbineCost.py**

    Calculates the CapEx cost of the turbine itself

    This is a simple feedthrough calculation that requires an input CapEx assumption ($/kW)

    The structure of this class contains many superfluous elements, but it has been made to match
    the other LandBOSSE cost modules

    turbine_capex
        (float) CapEx cost of turbine ($/kW)

    number_turbines
        (int) Number of turbines
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
        
    def calculate_costs(self, calculate_costs_input_dict , calculate_costs_output_dict):
        """
        Function to calculate turbine cost in USD

        Parameters
        -------
        turbine_capex
            (in $/kW, default $1500/kW)

        number_turbines
            (unitless)

        Returns:
        -------
        turbine_cost
            (in USD)

        """

        turbine_capex = calculate_costs_input_dict["turbine_capex"]
        if np.isnan(turbine_capex):
            turbine_capex = 0.0

        num_turbines = calculate_costs_input_dict["num_turbines"]
        turbine_rating_kW = calculate_costs_input_dict["turbine_rating_MW"] * 1000

        turbine_cost = turbine_capex * num_turbines * turbine_rating_kW

        calculate_costs_output_dict["turbine_cost"] = turbine_cost

        calculate_costs_output_dict["turbine_cost_output_df"] = pd.DataFrame(
            [["Other", calculate_costs_output_dict["turbine_cost"], "Turbine"]],
            columns=["Type of cost", "Cost USD", "Phase of construction"],
        )


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

        result.append({
            "unit": "usd",
            "type": "variable",
            "variable_df_key_col_name": "Turbine CapEx",
            "value": float(self.output_dict["turbine_cost"]),
        })

        for _dict in result:
            _dict["project_id_with_serial"] = self.project_name
            _dict["module"] = module

        self.output_dict["turbine_cost_csv"] = result

    def run_module(self):
        """
        Runs the TurbineCost module and populates the IO dictionaries with calculated values.

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
            self.output_dict["turbine_module_type_operation"] = self.outputs_for_costs_by_module_type_operation(
                input_df=self.output_dict["turbine_cost_output_df"],
                project_id=self.project_name,
                total_or_turbine=True
            )
            return 0, 0
        except Exception as error:
            traceback.print_exc()
            print(f"Fail {self.project_name} TurbineCost")
            return 1, error
