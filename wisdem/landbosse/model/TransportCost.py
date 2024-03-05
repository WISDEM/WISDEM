import traceback

import pandas as pd
import scipy.interpolate
from wisdem.landbosse.model.CostModule import CostModule
import numpy as np

class TransportCost(CostModule):
    """
    **TransportCost.py**

    Calculates the costs associated with transportation for land-based wind projects *(module is currently based on curve fit of empirical data)*

    Get number of blades, nacelles

    Return total trasnport costs

    \n\n**Keys in the input dictionary are the following:**

    rotor_diameter_m
        (float) Determines the approximate blade length [in m]

    number_turbines
        (int) Number of turbines

    \n\n**Keys in the output dictionary are the following:**

    transport_cost
        (float) cost of transportation [in USD]


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

    def calculate_costs(self, calculate_costs_input_dict, calculate_costs_output_dict):
        """
        Function to calculate Transport Cost in USD

        Parameters
        -------
        interconnect_voltage_kV
            (in kV)

        project_size_megawatts
            (in MW)


        Returns:
        -------
        substation_cost
            (in USD)

        """

        blade_length = 0.5 * calculate_costs_input_dict["rotor_diameter_m"]
        n_turb = calculate_costs_input_dict["num_turbines"]
        # Transport cost is $/blade * nblades + infrastructure costs + cost for tower & nacelle
        # Blade transport from emp.lbl.gov/publications/supersized-wind-turbine-blade-study
        xlen = np.array([-500., 0., 65., 75., 95., 115.])
        ycost = 1e3 * np.array([0.0, 0.0, 52., 70., 120., 171.])
        yinfra = 1e6 * np.array([0.0, 0.0, 0.0, 0.2, 1.0, 5.0])

        f_blade = scipy.interpolate.interp1d(xlen, ycost, fill_value='extrapolate', assume_sorted=True)
        cost_per_blade = f_blade(blade_length)

        f_infra = scipy.interpolate.interp1d(xlen, yinfra, fill_value='extrapolate', assume_sorted=True)
        cost_infra = f_infra(blade_length)

        # Multiply by 4x for 3 blades + 1 tower
        calculate_costs_output_dict["transport_cost_usd"] = 4 * cost_per_blade * n_turb + cost_infra
        
        calculate_costs_output_dict["transport_cost_output_df"] = pd.DataFrame(
            [["Other", calculate_costs_output_dict["transport_cost_usd"], "Transport"]],
            columns=["Type of cost", "Cost USD", "Phase of construction"],
        )

        calculate_costs_output_dict["total_transport_cost"] = calculate_costs_output_dict["transport_cost_output_df"]

        return calculate_costs_output_dict["transport_cost_output_df"]

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

        for row in self.output_dict["transport_cost_output_df"].itertuples():
            dashed_row = "{} <--> {} <--> {}".format(row[1], row[3], np.ceil(row[2]))
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

        self.output_dict["transport_cost_csv"] = result
        return result

    def run_module(self):
        """
        Runs the TransportCost module and populates the IO dictionaries with calculated values.

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
            # self.outputs_for_module_type_operation(self.input_dict, self.output_dict)
            self.output_dict["transport_module_type_operation"] = self.outputs_for_costs_by_module_type_operation(
                input_df=self.output_dict["transport_cost_output_df"],
                project_id=self.project_name,
                total_or_turbine=True,
            )
            return 0, 0
        except Exception as error:
            traceback.print_exc()
            print(f"Fail {self.project_name} TransportCost")
            return 1, error
