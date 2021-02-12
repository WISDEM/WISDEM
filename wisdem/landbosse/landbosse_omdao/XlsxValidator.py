import pandas as pd


class XlsxValidator:
    """
    XlsxValidator is for comparing the results of a previous model run
    to the results of a current model run.
    """

    def compare_expected_to_actual(self, expected_xlsx, actual_module_type_operation_list, validation_output_xlsx):
        """
        This compares the expected costs as calculated by a prior model run
        with the actual results from a current model run.

        It compares the results row by row and prints any differences.

        Parameters
        ----------
        expected_xlsx : str
            The absolute filename of the expected output .xlsx file.

        actual_module_type_operation_list : str
            The module_type_operation_list as returned by a subclass of
            XlsxManagerRunner.

        validation_output_xlsx : str
            The absolute pathname to the output file with the comparison
            results.

        Returns
        -------
        bool
            True if the expected and actual results are equal. It returns
            False otherwise.
        """
        # First, make the list of dictionaries into a dataframe, and drop
        # the raw_cost and raw_cost_total_or_per_turbine columns.
        actual_df = pd.DataFrame(actual_module_type_operation_list)
        actual_df.drop(["raw_cost", "raw_cost_total_or_per_turbine"], axis=1, inplace=True)
        expected_df = pd.read_excel(expected_xlsx, "costs_by_module_type_operation", engine="openpyxl")
        # expected_df = expected_df.dropna(inplace=True, how='all')
        expected_df.rename(
            columns={
                "Project ID with serial": "project_id_with_serial",
                "Number of turbines": "num_turbines",
                "Turbine rating MW": "turbine_rating_MW",
                "Module": "module",
                "Operation ID": "operation_id",
                "Type of cost": "type_of_cost",
                "Cost per turbine": "cost_per_turbine",
                "Cost per project": "cost_per_project",
                "USD/kW per project": "usd_per_kw_per_project",
            },
            inplace=True,
        )

        cost_per_project_actual = actual_df[
            ["cost_per_project", "project_id_with_serial", "module", "operation_id", "type_of_cost"]
        ]
        cost_per_project_expected = expected_df[
            ["cost_per_project", "project_id_with_serial", "module", "operation_id", "type_of_cost"]
        ]

        comparison = cost_per_project_actual.merge(
            cost_per_project_expected, on=["project_id_with_serial", "module", "operation_id", "type_of_cost"]
        )

        comparison.rename(
            columns={
                "cost_per_project_x": "cost_per_project_actual",
                "cost_per_project_y": "cost_per_project_expected",
            },
            inplace=True,
        )

        comparison["difference_validation"] = (
            comparison["cost_per_project_actual"] - comparison["cost_per_project_expected"]
        )

        # Regardless of the outcome, write the end result of the comparison
        # to the validation output file.
        columns_for_comparison_output = [
            "project_id_with_serial",
            "module",
            "operation_id",
            "type_of_cost",
            "cost_per_project_actual",
            "cost_per_project_expected",
            "difference_validation",
        ]
        comparison.to_excel(validation_output_xlsx, index=False, columns=columns_for_comparison_output)

        # If the comparison dataframe is empty, that means there are no common
        # projects in the expected data that match the actual data.
        if len(comparison) < 1:
            print("=" * 80)
            print("Validation error: There are no common projects between actual and expected data.")
            print("=" * 80)
            return False

        # Find all rows where the difference is unequal to 0. These are rows
        # that failed validation. Note that, after the join, the rows may be
        # in a different order than the originals.
        #
        # Round the difference to a given number of decimal places.
        failed_rows = comparison[comparison["difference_validation"].round(decimals=4) != 0]

        if len(failed_rows) > 0:
            print("=" * 80)
            print("The following rows failed validation:")
            print(failed_rows)
            print("=" * 80)
            return False
        else:
            return True
