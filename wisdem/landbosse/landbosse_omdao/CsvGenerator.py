import pandas as pd


class CsvGenerator:
    """
    This class generates CSV files.
    """

    def __init__(self, file_ops):
        """
        Parameters
        ----------
        file_ops : XlsxFileOperations
            An instance of XlsxFileOperations to manage file names.
        """
        self.file_ops = file_ops

    def create_details_dataframe(self, details):
        """
        This writes the details .csv.

        Parameters
        ----------
        details : list[dict]
            A list of dictionaries to be converted into a Pandas dataframe

        Returns
        -------
        pd.DataFrame
            The dataframe that can be written to a .csv file.
        """

        # This the list of details to write to the .csv
        details_to_write_to_csv = []
        for row in details:
            new_row = {}
            new_row["Project ID with serial"] = row["project_id_with_serial"]
            new_row["Module"] = row["module"]
            new_row["Variable name"] = row["variable_df_key_col_name"]
            new_row["Unit"] = row["unit"]

            value = row["value"]
            value_is_number = self._is_numeric(value)
            if value_is_number:
                new_row["Numeric value"] = value
            else:
                new_row["Non-numeric value"] = value

            # If there is a last_number, which means this is a dataframe row that has a number
            # at the end, write this into the numeric value column. This overrides automatic
            # type detection.

            if "last_number" in row:
                new_row["Numeric value"] = row["last_number"]

            details_to_write_to_csv.append(new_row)

        details = pd.DataFrame(details_to_write_to_csv)

        return details

    def create_costs_dataframe(self, costs):
        """
        Parameters
        ----------
        costs : list[dict]
            The list of dictionaries of costs.

        Returns
        -------
        pd.DataFrame
            A dataframe to be written as a .csv
        """
        new_rows = []
        for row in costs:
            new_row = {
                "Project ID with serial": row["project_id_with_serial"],
                "Number of turbines": row["num_turbines"],
                "Turbine rating MW": row["turbine_rating_MW"],
                "Rotor diameter m": row["rotor_diameter_m"],
                "Module": row["module"],
                "Type of cost": row["type_of_cost"],
                "Cost per turbine": row["cost_per_turbine"],
                "Cost per project": row["cost_per_project"],
                "Cost per kW": row["usd_per_kw_per_project"],
            }
            new_rows.append(new_row)
        costs_df = pd.DataFrame(new_rows)
        return costs_df

    def _is_numeric(self, value):
        """
        This method tests if a value is a numeric (that is, can be parsed
        by float()) or non numeric (which cannot be parsed).

        The decision from this method determines whether values go into
        the numeric or non-numeric columns.

        Parameters
        ----------
        value
            The value to be tested.

        Returns
        -------
        bool
            True if the value is numeric, False otherwise.
        """
        try:
            float(value)
        except ValueError:
            return False
        return True
