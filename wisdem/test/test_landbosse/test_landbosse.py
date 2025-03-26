import openmdao.api as om
import pandas as pd
import pytest

from wisdem.landbosse.landbosse_omdao.landbosse import LandBOSSE
from wisdem.landbosse.landbosse_omdao.OpenMDAODataframeCache import OpenMDAODataframeCache


@pytest.fixture
def landbosse_costs_by_module_type_operation():
    """
    Executes LandBOSSE and extracts cost output for the regression
    test.
    """
    prob = om.Problem(reports=False)
    prob.model = LandBOSSE()
    prob.setup()
    prob.run_model()
    # prob.model.list_inputs(units=True)
    landbosse_costs_by_module_type_operation = prob["landbosse_costs_by_module_type_operation"]
    return landbosse_costs_by_module_type_operation


def compare_expected_to_actual(expected_df, actual_module_type_operation_list, validation_output_csv):
    """
    This compares the expected costs as calculated by a prior model run
    with the actual results from a current model run.

    It compares the results row by row and prints any differences.

    Parameters
    ----------
    expected_df : pd.DataFrame
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

    columns_to_compare = ["Cost / project", "Project ID with serial", "Module", "Type of cost"]
    cost_per_project_actual = actual_df[columns_to_compare]
    cost_per_project_expected = expected_df[columns_to_compare]

    comparison = cost_per_project_actual.merge(
        cost_per_project_expected, on=["Project ID with serial", "Module", "Type of cost"]
    )

    comparison.rename(
        columns={"Cost / project_x": "Cost / project actual", "Cost / project_y": "Cost / project expected"},
        inplace=True,
    )

    comparison["% delta"] = (comparison["Cost / project actual"] / comparison["Cost / project expected"] - 1) * 100

    comparison.to_csv(validation_output_csv, index=False)

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
    failed_rows = comparison[~pd.isnull(comparison["% delta"]) & comparison["% delta"].round(decimals=4) != 0]

    if len(failed_rows) > 0:
        print("=" * 80)
        print("The following rows failed validation:")
        print(failed_rows)
        print("=" * 80)
        return False
    else:
        return True


def test_landbosse(landbosse_costs_by_module_type_operation):
    """
    This runs the regression test by comparing against the expected validation
    data.
    """
    OpenMDAODataframeCache._cache = {}  # Clear the cache
    expected_validation_data_sheets = OpenMDAODataframeCache.read_all_sheets_from_xlsx("ge15_expected_validation")
    costs_by_module_type_operation = expected_validation_data_sheets["costs_by_module_type_operation"]
    result = compare_expected_to_actual(
        costs_by_module_type_operation, landbosse_costs_by_module_type_operation, "test.csv"
    )
    assert result
