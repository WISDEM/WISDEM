import pandas as pd

from .XlsxDataframeCache import XlsxDataframeCache
from .XlsxFileOperations import XlsxFileOperations
from .XlsxReader import XlsxReader


class XlsxManagerRunner:
    """
    This class runs a new instance of Manager for each project listed
    in the project_xlsx file.

    This class is meant to be subclassed depending on whether a serial
    or parallel manager runner is needed.
    """

    def __init__(self, file_ops=None):
        """
        The constructor simply creates an XlsxFileOperations instance
        to live throughout the lifetime of the instance

        Parameters
        ----------
        file_ops : XlsxFileOperations
            The file operation instance used to create filenames. If this
            is left at the default of None, a new instance of
            XlsxFileOperations is created.
        """
        self.file_ops = file_ops if file_ops is not None else XlsxFileOperations()

    def run_from_project_list_xlsx(self, projects_xlsx,  enable_cost_and_scaling_modifications=True):
        """
        This function runs all the scenarios in the projects_xlsx file. It creates
        the OrderedDict that holds the results of all the runs. See the return
        section below for more details on what the OrderedDict contains.

        This method is meant to be overriden by subclasses. If this method is
        called directly on this class, a NotImplementedError is raised. However, this
        docstring does specify the contract that the inheriting class should raise.

        Parameters
        ----------
        projects_xlsx : str
            A path name (preferably created with os.path.join()) specific to the
            operating system that is the main input .xlsx file that controls
            running of all the projects. Crucially, this file contains names of
            other. It is recommended that all input file be kept in the same
            input directory. Each line of projects_xlsx becomes a project_series.
            This is implemented by subclasses.

        enable_cost_and_scaling_modifications : bool
            If True, this method modifies each row of the project list AFTER it has been
            modified by the parameters for to scale certain input values based
            on what has been parametrically modified. This is implemented by subclasses.

        Returns
        -------
        OrderedDict or dict, list
            First element of tuple is an ordered dict that is the result of
            all the runs. Each key is the name of a project and each value
            is the output dictionary of that project. The second element
            is the list of rows for the csv. The third element is the list
            of costs for the spreadsheets. For the first element, serial
            processes implement an OrderedDict (because order can be guaranteed)
            and for the second element processes implement a dict (because
            order cannot be guaranteed).

        Raises
        ------
        NotImplementedError
            NotImplementedError is raised if the method is called on the
            superclass.
        """
        raise NotImplementedError('run_from_project_list_xlsx() can only be called on subclasses')

    def extract_module_type_operation_lists(self, runs_dict):
        """
        This method extract all the cost_by_module_type_operation lists for
        output in an Excel file.

        It finds values for the keys ending in '_module_type_operation'. It
        then concatenates them
        together so they can be easily written to a .csv or .xlsx

        Parameters
        ----------
        runs_dict : dict
            Values are the names of the projects. Keys are the lists of
            dictionaries that are lines for the .csv

        Returns
        -------
        list
            List of dicts to write to the .csv.
        """
        result = []
        for project_results in runs_dict.values():
            for key, value in project_results.items():
                if key.endswith('_module_type_operation'):
                    result.extend(value)
        return result

    def extract_details_lists(self, runs_dict):
        """
        This method extract all .csv lists from the OrderDict of runs to output
        into an Excel or .csv file.

        It finds values for the keys ending in '_csv'. It then concatenates them
        together so they can be easily written to a .csv, .xlsx or other
        columnar format. (The actual writing is left to other functions.

        Parameters
        ----------
        runs_dict : dict
            Values are the names of the projects. Keys are the lists of
            dictionaries that are lines for the .csv

        Returns
        -------
        list
            List of dicts to write to the .csv.
        """
        runs_for_csv = []
        for project_results in runs_dict.values():
            for key, value in project_results.items():
                if key.endswith('_csv'):
                    runs_for_csv.extend(value)
        return runs_for_csv

    def read_project_and_parametric_list_from_xlsx(self):
        """
        This method reads both the project and parametric list from the
        project_list xlsx. It returns them as a tuple.

        To ensure backward compatibility with project list spreadsheets
        that contain only one sheet, the following actions take place

        1. Project list spreadsheet contains one sheet: The only sheet
        that is present is read as the project list, and a dataframe for
        parametrics is returned that has all the columns but no rows so
        it can be used in joins. It doe snot matter what the sheet name
        is.

        2. Project list spreadsheet contains two sheets: The sheet named
        "Project list" is assumed to be the project list and the sheet
        named "Parametric list" is assumed to be the specifications for
        the parametric variables. If one or both of those tab names are
        missing, an exception is raised

        Returns
        -------
        pandas.DataFrame
            The enhanced project list that has support for all parametric
            adjustments for each step.

        Raises
        ------
        KeyError
            When the spreadsheet contains multiple sheets and one or
            both of "Project list" or "Parametric list" are undefined.
        """
        path_to_project_list = self.file_ops.landbosse_input_dir()
        sheets = XlsxDataframeCache.read_all_sheets_from_xlsx('project_list', path_to_project_list)

        # If there is one sheet, make an empty dataframe as a placeholder.
        if len(sheets.values()) == 1:
            first_sheet = list(sheets.values())[0]
            project_list = first_sheet
            parametric_list = pd.DataFrame()

        # If the parametric and project lists exist, read them
        elif 'Parametric list' in sheets.keys() and 'Project list' in sheets.keys():
            project_list = sheets['Project list']
            parametric_list = sheets['Parametric list']

        # Otherwise, raise an exception
        else:
            raise KeyError("Project list needs to have a single sheet or sheets named 'Project list' and 'Parametric list'.")

        # Instantiate and XlsxReader to assemble master input dictionary
        xlsx_reader = XlsxReader()

        # Join in the parametric variable modifications
        parametric_value_list = xlsx_reader.create_parametric_value_list(parametric_list)
        extended_project_list = xlsx_reader.outer_join_projects_to_parametric_values(project_list,
                                                                                 parametric_value_list)

        return extended_project_list
