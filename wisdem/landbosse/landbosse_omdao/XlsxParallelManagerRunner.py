import os
from concurrent import futures

import pandas as pd

from ..model import Manager
from .XlsxFileOperations import XlsxFileOperations
from .XlsxReader import XlsxReader
from .XlsxManagerRunner import XlsxManagerRunner
from .XlsxDataframeCache import XlsxDataframeCache
from .XlsxGenerator import XlsxGenerator


class XlsxParallelManagerRunner(XlsxManagerRunner):
    """
    This subclass implementation of XlsxManagerRunner runs all projects
    with a ProcessPoolExecutor.
    """

    def run_from_project_list_xlsx(self, projects_xlsx, enable_cost_and_scaling_modifications=False):
        """
        This function runs all the scenarios in the projects_xlsx file. It creates
        the OrderedDict that holds the results of all the runs. See the return
        section below for more details on what the OrderedDict contains.

        This is a concrete implementation of the super class method.

        Parameters
        ----------
        projects_xlsx : str
            A path name (preferably created with os.path.join()) specific to the
            operating system that is the main input .xlsx file that controls
            running of all the projects. Crucially, this file contains names of
            other. It is recommended that all input file be kept in the same
            input directory.

        enable_cost_and_scaling_modifications : bool
            If True, this method modifies each row of the project list AFTER it has been
            modified by the parameters for to scale certain input values based
            on what has been parametrically modified.

        Returns
        -------
        dict, list, module_type_operation_lists
            First element of tuple is a dict that is the result of
            all the runs. Each key is the name of a project and each value
            is the output dictionary of that project. The second element
            is the list of rows for the csv. The third element is the list
            of costs for the spreadsheets.
        """
        # Load the project list
        print('Calculating parametric values')
        extended_project_list_before_parameter_modifications = self.read_project_and_parametric_list_from_xlsx()

        # Prepare the file operations
        file_ops = XlsxFileOperations()

        # Instantiate an XlsxReader to handle the parametrics and master input
        # dictionaries
        xlsx_reader = XlsxReader()

        # Get a list ready to hold the project parameters after they have been modified
        # After all rows have been added to this list (each row is a series) then the
        # whole list will be transformed into a dataframe.
        #
        # See notes at https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.append.html
        # for why this is more performant than appending to a dataframe.
        extended_project_list_after_parameter_modifications = []

        # Prep all task for the executor
        all_tasks = []
        print(f'Found {len(extended_project_list_before_parameter_modifications)} projects for execution')
        for _, project_parameters in extended_project_list_before_parameter_modifications.iterrows():

            # If project_parameters['Project ID with serial'] is null, that means there are no
            # parametric modifications to the project data dataframes. Hence,
            # just the plain Project ID without a serial number should be used.
            if pd.isnull(project_parameters['Project ID with serial']):
                project_id_with_serial = project_parameters['Project ID']
            else:
                project_id_with_serial = project_parameters['Project ID with serial']

            print(f'Preparing {project_id_with_serial}')

            project_data_basename = project_parameters['Project data file']
            task = dict()

            task['project_data_sheets'] = XlsxDataframeCache.read_all_sheets_from_xlsx(project_data_basename)

            # Transform the dataframes so that they have the right values for
            # the parametric variables.
            xlsx_reader.modify_project_data_and_project_list(task['project_data_sheets'], project_parameters)

            # Apply cost and scaling modifications if needed.
            if enable_cost_and_scaling_modifications:
                xlsx_reader.apply_cost_and_scaling_modifications_to_project_parameters(project_parameters)

            # Append the modified project parameters
            extended_project_list_after_parameter_modifications.append(project_parameters)

            # Write all project_data sheets
            parametric_project_data_path = \
                os.path.join(file_ops.parametric_project_data_output_path(), f'{project_id_with_serial}_project_data.xlsx')
            XlsxGenerator.write_project_data(task['project_data_sheets'], parametric_project_data_path)

            task['project_data_basename'] = project_data_basename
            task['project_id_with_serial'] = project_id_with_serial
            task['project_series'] = project_parameters
            all_tasks.append(task)

        # Execute every project
        with futures.ProcessPoolExecutor() as executor:
            executor_result = executor.map(run_single_project, all_tasks)

        # Get the output dictionary ready
        runs_dict = {project_id_with_serial: result for project_id_with_serial, result in executor_result}

        # Assemble the dictionary with content for the details, details with inputs,
        #  cost_by_module_type_operation and cost_by_module_type_operation_with_input tabs
        final_result = dict()
        final_result['details_list'] = self.extract_details_lists(runs_dict)
        final_result['module_type_operation_list'] = self.extract_module_type_operation_lists(runs_dict)
        final_result['extended_project_list'] = pd.DataFrame(extended_project_list_after_parameter_modifications)

        # Return the runs for all the scenarios.
        return final_result


"""
The following function is deliberately defined outside of the class.
This makes it easier to think about it being a pure function for
parallel processes.
"""


def run_single_project(task_dict):
    """
    The dictionary project_definition_dict contains the following keys.

    For each process another logger is created, so that each process does
    not attempt to use the same logger.

    input_xlsx : str
        The filename for the input .xlsx that has all the dataframes
        for the for ErectionCost and FoundationCost

    project_series : pd.Series
        The series that has the non-dataframe values for each project,
        including the project name.

    project_id : str
        The string that is the name of the project.

    Basically, the map operation goes like this:

    task_dict -> master_input_dict -> master_output_dict

    Wrapped in a functional executor, this maps projects into their
    output dictionaries.

    Parameters
    ----------
    task_dict : dict
        The configuration of the task.

    Returns
    -------
    tuple : (str, dict)
        The str is the project_id. The dict is the resulting output
        dictionary.
    """
    project_data_basename = task_dict['project_data_basename']
    project_series = task_dict['project_series']
    project_id_with_serial = task_dict['project_id_with_serial']
    project_data_sheets = task_dict['project_data_sheets']

    # Log each project. Use print because it works better for multiple processes.
    print(f'Start {project_id_with_serial}, project data in {project_data_basename}')

    # Read the Excel
    xlsx_reader = XlsxReader()
    master_input_dict = xlsx_reader.create_master_input_dictionary(project_data_sheets, project_series)

    # Now run the manager and accumulate its result into the runs_dict
    output_dict = dict()
    output_dict['project_series'] = project_series
    mc = Manager(input_dict=master_input_dict, output_dict=output_dict)
    mc.execute_landbosse(project_name=project_id_with_serial)

    print(f'End {project_id_with_serial}')

    return project_id_with_serial, output_dict
