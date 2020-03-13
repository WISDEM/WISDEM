import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell
import pandas as pd
import os
import traceback

from .XlsxFileOperations import XlsxFileOperations


class XlsxGenerator:
    """
    This class is for writing data to Excel files. It is a context manager
    so it used in the following manner:

    with XlsxGenerator(output_xlsx) as xlsx:
        xlsx.joined_with_validation(...)

    Each method call on the context manager makes one or more tabs on the
    output Excel workbook.
    """

    def __init__(self, output_xlsx, file_ops):
        """
        This constructor sets the name of the .xlsx file for writing

        It sets the parameter self.workbook, which is the attribute that
        references the workbook to which all tabs should be written.

        Parameters
        ----------
        output_xlsx : str
            The name of the .xlsx file to write. Do not include the .xlsx at
            the end of the filename. Also, this filename will be timestamped
            before it is written.

        file_ops : XlsxFileOperations
            An instance of XlsxFileOperations to manage file names.
        """

        # Set all instance attributes to None first in the constructor as good
        # coding practice.
        self.workbook = None
        self.header_format = None
        self.scientific_format = None
        self.percent_format = None
        self.output_xlsx_path = os.path.join(file_ops.landbosse_output_dir(), f'{output_xlsx}.xlsx')
        self.file_ops = file_ops

    @classmethod
    def write_project_data(cls, project_data_dataframes, project_data_output_xlsx_path):
        """
        This writes a dictionary full of dataframes to an Excel spreadsheet.

        There is no special formatting done to the output Excel sheets.

        This is a class method so an instance of this class does not need
        to be created to write project data. This enables this method to
        be nicely accessed from XlsxManagerRunner subclasses.

        Parameters
        ----------
        project_data_dataframes : dict
            The dataframes to write to the output sheets. The key names will
            be used as the sheet names.

        project_data_output_xlsx_path : str
            The absolute pathname to write the .xlsx file to.
        """
        with pd.ExcelWriter(project_data_output_xlsx_path, mode='w') as writer:
            for name, df in project_data_dataframes.items():
                df.to_excel(writer, sheet_name=name)

    def __enter__(self):
        """
        Opens the workbook for writing and sets the formatting.

        Returns
        -------
        self
            Returns self for easy use in the context manager.
        """
        self.workbook = xlsxwriter.Workbook(self.output_xlsx_path, {'nan_inf_to_errors': True})
        self.set_workbook_formats()
        return self

    def __exit__(self, exception_type, exception_val, exception_traceback):
        """
        Closes the workbook

        Parameters
        ----------
        exception_type
            The type of the exception, if there is one.
        exception_val
            The value of the exception, if there is one.
        exception_traceback
            Traceback of the exception problem.

        Returns
        -------
        bool
            True if the workbook is closed properly. False if something
            prevented normal closure.
        """
        if exception_type:
            print('exception_type: {}'.format(exception_type))
            print('exception_val: {}'.format(exception_val))
            print('exception_traceback:')
            traceback.print_tb(exception_traceback)
        self.workbook.close()
        return True

    def set_workbook_formats(self):
        """
        This method creates the formats in the workbook. It is called upon entry
        into the context manager.
        """
        self.header_format = self.workbook.add_format()
        self.header_format.set_bold()
        self.header_format.set_text_wrap()
        self.scientific_format = self.workbook.add_format()
        self.scientific_format.set_num_format('0.00E+00')
        self.scientific_format.set_align('left')
        self.percent_format = self.workbook.add_format()
        self.percent_format.set_num_format('0.0%')
        self.percent_format.set_align('left')
        self.accounting_format = self.workbook.add_format()
        self.accounting_format.set_num_format('$ #,##0')

    def tab_costs_by_module_type_operation(self, rows):
        """
        This writes the costs_by_module_type_operation tab.

        Parameters
        ----------
        rows : list
            List of dictionaries that are each row in the output
            sheet.
        """
        worksheet = self.workbook.add_worksheet('costs_by_module_type_operation')
        for idx, col_name in enumerate(['Project ID with serial',
                                        'Number of turbines',
                                        'Turbine rating MW',
                                        'Rotor diameter m',
                                        'Module',
                                        'Operation ID',
                                        'Type of cost',
                                        'Cost per turbine',
                                        'Cost per project',
                                        'USD/kW per project']):
            worksheet.write(0, idx, col_name, self.header_format)
        for row_idx, row in enumerate(rows):
            worksheet.write(row_idx + 1, 0, row['project_id_with_serial'])
            worksheet.write(row_idx + 1, 1, row['num_turbines'])
            worksheet.write(row_idx + 1, 2, row['turbine_rating_MW'])
            worksheet.write(row_idx + 1, 3, row['rotor_diameter_m'])
            worksheet.write(row_idx + 1, 4, row['module'])
            worksheet.write(row_idx + 1, 5, row['operation_id'])
            worksheet.write(row_idx + 1, 6, row['type_of_cost'])
            worksheet.write(row_idx + 1, 7, row['cost_per_turbine'], self.accounting_format)
            worksheet.write(row_idx + 1, 8, row['cost_per_project'], self.accounting_format)
            worksheet.write(row_idx + 1, 9, row['usd_per_kw_per_project'], self.accounting_format)
            worksheet.set_column(0, 5, 25)
            worksheet.set_column(6, 10, 17)
        worksheet.freeze_panes(1, 0)  # Freeze the first row.

    def tab_details(self, rows):
        """
        This writes a detailed outputs tab. It takes a list of dictionaries
        as the parameters and in each of those dictionaries it looks at the keys:

        ['project_id', 'module', 'type', 'variable_df_key_col_name', 'unit', 'numeric value', 'non_numeric_value']

        The values of each of those keys become each cell in the row.

        Parameters
        ----------
        rows : list
            list of dicts. See above.
        """
        worksheet = self.workbook.add_worksheet('details')
        worksheet.set_column(3, 3, 66)
        worksheet.set_column(4, 4, 17)
        worksheet.set_column(5, 5, 66)
        worksheet.set_column(0, 2, 17)

        for idx, col_name in enumerate(['Project ID with serial', 'Module', 'Variable or DataFrame', 'name', 'unit', 'Numeric value', 'Non-numeric value']):
            worksheet.write(0, idx, col_name, self.header_format)

        # Go through each row and create Excel rows from each of those rows.
        for row_idx, row in enumerate(rows):
            worksheet.write(row_idx + 1, 0, row['project_id_with_serial'])
            worksheet.write(row_idx + 1, 1, row['module'])
            worksheet.write(row_idx + 1, 2, row['type'])
            worksheet.write(row_idx + 1, 3, row['variable_df_key_col_name'])
            worksheet.write(row_idx + 1, 4, row['unit'])

            value = row['value']
            value_is_number = self._is_numeric(value)
            if value_is_number:
                worksheet.write(row_idx + 1, 5, value, self.scientific_format)
            else:
                worksheet.write(row_idx + 1, 6, value)

            # If there is a last_number, which means this is a dataframe row that has a number
            # at the end, write this into the numeric value column. This overrides automatic
            # type detection.

            if 'last_number' in row:
                worksheet.write(row_idx + 1, 5, row['last_number'], self.scientific_format)

            # Certain data are pairs of numeric and non-numeric values. If a key of
            # "non_numeric_value" exists, put that in column 6.
            # An example is mobilization of an LB75-SL3F-Offload at some numeric cost

            if 'non_numeric_value' in row:
                worksheet.write(row_idx + 1, 6, row['non_numeric_value'])

        worksheet.freeze_panes(1, 0)  # Freeze the first row.

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
