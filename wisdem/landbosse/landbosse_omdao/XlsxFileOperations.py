import os
import sys
from datetime import datetime
from shutil import copy2
from shutil import copytree

from .XlsxOperationException import XlsxOperationException


class XlsxFileOperations:
    """
    This class is made to handle file naming and copying.
    """

    def __init__(self):
        """
        The __init__() method just makes a timestamp that will be used throughout
        the lifetime of this instance.
        """
        dt = datetime.now()
        self.timestamp = f'{dt.year}-{dt.month}-{dt.day}-{dt.hour}-{dt.minute}-{dt.second}'

    def get_input_output_paths_from_argv_or_env(self):
        """
        This uses the sys.argv object to inspect the command line to find input
        and output paths as specified on the command line. It expects the
        command line to have options in the following form:

        -i [input file] -o [output file]

        If one or both of these is missing, then it is filled with the defaults
        from the environment variables LANDBOSSE_INPUT_DIR or LANDBOSSE_OUTPUT_DIR

        If all of them are missing the method defaults to 'inputs/' and 'outputs/'

        There is a third option this method looks for:

        --validate or -v

        If that parameter is present, then --output or -o is optional. If enabled,
        validation mode will accept an input directory with --input that must also have a
        landbosse-output.xlsx file. The model will then run on the inputs. Instead
        of writing a file of the output, the output will stay in memory. The
        landbosse-output.xlsx will then be loaded and compared against the in
        memory output. If the outputs are the same, the validation passes
        because the results were reproduced. Otherwise, the validation failed
        because something broke in the model.

        There is a fourth option this method looks for:

        --scaling or -s

        This enables auto scaling of input parameters as found in XlsxReader's
        apply_cost_and_scaling_modifications_to_project_parameters() method.

        Parameters
        ----------
        This function takes no parameters.

        Returns
        -------
        str, str, bool
            The first two strings are paths to the input and output files
            respectively. The third bool is True if validation mode is
            enabled, False for normal operation.
        """

        # Get the fallback paths from the environment variables and set their
        # defaults. These defualts are used

        input_path_from_env = os.environ.get('LANDBOSSE_INPUT_DIR', 'input')
        output_path_from_env = os.environ.get('LANDBOSSE_OUTPUT_DIR', 'output')

        # input and output paths from command line are initially set to None
        # to indicate they have not been found yet.

        input_path_from_arg = None
        output_path_from_arg = None

        # This is for validation option detection
        validation_enabled =  '--validate' in sys.argv or '-v' in sys.argv

        # This is for scaling study operation
        enable_scaling_study = '--scaling' in sys.argv or '-s' in sys.argv

        # If validation and scaling study are simultaneously enabled, raise
        # an error

        if validation_enabled and enable_scaling_study:
            raise XlsxOperationException('--scaling-study and --validate cannot be enabled at the same time.')

        # Look for the input path on command line
        if '--input' in sys.argv and sys.argv.index('--input') + 1 < len(sys.argv):
            input_idx = sys.argv.index('--input') + 1
            input_path_from_arg = sys.argv[input_idx]

        if '-i' in sys.argv and sys.argv.index('-i') + 1 < len(sys.argv):
            input_idx = sys.argv.index('-i') + 1
            input_path_from_arg = sys.argv[input_idx]

        # Look for the output path on command line
        if '--output' in sys.argv and sys.argv.index('--output') + 1 < len(sys.argv):
            output_idx = sys.argv.index('--output') + 1
            output_path_from_arg = sys.argv[output_idx]

        if '-o' in sys.argv and sys.argv.index('-o') + 1 < len(sys.argv):
            output_idx = sys.argv.index('-o') + 1
            output_path_from_arg = sys.argv[output_idx]

        # Find the final input and output paths. If a command line argument was
        # found for input and/or output, that is used. If it wasn't found,
        # the value from the environment variable search is returned, which includes
        # the default if the environment variable itself wasn't found.

        input_path = input_path_from_arg if input_path_from_arg is not None else input_path_from_env
        output_path = output_path_from_arg if output_path_from_arg is not None else output_path_from_env

        # Return the state of the command line arguments.
        return input_path, output_path, validation_enabled, enable_scaling_study

    def landbosse_input_dir(self):
        """
        See the get_input_output_paths_from_argv_or_env() function above. This
        function is simply a wrapper around that function to get the input
        path.

        Returns
        -------
        str
            The input directory.
        """

        # The last three elements of the returned tuple are not used here.
        # See get_input_output_paths_from_argv_or_env() docstring for more
        # details.

        input_path, _, _, _ = self.get_input_output_paths_from_argv_or_env()
        return input_path

    def landbosse_output_dir(self):
        """
        See the get_input_output_paths_from_argv_or_env() function above. This
        method gets the base path from there. Then, it checks for a timestamped
        directory that matches the timestamp in this instance. If it finds that
        directory, it simply returns the path to that directory. If it does
        not find that directory, it creates the directory and returns the path
        to the newly created directory.

        Returns
        -------
        str
            The output directory.
        """
        _, output_base_path, _, _ = self.get_input_output_paths_from_argv_or_env()
        output_path = os.path.join(output_base_path, f'landbosse-{self.timestamp}')

        if os.path.exists(output_path) and not os.path.isdir(output_path):
            raise FileExistsError(f'Cannot overwrite {output_path} with LandBOSSE data.')
        elif not os.path.exists(output_path):
            os.mkdir(output_path)
            return output_path
        else:
            return output_path

    def parametric_project_data_output_path(self):
        """
        Returns th path to the project output data folder.

        This folder is to put the parameterized project data sheets generated
        during model runs.

        If the directory does not exist, it is created.

        Returns
        -------
        str
            Path to project data output folder.
        """
        path = os.path.join(self.landbosse_output_dir(), 'calculated_parametric_inputs', 'parametric_project_data')

        if os.path.exists(path) and not os.path.isdir(path):
            raise XlsxOperationException(f'Attempt to write project data to {path} failed. File exists and is not a directory.')

        os.makedirs(path, exist_ok=True)
        return path

    def extended_project_list_path(self):
        """
        This returns the path to which the extended project list, which has all
        the parametric values, should be copied.

        If the folder does not exist yet, this method creates it.

        Returns
        -------
        str
            The absolute path to the destiantion of the extended project
            list.
        """
        path = os.path.join(self.landbosse_output_dir(), 'calculated_parametric_inputs')

        if os.path.exists(path) and not os.path.isdir(path):
            raise XlsxOperationException(f'Attempt to write project data to {path} failed. File exists and is not a directory.')

        os.makedirs(path, exist_ok=True)
        return path

    def copy_input_data(self):
        """
        This copies all input data to the outputs folder. The input data it copies
        are all the data BEFORE they have been modified for parametric runs.
        """
        dst_inputs_copy_path = os.path.join(self.landbosse_output_dir(), 'inputs')
        os.makedirs(dst_inputs_copy_path, exist_ok=True)

        src_project_list_xlsx = os.path.join(self.landbosse_input_dir(), 'project_list.xlsx')
        dst_project_list_xlsx = os.path.join(dst_inputs_copy_path, 'project_list.xlsx')

        src_project_data_dir = os.path.join(self.landbosse_input_dir(), 'project_data')
        dst_project_data_dir = os.path.join(dst_inputs_copy_path, 'project_data')

        copy2(src_project_list_xlsx, dst_project_list_xlsx)
        copytree(src_project_data_dir, dst_project_data_dir)

        src_expected_validation_data = os.path.join(self.landbosse_input_dir(),
                                                    'landbosse-expected-validation-data.xlsx')

        dst_expected_validation_data = os.path.join(self.landbosse_output_dir(),
                                                    'landbosse-expected-validation-data.xlsx')

        if os.path.isfile(src_expected_validation_data):
            copy2(src_expected_validation_data, dst_expected_validation_data)
