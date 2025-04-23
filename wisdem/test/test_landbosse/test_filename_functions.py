import os


def landbosse_test_input_dir():
    """
    This function is to find the input directory of the landbosse.

    This function checks to see if the LANDBOSSE_INPUT_DIR environment
    variable is defined. If so, it returns the value of that variable.

    If the environment variable is not defined, it returns 'inputs'
    relative to the current working directory.

    Returns
    -------
    str
        The input directory.
    """

    test_source_dir = os.path.dirname(os.path.realpath(__file__))
    default_test_inputs_dir = os.path.join(test_source_dir, 'inputs')
    return os.environ.get('LANDBOSSE_TEST_INPUT_DIR', default_test_inputs_dir)

