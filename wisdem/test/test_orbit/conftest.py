"""Shared pytest settings and fixtures."""


import os

import pytest

from wisdem.orbit.library import initialize_library

collect_ignore = ["setup.py"]


def get_test_library_path():
    """Retrieves the testing suite library path.

    Returns
    -------
    str
        Path to <path>/tests/data/library.
    """
    #repository = "ORBIT-dev"
    #root = os.path.split(os.path.abspath(os.path.basename(__file__)))[0]
    #cutoff = root.index(repository) + len(repository)
    #repository = root[:cutoff]
    #library_path = os.path.join(repository, "tests", "data", "library")
    library_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'library')
    return library_path


def pytest_configure():
    """Creates the default library for pytest testing suite and initializes it
    when required.
    """
    pytest.library = get_test_library_path()
    initialize_library(pytest.library)
