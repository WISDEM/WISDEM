__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

from wisdem.orbit.phases import BasePhase

# Tests for key checking.
expected_config = {
    "param1": "str",
    "param2": {"param3": "int | float", "param4": "str (optional)"},
    "param5 (variable)": "int",
    "param6": {"param7": "float (optional)"},
}


def test_good_config():

    config = deepcopy(expected_config)
    missing = BasePhase._check_keys(expected_config, config)
    assert len(missing) == 0


def test_missing_key():

    config = deepcopy(expected_config)
    _ = config.pop("param1")
    missing = BasePhase._check_keys(expected_config, config)
    assert len(missing) == 1


def test_optional():

    config = deepcopy(expected_config)
    _ = config["param2"].pop("param4")
    missing = BasePhase._check_keys(expected_config, config)
    assert len(missing) == 0


def test_variable_key():

    config = deepcopy(expected_config)
    _ = config.pop("param5 (variable)")

    missing = BasePhase._check_keys(expected_config, config)
    assert len(missing) == 0


def test_optional_dict():

    config = deepcopy(expected_config)
    _ = config.pop("param6")

    missing = BasePhase._check_keys(expected_config, config)
    assert len(missing) == 0
