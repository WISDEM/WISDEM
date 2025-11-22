from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).parent


def pytest_addoption(parser):  # noqa: D103
    parser.addoption(
        "--unit",
        action="store_true",
        default=False,
        help="run all tests in 'wisdem/test', except for 'test/test_examples/'.",
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run all tests in 'test/test_examples/.",
    )


def pytest_configure(config):  # noqa: D103
    # Check for the options
    unit = config.getoption("--unit")
    integration = config.getoption("--integration")

    # Provide the appropriate directories
    unit_tests = [
        str(name)
        for el in sorted(TEST_ROOT.iterdir())
        if "test_examples" not in el.parts and el.is_dir()
        for name in sorted(el.iterdir())
        if name.name.startswith("test_") and name.suffix == ".py"
    ]
    integration_tests = [
        str(name)
        for name in sorted((TEST_ROOT / "test_examples").iterdir())
        if name.name.startswith("test_") and name.suffix == ".py"
    ]

    # If both, run them all; if neither skip any modifications; otherwise run just the
    # appropriate subset
    if integration and unit:
        config.args = unit_tests + regression_tests
    elif integration:
        config.args = integration_tests
    elif unit:
        config.args = unit_tests
