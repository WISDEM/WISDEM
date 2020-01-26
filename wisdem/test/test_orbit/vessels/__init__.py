__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"

import pytest

from wisdem.orbit.library import extract_library_specs
from wisdem.orbit.vessels import Vessel

wtiv = extract_library_specs("wtiv", "test_wtiv")
feeder = extract_library_specs("feeder", "test_feeder")
scour_protection = extract_library_specs(
    "scour_protection_install_vessel", "test_scour_protection_vessel"
)
cable_lay = extract_library_specs(
    "array_cable_lay_vessel", "test_cable_lay_vessel"
)


WTIV = Vessel(wtiv["name"], wtiv)
FEEDER = Vessel(feeder["name"], feeder)
SCOUR_PROTECTION = Vessel(scour_protection["name"], scour_protection)
CABLE_LAY = Vessel(cable_lay["name"], cable_lay)
