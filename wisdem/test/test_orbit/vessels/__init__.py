__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os

from wisdem.orbit.vessels import Vessel

WTIV_SPECS = {
    "name": "Example WTIV",
    "day_rate": 250000,
    "crane_specs": {
        "boom_length": 100,
        "max_hook_height": 100,
        "max_lift": 1200,
        "max_windspeed": 15,
        "radius": 30,
    },
    "jacksys_specs": {
        "air_gap": 10,
        "leg_length": 110,
        "leg_pen": 5,
        "max_depth": 75,
        "max_extension": 85,
        "num_legs": 6,
        "speed_above_depth": 1,
        "speed_below_depth": 2.5,
    },
    "storage_specs": {
        "max_cargo": 8000,
        "max_deck_load": 15,
        "max_deck_space": 4000,
    },
    "transport_specs": {
        "max_waveheight": 3,
        "max_windspeed": 20,
        "transit_speed": 10,
    },
    "vessel_specs": {
        "beam_length": 50,
        "max_draft": 6,
        "min_draft": 5,
        "overall_length": 150,
    },
}


WTIV = Vessel(WTIV_SPECS["name"], WTIV_SPECS)


FEEDER_SPECS = {
    "name": "Example Feeder",
    "day_rate": 50000,
    "jacksys_specs": {
        "air_gap": 10,
        "leg_length": 85,
        "leg_pen": 5,
        "max_depth": 40,
        "max_extension": 60,
        "num_legs": 4,
        "speed_above_depth": 0.5,
        "speed_below_depth": 0.5,
    },
    "storage_specs": {
        "max_cargo": 1200,
        "max_deck_load": 8,
        "max_deck_space": 1000,
    },
    "transport_specs": {
        "max_waveheight": 2.5,
        "max_windspeed": 20,
        "transit_speed": 6,
    },
    "vessel_specs": {
        "beam_length": 35,
        "max_draft": 5,
        "min_draft": 4,
        "overall_length": 60,
    },
}


FEEDER = Vessel(FEEDER_SPECS["name"], FEEDER_SPECS)


SCOUR_PROTECTION_SPECS = {
    "name": "Example Scour Protection Install Vessel",
    "day_rate": 50000,
    "storage_specs": {
        "max_cargo": 32000,
        "max_deck_load": 8,
        "max_deck_space": 600,
    },
    "transport_specs": {
        "cable_lay_speed": 0.5,
        "max_waveheight": 1.5,
        "max_windspeed": 25,
        "transit_speed": 11.5,
    },
    "vessel_specs": {
        "beam_length": 35,
        "max_draft": 5,
        "min_draft": 4,
        "overall_length": 60,
    },
}

SCOUR_PROTECTION = Vessel(
    SCOUR_PROTECTION_SPECS["name"], SCOUR_PROTECTION_SPECS
)

CABLE_LAY_SPECS = {
    "carousel_specs": {
        "capacity": 5000,
        "max_cable_size": 300,
        "min_cable_size": 50,
    },
    "storage_specs": {
        "helideck": False,
        "max_cargo": 6000,
        "max_deck_load": 1,
        "max_deck_space": 1,
    },
    "transport_specs": {
        "cable_lay_speed": 0.5,
        "max_waveheight": 1.5,
        "max_windspeed": 25,
        "transit_speed": 11.5,
    },
    "vessel_specs": {
        "beam_length": 30.0,
        "day_rate": 50000,
        "min_draft": 4.8,
        "overall_length": 99.0,
    },
}

CABLE_LAY_VESSEL = Vessel("Example Cable Lay", CABLE_LAY_SPECS)
