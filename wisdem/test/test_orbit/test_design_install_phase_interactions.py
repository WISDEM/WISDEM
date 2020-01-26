__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

from wisdem.orbit import ProjectManager

config = {
    "wtiv": "test_wtiv",
    "site": {"depth": 20, "distance": 20, "mean_windspeed": 9},
    "plant": {"num_turbines": 20},
    "turbine": {
        "hub_height": 130,
        "rotor_diameter": 154,
        "rated_windspeed": 11,
    },
    "port": {"num_cranes": 1, "monthly_rate": 2e6},
    "monopile": {
        "type": "Monopile",
        "length": 60,
        "diameter": 8,
        "deck_space": 0,
        "weight": 600,
    },
    "transition_piece": {
        "type": "Transition Piece",
        "deck_space": 0,
        "weight": 500,
    },
    "monopile_design": {},
    "design_phases": ["MonopileDesign"],
    "install_phases": ["MonopileInstallation"],
}


def test_monopile_definition():

    test_config = deepcopy(config)
    _ = test_config.pop("transition_piece")

    project = ProjectManager(test_config)
    project.run_project()

    for key, value in config["monopile"].items():
        if key == "type":
            continue

        assert project.config["monopile"][key] == value

    for key, value in config["transition_piece"].items():
        if key == "type":
            continue

        assert project.config["transition_piece"][key] != value


def test_transition_piece_definition():

    test_config = deepcopy(config)
    _ = test_config.pop("monopile")

    project = ProjectManager(test_config)
    project.run_project()

    for key, value in config["monopile"].items():
        if key == "type":
            continue

        assert project.config["monopile"][key] != value

    for key, value in config["transition_piece"].items():
        if key == "type":
            continue

        assert project.config["transition_piece"][key] == value


def test_mono_and_tp_definition():

    test_config = deepcopy(config)

    project = ProjectManager(test_config)
    project.run_project()

    for key, value in config["monopile"].items():
        if key == "type":
            continue

        assert project.config["monopile"][key] == value

    for key, value in config["transition_piece"].items():
        if key == "type":
            continue

        assert project.config["transition_piece"][key] == value
