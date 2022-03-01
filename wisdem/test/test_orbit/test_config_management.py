__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from wisdem.orbit import ProjectManager, load_config, save_config
from wisdem.orbit.core.library import extract_library_specs

complete_project = extract_library_specs("config", "complete_project")


def test_save_and_load_equality(tmp_yaml_del):

    save_config(complete_project, "tmp.yaml", overwrite=True)
    new = load_config("tmp.yaml")

    assert new == complete_project


def test_orbit_version_ProjectManager():

    config = ProjectManager.compile_input_dict(["MonopileDesign", "MonopileInstallation"])
    assert "orbit_version" in config.keys()
