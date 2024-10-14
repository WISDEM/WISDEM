__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


from copy import deepcopy

from numpy.testing import assert_almost_equal

from wisdem.orbit import ProjectManager
from wisdem.orbit.core.library import extract_library_specs

fixed = extract_library_specs("config", "complete_project")
floating = extract_library_specs("config", "complete_floating_project")


def test_fixed_phase_cost_passing():

    project = ProjectManager(fixed)
    project.run()

    assert_almost_equal(
        project.phases["MonopileDesign"].total_cost,
        project.phases["MonopileInstallation"].system_capex,
    )

    assert_almost_equal(
        project.phases["ScourProtectionDesign"].total_cost,
        project.phases["ScourProtectionInstallation"].system_capex,
    )

    assert_almost_equal(
        project.phases["ArraySystemDesign"].total_cost,
        project.phases["ArrayCableInstallation"].system_capex,
    )

    assert_almost_equal(
        project.phases["ExportSystemDesign"].total_cost,
        project.phases["ExportCableInstallation"].system_capex,
    )

    assert_almost_equal(
        project.phases["OffshoreSubstationDesign"].total_cost,
        project.phases["OffshoreSubstationInstallation"].system_capex,
    )


def test_floating_phase_cost_passing():

    project = ProjectManager(floating)
    project.run()

    assert_almost_equal(
        project.phases["MooringSystemDesign"].total_cost,
        project.phases["MooringSystemInstallation"].system_capex,
    )

    assert_almost_equal(
        project.phases["SemiSubmersibleDesign"].total_cost,
        project.phases["MooredSubInstallation"].system_capex,
    )

    assert_almost_equal(
        project.phases["ArraySystemDesign"].total_cost,
        project.phases["ArrayCableInstallation"].system_capex,
    )

    assert_almost_equal(
        project.phases["ExportSystemDesign"].total_cost,
        project.phases["ExportCableInstallation"].system_capex,
    )

    assert_almost_equal(
        project.phases["OffshoreSubstationDesign"].total_cost,
        project.phases["OffshoreSubstationInstallation"].system_capex,
    )

    spar = deepcopy(floating)
    spar["design_phases"].remove("SemiSubmersibleDesign")
    spar["design_phases"].append("SparDesign")

    project = ProjectManager(spar)
    project.run()

    assert_almost_equal(
        project.phases["SparDesign"].total_cost,
        project.phases["MooredSubInstallation"].system_capex,
    )
