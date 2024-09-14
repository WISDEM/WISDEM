"""Tests for the `ExportSystemDesign` class."""

__author__ = "Rob Hammond"
__copyright__ = "Copyright 2020, National Renewable Energy Laboratory"
__maintainer__ = "Rob Hammond"
__email__ = "rob.hammond@nrel.gov"

import warnings
from copy import deepcopy

import pytest

from wisdem.orbit.core.library import extract_library_specs
from wisdem.orbit.phases.design import ExportSystemDesign

config = extract_library_specs("config", "export_design")


def test_export_system_creation():
    export = ExportSystemDesign(config)
    export.run()

    assert export.num_cables
    assert export.length
    assert export.mass
    assert export.cable
    assert export.total_length
    assert export.total_mass


def test_number_cables():
    export = ExportSystemDesign(config)
    print(export.config)
    export.run()

    assert export.num_cables == 9


def test_cable_length():
    export = ExportSystemDesign(config)
    export.run()

    length = (0.02 + 3 + 30) * 1.01
    assert export.length == length


def test_cable_mass():
    export = ExportSystemDesign(config)
    export.run()

    length = (0.02 + 3 + 30) * 1.01
    mass = length * export.cable.linear_density
    assert export.mass == pytest.approx(mass, abs=1e-10)


def test_total_cable():
    export = ExportSystemDesign(config)
    export.run()

    length = 0.02 + 3 + 30
    length += length * 0.01
    mass = length * export.cable.linear_density
    assert export.total_mass == pytest.approx(mass * 9, abs=1e-10)
    assert export.total_length == pytest.approx(length * 9, abs=1e-10)


def test_cables_property():
    export = ExportSystemDesign(config)
    export.run()

    assert (
        export.sections_cables == export.cable.name
    ).sum() == export.num_cables


def test_cable_lengths_property():
    export = ExportSystemDesign(config)
    export.run()

    cable_name = export.cable.name
    assert (
        export.cable_lengths_by_type[cable_name] == export.length
    ).sum() == export.num_cables


def test_total_cable_len_property():
    export = ExportSystemDesign(config)
    export.run()

    cable_name = export.cable.name
    assert export.total_cable_length_by_type[cable_name] == pytest.approx(
        export.total_length,
        abs=1e-10,
    )


def test_design_result():
    export = ExportSystemDesign(config)
    export.run()

    _ = export.cable.name
    cables = export.design_result["export_system"]["cable"]
    # landfall = export.design_results["export_system"]["landfall"]

    assert cables["sections"] == [export.length]
    assert cables["number"] == 9
    assert cables["linear_density"] == export.cable.linear_density
    # assert landfall["interconnection_distance"] == 3


def test_floating_length_calculations():

    base = deepcopy(config)
    base["site"]["depth"] = 250
    base["export_system_design"]["touchdown_distance"] = 0

    sim = ExportSystemDesign(base)
    sim.run()

    base_length = sim.total_length

    with_cat = deepcopy(config)
    with_cat["site"]["depth"] = 250

    new = ExportSystemDesign(with_cat)
    new.run()

    assert new.total_length < base_length


def test_deprecated_landfall():

    base = deepcopy(config)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim = ExportSystemDesign(base)
        sim.run()

    deprecated = deepcopy(base)
    deprecated["landfall"] = {"interconnection_distance": 4}

    with pytest.deprecated_call():
        sim = ExportSystemDesign(deprecated)
        sim.run()
