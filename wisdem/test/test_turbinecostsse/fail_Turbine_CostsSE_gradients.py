
#!/usr/bin/env python
# encoding: utf-8
"""
test_turbine_costsse.py

Created by Katherine Dykes on 2014-01-07.
Copyright (c) NREL. All rights reserved.
"""

import unittest
from commonse.utilities import check_gradient_unit_test

from turbine_costsse.turbine_costsse import TowerCostAdder, TowerCost
from turbine_costsse.turbine_costsse import BladeCost, HubCost, PitchSystemCost, SpinnerCost, \
    HubSystemCostAdder, RotorCostAdder
from turbine_costsse.turbine_costsse import LowSpeedShaftCost, BearingsCost, GearboxCost, \
    HighSpeedSideCost, GeneratorCost, BedplateCost, \
    YawSystemCost, NacelleSystemCostAdder
from turbine_costsse.turbine_costsse import TurbineCostAdder

from turbine_costsse.nrel_csm_tcc import tower_csm_component
from turbine_costsse.nrel_csm_tcc import blades_csm_component
from turbine_costsse.nrel_csm_tcc import hub_csm_component
from turbine_costsse.nrel_csm_tcc import nacelle_csm_component
from turbine_costsse.nrel_csm_tcc import rotor_mass_adder, tcc_csm_component

# turbine_costsse Model
# ----------------------------------------------------------
# Tower Components


class TestTowerCost(unittest.TestCase):

    def test1(self):

        tower = TowerCost()

        tower.tower_mass = 434559.0
        tower.year = 2009
        tower.month = 12

        check_gradient_unit_test(self, tower)


class TestTowerCostAdder(unittest.TestCase):

    def test1(self):

        tower = TowerCostAdder()

        tower.tower_cost = 1000000.0

        check_gradient_unit_test(self, tower)

# Rotor Components


class TestBladeCost(unittest.TestCase):

    def test1(self):

        blade = BladeCost()

        blade.blade_mass = 17650.67
        blade.year = 2009
        blade.month = 12

        check_gradient_unit_test(self, blade)


class TestHubCost(unittest.TestCase):

    def test1(self):

        hub = HubCost()

        hub.hub_mass = 31644.5
        hub.year = 2009
        hub.month = 12

        check_gradient_unit_test(self, hub)


class TestPitchSystemCost(unittest.TestCase):

    def test1(self):

        pitch = PitchSystemCost()

        pitch.pitch_system_mass = 17004.0
        pitch.year = 2009
        pitch.month = 12

        check_gradient_unit_test(self, pitch)


class TestSpinnerCost(unittest.TestCase):

    def test1(self):

        spinner = SpinnerCost()

        spinner.spinner_mass = 1810.5
        spinner.year = 2009
        spinner.month = 12

        check_gradient_unit_test(self, spinner)


class TestHubSystemCostAdder(unittest.TestCase):

    def test1(self):

        hub = HubSystemCostAdder()

        hub.hub_cost = 20000.0
        hub.pitch_system_cost = 20000.0
        hub.spinner_cost = 20000.0

        check_gradient_unit_test(self, hub)


class TestRotorCostAdder(unittest.TestCase):

    def test1(self):

        rotor = RotorCostAdder()

        rotor.blade_cost = 20000.0
        rotor.blade_number = 3
        rotor.hub_system_cost = 20000.0

        check_gradient_unit_test(self, rotor)


# Nacelle Components
class TestLowSpeedShaftCost(unittest.TestCase):

    def test1(self):

        lss = LowSpeedShaftCost()

        lss.low_speed_shaft_mass = 31257.3
        lss.year = 2009
        lss.month = 12

        check_gradient_unit_test(self, lss)


class TestBearingsCost(unittest.TestCase):

    def test1(self):

        bearings = BearingsCost()

        bearings.main_bearing_mass = 9731.41 / 2.0
        bearings.second_bearing_mass = 9731.41 / 2.0
        bearings.year = 2009
        bearings.month = 12

        check_gradient_unit_test(self, bearings)


class TestGearboxCost(unittest.TestCase):

    def test1(self):

        gearbox = GearboxCost()

        gearbox.gearbox_mass = 30237.60
        gearbox.year = 2009
        gearbox.month = 12
        gearbox.drivetrain_design = 'geared'

        check_gradient_unit_test(self, gearbox)


class TestHighSpeedSideCost(unittest.TestCase):

    def test1(self):

        hss = HighSpeedSideCost()

        hss.high_speed_side_mass = 1492.45
        hss.year = 2009
        hss.month = 12

        check_gradient_unit_test(self, hss)


class TestGeneratorCost(unittest.TestCase):

    def test1(self):

        generator = GeneratorCost()

        generator.generator_mass = 16699.85
        generator.year = 2009
        generator.month = 12
        generator.drivetrain_design = 'geared'
        generator.machine_rating = 5000.0

        check_gradient_unit_test(self, generator)


class TestBedplateCost(unittest.TestCase):

    def test1(self):

        bedplate = BedplateCost()

        bedplate.bedplate_mass = 93090.6
        bedplate.year = 2009
        bedplate.month = 12

        check_gradient_unit_test(self, bedplate)


class TestYawSystemCost(unittest.TestCase):

    def test1(self):

        yaw = YawSystemCost()

        yaw.yaw_system_mass = 11878.24
        yaw.year = 2009
        yaw.month = 12

        check_gradient_unit_test(self, yaw)


class TestNacelleSystemCostAdder(unittest.TestCase):

    def test1(self):

        nacelle = NacelleSystemCostAdder()

        nacelle.bedplate_mass = 93090.6
        nacelle.machine_rating = 5000.0
        nacelle.drivetrainDesign = 1
        nacelle.crane = True
        nacelle.offshore = True
        nacelle.year = 2009
        nacelle.month = 12
        nacelle.lss_cost = 10000.0
        nacelle.bearings_cost = 10000.0
        nacelle.gearbox_cost = 10000.0
        nacelle.hss_cost = 10000.0
        nacelle.bedplate_cost = 10000.0
        nacelle.bedplateCost2002 = 8000.0
        nacelle.yaw_system_cost = 10000.0

        check_gradient_unit_test(self, nacelle, tol=1e-5)

# Turbine Components


class TestTurbineCostAdder(unittest.TestCase):

    def test1(self):

        turbine = TurbineCostAdder()

        turbine.offshore = True
        turbine.rotor_cost = 2000000.0
        turbine.nacelle_cost = 5000000.0
        turbine.tower_cost = 1000000.0

        check_gradient_unit_test(self, turbine)


# NREL CSM TCC Components
# ----------------------------------------------------------
# Tower Component
class Test_tower_csm_component(unittest.TestCase):

    def test1(self):

        tower = tower_csm_component()

        tower.rotor_diameter = 126.0
        tower.hub_height = 90.0
        tower.year = 2009
        tower.month = 12
        tower.advanced = False

        check_gradient_unit_test(self, tower)

# Rotor Components


class Test_blade_csm_component(unittest.TestCase):

    def test1(self):

        blades = blades_csm_component()

        blades.rotor_diameter = 126.0
        blades.advanced_blade = False
        blades.year = 2009
        blades.month = 12
        blades.advanced_blade = False

        check_gradient_unit_test(self, blades)


class Test_hub_csm_component(unittest.TestCase):

    def test1(self):

        hub = hub_csm_component()

        hub.blade_mass = 25614.377
        hub.rotor_diameter = 126.0
        hub.blade_number = 3
        hub.year = 2009
        hub.month = 12

        check_gradient_unit_test(self, hub)

# Nacelle Components


class Test_nacelle_csm_component(unittest.TestCase):

    def test1(self):

        nac = nacelle_csm_component()

        nac.rotor_diameter = 126.0
        nac.machine_rating = 5000.0
        nac.rotor_mass = 123193.30
        nac.rotor_thrust = 500930.1
        nac.rotor_torque = 4365249
        nac.drivetrain_design = 'geared'
        nac.offshore = True
        nac.crane = True
        nac.advanced_bedplate = 0
        nac.year = 2009
        nac.month = 12

        check_gradient_unit_test(self, nac, display=False)

# Turbine Components


class Test_rotor_mass_adder(unittest.TestCase):

    def test1(self):

        rotor = rotor_mass_adder()

        rotor.blade_mass = 17000.
        rotor.hub_system_mass = 35000.
        rotor.blade_number = 3

        check_gradient_unit_test(self, rotor)


# Turbine Components
class Test_tcc_csm_component(unittest.TestCase):

    def test1(self):

        trb = tcc_csm_component()

        trb.rotor_diameter = 126.0
        trb.advanced_blade = True
        trb.blade_number = 3
        trb.hub_height = 90.0
        trb.machine_rating = 5000.0
        trb.offshore = True
        trb.year = 2009
        trb.month = 12

        check_gradient_unit_test(self, trb)

#----------------------------------------------------

if __name__ == "__main__":
    unittest.main()
