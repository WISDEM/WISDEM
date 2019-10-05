
#!/usr/bin/env python
# encoding: utf-8
"""
test_turbine_costsse.py

Created by Katherine Dykes on 2014-01-07.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
from commonse.utilities import check_gradient_unit_test

from turbine_costsse.turbine_costsse import TowerCostAdder, TowerCost, Tower_CostsSE
from turbine_costsse.turbine_costsse import BladeCost, HubCost, PitchSystemCost, SpinnerCost, \
    HubSystemCostAdder, RotorCostAdder, Rotor_CostsSE
from turbine_costsse.turbine_costsse import LowSpeedShaftCost, BearingsCost, GearboxCost, \
    HighSpeedSideCost, GeneratorCost, BedplateCost, \
    YawSystemCost, NacelleSystemCostAdder, Nacelle_CostsSE
from turbine_costsse.turbine_costsse import TurbineCostAdder, Turbine_CostsSE

from turbine_costsse.nrel_csm_tcc import tower_csm_component
from turbine_costsse.nrel_csm_tcc import blades_csm_component
from turbine_costsse.nrel_csm_tcc import hub_csm_component
from turbine_costsse.nrel_csm_tcc import nacelle_csm_component
from turbine_costsse.nrel_csm_tcc import rotor_mass_adder, tcc_csm_component, tcc_csm_assembly

# turbine_costsse Model
# ----------------------------------------------------------
# Tower Components
class TestTower_CostsSE(unittest.TestCase):

    def setUp(self):

        # simple test of module
        self.tower = Tower_CostsSE()
    
        self.tower.tower_mass = 434559.0
        self.tower.year = 2009
        self.tower.month =  12
    
    def test_functionality(self):
        
        self.tower.run()
        
        self.assertEqual(round(self.tower.cost,2), 987180.59)

class TestTowerCost(unittest.TestCase):

    def setUp(self):

        self.tower = TowerCost()

        self.tower.tower_mass = 434559.0
        self.tower.year = 2009
        self.tower.month = 12
    
    def test_functionality(self):
        
        self.tower.run()
        
        self.assertEqual(round(self.tower.cost,2), 987180.59)

    def test_gradient(self):

        check_gradient_unit_test(self, self.tower)


class TestTowerCostAdder(unittest.TestCase):

    def setUp(self):

        self.tower = TowerCostAdder()

        self.tower.tower_cost = 1000000.0

    def test_functionality(self):
        
        self.tower.run()
        
        self.assertEqual(round(self.tower.cost,2), 1000000.0)

    def test_gradient(self):

        check_gradient_unit_test(self, self.tower)

# Rotor Components

class TestRotor_CostSE(unittest.TestCase):

    def setUp(self):

        self.rotor = Rotor_CostsSE()
    
        # Blade Test 1
        self.rotor.blade_number = 3
        self.rotor.advanced = True
        self.rotor.blade_mass = 17650.67
        self.rotor.hub_mass = 31644.5
        self.rotor.pitch_system_mass = 17004.0
        self.rotor.spinner_mass = 1810.5
        self.rotor.year = 2009
        self.rotor.month = 12

    def test_functionality(self):
    
        self.rotor.run()
        
        self.assertEqual(round(self.rotor.cost,2), 1478411.14) 


class TestBladeCost(unittest.TestCase):

    def setUp(self):

        self.blade = BladeCost()

        self.blade.blade_mass = 17650.67
        self.blade.year = 2009
        self.blade.month = 12

    def test_functionality(self):
    
        self.blade.run()
        
        self.assertEqual(round(self.blade.cost,2), 252437.55)        

    def test_gradient(self):

        check_gradient_unit_test(self, self.blade)


class TestHubCost(unittest.TestCase):

    def setUp(self):

        self.hub = HubCost()

        self.hub.hub_mass = 31644.5
        self.hub.year = 2009
        self.hub.month = 12

    def test_functionality(self):
    
        self.hub.run()
        
        self.assertEqual(round(self.hub.cost,2), 175513.66) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.hub)


class TestPitchSystemCost(unittest.TestCase):

    def setUp(self):

        self.pitch = PitchSystemCost()

        self.pitch.pitch_system_mass = 17004.0
        self.pitch.year = 2009
        self.pitch.month = 12

    def test_functionality(self):
    
        self.pitch.run()
        
        self.assertEqual(round(self.pitch.cost,2), 535075.83) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.pitch)


class TestSpinnerCost(unittest.TestCase):

    def setUp(self):

        self.spinner = SpinnerCost()

        self.spinner.spinner_mass = 1810.5
        self.spinner.year = 2009
        self.spinner.month = 12

    def test_functionality(self):
    
        self.spinner.run()
        
        self.assertEqual(round(self.spinner.cost,2), 10508.99) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.spinner)


class TestHubSystemCostAdder(unittest.TestCase):

    def setUp(self):

        self.hub = HubSystemCostAdder()

        self.hub.hub_cost = 20000.0
        self.hub.pitch_system_cost = 20000.0
        self.hub.spinner_cost = 20000.0

    def test_functionality(self):
    
        self.hub.run()
        
        self.assertEqual(round(self.hub.cost,2), 60000.00) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.hub)


class TestRotorCostAdder(unittest.TestCase):

    def setUp(self):

        self.rotor = RotorCostAdder()

        self.rotor.blade_cost = 20000.0
        self.rotor.blade_number = 3
        self.rotor.hub_system_cost = 20000.0

    def test_functionality(self):
    
        self.rotor.run()
        
        self.assertEqual(round(self.rotor.cost,2), 80000.00) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.rotor)


# Nacelle Components

class TestNacelle_CostSE(unittest.TestCase):

    def setUp(self):

        self.nacelle = Nacelle_CostsSE()
    
        self.nacelle.low_speed_shaft_mass = 31257.3
        #nacelle.bearingsMass = 9731.41
        self.nacelle.main_bearing_mass = 9731.41 / 2.0
        self.nacelle.second_bearing_mass = 9731.41 / 2.0
        self.nacelle.gearbox_mass = 30237.60
        self.nacelle.high_speed_side_mass = 1492.45
        self.nacelle.generator_mass = 16699.85
        self.nacelle.bedplate_mass = 93090.6
        self.nacelle.yaw_system_mass = 11878.24
        self.nacelle.machine_rating = 5000.0
        self.nacelle.drivetrain_design = 'geared'
        self.nacelle.crane = True
        self.nacelle.offshore = True
        self.nacelle.year = 2009
        self.nacelle.month = 12

    def test_functionality(self):
    
        self.nacelle.run()
        
        self.assertEqual(round(self.nacelle.cost,2), 2917983.93) 

class TestLowSpeedShaftCost(unittest.TestCase):

    def setUp(self):

        self.lss = LowSpeedShaftCost()

        self.lss.low_speed_shaft_mass = 31257.3
        self.lss.year = 2009
        self.lss.month = 12

    def test_functionality(self):
    
        self.lss.run()
        
        self.assertEqual(round(self.lss.cost,2), 183363.66) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.lss)


class TestBearingsCost(unittest.TestCase):

    def setUp(self):

        self.bearings = BearingsCost()

        self.bearings.main_bearing_mass = 9731.41 / 2.0
        self.bearings.second_bearing_mass = 9731.41 / 2.0
        self.bearings.year = 2009
        self.bearings.month = 12

    def test_functionality(self):
    
        self.bearings.run()
        
        self.assertEqual(round(self.bearings.cost,2), 56660.73)

    def test_gradient(self):

        check_gradient_unit_test(self, self.bearings)


class TestGearboxCost(unittest.TestCase):

    def setUp(self):

        self.gearbox = GearboxCost()

        self.gearbox.gearbox_mass = 30237.60
        self.gearbox.year = 2009
        self.gearbox.month = 12
        self.gearbox.drivetrain_design = 'geared'

    def test_functionality(self):
    
        self.gearbox.run()
        
        self.assertEqual(round(self.gearbox.cost,2), 648030.64)

    def test_gradient(self):

        check_gradient_unit_test(self, self.gearbox)


class TestHighSpeedSideCost(unittest.TestCase):

    def setUp(self):

        self.hss = HighSpeedSideCost()

        self.hss.high_speed_side_mass = 1492.45
        self.hss.year = 2009
        self.hss.month = 12

    def test_functionality(self):
    
        self.hss.run()
        
        self.assertEqual(round(self.hss.cost,2), 15218.23)

    def test_gradient(self):

        check_gradient_unit_test(self, self.hss)


class TestGeneratorCost(unittest.TestCase):

    def setUp(self):

        self.generator = GeneratorCost()

        self.generator.generator_mass = 16699.85
        self.generator.year = 2009
        self.generator.month = 12
        self.generator.drivetrain_design = 'geared'
        self.generator.machine_rating = 5000.0

    def test_functionality(self):
    
        self.generator.run()
        
        self.assertEqual(round(self.generator.cost,2), 435157.71)

    def test_gradient(self):

        check_gradient_unit_test(self, self.generator)


class TestBedplateCost(unittest.TestCase):

    def setUp(self):

        self.bedplate = BedplateCost()

        self.bedplate.bedplate_mass = 93090.6
        self.bedplate.year = 2009
        self.bedplate.month = 12

    def test_functionality(self):
    
        self.bedplate.run()
        
        self.assertEqual(round(self.bedplate.cost,2), 138167.19)

    def test_gradient(self):

        check_gradient_unit_test(self, self.bedplate)


class TestYawSystemCost(unittest.TestCase):

    def setUp(self):

        self.yaw = YawSystemCost()

        self.yaw.yaw_system_mass = 11878.24
        self.yaw.year = 2009
        self.yaw.month = 12

    def test_functionality(self):
    
        self.yaw.run()
        
        self.assertEqual(round(self.yaw.cost,2), 137698.39)

    def test_gradient(self):

        check_gradient_unit_test(self, self.yaw)


class TestNacelleSystemCostAdder(unittest.TestCase):

    def setUp(self):

        self.nacelle = NacelleSystemCostAdder()

        self.nacelle.bedplate_mass = 93090.6
        self.nacelle.machine_rating = 5000.0
        self.nacelle.crane = True
        self.nacelle.offshore = True
        self.nacelle.year = 2009
        self.nacelle.month = 12
        self.nacelle.lss_cost = 183363.66
        self.nacelle.bearings_cost = 56660.73
        self.nacelle.gearbox_cost = 648030.64
        self.nacelle.hss_cost = 15218.23
        self.nacelle.generator_cost = 435157.71
        self.nacelle.bedplate_cost = 138167.19
        self.nacelle.bedplateCost2002 = 105872.02
        self.nacelle.yaw_system_cost = 137698.39

    def test_functionality(self):
    
        self.nacelle.run()
        
        self.assertEqual(round(self.nacelle.cost,2), 2917983.91)

    def test_gradient(self):

        check_gradient_unit_test(self, self.nacelle, tol=1e-5)

# Turbine Components

class TestTurbine_CostSE(unittest.TestCase):

    def setUp(self):

        self.turbine = Turbine_CostsSE()
    
        self.turbine.blade_mass = 17650.67  # inline with the windpact estimates
        self.turbine.hub_mass = 31644.5
        self.turbine.pitch_system_mass = 17004.0
        self.turbine.spinner_mass = 1810.5
        self.turbine.low_speed_shaft_mass = 31257.3
        #bearingsMass = 9731.41
        self.turbine.main_bearing_mass = 9731.41 / 2
        self.turbine.second_bearing_mass = 9731.41 / 2
        self.turbine.gearbox_mass = 30237.60
        self.turbine.high_speed_side_mass = 1492.45
        self.turbine.generator_mass = 16699.85
        self.turbine.bedplate_mass = 93090.6
        self.turbine.yaw_system_mass = 11878.24
        self.turbine.tower_mass = 434559.0
        self.turbine.machine_rating = 5000.0
        self.turbine.advanced = True
        self.turbine.blade_number = 3
        self.turbine.drivetrain_design = 'geared'
        self.turbine.crane = True
        self.turbine.offshore = True
        self.turbine.year = 2010
        self.turbine.month =  12

    def test_functionality(self):
    
        self.turbine.run()
        
        self.assertEqual(round(self.turbine.turbine_cost,2), 6153564.42) 

class TestTurbineCostAdder(unittest.TestCase):

    def setUp(self):

        self.turbine = TurbineCostAdder()

        self.turbine.offshore = True
        self.turbine.rotor_cost = 1519510.91
        self.turbine.nacelle_cost = 3043115.22
        self.turbine.tower_cost = 1031523.34

    def test_functionality(self):
    
        self.turbine.run()
        
        self.assertEqual(round(self.turbine.turbine_cost,2), 6153564.42) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.turbine)


# NREL CSM TCC Components
# ----------------------------------------------------------

# Tower Component

class Test_tower_csm_component(unittest.TestCase):

    def setUp(self):

        self.tower = tower_csm_component()

        self.tower.rotor_diameter = 126.0
        self.tower.hub_height = 90.0
        self.tower.year = 2009
        self.tower.month = 12
        self.tower.advanced = False

    def test_functionality(self):
    
        self.tower.run()
        
        self.assertEqual(round(self.tower.tower_cost,2), 1009500.24) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.tower)


# Rotor Components

class Test_blade_csm_component(unittest.TestCase):

    def setUp(self):

        self.blades = blades_csm_component()

        self.blades.rotor_diameter = 126.0
        self.blades.advanced_blade = False
        self.blades.year = 2009
        self.blades.month = 12
        self.blades.advanced_blade = False

    def test_functionality(self):
    
        self.blades.run()
        
        self.assertEqual(round(self.blades.blade_cost,2), 276143.07) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.blades)


class Test_hub_csm_component(unittest.TestCase):

    def setUp(self):

        self.hub = hub_csm_component()

        self.hub.blade_mass = 25614.377
        self.hub.rotor_diameter = 126.0
        self.hub.blade_number = 3
        self.hub.year = 2009
        self.hub.month = 12

    def test_functionality(self):
    
        self.hub.run()
        
        self.assertEqual(round(self.hub.hub_system_cost,2), 421290.41) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.hub)

# Nacelle Components


class Test_nacelle_csm_component(unittest.TestCase):

    def setUp(self):

        self.nac = nacelle_csm_component()

        self.nac.rotor_diameter = 126.0
        self.nac.machine_rating = 5000.0
        self.nac.rotor_mass = 123193.30
        self.nac.rotor_thrust = 500930.1
        self.nac.rotor_torque = 4365249
        self.nac.drivetrain_design = 'geared'
        self.nac.offshore = True
        self.nac.crane = True
        self.nac.advanced_bedplate = 0
        self.nac.year = 2009
        self.nac.month = 12

    def test_functionality(self):
    
        self.nac.run()
        
        self.assertEqual(round(self.nac.nacelle_cost,2), 3275147.05) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.nac, display=False)

# Turbine Components


class Test_rotor_mass_adder(unittest.TestCase):

    def setUp(self):

        self.rotor = rotor_mass_adder()

        self.rotor.blade_mass = 17000.
        self.rotor.hub_system_mass = 35000.
        self.rotor.blade_number = 3

    def test_functionality(self):
    
        self.rotor.run()
        
        self.assertEqual(round(self.rotor.rotor_mass,2), 86000.00) 

    def test_gradient(self):

        check_gradient_unit_test(self, self.rotor)


# Turbine Components
class Test_tcc_csm_assembly(unittest.TestCase):

    def setUp(self):

        self.trb = tcc_csm_assembly()

        self.trb.rotor_diameter = 126.0
        self.trb.advanced_blade = True
        self.trb.blade_number = 3
        self.trb.hub_height = 90.0
        self.trb.machine_rating = 5000.0
        self.trb.offshore = True
        self.trb.year = 2009
        self.trb.month = 12
        self.trb.drivetrain_design = 'geared'
    
        # Rotor force calculations for nacelle inputs
        maxTipSpd = 80.0
        maxEfficiency = 0.90201
        ratedWindSpd = 11.5064
        thrustCoeff = 0.50
        airDensity = 1.225
    
        ratedHubPower  = self.trb.machine_rating / maxEfficiency 
        rotorSpeed     = (maxTipSpd/(0.5*self.trb.rotor_diameter)) * (60.0 / (2*np.pi))
        self.trb.rotor_thrust  = airDensity * thrustCoeff * np.pi * self.trb.rotor_diameter**2 * (ratedWindSpd**2) / 8
        self.trb.rotor_torque = ratedHubPower/(rotorSpeed*(np.pi/30))*1000

    def test_functionality(self):
    
        self.trb.run()
        
        self.assertEqual(round(self.trb.turbine_cost,2), 5925727.43)

#----------------------------------------------------

if __name__ == "__main__":
    unittest.main()
