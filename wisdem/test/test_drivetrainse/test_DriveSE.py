
"""
test_DriveSE.py

Created by Katherine Dykes on 2014-01-07.
Copyright (c) NREL. All rights reserved.
"""

import unittest
import numpy as np
from math import pi
from commonse.utilities import check_gradient_unit_test

# from drivese.drive_smooth import YawSystemSmooth, BedplateSmooth
from drivese.drive import Drive3pt, Drive4pt, sys_print
from drivese.drivese_components import LowSpeedShaft_drive, Gearbox_drive, MainBearing_drive, SecondBearing_drive, Bedplate_drive, YawSystem_drive, LowSpeedShaft_drive3pt, \
    LowSpeedShaft_drive4pt, Transformer_drive, HighSpeedSide_drive, Generator_drive, NacelleSystemAdder_drive, AboveYawMassAdder_drive, RNASystemAdder_drive
from drivese.hub import Hub_System_Adder_drive, Hub_drive, PitchSystem_drive, Spinner_drive


# Hub Components
# commented out since Hub_system_Adder_drive was implemented instead of Hub_SE
# class Test_HubSE(unittest.TestCase):

#     def setUp(self):

#         self.hub = Hub_System_Adder_drive()

#         self.hub.rotor_diameter = 126.0 # m
#         self.hub.blade_number  = 3
#         self.hub.blade_root_diameter   = 3.542
#         self.hub.machine_rating = 5000.0
#         # self.hub.blade_mass = 17740.0 # kg
    
#         AirDensity= 1.225 # kg/(m^3)
#         Solidity  = 0.0517
#         RatedWindSpeed = 11.05 # m/s
#         self.hub.rotor_bending_moment = (3.06 * pi / 8) * AirDensity * (RatedWindSpeed ** 2) * (Solidity * (self.hub.rotor_diameter ** 3)) / self.hub.blade_number
    
#         self.hub.gamma = 5.0
#         self.hub.MB1_location = np.array([-3.2, 0.0, 1.0])

#     def test_functionality(self):
        
#         self.hub.run()
        
#         self.assertEqual(round(self.hub.hub_system_mass,1), 45025.7)

class Test_Hub(unittest.TestCase):

    def setUp(self):

        self.hub = Hub_drive()

        # self.hub.rotor_diameter = 126.0 # m
        self.hub.blade_number  = 3
        self.hub.blade_root_diameter   = 3.542
        self.hub.machine_rating = 5000.0

        # AirDensity= 1.225 # kg/(m^3)
        # Solidity  = 0.0517
        # RatedWindSpeed = 11.05 # m/s
        # self.hub.rotor_bending_moment = (3.06 * pi / 8) * AirDensity * (RatedWindSpeed ** 2) * (Solidity * (self.hub.rotor_diameter ** 3)) / self.hub.blade_number

    def test_functionality(self):
        
        self.hub.run()
        
        self.assertEqual(round(self.hub.mass,1), 29852.8)


class Test_PitchSystem(unittest.TestCase):

    def setUp(self):

        self.pitch = PitchSystem_drive()

        self.pitch.blade_mass = 17740.0 # kg
        self.pitch.rotor_diameter = 126.0 # m
        self.pitch.blade_number  = 3
        self.pitch.hub_diameter   = 3.542

        AirDensity= 1.225 # kg/(m^3)
        Solidity  = 0.0517
        RatedWindSpeed = 11.05 # m/s
        self.pitch.rotor_bending_moment = (3.06 * pi / 8) * AirDensity * (RatedWindSpeed ** 2) * (Solidity * (self.pitch.rotor_diameter ** 3)) / self.pitch.blade_number

        self.pitch.gamma = 5.0
        self.pitch.MB1_location = np.array([-3.2, 0.0, 1.0])


    def test_functionality(self):
        
        self.pitch.run()
        
        self.assertEqual(round(self.pitch.mass,1), 13362.4)

class Test_Spinner(unittest.TestCase):

    def setUp(self):

        self.spinner = Spinner_drive()

        self.spinner.rotor_diameter = 126.0 # m
        self.spinner.hub_diameter   = 3.542

        self.spinner.gamma = 5.0
        self.spinner.MB1_location = np.array([-3.2, 0.0, 1.0])

    def test_functionality(self):
        
        self.spinner.run()
        
        self.assertEqual(round(self.spinner.mass,1), 1810.5)


# Nacelle Components

class Test_Drive3pt(unittest.TestCase):

    def setUp(self):

        self.nace = Drive3pt()

        self.nace.rotor_diameter = 126.0 # m
        self.nace.rotor_speed = 12.1 # #rpm m/s
        self.nace.machine_rating = 5000.0
        self.nace.DrivetrainEfficiency = 0.95
        self.nace.rotor_torque =  1.5 * (self.nace.machine_rating * 1000 / self.nace.DrivetrainEfficiency) / (self.nace.rotor_speed * (pi / 30)) # 
        self.nace.rotor_thrust = 599610.0 # N
        self.nace.rotor_mass = 0.0 #accounted for in F_z # kg
        self.nace.rotor_speed = 12.1 #rpm
        self.nace.rotor_bending_moment = -16665000.0 # Nm same as rotor_bending_moment_y
        self.nace.rotor_bending_moment_x = 330770.0# Nm
        self.nace.rotor_bending_moment_y = -16665000.0 # Nm
        self.nace.rotor_bending_moment_z = 2896300.0 # Nm
        self.nace.rotor_force_x = 599610.0 # N
        self.nace.rotor_force_y = 186780.0 # N
        self.nace.rotor_force_z = -842710.0 # N
    
        # NREL 5 MW Drivetrain variables
        self.nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
        self.nace.machine_rating = 5000.0 # kW
        self.nace.gear_ratio = 96.76 # 97:1 as listed in the 5 MW reference document
        self.nace.gear_configuration = 'eep' # epicyclic-epicyclic-parallel
        self.nace.crane = True # onboard crane present
        self.nace.shaft_angle = 5.0 #deg
        self.nace.shaft_ratio = 0.10
        self.nace.Np = [3,3,1]
        self.nace.ratio_type = 'optimal'
        self.nace.shaft_type = 'normal'
        self.nace.uptower_transformer=True
        self.nace.shrink_disc_mass = 333.3*self.nace.machine_rating/1000.0 # estimated
        self.nace.carrier_mass = 8000.0 # estimated
        self.nace.mb1Type = 'SRB'
        self.nace.mb2Type = 'SRB'
        self.nace.flange_length = 0.5
        self.nace.overhang = 5.0
        self.nace.L_rb = 1.912 # length from hub center to main bearing, leave zero if unknow
    
        self.nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs
    
        # NREL 5 MW Tower Variables
        self.nace.tower_top_diameter = 3.78 # m

    def test_functionality(self):
        
        self.nace.run()
        
        self.assertEqual(round(self.nace.nacelle_mass,1), 191196.9)

class Test_Drive4pt(unittest.TestCase):

    def setUp(self):

        self.nace = Drive4pt()
        self.nace.rotor_diameter = 126.0 # m
        self.nace.rotor_speed = 12.1 # #rpm m/s
        self.nace.machine_rating = 5000.0
        self.nace.DrivetrainEfficiency = 0.95
        self.nace.rotor_torque =  1.5 * (self.nace.machine_rating * 1000 / self.nace.DrivetrainEfficiency) / (self.nace.rotor_speed * (pi / 30)) # 6.35e6 #4365248.74 # Nm
        self.nace.rotor_thrust = 599610.0 # N
        self.nace.rotor_mass = 0.0 #accounted for in F_z # kg
        self.nace.rotor_speed = 12.1 #rpm
        self.nace.rotor_bending_moment = -16665000.0 # Nm same as rotor_bending_moment_y
        self.nace.rotor_bending_moment_x = 330770.0# Nm
        self.nace.rotor_bending_moment_y = -16665000.0 # Nm
        self.nace.rotor_bending_moment_z = 2896300.0 # Nm
        self.nace.rotor_force_x = 599610.0 # N
        self.nace.rotor_force_y = 186780.0 # N
        self.nace.rotor_force_z = -842710.0 # N
    
        # NREL 5 MW Drivetrain variables
        self.nace.drivetrain_design = 'geared' # geared 3-stage Gearbox with induction generator machine
        self.nace.machine_rating = 5000.0 # kW
        self.nace.gear_ratio = 96.76 # 97:1 as listed in the 5 MW reference document
        self.nace.gear_configuration = 'eep' # epicyclic-epicyclic-parallel
        self.nace.crane = True # onboard crane present
        self.nace.shaft_angle = 5.0 #deg
        self.nace.shaft_ratio = 0.10
        self.nace.Np = [3,3,1]
        self.nace.ratio_type = 'optimal'
        self.nace.shaft_type = 'normal'
        self.nace.uptower_transformer=False
        self.nace.shrink_disc_mass = 333.3*self.nace.machine_rating/1000.0 # estimated
        self.nace.carrier_mass = 8000.0 # estimated
        self.nace.mb1Type = 'CARB'
        self.nace.mb2Type = 'SRB'
        self.nace.flange_length = 0.5 #m
        self.nace.overhang = 5.0
        self.nace.gearbox_cm = 0.1
        self.nace.hss_length = 1.5
        self.nace.L_rb = 1.912 # length from hub center to main bearing, leave zero if unknown

        self.nace.check_fatigue = 0 #0 if no fatigue check, 1 if parameterized fatigue check, 2 if known loads inputs

        self.nace.tower_top_diameter = 3.78 # m

    def test_functionality(self):
        
        self.nace.run()
        #sys_print(self.nace)
        
        self.assertEqual(round(self.nace.nacelle_mass,1), 190223.5)

'''
class Test_LowSpeedShaft(unittest.TestCase):

    def setUp(self):

        self.lss = LowSpeedShaft()

        self.lss.rotor_diameter = 126. # rotor diameter [m]
        rotor_speed = 12.1
        DrivetrainEfficiency = 0.95
        machine_rating = 5000.0
        self.lss.rotor_torque = 1.5 * (machine_rating * 1000. / DrivetrainEfficiency) / (rotor_speed * (pi / 30.))
        self.lss.rotor_mass = 142585.75 # rotor mass [kg]

    def test_functionality(self):
        
        self.lss.run()
        
        self.assertEqual(round(self.lss.mass,1), 42381.5)

class Test_MainBearing(unittest.TestCase):

    def setUp(self):

        self.mb = MainBearing()

        self.mb.lss_design_torque = 18691533.1165
        self.mb.lss_diameter = 1.2115
        self.mb.lss_mass = 42381.5447
        self.mb.rotor_speed = 12.1
        self.mb.rotor_diameter = 126.0

    def test_functionality(self):
        
        self.mb.run()
        
        self.assertEqual(round(self.mb.mass,1), 7348.5)

class Test_SecondBearing(unittest.TestCase):

    def setUp(self):

        self.sb = MainBearing()

        self.sb.lss_design_torque = 18691533.1165
        self.sb.lss_diameter = 1.2115
        self.sb.lss_mass = 42381.5447
        self.sb.rotor_speed = 12.1
        self.sb.rotor_diameter = 126.0

    def test_functionality(self):
        
        self.sb.run()
        
        self.assertEqual(round(self.sb.mass,1), 7348.5)
'''

class Test_Gearbox_drive(unittest.TestCase):

    def setUp(self):

        self.gbx = Gearbox_drive()

        self.gbx.gear_ratio = 96.76 # 97:1 as listed in the 5 MW reference document
        self.gbx.gear_configuration = 'eep' # epicyclic-epicyclic-parallel
        self.gbx.Np = [3,3,1]
        self.gbx.ratio_type = 'optimal'
        self.gbx.shaft_type = 'normal'
        self.gbx.cm_input = 0.1
        self.gbx.hss_length = 1.5

        self.gbx.rotor_diameter = 126.0 # m
        self.gbx.rotor_speed = 12.1 # #rpm m/s
        self.gbx.machine_rating = 5000.0
        self.gbx.DrivetrainEfficiency = 0.95
        self.gbx.rotor_torque =  1.5 * (self.gbx.machine_rating * 1000 / self.gbx.DrivetrainEfficiency) / (self.gbx.rotor_speed * (pi / 30)) # 6.35e6 #4365248.74 # Nm

    def test_functionality(self):
        
        self.gbx.run()
        
        self.assertEqual(round(self.gbx.mass,1), 55658.3)

class Test_HighSpeedSide(unittest.TestCase):

    def setUp(self):

        self.hss = HighSpeedSide_drive()

        self.hss.gear_ratio = 96.76
        self.hss.rotor_diameter = 126. # rotor diameter [m]
        rotor_speed = 12.1
        DrivetrainEfficiency = 0.95
        machine_rating = 5000.0
        self.hss.rotor_torque = 1.5 * (machine_rating * 1000. / DrivetrainEfficiency) / (rotor_speed * (pi / 30.))
        self.hss.lss_diameter = 1.2115

        self.hss.gearbox_length = 1.512
        self.hss.gearbox_height = 1.89
        self.hss.gearbox_cm = np.array([0.1, 0., 0.756])
        self.hss.length_in = 1.5

    def test_functionality(self):
        
        self.hss.run()
        
        self.assertEqual(round(self.hss.mass,1), 2414.7)

class Test_Generator(unittest.TestCase):

    def setUp(self):

        self.gen = Generator_drive()

        self.gen.gear_ratio = 96.76
        self.gen.rotor_diameter = 126. # rotor diameter [m]
        self.gen.machine_rating = 5000.
        self.gen.drivetrain_design = 'geared'

        self.gen.highSpeedSide_length = 1.5
        self.gen.highSpeedSide_cm = np.array([1.606, 0., 1.134])
        self.gen.rotor_speed = 12.1

    def test_functionality(self):
        
        self.gen.run()
        
        self.assertEqual(round(self.gen.mass,1), 16699.9)


class Test_Bedplate(unittest.TestCase):

    def setUp(self):

        self.bpl = Bedplate_drive()

        self.bpl.gbx_length = 1.512
        self.bpl.gbx_location = 0.0
        self.bpl.gbx_mass = 0.0
        self.bpl.hss_location = 1.606
        self.bpl.hss_mass = 2414.6771802
        self.bpl.generator_location  = 4.057
        self.bpl.generator_mass  = 16699.851325
        self.bpl.lss_location  = -2.52145326316
        self.bpl.lss_mass  = 18564.9450561
        self.bpl.lss_length  = 3.165
        self.bpl.mb1_location = -3.18633453315
        self.bpl.FW_mb1 = 0.25
        self.bpl.mb1_mass  = 4438.51851852
        self.bpl.mb2_location = -0.745657522828
        self.bpl.mb2_mass = 1406.85185185
        self.bpl.transformer_mass = 0.0
        self.bpl.transformer_location = 0.0
        self.bpl.tower_top_diameter = 3.78
        self.bpl.rotor_diameter = 126.0
        self.bpl.machine_rating = 5000.0
        self.bpl.rotor_mass  = 0.0
        self.bpl.rotor_bending_moment_y = -16665000.0
        self.bpl.rotor_force_z  = -842710.0
        self.bpl.flange_length  = 0.5
        self.bpl.L_rb = 1.912
        self.bpl.overhang  = 5.0
    
        #parameters
        self.bpl.uptower_transformer = True

    def test_functionality(self):
        
        self.bpl.run()
        
        self.assertEqual(round(self.bpl.mass,1), 64485.7)

class Test_YawSystem(unittest.TestCase):

    def setUp(self):

        self.yaw = YawSystem_drive()

        #variables
        self.yaw.rotor_diameter = 126.
        self.yaw.rotor_thrust = 599610.0
        self.yaw.tower_top_diameter = 3.78
        self.yaw.above_yaw_mass = 164945.719525
        self.yaw.bedplate_height = 1.4496
    
        #parameters
        self.yaw.yaw_motors_number = 0

    def test_functionality(self):
        
        self.yaw.run()
        
        self.assertEqual(round(self.yaw.mass,1), 6044.7)

class Test_Transformer(unittest.TestCase):

    def setUp(self):

        self.trans = Transformer_drive()

        self.trans.machine_rating = 5000.0
        self.trans.uptower_transformer = False
        self.trans.tower_top_diameter =3.78
        self.trans.rotor_mass = 0.0
        self.trans.overhang = 5.0
        self.trans.generator_cm = np.array([4.057, 0.0, 1.134])
        self.trans.rotor_diameter = 126.0
        self.trans.RNA_mass = 217013.124235
        self.trans.RNA_cm = -2.64480400163

    def test_functionality(self):
        
        self.trans.run()
        
        self.assertEqual(round(self.trans.mass,1), 0.0)


'''
class Test_AboveYawMassAdder(unittest.TestCase):

    def setUp(self):

        self.yawadder = AboveYawMassAdder()

        self.yawadder.machine_rating = 5000.0
        self.yawadder.lss_mass = 42381.5
        self.yawadder.main_bearing_mass = 14696.2/2.
        self.yawadder.second_bearing_mass = 14696.2/2.
        self.yawadder.gearbox_mass = 48664.7
        self.yawadder.hss_mass = 2414.7
        self.yawadder.generator_mass = 16699.9
        self.yawadder.bedplate_mass = 108512.5
        self.yawadder.bedplate_length = 10.4006
        self.yawadder.bedplate_width = 5.20032
        self.yawadder.crane = True

    def test_functionality(self):
        
        self.yawadder.run()
        
        self.assertEqual(round(self.yawadder.above_yaw_mass,1), 259430.9)


class Test_NacelleSystemAdder(unittest.TestCase):

    def setUp(self):

        self.nac = NacelleSystemAdder()

    def test_functionality(self):

        self.nac.above_yaw_mass = 259430.9
        self.nac.yawMass = 13789.0
        self.nac.machine_rating = 5000.0
        self.nac.lss_mass = 42381.5
        self.nac.main_bearing_mass = 14696.2/2.
        self.nac.second_bearing_mass = 14696.2/2.
        self.nac.gearbox_mass = 48664.7
        self.nac.hss_mass = 2414.7
        self.nac.generator_mass = 16699.9
        self.nac.bedplate_mass = 108512.5
        self.nac.mainframe_mass = 125076.5
        self.nac.lss_cm = np.random.rand(3)  # np.array([-2.0, 1.0, 1.0])
        self.nac.main_bearing_cm = np.random.rand(3)  # np.array([-2.0, 1.0, 1.0])
        self.nac.second_bearing_cm = np.random.rand(3)  # np.array([-2.0, 1.0, 1.0])
        self.nac.gearbox_cm = np.random.rand(3)  # np.array([-2.0, 1.0, 1.0])
        self.nac.hss_cm = np.random.rand(3)  # np.array([-2.0, 1.0, 1.0])
        self.nac.generator_cm = np.random.rand(3)  # np.array([-2.0, 1.0, 1.0])
        self.nac.bedplate_cm = np.random.rand(3)  # np.array([-2.0, 1.0, 1.0])
        self.nac.lss_I = np.random.rand(3)  # np.array([1000., 1000., 1000.])
        self.nac.main_bearing_I = np.random.rand(3)  # np.array([1000., 1000., 1000.])
        self.nac.second_bearing_I = np.random.rand(3)  # np.array([1000., 1000., 1000.])
        self.nac.gearbox_I = np.random.rand(3)  # np.array([1000., 1000., 1000.])
        self.nac.hss_I = np.random.rand(3)  # np.array([1000., 1000., 1000.])
        self.nac.generator_I = np.random.rand(3)  # np.array([1000., 1000., 1000.])
        self.nac.bedplate_I = np.random.rand(3)  # np.array([1000., 1000., 1000.])

        self.nac.run()

        self.assertEqual(round(self.nac.nacelle_mass,1), 273219.9)

'''

# Gradient tests for drive smooth
# TODO: no unit tests since drive_smooth needs updating
# class TestBearingSmooth(unittest.TestCase):

#     def test_gradient(self):
#         comp = BearingSmooth()
#         comp.bearing_type = 'SRB'
#         comp.lss_diameter = 0.721049014299
#         comp.rotor_diameter = 125.740528176
#         comp.bearing_switch = 'main'

#         check_gradient_unit_test(self, comp)


# class TestYawSystemSmooth(unittest.TestCase):

#     def test_gradient(self):
#         comp = YawSystemSmooth()
#         comp.rotor_diameter = 125.740528176
#         comp.tower_top_diameter = 3.87

#         check_gradient_unit_test(self, comp)


# class TestBedplateSmooth(unittest.TestCase):

#     def test_gradient(self):
#         comp = BedplateSmooth()
#         comp.hss_location = 0.785878301101
#         comp.hss_mass = 2288.26758514
#         comp.generator_location = 1.5717566022
#         comp.generator_mass = 16699.851325
#         comp.lss_location = -3.14351320441
#         comp.lss_mass = 12546.3193435
#         comp.mb1_location = -1.25740528176
#         comp.mb1_mass = 3522.06734168
#         comp.mb2_location = -4.40091848617
#         comp.mb2_mass = 5881.81400444
#         comp.tower_top_diameter = 3.87
#         comp.rotor_diameter = 125.740528176
#         comp.machine_rating = 5000.0
#         comp.rotor_mass = 93910.5225629
#         comp.rotor_bending_moment_y = -2325000.0
#         comp.rotor_force_z = -921262.226342
#         comp.h0_rear = 1.35
#         comp.h0_front = 1.7

#         check_gradient_unit_test(self, comp)
        
if __name__ == "__main__":
    unittest.main()
    