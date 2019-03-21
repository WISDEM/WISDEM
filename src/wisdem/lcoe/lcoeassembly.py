# LCOE assembly for DNV GL Supersized Blades analysis
import numpy as np

# openmdao imports
from openmdao.api import Component, Problem, Group, IndepVarComp, ScipyOptimizer, DumpRecorder

# FUSED wrapper
from fusedwind.fused_openmdao import FUSED_Component, FUSED_Group, FUSED_add, FUSED_connect, FUSED_print, \
                                     FUSED_Problem, FUSED_setup, FUSED_run, FUSED_VarComp

# LCOE analysis model imports
# FUSED drivetrain components
from drivese.fused_drivese import FUSED_Gearbox, FUSED_MainBearing, FUSED_Bedplate, FUSED_YawSystem, FUSED_LowSpeedShaft3pt, \
    FUSED_LowSpeedShaft4pt, FUSED_Transformer, FUSED_HighSpeedSide, FUSED_Generator, FUSED_NacelleSystemAdder, FUSED_AboveYawMassAdder, FUSED_RNASystemAdder

from drivese.fused_hubse import FUSED_Hub, FUSED_PitchSystem, FUSED_Spinner, FUSED_Hub_System_Adder

# tower model imports
from towerse.tower import TowerSE
from commonse.environment import PowerWind, LogWind
from commonse.rna import RNAMass

# global tower variables
nPoints = 3
nFull   = 5*(nPoints-1) + 1
wind = 'PowerWind'
nLC = 2
nDEL = 35

# turbine cost model imports
from turbine_costsse.turbine_costsse_2015 import Turbine_CostsSE_2015


# create workflow for LCOE
def example_task37_lcoe(lcoe_group, mb1Type, mb2Type, IEC_Class, gear_configuration, shaft_factor, drivetrain_design, uptower_transformer, yaw_motors_number, crane, blade_number):

    # Add common inputs
    # Add common inputs for rotor
    FUSED_add(lcoe_group, 'rotorvars',FUSED_VarComp([('rotor_diameter', 0.0),
                                                     ('rotor_bending_moment_x', 0.0),
                                                     ('rotor_bending_moment_y', 0.0),
                                                     ('rotor_bending_moment_z', 0.0),
                                                     ('rotor_thrust', 0.0),
                                                     ('rotor_force_y',0.0),
                                                     ('rotor_force_z', 0.0),
                                                     ('rotor_torque', 0.0),
                                                     ('rotor_mass', 0.0), # remove if using rna component
                                                     ('blade_mass', 0.0),                                                  
                                                     ]),['*'])

    # Add common inputs for drivetrain
    FUSED_add(lcoe_group, 'drivevars',FUSED_VarComp([('machine_rating', 0.0),
                                                     ('gear_ratio', 0.0),
                                                     ('flange_length', 0.0),
                                                     ('overhang', 0.0),
                                                     ('distance_hub2mb', 0.0),
                                                     ('gearbox_input_cm', 0.0),
                                                     ('hss_input_length', 0.0),
                                                     ]),['*'])

    # Add common inputs for tower
    FUSED_add(lcoe_group, 'towervars',FUSED_VarComp([('tower_top_diameter', 0.0)]),['*'])

    ###
    # Select components
    # hub
    FUSED_add(lcoe_group, 'hub', FUSED_Component(FUSED_Hub(blade_number)), ['*'])
    FUSED_add(lcoe_group, 'pitchSystem', FUSED_Component(FUSED_PitchSystem(blade_number)), ['*'])
    FUSED_add(lcoe_group, 'spinner', FUSED_Component(FUSED_Spinner()), ['*'])
    FUSED_add(lcoe_group, 'adder', FUSED_Component(FUSED_Hub_System_Adder()), ['*'])

    # drivetrain
    FUSED_add(lcoe_group, 'lowSpeedShaft', FUSED_Component(FUSED_LowSpeedShaft4pt(mb1Type, mb2Type, IEC_Class)), ['*'])
    FUSED_add(lcoe_group, 'mainBearing', FUSED_Component(FUSED_MainBearing('main')), ['lss_design_torque','rotor_diameter']) #explicit connections for bearings
    FUSED_add(lcoe_group, 'secondBearing', FUSED_Component(FUSED_MainBearing('second')), ['lss_design_torque','rotor_diameter']) #explicit connections for bearings
    FUSED_add(lcoe_group, 'gearbox', FUSED_Component(FUSED_Gearbox(gear_configuration, shaft_factor)), ['*'])
    FUSED_add(lcoe_group, 'highSpeedSide', FUSED_Component(FUSED_HighSpeedSide()), ['*'])
    FUSED_add(lcoe_group, 'generator', FUSED_Component(FUSED_Generator(drivetrain_design)), ['*'])
    FUSED_add(lcoe_group, 'bedplate', FUSED_Component(FUSED_Bedplate(uptower_transformer)), ['*'])
    FUSED_add(lcoe_group, 'transformer', FUSED_Component(FUSED_Transformer(uptower_transformer)), ['*'])
    FUSED_add(lcoe_group, 'rna', FUSED_Component(FUSED_RNASystemAdder()), ['*'])
    FUSED_add(lcoe_group, 'above_yaw_massAdder', FUSED_Component(FUSED_AboveYawMassAdder(crane)), ['*'])
    FUSED_add(lcoe_group, 'yawSystem', FUSED_Component(FUSED_YawSystem(yaw_motors_number)), ['*'])
    FUSED_add(lcoe_group, 'nacelleSystem', FUSED_Component(FUSED_NacelleSystemAdder()), ['*'])

    # rna
    FUSED_add(lcoe_group, 'rnamass', RNAMass(), ['*'])

    # tower
    FUSED_add(lcoe_group, 'tower', TowerSE(nLC, nPoints, nFull, nDEL, wind=wind), ['*'])

    # turbine costs
    FUSED_add(lcoe_group, 'tcc', Turbine_CostsSE_2015(), ['*'])

    # lcoe 
    FUSED_add(lcoe_group, 'lcoe', LCOEcalc(), ['*'])

    ###
    # Connect nacelle components where explicit connections needed (for main bearings)
    FUSED_connect(lcoe_group, 'lss_mb1_mass', ['mainBearing.bearing_mass'])
    FUSED_connect(lcoe_group, 'lss_diameter1', ['mainBearing.lss_diameter', 'lss_diameter'])
    FUSED_connect(lcoe_group, 'lss_mb1_cm', ['mainBearing.lss_mb_cm'])
    FUSED_connect(lcoe_group, 'mainBearing.mb_mass', ['mb1_mass'])
    FUSED_connect(lcoe_group, 'mainBearing.mb_cm', ['mb1_cm'])
    FUSED_connect(lcoe_group, 'mainBearing.mb_I', ['mb1_I'])

    FUSED_connect(lcoe_group, 'lss_mb2_mass', ['secondBearing.bearing_mass'])
    FUSED_connect(lcoe_group, 'lss_diameter2', ['secondBearing.lss_diameter'])
    FUSED_connect(lcoe_group, 'lss_mb2_cm', ['secondBearing.lss_mb_cm'])
    FUSED_connect(lcoe_group, 'secondBearing.mb_mass', ['mb2_mass'])
    FUSED_connect(lcoe_group, 'secondBearing.mb_cm', ['mb2_cm'])
    FUSED_connect(lcoe_group, 'secondBearing.mb_I', ['mb2_I'])

    # Connect rna to tower inputs where needed
    #FUSED_connect(lcoe_group, 'rna_mass', 'rna_mass') # mass has same name for tower and rna component
    FUSED_connect(lcoe_group, 'rna_cm', 'rna_cg')
    FUSED_connect(lcoe_group, 'rna_I_TT', 'rna_I')
    # Manually inputing now due to issues with connections
    '''
    FUSED_connect(lcoe_group, 'hub_height - nacelle_cm[2]', 'tower_section_height[-1]')
    FUSED_connect(lcoe_group, 'rotor_thrust', 'pre1.rna_F', [0])
    FUSED_connect(lcoe_group, 'rotor_force_y', 'pre1.rna_F', [1])
    FUSED_connect(lcoe_group, 'rotor_force_z', 'pre1.rna_F', [2])
    FUSED_connect(lcoe_group, 'rotor_bending_moment_x', 'pre1.rna_M', [0])
    FUSED_connect(lcoe_group, 'rotor_bending_moment_y', 'pre1.rna_M', [1])
    FUSED_connect(lcoe_group, 'rotor_bending_moment_z', 'pre1.rna_M', [2])
    '''   

# adjusting from common se for this particular application
class RNAMass(Component):
    def __init__(self):

        super(RNAMass, self).__init__()
        # variables
        self.add_param('blade_mass', 0.0, units='kg', desc='mass of all blade')
        self.add_param('hub_system_mass', 0.0, units='kg', desc='mass of hub')
        self.add_param('nacelle_mass', 0.0, units='kg', desc='mass of nacelle')

        self.add_param('hub_system_cm', np.zeros(3), units='m', desc='location of hub center of mass relative to tower top in yaw-aligned c.s.')
        self.add_param('nacelle_cm', np.zeros(3), units='m', desc='location of nacelle center of mass relative to tower top in yaw-aligned c.s.')

        # order for all moments of inertia is (xx, yy, zz, xy, xz, yz) in the yaw-aligned coorinate system
        self.add_param('blades_I', np.zeros(3), units='kg*m**2', desc='mass moments of inertia of all blades about hub center')
        self.add_param('hub_system_I', np.zeros(3), units='kg*m**2', desc='mass moments of inertia of hub about its center of mass')
        self.add_param('nacelle_I', np.zeros(3), units='kg*m**2', desc='mass moments of inertia of nacelle about its center of mass')

        # outputs
        #self.add_output('rotor_mass', 0.0, units='kg', desc='mass of blades and hub')
        self.add_output('rna_mass', 0.0, units='kg', desc='total mass of RNA')
        self.add_output('rna_cm', np.zeros(3), units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')
        self.add_output('rna_I_TT', np.zeros(6), units='kg*m**2', desc='mass moments of inertia of RNA about tower top in yaw-aligned coordinate system')

    def _assembleI(self, I):
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I[0], I[1], I[2], 0.0, 0.0, 0.0
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


    def _unassembleI(self, I):
        return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])


    def solve_nonlinear(self, params, unknowns, resids):

        rotor_mass = params['blade_mass']*3 + params['hub_system_mass'] # change blade number to input when updating
        nac_mass = params['nacelle_mass']

        # rna mass
        #unknowns['rotor_mass'] = rotor_mass # commenting out for now
        unknowns['rna_mass'] = rotor_mass + nac_mass

        # rna cm
        unknowns['rna_cm'] = (rotor_mass*params['hub_system_cm'] + nac_mass*params['nacelle_cm'])/unknowns['rna_mass']

        #TODO check if the use of assembleI and unassembleI functions are correct
        # rna I
        blades_I = self._assembleI(params['blades_I'])
        hub_I = self._assembleI(params['hub_system_I'])
        nac_I = self._assembleI(params['nacelle_I'])
        rotor_I = blades_I + hub_I

        R = params['hub_system_cm']
        rotor_I_TT = rotor_I + rotor_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        R = params['nacelle_cm']
        nac_I_TT = nac_I + params['nacelle_mass']*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        unknowns['rna_I_TT'] = self._unassembleI(rotor_I_TT + nac_I_TT)

# simple lcoe calculator
class LCOEcalc(Component):

    def __init__(self):

        super(LCOEcalc, self).__init__()

        # Variables
        self.add_param('turbine_cost', 0.0, units='USD', desc='turbine capital cost per turbine')
        self.add_param('turbine_number', 0, desc='number of turbines')
        self.add_param('bos_cost', 0.0, units='USD', desc='tower cost')
        self.add_param('opex', 0.0, units='USD', desc='annual operating expenses')
        self.add_param('fcr', 0.0, desc='fixed charge rate')
        self.add_param('aep', 0.0, desc='annual energy production') # consider bringing in AEP calculator if AEP not supplied
    
        # Outputs
        self.add_output('coe', 0.0, units='USD / (kW * hr)', desc='Overall plant cost of energy')

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['coe'] = ((params['turbine_cost']*params['turbine_number'] + params['bos_cost'])*params['fcr'] + params['opex']) / params['aep']

if __name__ == "__main__":

    pass