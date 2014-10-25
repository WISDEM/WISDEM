import math
import numpy as np
from openmdao.main.api import VariableTree, Component, Assembly, set_as_top
from openmdao.main.datatypes.api import Int, Float, Array, VarTree, Slot,Instance, Bool

from commonse.utilities import hstack, vstack
from commonse.csystem import DirectionVector

class RNAMass(Component):

    # variables
    blades_mass = Float(iotype='in', units='kg', desc='mass of all blade')
    hub_mass = Float(iotype='in', units='kg', desc='mass of hub')
    nac_mass = Float(iotype='in', units='kg', desc='mass of nacelle')

    hub_cm = Array(iotype='in', units='m', desc='location of hub center of mass relative to tower top in yaw-aligned c.s.')
    nac_cm = Array(iotype='in', units='m', desc='location of nacelle center of mass relative to tower top in yaw-aligned c.s.')

    # TODO: check on this???
    # order for all moments of inertia is (xx, yy, zz, xy, xz, yz) in the yaw-aligned coorinate system
    blades_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of all blades about hub center')
    hub_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of hub about its center of mass')
    nac_I = Array(iotype='in', units='kg*m**2', desc='mass moments of inertia of nacelle about its center of mass')

    # outputs
    rna_mass = Float(iotype='out', units='kg', desc='total mass of RNA')
    rna_cm = Array(iotype='out', units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')
    rna_I_TT = Array(iotype='out', units='kg*m**2', desc='mass moments of inertia of RNA about tower top in yaw-aligned coordinate system')


    def _assembleI(self, Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


    def _unassembleI(self, I):
        return np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])


    def execute(self):

        self.rotor_mass = self.blades_mass + self.hub_mass
        self.nac_mass = self.nac_mass

        # rna mass
        self.rna_mass = self.rotor_mass + self.nac_mass

        # rna cm
        self.rna_cm = (self.rotor_mass*self.hub_cm + self.nac_mass*self.nac_cm)/self.rna_mass

        # rna I
        blades_I = self._assembleI(*self.blades_I)
        hub_I = self._assembleI(*self.hub_I)
        nac_I = self._assembleI(*self.nac_I)
        rotor_I = blades_I + hub_I

        R = self.hub_cm
        rotor_I_TT = rotor_I + self.rotor_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        R = self.nac_cm
        nac_I_TT = nac_I + self.nac_mass*(np.dot(R, R)*np.eye(3) - np.outer(R, R))

        self.rna_I_TT = self._unassembleI(rotor_I_TT + nac_I_TT)


    def list_deriv_vars(self):

        inputs = ('blades_mass', 'hub_mass', 'nac_mass', 'hub_cm', 'nac_cm', 'blades_I', 'hub_I', 'nac_I')
        outputs = ('rna_mass', 'rna_cm', 'rna_I_TT')

        return inputs, outputs


    def provideJ(self):

        # mass
        dmass = np.hstack([np.array([1.0, 1.0, 1.0]), np.zeros(2*3+3*6)])

        # cm
        top = (self.rotor_mass*self.hub_cm + self.nac_mass*self.nac_cm)
        dcm_dblademass = (self.rna_mass*self.hub_cm - top)/self.rna_mass**2
        dcm_dhubmass = (self.rna_mass*self.hub_cm - top)/self.rna_mass**2
        dcm_dnacmass = (self.rna_mass*self.nac_cm - top)/self.rna_mass**2
        dcm_dhubcm = self.rotor_mass/self.rna_mass*np.eye(3)
        dcm_dnaccm = self.nac_mass/self.rna_mass*np.eye(3)

        dcm = hstack([dcm_dblademass, dcm_dhubmass, dcm_dnacmass, dcm_dhubcm,
            dcm_dnaccm, np.zeros((3, 3*6))])

        # I
        R = self.hub_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        dI_dblademass = const
        dI_dhubmass = const
        dI_drx = self.rotor_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = self.rotor_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = self.rotor_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        dI_dhubcm = np.vstack([dI_drx, dI_dry, dI_drz]).T

        R = self.nac_cm
        const = self._unassembleI(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        dI_dnacmass = const
        dI_drx = self.nac_mass*self._unassembleI(2*R[0]*np.eye(3) - np.array([[2*R[0], R[1], R[2]], [R[1], 0.0, 0.0], [R[2], 0.0, 0.0]]))
        dI_dry = self.nac_mass*self._unassembleI(2*R[1]*np.eye(3) - np.array([[0.0, R[0], 0.0], [R[0], 2*R[1], R[2]], [0.0, R[2], 0.0]]))
        dI_drz = self.nac_mass*self._unassembleI(2*R[2]*np.eye(3) - np.array([[0.0, 0.0, R[0]], [0.0, 0.0, R[1]], [R[0], R[1], 2*R[2]]]))
        dI_dnaccm = np.vstack([dI_drx, dI_dry, dI_drz]).T

        dI_dbladeI = np.eye(6)
        dI_dhubI = np.eye(6)
        dI_dnacI = np.eye(6)

        dI = hstack([dI_dblademass, dI_dhubmass, dI_dnacmass, dI_dhubcm, dI_dnaccm,
            dI_dbladeI, dI_dhubI, dI_dnacI])

        J = np.vstack([dmass, dcm, dI])

        return J


class RotorLoads(Component):

    # variables
    F = Array(np.array([0.0,0.0,0.0]),iotype='in', desc='forces in hub-aligned coordinate system')
    M = Array(np.array([0.0,0.0,0.0]),iotype='in', desc='moments in hub-aligned coordinate system')
    r_hub = Array(iotype='in', desc='position of rotor hub relative to tower top in yaw-aligned c.s.')
    m_RNA = Float(iotype='in', units='kg', desc='mass of rotor nacelle assembly')
    rna_cm = Array(iotype='in', units='m', desc='location of RNA center of mass relative to tower top in yaw-aligned c.s.')

    rna_weightM = Bool(True,iotype='in', units=None, desc='Flag to indicate whether or not the RNA weight should be considered.\
                      An upwind overhang may lead to unconservative estimates due to the P-Delta effect(suggest not using). For downwind turbines set to True. ')

    # These are used for backwards compatibility - do not use
    T = Float(iotype='in', desc='thrust in hub-aligned coordinate system') #THIS MEANS STILL YAWED THOUGH (Shaft tilt)
    Q = Float(iotype='in', desc='torque in hub-aligned coordinate system')

    # parameters
    downwind = Bool(False, iotype='in')
    tilt = Float(iotype='in', units='deg')
    g = Float(9.81, iotype='in', units='m/s**2', desc='Gravity Acceleration (ABSOLUTE VALUE!)')

    # out
    top_F = Array(iotype='out')  # in yaw-aligned
    top_M = Array(iotype='out')

    missing_deriv_policy = 'assume_zero'


    def execute(self):

        if self.T != 0:
            F = [self.T, 0.0, 0.0]
            M = [self.Q, 0.0, 0.0]
        else:
            F = self.F
            M = self.M

        F = DirectionVector.fromArray(F).hubToYaw(self.tilt)
        M = DirectionVector.fromArray(M).hubToYaw(self.tilt)

        # change x-direction if downwind
        r_hub = np.copy(self.r_hub)
        rna_cm = np.copy(self.rna_cm)
        if self.downwind:
            r_hub[0] *= -1
            rna_cm[0] *= -1
        r_hub = DirectionVector.fromArray(r_hub)
        rna_cm = DirectionVector.fromArray(rna_cm)
        self.save_rhub = r_hub
        self.save_rcm = rna_cm

        # aerodynamic moments
        M = M + r_hub.cross(F)
        self.saveF = F


        # add weight loads
        F_w = DirectionVector(0.0, 0.0, -self.m_RNA*self.g)
        M_w = rna_cm.cross(F_w)
        self.saveF_w = F_w

        F += F_w

        if self.rna_weightM:
           M += M_w
        else:
 			#REMOVE WEIGHT EFFECT TO ACCOUNT FOR P-Delta Effect
			print "!!!! No weight effect on rotor moments -TowerSE  !!!!"

		# put back in array
        self.top_F = np.array([F.x, F.y, F.z])
        self.top_M = np.array([M.x, M.y, M.z])


    def list_deriv_vars(self):

        inputs = ('F', 'M', 'r_hub', 'm_RNA', 'rna_cm')
        outputs = ('top_F', 'top_M')

        return inputs, outputs

    def provideJ(self):

        dF = DirectionVector.fromArray(self.F).hubToYaw(self.tilt)
        dFx, dFy, dFz = dF.dx, dF.dy, dF.dz

        dtopF_dFx = np.array([dFx['dx'], dFy['dx'], dFz['dx']])
        dtopF_dFy = np.array([dFx['dy'], dFy['dy'], dFz['dy']])
        dtopF_dFz = np.array([dFx['dz'], dFy['dz'], dFz['dz']])
        dtopF_dF = hstack([dtopF_dFx, dtopF_dFy, dtopF_dFz])
        dtopF_w_dm = np.array([0.0, 0.0, -self.g])

        dtopF = hstack([dtopF_dF, np.zeros((3, 6)), dtopF_w_dm, np.zeros((3, 3))])


        dM = DirectionVector.fromArray(self.M).hubToYaw(self.tilt)
        dMx, dMy, dMz = dM.dx, dM.dy, dM.dz
        dMxcross, dMycross, dMzcross = self.save_rhub.cross_deriv(self.saveF, 'dr', 'dF')

        dtopM_dMx = np.array([dMx['dx'], dMy['dx'], dMz['dx']])
        dtopM_dMy = np.array([dMx['dy'], dMy['dy'], dMz['dy']])
        dtopM_dMz = np.array([dMx['dz'], dMy['dz'], dMz['dz']])
        dtopM_dM = hstack([dtopM_dMx, dtopM_dMy, dtopM_dMz])
        dM_dF = np.array([dMxcross['dF'], dMycross['dF'], dMzcross['dF']])

        dtopM_dFx = np.dot(dM_dF, dtopF_dFx)
        dtopM_dFy = np.dot(dM_dF, dtopF_dFy)
        dtopM_dFz = np.dot(dM_dF, dtopF_dFz)
        dtopM_dF = hstack([dtopM_dFx, dtopM_dFy, dtopM_dFz])
        dtopM_dr = np.array([dMxcross['dr'], dMycross['dr'], dMzcross['dr']])

        dMx_w_cross, dMy_w_cross, dMz_w_cross = self.save_rcm.cross_deriv(self.saveF_w, 'dr', 'dF')

        dtopM_drnacm = np.array([dMx_w_cross['dr'], dMy_w_cross['dr'], dMz_w_cross['dr']])
        dtopM_dF_w = np.array([dMx_w_cross['dF'], dMy_w_cross['dF'], dMz_w_cross['dF']])
        dtopM_dm = np.dot(dtopM_dF_w, dtopF_w_dm)

        if self.downwind:
            dtopM_dr[:, 0] *= -1
            dtopM_drnacm[:, 0] *= -1

        dtopM = hstack([dtopM_dF, dtopM_dM, dtopM_dr, dtopM_dm, dtopM_drnacm])

        J = vstack([dtopF, dtopM])

        return J