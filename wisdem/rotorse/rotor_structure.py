from __future__ import print_function

import numpy as np
from scipy.optimize import curve_fit
import os, copy
from openmdao.api import IndepVarComp, ExplicitComponent, Group, Problem, ExecComp
from wisdem.ccblade.ccblade_component import CCBladePower, CCBladeLoads, CCBladeGeometry
from wisdem.commonse import gravity, NFREQ
from wisdem.commonse.csystem import DirectionVector
from wisdem.commonse.utilities import trapz_deriv, interp_with_deriv
from wisdem.commonse.akima import Akima, akima_interp_with_derivs
import wisdem.pBeam._pBEAM as _pBEAM
import wisdem.ccblade._bem as _bem

from wisdem.rotorse import RPM2RS, RS2RPM
from wisdem.rotorse.rotor_geometry import RotorGeometry
from wisdem.rotorse.rotor_geometry_yaml import ReferenceBlade
from wisdem.rotorse.precomp import _precomp
        
# ---------------------
# Components
# ---------------------

        
class BladeCurvature(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
    
    def setup(self):
        NPTS = self.options['NPTS']

        self.add_input('r', val=np.zeros(NPTS), units='m', desc='location in blade z-coordinate')
        self.add_input('precurve', val=np.zeros(NPTS), units='m', desc='location in blade x-coordinate')
        self.add_input('presweep', val=np.zeros(NPTS), units='m', desc='location in blade y-coordinate')
        self.add_input('precone', val=0.0, units='deg', desc='precone angle')

        self.add_output('totalCone', val=np.zeros(NPTS), units='deg', desc='total cone angle from precone and curvature')
        self.add_output('x_az', val=np.zeros(NPTS), units='m', desc='location of blade in azimuth x-coordinate system')
        self.add_output('y_az', val=np.zeros(NPTS), units='m', desc='location of blade in azimuth y-coordinate system')
        self.add_output('z_az', val=np.zeros(NPTS), units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_output('s', val=np.zeros(NPTS), units='m', desc='cumulative path length along blade')

        #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):

        # x_az, y_az, z_az, cone, s = _bem.definecurvature(r, precurve, presweep, 0.0)
        r = inputs['r']
        precurve = inputs['precurve']
        presweep = inputs['presweep']
        precone = inputs['precone']

        n = len(r)
        dx_dx = np.eye(3*n)

        x_az, x_azd, y_az, y_azd, z_az, z_azd, cone, coned, s, sd = _bem.definecurvature_dv2(r, dx_dx[:, :n],
                                                                                             precurve, dx_dx[:, n:2*n],
                                                                                             presweep, dx_dx[:, 2*n:],
                                                                                             0.0, np.zeros(3*n))

        totalCone = precone + np.degrees(cone)
        s = r[0] + s

        outputs['totalCone'] = totalCone
        outputs['x_az'] = x_az
        outputs['y_az'] = y_az
        outputs['z_az'] = z_az
        outputs['s'] = s
        '''
        dxaz_dr = x_azd[:n, :].T
        dxaz_dprecurve = x_azd[n:2*n, :].T
        dxaz_dpresweep = x_azd[2*n:, :].T

        dyaz_dr = y_azd[:n, :].T
        dyaz_dprecurve = y_azd[n:2*n, :].T
        dyaz_dpresweep = y_azd[2*n:, :].T

        dzaz_dr = z_azd[:n, :].T
        dzaz_dprecurve = z_azd[n:2*n, :].T
        dzaz_dpresweep = z_azd[2*n:, :].T

        dcone_dr = np.degrees(coned[:n, :]).T
        dcone_dprecurve = np.degrees(coned[n:2*n, :]).T
        dcone_dpresweep = np.degrees(coned[2*n:, :]).T

        ds_dr = sd[:n, :].T
        ds_dr[:, 0] += 1
        ds_dprecurve = sd[n:2*n, :].T
        ds_dpresweep = sd[2*n:, :].T

        J = {}
        J['x_az', 'r'] = dxaz_dr
        J['x_az', 'precurve'] = dxaz_dprecurve
        J['x_az', 'presweep'] = dxaz_dpresweep
        J['x_az', 'precone'] = np.zeros(n)
        J['y_az', 'r'] = dyaz_dr
        J['y_az', 'precurve'] = dyaz_dprecurve
        J['y_az', 'presweep'] = dyaz_dpresweep
        J['y_az', 'precone'] = np.zeros(n)
        J['z_az', 'r'] = dzaz_dr
        J['z_az', 'precurve'] = dzaz_dprecurve
        J['z_az', 'presweep'] = dzaz_dpresweep
        J['z_az', 'precone'] = np.zeros(n)
        J['totalCone', 'r'] = dcone_dr
        J['totalCone', 'precurve'] = dcone_dprecurve
        J['totalCone', 'presweep'] = dcone_dpresweep
        J['totalCone', 'precone'] = np.ones(n)
        J['s', 'r'] = ds_dr
        J['s', 'precurve'] = ds_dprecurve
        J['s', 'presweep'] = ds_dpresweep
        J['s', 'precone'] = np.zeros(n)

        self.J = J

    def compute_partials(self, inputs, J):
        J = self.J
        '''

class CurveFEM(ExplicitComponent):
    """natural frequencies for curved blades"""
    def initialize(self):
        self.options.declare('NPTS')
    
    def setup(self):
        NPTS = self.options['NPTS']

        self.add_input('Omega', val=0.0, units='rpm', desc='rotor rotation frequency')
        self.add_input('z', val=np.zeros(NPTS), units='m', desc='locations of properties along beam')
        self.add_input('EA', val=np.zeros(NPTS), units='N', desc='axial stiffness')
        self.add_input('EIxx', val=np.zeros(NPTS), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('EIyy', val=np.zeros(NPTS), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('EIxy', val=np.zeros(NPTS), units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_input('GJ', val=np.zeros(NPTS), units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_input('rhoA', val=np.zeros(NPTS), units='kg/m', desc='mass per unit length')
        self.add_input('rhoJ', val=np.zeros(NPTS), units='kg*m', desc='polar mass moment of inertia per unit length')
        self.add_input('x_ec', val=np.zeros(NPTS), units='m', desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_input('y_ec', val=np.zeros(NPTS), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_input('Tw_iner', val=np.zeros(NPTS), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')
        self.add_input('flap_iner', val=np.zeros(NPTS), units='kg/m', desc='Section flap inertia about the Y_G axis per unit length.')
        self.add_input('edge_iner', val=np.zeros(NPTS), units='kg/m', desc='Section lag inertia about the X_G axis per unit length')
        self.add_input('theta', val=np.zeros(NPTS), units='deg', desc='structural twist distribution')
        self.add_input('precurve', val=np.zeros(NPTS), units='m', desc='structural precuve (see FAST definition)')
        self.add_input('presweep', val=np.zeros(NPTS), units='m', desc='structural presweep (see FAST definition)')

        self.add_output('freq_curvefem', val=np.zeros(NFREQ), units='Hz', desc='first nF natural frequencies')
        self.add_output('modes_coef', val=np.zeros((3, 5)), desc='mode shapes as 6th order polynomials, in the format accepted by ElastoDyn, [[c_x2, c_],..]')


    def compute(self, inputs, outputs):

        mycurve = _pBEAM.CurveFEM(inputs['Omega'], inputs['Tw_iner'], inputs['z'], inputs['precurve'], inputs['presweep'], inputs['rhoA'], True)
        # mycurve = _pBEAM.CurveFEM(inputs['Omega'], inputs['theta'], inputs['z'], inputs['precurve'], inputs['presweep'], inputs['rhoA'], True)
        n = len(inputs['z'])
        freq, eig_vec = mycurve.frequencies(inputs['EA'], inputs['EIxx'], inputs['EIyy'], inputs['GJ'], inputs['rhoJ'], n)
        outputs['freq_curvefem'] = freq[:NFREQ]
        
        # Parse eigen vectors
        R = inputs['z']
        R = np.asarray([(Ri-R[0])/(R[-1]-R[0]) for Ri in R])
        ndof = 6

        flap = np.zeros((NFREQ, n))
        edge = np.zeros((NFREQ, n))
        for i in range(NFREQ):
            eig_vec_i = eig_vec[:,i]
            for j in range(n):
                flap[i,j] = eig_vec_i[0+j*ndof]
                edge[i,j] = eig_vec_i[1+j*ndof]


        # Mode shape polynomial fit
        def mode_fit(x, a, b, c, d, e):
            return a*x**2. + b*x**3. + c*x**4. + d*x**5. + e*x**6.
        # First Flapwise
        coef, pcov = curve_fit(mode_fit, R, flap[0,:])
        coef_norm = [c/sum(coef) for c in coef]
        outputs['modes_coef'][0,:] = coef_norm
        # Second Flapwise
        coef, pcov = curve_fit(mode_fit, R, flap[1,:])
        coef_norm = [c/sum(coef) for c in coef]
        outputs['modes_coef'][1,:] = coef_norm
        # First Edgewise
        coef, pcov = curve_fit(mode_fit, R, edge[0,:])
        coef_norm = [c/sum(coef) for c in coef]
        outputs['modes_coef'][2,:] = coef_norm


        # # temp
        # from bmodes import BModes_tools
        # r = np.asarray([(ri-inputs['z'][0])/(inputs['z'][-1]-inputs['z'][0]) for ri in inputs['z']])
        # prop = np.column_stack((r, inputs['theta'], inputs['Tw_iner'], inputs['rhoA'], inputs['flap_iner'], inputs['edge_iner'], inputs['EIyy'], \
        #         inputs['EIxx'], inputs['GJ'], inputs['EA'], np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)))
        
        # bm = BModes_tools()
        # bm.setup.radius = inputs['z'][-1]
        # bm.setup.hub_rad = inputs['z'][0]
        # bm.setup.precone = -2.5
        # bm.prop = prop
        # bm.exe_BModes = 'C:/Users/egaertne/WT_Codes/bModes/BModes.exe'
        # bm.execute()
        # print(bm.freq)

        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12., 6.), sharex=True, sharey=True)
        # # fig.subplots_adjust(bottom=0.2, top=0.9)
        # fig.subplots_adjust(bottom=0.15, left=0.1, hspace=0, wspace=0)
        # i = 0
        # k_flap = bm.flap_disp[i,-1]/flap[i,-1]
        # k_edge = bm.lag_disp[i,-1]/edge[i,-1]
        # ax[0,0].plot(R, flap[i,:]*k_flap ,'k',label='CurveFEM')
        # ax[0,0].plot(bm.r[i,:], bm.flap_disp[i,:],'bx',label='BModes')
        # ax[0,0].set_ylabel('Flapwise Disp.')
        # ax[0,0].set_title('1st Mode')
        # ax[1,0].plot(R, edge[i,:]*k_edge ,'k')
        # ax[1,0].plot(bm.r[i,:], bm.lag_disp[i,:],'bx')
        # ax[1,0].set_ylabel('Edgewise Disp.')

        # i = 1
        # k_flap = bm.flap_disp[i,-1]/flap[i,-1]
        # k_edge = bm.lag_disp[i,-1]/edge[i,-1]
        # ax[0,1].plot(R, flap[i,:]*k_flap ,'k')
        # ax[0,1].plot(bm.r[i,:], bm.flap_disp[i,:],'bx')
        # ax[0,1].set_title('2nd Mode')
        # ax[1,1].plot(R, edge[i,:]*k_edge ,'k')
        # ax[1,1].plot(bm.r[i,:], bm.lag_disp[i,:],'bx')

        # i = 2
        # k_flap = bm.flap_disp[i,-1]/flap[i,-1]
        # k_edge = bm.lag_disp[i,-1]/edge[i,-1]
        # ax[0,2].plot(R, flap[i,:]*k_flap ,'k')
        # ax[0,2].plot(bm.r[i,:], bm.flap_disp[i,:],'bx')
        # ax[0,2].set_title('3rd Mode')
        # ax[1,2].plot(R, edge[i,:]*k_edge ,'k')
        # ax[1,2].plot(bm.r[i,:], bm.lag_disp[i,:],'bx')
        # fig.legend(loc='lower center', ncol=2)

        # i = 3
        # k_flap = bm.flap_disp[i,-1]/flap[i,-1]
        # k_edge = bm.lag_disp[i,-1]/edge[i,-1]
        # ax[0,3].plot(R, flap[i,:]*k_flap ,'k')
        # ax[0,3].plot(bm.r[i,:], bm.flap_disp[i,:],'bx')
        # ax[0,3].set_title('4th Mode')
        # ax[1,3].plot(R, edge[i,:]*k_edge ,'k')
        # ax[1,3].plot(bm.r[i,:], bm.lag_disp[i,:],'bx')
        # fig.legend(loc='lower center', ncol=2)
        # fig.text(0.5, 0.075, 'Blade Spanwise Position, $r/R$', ha='center')

        # (n,m)=np.shape(ax)
        # for i in range(n):
        #     for j in range(m):
        #         ax[i,j].tick_params(axis='both', which='major', labelsize=8)
        #         ax[i,j].grid(True, linestyle=':')

        # plt.show()



class RotorWithpBEAM(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
    
    def setup(self):
        NPTS = self.options['NPTS']

        # all inputs/outputs in airfoil coordinate system
        self.add_input('Px_defl', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil x-direction at max deflection condition')
        self.add_input('Py_defl', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil y-direction at max deflection condition')
        self.add_input('Pz_defl', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil z-direction at max deflection condition')

        self.add_input('Px_strain', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil x-direction at max strain condition')
        self.add_input('Py_strain', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil y-direction at max strain condition')
        self.add_input('Pz_strain', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil z-direction at max strain condition')

        self.add_input('Px_pc_defl', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil x-direction for deflection used in generated power curve')
        self.add_input('Py_pc_defl', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil y-direction for deflection used in generated power curve')
        self.add_input('Pz_pc_defl', val=np.zeros(NPTS), desc='distributed load (force per unit length) in airfoil z-direction for deflection used in generated power curve')

        self.add_input('xu_strain_spar', val=np.zeros(NPTS), desc='x-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_input('xl_strain_spar', val=np.zeros(NPTS), desc='x-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_input('yu_strain_spar', val=np.zeros(NPTS), desc='y-position of midpoint of spar cap on upper surface for strain calculation')
        self.add_input('yl_strain_spar', val=np.zeros(NPTS), desc='y-position of midpoint of spar cap on lower surface for strain calculation')
        self.add_input('xu_strain_te', val=np.zeros(NPTS), desc='x-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_input('xl_strain_te', val=np.zeros(NPTS), desc='x-position of midpoint of trailing-edge panel on lower surface for strain calculation')
        self.add_input('yu_strain_te', val=np.zeros(NPTS), desc='y-position of midpoint of trailing-edge panel on upper surface for strain calculation')
        self.add_input('yl_strain_te', val=np.zeros(NPTS), desc='y-position of midpoint of trailing-edge panel on lower surface for strain calculation')

        self.add_input('Mx_damage', val=np.zeros(NPTS), units='N*m', desc='damage equivalent moments about airfoil x-direction')
        self.add_input('My_damage', val=np.zeros(NPTS), units='N*m', desc='damage equivalent moments about airfoil y-direction')
        self.add_input('strain_ult_spar', val=0.0, desc='ultimate strain in spar cap')
        self.add_input('strain_ult_te', val=0.0, desc='uptimate strain in trailing-edge panels')
        self.add_input('gamma_fatigue', val=0.0, desc='safety factor for fatigue')
        self.add_input('m_damage', val=0.0, desc='slope of S-N curve for fatigue analysis')
        self.add_input('lifetime', val=0.0, units='year', desc='number of years used in fatigue analysis')

        self.add_input('z', val=np.zeros(NPTS), units='m', desc='locations of properties along beam')
        self.add_input('EA', val=np.zeros(NPTS), units='N', desc='axial stiffness')
        self.add_input('EIxx', val=np.zeros(NPTS), units='N*m**2', desc='edgewise stiffness (bending about :ref:`x-direction of airfoil aligned coordinate system <blade_airfoil_coord>`)')
        self.add_input('EIyy', val=np.zeros(NPTS), units='N*m**2', desc='flatwise stiffness (bending about y-direction of airfoil aligned coordinate system)')
        self.add_input('EIxy', val=np.zeros(NPTS), units='N*m**2', desc='coupled flap-edge stiffness')
        self.add_input('GJ', val=np.zeros(NPTS), units='N*m**2', desc='torsional stiffness (about axial z-direction of airfoil aligned coordinate system)')
        self.add_input('rhoA', val=np.zeros(NPTS), units='kg/m', desc='mass per unit length')
        self.add_input('rhoJ', val=np.zeros(NPTS), units='kg*m', desc='polar mass moment of inertia per unit length')
        self.add_input('x_ec', val=np.zeros(NPTS), units='m', desc='x-distance to elastic center from point about which above structural properties are computed (airfoil aligned coordinate system)')
        self.add_input('y_ec', val=np.zeros(NPTS), units='m', desc='y-distance to elastic center from point about which above structural properties are computed')

        # outputs
        self.add_output('blade_mass', val=0.0, units='kg', desc='mass of one blades')
        self.add_output('blade_moment_of_inertia', val=0.0, units='kg*m**2', desc='out of plane moment of inertia of a blade')
        self.add_output('freq_pbeam', val=np.zeros(NFREQ), units='Hz', desc='first nF natural frequencies of blade')
        self.add_output('freq_distance', val=0.0, desc='ration of 2nd and 1st natural frequencies, should be ratio of edgewise to flapwise')
        self.add_output('dx_defl', val=np.zeros(NPTS), desc='deflection of blade section in airfoil x-direction under max deflection loading')
        self.add_output('dy_defl', val=np.zeros(NPTS), desc='deflection of blade section in airfoil y-direction under max deflection loading')
        self.add_output('dz_defl', val=np.zeros(NPTS), desc='deflection of blade section in airfoil z-direction under max deflection loading')
        self.add_output('dx_pc_defl', val=np.zeros(NPTS), desc='deflection of blade section in airfoil x-direction under power curve loading')
        self.add_output('dy_pc_defl', val=np.zeros(NPTS), desc='deflection of blade section in airfoil y-direction under power curve loading')
        self.add_output('dz_pc_defl', val=np.zeros(NPTS), desc='deflection of blade section in airfoil z-direction under power curve loading')
        self.add_output('strainU_spar', val=np.zeros(NPTS), desc='strain in spar cap on upper surface at location xu,yu_strain with loads P_strain')
        self.add_output('strainL_spar', val=np.zeros(NPTS), desc='strain in spar cap on lower surface at location xl,yl_strain with loads P_strain')
        self.add_output('strainU_te', val=np.zeros(NPTS), desc='strain in trailing-edge panels on upper surface at location xu,yu_te with loads P_te')
        self.add_output('strainL_te', val=np.zeros(NPTS), desc='strain in trailing-edge panels on lower surface at location xl,yl_te with loads P_te')
        self.add_output('damageU_spar', val=np.zeros(NPTS), desc='fatigue damage on upper surface in spar cap')
        self.add_output('damageL_spar', val=np.zeros(NPTS), desc='fatigue damage on lower surface in spar cap')
        self.add_output('damageU_te', val=np.zeros(NPTS), desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_output('damageL_te', val=np.zeros(NPTS), desc='fatigue damage on lower surface in trailing-edge panels')

        # #self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

        self.EI11 = None
        self.EI22 = None
        self.EA   = None
        self.ca   = None
        self.sa   = None
        
    def principalCS(self, EIyy_in, EIxx_in, y_ec_in, x_ec_in, EA, EIxy):

        # rename (with swap of x, y for profile c.s.)
        EIxx , EIyy = EIyy_in , EIxx_in
        x_ec , y_ec = y_ec_in , x_ec_in
        self.EA     = EA
        EIxy        = EIxy

        # translate to elastic center
        EIxx -= y_ec**2*EA
        EIyy -= x_ec**2*EA
        EIxy -= x_ec*y_ec*EA

        # get rotation angle
        alpha = 0.5*np.arctan2(2*EIxy, EIyy-EIxx)

        self.EI11 = EIxx - EIxy*np.tan(alpha)
        self.EI22 = EIyy + EIxy*np.tan(alpha)

        # get moments and positions in principal axes
        self.ca = np.cos(alpha)
        self.sa = np.sin(alpha)


    def strain(self, blade, xu, yu, xl, yl):

        Vx, Vy, Fz, Mx, My, Tz = blade.shearAndBending()

        # use profile c.s. to use Hansen's notation
        Vx, Vy = Vy, Vx
        Mx, My = My, Mx
        xu, yu = yu, xu
        xl, yl = yl, xl

        # convert to principal xes
        M1 = Mx*self.ca + My*self.sa
        M2 = -Mx*self.sa + My*self.ca

        x = xu*self.ca + yu*self.sa
        y = -xu*self.sa + yu*self.ca

        # compute strain
        strainU = -(M1/self.EI11*y - M2/self.EI22*x + Fz/self.EA)  # negative sign because 3 is opposite of z

        x = xl*self.ca + yl*self.sa
        y = -xl*self.sa + yl*self.ca

        strainL = -(M1/self.EI11*y - M2/self.EI22*x + Fz/self.EA)

        return strainU, strainL

    def damage(self, Mx, My, xu, yu, xl, yl, emax=0.01, eta=1.755, m=10.0, N=365*24*3600*24):

        # use profil ec.s. to use Hansen's notation
        Mx, My = My, Mx
        Fz = 0.0
        xu, yu = yu, xu
        xl, yl = yl, xl

        # convert to principal xes
        M1 = Mx*self.ca + My*self.sa
        M2 = -Mx*self.sa + My*self.ca

        x = xu*self.ca + yu*self.sa
        y = -xu*self.sa + yu*self.ca

        # compute strain
        strainU = -(M1/self.EI11*y - M2/self.EI22*x + Fz/self.EA)  # negative sign because 3 is opposite of z

        x = xl*self.ca + yl*self.sa
        y = -xl*self.sa + yl*self.ca

        strainL = -(M1/self.EI11*y - M2/self.EI22*x + Fz/self.EA)

        # number of cycles to failure
        NfU = (emax/(eta*strainU))**m
        NfL = (emax/(eta*strainL))**m

        # damage- use log-based utilization version
        #damageU = N/NfU
        #damageL = N/NfL

        damageU = np.log(N) - m*(np.log(emax) - np.log(eta) - np.log(np.abs(strainU)))
        damageL = np.log(N) - m*(np.log(emax) - np.log(eta) - np.log(np.abs(strainL)))

        return damageU, damageL

    def compute(self, inputs, outputs):

        Px_defl = inputs['Px_defl']
        Py_defl = inputs['Py_defl']
        Pz_defl = inputs['Pz_defl']

        Px_defl = inputs['Px_defl']
        Py_defl = inputs['Py_defl']
        Pz_defl = inputs['Pz_defl']
        Px_strain = inputs['Px_strain']
        Py_strain = inputs['Py_strain']
        Pz_strain = inputs['Pz_strain']
        Px_pc_defl = inputs['Px_pc_defl']
        Py_pc_defl = inputs['Py_pc_defl']
        Pz_pc_defl = inputs['Pz_pc_defl']

        xu_strain_spar = inputs['xu_strain_spar']
        xl_strain_spar = inputs['xl_strain_spar']
        yu_strain_spar = inputs['yu_strain_spar']
        yl_strain_spar = inputs['yl_strain_spar']
        xu_strain_te = inputs['xu_strain_te']
        xu_strain_te = inputs['xu_strain_te']
        xl_strain_te = inputs['xl_strain_te']
        yu_strain_te = inputs['yu_strain_te']
        yl_strain_te = inputs['yl_strain_te']

        Mx_damage = inputs['Mx_damage']
        My_damage = inputs['My_damage']
        strain_ult_spar = inputs['strain_ult_spar']
        strain_ult_te = inputs['strain_ult_te']
        gamma_fatigue = inputs['gamma_fatigue']
        m_damage = inputs['m_damage']
        N_damage = 365*24*3600*inputs['lifetime']

        # outputs
        nsec = len(inputs['z'])
        
        # create finite element objects
        p_section = _pBEAM.SectionData(nsec, inputs['z'], inputs['EA'], inputs['EIxx'],
            inputs['EIyy'], inputs['GJ'], inputs['rhoA'], inputs['rhoJ'])
        p_tip = _pBEAM.TipData()  # no tip mass
        p_base = _pBEAM.BaseData(np.ones(6), 1.0)  # rigid base


        # ----- tip deflection -----

        # evaluate displacements
        p_loads = _pBEAM.Loads(nsec, Px_defl, Py_defl, Pz_defl)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        dx_defl, dy_defl, dz_defl, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()

        p_loads = _pBEAM.Loads(nsec, Px_pc_defl, Py_pc_defl, Pz_pc_defl)
        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)
        dx_pc_defl, dy_pc_defl, dz_pc_defl, dtheta_r1, dtheta_r2, dtheta_z = blade.displacement()


        # --- mass ---
        blade_mass = blade.mass()

        # --- moments of inertia
        blade_moment_of_inertia = blade.outOfPlaneMomentOfInertia()

        # ----- natural frequencies ----
        freq = blade.naturalFrequencies(NFREQ)

        # ----- strain -----
        self.principalCS(inputs['EIyy'], inputs['EIxx'], inputs['y_ec'], inputs['x_ec'], inputs['EA'], inputs['EIxy'])

        p_loads = _pBEAM.Loads(nsec, Px_strain, Py_strain, Pz_strain)

        blade = _pBEAM.Beam(p_section, p_loads, p_tip, p_base)

        strainU_spar, strainL_spar = self.strain(blade, xu_strain_spar, yu_strain_spar, xl_strain_spar, yl_strain_spar)

        strainU_te, strainL_te = self.strain(blade, xu_strain_te, yu_strain_te, xl_strain_te, yl_strain_te)

        damageU_spar, damageL_spar = self.damage(Mx_damage, My_damage, xu_strain_spar, yu_strain_spar, xl_strain_spar, yl_strain_spar,
                                                 emax=strain_ult_spar, eta=gamma_fatigue, m=m_damage, N=N_damage)

        damageU_te, damageL_te = self.damage(Mx_damage, My_damage, xu_strain_te, yu_strain_te, xl_strain_te, yl_strain_te,
                                             emax=strain_ult_te, eta=gamma_fatigue, m=m_damage, N=N_damage)

        outputs['blade_mass'] = blade_mass
        outputs['blade_moment_of_inertia'] = blade_moment_of_inertia
        outputs['freq_pbeam'] = freq
        outputs['freq_distance'] = np.float(freq[1]/freq[0])
        outputs['dx_defl'] = dx_defl
        outputs['dy_defl'] = dy_defl
        outputs['dz_defl'] = dz_defl
        outputs['dx_pc_defl'] = dx_pc_defl
        outputs['dy_pc_defl'] = dy_pc_defl
        outputs['dz_pc_defl'] = dz_pc_defl
        outputs['strainU_spar'] = strainU_spar
        outputs['strainL_spar'] = strainL_spar
        outputs['strainU_te'] = strainU_te
        outputs['strainL_te'] = strainL_te
        outputs['damageU_spar'] = damageU_spar
        outputs['damageL_spar'] = damageL_spar
        outputs['damageU_te'] = damageU_te
        outputs['damageL_te'] = damageL_te
        

class DamageLoads(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
    
    def setup(self):
        NPTS = self.options['NPTS']

        self.add_input('rstar', np.zeros(NPTS+1), desc='nondimensional radial locations of damage equivalent moments')
        self.add_input('Mxb', np.zeros(NPTS+1), units='N*m', desc='damage equivalent moments about blade c.s. x-direction')
        self.add_input('Myb', np.zeros(NPTS+1), units='N*m', desc='damage equivalent moments about blade c.s. y-direction')
        self.add_input('theta', val=np.zeros(NPTS), units='deg', desc='structural twist')
        self.add_input('z', val=np.zeros(NPTS), units='m', desc='structural radial locations')

        self.add_output('Mxa', val=np.zeros(NPTS), units='N*m', desc='damage equivalent moments about airfoil c.s. x-direction')
        self.add_output('Mya', val=np.zeros(NPTS), units='N*m', desc='damage equivalent moments about airfoil c.s. y-direction')

        #self.declare_partials(['Mxa', 'Mya'], ['rstar', 'Mxb', 'Myb', 'theta', 'z'])


    def compute(self, inputs, outputs):
        Mxb = inputs['Mxb']
        Myb = inputs['Myb']
        theta = inputs['theta']
        r = inputs['z']

        rstar = (r-r[0])/(r[-1]-r[0])

        myakima = Akima(inputs['rstar'], Mxb)
        Mxb, dMxbstr_drstarstr, dMxbstr_drstar, dMxbstr_dMxb = myakima(rstar)

        myakima = Akima(inputs['rstar'], Myb)
        Myb, dMybstr_drstarstr, dMybstr_drstar, dMybstr_dMyb = myakima(rstar)

        Ma = DirectionVector(Mxb, Myb, 0.0).bladeToAirfoil(theta)
        Mxa = Ma.x
        Mya = Ma.y

        outputs['Mxa'] = Mxa
        outputs['Mya'] = Mya
        '''
        n = len(r)
        drstarstr_dr = np.zeros((n, n))
        for i in range(1, n-1):
            drstarstr_dr[i, i] = 1.0/(r[-1] - r[0])
        drstarstr_dr[1:, 0] = (r[1:] - r[-1])/(r[-1] - r[0])**2
        drstarstr_dr[:-1, -1] = -(r[:-1] - r[0])/(r[-1] - r[0])**2

        dMxbstr_drstarstr = np.diag(dMxbstr_drstarstr)
        dMybstr_drstarstr = np.diag(dMybstr_drstarstr)

        dMxbstr_dr = np.dot(dMxbstr_drstarstr, drstarstr_dr)
        dMybstr_dr = np.dot(dMybstr_drstarstr, drstarstr_dr)

        dMxa_dr = np.dot(np.diag(Ma.dx['dx']), dMxbstr_dr)\
            + np.dot(np.diag(Ma.dx['dy']), dMybstr_dr)
        dMxa_drstar = np.dot(np.diag(Ma.dx['dx']), dMxbstr_drstar)\
            + np.dot(np.diag(Ma.dx['dy']), dMybstr_drstar)
        dMxa_dMxb = np.dot(np.diag(Ma.dx['dx']), dMxbstr_dMxb)
        dMxa_dMyb = np.dot(np.diag(Ma.dx['dy']), dMybstr_dMyb)
        dMxa_dtheta = np.diag(Ma.dx['dtheta'])

        dMya_dr = np.dot(np.diag(Ma.dy['dx']), dMxbstr_dr)\
            + np.dot(np.diag(Ma.dy['dy']), dMybstr_dr)
        dMya_drstar = np.dot(np.diag(Ma.dy['dx']), dMxbstr_drstar)\
            + np.dot(np.diag(Ma.dy['dy']), dMybstr_drstar)
        dMya_dMxb = np.dot(np.diag(Ma.dy['dx']), dMxbstr_dMxb)
        dMya_dMyb = np.dot(np.diag(Ma.dy['dy']), dMybstr_dMyb)
        dMya_dtheta = np.diag(Ma.dy['dtheta'])

        J={}
        J['Mxa', 'rstar'] = dMxa_drstar
        J['Mxa', 'Mxb'] = dMxa_dMxb
        J['Mxa', 'Myb'] = dMxa_dMyb
        J['Mxa', 'theta'] = dMxa_dtheta
        J['Mxa', 'z'] = dMxa_dr

        J['Mya', 'rstar'] = dMya_drstar
        J['Mya', 'Mxb'] = dMya_dMxb
        J['Mya', 'Myb'] = dMya_dMyb
        J['Mya', 'theta'] = dMya_dtheta
        J['Mya', 'z'] = dMya_dr
        self.J = J

    def compute_partials(self, inputs, J):
        J = self.J
        '''
        



class TotalLoads(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
    
    def setup(self):
        NPTS = self.options['NPTS']

        # variables
        self.add_input('aeroloads_r', val=np.zeros(NPTS), units='m', desc='radial positions along blade going toward tip')
        self.add_input('aeroloads_Px', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_input('aeroloads_Py', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_input('aeroloads_Pz', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_input('aeroloads_Omega', val=0.0, units='rpm', desc='rotor rotation speed')
        self.add_input('aeroloads_pitch', val=0.0, units='deg', desc='pitch angle')
        self.add_input('aeroloads_azimuth', val=0.0, units='deg', desc='azimuthal angle')

        self.add_input('z', val=np.zeros(NPTS), units='m', desc='structural radial locations')
        self.add_input('theta', val=np.zeros(NPTS), units='deg', desc='structural twist')
        self.add_input('tilt', val=0.0, units='deg', desc='tilt angle')
        self.add_input('totalCone', val=np.zeros(NPTS), units='deg', desc='total cone angle from precone and curvature')
        self.add_input('z_az', val=np.zeros(NPTS), units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_input('rhoA', val=np.zeros(NPTS), units='kg/m', desc='mass per unit length')

        self.add_input('dynamicFactor', val=1.0, desc='a dynamic amplification factor to adjust the static deflection calculation') #)

        # outputs
        self.add_output('Px_af', val=np.zeros(NPTS), desc='total distributed loads in airfoil x-direction')
        self.add_output('Py_af', val=np.zeros(NPTS), desc='total distributed loads in airfoil y-direction')
        self.add_output('Pz_af', val=np.zeros(NPTS), desc='total distributed loads in airfoil z-direction')

        #self.declare_partials(['Px_af', 'Py_af', 'Pz_af'],
        #                      ['aeroloads_r', 'aeroloads_Px', 'aeroloads_Py', 'aeroloads_Pz', 'aeroloads_Omega',
        #                       'aeroloads_pitch', 'aeroloads_azimuth', 'z', 'theta', 'tilt', 'totalCone', 'rhoA', 'z_az'])


    def compute(self, inputs, outputs):

        dynamicFactor = inputs['dynamicFactor']
        r = inputs['z']
        theta = inputs['theta']
        tilt = inputs['tilt']
        totalCone = inputs['totalCone']
        z_az = inputs['z_az']
        rhoA = inputs['rhoA']


        # totalCone = precone
        # z_az = r*cosd(precone)
        totalCone = totalCone
        z_az = z_az

        # keep all in blade c.s. then rotate all at end

        # rename
        # aero = aeroloads

        # --- aero loads ---

        # interpolate aerodynamic loads onto structural grid
        P_a = DirectionVector(0, 0, 0)
        myakima = Akima(inputs['aeroloads_r'], inputs['aeroloads_Px'])
        P_a.x, dPax_dr, dPax_daeror, dPax_daeroPx = myakima(r)

        myakima = Akima(inputs['aeroloads_r'], inputs['aeroloads_Py'])
        P_a.y, dPay_dr, dPay_daeror, dPay_daeroPy = myakima(r)

        myakima = Akima(inputs['aeroloads_r'], inputs['aeroloads_Pz'])
        P_a.z, dPaz_dr, dPaz_daeror, dPaz_daeroPz = myakima(r)


        # --- weight loads ---

        # yaw c.s.
        weight = DirectionVector(0.0, 0.0, -rhoA*gravity)

        P_w = weight.yawToHub(tilt).hubToAzimuth(inputs['aeroloads_azimuth'])\
            .azimuthToBlade(totalCone)


        # --- centrifugal loads ---

        # azimuthal c.s.
        Omega = inputs['aeroloads_Omega']*RPM2RS
        load = DirectionVector(0.0, 0.0, rhoA*Omega**2*z_az)

        P_c = load.azimuthToBlade(totalCone)


        # --- total loads ---
        P = P_a + P_w + P_c

        # rotate to airfoil c.s.
        theta = np.array(theta) + inputs['aeroloads_pitch']
        P = P.bladeToAirfoil(theta)

        Px_af = dynamicFactor * P.x
        Py_af = dynamicFactor * P.y
        Pz_af = dynamicFactor * P.z

        outputs['Px_af'] = Px_af
        outputs['Py_af'] = Py_af
        outputs['Pz_af'] = Pz_af

        '''
        dPwx, dPwy, dPwz = P_w.dx, P_w.dy, P_w.dz
        dPcx, dPcy, dPcz = P_c.dx, P_c.dy, P_c.dz
        dPx, dPy, dPz = P.dx, P.dy, P.dz
        Omega = inputs['aeroloads_Omega']*RPM2RS
        z_az = z_az


        dPx_dOmega = dPcx['dz']*rhoA*z_az*2*Omega*RPM2RS
        dPy_dOmega = dPcy['dz']*rhoA*z_az*2*Omega*RPM2RS
        dPz_dOmega = dPcz['dz']*rhoA*z_az*2*Omega*RPM2RS

        dPx_dr = np.diag(dPax_dr)
        dPy_dr = np.diag(dPay_dr)
        dPz_dr = np.diag(dPaz_dr)

        dPx_dprecone = np.diag(dPwx['dprecone'] + dPcx['dprecone'])
        dPy_dprecone = np.diag(dPwy['dprecone'] + dPcy['dprecone'])
        dPz_dprecone = np.diag(dPwz['dprecone'] + dPcz['dprecone'])

        dPx_dzaz = np.diag(dPcx['dz']*rhoA*Omega**2)
        dPy_dzaz = np.diag(dPcy['dz']*rhoA*Omega**2)
        dPz_dzaz = np.diag(dPcz['dz']*rhoA*Omega**2)

        dPx_drhoA = np.diag(-dPwx['dz']*gravity + dPcx['dz']*Omega**2*z_az)
        dPy_drhoA = np.diag(-dPwy['dz']*gravity + dPcy['dz']*Omega**2*z_az)
        dPz_drhoA = np.diag(-dPwz['dz']*gravity + dPcz['dz']*Omega**2*z_az)

        dPxaf_daeror = (dPx['dx']*dPax_daeror.T + dPx['dy']*dPay_daeror.T + dPx['dz']*dPaz_daeror.T).T
        dPyaf_daeror = (dPy['dx']*dPax_daeror.T + dPy['dy']*dPay_daeror.T + dPy['dz']*dPaz_daeror.T).T
        dPzaf_daeror = (dPz['dx']*dPax_daeror.T + dPz['dy']*dPay_daeror.T + dPz['dz']*dPaz_daeror.T).T

        dPxaf_dPxaero = (dPx['dx']*dPax_daeroPx.T).T
        dPxaf_dPyaero = (dPx['dy']*dPay_daeroPy.T).T
        dPxaf_dPzaero = (dPx['dz']*dPaz_daeroPz.T).T

        dPyaf_dPxaero = (dPy['dx']*dPax_daeroPx.T).T
        dPyaf_dPyaero = (dPy['dy']*dPay_daeroPy.T).T
        dPyaf_dPzaero = (dPy['dz']*dPaz_daeroPz.T).T

        dPzaf_dPxaero = (dPz['dx']*dPax_daeroPx.T).T
        dPzaf_dPyaero = (dPz['dy']*dPay_daeroPy.T).T
        dPzaf_dPzaero = (dPz['dz']*dPaz_daeroPz.T).T

        dPxaf_dOmega = dPx['dx']*dPx_dOmega + dPx['dy']*dPy_dOmega + dPx['dz']*dPz_dOmega
        dPyaf_dOmega = dPy['dx']*dPx_dOmega + dPy['dy']*dPy_dOmega + dPy['dz']*dPz_dOmega
        dPzaf_dOmega = dPz['dx']*dPx_dOmega + dPz['dy']*dPy_dOmega + dPz['dz']*dPz_dOmega

        dPxaf_dpitch = dPx['dtheta']
        dPyaf_dpitch = dPy['dtheta']
        dPzaf_dpitch = dPz['dtheta']

        dPxaf_dazimuth = dPx['dx']*dPwx['dazimuth'] + dPx['dy']*dPwy['dazimuth'] + dPx['dz']*dPwz['dazimuth']
        dPyaf_dazimuth = dPy['dx']*dPwx['dazimuth'] + dPy['dy']*dPwy['dazimuth'] + dPy['dz']*dPwz['dazimuth']
        dPzaf_dazimuth = dPz['dx']*dPwx['dazimuth'] + dPz['dy']*dPwy['dazimuth'] + dPz['dz']*dPwz['dazimuth']

        dPxaf_dr = dPx['dx']*dPx_dr + dPx['dy']*dPy_dr + dPx['dz']*dPz_dr
        dPyaf_dr = dPy['dx']*dPx_dr + dPy['dy']*dPy_dr + dPy['dz']*dPz_dr
        dPzaf_dr = dPz['dx']*dPx_dr + dPz['dy']*dPy_dr + dPz['dz']*dPz_dr

        dPxaf_dtheta = np.diag(dPx['dtheta'])
        dPyaf_dtheta = np.diag(dPy['dtheta'])
        dPzaf_dtheta = np.diag(dPz['dtheta'])

        dPxaf_dtilt = dPx['dx']*dPwx['dtilt'] + dPx['dy']*dPwy['dtilt'] + dPx['dz']*dPwz['dtilt']
        dPyaf_dtilt = dPy['dx']*dPwx['dtilt'] + dPy['dy']*dPwy['dtilt'] + dPy['dz']*dPwz['dtilt']
        dPzaf_dtilt = dPz['dx']*dPwx['dtilt'] + dPz['dy']*dPwy['dtilt'] + dPz['dz']*dPwz['dtilt']

        dPxaf_dprecone = dPx['dx']*dPx_dprecone + dPx['dy']*dPy_dprecone + dPx['dz']*dPz_dprecone
        dPyaf_dprecone = dPy['dx']*dPx_dprecone + dPy['dy']*dPy_dprecone + dPy['dz']*dPz_dprecone
        dPzaf_dprecone = dPz['dx']*dPx_dprecone + dPz['dy']*dPy_dprecone + dPz['dz']*dPz_dprecone

        dPxaf_drhoA = dPx['dx']*dPx_drhoA + dPx['dy']*dPy_drhoA + dPx['dz']*dPz_drhoA
        dPyaf_drhoA = dPy['dx']*dPx_drhoA + dPy['dy']*dPy_drhoA + dPy['dz']*dPz_drhoA
        dPzaf_drhoA = dPz['dx']*dPx_drhoA + dPz['dy']*dPy_drhoA + dPz['dz']*dPz_drhoA

        dPxaf_dzaz = dPx['dx']*dPx_dzaz + dPx['dy']*dPy_dzaz + dPx['dz']*dPz_dzaz
        dPyaf_dzaz = dPy['dx']*dPx_dzaz + dPy['dy']*dPy_dzaz + dPy['dz']*dPz_dzaz
        dPzaf_dzaz = dPz['dx']*dPx_dzaz + dPz['dy']*dPy_dzaz + dPz['dz']*dPz_dzaz

        J = {}
        J['Px_af', 'aeroloads_r'] = dynamicFactor * dPxaf_daeror
        J['Px_af', 'aeroloads_Px'] = dynamicFactor * dPxaf_dPxaero
        J['Px_af', 'aeroloads_Py'] = dynamicFactor * dPxaf_dPyaero
        J['Px_af', 'aeroloads_Pz'] = dynamicFactor * dPxaf_dPzaero
        J['Px_af', 'aeroloads_Omega'] = dynamicFactor * dPxaf_dOmega
        J['Px_af', 'aeroloads_pitch'] = dynamicFactor * dPxaf_dpitch
        J['Px_af', 'aeroloads_azimuth'] = dynamicFactor * dPxaf_dazimuth
        J['Px_af', 'z'] = dynamicFactor * dPxaf_dr
        J['Px_af', 'theta'] = dynamicFactor * dPxaf_dtheta
        J['Px_af', 'tilt'] = dynamicFactor * dPxaf_dtilt
        J['Px_af', 'totalCone'] = dynamicFactor * dPxaf_dprecone
        J['Px_af', 'rhoA'] = dynamicFactor * dPxaf_drhoA
        J['Px_af', 'z_az'] = dynamicFactor * dPxaf_dzaz

        J['Py_af', 'aeroloads_r'] = dynamicFactor * dPyaf_daeror
        J['Py_af', 'aeroloads_Px'] = dynamicFactor * dPyaf_dPxaero
        J['Py_af', 'aeroloads_Py'] = dynamicFactor * dPyaf_dPyaero
        J['Py_af', 'aeroloads_Pz'] = dynamicFactor * dPyaf_dPzaero
        J['Py_af', 'aeroloads_Omega'] = dynamicFactor * dPyaf_dOmega
        J['Py_af', 'aeroloads_pitch'] = dynamicFactor * dPyaf_dpitch
        J['Py_af', 'aeroloads_azimuth'] = dynamicFactor * dPyaf_dazimuth
        J['Py_af', 'z'] = dynamicFactor * dPyaf_dr
        J['Py_af', 'theta'] = dynamicFactor * dPyaf_dtheta
        J['Py_af', 'tilt'] = dynamicFactor * dPyaf_dtilt
        J['Py_af', 'totalCone'] = dynamicFactor * dPyaf_dprecone
        J['Py_af', 'rhoA'] = dynamicFactor * dPyaf_drhoA
        J['Py_af', 'z_az'] = dynamicFactor * dPyaf_dzaz

        J['Pz_af', 'aeroloads_r'] = dynamicFactor * dPzaf_daeror
        J['Pz_af', 'aeroloads_Px'] = dynamicFactor * dPzaf_dPxaero
        J['Pz_af', 'aeroloads_Py'] = dynamicFactor * dPzaf_dPyaero
        J['Pz_af', 'aeroloads_Pz'] = dynamicFactor * dPzaf_dPzaero
        J['Pz_af', 'aeroloads_Omega'] = dynamicFactor * dPzaf_dOmega
        J['Pz_af', 'aeroloads_pitch'] = dynamicFactor * dPzaf_dpitch
        J['Pz_af', 'aeroloads_azimuth'] = dynamicFactor * dPzaf_dazimuth
        J['Pz_af', 'z'] = dynamicFactor * dPzaf_dr
        J['Pz_af', 'theta'] = dynamicFactor * dPzaf_dtheta
        J['Pz_af', 'tilt'] = dynamicFactor * dPzaf_dtilt
        J['Pz_af', 'totalCone'] = dynamicFactor * dPzaf_dprecone
        J['Pz_af', 'rhoA'] = dynamicFactor * dPzaf_drhoA
        J['Pz_af', 'z_az'] = dynamicFactor * dPzaf_dzaz
        self.J = J


    def compute_partials(self, inputs, J):
        J = self.J
        '''        

class TipDeflection(ExplicitComponent):
    def setup(self):
        # variables
        self.add_input('dx', val=0.0, desc='deflection at tip in airfoil x-direction')
        self.add_input('dy', val=0.0, desc='deflection at tip in airfoil y-direction')
        self.add_input('dz', val=0.0, desc='deflection at tip in airfoil z-direction')
        self.add_input('theta', val=0.0, units='deg', desc='twist at tip section')
        self.add_input('pitch', val=0.0, units='deg', desc='blade pitch angle')
        self.add_input('azimuth', val=0.0, units='deg', desc='azimuth angle')
        self.add_input('tilt', val=0.0, units='deg', desc='tilt angle')
        self.add_input('totalConeTip', val=0.0, units='deg', desc='total coning angle including precone and curvature')

        self.add_input('hub_height', val=0.0, units='m', desc='Tower top hub height')
        self.add_discrete_input('downwind', val=False)
        self.add_input('Rtip', val=0.0, units='m', desc='tip location in z_b')
        self.add_input('precurveTip', val=0.0, units='m', desc='tip location in x_b')
        self.add_input('presweepTip', val=0.0, units='m', desc='tip location in y_b')
        self.add_input('precone', val=0.0, units='deg', desc='precone angle')
        self.add_input('gamma_m', 0.0, desc='safety factor on materials')

        # parameters
        self.add_input('dynamicFactor', val=1.0, desc='a dynamic amplification factor to adjust the static deflection calculation') #)

        # outputs
        self.add_output('tip_deflection', val=0.0, units='m', desc='deflection at tip in yaw x-direction')
        self.add_output('tip_position', val=np.zeros(3), units='m', desc='Position coordinates of deflected tip in yaw c.s.')
        self.add_output('ground_clearance', val=0.0, units='m', desc='distance between blade tip and ground')

        #self.declare_partials(['tip_deflection'],
        #                      ['dx', 'dy', 'dz', 'theta', 'pitch', 'azimuth', 'tilt',
        #                       'totalConeTip','precurveTip','presweepTip','Rtip'])
        #self.declare_partials('tip_position', '*', method='fd', form='central', step=1e-6)
        #self.declare_partials('ground_clearance', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        dx            = inputs['dx']
        dy            = inputs['dy']
        dz            = inputs['dz']
        theta         = inputs['theta']
        pitch         = inputs['pitch']
        azimuth       = 180.0 #inputs['azimuth']
        precone       = inputs['precone']
        tilt          = inputs['tilt']
        totalConeTip  = inputs['totalConeTip']
        dynamicFactor = inputs['dynamicFactor']
        precurveTip   = inputs['precurveTip']
        presweepTip   = inputs['presweepTip']
        rtip          = inputs['Rtip']
        upwind        = not discrete_inputs['downwind']

        theta = theta + pitch

        dr = DirectionVector(dx, dy, dz)
        delta = dr.airfoilToBlade(theta).bladeToAzimuth(totalConeTip).azimuthToHub(azimuth).hubToYaw(tilt)

        tip_deflection = dynamicFactor * delta.x

        outputs['tip_deflection'] = tip_deflection

        # coordinates of blade tip in yaw c.s.
        # TODO: Combine intelligently with other Direction Vector
        dR = DirectionVector(precurveTip, presweepTip, rtip)
        blade_yaw = dR.bladeToAzimuth(totalConeTip).azimuthToHub(azimuth).hubToYaw(tilt)
        

        # find corresponding radius of tower
        coeff = 1.0 if upwind else -1.0
        z_pos = inputs['hub_height'] + blade_yaw.z
        x_pos = coeff*blade_yaw.x + inputs['gamma_m'] * tip_deflection
        outputs['tip_position'] = np.array([x_pos, 0.0, z_pos])
        outputs['ground_clearance'] = z_pos
        '''
        dx = dynamicFactor * delta.dx['dx']
        dy = dynamicFactor * delta.dx['dy']
        dz = dynamicFactor * delta.dx['dz']
        dtheta = dpitch = dynamicFactor * delta.dx['dtheta']
        dazimuth = dynamicFactor * delta.dx['dazimuth']
        dtilt = dynamicFactor * delta.dx['dtilt']
        dtotalConeTip = dynamicFactor * delta.dx['dprecone']
        J = {}
        J['tip_deflection', 'dx'] = dx
        J['tip_deflection', 'dy'] = dy
        J['tip_deflection', 'dz'] = dz
        J['tip_deflection', 'theta'] = dtheta
        J['tip_deflection', 'pitch'] = dpitch
        J['tip_deflection', 'azimuth'] = dazimuth
        J['tip_deflection', 'tilt'] = dtilt
        J['tip_deflection', 'totalConeTip'] = dtotalConeTip
        J['tip_deflection', 'dynamicFactor'] = delta.x.tolist()
        self.J = J

    def compute_partials(self, inputs, J, discrete_inputs):
        J = self.J
        '''

        

class BladeDeflection(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
        self.options.declare('NINPUT')
    
    def setup(self):
        NPTS = self.options['NPTS']
        NINPUT = self.options['NINPUT']

        self.add_input('dx', val=np.zeros(NPTS), desc='deflections in airfoil x-direction')
        self.add_input('dy', val=np.zeros(NPTS), desc='deflections in airfoil y-direction')
        self.add_input('dz', val=np.zeros(NPTS), desc='deflections in airfoil z-direction')
        self.add_input('pitch', val=0.0, units='deg', desc='blade pitch angle')
        self.add_input('theta', val=np.zeros(NPTS), units='deg', desc='structural twist')

        self.add_input('r_in', val=np.zeros(NINPUT), units='m', desc='Spline control points for inputs')
        self.add_input('Rhub', val=0.0, units='m', desc='hub radius')
        self.add_input('r', val=np.zeros(NPTS), units='m', desc='undeflected radial locations')
        self.add_input('precurve', val=np.zeros(NPTS), units='m', desc='undeflected precurve locations')
        self.add_input('bladeLength', val=0.0, units='m', desc='original blade length (only an actual length if no curvature)')

        self.add_output('delta_bladeLength', val=0.0, units='m', desc='adjustment to blade length to account for curvature from loading')
        self.add_output('delta_precurve_sub', val=np.zeros(NINPUT), units='m', desc='adjustment to precurve to account for curvature from loading')


        #self.declare_partials(['delta_bladeLength', 'delta_precurve_sub'],
        #                      ['dx', 'dy', 'dz', 'pitch', 'theta', 'r_in', 'Rhub',
        #                       'r', 'precurve', 'bladeLength'])


    def compute(self, inputs, outputs):

        dx = inputs['dx']
        dy = inputs['dy']
        dz = inputs['dz']
        pitch = inputs['pitch']
        theta = inputs['theta']
        r_in0 = inputs['r_in']
        Rhub0 = inputs['Rhub']
        r_pts0 = inputs['r']
        precurve0 = inputs['precurve']
        bladeLength0 = inputs['bladeLength']


        theta = theta + pitch

        dr = DirectionVector(dx, dy, dz)
        delta = dr.airfoilToBlade(theta)

        precurve_out = precurve0 + delta.x

        length0 = Rhub0 + np.sum(np.sqrt((precurve0[1:] - precurve0[:-1])**2 +
                                            (r_pts0[1:] - r_pts0[:-1])**2))
        length = Rhub0 + np.sum(np.sqrt((precurve_out[1:] - precurve_out[:-1])**2 +
                                           (r_pts0[1:] - r_pts0[:-1])**2))

        shortening = length0/length

        delta_bladeLength = bladeLength0 * (shortening - 1)
        # TODO: linearly interpolation is not C1 continuous.  it should work OK for now, but is not ideal
        delta_precurve_sub, dpcs_drsubpc0, dpcs_drstr0, dpcs_ddeltax = \
            interp_with_deriv(r_in0, r_pts0, delta.x)

        outputs['delta_bladeLength'] = delta_bladeLength
        outputs['delta_precurve_sub'] = delta_precurve_sub
        '''
        n = len(theta)

        ddeltax_ddx = delta.dx['dx']
        ddeltax_ddy = delta.dx['dy']
        ddeltax_ddz = delta.dx['dz']
        ddeltax_dtheta = delta.dx['dtheta']
        ddeltax_dthetastr = ddeltax_dtheta
        ddeltax_dpitch = ddeltax_dtheta

        dl0_drhub0 = 1.0
        dl_drhub0 = 1.0
        dl0_dprecurvestr0 = np.zeros(n)
        dl_dprecurvestr0 = np.zeros(n)
        dl0_drstr0 = np.zeros(n)
        dl_drstr0 = np.zeros(n)

        precurve_out = precurve0 + delta.x

        for i in range(1, n-1):
            sm0 = np.sqrt((precurve0[i] - precurve0[i-1])**2 + (r_pts0[i] - r_pts0[i-1])**2)
            sm = np.sqrt((precurve_out[i] - precurve_out[i-1])**2 + (r_pts0[i] - r_pts0[i-1])**2)
            sp0 = np.sqrt((precurve0[i+1] - precurve0[i])**2 + (r_pts0[i+1] - r_pts0[i])**2)
            sp = np.sqrt((precurve_out[i+1] - precurve_out[i])**2 + (r_pts0[i+1] - r_pts0[i])**2)
            dl0_dprecurvestr0[i] = (precurve0[i] - precurve0[i-1]) / sm0 \
                - (precurve0[i+1] - precurve0[i]) / sp0
            dl_dprecurvestr0[i] = (precurve_out[i] - precurve_out[i-1]) / sm \
                - (precurve_out[i+1] - precurve_out[i]) / sp
            dl0_drstr0[i] = (r_pts0[i] - r_pts0[i-1]) / sm0 \
                - (r_pts0[i+1] - r_pts0[i]) / sp0
            dl_drstr0[i] = (r_pts0[i] - r_pts0[i-1]) / sm \
                - (r_pts0[i+1] - r_pts0[i]) / sp

        sfirst0 = np.sqrt((precurve0[1] - precurve0[0])**2 + (r_pts0[1] - r_pts0[0])**2)
        sfirst = np.sqrt((precurve_out[1] - precurve_out[0])**2 + (r_pts0[1] - r_pts0[0])**2)
        slast0 = np.sqrt((precurve0[n-1] - precurve0[n-2])**2 + (r_pts0[n-1] - r_pts0[n-2])**2)
        slast = np.sqrt((precurve_out[n-1] - precurve_out[n-2])**2 + (r_pts0[n-1] - r_pts0[n-2])**2)
        dl0_dprecurvestr0[0] = -(precurve0[1] - precurve0[0]) / sfirst0
        dl0_dprecurvestr0[n-1] = (precurve0[n-1] - precurve0[n-2]) / slast0
        dl_dprecurvestr0[0] = -(precurve_out[1] - precurve_out[0]) / sfirst
        dl_dprecurvestr0[n-1] = (precurve_out[n-1] - precurve_out[n-2]) / slast
        dl0_drstr0[0] = -(r_pts0[1] - r_pts0[0]) / sfirst0
        dl0_drstr0[n-1] = (r_pts0[n-1] - r_pts0[n-2]) / slast0
        dl_drstr0[0] = -(r_pts0[1] - r_pts0[0]) / sfirst
        dl_drstr0[n-1] = (r_pts0[n-1] - r_pts0[n-2]) / slast

        dl_ddeltax = dl_dprecurvestr0
        dl_ddx = dl_ddeltax * ddeltax_ddx
        dl_ddy = dl_ddeltax * ddeltax_ddy
        dl_ddz = dl_ddeltax * ddeltax_ddz
        dl_dthetastr = dl_ddeltax * ddeltax_dthetastr
        dl_dpitch = np.dot(dl_ddeltax, ddeltax_dpitch)

        dshort_dl = -length0/length**2
        dshort_dl0 = 1.0/length
        dshort_drhub0 = dshort_dl0*dl0_drhub0 + dshort_dl*dl_drhub0
        dshort_dprecurvestr0 = dshort_dl0*dl0_dprecurvestr0 + dshort_dl*dl_dprecurvestr0
        dshort_drstr0 = dshort_dl0*dl0_drstr0 + dshort_dl*dl_drstr0
        dshort_ddx = dshort_dl*dl_ddx
        dshort_ddy = dshort_dl*dl_ddy
        dshort_ddz = dshort_dl*dl_ddz
        dshort_dthetastr = dshort_dl*dl_dthetastr
        dshort_dpitch = dshort_dl*dl_dpitch

        dbl_dbl0 = (shortening - 1)
        dbl_drhub0 = bladeLength0 * dshort_drhub0
        dbl_dprecurvestr0 = bladeLength0 * dshort_dprecurvestr0
        dbl_drstr0 = bladeLength0 * dshort_drstr0
        dbl_ddx = bladeLength0 * dshort_ddx
        dbl_ddy = bladeLength0 * dshort_ddy
        dbl_ddz = bladeLength0 * dshort_ddz
        dbl_dthetastr = bladeLength0 * dshort_dthetastr
        dbl_dpitch = bladeLength0 * dshort_dpitch

        m = len(r_in0)
        dpcs_ddx = dpcs_ddeltax*ddeltax_ddx
        dpcs_ddy = dpcs_ddeltax*ddeltax_ddy
        dpcs_ddz = dpcs_ddeltax*ddeltax_ddz
        dpcs_dpitch = np.dot(dpcs_ddeltax, ddeltax_dpitch)
        dpcs_dthetastr = dpcs_ddeltax*ddeltax_dthetastr

        J = {}
        J['delta_bladeLength', 'dx'] = np.reshape(dbl_ddx, (1, len(dbl_ddx)))
        J['delta_bladeLength', 'dy'] = np.reshape(dbl_ddy, (1, len(dbl_ddy)))
        J['delta_bladeLength', 'dz'] = np.reshape(dbl_ddz, (1, len(dbl_ddz)))
        J['delta_bladeLength', 'pitch'] = dbl_dpitch
        J['delta_bladeLength', 'theta'] = np.reshape(dbl_dthetastr, (1, len(dbl_dthetastr)))
        J['delta_bladeLength', 'r_in'] = np.zeros((1, m))
        J['delta_bladeLength', 'Rhub'] = dbl_drhub0
        J['delta_bladeLength', 'r'] = np.reshape(dbl_drstr0, (1, len(dbl_drstr0)))
        J['delta_bladeLength', 'precurve'] = np.reshape(dbl_dprecurvestr0, (1, len(dbl_dprecurvestr0)))
        J['delta_bladeLength', 'bladeLength'] = dbl_dbl0

        J['delta_precurve_sub', 'dx'] = dpcs_ddx
        J['delta_precurve_sub', 'dy'] = dpcs_ddy
        J['delta_precurve_sub', 'dz'] = dpcs_ddz
        J['delta_precurve_sub', 'pitch'] = dpcs_dpitch
        J['delta_precurve_sub', 'theta'] = dpcs_dthetastr
        J['delta_precurve_sub', 'r_in'] = dpcs_drsubpc0
        J['delta_precurve_sub', 'Rhub'] = np.zeros(m)
        J['delta_precurve_sub', 'r'] = dpcs_drstr0
        J['delta_precurve_sub', 'precurve'] = np.zeros((m, n))
        J['delta_precurve_sub', 'bladeLength'] = np.zeros(m)
        self.J = J

    def compute_partials(self, inputs, J):
        J = self.J        
        '''
        


class RootMoment(ExplicitComponent):
    """blade root bending moment"""
    def initialize(self):
        self.options.declare('NPTS')
    
    def setup(self):
        NPTS = self.options['NPTS']

        self.add_input('aeroloads_r', val=np.zeros(NPTS), units='m', desc='radial positions along blade going toward tip')
        self.add_input('aeroloads_Px', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned x-direction')
        self.add_input('aeroloads_Py', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned y-direction')
        self.add_input('aeroloads_Pz', val=np.zeros(NPTS), units='N/m', desc='distributed loads in blade-aligned z-direction')
        self.add_input('r', val=np.zeros(NPTS), units='m')
        self.add_input('totalCone', val=np.zeros(NPTS), units='deg', desc='total cone angle from precone and curvature')
        self.add_input('x_az', val=np.zeros(NPTS), units='m', desc='location of blade in azimuth x-coordinate system')
        self.add_input('y_az', val=np.zeros(NPTS), units='m', desc='location of blade in azimuth y-coordinate system')
        self.add_input('z_az', val=np.zeros(NPTS), units='m', desc='location of blade in azimuth z-coordinate system')
        self.add_input('s', val=np.zeros(NPTS), units='m', desc='cumulative path length along blade')

        self.add_input('dynamicFactor', val=1.0, desc='a dynamic amplification factor to adjust the static deflection calculation') #)

        self.add_output('root_bending_moment', val=0.0, units='N*m', desc='total magnitude of bending moment at root of blade')
        self.add_output('Mxyz', val=np.array([0.0, 0.0, 0.0]), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s.')
        self.add_output('Fxyz', val=np.array([0.0, 0.0, 0.0]), units='N', desc='individual forces [x,y,z] at the blade root in blade c.s.')

        #self.declare_partials(['root_bending_moment'],
        #                      ['r', 'aeroloads_r', 'aeroloads_Px', 'aeroloads_Py', 'aeroloads_Pz',
        #                       'totalCone', 'x_az', 'y_az', 'z_az', 's'])

    def compute(self, inputs, outputs):

        r_pts = inputs['r']
        totalCone = inputs['totalCone']
        x_az = inputs['x_az']
        y_az = inputs['y_az']
        z_az = inputs['z_az']
        s = inputs['s']

        r = r_pts
        x_az = x_az
        y_az = y_az
        z_az = z_az

        aeroloads_Px = inputs['aeroloads_Px'] * inputs['dynamicFactor']
        aeroloads_Py = inputs['aeroloads_Py'] * inputs['dynamicFactor']
        aeroloads_Pz = inputs['aeroloads_Pz'] * inputs['dynamicFactor']
        
        # aL = aeroloads
        # TODO: linearly interpolation is not C1 continuous.  it should work OK for now, but is not ideal
        Px, dPx_dr, dPx_dalr, dPx_dalPx = interp_with_deriv(r, inputs['aeroloads_r'], aeroloads_Px)
        Py, dPy_dr, dPy_dalr, dPy_dalPy = interp_with_deriv(r, inputs['aeroloads_r'], aeroloads_Py)
        Pz, dPz_dr, dPz_dalr, dPz_dalPz = interp_with_deriv(r, inputs['aeroloads_r'], aeroloads_Pz)

        # print 'al.Pz: ', aL.Pz #check=0

        Fx = np.trapz(Px, inputs['s'])
        Fy = np.trapz(Py, inputs['s'])
        Fz = np.trapz(Pz, inputs['s'])

        # loads in azimuthal c.s.
        P = DirectionVector(Px, Py, Pz).bladeToAzimuth(totalCone)

        # distributed bending load in azimuth coordinate ysstem
        az = DirectionVector(x_az, y_az, z_az)
        Mp = az.cross(P)

        # integrate
        Mx = np.trapz(Mp.x, s)
        My = np.trapz(Mp.y, s)
        Mz = np.trapz(Mp.z, s)

        # get total magnitude
        model_bending_moment = np.sqrt(Mx**2 + My**2 + Mz**2)

        outputs['Mxyz'] = np.array([Mx, My, Mz])
        outputs['Fxyz'] = np.array([Fx,Fy,Fz])
        # print 'Forces: ', outputs['Fxyz']
        outputs['root_bending_moment'] = model_bending_moment

        '''
        # dx_dr = -sind(precone)
        # dz_dr = cosd(precone)

        # dx_dprecone = -r*cosd(precone)*np.pi/180.0
        # dz_dprecone = -r*sind(precone)*np.pi/180.0

        dPx_dr = (P.dx['dx']*dPx_dr.T + P.dx['dy']*dPy_dr.T + P.dx['dz']*dPz_dr.T).T
        dPy_dr = (P.dy['dx']*dPx_dr.T + P.dy['dy']*dPy_dr.T + P.dy['dz']*dPz_dr.T).T
        dPz_dr = (P.dz['dx']*dPx_dr.T + P.dz['dy']*dPy_dr.T + P.dz['dz']*dPz_dr.T).T

        dPx_dalr = (P.dx['dx']*dPx_dalr.T + P.dx['dy']*dPy_dalr.T + P.dx['dz']*dPz_dalr.T).T
        dPy_dalr = (P.dy['dx']*dPx_dalr.T + P.dy['dy']*dPy_dalr.T + P.dy['dz']*dPz_dalr.T).T
        dPz_dalr = (P.dz['dx']*dPx_dalr.T + P.dz['dy']*dPy_dalr.T + P.dz['dz']*dPz_dalr.T).T

        dPx_dalPx = (P.dx['dx']*dPx_dalPx.T).T
        dPx_dalPy = (P.dx['dy']*dPy_dalPy.T).T
        dPx_dalPz = (P.dx['dz']*dPz_dalPz.T).T

        dPy_dalPx = (P.dy['dx']*dPx_dalPx.T).T
        dPy_dalPy = (P.dy['dy']*dPy_dalPy.T).T
        dPy_dalPz = (P.dy['dz']*dPz_dalPz.T).T

        dPz_dalPx = (P.dz['dx']*dPx_dalPx.T).T
        dPz_dalPy = (P.dz['dy']*dPy_dalPy.T).T
        dPz_dalPz = (P.dz['dz']*dPz_dalPz.T).T


        # dazx_dr = np.diag(az.dx['dx']*dx_dr + az.dx['dz']*dz_dr)
        # dazy_dr = np.diag(az.dy['dx']*dx_dr + az.dy['dz']*dz_dr)
        # dazz_dr = np.diag(az.dz['dx']*dx_dr + az.dz['dz']*dz_dr)

        # dazx_dprecone = (az.dx['dx']*dx_dprecone.T + az.dx['dz']*dz_dprecone.T).T
        # dazy_dprecone = (az.dy['dx']*dx_dprecone.T + az.dy['dz']*dz_dprecone.T).T
        # dazz_dprecone = (az.dz['dx']*dx_dprecone.T + az.dz['dz']*dz_dprecone.T).T

        dMpx, dMpy, dMpz = az.cross_deriv_array(P, namea='az', nameb='P')

        dMpx_dr = dMpx['dPx']*dPx_dr.T + dMpx['dPy']*dPy_dr.T + dMpx['dPz']*dPz_dr.T
        dMpy_dr = dMpy['dPx']*dPx_dr.T + dMpy['dPy']*dPy_dr.T + dMpy['dPz']*dPz_dr.T
        dMpz_dr = dMpz['dPx']*dPx_dr.T + dMpz['dPy']*dPy_dr.T + dMpz['dPz']*dPz_dr.T

        dMpx_dtotalcone = dMpx['dPx']*P.dx['dprecone'].T + dMpx['dPy']*P.dy['dprecone'].T + dMpx['dPz']*P.dz['dprecone'].T
        dMpy_dtotalcone = dMpy['dPx']*P.dx['dprecone'].T + dMpy['dPy']*P.dy['dprecone'].T + dMpy['dPz']*P.dz['dprecone'].T
        dMpz_dtotalcone = dMpz['dPx']*P.dx['dprecone'].T + dMpz['dPy']*P.dy['dprecone'].T + dMpz['dPz']*P.dz['dprecone'].T

        dMpx_dalr = (dMpx['dPx']*dPx_dalr.T + dMpx['dPy']*dPy_dalr.T + dMpx['dPz']*dPz_dalr.T).T
        dMpy_dalr = (dMpy['dPx']*dPx_dalr.T + dMpy['dPy']*dPy_dalr.T + dMpy['dPz']*dPz_dalr.T).T
        dMpz_dalr = (dMpz['dPx']*dPx_dalr.T + dMpz['dPy']*dPy_dalr.T + dMpz['dPz']*dPz_dalr.T).T

        dMpx_dalPx = (dMpx['dPx']*dPx_dalPx.T + dMpx['dPy']*dPy_dalPx.T + dMpx['dPz']*dPz_dalPx.T).T
        dMpy_dalPx = (dMpy['dPx']*dPx_dalPx.T + dMpy['dPy']*dPy_dalPx.T + dMpy['dPz']*dPz_dalPx.T).T
        dMpz_dalPx = (dMpz['dPx']*dPx_dalPx.T + dMpz['dPy']*dPy_dalPx.T + dMpz['dPz']*dPz_dalPx.T).T

        dMpx_dalPy = (dMpx['dPx']*dPx_dalPy.T + dMpx['dPy']*dPy_dalPy.T + dMpx['dPz']*dPz_dalPy.T).T
        dMpy_dalPy = (dMpy['dPx']*dPx_dalPy.T + dMpy['dPy']*dPy_dalPy.T + dMpy['dPz']*dPz_dalPy.T).T
        dMpz_dalPy = (dMpz['dPx']*dPx_dalPy.T + dMpz['dPy']*dPy_dalPy.T + dMpz['dPz']*dPz_dalPy.T).T

        dMpx_dalPz = (dMpx['dPx']*dPx_dalPz.T + dMpx['dPy']*dPy_dalPz.T + dMpx['dPz']*dPz_dalPz.T).T
        dMpy_dalPz = (dMpy['dPx']*dPx_dalPz.T + dMpy['dPy']*dPy_dalPz.T + dMpy['dPz']*dPz_dalPz.T).T
        dMpz_dalPz = (dMpz['dPx']*dPx_dalPz.T + dMpz['dPy']*dPy_dalPz.T + dMpz['dPz']*dPz_dalPz.T).T

        dMx_dMpx, dMx_ds = trapz_deriv(Mp.x, s)
        dMy_dMpy, dMy_ds = trapz_deriv(Mp.y, s)
        dMz_dMpz, dMz_ds = trapz_deriv(Mp.z, s)

        dMx_dr = np.dot(dMx_dMpx, dMpx_dr)
        dMy_dr = np.dot(dMy_dMpy, dMpy_dr)
        dMz_dr = np.dot(dMz_dMpz, dMpz_dr)

        dMx_dalr = np.dot(dMx_dMpx, dMpx_dalr)
        dMy_dalr = np.dot(dMy_dMpy, dMpy_dalr)
        dMz_dalr = np.dot(dMz_dMpz, dMpz_dalr)

        dMx_dalPx = np.dot(dMx_dMpx, dMpx_dalPx)
        dMy_dalPx = np.dot(dMy_dMpy, dMpy_dalPx)
        dMz_dalPx = np.dot(dMz_dMpz, dMpz_dalPx)

        dMx_dalPy = np.dot(dMx_dMpx, dMpx_dalPy)
        dMy_dalPy = np.dot(dMy_dMpy, dMpy_dalPy)
        dMz_dalPy = np.dot(dMz_dMpz, dMpz_dalPy)

        dMx_dalPz = np.dot(dMx_dMpx, dMpx_dalPz)
        dMy_dalPz = np.dot(dMy_dMpy, dMpy_dalPz)
        dMz_dalPz = np.dot(dMz_dMpz, dMpz_dalPz)

        dMx_dtotalcone = dMx_dMpx * dMpx_dtotalcone
        dMy_dtotalcone = dMy_dMpy * dMpy_dtotalcone
        dMz_dtotalcone = dMz_dMpz * dMpz_dtotalcone

        dMx_dazx = dMx_dMpx * dMpx['dazx']
        dMx_dazy = dMx_dMpx * dMpx['dazy']
        dMx_dazz = dMx_dMpx * dMpx['dazz']

        dMy_dazx = dMy_dMpy * dMpy['dazx']
        dMy_dazy = dMy_dMpy * dMpy['dazy']
        dMy_dazz = dMy_dMpy * dMpy['dazz']

        dMz_dazx = dMz_dMpz * dMpz['dazx']
        dMz_dazy = dMz_dMpz * dMpz['dazy']
        dMz_dazz = dMz_dMpz * dMpz['dazz']

        drbm_dr = (Mx*dMx_dr + My*dMy_dr + Mz*dMz_dr)/model_bending_moment
        drbm_dalr = (Mx*dMx_dalr + My*dMy_dalr + Mz*dMz_dalr)/model_bending_moment
        drbm_dalPx = (Mx*dMx_dalPx + My*dMy_dalPx + Mz*dMz_dalPx)/model_bending_moment
        drbm_dalPy = (Mx*dMx_dalPy + My*dMy_dalPy + Mz*dMz_dalPy)/model_bending_moment
        drbm_dalPz = (Mx*dMx_dalPz + My*dMy_dalPz + Mz*dMz_dalPz)/model_bending_moment
        drbm_dtotalcone = (Mx*dMx_dtotalcone + My*dMy_dtotalcone + Mz*dMz_dtotalcone)/model_bending_moment
        drbm_dazx = (Mx*dMx_dazx + My*dMy_dazx + Mz*dMz_dazx)/model_bending_moment
        drbm_dazy = (Mx*dMx_dazy + My*dMy_dazy + Mz*dMz_dazy)/model_bending_moment
        drbm_dazz = (Mx*dMx_dazz + My*dMy_dazz + Mz*dMz_dazz)/model_bending_moment
        drbm_ds = (Mx*dMx_ds + My*dMy_ds + Mz*dMz_ds)/model_bending_moment

        J = {}
        J['root_bending_moment', 'r'] = np.reshape(drbm_dr, (1, len(drbm_dr)))
        J['root_bending_moment', 'aeroloads_r'] = np.reshape(drbm_dalr, (1, len(drbm_dalr)))
        J['root_bending_moment', 'aeroloads_Px'] = np.reshape(drbm_dalPx, (1, len(drbm_dalPx)))
        J['root_bending_moment', 'aeroloads_Py'] = np.reshape(drbm_dalPy, (1, len(drbm_dalPy)))
        J['root_bending_moment', 'aeroloads_Pz'] = np.reshape(drbm_dalPz, (1, len(drbm_dalPz)))
        J['root_bending_moment', 'totalCone'] = np.reshape(drbm_dtotalcone, (1, len(drbm_dtotalcone)))
        J['root_bending_moment', 'x_az'] = np.reshape(drbm_dazx, (1, len(drbm_dazx)))
        J['root_bending_moment', 'y_az'] = np.reshape(drbm_dazy, (1, len(drbm_dazy)))
        J['root_bending_moment', 'z_az'] = np.reshape(drbm_dazz, (1, len(drbm_dazz)))
        J['root_bending_moment', 's'] = np.reshape(drbm_ds, (1, len(drbm_ds)))
        self.J = J
        
    def compute_partials(self, inputs, J):        
        J = self.J
        
        '''


class MassProperties(ExplicitComponent):
    def setup(self):
        # variables
        self.add_input('blade_mass',                val=0.0, units='kg',        desc='mass of one blade')
        self.add_input('blade_moment_of_inertia',   val=0.0, units='kg*m**2',   desc='mass moment of inertia of blade about hub')
        self.add_input('tilt',                      val=0.0, units='deg',       desc='rotor tilt angle (used to translate moments of inertia from hub to yaw c.s.')

        # parameters
        self.add_discrete_input('nBlades',  val=3,                      desc='number of blades')

        # outputs
        self.add_output('mass_all_blades', val=0.0, units='kg',         desc='mass of all blades')
        self.add_output('I_all_blades',    shape=6, units='kg*m**2',    desc='mass moments of inertia of all blades in yaw c.s. order:Ixx, Iyy, Izz, Ixy, Ixz, Iyz')

        #self.declare_partials(['mass_all_blades', 'I_all_blades'], 
        #                      ['blade_mass', 'blade_moment_of_inertia', 'tilt'])

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        blade_mass = inputs['blade_mass']
        blade_moment_of_inertia = inputs['blade_moment_of_inertia']
        tilt = inputs['tilt']
        nBlades = discrete_inputs['nBlades']

        mass_all_blades = nBlades * blade_mass

        Ibeam = nBlades * blade_moment_of_inertia

        Ixx = Ibeam
        Iyy = Ibeam/2.0  # azimuthal average for 2 blades, exact for 3+
        Izz = Ibeam/2.0
        Ixy = 0.0
        Ixz = 0.0
        Iyz = 0.0  # azimuthal average for 2 blades, exact for 3+

        # rotate to yaw c.s.
        I = DirectionVector(Ixx, Iyy, Izz).hubToYaw(tilt)  # because off-diagonal components are all zero

        I_all_blades = np.array([I.x, I.y, I.z, Ixy, Ixz, Iyz])

        outputs['mass_all_blades'] = mass_all_blades
        outputs['I_all_blades'] = I_all_blades
        '''
        dIx_dmoi = nBlades*(I.dx['dx'] + I.dx['dy']/2.0 + I.dx['dz']/2.0)
        dIy_dmoi = nBlades*(I.dy['dx'] + I.dy['dy']/2.0 + I.dy['dz']/2.0)
        dIz_dmoi = nBlades*(I.dz['dx'] + I.dz['dy']/2.0 + I.dz['dz']/2.0)

        J = {}
        J['mass_all_blades', 'blade_mass'] = nBlades
        J['mass_all_blades', 'blade_moment_of_inertia'] = 0.0
        J['mass_all_blades', 'tilt'] = 0.0
        J['I_all_blades', 'blade_mass'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        J['I_all_blades', 'blade_moment_of_inertia'] = np.array([dIx_dmoi, dIy_dmoi, dIz_dmoi, 0.0, 0.0, 0.0])
        J['I_all_blades', 'tilt'] = np.array([ I.dx['dtilt'],  I.dy['dtilt'],  I.dz['dtilt'], 0.0, 0.0, 0.0])
        self.J = J
        
    def compute_partials(self, inputs, J, discrete_inputs):
        J = self.J
        '''        


class ExtremeLoads(ExplicitComponent):
    def setup(self):
        # variables
        self.add_input('T', units='N', shape=((2,)), desc='rotor thrust, index 0 is at worst-case, index 1 feathered')
        self.add_input('Q', units='N*m', shape=((2,)), desc='rotor torque, index 0 is at worst-case, index 1 feathered')

        # parameters
        self.add_discrete_input('nBlades', val=3, desc='number of blades')

        # outputs
        self.add_output('T_extreme', val=0.0, units='N', desc='rotor thrust at survival wind condition')
        self.add_output('Q_extreme', val=0.0, units='N*m', desc='rotor torque at survival wind condition')

        #self.declare_partials(['T_extreme'],['T'])
        #self.declare_partials(['Q_extreme'],['Q'])

        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        n = float(discrete_inputs['nBlades'])
        T = inputs['T']
        Q = inputs['Q']
        T_extreme = (T[0] + T[1]*(n-1)) / n
        Q_extreme = (Q[0] + Q[1]*(n-1)) / n
        outputs['T_extreme'] = T_extreme
        outputs['Q_extreme'] = 0.0

        '''
    def compute_partials(self, inputs, J, discrete_inputs):
        n = float(discrete_inputs['nBlades'])
        
        J['T_extreme', 'T'] = np.reshape(np.array([[1.0/n], [(n-1)/n]]), (1, 2))
        # J['Q_extreme', 'Q'] = np.reshape(np.array([1.0/n, (n-1)/n]), (1, 2))
        J['Q_extreme', 'Q'] = 0.0
        '''


class GustETM(ExplicitComponent):
    def setup(self):
        # variables
        self.add_input('V_mean', val=0.0, units='m/s', desc='IEC average wind speed for turbine class')
        self.add_input('V_hub', val=0.0, units='m/s', desc='hub height wind speed')

        # parameters
        self.add_discrete_input('turbulence_class', val='A', desc='IEC turbulence class')
        self.add_discrete_input('std', val=3, desc='number of standard deviations for strength of gust')

        # out
        self.add_output('V_gust', val=0.0, units='m/s', desc='gust wind speed')

        #self.declare_partials(['V_gust'], ['V_mean', 'V_hub'])


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        V_mean = inputs['V_mean']
        V_hub = inputs['V_hub']
        turbulence_class = discrete_inputs['turbulence_class']
        std = discrete_inputs['std']

        if turbulence_class == 'A':
            Iref = 0.16
        elif turbulence_class == 'B':
            Iref = 0.14
        elif turbulence_class == 'C':
            Iref = 0.12

        c = 2.0
        sigma = c * Iref * (0.072*(V_mean/c + 3)*(V_hub/c - 4) + 10)
        V_gust = V_hub + std*sigma
        outputs['V_gust'] = V_gust
        '''
        J = {}
        J['V_gust', 'V_mean'] = std*(c*Iref*0.072/c*(V_hub/c - 4))
        J['V_gust', 'V_hub'] = 1.0 + std*(c*Iref*0.072*(V_mean/c + 3)/c)
        self.J = J
        
    def compute_partials(self, inputs, J, discrete_inputs):
        J = self.J
        '''

        
class SetupPCModVarSpeed(ExplicitComponent):
    def setup(self):
        self.add_input('control_tsr', desc='tip-speed ratio in Region 2 (should be optimized externally)')
        self.add_input('control_pitch', units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
        self.add_input('Vrated', val=0.0, units='m/s', desc='rated wind speed')
        self.add_input('R', val=0.0, units='m', desc='rotor radius')
        self.add_input('Vfactor', val=0.0, desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation')

        self.add_output('Uhub', val=0.0, units='m/s', desc='freestream velocities to run')
        self.add_output('Omega', val=0.0, units='rpm', desc='rotation speeds to run')
        self.add_output('pitch', val=0.0, units='deg', desc='pitch angles to run')
        self.add_output('azimuth', val=0.0, units='deg')

        #self.declare_partials(['Uhub', 'Omega', 'pitch'], ['control_tsr', 'control_pitch', 'Vrated', 'R'])

    def compute(self, inputs, outputs):

        Vrated = inputs['Vrated']
        R = inputs['R']
        Vfactor = inputs['Vfactor']

        Uhub = Vfactor * Vrated
        Omega = inputs['control_tsr']*Uhub/R*RS2RPM
        pitch = inputs['control_pitch']

        outputs['Uhub'] = Uhub
        outputs['Omega'] = Omega
        outputs['pitch'] = pitch
        outputs['azimuth'] = 0.0
        '''
        J = {}
        J['Uhub', 'control_tsr'] = 0.0
        J['Uhub', 'Vrated'] = Vfactor
        J['Uhub', 'R'] = 0.0
        J['Uhub', 'control_pitch'] = 0.0
        J['Omega', 'control_tsr'] = Uhub/R*RS2RPM
        J['Omega', 'Vrated'] = inputs['control_tsr']*Vfactor/R*RS2RPM
        J['Omega', 'R'] = -inputs['control_tsr']*Uhub/R**2*RS2RPM
        J['Omega', 'control_pitch'] = 0.0
        J['pitch', 'control_tsr'] = 0.0
        J['pitch', 'Vrated'] = 0.0
        J['pitch', 'R'] = 0.0
        J['pitch', 'control_pitch'] = 1.0
        self.J = J
        
    def compute_partials(self, inputs, J):
        J = self.J
        '''
        


class ConstraintsStructures(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
    
    def setup(self):
        NPTS = self.options['NPTS']

        self.add_discrete_input('nBlades', val=3, desc='number of blades')
        self.add_input('freq_pbeam', val=np.zeros(NFREQ), units='Hz', desc='1st nF natural frequencies')
        self.add_input('freq_curvefem', val=np.zeros(NFREQ), units='Hz', desc='1st nF natural frequencies')
        self.add_input('Omega', val=0.0, units='rpm', desc='rotation speed')
        self.add_input('strainU_spar', val=np.zeros(NPTS), desc='axial strain and specified locations')
        self.add_input('strainL_spar', val=np.zeros(NPTS), desc='axial strain and specified locations')
        self.add_input('strainU_te', val=np.zeros(NPTS), desc='axial strain and specified locations')
        self.add_input('strainL_te', val=np.zeros(NPTS), desc='axial strain and specified locations')
        self.add_input('strain_ult_spar', val=0.0, desc='ultimate strain in spar cap')
        self.add_input('strain_ult_te', val=0.0, desc='uptimate strain in trailing-edge panels')
        self.add_input('eps_crit_spar', val=np.zeros(NPTS), desc='critical strain in spar from panel buckling calculation')
        self.add_input('eps_crit_te', val=np.zeros(NPTS), desc='critical strain in trailing-edge panels from panel buckling calculation')
        self.add_input('damageU_spar', val=np.zeros(NPTS), desc='fatigue damage on upper surface in spar cap')
        self.add_input('damageL_spar', val=np.zeros(NPTS), desc='fatigue damage on lower surface in spar cap')
        self.add_input('damageU_te', val=np.zeros(NPTS), desc='fatigue damage on upper surface in trailing-edge panels')
        self.add_input('damageL_te', val=np.zeros(NPTS), desc='fatigue damage on lower surface in trailing-edge panels')
        self.add_input('gamma_f', 0.0, desc='safety factor on loads')
        self.add_input('gamma_m', 0.0, desc='safety factor on materials')
        self.add_input('gamma_freq', 0.0, desc='partial safety factor for fatigue')
        
        self.add_output('Pn_margin', shape=5, desc='Blade natural frequency (pBeam) relative to blade passing frequency')
        self.add_output('P1_margin', shape=5, desc='Blade natural frequency (pBeam) relative to rotor passing frequency')
        self.add_output('Pn_margin_cfem', shape=5, desc='Blade natural frequency (curvefem) relative to blade passing frequency')
        self.add_output('P1_margin_cfem', shape=5, desc='Blade natural frequency (curvefem) relative to rotor passing frequency')
        self.add_output('rotor_strain_sparU', val=np.zeros(NPTS), desc='Strain in upper spar relative to ultimate allowable')
        self.add_output('rotor_strain_sparL', val=np.zeros(NPTS), desc='Strain in lower spar relative to ultimate allowable')
        self.add_output('rotor_strain_teU', val=np.zeros(NPTS), desc='Strain in upper trailing edge relative to ultimate allowable')
        self.add_output('rotor_strain_teL', val=np.zeros(NPTS), desc='Strain in lower trailing edge relative to ultimate allowable')
        self.add_output('rotor_buckling_sparU', val=np.zeros(NPTS), desc='Buckling in upper spar relative to ultimate allowable')
        self.add_output('rotor_buckling_sparL', val=np.zeros(NPTS), desc='Buckling in lower spar relative to ultimate allowable')
        self.add_output('rotor_buckling_teU', val=np.zeros(NPTS), desc='Buckling in upper trailing edge relative to ultimate allowable')
        self.add_output('rotor_buckling_teL', val=np.zeros(NPTS), desc='Buckling in lower trailing edge relative to ultimate allowable')
        self.add_output('rotor_damage_sparU', val=np.zeros(NPTS), desc='Damage in upper spar relative to ultimate allowable')
        self.add_output('rotor_damage_sparL', val=np.zeros(NPTS), desc='Damage in lower spar relative to ultimate allowable')
        self.add_output('rotor_damage_teU', val=np.zeros(NPTS), desc='Damage in upper trailing edge relative to ultimate allowable')
        self.add_output('rotor_damage_teL', val=np.zeros(NPTS), desc='Damage in lower trailing edge relative to ultimate allowable')

        #self.declare_partials(['Pn_margin','P1_margin'], ['Omega','gamma_freq','freq_pbeam'])
        #self.declare_partials(['Pn_margin_cfem','P1_margin_cfem'], ['Omega','gamma_freq','freq_curvefem'])
        
        #self.declare_partials(['rotor_strain_sparU','rotor_strain_sparL','rotor_strain_teU','rotor_strain_teL'], ['gamma_f','gamma_m'])
        #self.declare_partials(['rotor_strain_sparU','rotor_strain_sparL'], ['strain_ult_spar'])
        #self.declare_partials(['rotor_strain_sparU'], ['strainU_spar'])
        #self.declare_partials(['rotor_strain_sparL'], ['strainL_spar'])
        #self.declare_partials(['rotor_strain_teU','rotor_strain_teL'], ['strain_ult_te'])
        #self.declare_partials(['rotor_strain_teU'], ['strainU_te'])
        #self.declare_partials(['rotor_strain_teL'], ['strainL_te'])

        #self.declare_partials(['rotor_buckling_sparU','rotor_buckling_sparL','rotor_buckling_teU','rotor_buckling_teL'], ['gamma_f'])
        #self.declare_partials(['rotor_buckling_sparU','rotor_buckling_sparL'], ['eps_crit_spar'])
        #self.declare_partials(['rotor_buckling_sparU'], ['strainU_spar'])
        #self.declare_partials(['rotor_buckling_sparL'], ['strainL_spar'])
        #self.declare_partials(['rotor_buckling_teU','rotor_buckling_teL'], ['eps_crit_te'])
        #self.declare_partials(['rotor_buckling_teU'], ['strainU_te'])
        #self.declare_partials(['rotor_buckling_teL'], ['strainL_te'])

        #self.declare_partials(['rotor_damage_sparU'], ['damageU_spar'])
        #self.declare_partials(['rotor_damage_sparL'], ['damageL_spar'])
        #self.declare_partials(['rotor_damage_teU'], ['damageU_te'])
        #self.declare_partials(['rotor_damage_teL'], ['damageL_te'])
        
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack variables
        omega           = inputs['Omega'] / 60.0 #Hz
        gamma_freq      = inputs['gamma_freq']
        gamma_f         = inputs['gamma_f']
        gamma_m         = inputs['gamma_m']
        gamma_strain    = gamma_f * gamma_m
        strain_ult_spar = inputs['strain_ult_spar']
        strain_ult_te   = inputs['strain_ult_te']
        eps_crit_spar   = inputs['eps_crit_spar']
        eps_crit_te     = inputs['eps_crit_te']
        nBlades         = discrete_inputs['nBlades']
        NPTS            = self.options['NPTS']
        strainU_spar    = inputs['strainU_spar']
        strainL_spar    = inputs['strainL_spar']
        strainU_te      = inputs['strainU_te']
        strainL_te      = inputs['strainL_te']
        
        outputs['Pn_margin'] = (nBlades*omega*gamma_freq) / inputs['freq_pbeam']
        outputs['P1_margin'] = (        omega*gamma_freq) / inputs['freq_pbeam']
        
        outputs['Pn_margin_cfem'] = (nBlades*omega*gamma_freq) / inputs['freq_curvefem']
        outputs['P1_margin_cfem'] = (        omega*gamma_freq) / inputs['freq_curvefem']

        if strain_ult_spar != 0.:
            outputs['rotor_strain_sparU'] = strainU_spar * gamma_strain / strain_ult_spar
            outputs['rotor_strain_sparL'] = strainL_spar * gamma_strain / strain_ult_spar
            outputs['rotor_strain_teU']   = strainU_te * gamma_strain / strain_ult_te
            outputs['rotor_strain_teL']   = strainL_te * gamma_strain / strain_ult_te

        outputs['rotor_buckling_sparU'] = np.divide( strainU_spar * gamma_f, eps_crit_spar, out=np.zeros(NPTS), where=eps_crit_spar!=0)
        outputs['rotor_buckling_sparL'] = np.divide( strainL_spar * gamma_f, eps_crit_spar, out=np.zeros(NPTS), where=eps_crit_spar!=0)
        outputs['rotor_buckling_teU']   = np.divide( strainU_te   * gamma_f, eps_crit_te, out=np.zeros(NPTS), where=eps_crit_te!=0)
        outputs['rotor_buckling_teL']   = np.divide( strainL_te   * gamma_f, eps_crit_te, out=np.zeros(NPTS), where=eps_crit_te!=0)

        outputs['rotor_damage_sparU'] = inputs['damageU_spar']
        outputs['rotor_damage_sparL'] = inputs['damageL_spar']
        outputs['rotor_damage_teU']   = inputs['damageU_te']
        outputs['rotor_damage_teL']   = inputs['damageL_te']
        '''
        myones = np.ones((NPTS,))
        J = {}
        J['Pn_margin','Omega']      = (nBlades*gamma_freq) / inputs['freq_pbeam']
        J['Pn_margin','gamma_freq'] = (nBlades*omega) / inputs['freq_pbeam']
        J['Pn_margin','freq_pbeam']       = -np.diag(outputs['Pn_margin'])  / inputs['freq_pbeam']
        J['P1_margin','Omega']      = gamma_freq / inputs['freq_pbeam']
        J['P1_margin','gamma_freq'] = omega / inputs['freq_pbeam']
        J['P1_margin','freq_pbeam']       = -np.diag(outputs['P1_margin'])  / inputs['freq_pbeam']

        J['Pn_margin_cfem','Omega']      = (nBlades*gamma_freq) / inputs['freq_curvefem']
        J['Pn_margin_cfem','gamma_freq'] = (nBlades*omega) / inputs['freq_curvefem']
        J['Pn_margin_cfem','freq_curvefem']  = -np.diag(outputs['Pn_margin_cfem'])  / inputs['freq_curvefem']
        J['P1_margin_cfem','Omega']      = gamma_freq / inputs['freq_curvefem']
        J['P1_margin_cfem','gamma_freq'] = omega / inputs['freq_curvefem']
        J['P1_margin_cfem','freq_curvefem']  = -np.diag(outputs['P1_margin_cfem'])  / inputs['freq_curvefem']
        
        if strain_ult_spar != 0.:
            J['rotor_strain_sparU', 'gamma_f'] = strainU_spar * gamma_m / strain_ult_spar
            J['rotor_strain_sparL', 'gamma_f'] = strainL_spar * gamma_m / strain_ult_spar
            J['rotor_strain_teU'  , 'gamma_f'] = strainU_te   * gamma_m / strain_ult_te
            J['rotor_strain_teL'  , 'gamma_f'] = strainL_te   * gamma_m / strain_ult_te

            J['rotor_strain_sparU', 'gamma_m'] = strainU_spar * gamma_f / strain_ult_spar
            J['rotor_strain_sparL', 'gamma_m'] = strainL_spar * gamma_f / strain_ult_spar
            J['rotor_strain_teU'  , 'gamma_m'] = strainU_te   * gamma_f / strain_ult_te
            J['rotor_strain_teL'  , 'gamma_m'] = strainL_te   * gamma_f / strain_ult_te

            J['rotor_strain_sparU', 'strainU_spar'] = gamma_strain * np.diag(myones) / strain_ult_spar
            J['rotor_strain_sparL', 'strainL_spar'] = gamma_strain * np.diag(myones) / strain_ult_spar
            J['rotor_strain_teU'  , 'strainU_te']   = gamma_strain * np.diag(myones) / strain_ult_te
            J['rotor_strain_teL'  , 'strainL_te']   = gamma_strain * np.diag(myones) / strain_ult_te

            J['rotor_strain_sparU', 'strain_ult_spar'] = -outputs['rotor_strain_sparU'] / strain_ult_spar
            J['rotor_strain_sparL', 'strain_ult_spar'] = -outputs['rotor_strain_sparL'] / strain_ult_spar
            J['rotor_strain_teU'  , 'strain_ult_te']   = -outputs['rotor_strain_teU']   / strain_ult_te
            J['rotor_strain_teL'  , 'strain_ult_te']   = -outputs['rotor_strain_teL']   / strain_ult_te
        
        J['rotor_buckling_sparU', 'gamma_f'] = np.divide(strainU_spar, eps_crit_spar, out=np.zeros(NPTS), where=eps_crit_spar!=0.0)
        J['rotor_buckling_sparL', 'gamma_f'] = np.divide(strainL_spar, eps_crit_spar, out=np.zeros(NPTS), where=eps_crit_spar!=0.0)
        J['rotor_buckling_teU'  , 'gamma_f'] = np.divide(strainU_te  , eps_crit_te, out=np.zeros(NPTS), where=eps_crit_te!=0.0)
        J['rotor_buckling_teL'  , 'gamma_f'] = np.divide(strainL_te  , eps_crit_te, out=np.zeros(NPTS), where=eps_crit_te!=0.0)

        J['rotor_buckling_sparU', 'strainU_spar'] = np.diag(myones) * np.divide(gamma_f, eps_crit_spar, out=np.zeros(NPTS), where=eps_crit_spar!=0.0)
        J['rotor_buckling_sparL', 'strainL_spar'] = np.diag(myones) * np.divide(gamma_f, eps_crit_spar, out=np.zeros(NPTS), where=eps_crit_spar!=0.0)
        J['rotor_buckling_teU'  , 'strainU_te']   = np.diag(myones) * np.divide(gamma_f, eps_crit_te, out=np.zeros(NPTS), where=eps_crit_te!=0.0)
        J['rotor_buckling_teL'  , 'strainL_te']   = np.diag(myones) * np.divide(gamma_f, eps_crit_te, out=np.zeros(NPTS), where=eps_crit_te!=0.0)

        J['rotor_buckling_sparU', 'eps_crit_spar'] = -np.diag(np.divide(outputs['rotor_buckling_sparU'], eps_crit_spar, out=np.zeros(NPTS), where=eps_crit_spar!=0.0))
        J['rotor_buckling_sparL', 'eps_crit_spar'] = -np.diag(np.divide(outputs['rotor_buckling_sparL'], eps_crit_spar, out=np.zeros(NPTS), where=eps_crit_spar!=0.0))
        J['rotor_buckling_teU'  , 'eps_crit_te']   = -np.diag(np.divide(outputs['rotor_buckling_teU']  , eps_crit_te, out=np.zeros(NPTS), where=eps_crit_te!=0.0))
        J['rotor_buckling_teL'  , 'eps_crit_te']   = -np.diag(np.divide(outputs['rotor_buckling_teL']  , eps_crit_te, out=np.zeros(NPTS), where=eps_crit_te!=0.0))
        
        J['rotor_damage_sparU', 'damageU_spar'] = np.diag(myones)
        J['rotor_damage_sparL', 'damageL_spar'] = np.diag(myones)
        J['rotor_damage_teU', 'damageU_te']     = np.diag(myones)
        J['rotor_damage_teL', 'damageL_te']     = np.diag(myones)
        self.J = J
        
    def compute_partials(self, inputs, J, discrete_inputs):
        J = self.J
        '''

        
    
class OutputsStructures(ExplicitComponent):
    def initialize(self):
        self.options.declare('NPTS')
        self.options.declare('NINPUT')
    
    def setup(self):
        NPTS   = self.options['NPTS']
        NINPUT = self.options['NINPUT']

        # structural outputs
        self.add_input('mass_one_blade_in',     val=0.0,            units='kg',     desc='mass of one blade')
        self.add_input('modes_coef_curvefem_in',val=np.zeros((3, 5)),               desc='mode shapes as 6th order polynomials, in the format accepted by ElastoDyn, [[c_x2, c_],..]')
        self.add_input('root_bending_moment_in',val=0.0,            units='N*m',    desc='total magnitude of bending moment at root of blade')
        self.add_input('Mxyz_in',               val=np.zeros(3),    units='N*m',    desc='bending moment at root of blade, x,y,z')
        self.add_input('delta_bladeLength_out_in',  val=0.0,        units='m',      desc='adjustment to blade length to account for curvature from loading')
        self.add_input('delta_precurve_sub_out_in', val=np.zeros(NINPUT), units='m',desc='adjustment to precurve to account for curvature from loading')
        # additional drivetrain moments output
        self.add_input('Fxyz_1_in', val=np.zeros((3,)), units='N',   desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #1)')
        self.add_input('Fxyz_2_in', val=np.zeros((3,)), units='N',   desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #2)')
        self.add_input('Fxyz_3_in', val=np.zeros((3,)), units='N',   desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #3)')
        self.add_input('Fxyz_4_in', val=np.zeros((3,)), units='N',   desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #4)')
        self.add_input('Fxyz_5_in', val=np.zeros((3,)), units='N',   desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #5)')
        self.add_input('Fxyz_6_in', val=np.zeros((3,)), units='N',   desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #6)')
        self.add_input('Mxyz_1_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #1)')
        self.add_input('Mxyz_2_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #2)')
        self.add_input('Mxyz_3_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #3)')
        self.add_input('Mxyz_4_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #4)')
        self.add_input('Mxyz_5_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #5)')
        self.add_input('Mxyz_6_in', val=np.zeros((3,)), units='N*m', desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #6)')
        self.add_input('TotalCone_in',      val=0.0,    units='rad', desc='total cone angle for blades at rated')
        self.add_input('Pitch_in',          val=0.0,    units='rad', desc='pitch angle at rated')
        self.add_discrete_input('nBlades',  val=3,                   desc='Number of blades on rotor')

        # structural outputs
        self.add_output('mass_one_blade',   val=0.0,    units='kg',         desc='mass of one blade')
        self.add_output('modes_coef_curvefem', val=np.zeros((3, 5)),        desc='mode shapes as 6th order polynomials, in the format accepted by ElastoDyn, [[c_x2, c_],..]')
        self.add_output('root_bending_moment', val=0.0, units='N*m',        desc='total magnitude of bending moment at root of blade')
        self.add_output('Mxyz',             val=np.zeros(3), units='N*m',   desc='bending moment at root of blade, x,y,z')
        self.add_output('delta_bladeLength_out', val=0.0, units='m',        desc='adjustment to blade length to account for curvature from loading')
        self.add_output('delta_precurve_sub_out',val=np.zeros(NINPUT), units='m', desc='adjustment to precurve to account for curvature from loading')
        # additional drivetrain moments output
        self.add_output('Fxyz_1', val=np.zeros((3,)), units='N',    desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #1)')
        self.add_output('Fxyz_2', val=np.zeros((3,)), units='N',    desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #2)')
        self.add_output('Fxyz_3', val=np.zeros((3,)), units='N',    desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #3)')
        self.add_output('Fxyz_4', val=np.zeros((3,)), units='N',    desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #4)')
        self.add_output('Fxyz_5', val=np.zeros((3,)), units='N',    desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #5)')
        self.add_output('Fxyz_6', val=np.zeros((3,)), units='N',    desc='individual forces [x,y,z] at the blade root in blade c.s. (blade #6)')
        self.add_output('Mxyz_1', val=np.zeros((3,)), units='N*m',  desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #1)')
        self.add_output('Mxyz_2', val=np.zeros((3,)), units='N*m',  desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #2)')
        self.add_output('Mxyz_3', val=np.zeros((3,)), units='N*m',  desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #3)')
        self.add_output('Mxyz_4', val=np.zeros((3,)), units='N*m',  desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #4)')
        self.add_output('Mxyz_5', val=np.zeros((3,)), units='N*m',  desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #5)')
        self.add_output('Mxyz_6', val=np.zeros((3,)), units='N*m',  desc='individual moments [x,y,z] at the blade root in blade c.s. (blade #6)')
        self.add_output('Fxyz_total',   val=np.zeros((3,)), units='N',      desc='Total force [x,y,z] at the blade root in *hub* c.s.')
        self.add_output('Mxyz_total',   val=np.zeros((3,)), units='N*m',    desc='individual moments [x,y,z] at the hub in *hub* c.s.')
        self.add_output('TotalCone',    val=0.0,            units='rad',    desc='total cone angle for blades at rated')
        self.add_output('Pitch',        val=0.0,            units='rad',    desc='pitch angle at rated')

        #self.declare_partials(['mass_one_blade'],   ['mass_one_blade_in'])
        #self.declare_partials(['root_bending_moment'], ['root_bending_moment_in'])
        #self.declare_partials(['Mxyz'],             ['Mxyz_in'])
        #self.declare_partials(['delta_bladeLength_out'],    ['delta_bladeLength_out_in'])
        #self.declare_partials(['delta_precurve_sub_out'],   ['delta_precurve_sub_out_in'])

        #for k in range(6):
            #kstr = '_'+str(k+1)
            #self.declare_partials(['Fxyz_total'], ['Fxyz'+kstr+'_in'])
            #self.declare_partials(['Mxyz_total'], ['Mxyz'+kstr+'_in'])
        #for k in range(1,7):
            #kstr = '_'+str(k)
            #self.declare_partials(['Fxyz'+kstr], ['Fxyz'+kstr+'_in'])
            #self.declare_partials(['Mxyz'+kstr], ['Mxyz'+kstr+'_in'])
        #self.declare_partials(['TotalCone'], ['TotalCone_in'])
        #self.declare_partials(['Pitch'], ['Pitch_in'])


    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        outputs['mass_one_blade'] = inputs['mass_one_blade_in']
        outputs['modes_coef_curvefem'] = inputs['modes_coef_curvefem_in']
        outputs['root_bending_moment'] = inputs['root_bending_moment_in']
        outputs['Mxyz'] = inputs['Mxyz_in']
        outputs['delta_bladeLength_out'] = inputs['delta_bladeLength_out_in']
        outputs['delta_precurve_sub_out'] = inputs['delta_precurve_sub_out_in']
        outputs['TotalCone'] = inputs['TotalCone_in']
        outputs['Pitch'] = inputs['Pitch_in']

        for k in range(1,7):
            kstr = '_'+str(k)
            outputs['Fxyz'+kstr] = np.copy( inputs['Fxyz'+kstr+'_in'] )
            outputs['Mxyz'+kstr] = np.copy( inputs['Mxyz'+kstr+'_in'] )

        # TODO: This is meant to sum up forces and torques across all blades while taking into account coordinate systems
        # This may not be necessary as CCBlade returns total thrust (T) and torque (Q), which are the only non-zero F & M entries anyway
        # The difficulty is that the answers don't match exactly.
        F_hub   = np.copy( np.array([inputs['Fxyz_1_in'], inputs['Fxyz_2_in'], inputs['Fxyz_3_in'], inputs['Fxyz_4_in'], inputs['Fxyz_5_in'], inputs['Fxyz_6_in']]) )
        M_hub   = np.copy( np.array([inputs['Mxyz_1_in'], inputs['Mxyz_2_in'], inputs['Mxyz_3_in'], inputs['Mxyz_4_in'], inputs['Mxyz_5_in'], inputs['Mxyz_6_in']]) )
        
        nBlades = discrete_inputs['nBlades']
        angles  = np.linspace(0, 360, nBlades+1)
        # Initialize summation
        F_hub_tot = np.zeros((3,))
        M_hub_tot = np.zeros((3,))
        dFx_dF    = np.zeros(F_hub.shape)
        dFy_dF    = np.zeros(F_hub.shape)
        dFz_dF    = np.zeros(F_hub.shape)
        dMx_dM    = np.zeros(M_hub.shape)
        dMy_dM    = np.zeros(M_hub.shape)
        dMz_dM    = np.zeros(M_hub.shape)
        # Convert from blade to hub c.s.
        for row in range(nBlades):
            myF = DirectionVector.fromArray(F_hub[row,:]).azimuthToHub(angles[row])
            myM = DirectionVector.fromArray(M_hub[row,:]).azimuthToHub(angles[row])
            
            F_hub_tot += myF.toArray()
            M_hub_tot += myM.toArray()

            dFx_dF[row,:] = np.array([myF.dx['dx'], myF.dx['dy'], myF.dx['dz']])
            dFy_dF[row,:] = np.array([myF.dy['dx'], myF.dy['dy'], myF.dy['dz']])
            dFz_dF[row,:] = np.array([myF.dz['dx'], myF.dz['dy'], myF.dz['dz']])

            dMx_dM[row,:] = np.array([myM.dx['dx'], myM.dx['dy'], myM.dx['dz']])
            dMy_dM[row,:] = np.array([myM.dy['dx'], myM.dy['dy'], myM.dy['dz']])
            dMz_dM[row,:] = np.array([myM.dz['dx'], myM.dz['dy'], myM.dz['dz']])

        # Now sum over all blades
        outputs['Fxyz_total'] = F_hub_tot
        outputs['Mxyz_total'] = M_hub_tot
        '''
        self.J = {}
        for k in range(6):
            kstr = '_'+str(k+1)
            self.J['Fxyz_total','Fxyz'+kstr+'_in'] = np.vstack([dFx_dF[k,:], dFy_dF[k,:], dFz_dF[k,:]])
            self.J['Mxyz_total','Mxyz'+kstr+'_in'] = np.vstack([dMx_dM[k,:], dMy_dM[k,:], dMz_dM[k,:]])

    def compute_partials(self, inputs, J, discrete_inputs):
        
        J = self.J

        J['mass_one_blade', 'mass_one_blade_in'] = 1
        J['root_bending_moment', 'root_bending_moment_in'] = 1
        J['Mxyz', 'Mxyz_in'] = np.diag(np.ones(len(inputs['Mxyz_in'])))
        J['delta_bladeLength_out', 'delta_bladeLength_out_in'] = 1
        J['delta_precurve_sub_out', 'delta_precurve_sub_out_in'] = np.diag(np.ones(len(inputs['delta_precurve_sub_out_in'])))

        for k in range(1,7):
            kstr = '_'+str(k)
            J['Fxyz'+kstr, 'Fxyz'+kstr+'_in'] = np.diag(np.ones(len(inputs['Fxyz'+kstr+'_in'])))
            J['Mxyz'+kstr, 'Mxyz'+kstr+'_in'] = np.diag(np.ones(len(inputs['Mxyz'+kstr+'_in'])))
        J['TotalCone', 'TotalCone_in'] = 1
        J['Pitch', 'Pitch_in'] = 1
        # J = self.J
        '''        



class RotorStructure(Group):
    def initialize(self):
        self.options.declare('RefBlade')
        self.options.declare('topLevelFlag',default=False)
        self.options.declare('Analysis_Level',default=0)
        self.options.declare('user_update_routine', default=None)
        
    def setup(self):
        RefBlade            = self.options['RefBlade']
        NPTS                = len(RefBlade['pf']['s'])
        NINPUT              = len(RefBlade['ctrl_pts']['r_in'])
        NAFgrid             = len(RefBlade['airfoils_aoa'])
        NRe                 = len(RefBlade['airfoils_Re'])
        topLevelFlag        = self.options['topLevelFlag']
        Analysis_Level      = self.options['Analysis_Level']
        user_update_routine = self.options['user_update_routine']
        
        structIndeps = IndepVarComp()
        structIndeps.add_discrete_output('fst_vt_in', val={})
        structIndeps.add_output('VfactorPC',                val=0.7,            desc='fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation')
        structIndeps.add_discrete_output('turbulence_class', val='A',           desc='IEC turbulence class class')
        structIndeps.add_discrete_output('gust_stddev',     val=3)
        structIndeps.add_output('pitch_extreme',            val=0.0, units='deg', desc='worst-case pitch at survival wind condition')
        structIndeps.add_output('azimuth_extreme',          val=0.0, units='deg', desc='worst-case azimuth at survival wind condition')
        structIndeps.add_output('rstar_damage',             val=np.zeros(NPTS+1), desc='nondimensional radial locations of damage equivalent moments')
        structIndeps.add_output('Mxb_damage',               val=np.zeros(NPTS+1), units='N*m', desc='damage equivalent moments about blade c.s. x-direction')
        structIndeps.add_output('Myb_damage',               val=np.zeros(NPTS+1), units='N*m', desc='damage equivalent moments about blade c.s. y-direction')
        structIndeps.add_output('strain_ult_spar',          val=0.01,           desc='ultimate strain in spar cap')
        structIndeps.add_output('strain_ult_te',            val=2500*1e-6,      desc='uptimate strain in trailing-edge panels')
        structIndeps.add_output('m_damage',                 val=10.0,           desc='slope of S-N curve for fatigue analysis')
        structIndeps.add_output('gamma_fatigue',            val=1.755,          desc='safety factor for fatigue')
        structIndeps.add_output('gamma_freq',               val=1.1,            desc='safety factor for resonant frequencies')
        structIndeps.add_output('gamma_f',                  val=1.35,           desc='safety factor for loads/stresses')
        structIndeps.add_output('gamma_m',                  val=1.1,            desc='safety factor for materials')
        structIndeps.add_output('dynamic_amplification',    val=1.0,            desc='a dynamic amplification factor to adjust the static deflection calculation')
        structIndeps.add_output('azimuth_load180',          val=180.0,  units='deg')
        structIndeps.add_output('azimuth_load0',            val=0.0,    units='deg')
        structIndeps.add_output('azimuth_load120',          val=120.0,  units='deg')
        structIndeps.add_output('azimuth_load240',          val=240.0,  units='deg')

        self.add_subsystem('structIndeps', structIndeps, promotes=['*'])

        if topLevelFlag:
            sharedIndeps = IndepVarComp()
            sharedIndeps.add_output('hub_height',    val=0.0,        units='m')
            sharedIndeps.add_output('rho',           val=1.225,      units='kg/m**3')
            sharedIndeps.add_output('mu',            val=1.81e-5,    units='kg/(m*s)')
            sharedIndeps.add_output('V_hub',         val=0.0,        units='m/s')
            sharedIndeps.add_output('Omega_rated',   val=0.0,        units='rpm')
            sharedIndeps.add_output('shearExp',      val=0.2)
            sharedIndeps.add_output('lifetime',      val=20.0,       units='year', desc='project lifetime for fatigue analysis')
            sharedIndeps.add_output('control_tsr',   val=0.0, desc='tip-speed ratio in Region 2 (should be optimized externally)')
            sharedIndeps.add_output('control_pitch', val=0.0,       units='deg', desc='pitch angle in region 2 (and region 3 for fixed pitch machines)')
            sharedIndeps.add_discrete_output('tiploss', True)
            sharedIndeps.add_discrete_output('hubloss', True)
            sharedIndeps.add_discrete_output('wakerotation', True)
            sharedIndeps.add_discrete_output('usecd', True)
            sharedIndeps.add_discrete_output('nSector', val=4, desc='number of sectors to divide rotor face into in computing thrust and power')
            self.add_subsystem('sharedIndeps', sharedIndeps, promotes=['*'])
            
            # Geometry
            self.add_subsystem('rotorGeometry', RotorGeometry(RefBlade=RefBlade, topLevelFlag=topLevelFlag, user_update_routine=user_update_routine), promotes=['*'])

        # --- add structures ---
        promoteList = ['nSector','rho','mu','shearExp','tiploss','hubloss','wakerotation','usecd',
                       'precone','precurveTip','tilt','yaw','nBlades','hub_height',
                       'chord','theta','precurve','Rhub','Rtip','r',
                       'airfoils_cl','airfoils_cd','airfoils_cm','airfoils_aoa','airfoils_Re']
        self.add_subsystem('curvature', BladeCurvature(NPTS=NPTS), promotes=['r','precone','precurve','presweep','totalCone','x_az','y_az','z_az','s'])
        self.add_subsystem('gust',      GustETM(), promotes=['V_mean','turbulence_class','V_hub'])
        self.add_subsystem('setuppc',   SetupPCModVarSpeed(),promotes=['R','control_tsr','control_pitch'])

        self.add_subsystem('aero_rated',            CCBladeLoads(naero=NPTS, npower=1, n_aoa_grid=NAFgrid, n_Re_grid=NRe), promotes=promoteList)
        self.add_subsystem('aero_extrm',            CCBladeLoads(naero=NPTS, npower=1, n_aoa_grid=NAFgrid, n_Re_grid=NRe), promotes=promoteList)
        self.add_subsystem('aero_extrm_forces',     CCBladePower(naero=NPTS, npower=2, n_aoa_grid=NAFgrid, n_Re_grid=NRe), promotes=promoteList)
        self.add_subsystem('aero_defl_powercurve',  CCBladeLoads(naero=NPTS, npower=1, n_aoa_grid=NAFgrid, n_Re_grid=NRe), promotes=promoteList)

        # Out of plane loads
        self.add_subsystem('aero_rated_0',    CCBladeLoads(naero=NPTS, npower=1,  n_aoa_grid=NAFgrid, n_Re_grid=NRe), promotes=promoteList)
        self.add_subsystem('aero_rated_120',  CCBladeLoads(naero=NPTS, npower=1,  n_aoa_grid=NAFgrid, n_Re_grid=NRe), promotes=promoteList)
        self.add_subsystem('aero_rated_240',  CCBladeLoads(naero=NPTS, npower=1,  n_aoa_grid=NAFgrid, n_Re_grid=NRe), promotes=promoteList)
        
        self.add_subsystem('loads_defl',        TotalLoads(NPTS=NPTS), promotes=['tilt','theta','rhoA','z','totalCone','z_az'])
        self.add_subsystem('loads_pc_defl',     TotalLoads(NPTS=NPTS), promotes=['tilt','theta','rhoA','z','totalCone','z_az'])
        self.add_subsystem('loads_strain',      TotalLoads(NPTS=NPTS), promotes=['tilt','theta','rhoA','z','totalCone','z_az'])

        self.add_subsystem('damage',        DamageLoads(NPTS=NPTS), promotes=['theta','z'])
        self.add_subsystem('struc',         RotorWithpBEAM(NPTS=NPTS), promotes=['*'])#gamma_fatigue','lifetime'])
        self.add_subsystem('curvefem',      CurveFEM(NPTS=NPTS), promotes=['*'])
        self.add_subsystem('tip',           TipDeflection(), promotes=['gamma_m','precone','tilt',
                                                                       'precurveTip','presweepTip',
                                                                       'downwind','hub_height','Rtip',
                                                                       'tip_deflection','tip_position','ground_clearance'])
        if not Analysis_Level>1:
            self.add_subsystem('root_moment',   RootMoment(NPTS=NPTS), promotes=['r','totalCone','x_az','y_az','z_az','s'])
        self.add_subsystem('mass',              MassProperties(), promotes=['tilt','nBlades','mass_all_blades','I_all_blades'])
        self.add_subsystem('extreme',           ExtremeLoads(), promotes=['nBlades'])
        self.add_subsystem('blade_defl',        BladeDeflection(NPTS=NPTS, NINPUT=NINPUT), promotes=['bladeLength','theta','Rhub',
                                                                                                     'precurve','r','r_in'])

        self.add_subsystem('root_moment_0',     RootMoment(NPTS=NPTS), promotes=['r','totalCone','x_az','y_az','z_az','s'])
        self.add_subsystem('root_moment_120',   RootMoment(NPTS=NPTS), promotes=['r','totalCone','x_az','y_az','z_az','s'])
        self.add_subsystem('root_moment_240',   RootMoment(NPTS=NPTS), promotes=['r','totalCone','x_az','y_az','z_az','s'])

        self.add_subsystem('output_struc',      OutputsStructures(NPTS=NPTS, NINPUT=NINPUT), promotes=['*'])
        self.add_subsystem('constraints',       ConstraintsStructures(NPTS=NPTS), promotes=['*'])

        
        # connections to gust
        if topLevelFlag:
            self.connect('V_hub', 'setuppc.Vrated')
        self.connect('gust_stddev', 'gust.std')
        
        # connections to setuppc
        #self.connect('geom.R',      'setuppc.R')
        self.connect('VfactorPC',   'setuppc.Vfactor')
        self.connect('gust.V_gust',            ['aero_rated.V_load',    'aero_rated_0.V_load',      'aero_rated_120.V_load',      'aero_rated_240.V_load'])
        
        if topLevelFlag:
            self.connect('Omega_rated', ['Omega', 'aero_rated.Omega_load', 'aero_rated_0.Omega_load','aero_rated_120.Omega_load','aero_rated_240.Omega_load'])
        
        # connections to aero_extrm (for max strain)
        if topLevelFlag:
            self.connect('V_extreme50',    'aero_extrm.V_load')
        self.connect('pitch_extreme',               'aero_extrm.pitch_load')
        self.connect('azimuth_extreme',             'aero_extrm.azimuth_load')
        self.aero_extrm.Omega_load = 0.0  # parked case

        # connections to aero_extrm_forces (for tower thrust)
        self.aero_extrm_forces.Uhub  = np.zeros(2)
        self.aero_extrm_forces.Omega = np.zeros(2)  # parked case
        self.aero_extrm_forces.pitch = np.zeros(2)
        if topLevelFlag:
            self.connect('V_extreme_full', 'aero_extrm_forces.Uhub')
        self.aero_extrm_forces.pitch =  np.array([0.0, 90.0])  # feathered
        self.aero_extrm_forces.T =      np.zeros(2)
        self.aero_extrm_forces.Q =      np.zeros(2)

        # connections to aero_defl_powercurve (for gust reversal)
        self.connect('setuppc.Uhub',    'aero_defl_powercurve.V_load')
        self.connect('setuppc.Omega',   'aero_defl_powercurve.Omega_load')
        self.connect('setuppc.pitch',   'aero_defl_powercurve.pitch_load')
        self.connect('setuppc.azimuth', 'aero_defl_powercurve.azimuth_load')
        self.aero_defl_powercurve.azimuth_load = 0.0

        # connections to loads_defl
        self.connect('aero_rated.loads_Omega',  'loads_defl.aeroloads_Omega')
        self.connect('aero_rated.loads_Px',     'loads_defl.aeroloads_Px')
        self.connect('aero_rated.loads_Py',     'loads_defl.aeroloads_Py')
        self.connect('aero_rated.loads_Pz',     'loads_defl.aeroloads_Pz')
        self.connect('aero_rated.loads_azimuth','loads_defl.aeroloads_azimuth')
        self.connect('aero_rated.loads_pitch',  'loads_defl.aeroloads_pitch')
        self.connect('aero_rated.loads_r',      'loads_defl.aeroloads_r')
        self.connect('dynamic_amplification',   'loads_defl.dynamicFactor')

        # connections to loads_pc_defl
        self.connect('aero_defl_powercurve.loads_Omega',    'loads_pc_defl.aeroloads_Omega')
        self.connect('aero_defl_powercurve.loads_Px',       'loads_pc_defl.aeroloads_Px')
        self.connect('aero_defl_powercurve.loads_Py',       'loads_pc_defl.aeroloads_Py')
        self.connect('aero_defl_powercurve.loads_Pz',       'loads_pc_defl.aeroloads_Pz')
        self.connect('aero_defl_powercurve.loads_azimuth',  'loads_pc_defl.aeroloads_azimuth')
        self.connect('aero_defl_powercurve.loads_pitch',    'loads_pc_defl.aeroloads_pitch')
        self.connect('aero_defl_powercurve.loads_r',        'loads_pc_defl.aeroloads_r')
        self.connect('dynamic_amplification',               'loads_pc_defl.dynamicFactor')

        # connections to loads_strain
        if Analysis_Level<1:
            self.connect('aero_extrm.loads_Px',         'loads_strain.aeroloads_Px')
            self.connect('aero_extrm.loads_Py',         'loads_strain.aeroloads_Py')
            self.connect('aero_extrm.loads_Pz',         'loads_strain.aeroloads_Pz')
            self.connect('aero_extrm.loads_Omega',      'loads_strain.aeroloads_Omega')
            self.connect('aero_extrm.loads_azimuth',    'loads_strain.aeroloads_azimuth')
            self.connect('aero_extrm.loads_pitch',      'loads_strain.aeroloads_pitch')
            
        self.connect('aero_extrm.loads_r',      'loads_strain.aeroloads_r')
        self.connect('dynamic_amplification',   'loads_strain.dynamicFactor')

        # connections to damage
        self.connect('rstar_damage',    'damage.rstar')
        self.connect('Mxb_damage',      'damage.Mxb')
        self.connect('Myb_damage',      'damage.Myb')

        # connections to struc
        self.connect('damage.Mxa',          'Mx_damage')
        self.connect('damage.Mya',          'My_damage')
        self.connect('loads_defl.Px_af',    'Px_defl')
        self.connect('loads_defl.Py_af',    'Py_defl')
        self.connect('loads_defl.Pz_af',    'Pz_defl')
        self.connect('loads_pc_defl.Px_af', 'Px_pc_defl')
        self.connect('loads_pc_defl.Py_af', 'Py_pc_defl')
        self.connect('loads_pc_defl.Pz_af', 'Pz_pc_defl')
        self.connect('loads_strain.Px_af',  'Px_strain')
        self.connect('loads_strain.Py_af',  'Py_strain')
        self.connect('loads_strain.Pz_af',  'Pz_strain')

        # connections to tip
        if Analysis_Level<1:
            self.connect('dx_defl', 'tip.dx', src_indices=[NPTS-1])
            self.connect('dy_defl', 'tip.dy', src_indices=[NPTS-1])
            self.connect('dz_defl', 'tip.dz', src_indices=[NPTS-1])
        if topLevelFlag:
            self.connect('theta', 'tip.theta', src_indices=[NPTS-1])
        self.connect('aero_rated.loads_pitch',      'tip.pitch')
        self.connect('aero_rated.loads_azimuth',    'tip.azimuth')
        self.connect('totalCone',         'tip.totalConeTip', src_indices=[NPTS-1])
        self.connect('dynamic_amplification',       'tip.dynamicFactor')


        # connections to root moment
        if not Analysis_Level>1:
            self.connect('aero_rated.loads_Px',     'root_moment.aeroloads_Px')
            self.connect('aero_rated.loads_Py',     'root_moment.aeroloads_Py')
            self.connect('aero_rated.loads_Pz',     'root_moment.aeroloads_Pz')
            self.connect('aero_rated.loads_r',      'root_moment.aeroloads_r')
            self.connect('dynamic_amplification',   'root_moment.dynamicFactor')

        # connections to mass
        self.connect('blade_mass',                'mass.blade_mass')
        self.connect('blade_moment_of_inertia',   'mass.blade_moment_of_inertia')

        # connectsion to extreme
        self.connect('aero_extrm_forces.T', 'extreme.T')
        self.connect('aero_extrm_forces.Q', 'extreme.Q')

        # connections to blade_defl
        self.connect('dx_pc_defl',                    'blade_defl.dx')
        self.connect('dy_pc_defl',                    'blade_defl.dy')
        self.connect('dz_pc_defl',                    'blade_defl.dz')
        self.connect('aero_defl_powercurve.loads_pitch',    'blade_defl.pitch')

        # connect to outputs
        self.connect('blade_mass',                'mass_one_blade_in')
        self.connect('modes_coef', 'modes_coef_curvefem_in')
        if Analysis_Level<1:
            self.connect('root_moment.root_bending_moment', 'root_bending_moment_in')
            self.connect('root_moment.Mxyz',                'Mxyz_in')
        self.connect('blade_defl.delta_bladeLength',    'delta_bladeLength_out_in')
        self.connect('blade_defl.delta_precurve_sub',   'delta_precurve_sub_out_in')
        
        self.connect('azimuth_load180', 'aero_rated.azimuth_load') # Blade position closest to root for max tip deflection constraint
        self.connect('azimuth_load0',   'aero_rated_0.azimuth_load')
        self.connect('azimuth_load120', 'aero_rated_120.azimuth_load')
        self.connect('azimuth_load240', 'aero_rated_240.azimuth_load')
        
        # connections to root moment for drivetrain
        self.connect('aero_rated_0.loads_Px',   'root_moment_0.aeroloads_Px')
        self.connect('aero_rated_120.loads_Px', 'root_moment_120.aeroloads_Px')
        self.connect('aero_rated_240.loads_Px', 'root_moment_240.aeroloads_Px')
        self.connect('aero_rated_0.loads_Py',   'root_moment_0.aeroloads_Py')
        self.connect('aero_rated_120.loads_Py', 'root_moment_120.aeroloads_Py')
        self.connect('aero_rated_240.loads_Py', 'root_moment_240.aeroloads_Py')
        self.connect('aero_rated_0.loads_Pz',   'root_moment_0.aeroloads_Pz')
        self.connect('aero_rated_120.loads_Pz', 'root_moment_120.aeroloads_Pz')
        self.connect('aero_rated_240.loads_Pz', 'root_moment_240.aeroloads_Pz')
        self.connect('aero_rated_0.loads_r',    'root_moment_0.aeroloads_r')
        self.connect('aero_rated_120.loads_r',  'root_moment_120.aeroloads_r')
        self.connect('aero_rated_240.loads_r',  'root_moment_240.aeroloads_r')
                
        self.connect('dynamic_amplification', ['root_moment_0.dynamicFactor', 'root_moment_120.dynamicFactor','root_moment_240.dynamicFactor'])

        # connections to root Mxyz outputs
        self.connect('root_moment_0.Mxyz',      'Mxyz_1_in')
        self.connect('root_moment_120.Mxyz',    'Mxyz_2_in')
        self.connect('root_moment_240.Mxyz',    'Mxyz_3_in')
        self.connect('totalCone',     'TotalCone_in', src_indices=[NPTS-1])
        # self.connect('aero_rated.pitch_load',   'Pitch_in')
        self.connect('root_moment_0.Fxyz',      'Fxyz_1_in')
        self.connect('root_moment_120.Fxyz',    'Fxyz_2_in')
        self.connect('root_moment_240.Fxyz',    'Fxyz_3_in')


def Init_RotorStructure_wRefBlade(rotor, blade):
    # === blade grid ===
    rotor['hubFraction']      = blade['config']['hubD']/2./blade['pf']['r'][-1] #0.025  # (Float): hub location as fraction of radius
    rotor['bladeLength']      = blade['ctrl_pts']['bladeLength'] #61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
    rotor['precone']          = blade['config']['cone_angle'] #2.5  # (Float, deg): precone angle
    rotor['tilt']             = blade['config']['tilt_angle'] #5.0  # (Float, deg): shaft tilt
    rotor['yaw']              = 0.0  # (Float, deg): yaw error
    rotor['nBlades']          = blade['config']['number_of_blades'] #3  # (Int): number of blades
    # ------------------
    
    # === blade geometry ===
    rotor['r_max_chord']      = blade['ctrl_pts']['r_max_chord']  # 0.23577 #(Float): location of max chord on unit radius
    rotor['chord_in']         = np.array(blade['ctrl_pts']['chord_in']) # np.array([3.2612, 4.3254, 4.5709, 3.7355, 2.69923333, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
    rotor['theta_in']         = np.array(blade['ctrl_pts']['theta_in']) # np.array([0.0, 13.2783, 12.30514836,  6.95106536,  2.72696309, -0.0878099]) # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
    rotor['precurve_in']      = np.array(blade['ctrl_pts']['precurve_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['presweep_in']      = np.array(blade['ctrl_pts']['presweep_in']) #np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
    rotor['sparT_in']         = np.array(blade['ctrl_pts']['sparT_in']) # np.array([0.0, 0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
    rotor['teT_in']           = np.array(blade['ctrl_pts']['teT_in']) # np.array([0.0, 0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
    # if 'le_var' in blade['precomp']['le_var']:
    #     rotor['leT_in']       = np.array(blade['ctrl_pts']['leT_in']) # (Array, m): leading-edge thickness parameters
    # rotor['thickness_in']     = np.array(blade['ctrl_pts']['thickness_in'])
    rotor['airfoil_position'] = np.array(blade['outer_shape_bem']['airfoil_position']['grid'])
    # ------------------

    # === atmosphere ===
    rotor['aero_rated_0.rho']       = 1.225  # (Float, kg/m**3): density of air
    rotor['aero_rated_0.mu']        = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
    rotor['aero_rated_0.shearExp']  = 0.25  # (Float): shear exponent
    rotor['hub_height']       = blade['config']['hub_height']  # (Float, m): hub height
    rotor['turbine_class']    = blade['config']['turbine_class'].upper() #TURBINE_CLASS['I']  # (Enum): IEC turbine class
    rotor['turbulence_class'] = blade['config']['turbulence_class'].upper()  # (Enum): IEC turbulence class class
    rotor['gust_stddev']      = 3
    # ----------------------

    # === control ===
    rotor['control_tsr']      = blade['config']['tsr'] #7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
    rotor['control_pitch']    = blade['config']['pitch'] #0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
    rotor['pitch_extreme']    = 0.0  # (Float, deg): worst-case pitch at survival wind condition
    rotor['azimuth_extreme']  = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
    rotor['VfactorPC']        = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
    # ----------------------

    # === aero and structural analysis options ===
    rotor['nSector'] = 4  # (Int): number of sectors to divide rotor face into in computing thrust and power
    rotor['dynamic_amplification'] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
    # ----------------------

    # === fatigue ===
    r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
                   0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
                   0.97777724])  # (Array): new aerodynamic grid on unit radius
    rstar_damage = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
        0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
    Mxb_damage = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
        1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
        1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
    Myb_damage = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
        1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
        3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
    xp = np.r_[0.0, r_aero]
    xx = np.r_[0.0, blade['pf']['s']]
    rotor['rstar_damage'] = np.interp(xx, xp, rstar_damage)
    rotor['Mxb_damage'] = np.interp(xx, xp, Mxb_damage)
    rotor['Myb_damage'] = np.interp(xx, xp, Myb_damage)
    rotor['strain_ult_spar'] = 1.0e-2  # (Float): ultimate strain in spar cap
    rotor['strain_ult_te'] = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
    rotor['gamma_fatigue'] = 1.755 # (Float): safety factor for fatigue
    rotor['gamma_f'] = 1.35 # (Float): safety factor for loads/stresses
    rotor['gamma_m'] = 1.1 # (Float): safety factor for materials
    rotor['gamma_freq'] = 1.1 # (Float): safety factor for resonant frequencies
    rotor['m_damage'] = 10.0  # (Float): slope of S-N curve for fatigue analysis
    rotor['lifetime'] = 20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
    # ----------------

    # Adding in only in rotor_structure- otherwise would have been connected in larger assembly
    rotor['gust.V_hub'] = 11.7386065326
    rotor['aero_rated.Omega_load'] = 12.1
    rotor['aero_rated.pitch_load'] = rotor['control_pitch']

    return rotor

if __name__ == '__main__':
    # Turbine Ontology input
    fname_input  = "turbine_inputs/nrel5mw_mod_update.yaml"

    fname_output = "turbine_inputs/nrel5mw_mod_out.yaml"
    fname_schema = "turbine_inputs/IEAontology_schema.yaml"

    # Initialize blade design
    refBlade = ReferenceBlade()
    refBlade.verbose = True
    refBlade.NINPUT  = 8
    refBlade.NPTS    = 50
    refBlade.spar_var = ['Spar_Cap_SS', 'Spar_Cap_PS']
    refBlade.te_var   = 'TE_reinforcement'
    # refBlade.le_var   = 'le_reinf'
    refBlade.fname_schema = fname_schema
    
    blade = refBlade.initialize(fname_input)
    rotor = Problem()
    rotor.model = RotorStructure(RefBlade=blade, topLevelFlag=True)
    
    #rotor.setup(check=False)
    rotor.setup()
    rotor = Init_RotorStructure_wRefBlade(rotor, blade)

    # === run and outputs ===
    rotor.run_driver()
    #rotor.check_partials(compact_print=True, method='fd', form='central', includes='load*') #step=1e-6

    # Write Ontology File Out
    blade_out = copy.deepcopy(blade)
    blade_out['ctrl_pts']['bladeLength'] = rotor['bladeLength']
    blade_out['ctrl_pts']['r_in']        = rotor['r_in']
    blade_out['ctrl_pts']['chord_in']    = rotor['chord_in']
    blade_out['ctrl_pts']['theta_in']    = rotor['theta_in']
    blade_out['ctrl_pts']['precurve_in'] = rotor['precurve_in']
    blade_out['ctrl_pts']['presweep_in'] = rotor['presweep_in']
    blade_out['ctrl_pts']['sparT_in']    = rotor['sparT_in']
    blade_out['ctrl_pts']['teT_in']      = rotor['teT_in']
    # if 'le_var' in blade['precomp']['le_var']:
    #     blade_out['ctrl_pts']['leT_in']  = rotor['leT_in']
    # Update
    refBlade.verbose  = False
    blade_out = refBlade.update(blade_out)
    # refBlade.write_ontology(fname_output, blade_out, refBlade.wt_ref)


    print('mass_one_blade =', rotor['mass_one_blade'])
    print('mass_all_blades =', rotor['mass_all_blades'])
    print('I_all_blades =', rotor['I_all_blades'])
    print('freq pbeam =', rotor['freq_pbeam'])
    print('freq curvefem =', rotor['freq_curvefem'])
    print('tip_deflection =', rotor['tip_deflection'])
    print('root_bending_moment =', rotor['root_bending_moment'])

    print('CurveFEM calculated mode shape curve fit coef. for ElastoDyn =')
    print(rotor['modes_coef_curvefem'])

    # # Write precomp files out
    # from rotorse.precomp import PreCompWriter
    # dir_out     = 'temp'
    # materials   = rotor['materials']
    # upper       = rotor['upperCS']
    # lower       = rotor['lowerCS']
    # webs        = rotor['websCS']
    # profile     = rotor['profile']
    # chord       = rotor['chord']
    # twist       = rotor['theta']
    # p_le        = rotor['le_location']
    # precomp_out = PreCompWriter(dir_out, materials, upper, lower, webs, profile, chord, twist, p_le)
    # precomp_out.execute()
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(rotor['r'], rotor['strainU_spar'], label='suction')
    plt.plot(rotor['r'], rotor['strainL_spar'], label='pressure')
    plt.plot(rotor['r'], rotor['eps_crit_spar'], label='critical')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r')
    plt.ylabel('strain')
    plt.legend()
    # plt.save('/Users/sning/Desktop/strain_spar.pdf')
    # plt.save('/Users/sning/Desktop/strain_spar.png')

    plt.figure()
    plt.plot(rotor['r'], rotor['strainU_te'], label='suction')
    plt.plot(rotor['r'], rotor['strainL_te'], label='pressure')
    plt.plot(rotor['r'], rotor['eps_crit_te'], label='critical')
    plt.ylim([-5e-3, 5e-3])
    plt.xlabel('r')
    plt.ylabel('strain')
    plt.legend()
    # plt.save('/Users/sning/Desktop/strain_te.pdf')
    # plt.save('/Users/sning/Desktop/strain_te.png')

    plt.show()
