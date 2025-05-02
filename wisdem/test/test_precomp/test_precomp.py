import unittest
import os

import numpy as np
import numpy.testing as npt

import wisdem.precomp.properties as prop
import wisdem.precomp.precomp_to_beamdyn as pc2bd
#from wisdem.precomp._precomp import precomp as _precomp

try:
   import cPickle as pickle
except Exception:
   import pickle

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
# https://stackoverflow.com/questions/20716812/saving-and-loading-multiple-objects-in-pickle-file
def loadall(filename):
    with open(os.path.join(mydir, filename), "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

class TestPreCompProperties(unittest.TestCase):
    def test_tw_rate(self):

        # From NREL 5 MW
        myr = np.array([ 1.5,         3.62068966,  5.74137931,  7.86206897,  9.98275862, 12.10344828,
                         14.22413793, 16.34482759, 18.46551724, 20.5862069 , 22.70689655, 24.82758621,
                         26.94827586, 29.06896552, 31.18965517, 33.31034483, 35.43103448, 37.55172414,
                         39.67241379, 41.79310345, 43.9137931 , 46.03448276, 48.15517241, 50.27586207,
                         52.39655172, 54.51724138, 56.63793103, 58.75862069, 60.87931034, 63.        ])
        th0 = np.zeros(myr.shape)
        #th_prime_fort = _precomp.tw_rate(myr, th0)
        th_prime_fort = th0
        th_prime_py = prop.tw_rate(myr, th0)
        npt.assert_almost_equal(th_prime_fort, th_prime_py)

        th0 = np.linspace(10., -10., myr.size)
        #th_prime_fort = _precomp.tw_rate(myr, th0)
        th_prime_fort = -0.32520325*np.ones(th0.shape)
        th_prime_py = prop.tw_rate(myr, th0)
        npt.assert_almost_equal(th_prime_fort, th_prime_py)

    def test_properties(self):
        fnames = ['section_dump_nrel5mw.pkl', 'section_dump_iea15mw.pkl']
        
        for f in fnames:
            with self.subTest(f=f):
                myitems = loadall(f)
                nsec = myitems.__next__()
                for k in range(nsec):
                    with self.subTest(i=k):
                        chord = myitems.__next__()
                        theta = myitems.__next__()
                        th_prime = myitems.__next__()
                        le_loc = myitems.__next__()
                        xnode = myitems.__next__()
                        ynode = myitems.__next__()
                        E1 = myitems.__next__()
                        E2 = myitems.__next__()
                        G12 = myitems.__next__()
                        nu12 = myitems.__next__()
                        rho = myitems.__next__()
                        locU = myitems.__next__()
                        n_laminaU = myitems.__next__()
                        n_pliesU = myitems.__next__()
                        tU = myitems.__next__()
                        thetaU = myitems.__next__()
                        mat_idxU = myitems.__next__()
                        locL = myitems.__next__()
                        n_laminaL = myitems.__next__()
                        n_pliesL = myitems.__next__()
                        tL = myitems.__next__()
                        thetaL = myitems.__next__()
                        mat_idxL = myitems.__next__()
                        nwebs = myitems.__next__()
                        locW = myitems.__next__()
                        n_laminaW = myitems.__next__()
                        n_pliesW = myitems.__next__()
                        tW = myitems.__next__()
                        thetaW = myitems.__next__()
                        mat_idxW = myitems.__next__()

                        results_fort = myitems.__next__()
                        results_py = prop.properties(chord,
                                                     theta,
                                                     th_prime,
                                                     le_loc,
                                                     xnode,
                                                     ynode,
                                                     E1,
                                                     E2,
                                                     G12,
                                                     nu12,
                                                     rho,
                                                     locU,
                                                     n_laminaU,
                                                     n_pliesU,
                                                     tU,
                                                     thetaU,
                                                     mat_idxU,
                                                     locL,
                                                     n_laminaL,
                                                     n_pliesL,
                                                     tL,
                                                     thetaL,
                                                     mat_idxL,
                                                     nwebs,
                                                     locW,
                                                     n_laminaW,
                                                     n_pliesW,
                                                     tW,
                                                     thetaW,
                                                     mat_idxW,
                                                     )

                        npt.assert_almost_equal(results_fort, results_py, decimal=5)

    def test_match_anba(self):

        # Stiffness and inertia matrices from https://github.com/WISDEM/SONATA/tree/develop/examples/1_IEA15MW
        # for the spanwise section 0.517241 
        K_anba = np.array([
            [1.0715521486176741e+08, 3.5706427831706675e+06, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 8.7833838441895731e+06],
            [3.5706427832182581e+06, 3.0025188389417487e+08, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.2073068428051447e+07],
            [0.0000000000000000e+00, 0.0000000000000000e+00, 1.9613892805330948e+10, 5.4105970862822247e+09, -2.0501694138219514e+08, 0.0000000000000000e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00, 5.4105970862821531e+09, 1.4137313098254759e+10, -2.3555582131167597e+08, 0.0000000000000000e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00, -2.0501694138206977e+08, -2.3555582131167600e+08, 4.3043864825780182e+09, 0.0000000000000000e+00],
            [8.7833838441405557e+06, 1.2073068427973984e+07, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.9466974220949066e+08],
        ])
        

        I_anba = np.array([
            [3.6479516547101201e+02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, -2.5786979371895751e+02],
            [0.0000000000000000e+00, 3.6479516547101201e+02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 8.2707311243991697e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00, 3.6479516547101201e+02, 2.5786979371895751e+02, -8.2707311243991697e+00, 0.0000000000000000e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00, 2.5786979371895751e+02, 6.7555496585520257e+02, -1.5049443526260998e+01, 0.0000000000000000e+00],
            [0.0000000000000000e+00, 0.0000000000000000e+00, -8.2707311243991697e+00, -1.5049443526260998e+01, 5.8584394201929328e+01, 0.0000000000000000e+00],
            [-2.5786979371895751e+02, 8.2707311243991697e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 7.3413936005712992e+02],
        ])
        
        
        
        fnames = ['section_dump_iea15mw.pkl']
        
        for f in fnames:
            with self.subTest(f=f):
                myitems = loadall(f)
                nsec = myitems.__next__()
                for k in range(nsec):
                    with self.subTest(i=k):
                        chord = myitems.__next__()
                        theta = myitems.__next__()
                        th_prime = myitems.__next__()
                        le_loc = myitems.__next__()
                        xnode = myitems.__next__()
                        ynode = myitems.__next__()
                        E1 = myitems.__next__()
                        E2 = myitems.__next__()
                        G12 = myitems.__next__()
                        nu12 = myitems.__next__()
                        rho = myitems.__next__()
                        locU = myitems.__next__()
                        n_laminaU = myitems.__next__()
                        n_pliesU = myitems.__next__()
                        tU = myitems.__next__()
                        thetaU = myitems.__next__()
                        mat_idxU = myitems.__next__()
                        locL = myitems.__next__()
                        n_laminaL = myitems.__next__()
                        n_pliesL = myitems.__next__()
                        tL = myitems.__next__()
                        thetaL = myitems.__next__()
                        mat_idxL = myitems.__next__()
                        nwebs = myitems.__next__()
                        locW = myitems.__next__()
                        n_laminaW = myitems.__next__()
                        n_pliesW = myitems.__next__()
                        tW = myitems.__next__()
                        thetaW = myitems.__next__()
                        mat_idxW = myitems.__next__()

                        results_fort = myitems.__next__()
                        results_py = prop.properties(chord,
                                                     theta,
                                                     th_prime,
                                                     le_loc,
                                                     xnode,
                                                     ynode,
                                                     E1,
                                                     E2,
                                                     G12,
                                                     nu12,
                                                     rho,
                                                     locU,
                                                     n_laminaU,
                                                     n_pliesU,
                                                     tU,
                                                     thetaU,
                                                     mat_idxU,
                                                     locL,
                                                     n_laminaL,
                                                     n_pliesL,
                                                     tL,
                                                     thetaL,
                                                     mat_idxL,
                                                     nwebs,
                                                     locW,
                                                     n_laminaW,
                                                     n_pliesW,
                                                     tW,
                                                     thetaW,
                                                     mat_idxW,
                                                     )

                    if k==15:
                        (eifbar,eilbar,gjbar,eabar,eiflbar,
                        sfbar,slbar,sftbar,sltbar,satbar,
                        z_sc,y_sc,ztc_ref,ytc_ref,
                        mass,area,iflap_eta,ilag_zeta,tw_iner,
                        zcm_ref,ycm_ref) = results_py

                        EIxx  = eilbar
                        EIyy  = eifbar
                        GJ  = gjbar
                        EA  = eabar
                        EIxy  = eiflbar
                        EA_EIxx  = slbar
                        EA_EIyy  = sfbar
                        EIxx_GJ  = sltbar
                        EIyy_GJ  = sftbar
                        EA_GJ  = satbar
                        x_sc  = z_sc
                        y_sc  = y_sc
                        x_tc  = ztc_ref
                        y_tc  = ytc_ref
                        rhoA  = mass
                        A  = area
                        flap_iner  = iflap_eta
                        edge_iner  = ilag_zeta
                        Tw_iner  = tw_iner
                        x_cg  = zcm_ref
                        y_cg  = ycm_ref

                        K_precomp = np.zeros((6,6))
                        I_precomp = np.zeros((6,6))

                        # Build stiffness matrix at the reference axis
                        K_precomp = pc2bd.pc2bd_K(
                            EA,
                            EIxx,
                            EIyy,
                            EIxy,
                            EA_EIxx,
                            EA_EIyy,
                            EIxx_GJ,
                            EIyy_GJ,
                            EA_GJ,
                            GJ,
                            flap_iner+edge_iner,
                            edge_iner,
                            flap_iner,
                            x_sc,
                            y_sc,
                            )
                        # Build inertia matrix at the reference axis
                        I_precomp = pc2bd.pc2bd_I(
                            rhoA,
                            edge_iner,
                            flap_iner,
                            edge_iner+flap_iner,
                            x_cg,
                            y_cg,
                            np.deg2rad(Tw_iner),
                            np.deg2rad(theta),
                            )
                        # Only check diagonal elements of K and I
                        ix = np.arange(6)
                        max_pc_error_K = np.array([25, 25, 11, 16, 11, 5]) # Relative errors at the time of setting up test
                        rel_err_K = np.zeros(len(ix))
                        for i in range(len(ix)):
                            rel_err_K[i] = abs((K_precomp[ix[i],ix[i]]-K_anba[ix[i],ix[i]])/K_anba[ix[i],ix[i]]*100.)
                        npt.assert_array_less(rel_err_K, max_pc_error_K)
                        
                        # Check all elements of I
                        max_pc_error_I = np.array([9, 9, 9, 12, 12, 12]) # Relative errors at the time of setting up test. Updated after switching rotation and translation
                        rel_err_I = np.zeros(len(ix))
                        for i in range(len(ix)):
                            rel_err_I[i] = abs((I_precomp[ix[i],ix[i]]-I_anba[ix[i],ix[i]])/I_anba[ix[i],ix[i]]*100.)
                        npt.assert_array_less(rel_err_I, max_pc_error_I)




if __name__ == "__main__":
    unittest.main()
