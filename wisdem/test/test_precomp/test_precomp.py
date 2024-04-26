import unittest
import os

import numpy as np
import numpy.testing as npt

import wisdem.precomp.properties as prop
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


if __name__ == "__main__":
    unittest.main()
