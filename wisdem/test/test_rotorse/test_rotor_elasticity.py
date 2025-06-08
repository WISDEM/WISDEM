import unittest
import os

import numpy as np
import numpy.testing as npt
import openmdao.api as om

import wisdem.rotorse.rotor_elasticity as rel
from wisdem.commonse.cross_sections import Tube
from wisdem.rotorse.rotor_elasticity import generate_KI, KI_to_Elastic
import wisdem.precomp.properties as prop
import wisdem.precomp.precomp_to_beamdyn as pc2bd

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

class TestRE(unittest.TestCase):
    def setUp(self):
        self.discrete_inputs = {}
        self.discrete_outputs = {}
        self.inputs = {}
        self.outputs = {}
        self.options = {}
        self.options = {}
        self.options["WISDEM"] = {}
        self.options["WISDEM"]["RotorSE"] = {}
        self.options["WISDEM"]["RotorSE"]["n_xy"] = npts = 101
        self.options["WISDEM"]["RotorSE"]["n_span"] = nspan = 31
        self.options["WISDEM"]["RotorSE"]["n_webs"] = nweb = 0
        self.options["WISDEM"]["RotorSE"]["n_layers"] = nlay = 1
        self.options["WISDEM"]["RotorSE"]["te_ss"] = "none"
        self.options["WISDEM"]["RotorSE"]["te_ps"] = "none"
        self.options["WISDEM"]["RotorSE"]["spar_cap_ss"] = "none"
        self.options["WISDEM"]["RotorSE"]["spar_cap_ps"] = "none"
        self.options["WISDEM"]["RotorSE"]["layer_name"] = ["mylayer"]
        self.options["WISDEM"]["RotorSE"]["layer_mat"] = ["mymat"]
        self.options["materials"] = {}
        self.options["materials"]["n_mat"] = nmat = 1
        self.options["General"] = {}
        self.options["General"]["verbosity"] = False

        # Use a tubular blade
        self.inputs["r"] = np.linspace(0.0, 10.0, nspan)
        self.inputs["theta"] = np.zeros(nspan)
        self.inputs["chord"] = 2 * np.ones(nspan)
        self.inputs["section_offset_y"] = 1. * np.ones(nspan) # 0=LE, 1=TE
        self.inputs["precurve"] = np.zeros(nspan)
        self.inputs["presweep"] = np.zeros(nspan)
        self.inputs["coord_xy_interp"] = np.zeros( (nspan, npts, 2) )
        angle = np.linspace(0, 2*np.pi, npts)
        for k in range(nspan):
            self.inputs["coord_xy_interp"][k,:,0] = 0.5 * np.cos(angle) + 0.5
            self.inputs["coord_xy_interp"][k,:,1] = 0.5 * np.sin(angle)
        self.inputs["coord_xy_interp"][:,0,1] = 0.0
        self.inputs["coord_xy_interp"][:,-1,1] = 0.0
        self.inputs["uptilt"] = np.zeros(1)
        self.discrete_inputs["n_blades"] = 3
        self.inputs["web_start_nd"] = self.inputs["web_end_nd"] = np.zeros((nweb, nspan))
        self.discrete_inputs["build_layer"] = np.zeros(nlay)
        self.inputs["layer_thickness"] = 0.01 * np.ones((nlay, nspan))
        self.inputs["layer_start_nd"] = np.zeros((nlay, nspan))
        self.inputs["layer_end_nd"] = np.ones((nlay, nspan))
        self.inputs["fiber_orientation"] = 90 * np.ones((nlay, nspan))

        self.discrete_inputs["mat_name"] = ["mymat"]
        self.discrete_inputs["orth"] = np.zeros(nmat)
        self.inputs["E"] = 1e10 * np.ones((nmat,3))
        self.inputs["G"] = 1e8 * np.ones((nmat,3))
        self.inputs["nu"] = 0.33 * np.ones((nmat,3))
        self.inputs["rho"] = 1e3 * np.ones(nmat)
        self.inputs["joint_position"] = self.inputs["joint_mass"] = 0.0

        self.mytube = Tube(self.inputs["chord"][0], self.inputs["layer_thickness"][0,0])

    def run_precomp(self):

        self.myobj = rel.RunPreComp(modeling_options=self.options, opt_options={})
        rotorse_options = self.options["WISDEM"]["RotorSE"]
        self.myobj.n_span = rotorse_options["n_span"]
        self.myobj.n_webs = rotorse_options["n_webs"]
        self.myobj.n_layers = rotorse_options["n_layers"]
        self.myobj.n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
        mat_init_options = self.options["materials"]
        self.myobj.n_mat = mat_init_options["n_mat"]
        self.myobj.verbosity = self.options["General"]["verbosity"]

        self.myobj.te_ss_var = rotorse_options["te_ss"]
        self.myobj.te_ps_var = rotorse_options["te_ps"]
        self.myobj.spar_cap_ss_var = rotorse_options["spar_cap_ss"]
        self.myobj.spar_cap_ps_var = rotorse_options["spar_cap_ps"]
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        for k in self.outputs.keys():
            if type(self.outputs[k]) == type(np.array([])):
                self.inputs[k] = self.outputs[k]
            else:
                self.inputs[k] = np.array([ self.outputs[k] ])
        self.myobj = rel.TotalBladeProperties(modeling_options=self.options, opt_options={})

    def test_no_pitch(self):
        self.run_precomp()
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["z"], self.inputs["r"])
        npt.assert_almost_equal(self.outputs["x_cg"], 0.0)
        npt.assert_almost_equal(self.outputs["x_tc"], 0.0)
        npt.assert_almost_equal(self.outputs["x_sc"], 0.0)
        npt.assert_almost_equal(self.outputs["y_cg"], 0.0)
        npt.assert_almost_equal(self.outputs["y_tc"], 0.0)
        npt.assert_almost_equal(self.outputs["y_sc"], 0.0)
        npt.assert_almost_equal(self.outputs["EIxy"], 0.0, decimal=6)
        npt.assert_almost_equal(self.outputs["A"], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["EIxx"]/self.inputs["E"][0,0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(self.outputs["EIyy"]/self.inputs["E"][0,0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(self.outputs["GJ"]/self.inputs["G"][0,0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(self.outputs["EA"]/self.inputs["E"][0,0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["rhoA"]/self.inputs["rho"][0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["rhoJ"]/self.inputs["rho"][0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(self.outputs["Tw_iner"], 0.0)
        npt.assert_almost_equal(self.outputs["flap_iner"]/self.inputs["rho"][0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(self.outputs["edge_iner"]/self.inputs["rho"][0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(self.outputs["blade_mass"], self.mytube.Area*self.inputs["rho"][0]*self.inputs["r"][-1], decimal=-1)
        npt.assert_almost_equal(self.outputs["mass_all_blades"], 3*self.outputs["blade_mass"], decimal=3)
        idx = np.int_(np.floor(0.5*self.inputs["r"].size))
        npt.assert_almost_equal(self.outputs["blade_span_cg"], self.inputs["r"][idx], decimal=1)
        npt.assert_almost_equal(self.outputs["blade_moment_of_inertia"], self.outputs["blade_mass"]*self.inputs["r"][-1]**2/3.0, decimal=-1)

        '''
        self.outputs["xu_spar"]
        self.outputs["xl_spar"]
        self.outputs["yu_spar"]
        self.outputs["yl_spar"]
        self.outputs["xu_te"]
        self.outputs["xl_te"]
        self.outputs["yu_te"]
        self.outputs["yl_te"]
        self.outputs["I_all_blades"]
        '''
        
    def test_with_pitch(self):
        self.inputs["theta"] = 45 * np.ones(self.inputs["theta"].shape)
        
        self.run_precomp()
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["z"], self.inputs["r"])
        npt.assert_almost_equal(self.outputs["x_cg"], 0.0)
        npt.assert_almost_equal(self.outputs["x_tc"], 0.0)
        npt.assert_almost_equal(self.outputs["x_sc"], 0.0)
        npt.assert_almost_equal(self.outputs["y_cg"], 0.0)
        npt.assert_almost_equal(self.outputs["y_tc"], 0.0)
        npt.assert_almost_equal(self.outputs["y_sc"], 0.0)
        npt.assert_almost_equal(self.outputs["EIxy"], 0.0, decimal=6)
        npt.assert_almost_equal(self.outputs["A"], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["EIxx"]/self.inputs["E"][0,0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(self.outputs["EIyy"]/self.inputs["E"][0,0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(self.outputs["GJ"]/self.inputs["G"][0,0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(self.outputs["EA"]/self.inputs["E"][0,0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["rhoA"]/self.inputs["rho"][0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["rhoJ"]/self.inputs["rho"][0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(self.outputs["Tw_iner"], 45.0)
        npt.assert_almost_equal(self.outputs["flap_iner"]/self.inputs["rho"][0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(self.outputs["edge_iner"]/self.inputs["rho"][0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(self.outputs["blade_mass"], self.mytube.Area*self.inputs["rho"][0]*self.inputs["r"][-1], decimal=-1)
        npt.assert_almost_equal(self.outputs["mass_all_blades"], 3*self.outputs["blade_mass"], decimal=3)
        idx = np.int_(np.floor(0.5*self.inputs["r"].size))
        npt.assert_almost_equal(self.outputs["blade_span_cg"], self.inputs["r"][idx], decimal=1)
        npt.assert_almost_equal(self.outputs["blade_moment_of_inertia"], self.outputs["blade_mass"]*self.inputs["r"][-1]**2/3.0, decimal=-1)
        
        
    def test_with_le_pitch_axis(self):
        self.inputs["section_offset_y"] = np.zeros(self.inputs["section_offset_y"].shape)
        
        self.run_precomp()
        self.myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(self.outputs["z"], self.inputs["r"])
        npt.assert_almost_equal(self.outputs["x_cg"], 0.0)
        npt.assert_almost_equal(self.outputs["x_tc"], 0.0)
        npt.assert_almost_equal(self.outputs["x_sc"], 0.0)
        npt.assert_almost_equal(self.outputs["y_cg"], 1.0)
        npt.assert_almost_equal(self.outputs["y_tc"], 1.0)
        npt.assert_almost_equal(self.outputs["y_sc"], 1.0)
        npt.assert_almost_equal(self.outputs["EIxy"], 0.0, decimal=6)
        npt.assert_almost_equal(self.outputs["A"], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["EIxx"]/self.inputs["E"][0,0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(self.outputs["EIyy"]/self.inputs["E"][0,0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(self.outputs["GJ"]/self.inputs["G"][0,0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(self.outputs["EA"]/self.inputs["E"][0,0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["rhoA"]/self.inputs["rho"][0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(self.outputs["rhoJ"]/self.inputs["rho"][0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(self.outputs["Tw_iner"], 0.0)
        npt.assert_almost_equal(self.outputs["flap_iner"]/self.inputs["rho"][0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(self.outputs["edge_iner"]/self.inputs["rho"][0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(self.outputs["blade_mass"], self.mytube.Area*self.inputs["rho"][0]*self.inputs["r"][-1], decimal=-1)
        npt.assert_almost_equal(self.outputs["mass_all_blades"], 3*self.outputs["blade_mass"], decimal=3)
        idx = np.int_(np.floor(0.5*self.inputs["r"].size))
        npt.assert_almost_equal(self.outputs["blade_span_cg"], self.inputs["r"][idx], decimal=1)
        npt.assert_almost_equal(self.outputs["blade_moment_of_inertia"], self.outputs["blade_mass"]*self.inputs["r"][-1]**2/3.0, decimal=-1)

    def test_KI_to_Elastic(self):

        fnames = ['../test_precomp/section_dump_iea15mw.pkl']
        
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
                        

                        dummy_options = {}
                        dummy_options['WISDEM'] = {}
                        dummy_options['WISDEM']['RotorSE'] = {}
                        dummy_options['WISDEM']['RotorSE']["n_span"] = 1

                        prob = om.Problem()
                        prob.model.add_subsystem("KI_reverse", KI_to_Elastic(modeling_options=dummy_options), promotes=["*"])
                        prob.setup()

                        prob.set_val("K", K_precomp)
                        prob.set_val("I", I_precomp)
                        prob.set_val("theta", theta)

                        prob.run_model()
                        Tw_iner_back = prob.get_val("Tw_iner")
                        x_cg_back = prob.get_val("x_cg")
                        y_cg_back = prob.get_val("y_cg")
                        edge_iner_back = prob.get_val("edge_iner")
                        flap_iner_back = prob.get_val("flap_iner")
                        EIxx_back = prob.get_val("EIxx")
                        EIyy_back = prob.get_val("EIyy")
                        x_tc_back = prob.get_val("x_tc")
                        y_tc_back = prob.get_val("y_tc")
                        x_sc_back = prob.get_val("x_sc")
                        y_sc_back = prob.get_val("y_sc")
                        GJ_back = prob.get_val("GJ")
                        npt.assert_almost_equal(Tw_iner, Tw_iner_back, decimal=6)
                        npt.assert_almost_equal(x_cg, x_cg_back, decimal=6)
                        npt.assert_almost_equal(np.abs(edge_iner-edge_iner_back)/edge_iner, 0.0, decimal=6)
                        npt.assert_almost_equal(np.abs(flap_iner-flap_iner_back)/flap_iner, 0.0, decimal=6)
                        npt.assert_almost_equal(y_cg, y_cg_back, decimal=6)
                        npt.assert_almost_equal(x_tc, x_tc_back, decimal=2)
                        npt.assert_almost_equal(y_tc, y_tc_back, decimal=2)
                        npt.assert_almost_equal(x_sc, x_sc_back, decimal=2)
                        npt.assert_almost_equal(y_sc, y_sc_back, decimal=2)
                        npt.assert_almost_equal(np.abs(EIxx-EIxx_back)/EIxx, 0.0, decimal=6)
                        npt.assert_almost_equal(np.abs(EIyy-EIyy_back)/EIyy, 0.0, decimal=6)
                        npt.assert_almost_equal(np.abs(GJ-GJ_back)/GJ, 0.0, decimal=6)

        
if __name__ == "__main__":
    unittest.main()
