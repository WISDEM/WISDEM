import unittest

import numpy as np
import numpy.testing as npt

import wisdem.rotorse.rotor_elasticity as rel
from wisdem.commonse.cross_sections import Tube


class TestRE(unittest.TestCase):
    def setUp(self):
        self.discrete_inputs = {}
        self.discrete_outputs = {}
        self.inputs = {}
        options = {}
        opt_options = {}
        options = {}
        options["WISDEM"] = {}
        options["WISDEM"]["RotorSE"] = {}
        options["WISDEM"]["RotorSE"]["n_xy"] = npts = 101
        options["WISDEM"]["RotorSE"]["n_span"] = nspan = 31
        options["WISDEM"]["RotorSE"]["n_webs"] = nweb = 0
        options["WISDEM"]["RotorSE"]["n_layers"] = nlay = 1
        options["WISDEM"]["RotorSE"]["te_ss"] = "none"
        options["WISDEM"]["RotorSE"]["te_ps"] = "none"
        options["WISDEM"]["RotorSE"]["spar_cap_ss"] = "none"
        options["WISDEM"]["RotorSE"]["spar_cap_ps"] = "none"
        options["WISDEM"]["RotorSE"]["layer_name"] = ["mylayer"]
        options["WISDEM"]["RotorSE"]["layer_mat"] = ["mymat"]
        options["materials"] = {}
        options["materials"]["n_mat"] = nmat = 1
        options["General"] = {}
        options["General"]["verbosity"] = False

        # Use a tubular blade
        self.inputs["r"] = np.linspace(0.0, 10.0, nspan)
        self.inputs["theta"] = np.zeros(nspan)
        self.inputs["chord"] = 2 * np.ones(nspan)
        self.inputs["pitch_axis"] = 0.5 * np.ones(nspan) # 0=LE, 1=TE
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
        self.inputs["layer_web"] = np.zeros(nlay)
        self.discrete_inputs["definition_layer"] = np.ones(nlay)
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

        self.myobj = rel.RunPreComp(modeling_options=options, opt_options=opt_options)
        rotorse_options = options["WISDEM"]["RotorSE"]
        self.myobj.n_span = rotorse_options["n_span"]
        self.myobj.n_webs = rotorse_options["n_webs"]
        self.myobj.n_layers = rotorse_options["n_layers"]
        self.myobj.n_xy = rotorse_options["n_xy"]  # Number of coordinate points to describe the airfoil geometry
        mat_init_options = options["materials"]
        self.myobj.n_mat = mat_init_options["n_mat"]
        self.myobj.verbosity = options["General"]["verbosity"]

        self.myobj.te_ss_var = rotorse_options["te_ss"]
        self.myobj.te_ps_var = rotorse_options["te_ps"]
        self.myobj.spar_cap_ss_var = rotorse_options["spar_cap_ss"]
        self.myobj.spar_cap_ps_var = rotorse_options["spar_cap_ps"]

        self.mytube = Tube(self.inputs["chord"][0], self.inputs["layer_thickness"][0,0])
        
    def test_no_pitch(self):
        outputs = {}
        
        self.myobj.compute(self.inputs, outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(outputs["z"], self.inputs["r"])
        npt.assert_almost_equal(outputs["x_cg"], 0.0)
        npt.assert_almost_equal(outputs["x_ec"], 0.0)
        npt.assert_almost_equal(outputs["x_tc"], 0.0)
        npt.assert_almost_equal(outputs["x_sc"], 0.0)
        npt.assert_almost_equal(outputs["y_cg"], 0.0)
        npt.assert_almost_equal(outputs["y_ec"], 0.0)
        npt.assert_almost_equal(outputs["y_tc"], 0.0)
        npt.assert_almost_equal(outputs["y_sc"], 0.0)
        npt.assert_almost_equal(outputs["EIxy"], 0.0, decimal=6)
        npt.assert_almost_equal(outputs["A"], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["EIxx"]/self.inputs["E"][0,0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(outputs["EIyy"]/self.inputs["E"][0,0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(outputs["GJ"]/self.inputs["G"][0,0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(outputs["EA"]/self.inputs["E"][0,0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["rhoA"]/self.inputs["rho"][0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["rhoJ"]/self.inputs["rho"][0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(outputs["Tw_iner"], 0.0)
        npt.assert_almost_equal(outputs["flap_iner"]/self.inputs["rho"][0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(outputs["edge_iner"]/self.inputs["rho"][0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(outputs["blade_mass"], self.mytube.Area*self.inputs["rho"][0]*self.inputs["r"][-1], decimal=-1)
        npt.assert_almost_equal(outputs["mass_all_blades"], 3*outputs["blade_mass"], decimal=3)
        idx = np.int_(np.floor(0.5*self.inputs["r"].size))
        npt.assert_almost_equal(outputs["blade_span_cg"], self.inputs["r"][idx], decimal=1)
        npt.assert_almost_equal(outputs["blade_moment_of_inertia"], outputs["blade_mass"]*self.inputs["r"][-1]**2/3.0, decimal=-1)

        '''
        outputs["xu_spar"]
        outputs["xl_spar"]
        outputs["yu_spar"]
        outputs["yl_spar"]
        outputs["xu_te"]
        outputs["xl_te"]
        outputs["yu_te"]
        outputs["yl_te"]
        outputs["I_all_blades"]
        '''
        
    def test_with_pitch(self):
        outputs = {}
        self.inputs["theta"] = 45 * np.ones(self.inputs["theta"].shape)
        
        self.myobj.compute(self.inputs, outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(outputs["z"], self.inputs["r"])
        npt.assert_almost_equal(outputs["x_cg"], 0.0)
        npt.assert_almost_equal(outputs["x_ec"], 0.0)
        npt.assert_almost_equal(outputs["x_tc"], 0.0)
        npt.assert_almost_equal(outputs["x_sc"], 0.0)
        npt.assert_almost_equal(outputs["y_cg"], 0.0)
        npt.assert_almost_equal(outputs["y_ec"], 0.0)
        npt.assert_almost_equal(outputs["y_tc"], 0.0)
        npt.assert_almost_equal(outputs["y_sc"], 0.0)
        npt.assert_almost_equal(outputs["EIxy"], 0.0, decimal=6)
        npt.assert_almost_equal(outputs["A"], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["EIxx"]/self.inputs["E"][0,0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(outputs["EIyy"]/self.inputs["E"][0,0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(outputs["GJ"]/self.inputs["G"][0,0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(outputs["EA"]/self.inputs["E"][0,0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["rhoA"]/self.inputs["rho"][0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["rhoJ"]/self.inputs["rho"][0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(outputs["Tw_iner"], 45.0)
        npt.assert_almost_equal(outputs["flap_iner"]/self.inputs["rho"][0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(outputs["edge_iner"]/self.inputs["rho"][0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(outputs["blade_mass"], self.mytube.Area*self.inputs["rho"][0]*self.inputs["r"][-1], decimal=-1)
        npt.assert_almost_equal(outputs["mass_all_blades"], 3*outputs["blade_mass"], decimal=3)
        idx = np.int_(np.floor(0.5*self.inputs["r"].size))
        npt.assert_almost_equal(outputs["blade_span_cg"], self.inputs["r"][idx], decimal=1)
        npt.assert_almost_equal(outputs["blade_moment_of_inertia"], outputs["blade_mass"]*self.inputs["r"][-1]**2/3.0, decimal=-1)
        
        
    def test_with_le_pitch_axis(self):
        outputs = {}
        self.inputs["pitch_axis"] = np.zeros(self.inputs["pitch_axis"].shape)
        
        self.myobj.compute(self.inputs, outputs, self.discrete_inputs, self.discrete_outputs)

        npt.assert_equal(outputs["z"], self.inputs["r"])
        npt.assert_almost_equal(outputs["x_cg"], 0.0)
        npt.assert_almost_equal(outputs["x_ec"], 0.0)
        npt.assert_almost_equal(outputs["x_tc"], 0.0)
        npt.assert_almost_equal(outputs["x_sc"], 0.0)
        npt.assert_almost_equal(outputs["y_cg"], 1.0)
        npt.assert_almost_equal(outputs["y_ec"], 0.0)
        npt.assert_almost_equal(outputs["y_tc"], 1.0)
        npt.assert_almost_equal(outputs["y_sc"], 1.0)
        npt.assert_almost_equal(outputs["EIxy"], 0.0, decimal=6)
        npt.assert_almost_equal(outputs["A"], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["EIxx"]/self.inputs["E"][0,0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(outputs["EIyy"]/self.inputs["E"][0,0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(outputs["GJ"]/self.inputs["G"][0,0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(outputs["EA"]/self.inputs["E"][0,0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["rhoA"]/self.inputs["rho"][0], self.mytube.Area, decimal=3)
        npt.assert_almost_equal(outputs["rhoJ"]/self.inputs["rho"][0], self.mytube.J0, decimal=3)
        npt.assert_almost_equal(outputs["Tw_iner"], 0.0)
        npt.assert_almost_equal(outputs["flap_iner"]/self.inputs["rho"][0], self.mytube.Iyy, decimal=3)
        npt.assert_almost_equal(outputs["edge_iner"]/self.inputs["rho"][0], self.mytube.Ixx, decimal=3)
        npt.assert_almost_equal(outputs["blade_mass"], self.mytube.Area*self.inputs["rho"][0]*self.inputs["r"][-1], decimal=-1)
        npt.assert_almost_equal(outputs["mass_all_blades"], 3*outputs["blade_mass"], decimal=3)
        idx = np.int_(np.floor(0.5*self.inputs["r"].size))
        npt.assert_almost_equal(outputs["blade_span_cg"], self.inputs["r"][idx], decimal=1)
        npt.assert_almost_equal(outputs["blade_moment_of_inertia"], outputs["blade_mass"]*self.inputs["r"][-1]**2/3.0, decimal=-1)
        
if __name__ == "__main__":
    unittest.main()
