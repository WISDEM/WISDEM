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
        self.inputs["pitch_axis"] = np.zeros(self.inputs["pitch_axis"].shape)
        
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
        
if __name__ == "__main__":
    unittest.main()
