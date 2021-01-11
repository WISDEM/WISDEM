import unittest

import numpy as np
import numpy.testing as npt
import wisdem.drivetrainse.generator_models as gm


class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.inputs = {}
        self.outputs = {}
        self.discrete_inputs = {}
        self.discrete_outputs = {}

        self.inputs["machine_rating"] = 5e6
        self.inputs["rated_torque"] = 4.143289e6

        self.inputs["rho_Fe"] = 7700.0
        self.inputs["rho_Fes"] = 7850.0
        self.inputs["rho_Copper"] = 8900.0
        self.inputs["rho_PM"] = 7450.0

        self.inputs["B_r"] = 1.2
        self.inputs["E"] = 2e11
        self.inputs["G"] = 79.3e9
        self.inputs["P_Fe0e"] = 1.0
        self.inputs["P_Fe0h"] = 4.0
        self.inputs["S_N"] = -0.002
        self.inputs["alpha_p"] = 0.5 * np.pi * 0.7
        self.inputs["b_r_tau_r"] = 0.45
        self.inputs["b_ro"] = 0.004
        self.inputs["b_s_tau_s"] = 0.45
        self.inputs["b_so"] = 0.004
        self.inputs["cofi"] = 0.85
        self.inputs["freq"] = 60
        self.inputs["h_i"] = 0.001
        self.inputs["h_sy0"] = 0.0
        self.inputs["h_w"] = 0.005
        self.inputs["k_fes"] = 0.9
        self.inputs["k_fillr"] = 0.7
        self.inputs["k_fills"] = 0.65
        self.inputs["k_s"] = 0.2
        self.discrete_inputs["m"] = 3
        self.inputs["mu_0"] = np.pi * 4e-7
        self.inputs["mu_r"] = 1.06
        self.inputs["p"] = 3.0
        self.inputs["phi"] = np.deg2rad(90)
        self.discrete_inputs["q1"] = 6
        self.discrete_inputs["q2"] = 4
        self.inputs["ratio_mw2pp"] = 0.7
        self.inputs["resist_Cu"] = 1.8e-8 * 1.4
        self.inputs["sigma"] = 40e3
        self.inputs["v"] = 0.3
        self.inputs["y_tau_p"] = 1.0
        self.inputs["y_tau_pr"] = 10.0 / 12

    def testPMSG_Outer(self):

        myobj = gm.PMSG_Outer(n_pc=20)

        self.inputs["machine_rating"] = 10.321e6
        self.inputs["rated_torque"] = 10.25e6  # rev 1 9.94718e6
        self.inputs["P_mech"] = 10.71947704e6  # rev 1 9.94718e6
        self.inputs["shaft_rpm"] = np.linspace(2, 10, 20)
        self.inputs["rad_ag"] = 4.0  # rev 1  4.92
        self.inputs["len_s"] = 1.7  # rev 2.3
        self.inputs["h_s"] = 0.7  # rev 1 0.3
        self.inputs["p"] = 70  # 100.0    # rev 1 160
        self.inputs["h_m"] = 0.005  # rev 1 0.034
        self.inputs["h_ys"] = 0.04  # rev 1 0.045
        self.inputs["h_yr"] = 0.06  # rev 1 0.045
        self.inputs["b"] = 2.0
        self.inputs["c"] = 5.0
        self.inputs["B_tmax"] = 1.9
        self.inputs["E_p"] = 3300 / np.sqrt(3)
        self.inputs["D_nose"] = 2 * 1.1  # Nose outer radius
        self.inputs["D_shaft"] = 2 * 1.34  # Shaft outer radius =(2+0.25*2+0.3*2)*0.5
        self.inputs["t_r"] = 0.05  # Rotor disc thickness
        self.inputs["h_sr"] = 0.04  # Rotor cylinder thickness
        self.inputs["t_s"] = 0.053  # Stator disc thickness
        self.inputs["h_ss"] = 0.04  # Stator cylinder thickness
        self.inputs["y_sh"] = 0.0005 * 0  # Shaft deflection
        self.inputs["theta_sh"] = 0.00026 * 0  # Slope at shaft end
        self.inputs["y_bd"] = 0.0005 * 0  # deflection at bedplate
        self.inputs["theta_bd"] = 0.00026 * 0  # Slope at bedplate end
        self.inputs["u_allow_pcent"] = 8.5  # % radial deflection
        self.inputs["y_allow_pcent"] = 1.0  # % axial deflection
        self.inputs["z_allow_deg"] = 0.05  # torsional twist
        self.inputs["sigma"] = 60.0e3  # Shear stress
        self.inputs["B_r"] = 1.279
        self.inputs["ratio_mw2pp"] = 0.8
        self.inputs["h_0"] = 5e-3
        self.inputs["h_w"] = 4e-3
        self.inputs["k_fes"] = 0.8

        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs["R_out"], 4.105)
        self.assertAlmostEqual(self.outputs["K_rad"], 0.2125)
        self.assertAlmostEqual(self.outputs["Slot_aspect_ratio"], 6.24779759)
        self.assertAlmostEqual(self.outputs["tau_p"], 0.17951958)
        self.assertAlmostEqual(self.outputs["tau_s"], 0.14930045)
        self.assertAlmostEqual(self.outputs["b_s"], 0.11203948)
        self.assertAlmostEqual(self.outputs["b_t"], 0.03652435)
        self.assertAlmostEqual(self.outputs["h_t"], 0.709)
        self.assertAlmostEqual(self.outputs["b_m"], 0.14361566)
        self.assertAlmostEqual(self.outputs["B_g"], 0.56284833)
        self.assertAlmostEqual(self.outputs["B_symax"], 0.66753916)
        self.assertAlmostEqual(self.outputs["B_rymax"], 0.44502611)
        self.assertAlmostEqual(self.outputs["B_pm1"], 0.46480944)
        self.assertAlmostEqual(self.outputs["B_smax"], 0.23570673147616897)
        self.assertAlmostEqual(self.outputs["f"][-1], 11.66666667)
        self.assertAlmostEqual(self.outputs["I_s"][-1], 1721.4993685170261)
        self.assertAlmostEqual(self.outputs["R_s"][-1], 0.0862357)
        self.assertAlmostEqual(self.outputs["L_s"], 0.01000474)
        self.assertAlmostEqual(self.outputs["S"], 168.0)
        self.assertAlmostEqual(self.outputs["N_s"], 361.0)
        self.assertAlmostEqual(self.outputs["A_Cuscalc"], 389.46615218)
        self.assertAlmostEqual(self.outputs["J_actual"][-1], 3.125519048273891)
        self.assertAlmostEqual(self.outputs["A_1"][-1], 148362.95007673657)
        self.assertAlmostEqual(len(self.outputs["eandm_efficiency"]), 20)
        self.assertAlmostEqual(self.outputs["eandm_efficiency"][-1], 0.9543687904168252)
        self.assertAlmostEqual(self.outputs["Iron"], 89073.14254723)
        self.assertAlmostEqual(self.outputs["mass_PM"], 1274.81620149)
        self.assertAlmostEqual(self.outputs["Copper"], 13859.17179278)
        self.assertAlmostEqual(self.outputs["twist_r"], 0.00032341)
        self.assertAlmostEqual(self.outputs["twist_s"], 5.8057978e-05)
        self.assertAlmostEqual(self.outputs["Structural_mass"], 62323.08483264)
        self.assertAlmostEqual(self.outputs["generator_mass"], 166530.21537414)

    def testPMSG_Arms(self):

        myobj = gm.PMSG_Arms(n_pc=20)

        self.inputs["shaft_rpm"] = np.linspace(5, 12.1, 20)
        self.inputs["sigma"] = 40e3
        self.inputs["rad_ag"] = 3.26
        self.inputs["len_s"] = 1.60
        self.inputs["h_s"] = 0.070
        self.inputs["tau_p"] = 0.080
        self.inputs["h_m"] = 0.009
        self.inputs["h_ys"] = 0.075
        self.inputs["h_yr"] = 0.075
        self.inputs["n_s"] = 5.0
        self.inputs["b_st"] = 0.480
        self.inputs["n_r"] = 5.0
        self.inputs["b_arm"] = 0.530
        self.inputs["d_r"] = 0.700
        self.inputs["d_s"] = 0.350
        self.inputs["t_d"] = 0.105
        self.inputs["t_wr"] = 0.06
        self.inputs["t_ws"] = 0.06
        self.inputs["D_shaft"] = 2 * 0.43
        self.discrete_inputs["q1"] = 1

        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs["B_symax"], 0.31869052515138424)
        self.assertAlmostEqual(self.outputs["B_tmax"], 1.391188727217009)
        self.assertAlmostEqual(self.outputs["B_smax"], 0.0995499652108797)
        self.assertAlmostEqual(self.outputs["B_rymax"], 0.28682147263624586)
        self.assertAlmostEqual(self.outputs["B_pm1"], 0.674462389329781)
        self.assertAlmostEqual(self.outputs["B_g"], 0.7651537999693548)
        self.assertAlmostEqual(self.outputs["N_s"], 256.0)
        self.assertAlmostEqual(self.outputs["b_s"], 0.012001865684417254)
        self.assertAlmostEqual(self.outputs["b_t"], 0.01466894694762109)
        self.assertAlmostEqual(self.outputs["A_Cuscalc"], 253.53941258331452)
        self.assertAlmostEqual(self.outputs["b_m"], 0.055999999999999994)
        self.assertAlmostEqual(self.outputs["E_p"][-1], 2013.9466358469522)
        self.assertAlmostEqual(self.outputs["f"][-1], 25.813333333333333)
        self.assertAlmostEqual(self.outputs["I_s"][-1], 846.2065921065657)
        self.assertAlmostEqual(self.outputs["R_s"], 0.09770712863768102)
        self.assertAlmostEqual(self.outputs["L_s"], 0.011401106062847888)
        self.assertAlmostEqual(self.outputs["A_1"][-1], 63455.62872651724)
        self.assertAlmostEqual(self.outputs["J_s"][-1], 3.337574160500578)
        self.assertAlmostEqual(self.outputs["Losses"][-1], 351360.55854497873)
        self.assertAlmostEqual(self.outputs["K_rad"], 0.245398773006135)
        self.assertAlmostEqual(len(self.outputs["eandm_efficiency"]), 20)
        self.assertAlmostEqual(self.outputs["eandm_efficiency"][-1], 0.9343418267745142)
        self.assertAlmostEqual(self.outputs["S"], 768.0)
        self.assertAlmostEqual(self.outputs["Slot_aspect_ratio"], 5.832426544390113)
        self.assertAlmostEqual(self.outputs["Copper"], 6654.6915566955695)
        self.assertAlmostEqual(self.outputs["Iron"], 52673.28069829149)
        self.assertAlmostEqual(self.outputs["TC1"], 16.4856231252069)
        self.assertAlmostEqual(self.outputs["R_out"], 3.3680954773869343)
        self.assertAlmostEqual(self.outputs["Structural_mass"], 33718.05538799999)
        self.assertAlmostEqual(self.outputs["generator_mass"], 94729.99806753898)
        self.assertAlmostEqual(self.outputs["mass_PM"], 1683.970424551947)

    def testPMSG_disc(self):

        myobj = gm.PMSG_Disc(n_pc=20)

        self.inputs["shaft_rpm"] = np.linspace(5, 12.1, 20)
        self.inputs["sigma"] = 40e3
        self.inputs["rad_ag"] = 3.49
        self.inputs["len_s"] = 1.5
        self.inputs["h_s"] = 0.06
        self.inputs["tau_p"] = 0.07
        self.inputs["h_m"] = 0.0105
        self.inputs["h_ys"] = 0.085
        self.inputs["h_yr"] = 0.055
        self.inputs["n_s"] = 5.0
        self.inputs["b_st"] = 0.460
        self.inputs["t_d"] = 0.105
        self.inputs["d_s"] = 0.350
        self.inputs["t_ws"] = 0.150
        self.inputs["D_shaft"] = 2 * 0.43
        self.discrete_inputs["q1"] = 1

        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs["B_symax"], 0.2552951)
        self.assertAlmostEqual(self.outputs["B_smax"], 0.0687816)
        self.assertAlmostEqual(self.outputs["B_tmax"], 1.4426429)
        self.assertAlmostEqual(self.outputs["B_rymax"], 0.3550922)
        self.assertAlmostEqual(self.outputs["B_pm1"], 0.6994079)
        self.assertAlmostEqual(self.outputs["B_g"], 0.7934536)
        self.assertAlmostEqual(self.outputs["N_s"], 314.0000000)
        self.assertAlmostEqual(self.outputs["b_s"], 0.0104753)
        self.assertAlmostEqual(self.outputs["b_t"], 0.0128032)
        self.assertAlmostEqual(self.outputs["A_Cuscalc"], 187.2461758)
        self.assertAlmostEqual(self.outputs["b_m"], 0.0490000)
        self.assertAlmostEqual(self.outputs["E_p"][-1], 2555.3443951)
        self.assertAlmostEqual(self.outputs["f"][-1], 31.6616667)
        self.assertAlmostEqual(self.outputs["I_s"][-1], 657.7809841)
        self.assertAlmostEqual(self.outputs["R_s"], 0.1504414)
        self.assertAlmostEqual(self.outputs["L_s"], 0.0120779)
        self.assertAlmostEqual(self.outputs["A_1"][-1], 56514.1132206)
        self.assertAlmostEqual(self.outputs["J_s"][-1], 3.5129208)
        self.assertAlmostEqual(self.outputs["Losses"][-1], 338164.5549178)
        self.assertAlmostEqual(self.outputs["K_rad"], 0.2148997)
        self.assertAlmostEqual(len(self.outputs["eandm_efficiency"]), 20)
        self.assertAlmostEqual(self.outputs["eandm_efficiency"][-1], 0.936651530)
        self.assertAlmostEqual(self.outputs["S"], 942.0000000)
        self.assertAlmostEqual(self.outputs["Slot_aspect_ratio"], 5.7277538)
        self.assertAlmostEqual(self.outputs["Copper"], 5588.6107806)
        self.assertAlmostEqual(self.outputs["Iron"], 48400.5614350)
        self.assertAlmostEqual(self.outputs["TC1"], 16.4856231)
        self.assertAlmostEqual(self.outputs["R_out"], 3.6073317)
        self.assertAlmostEqual(self.outputs["Structural_mass"], 67726.0753421)
        self.assertAlmostEqual(self.outputs["generator_mass"], 123674.5978407)
        self.assertAlmostEqual(self.outputs["mass_PM"], 1959.3502831)

    def testEESG(self):

        myobj = gm.EESG(n_pc=20)

        self.inputs["shaft_rpm"] = np.linspace(5, 12.1, 20)
        self.inputs["sigma"] = 48.373e3
        self.inputs["rad_ag"] = 3.2
        self.inputs["len_s"] = 1.4
        self.inputs["h_s"] = 0.060
        self.inputs["tau_p"] = 0.170
        self.inputs["I_f"] = 69
        self.inputs["N_f"] = 100
        self.inputs["h_ys"] = 0.130
        self.inputs["h_yr"] = 0.120
        self.inputs["n_s"] = 5
        self.inputs["b_st"] = 0.470
        self.inputs["n_r"] = 5
        self.inputs["b_arm"] = 0.480
        self.inputs["d_r"] = 0.510
        self.inputs["d_s"] = 0.400
        self.inputs["t_wr"] = 0.140
        self.inputs["t_ws"] = 0.070
        self.inputs["D_shaft"] = 2 * 0.43
        self.discrete_inputs["q1"] = 2

        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs["B_symax"], 0.45940641371625035)
        self.assertAlmostEqual(self.outputs["B_tmax"], 1.8877594585933197)
        self.assertAlmostEqual(self.outputs["B_rymax"], 0.6181179999135469)
        self.assertAlmostEqual(self.outputs["B_g"], 1.1036753874387688)
        self.assertAlmostEqual(self.outputs["N_s"], 236.0)
        self.assertAlmostEqual(self.outputs["b_s"], 0.012779359946805938)
        self.assertAlmostEqual(self.outputs["b_t"], 0.015619217712762814)
        self.assertAlmostEqual(self.outputs["A_Cuscalc"], 228.43105904915618)
        self.assertAlmostEqual(self.outputs["f"][-1], 11.898333333333333)
        self.assertAlmostEqual(self.outputs["I_s"][-1], 858.2107484477335)
        self.assertAlmostEqual(self.outputs["R_s"], 0.0906017600502669)
        self.assertAlmostEqual(self.outputs["A_1"][-1], 60440.40365229799)
        self.assertAlmostEqual(self.outputs["J_s"][-1], 3.75697924800609)
        self.assertAlmostEqual(self.outputs["Losses"][-1], 444556.7203870894)
        self.assertAlmostEqual(self.outputs["K_rad"], 0.21874999999999997)
        self.assertAlmostEqual(len(self.outputs["eandm_efficiency"]), 20)
        self.assertAlmostEqual(self.outputs["eandm_efficiency"][-1], 0.918348408655116)
        self.assertAlmostEqual(self.outputs["S"], 708.0)
        self.assertAlmostEqual(self.outputs["Slot_aspect_ratio"], 4.695070821210912)
        self.assertAlmostEqual(self.outputs["Copper"], 16059.414075335235)
        self.assertAlmostEqual(self.outputs["Iron"], 72360.88485924648)
        self.assertAlmostEqual(self.outputs["TC1"], 13.632086598066605)
        self.assertAlmostEqual(self.outputs["R_out"], 3.219748743718593)
        self.assertAlmostEqual(self.outputs["Structural_mass"], 42403.44234)
        self.assertAlmostEqual(self.outputs["generator_mass"], 130823.74127458173)

    def testSCIG(self):

        myobj = gm.SCIG(n_pc=20)

        self.inputs["shaft_rpm"] = 100 * np.linspace(2, 12, 20)
        self.inputs["cofi"] = 0.9
        self.inputs["y_tau_p"] = 12.0 / 15.0
        self.inputs["sigma"] = 21.5e3
        self.inputs["rad_ag"] = 0.55
        self.inputs["len_s"] = 1.30
        self.inputs["h_s"] = 0.090
        self.inputs["I_0"] = 140
        self.inputs["B_symax"] = 1.4
        self.discrete_inputs["q1"] = 6
        self.inputs["h_0"] = 0.05

        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs["B_tsmax"], 1.684969767820678)
        self.assertAlmostEqual(self.outputs["B_trmax"], 1.8541950805798264)
        self.assertAlmostEqual(self.outputs["B_rymax"], 1.4)
        self.assertAlmostEqual(self.outputs["B_g"], 0.9267333723013729)
        self.assertAlmostEqual(self.outputs["B_g1"], 0.8617526608763629)
        self.assertAlmostEqual(self.outputs["h_ys"], 0.12135794161089408)
        self.assertAlmostEqual(self.outputs["b_s"], 0.014398966328953218)
        self.assertAlmostEqual(self.outputs["b_t"], 0.017598736624276155)
        self.assertAlmostEqual(self.outputs["D_ratio"], 1.393378075656171)
        self.assertAlmostEqual(self.outputs["D_ratio_UL"], 1.4)
        self.assertAlmostEqual(self.outputs["D_ratio_LL"], 1.37)
        self.assertAlmostEqual(self.outputs["A_Cuscalc"], 244.78242759220464)
        self.assertAlmostEqual(self.outputs["Slot_aspect_ratio1"], 6.250448674154435)
        self.assertAlmostEqual(self.outputs["h_yr"], 0.12135794161089408)
        self.assertAlmostEqual(self.outputs["tau_p"], 0.5759586531581288)
        self.assertAlmostEqual(self.outputs["b_r"], 0.021513941784534422)
        self.assertAlmostEqual(self.outputs["b_trmin"], 0.023894990015160978)
        self.assertAlmostEqual(self.outputs["b_tr"], 0.026294817736653178)
        self.assertAlmostEqual(self.outputs["rad_r"], 0.547848028863988)
        self.assertAlmostEqual(self.outputs["A_bar"], 0.000968127380304049)
        self.assertAlmostEqual(self.outputs["Slot_aspect_ratio2"], 2.3240743375043973)
        self.assertAlmostEqual(self.outputs["E_p"][-1], 3584.6363848734995)
        self.assertAlmostEqual(self.outputs["I_s"][-1], 383.73299436233305)
        self.assertAlmostEqual(self.outputs["A_1"][-1], 23985.033858413553)
        self.assertAlmostEqual(self.outputs["J_s"][-1], 1.5676492717917363)
        self.assertAlmostEqual(self.outputs["J_r"][-1], 0.3690452555761286)
        self.assertAlmostEqual(self.outputs["R_s"], 0.00820627176284264)
        self.assertAlmostEqual(self.outputs["R_R"], 0.026060741300117953)
        self.assertAlmostEqual(self.outputs["L_s"], 0.001078616172170861)
        self.assertAlmostEqual(self.outputs["L_sm"], 0.06791816664758084)
        self.assertAlmostEqual(self.outputs["generator_mass"], 40729.31598698678)
        self.assertAlmostEqual(self.outputs["K_rad"], 1.1818181818181817)
        self.assertAlmostEqual(self.outputs["Losses"][-1], 75068.46569051298)
        self.assertAlmostEqual(len(self.outputs["eandm_efficiency"]), 20)
        self.assertAlmostEqual(self.outputs["eandm_efficiency"][-1], 0.9849562794756211)
        self.assertAlmostEqual(self.outputs["Copper"], 1360.253987369221)
        self.assertAlmostEqual(self.outputs["Iron"], 11520.904698026085)
        self.assertAlmostEqual(self.outputs["R_out"], 0.7663579416108941)
        self.assertAlmostEqual(self.outputs["Structural_mass"], 27848.15730159148)
        self.assertAlmostEqual(self.outputs["TC1"], 0.29453832454167966)

    def testDFIG(self):

        myobj = gm.DFIG(n_pc=20)

        self.inputs["shaft_rpm"] = 100 * np.linspace(2, 12, 20)
        self.inputs["cofi"] = 0.9
        self.inputs["y_tau_p"] = 12.0 / 15.0
        self.inputs["sigma"] = 21.5e3
        self.inputs["rad_ag"] = 0.61
        self.inputs["len_s"] = 0.49
        self.inputs["h_s"] = 0.08
        self.inputs["I_0"] = 40.0
        self.inputs["B_symax"] = 1.3
        self.inputs["S_Nmax"] = -0.2
        self.inputs["k_fillr"] = 0.55
        self.inputs["h_0"] = 0.01
        self.inputs["k_fillr"] = 0.55
        self.inputs["k_fills"] = 0.65
        self.discrete_inputs["q1"] = 5
        self.discrete_inputs["q2"] = 4

        myobj.compute(self.inputs, self.outputs, self.discrete_inputs, self.discrete_outputs)
        self.assertAlmostEqual(self.outputs["B_g"], 1.1073196940043413)
        self.assertAlmostEqual(self.outputs["B_g1"], 1.0207571181956336)
        self.assertAlmostEqual(self.outputs["B_rymax"], 1.3)
        self.assertAlmostEqual(self.outputs["B_tsmax"], 2.0133085345533477)
        self.assertAlmostEqual(self.outputs["B_trmax"], 2.046984459493319)
        self.assertAlmostEqual(self.outputs["h_ys"], 0.1731961572673457)
        self.assertAlmostEqual(self.outputs["b_s"], 0.019163715186897738)
        self.assertAlmostEqual(self.outputs["b_t"], 0.023422318561763904)
        self.assertAlmostEqual(self.outputs["D_ratio"], 1.4232723889628618)
        self.assertAlmostEqual(self.outputs["A_Cuscalc"], 287.4557278034661)
        self.assertAlmostEqual(self.outputs["Slot_aspect_ratio1"], 4.1745558843775825)
        self.assertAlmostEqual(self.outputs["h_yr"], 0.1731961572673457)
        self.assertAlmostEqual(self.outputs["tau_p"], 0.6387905062299246)
        self.assertAlmostEqual(self.outputs["N_r"], 148.0)
        self.assertAlmostEqual(self.outputs["b_r"], 0.023870136274726762)
        self.assertAlmostEqual(self.outputs["b_trmin"], 0.02869464545814538)
        self.assertAlmostEqual(self.outputs["b_tr"], 0.02917461100244382)
        self.assertAlmostEqual(self.outputs["A_Curcalc"], 5.322395250445834)
        self.assertAlmostEqual(self.outputs["Slot_aspect_ratio2"], 0.41893351109971694)
        self.assertAlmostEqual(self.outputs["E_p"][-1], 1480.0153123457287)
        self.assertAlmostEqual(self.outputs["I_s"][-1], 959.242520719389)
        self.assertAlmostEqual(self.outputs["A_1"][-1], 45049.62948091099)
        self.assertAlmostEqual(self.outputs["J_s"][-1], 3.3370095911785915)
        self.assertAlmostEqual(self.outputs["J_r"][-1], 176.31699359563828)
        self.assertAlmostEqual(self.outputs["R_s"], 0.0037830355111420555)
        self.assertAlmostEqual(self.outputs["R_R"], 0.03168834286109277)
        self.assertAlmostEqual(self.outputs["L_r"], 0.00016404053494963115)
        self.assertAlmostEqual(self.outputs["L_s"], 0.0001855520925628366)
        self.assertAlmostEqual(self.outputs["L_sm"], 0.019568172828126774)
        self.assertAlmostEqual(self.outputs["generator_mass"], 19508.405570203206)
        self.assertAlmostEqual(self.outputs["K_rad"], 0.4016393442622951)
        self.assertAlmostEqual(self.outputs["Losses"][-1], 143901.77088462992)
        self.assertAlmostEqual(len(self.outputs["eandm_efficiency"]), 20)
        self.assertAlmostEqual(self.outputs["eandm_efficiency"][-1], 0.9654635749876888)
        self.assertAlmostEqual(self.outputs["Copper"], 354.97950026786026)
        self.assertAlmostEqual(self.outputs["Iron"], 6077.944572132577)
        self.assertAlmostEqual(self.outputs["Structural_mass"], 13075.48149780277)
        self.assertAlmostEqual(self.outputs["R_out"], 0.8681961572673457)
        self.assertAlmostEqual(self.outputs["TC1"], 0.24790308982258036)
        self.assertAlmostEqual(self.outputs["Current_ratio"][-1], 0.17760183748148742)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestGenerators))
    return suite


if __name__ == "__main__":
    result = unittest.TextTestRunner().run(suite())

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
