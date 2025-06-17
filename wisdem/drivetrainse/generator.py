"""generator.py
Created by Latha Sethuraman, Katherine Dykes.
Copyright (c) NREL. All rights reserved.

Electromagnetic design based on conventional magnetic circuit laws
Structural design based on McDonald's thesis """

import numpy as np
import openmdao.api as om

import wisdem.drivetrainse.generator_models as gm

# ----------------------------------------------------------------------------------------------


class Constraints(om.ExplicitComponent):
    """
    Provides a material cost estimate for a PMSG _arms generator. Manufacturing costs are excluded.

    Parameters
    ----------
    u_allow_s : float, [m]

    u_as : float, [m]

    z_allow_s : float, [m]

    z_as : float, [m]

    y_allow_s : float, [m]

    y_as : float, [m]

    b_allow_s : float, [m]

    b_st : float, [m]

    u_allow_r : float, [m]

    u_ar : float, [m]

    y_allow_r : float, [m]

    y_ar : float, [m]

    z_allow_r : float, [m]

    z_ar : float, [m]

    b_allow_r : float, [m]

    b_arm : float, [m]

    TC1 : float, [m**3]

    TC2r : float, [m**3]

    TC2s : float, [m**3]

    B_g : float, [T]

    B_smax : float, [T]

    K_rad : float

    K_rad_LL : float

    K_rad_UL : float

    D_ratio : float

    D_ratio_LL : float

    D_ratio_UL : float


    Returns
    -------
    con_uas : float, [m]

    con_zas : float, [m]

    con_yas : float, [m]

    con_bst : float, [m]

    con_uar : float, [m]

    con_yar : float, [m]

    con_zar : float, [m]

    con_br : float, [m]

    TCr : float, [m**3]

    TCs : float, [m**3]

    con_TC2r : float, [m**3]

    con_TC2s : float, [m**3]

    con_Bsmax : float, [T]

    K_rad_L : float

    K_rad_U : float

    D_ratio_L : float

    D_ratio_U : float


    """

    def setup(self):
        self.add_input("u_allow_s", val=0.0, units="m")
        self.add_input("u_as", val=0.0, units="m")
        self.add_input("z_allow_s", val=0.0, units="m")
        self.add_input("z_as", val=0.0, units="m")
        self.add_input("y_allow_s", val=0.0, units="m")
        self.add_input("y_as", val=0.0, units="m")
        self.add_input("b_allow_s", val=0.0, units="m")
        self.add_input("b_st", val=0.0, units="m")
        self.add_input("u_allow_r", val=0.0, units="m")
        self.add_input("u_ar", val=0.0, units="m")
        self.add_input("y_allow_r", val=0.0, units="m")
        self.add_input("y_ar", val=0.0, units="m")
        self.add_input("z_allow_r", val=0.0, units="m")
        self.add_input("z_ar", val=0.0, units="m")
        self.add_input("b_allow_r", val=0.0, units="m")
        self.add_input("b_arm", val=0.0, units="m")
        self.add_input("TC1", val=0.0, units="m**3")
        self.add_input("TC2r", val=0.0, units="m**3")
        self.add_input("TC2s", val=0.0, units="m**3")
        self.add_input("B_g", val=0.0, units="T")
        self.add_input("B_smax", val=0.0, units="T")
        self.add_input("K_rad", val=0.0)
        self.add_input("K_rad_LL", val=0.0)
        self.add_input("K_rad_UL", val=0.0)
        self.add_input("D_ratio", val=0.0)
        self.add_input("D_ratio_LL", val=0.0)
        self.add_input("D_ratio_UL", val=0.0)

        self.add_output("con_uas", val=0.0, units="m")
        self.add_output("con_zas", val=0.0, units="m")
        self.add_output("con_yas", val=0.0, units="m")
        self.add_output("con_bst", val=0.0, units="m")
        self.add_output("con_uar", val=0.0, units="m")
        self.add_output("con_yar", val=0.0, units="m")
        self.add_output("con_zar", val=0.0, units="m")
        self.add_output("con_br", val=0.0, units="m")
        self.add_output("TCr", val=0.0, units="m**3")
        self.add_output("TCs", val=0.0, units="m**3")
        self.add_output("con_TC2r", val=0.0, units="m**3")
        self.add_output("con_TC2s", val=0.0, units="m**3")
        self.add_output("con_Bsmax", val=0.0, units="T")
        self.add_output("K_rad_L", val=0.0)
        self.add_output("K_rad_U", val=0.0)
        self.add_output("D_ratio_L", val=0.0)
        self.add_output("D_ratio_U", val=0.0)

    def compute(self, inputs, outputs):
        outputs["con_uas"] = inputs["u_allow_s"] - inputs["u_as"]
        outputs["con_zas"] = inputs["z_allow_s"] - inputs["z_as"]
        outputs["con_yas"] = inputs["y_allow_s"] - inputs["y_as"]
        outputs["con_bst"] = inputs["b_allow_s"] - inputs["b_st"]  # b_st={'units':'m'}
        outputs["con_uar"] = inputs["u_allow_r"] - inputs["u_ar"]
        outputs["con_yar"] = inputs["y_allow_r"] - inputs["y_ar"]
        outputs["con_TC2r"] = inputs["TC2s"] - inputs["TC1"]
        outputs["con_TC2s"] = inputs["TC2s"] - inputs["TC1"]
        outputs["con_Bsmax"] = inputs["B_g"] - inputs["B_smax"]
        outputs["con_zar"] = inputs["z_allow_r"] - inputs["z_ar"]
        outputs["con_br"] = inputs["b_allow_r"] - inputs["b_arm"]  # b_r={'units':'m'}
        outputs["TCr"] = inputs["TC2r"] - inputs["TC1"]
        outputs["TCs"] = inputs["TC2s"] - inputs["TC1"]
        outputs["K_rad_L"] = inputs["K_rad"] - inputs["K_rad_LL"]
        outputs["K_rad_U"] = inputs["K_rad"] - inputs["K_rad_UL"]
        outputs["D_ratio_L"] = inputs["D_ratio"] - inputs["D_ratio_LL"]
        outputs["D_ratio_U"] = inputs["D_ratio"] - inputs["D_ratio_UL"]


# ----------------------------------------------------------------------------------------------
class MofI(om.ExplicitComponent):
    """
    Compute moments of inertia.

    Parameters
    ----------
    R_out : float, [m]
        Outer radius
    stator_mass : float, [kg]
        Total rotor mass
    rotor_mass : float, [kg]
        Total rotor mass
    generator_mass : float, [kg]
        Actual mass
    len_s : float, [m]
        Stator core length

    Returns
    -------
    generator_I : numpy array[3], [kg*m**2]
        Moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    rotor_I : numpy array[3], [kg*m**2]
        Moments of Inertia for the rotor about its center of mass
    stator_I : numpy array[3], [kg*m**2]
        Moments of Inertia for the stator about its center of mass

    """

    def setup(self):
        self.add_input("R_out", val=0.0, units="m")
        self.add_input("stator_mass", val=0.0, units="kg")
        self.add_input("rotor_mass", val=0.0, units="kg")
        self.add_input("generator_mass", val=0.0, units="kg")
        self.add_input("len_s", val=0.0, units="m")

        self.add_output("generator_I", val=np.zeros(3), units="kg*m**2")
        self.add_output("rotor_I", val=np.zeros(3), units="kg*m**2")
        self.add_output("stator_I", val=np.zeros(3), units="kg*m**2")

    def compute(self, inputs, outputs):
        R_out = float(inputs["R_out"][0])
        Mass = float(inputs["generator_mass"][0])
        m_stator = float(inputs["stator_mass"][0])
        m_rotor = float(inputs["rotor_mass"][0])
        len_s = float(inputs["len_s"][0])

        I = np.zeros(3)
        I[0] = 0.50 * Mass * R_out**2
        I[1] = I[2] = 0.5 * I[0] + Mass * len_s**2 / 12.0
        outputs["generator_I"] = I
        coeff = m_stator / Mass if m_stator > 0.0 else 0.5
        outputs["stator_I"] = coeff * I
        coeff = m_rotor / Mass if m_rotor > 0.0 else 0.5
        outputs["rotor_I"] = coeff * I


# ----------------------------------------------------------------------------------------------


class Cost(om.ExplicitComponent):
    """
    Provides a material cost estimate for a PMSG _arms generator. Manufacturing costs are excluded.

    Parameters
    ----------
    C_Cu : float, [USD/kg]
        Specific cost of copper
    C_Fe : float, [USD/kg]
        Specific cost of magnetic steel/iron
    C_Fes : float, [USD/kg]
        Specific cost of structural steel
    C_PM : float, [USD/kg]
        Specific cost of Magnet
    Copper : float, [kg]
        Copper mass
    Iron : float, [kg]
        Iron mass
    mass_PM : float, [kg]
        Magnet mass
    Structural_mass : float, [kg]
        Structural mass

    Returns
    -------
    generator_cost : float, [USD]
        Total cost

    """

    def setup(self):
        # Specific cost of material by type
        self.add_input("C_Cu", val=0.0, units="USD/kg")
        self.add_input("C_Fe", val=0.0, units="USD/kg")
        self.add_input("C_Fes", val=0.0, units="USD/kg")
        self.add_input("C_PM", val=0.0, units="USD/kg")

        # Mass of each material type
        self.add_input("Copper", val=0.0, units="kg")
        self.add_input("Iron", val=0.0, units="kg")
        self.add_input("mass_PM", val=0.0, units="kg")
        self.add_input("Structural_mass", val=0.0, units="kg")

        # Outputs
        self.add_output("generator_cost", val=0.0, units="USD")

        # self.declare_partials('*', '*', method='fd', form='central', step=1e-6)

    def compute(self, inputs, outputs):
        Copper = inputs["Copper"]
        Iron = inputs["Iron"]
        mass_PM = inputs["mass_PM"]
        Structural_mass = inputs["Structural_mass"]
        C_Cu = inputs["C_Cu"]
        C_Fes = inputs["C_Fes"]
        C_Fe = inputs["C_Fe"]
        C_PM = inputs["C_PM"]

        # Industrial electricity rate $/kWh https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
        k_e = 0.064

        # Material cost ($/kg) and electricity usage cost (kWh/kg)*($/kWh) for the materials with waste fraction
        K_copper = Copper * (1.26 * C_Cu + 96.2 * k_e)
        K_iron = Iron * (1.21 * C_Fe + 26.9 * k_e)
        K_pm = mass_PM * (1.0 * C_PM + 79.0 * k_e)
        K_steel = Structural_mass * (1.21 * C_Fes + 15.9 * k_e)
        # Account for capital cost and labor share from BLS MFP by NAICS
        outputs["generator_cost"] = (K_copper + K_pm) / 0.619 + (K_iron + K_steel) / 0.684


# ----------------------------------------------------------------------------------------------


class PowerElectronicsEff(om.ExplicitComponent):
    """
    Compute representative efficiency of power electronics

    Parameters
    ----------
    machine_rating : float, [W]
        Machine rating
    shaft_rpm : numpy array[n_pc], [rpm]
        rated speed of input shaft (lss for direct, hss for geared)
    eandm_efficiency : numpy array[n_pc]
        Generator electromagnetic efficiency values (<1)

    Returns
    -------
    converter_efficiency : numpy array[n_pc]
        Converter efficiency values (<1)
    transformer_efficiency : numpy array[n_pc]
        Transformer efficiency values (<1)
    generator_efficiency : numpy array[n_pc]
        Full generato and power electronics efficiency values (<1)

    """

    def initialize(self):
        self.options.declare("n_pc", default=20)

    def setup(self):
        n_pc = self.options["n_pc"]

        self.add_input("machine_rating", val=0.0, units="W")
        self.add_input("shaft_rpm", val=np.zeros(n_pc), units="rpm")
        self.add_input("eandm_efficiency", val=np.zeros(n_pc))

        self.add_output("converter_efficiency", val=np.zeros(n_pc))
        self.add_output("transformer_efficiency", val=np.zeros(n_pc))
        self.add_output("generator_efficiency", val=np.zeros(n_pc))

    def compute(self, inputs, outputs):
        # Unpack inputs
        rating = inputs["machine_rating"]
        rpmData = inputs["shaft_rpm"]
        rpmRatio = rpmData / rpmData[-1]

        # This converter efficiency is from the APEEM Group in 2020
        # See Sreekant Narumanchi, Kevin Bennion, Bidzina Kekelia, Ramchandra Kotecha
        # Converter constants
        v_dc, v_dc0, c0, c1, c2, c3 = 6600, 6200, -2.1e-10, 1.2e-5, 1.46e-3, -2e-4
        p_ac0, p_dc0 = 0.99 * rating, rating
        p_s0 = 1e-3 * p_dc0

        # calculated parameters
        a = p_dc0 * (1.0 + c1 * (v_dc - v_dc0))
        b = p_s0 * (1.0 + c2 * (v_dc - v_dc0))
        c = c0 * (1.0 + c3 * (v_dc - v_dc0))

        # Converter efficiency
        p_dc = rpmRatio * p_dc0
        p_ac = (p_ac0 / (a - b) - c * (a - b)) * (p_dc - b) + c * ((p_dc - b) ** 2)
        conv_eff = p_ac / p_dc

        # Transformer loss model is P_loss = P_0 + a^2 * P_k
        # a is output power/rated
        p0, pk, rT = 16.0, 111.0, 5.0 / 3.0
        a = rpmRatio * (1 / rT)
        # This gives loss in kW, so need to convert to efficiency
        trans_eff = 1.0 - (p0 + a * a * pk) / (1e-3 * rating)

        # Store outputs
        outputs["converter_efficiency"] = conv_eff
        outputs["transformer_efficiency"] = trans_eff
        outputs["generator_efficiency"] = conv_eff * trans_eff * inputs["eandm_efficiency"]


# ----------------------------------------------------------------------------------------------
class Generator(om.Group):
    def initialize(self):
        genTypes = ["scig", "dfig", "eesg", "pmsg_arms", "pmsg_disc", "pmsg_outer"]
        self.options.declare("design", values=genTypes + [m.upper() for m in genTypes])
        self.options.declare("n_pc", default=20)

    def setup(self):
        genType = self.options["design"]
        n_pc = self.options["n_pc"]

        # ivc = om.IndepVarComp()
        # sivc = om.IndepVarComp()

        self.set_input_defaults("B_r", val=1.2, units="T")
        self.set_input_defaults("P_Fe0e", val=1.0, units="W/kg")
        self.set_input_defaults("P_Fe0h", val=4.0, units="W/kg")
        self.set_input_defaults("S_N", val=-0.002)
        self.set_input_defaults("alpha_p", val=0.5 * np.pi * 0.7)
        self.set_input_defaults("b_r_tau_r", val=0.45)
        self.set_input_defaults("b_ro", val=0.004, units="m")
        self.set_input_defaults("b_s_tau_s", val=0.45)
        self.set_input_defaults("b_so", val=0.004, units="m")
        self.set_input_defaults("cofi", val=0.85)
        self.set_input_defaults("freq", val=60, units="Hz")
        self.set_input_defaults("h_i", val=0.001, units="m")
        self.set_input_defaults("h_sy0", val=0.0)
        self.set_input_defaults("h_w", val=0.005, units="m")
        self.set_input_defaults("k_fes", val=0.9)
        self.set_input_defaults("k_fillr", val=0.7)
        self.set_input_defaults("k_fills", val=0.65)
        self.set_input_defaults("k_s", val=0.2)
        #self.set_input_defaults("m", val=3)
        self.set_input_defaults("mu_0", val=np.pi * 4e-7, units="m*kg/s**2/A**2")
        self.set_input_defaults("mu_r", val=1.06, units="m*kg/s**2/A**2")
        self.set_input_defaults("p", val=3.0)
        self.set_input_defaults("phi", val=np.deg2rad(90), units="rad")
        #self.set_input_defaults("q1", val=6)
        #self.set_input_defaults("q2", val=4)
        self.set_input_defaults("ratio_mw2pp", val=0.7)
        self.set_input_defaults("resist_Cu", val=1.8e-8 * 1.4, units="ohm/m")
        self.set_input_defaults("sigma", val=40e3, units="Pa")
        self.set_input_defaults("y_tau_p", val=1.0)
        self.set_input_defaults("y_tau_pr", val=10.0 / 12)

        # self.set_input_defaults('I_0', val=0.0, units='A')
        # self.set_input_defaults('d_r', val=0.0, units='m')
        # self.set_input_defaults('h_m', val=0.0, units='m')
        # self.set_input_defaults('h_0', val=0.0, units ='m')
        # self.set_input_defaults('h_s', val=0.0, units='m')
        # self.set_input_defaults('len_s', val=0.0, units='m')
        # self.set_input_defaults('n_r', val=0.0)
        # self.set_input_defaults('rad_ag', val=0.0, units='m')
        # self.set_input_defaults('t_wr', val=0.0, units='m')

        # self.set_input_defaults('n_s', val=0.0)
        # self.set_input_defaults('b_st', val=0.0, units='m')
        # self.set_input_defaults('d_s', val=0.0, units='m')
        # self.set_input_defaults('t_ws', val=0.0, units='m')

        # self.set_input_defaults('rho_Copper', val=0.0, units='kg*m**-3')
        # self.set_input_defaults('rho_Fe', val=0.0, units='kg*m**-3')
        # self.set_input_defaults('rho_Fes', val=0.0, units='kg*m**-3')
        # self.set_input_defaults('rho_PM', val=0.0, units='kg*m**-3')

        # self.set_input_defaults('C_Cu',  val=0.0, units='USD/kg')
        # self.set_input_defaults('C_Fe',  val=0.0, units='USD/kg')
        # self.set_input_defaults('C_Fes', val=0.0, units='USD/kg')
        # self.set_input_defaults('C_PM',  val=0.0, units='USD/kg')

        # if genType.lower() in ['pmsg_outer']:
        #    self.set_input_defaults('r_g',0.0, units ='m')
        #    self.set_input_defaults('N_c',0.0)
        #    self.set_input_defaults('b',0.0)
        #    self.set_input_defaults('c',0.0)
        #    self.set_input_defaults('E_p',0.0, units ='V')
        #    self.set_input_defaults('h_yr', val=0.0, units ='m')
        #    self.set_input_defaults('h_ys', val=0.0, units ='m')
        #    self.set_input_defaults('h_sr',0.0,units='m',desc='Structural Mass')
        #    self.set_input_defaults('h_ss',0.0, units ='m')
        #    self.set_input_defaults('t_r',0.0, units ='m')
        #    self.set_input_defaults('t_s',0.0, units ='m')

        #    self.set_input_defaults('u_allow_pcent',0.0)
        #    self.set_input_defaults('y_allow_pcent',0.0)
        #    self.set_input_defaults('z_allow_deg',0.0,units='deg')
        #    self.set_input_defaults('B_tmax',0.0, units='T')

        #    self.set_input_defaults('P_mech', 0.0, units='W')
        #    self.set_input_defaults('y_sh', units ='m')
        #    self.set_input_defaults('theta_sh', 0.0, units='rad')
        #    self.set_input_defaults('D_nose',0.0, units ='m')
        #    self.set_input_defaults('y_bd', units ='m')
        #    self.set_input_defaults('theta_bd', 0.0, units='rad')

        # if genType.lower() in ['eesg','pmsg_arms','pmsg_disc']:
        #    self.set_input_defaults('tau_p', val=0.0, units='m')
        #    self.set_input_defaults('h_ys',  val=0.0, units='m')
        #    self.set_input_defaults('h_yr',  val=0.0, units='m')
        #    self.set_input_defaults('b_arm',   val=0.0, units='m')

        # elif genType.lower() in ['scig','dfig']:
        #    self.set_input_defaults('B_symax', val=0.0, units='T')
        #    self.set_input_defaults('S_Nmax', val=-0.2)

        # if topLevelFlag:
        #    self.add_subsystem('ivc', ivc, promotes=['*'])

        #    self.set_input_defaults('machine_rating', 0.0, units='W')
        #    self.set_input_defaults('shaft_rpm', np.linspace(1.0, 10.0, n_pc), units='rpm')
        #    self.set_input_defaults('rated_torque', 0.0, units='N*m')
        #    self.set_input_defaults('D_shaft', val=0.0, units='m')
        self.set_input_defaults("E", val=210e9, units="Pa")
        self.set_input_defaults("G", val=81e9, units="Pa")
        #    self.add_subsystem('sivc', sivc, promotes=['*'])

        # Easy Poisson ratio assuming isotropic
        self.add_subsystem(
            "poisson", om.ExecComp("v = 0.5*E/G - 1.0", E={"units": "Pa"}, G={"units": "Pa"}), promotes=["*"]
        )

        # Add generator design component and cost
        if genType.lower() == "scig":
            mygen = gm.SCIG

        elif genType.lower() == "dfig":
            mygen = gm.DFIG

        elif genType.lower() == "eesg":
            mygen = gm.EESG

        elif genType.lower() == "pmsg_arms":
            mygen = gm.PMSG_Arms

        elif genType.lower() == "pmsg_disc":
            mygen = gm.PMSG_Disc

        elif genType.lower() == "pmsg_outer":
            mygen = gm.PMSG_Outer

        self.add_subsystem("generator", mygen(n_pc=n_pc), promotes=["*"])
        self.add_subsystem("mofi", MofI(), promotes=["*"])
        self.add_subsystem("gen_cost", Cost(), promotes=["*"])
        self.add_subsystem("constr", Constraints(), promotes=["*"])
        self.add_subsystem("eff", PowerElectronicsEff(n_pc=n_pc), promotes=["*"])
