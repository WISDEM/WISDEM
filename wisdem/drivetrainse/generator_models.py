import sys

import numpy as np
import pandas as pd
import openmdao.api as om

from wisdem.commonse import gravity

eps = 1e-3


# Convenience functions for computing McDonald's C and F parameters
def chsMshc(x):
    return np.cosh(x) * np.sin(x) - np.sinh(x) * np.cos(x)


def chsPshc(x):
    return np.cosh(x) * np.sin(x) + np.sinh(x) * np.cos(x)


def carterFactor(airGap, slotOpening, slotPitch):
    """Return Carter factor
    (based on Langsdorff's empirical expression)
      See page 3-13 Boldea Induction machines Chapter 3
    """
    gma = (2 * slotOpening / airGap) ** 2 / (5 + 2 * slotOpening / airGap)
    return slotPitch / (slotPitch - airGap * gma * 0.5)


# ---------------
def carterFactorMcDonald(airGap, h_m, slotOpening, slotPitch):
    """Return Carter factor using Carter's equation
      (based on Schwartz-Christoffel's conformal mapping on simplified slot geometry)

    This code is based on Eq. B.3-5 in Appendix B of McDonald's thesis.
    It is used by PMSG_arms and PMSG_disc.

    h_m   : magnet height (m)
    b_so  : stator slot opening (m)
    tau_s : Stator slot pitch (m)
    """
    mu_r = 1.06  # relative permeability (probably for neodymium magnets, often given as 1.05 - GNS)
    g_1 = airGap + h_m / mu_r  # g
    b_over_a = slotOpening / (2 * g_1)
    gamma = 4 / np.pi * (b_over_a * np.arctan(b_over_a) - np.log(np.sqrt(1 + b_over_a**2)))
    return slotPitch / (slotPitch - gamma * g_1)


# ---------------
def carterFactorEmpirical(airGap, slotOpening, slotPitch):
    """Return Carter factor using Langsdorff's empirical expression"""
    sigma = (slotOpening / airGap) / (5 + slotOpening / airGap)
    return slotPitch / (slotPitch - sigma * slotOpening)


# ---------------
def carterFactorSalientPole(airGap, slotWidth, slotPitch):
    """Return Carter factor for salient pole rotor
    Where does this equation come from? It's different from other approximations above.
    Original code:
        tau_s = np.pi * dia / S                      # slot pitch
        b_s  = tau_s * b_s_tau_s                  # slot width
        b_t = tau_s - b_s                         # tooth width
        K_C1 = (tau_s + 10 * g_a) / (tau_s - b_s + 10 * g_a)  # salient pole rotor
    slotPitch - slotWidth == toothWidth
    """
    return (slotPitch + 10 * airGap) / (slotPitch - slotWidth + 10 * airGap)  # salient pole rotor


# ---------------------------------
def array_seq(q1, b, c, Total_number):
    Seq = np.array([1, 0, 0, 1, 0])
    diff = Total_number * 5 / 6
    G = np.prod(Seq.shape)
    return Seq, diff, G


# ---------------------------------
def winding_factor(Sin, b, c, p, m):
    S = int(Sin)

    # Step 1 Writing q1 as a fraction
    q1 = b / c

    # Step 2: Writing a binary sequence of b-c zeros and b ones
    Total_number = int(S / b)
    L = array_seq(q1, b, c, Total_number)

    # STep 3 : Repeat binary sequence Q_s/b times
    New_seq = np.tile(L[0], Total_number)
    Actual_seq1 = pd.DataFrame(New_seq[:, None].T)
    Winding_sequence = ["A", "C1", "B", "A1", "C", "B1"]

    New_seq2 = np.tile(Winding_sequence, int(L[1]))
    Actual_seq2 = pd.DataFrame(New_seq2[:, None].T)
    Seq_f = pd.concat([Actual_seq1, Actual_seq2], ignore_index=True)
    Seq_f.reset_index(drop=True)

    R = S if S % 2 == 0 else S + 1

    windings_df = pd.DataFrame(index=Seq_f.index, columns=Seq_f.columns[1:R], data=np.zeros((len(Seq_f), R-1)))
    windings_idx = list(windings_df.loc[0])
    windings_arrange = list(windings_df.loc[1])

    # Step #4 Arranging winding in Slots
    counter = 0
    for i in range(len(New_seq)):
        if counter >= len(windings_idx):
            break
        if Seq_f.loc[0, i] == 1:
            windings_idx[counter] = Seq_f.loc[1, i]
            counter += 1

    windings_arrange[0] = "C1"
    for k in range(1, R):
        if k >= len(windings_idx):
            break
        if windings_idx[k - 1] == "A":
            windings_arrange[k] = "A1"
        elif windings_idx[k - 1] == "B":
            windings_arrange[k] = "B1"
        elif windings_idx[k - 1] == "C":
            windings_arrange[k] = "C1"
        elif windings_idx[k - 1] == "A1":
            windings_arrange[k] = "A"
        elif windings_idx[k - 1] == "B1":
            windings_arrange[k] = "B"
        elif windings_idx[k - 1] == "C1":
            windings_arrange[k] = "C"

    Phase_A = np.zeros(1000)
    counter_A = 0
    # Windings_arrange.to_excel('test.xlsx')
    # Winding vector, W_A for Phase A
    for k in range(1, R):
        if windings_idx[k - 1] == "A" and windings_arrange[k - 1] == "A":
            Phase_A[counter_A] = k
            Phase_A[counter_A + 1] = k
            counter_A += 2
        elif windings_idx[k - 1] == "A1" and windings_arrange[k - 1] == "A1":
            Phase_A[counter_A] = -1 * k
            Phase_A[counter_A + 1] = -1 * k
            counter_A += 2
        elif windings_idx[k - 1] == "A" or windings_arrange[k - 1] == "A":
            Phase_A[counter_A] = k
            counter_A += 1
        elif windings_idx[k - 1] == "A1" or windings_arrange[k - 1] == "A1":
            Phase_A[counter_A] = -1 * k
            counter_A += 1

    W_A = np.trim_zeros(Phase_A)
    # Calculate winding factor
    K_w = 0

    for r in range(0, int(2 * (S) / 3)):
        Gamma = 2 * np.pi * p * np.abs(W_A[r]) / S
        K_w += np.sign(W_A[r]) * (np.exp(Gamma * 1j))

    K_w = np.abs(K_w) / (2 * S / 3)
    CPMR = np.lcm(S, int(2 * p))
    N_cog_s = CPMR / S
    N_cog_p = CPMR / p
    N_cog_t = CPMR * 0.5 / p
    A = np.lcm(S, int(2 * p))
    b_p_tau_p = 2 * 1 * p / S - 0
    b_t_tau_s = (2) * S * 0.5 / p - 2

    return K_w


# ---------------------------------
def shell_constant(R, t, l, x, E, v):
    Lambda = (3 * (1 - v**2) / (R**2 * t**2)) ** 0.25
    D = E * t**3 / (12 * (1 - v**2))
    C_14 = (np.sinh(Lambda * l)) ** 2 + (np.sin(Lambda * l)) ** 2
    C_11 = (np.sinh(Lambda * l)) ** 2 - (np.sin(Lambda * l)) ** 2
    F_2 = np.cosh(Lambda * x) * np.sin(Lambda * x) + np.sinh(Lambda * x) * np.cos(Lambda * x)
    C_13 = np.cosh(Lambda * l) * np.sinh(Lambda * l) - np.cos(Lambda * l) * np.sin(Lambda * l)
    F_1 = np.cosh(Lambda * x) * np.cos(Lambda * x)
    F_4 = np.cosh(Lambda * x) * np.sin(Lambda * x) - np.sinh(Lambda * x) * np.cos(Lambda * x)

    return D, Lambda, C_14, C_11, F_2, C_13, F_1, F_4


# ---------------------------------
def plate_constant(a, b, E, v, r_o, t):
    D = E * t**3 / (12 * (1 - v**2))
    C_2 = 0.25 * (1 - (b / a) ** 2 * (1 + 2 * np.log(a / b)))
    C_3 = 0.25 * (b / a) * (((b / a) ** 2 + 1) * np.log(a / b) + (b / a) ** 2 - 1)
    C_5 = 0.5 * (1 - (b / a) ** 2)
    C_6 = 0.25 * (b / a) * ((b / a) ** 2 - 1 + 2 * np.log(a / b))
    C_8 = 0.5 * (1 + v + (1 - v) * (b / a) ** 2)
    C_9 = (b / a) * (0.5 * (1 + v) * np.log(a / b) + 0.25 * (1 - v) * (1 - (b / a) ** 2))
    L_11 = (1 / 64) * (
        1 + 4 * (r_o / a) ** 2 - 5 * (r_o / a) ** 4 - 4 * (r_o / a) ** 2 * (2 + (r_o / a) ** 2) * np.log(a / r_o)
    )
    L_17 = 0.25 * (1 - 0.25 * (1 - v) * ((1 - (r_o / a) ** 4) - (r_o / a) ** 2 * (1 + (1 + v) * np.log(a / r_o))))

    return D, C_2, C_3, C_5, C_6, C_8, C_9, L_11, L_17


# ---------------------------------

debug = False
# ---------------------------------


class GeneratorBase(om.ExplicitComponent):
    """
    Base class for generators

    Parameters
    ----------
    B_r : float, [T]
        Remnant flux density
    E : float, [Pa]
        youngs modulus
    G : float, [Pa]
        Shear modulus
    P_Fe0e : float, [W/kg]
        specific eddy losses @ 1.5T, 50Hz
    P_Fe0h : float, [W/kg]
        specific hysteresis losses W / kg @ 1.5 T @50 Hz
    S_N : float
        Slip
    alpha_p : float

    b_r_tau_r : float
        Rotor Slot width / Slot pitch ratio
    b_ro : float, [m]
        Rotor slot opening width
    b_s_tau_s : float
        Stator Slot width/Slot pitch ratio
    b_so : float, [m]
        Stator slot opening width
    cofi : float
        power factor
    freq : float, [Hz]
        grid frequency
    h_i : float, [m]
        coil insulation thickness
    h_sy0 : float

    h_w : float, [m]
        Slot wedge height
    k_fes : float
        Stator iron fill factor per Grauers
    k_fillr : float
        Rotor slot fill factor
    k_fills : float
        Stator Slot fill factor
    k_s : float
        magnetic saturation factor for iron
    m : int
        Number of phases
    mu_0 : float, [m*kg/s**2/A**2]
        permeability of free space
    mu_r : float, [m*kg/s**2/A**2]
        relative permeability (neodymium)
    p : float
        number of pole pairs (taken as int within code)
    phi : numpy array[90], [rad]
        tilt angle (during transportation)
    q1 : int
        Stator slots per pole per phase
    q2 : int
        Rotor slots per pole per phase
    ratio_mw2pp : float
        ratio of magnet width to pole pitch(bm / self.tau_p)
    resist_Cu : float, [ohm/m]
        Copper resistivity
    sigma : float, [Pa]
        assumed max shear stress
    v : float
        poisson ratio
    y_tau_p : float
        Stator coil span to pole pitch
    y_tau_pr : float
        Rotor coil span to pole pitch
    I_0 : float, [A]
        no-load excitation current
    T_rated : float, [N*m]
        Rated torque
    d_r : float, [m]
        arm depth d_r
    h_m : float, [m]
        magnet height
    h_0 : float, [m]
        Slot height
    h_s : float, [m]
        Yoke height h_s
    len_s : float, [m]
        Stator core length
    machine_rating : float, [W]
        Machine rating
    shaft_rpm : numpy array[n_pc], [rpm]
        rated speed of input shaft (lss for direct, hss for geared)
    n_r : float
        number of arms n
    rad_ag : float, [m]
        airgap radius
    t_wr : float, [m]
        arm depth thickness
    n_s : float
        number of stator arms n_s
    b_st : float, [m]
        arm width b_st
    d_s : float, [m]
        arm depth d_s
    t_ws : float, [m]
        arm depth thickness
    D_shaft : float, [m]
        Shaft diameter
    rho_Copper : float, [kg*m**-3]
        Copper density
    rho_Fe : float, [kg*m**-3]
        Magnetic Steel density
    rho_Fes : float, [kg*m**-3]
        Structural Steel density
    rho_PM : float, [kg*m**-3]
        Magnet density

    Returns
    -------
    B_rymax : float, [T]
        Peak Rotor yoke flux density
    B_trmax : float, [T]
        maximum tooth flux density in rotor
    B_tsmax : float, [T]
        maximum tooth flux density in stator
    B_g : float, [T]
        Peak air gap flux density B_g
    B_g1 : float, [T]
        air gap flux density fundamental
    B_pm1 : float
        Fundamental component of peak air gap flux density
    N_s : float
        Number of turns in the stator winding
    b_s : float, [m]
        slot width
    b_t : float, [m]
        tooth width
    A_Curcalc : float, [mm**2]
        Conductor cross-section mm^2
    A_Cuscalc : float, [mm**2]
        Stator Conductor cross-section mm^2
    b_m : float
        magnet width
    mass_PM : float, [kg]
        Magnet mass
    Copper : float, [kg]
        Copper Mass
    Iron : float, [kg]
        Electrical Steel Mass
    Structural_mass : float, [kg]
        Structural Mass
    generator_mass : float, [kg]
        Actual mass
    f : float
        Generator output frequency
    I_s : float, [A]
        Generator output phase current
    R_s : float, [ohm]
        Stator resistance
    L_s : float
        Stator synchronising inductance
    J_s : float, [A*m**-2]
        Stator winding current density
    A_1 : float
        Specific current loading
    K_rad : float
        Stack length ratio
    Losses : numpy array[n_pc], [W]
        Total loss
    generator_efficiency : numpy array[n_pc]
        Generator electromagnetic efficiency values (<1)
    u_ar : float, [m]
        Rotor radial deflection
    u_as : float, [m]
        Stator radial deflection
    u_allow_r : float, [m]
        Allowable radial rotor
    u_allow_s : float, [m]
        Allowable radial stator
    y_ar : float, [m]
        Rotor axial deflection
    y_as : float, [m]
        Stator axial deflection
    y_allow_r : float, [m]
        Allowable axial
    y_allow_s : float, [m]
        Allowable axial
    z_ar : float, [m]
        Rotor circumferential deflection
    z_as : float, [m]
        Stator circumferential deflection
    z_allow_r : float, [m]
        Allowable circum rotor
    z_allow_s : float, [m]
        Allowable circum stator
    b_allow_r : float, [m]
        Allowable arm dimensions
    b_allow_s : float, [m]
        Allowable arm
    TC1 : float, [m**3]
        Torque constraint
    TC2r : float, [m**3]
        Torque constraint-rotor
    TC2s : float, [m**3]
        Torque constraint-stator
    R_out : float, [m]
        Outer radius
    S : float
        Stator slots
    Slot_aspect_ratio : float
        Slot aspect ratio
    Slot_aspect_ratio1 : float
        Stator slot aspect ratio
    Slot_aspect_ratio2 : float
        Rotor slot aspect ratio
    D_ratio : float
        Stator diameter ratio
    J_r : float
        Rotor winding Current density
    L_sm : float
        mutual inductance
    Q_r : float
        Rotor slots
    R_R : float
        Rotor resistance
    b_r : float
        rotor slot width
    b_tr : float
        rotor tooth width
    b_trmin : float
        minimum tooth width

    """

    def initialize(self):
        self.options.declare("n_pc", default=20)

    def setup(self):
        n_pc = self.options["n_pc"]

        # Constants and parameters
        self.add_input("B_r", val=1.2, units="T")
        self.add_input("E", val=0.0, units="Pa")
        self.add_input("G", val=0.0, units="Pa")
        self.add_input("P_Fe0e", val=1.0, units="W/kg")
        self.add_input("P_Fe0h", val=4.0, units="W/kg")
        self.add_input("S_N", val=-0.002)
        self.add_input("alpha_p", val=0.5 * np.pi * 0.7)
        self.add_input("b_r_tau_r", val=0.45)
        self.add_input("b_ro", val=0.004, units="m")
        self.add_input("b_s_tau_s", val=0.45)
        self.add_input("b_so", val=0.004, units="m")
        self.add_input("cofi", val=0.85)
        self.add_input("freq", val=60, units="Hz")
        self.add_input("h_i", val=0.001, units="m")
        self.add_input("h_sy0", val=0.0)
        self.add_input("h_w", val=0.005, units="m")
        self.add_input("k_fes", val=0.9)
        self.add_input("k_fillr", val=0.7)
        self.add_input("k_fills", val=0.65)
        self.add_input("k_s", val=0.2)
        self.add_discrete_input("m", val=3)
        self.add_input("mu_0", val=np.pi * 4e-7, units="m*kg/s**2/A**2")
        self.add_input("mu_r", val=1.06, units="m*kg/s**2/A**2")
        self.add_input("p", val=3.0)
        self.add_input("phi", val=np.deg2rad(90), units="rad")
        self.add_discrete_input("q1", val=6)
        self.add_discrete_input("q2", val=4)
        self.add_input("ratio_mw2pp", val=0.7)
        self.add_input("resist_Cu", val=1.8e-8 * 1.4, units="ohm/m")
        self.add_input("sigma", val=40e3, units="Pa")
        self.add_input("v", val=0.3)
        self.add_input("y_tau_p", val=1.0)
        self.add_input("y_tau_pr", val=10.0 / 12)

        # General inputs
        # self.add_input('r_s', val=0.0, units='m', desc='airgap radius r_s')
        self.add_input("I_0", val=0.0, units="A")
        self.add_input("rated_torque", val=0.0, units="N*m")
        self.add_input("d_r", val=0.0, units="m")
        self.add_input("h_m", val=0.0, units="m")
        self.add_input("h_0", val=0.0, units="m")
        self.add_input("h_s", val=0.0, units="m")
        self.add_input("len_s", val=0.0, units="m")
        self.add_input("machine_rating", val=0.0, units="W")
        self.add_input("shaft_rpm", val=np.zeros(n_pc), units="rpm")
        self.add_input("n_r", val=0.0)
        self.add_input("rad_ag", val=0.0, units="m")
        self.add_input("t_wr", val=0.0, units="m")

        # Structural design variables
        self.add_input("n_s", val=0.0)
        self.add_input("b_st", val=0.0, units="m")
        self.add_input("d_s", val=0.0, units="m")
        self.add_input("t_ws", val=0.0, units="m")
        self.add_input("D_shaft", val=0.0, units="m")

        # Material properties
        self.add_input("rho_Copper", val=8900.0, units="kg*m**-3")
        self.add_input("rho_Fe", val=7700.0, units="kg*m**-3")
        self.add_input("rho_Fes", val=7850.0, units="kg*m**-3")
        self.add_input("rho_PM", val=7450.0, units="kg*m**-3")

        # Magnetic loading
        self.add_output("B_rymax", val=0.0, units="T")
        self.add_output("B_trmax", val=0.0, units="T")
        self.add_output("B_tsmax", val=0.0, units="T")
        self.add_output("B_g", val=0.0, units="T")
        self.add_output("B_g1", val=0.0, units="T")
        self.add_output("B_pm1", val=0.0)

        # Stator design
        self.add_output("N_s", val=0.0)
        self.add_output("b_s", val=0.0, units="m")
        self.add_output("b_t", val=0.0, units="m")
        self.add_output("A_Curcalc", val=0.0, units="mm**2")
        self.add_output("A_Cuscalc", val=0.0, units="mm**2")

        # Rotor magnet dimension
        self.add_output("b_m", val=0.0)

        # Mass Outputs
        self.add_output("mass_PM", val=0.0, units="kg")
        self.add_output("Copper", val=0.0, units="kg")
        self.add_output("Iron", val=0.0, units="kg")
        self.add_output("Structural_mass", val=0.0, units="kg")
        self.add_output("generator_mass", val=0.0, units="kg")

        # Electrical performance
        self.add_output("f", val=np.zeros(n_pc))
        self.add_output("I_s", val=np.zeros(n_pc), units="A")
        self.add_output("R_s", val=np.zeros(n_pc), units="ohm")
        self.add_output("L_s", val=0.0)
        self.add_output("J_s", val=np.zeros(n_pc), units="A*m**-2")
        self.add_output("A_1", val=np.zeros(n_pc))

        # Objective functions
        self.add_output("K_rad", val=0.0)
        self.add_output("Losses", val=np.zeros(n_pc), units="W")
        self.add_output("eandm_efficiency", val=np.zeros(n_pc))

        # Structural performance
        self.add_output("u_ar", val=0.0, units="m")
        self.add_output("u_as", val=0.0, units="m")
        self.add_output("u_allow_r", val=0.0, units="m")
        self.add_output("u_allow_s", val=0.0, units="m")
        self.add_output("y_ar", val=0.0, units="m")
        self.add_output("y_as", val=0.0, units="m")
        self.add_output("y_allow_r", val=0.0, units="m")
        self.add_output("y_allow_s", val=0.0, units="m")
        self.add_output("z_ar", val=0.0, units="m")
        self.add_output("z_as", val=0.0, units="m")
        self.add_output("z_allow_r", val=0.0, units="m")
        self.add_output("z_allow_s", val=0.0, units="m")
        self.add_output("b_allow_r", val=0.0, units="m")
        self.add_output("b_allow_s", val=0.0, units="m")
        self.add_output("TC1", val=0.0, units="m**3")
        self.add_output("TC2r", val=0.0, units="m**3")
        self.add_output("TC2s", val=0.0, units="m**3")

        # Other parameters
        self.add_output("R_out", val=0.0, units="m")
        self.add_output("S", val=0.0)
        self.add_output("Slot_aspect_ratio", val=0.0)
        self.add_output("Slot_aspect_ratio1", val=0.0)
        self.add_output("Slot_aspect_ratio2", val=0.0)

        self.add_output("D_ratio", val=0.0)
        self.add_output("J_r", val=np.zeros(n_pc))
        self.add_output("L_sm", val=0.0)
        self.add_output("Q_r", val=0.0)
        self.add_output("R_R", val=0.0)
        self.add_output("b_r", val=0.0)
        self.add_output("b_tr", val=0.0)
        self.add_output("b_trmin", val=0.0)


# ----------------------------------------------------------------------------------------


class PMSG_Outer(GeneratorBase):
    """
    Estimates overall electromagnetic dimensions and Efficiency of PMSG -arms generator.

    Parameters
    ----------
    P_mech : float, [W]
        Shaft mechanical power
    N_c : float
        Number of turns per coil
    b : float
        Slot pole combination
    c : float
        Slot pole combination
    E_p : float, [V]
        Stator phase voltage
    h_yr : float, [m]
        rotor yoke height
    h_ys : float, [m]
        Yoke height
    h_sr : float, [m]
        Structural Mass
    h_ss : float, [m]
        Stator yoke height
    t_r : float, [m]
        Rotor disc thickness
    t_s : float, [m]
        Stator disc thickness
    y_sh : float, [m]
        Shaft deflection
    theta_sh : float, [rad]
        slope of shaft
    D_nose : float, [m]
        Nose outer diameter
    y_bd : float, [m]
        Deflection of the bedplate
    theta_bd : float, [rad]
        Slope at the bedplate
    u_allow_pcent : float
        Radial deflection as a percentage of air gap diameter
    y_allow_pcent : float
        Radial deflection as a percentage of air gap diameter
    z_allow_deg : float, [deg]
        Allowable torsional twist
    B_tmax : float, [T]
        Peak Teeth flux density

    Returns
    -------
    B_smax : float, [T]
        Peak Stator flux density
    B_symax : float, [T]
        Peak Stator flux density
    tau_p : float, [m]
        Pole pitch
    q : float, [N/m**2]
        Normal stress
    len_ag : float, [m]
        Air gap length
    h_t : float, [m]
        tooth height
    tau_s : float, [m]
        Slot pitch
    J_actual : float, [A/m**2]
        Current density
    T_e : float, [N*m]
        Electromagnetic torque
    twist_r : float, [deg]
        torsional twist
    twist_s : float, [deg]
        Stator torsional twist
    Structural_mass_rotor : float, [kg]
        Rotor mass (kg)
    Structural_mass_stator : float, [kg]
        Stator mass (kg)
    Mass_tooth_stator : float, [kg]
        Teeth and copper mass
    Mass_yoke_rotor : float, [kg]
        Rotor yoke mass
    Mass_yoke_stator : float, [kg]
        Stator yoke mass
    rotor_mass : float, [kg]
        Total rotor mass
    stator_mass : float, [kg]
        Total stator mass

    """

    def initialize(self):
        super(PMSG_Outer, self).initialize()

    def setup(self):
        super(PMSG_Outer, self).setup()

        n_pc = self.options["n_pc"]

        # PMSG_structrual inputs
        self.add_input("P_mech", units="W")
        self.add_input("N_c", 0.0)
        self.add_input("b", 0.0)
        self.add_input("c", 0.0)
        self.add_input("E_p", 0.0, units="V")
        self.add_input("h_yr", val=0.0, units="m")
        self.add_input("h_ys", val=0.0, units="m")
        self.add_input("h_sr", 0.0, units="m")
        self.add_input("h_ss", 0.0, units="m")
        self.add_input("t_r", 0.0, units="m")
        self.add_input("t_s", 0.0, units="m")
        self.add_input("y_sh", units="m")
        self.add_input("theta_sh", 0.0, units="rad")
        self.add_input("D_nose", 0.0, units="m")
        self.add_input("y_bd", units="m")
        self.add_input("theta_bd", 0.0, units="rad")
        self.add_input("u_allow_pcent", 0.0)
        self.add_input("y_allow_pcent", 0.0)
        self.add_input("z_allow_deg", 0.0, units="deg")

        # Magnetic loading
        self.add_input("B_tmax", 0.0, units="T")
        self.add_output("B_smax", val=0.0, units="T")
        self.add_output("B_symax", val=0.0, units="T")
        self.add_output("tau_p", 0.0, units="m")
        self.add_output("q", 0.0, units="N/m**2")
        self.add_output("len_ag", 0.0, units="m")

        # Stator design
        self.add_output("h_t", 0.0, units="m")
        self.add_output("tau_s", 0.0, units="m")

        # Electrical performance
        self.add_output("J_actual", val=np.zeros(n_pc), units="A/m**2")
        self.add_output("T_e", 0.0, units="N*m")

        # Material properties
        self.add_output("twist_r", 0.0, units="deg")
        self.add_output("twist_s", 0.0, units="deg")

        # Mass Outputs
        self.add_output("Structural_mass_rotor", 0.0, units="kg")
        self.add_output("Structural_mass_stator", 0.0, units="kg")
        self.add_output("Mass_tooth_stator", 0.0, units="kg")
        self.add_output("Mass_yoke_rotor", 0.0, units="kg")
        self.add_output("Mass_yoke_stator", 0.0, units="kg")
        self.add_output("rotor_mass", 0.0, units="kg")
        self.add_output("stator_mass", 0.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        rad_ag = float(inputs["rad_ag"][0])
        len_s = float(inputs["len_s"][0])
        b = float(inputs["b"][0])
        c = float(inputs["c"][0])
        h_m = float(inputs["h_m"][0])
        h_ys = float(inputs["h_ys"][0])
        h_yr = float(inputs["h_yr"][0])
        h_s = float(inputs["h_s"][0])
        h_ss = float(inputs["h_ss"][0])
        h_0 = float(inputs["h_0"][0])
        B_tmax = float(inputs["B_tmax"][0])
        E_p = float(inputs["E_p"][0])
        P_mech = float(inputs["P_mech"][0])
        P_av_v = float(inputs["machine_rating"][0])
        h_sr = float(inputs["h_sr"][0])
        t_r = float(inputs["t_r"][0])
        t_s = float(inputs["t_s"][0])
        R_sh = 0.5 * float(inputs["D_shaft"][0])
        R_no = 0.5 * float(inputs["D_nose"][0])
        y_sh = float(inputs["y_sh"][0])
        y_bd = float(inputs["y_bd"][0])
        rho_Fes = float(inputs["rho_Fes"][0])
        rho_Fe = float(inputs["rho_Fe"][0])
        sigma = float(inputs["sigma"][0])
        shaft_rpm = inputs["shaft_rpm"]

        # Grab constant values
        B_r = float(inputs["B_r"][0])
        E = float(inputs["E"][0])
        G = float(inputs["G"][0])
        P_Fe0e = float(inputs["P_Fe0e"][0])
        P_Fe0h = float(inputs["P_Fe0h"][0])
        cofi = float(inputs["cofi"][0])
        h_w = float(inputs["h_w"][0])
        k_fes = float(inputs["k_fes"][0])
        k_fills = float(inputs["k_fills"][0])
        m = int(discrete_inputs["m"])
        mu_0 = float(inputs["mu_0"][0])
        mu_r = float(inputs["mu_r"][0])
        p = inputs["p"]
        phi = float(inputs["phi"][0])
        ratio_mw2pp = float(inputs["ratio_mw2pp"][0])
        resist_Cu = float(inputs["resist_Cu"][0])
        v = float(inputs["v"][0])

        """
        #Assign values to universal constants
        B_r        = 1.279      # Tesla remnant flux density
        E          = 2e11       # N/m^2 young's modulus
        ratio      = 0.8        # ratio of magnet width to pole pitch(bm/self.tau_p)
        mu_0       = np.pi*4e-7 # permeability of free space
        mu_r       = 1.06       # relative permeability
        cofi       = 0.85       # power factor

        #Assign values to design constants
        h_0        = 0.005 # Slot opening height
        h_w        = 0.004 # Slot wedge height
        m          = 3     # no of phases
        #b_s_tau_s = 0.45   # slot width to slot pitch ratio
        k_fills     = 0.65  # Slot fill factor
        P_Fe0h     = 4	   # specific hysteresis losses W/kg @ 1.5 T
        P_Fe0e     = 1	   # specific hysteresis losses W/kg @ 1.5 T
        k_fes      = 0.8   # Iron fill factor

        #Assign values to universal constants
        phi        = 90*2*np.pi/360 # tilt angle (rotor tilt -90 degrees during transportation)
        v          = 0.3            # Poisson's ratio
        G          = 79.3e9
        """

        ######################## Electromagnetic design ###################################
        K_rad = len_s / (2 * rad_ag)  # Aspect ratio

        # Calculating air gap length
        dia = 2 * rad_ag  # air gap diameter
        len_ag = 0.001 * dia  # air gap length
        r_s = rad_ag - len_ag  # Stator outer radius
        b_so = 2 * len_ag  # Slot opening
        tau_p = np.pi * dia / (2 * p)  # pole pitch

        # Calculating winding factor
        Slot_pole = b / c
        S = Slot_pole * 2 * p * m
        testval = S / (m * np.gcd(int(S), int(p)))

        if float(np.round(testval, 3)).is_integer():
            k_w = winding_factor(int(S), b, c, int(p), m)
            b_m = ratio_mw2pp * tau_p  # magnet width
            alpha_p = np.pi / 2 * ratio_mw2pp
            tau_s = np.pi * (dia - 2 * len_ag) / S

            # Calculating Carter factor for statorand effective air gap length
            gamma = (
                4
                / np.pi
                * (
                    b_so / 2 / (len_ag + h_m / mu_r) * np.arctan(b_so / 2 / (len_ag + h_m / mu_r))
                    - np.log(np.sqrt(1 + (b_so / 2 / (len_ag + h_m / mu_r)) ** 2))
                )
            )
            k_C = tau_s / (tau_s - gamma * (len_ag + h_m / mu_r))  # carter coefficient
            g_eff = k_C * (len_ag + h_m / mu_r)

            # angular frequency in radians
            om_m = 2 * np.pi * shaft_rpm / 60
            om_e = p * om_m
            freq = om_e / 2 / np.pi  # outout frequency

            # Calculating magnetic loading
            B_pm1 = B_r * h_m / mu_r / (g_eff)
            B_g = B_r * h_m / (mu_r * g_eff) * (4 / np.pi) * np.sin(alpha_p)
            B_symax = B_pm1 * b_m / (2 * h_ys) * k_fes
            B_rymax = B_pm1 * b_m * k_fes / (2 * h_yr)
            b_t = B_pm1 * tau_s / B_tmax
            N_c = 2  # Number of turns per coil
            q = (B_g) ** 2 / 2 / mu_0

            # Stator winding length ,cross-section and resistance
            l_Cus = 2 * (len_s + np.pi / 4 * (tau_s + b_t))  # length of a turn

            # Calculating no-load voltage induced in the stator
            N_s = np.rint(E_p / (np.sqrt(2) * len_s * r_s * k_w * om_m * B_g))
            # Z              = P_av_v / (m*E_p)

            # Calculating leakage inductance in  stator
            V_1 = E_p / 1.1
            I_n = P_av_v / 3 / cofi / V_1
            J_s = 6.0
            A_Cuscalc = I_n / J_s
            A_slot = 2 * N_c * A_Cuscalc * (10**-6) / k_fills
            tau_s_new = np.pi * (dia - 2 * len_ag - 2 * h_w - 2 * h_0) / S
            b_s2 = tau_s_new - b_t  # Slot top width
            b_s1 = np.sqrt(b_s2**2 - 4 * np.pi * A_slot / S)
            b_s = (b_s1 + b_s2) * 0.5
            N_coil = 2 * S
            P_s = mu_0 * (h_s / 3 / b_s + h_w * 2 / (b_s2 + b_so) + h_0 / b_so)  # Slot permeance function
            L_ssigmas = S / 3 * 4 * N_c**2 * len_s * P_s  # slot leakage inductance
            L_ssigmaew = (
                N_coil * N_c**2 * mu_0 * tau_s * np.log((0.25 * np.pi * tau_s**2) / (0.5 * h_s * b_s))
            )  # end winding leakage inductance
            L_aa = 2 * np.pi / 3 * (N_c**2 * mu_0 * len_s * r_s / g_eff)
            L_m = L_aa
            L_ssigma = L_ssigmas + L_ssigmaew
            L_s = L_m + L_ssigma
            G_leak = np.abs((1.1 * E_p) ** 4 - (1 / 9) * (P_av_v * om_e * L_s) ** 2)

            # Calculating stator current and electrical loading
            I_s = np.sqrt(2 * (np.abs((E_p * 1.1) ** 2 - G_leak**0.5)) / (om_e * L_s) ** 2)
            A_1 = 6 * I_s * N_s / np.pi / dia
            J_actual = I_s / (A_Cuscalc * 2**0.5)
            L_Cus = N_s * l_Cus
            R_s = inputs["resist_Cu"] * (N_s) * l_Cus / (A_Cuscalc * (10**-6))
            B_smax = np.sqrt(2) * I_s * mu_0 / g_eff

            # Calculating Electromagnetically active mass
            wedge_area = (b_s * 0.5 - b_so * 0.5) * (2 * h_0 + h_w)
            V_Cus = m * L_Cus * (A_Cuscalc * (10**-6))  # copper volume
            h_t = h_s + h_w + h_0
            V_Fest = len_s * S * (b_t * (h_s + h_w + h_0) + wedge_area)  # volume of iron in stator tooth
            V_Fesy = (
                len_s
                * np.pi
                * ((rad_ag - len_ag - h_s - h_w - h_0) ** 2 - (rad_ag - len_ag - h_s - h_w - h_0 - h_ys) ** 2)
            )  # volume of iron in stator yoke
            V_Fery = len_s * np.pi * ((rad_ag + h_m + h_yr) ** 2 - (rad_ag + h_m) ** 2)
            Copper = V_Cus[-1] * inputs["rho_Copper"]
            M_Fest = V_Fest * rho_Fe  # Mass of stator tooth
            M_Fesy = V_Fesy * rho_Fe  # Mass of stator yoke
            M_Fery = V_Fery * rho_Fe  # Mass of rotor yoke
            Iron = M_Fest + M_Fesy + M_Fery
            mass_PM = 2 * np.pi * (rad_ag + h_m) * len_s * h_m * ratio_mw2pp * inputs["rho_PM"]

            # Calculating Losses
            ##1. Copper Losses
            K_R = 1.0  # Skin effect correction co-efficient
            P_Cu = m * (I_s / 2**0.5) ** 2 * R_s * K_R

            # Iron Losses ( from Hysteresis and eddy currents)
            P_Hyys = (
                M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))
            )  # Hysteresis losses in stator yoke
            P_Ftys = (
                M_Fesy * ((B_symax / 1.5) ** 2) * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
            )  # Eddy losses in stator yoke
            P_Fesynom = P_Hyys + P_Ftys
            P_Hyd = (
                M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))
            )  # Hysteresis losses in stator teeth
            P_Ftd = (
                M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
            )  # Eddy losses in stator teeth
            P_Festnom = P_Hyd + P_Ftd

            # Iron Losses ( from Hysteresis and eddy currents)
            P_Hyyr = (
                M_Fery * (B_rymax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))
            )  # Hysteresis losses in stator yoke
            P_Ftyr = (
                M_Fery * ((B_rymax / 1.5) ** 2) * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
            )  # Eddy losses in stator yoke
            P_Ferynom = P_Hyyr + P_Ftyr

            # additional stray losses due to leakage flux
            P_ad = 0.2 * (P_Hyys + P_Ftys + P_Hyd + P_Ftd + P_Hyyr + P_Ftyr)
            pFtm = 300  # specific magnet loss
            P_Ftm = pFtm * 2 * p * b_m * len_s
            Losses = P_Cu + P_Festnom + P_Fesynom + P_ad + P_Ftm + P_Ferynom
            gen_eff = (P_mech - Losses) / (P_mech)
            I_snom = gen_eff * (P_mech / m / E_p / cofi)  # rated current
            I_qnom = gen_eff * P_mech / (m * E_p)
            X_snom = om_e * (L_m + L_ssigma)
            T_e = np.pi * rad_ag**2 * len_s * 2 * sigma
            Stator = M_Fesy + M_Fest + Copper  # modified mass_stru_steel
            Rotor = M_Fery + mass_PM  # modified (N_r*(R_1-self.R_sh)*a_r*self.rho_Fes))

            Mass_tooth_stator = M_Fest + Copper
            Mass_yoke_rotor = M_Fery
            Mass_yoke_stator = M_Fesy
            R_out = (dia + 2 * h_m + 2 * h_yr + 2 * inputs["h_sr"]) * 0.5
            Losses = Losses
            generator_efficiency = gen_eff
        else:
            # Bad design
            for k in outputs.keys():
                outputs[k] = 1e30
            return

        ######################## Rotor inactive (structural) design ###################################
        # Radial deformation of rotor
        R = rad_ag + h_m
        L_r = len_s + t_r + 0.125
        constants_x_0 = shell_constant(R, t_r, L_r, 0, E, v)
        constants_x_L = shell_constant(R, t_r, L_r, L_r, E, v)
        f_d_denom1 = R / (E * ((R) ** 2 - (R_sh) ** 2)) * ((1 - v) * R**2 + (1 + v) * (R_sh) ** 2)
        f_d_denom2 = (
            t_r
            / (2 * constants_x_0[0] * (constants_x_0[1]) ** 3)
            * (
                constants_x_0[2] / (2 * constants_x_0[3]) * constants_x_0[4]
                - constants_x_0[5] / constants_x_0[3] * constants_x_0[6]
                - 0.5 * constants_x_0[7]
            )
        )
        f = q * (R) ** 2 * t_r / (E * (h_yr + h_sr) * (f_d_denom1 + f_d_denom2))
        u_d = (
            f
            / (constants_x_L[0] * (constants_x_L[1]) ** 3)
            * (
                (
                    constants_x_L[2] / (2 * constants_x_L[3]) * constants_x_L[4]
                    - constants_x_L[5] / constants_x_L[3] * constants_x_L[6]
                    - 0.5 * constants_x_L[7]
                )
            )
            + y_sh
        )

        u_ar = (q * (R) ** 2) / (E * (h_yr + h_sr)) - u_d
        u_ar = np.abs(u_ar + y_sh)
        u_allow_r = 2 * rad_ag / 1000 * inputs["u_allow_pcent"] / 100

        # axial deformation of rotor
        W_back_iron = plate_constant(R + h_sr + h_yr, R_sh, E, v, 0.5 * h_yr + R, t_r)
        W_ssteel = plate_constant(R + h_sr + h_yr, R_sh, E, v, h_yr + R + h_sr * 0.5, t_r)
        W_mag = plate_constant(R + h_sr + h_yr, R_sh, E, v, h_yr + R - 0.5 * h_m, t_r)
        W_ir = rho_Fe * gravity * np.sin(phi) * (L_r - t_r) * h_yr
        y_ai1r = (
            -W_ir
            * (0.5 * h_yr + R) ** 4
            / (R_sh * W_back_iron[0])
            * (W_back_iron[1] * W_back_iron[4] / W_back_iron[3] - W_back_iron[2])
        )
        W_sr = rho_Fes * gravity * np.sin(phi) * (L_r - t_r) * h_sr
        y_ai2r = (
            -W_sr
            * (h_sr * 0.5 + h_yr + R) ** 4
            / (R_sh * W_ssteel[0])
            * (W_ssteel[1] * W_ssteel[4] / W_ssteel[3] - W_ssteel[2])
        )
        W_m = np.sin(phi) * mass_PM / (2 * np.pi * (R - h_m * 0.5))
        y_ai3r = -W_m * (R - h_m) ** 4 / (R_sh * W_mag[0]) * (W_mag[1] * W_mag[4] / W_mag[3] - W_mag[2])
        w_disc_r = rho_Fes * gravity * np.sin(phi) * t_r
        a_ii = R + h_sr + h_yr
        r_oii = R_sh
        M_rb = (
            -w_disc_r
            * a_ii**2
            / W_ssteel[5]
            * (W_ssteel[6] * 0.5 / (a_ii * R_sh) * (a_ii**2 - r_oii**2) - W_ssteel[8])
        )
        Q_b = w_disc_r * 0.5 / R_sh * (a_ii**2 - r_oii**2)
        y_aiir = (
            M_rb * a_ii**2 / W_ssteel[0] * W_ssteel[1]
            + Q_b * a_ii**3 / W_ssteel[0] * W_ssteel[2]
            - w_disc_r * a_ii**4 / W_ssteel[0] * W_ssteel[7]
        )
        I = np.pi * 0.25 * (R**4 - (R_sh) ** 4)
        F_ecc = q * 2 * np.pi * K_rad * rad_ag**3
        M_ar = F_ecc * L_r * 0.5
        y_ar = (
            np.abs(y_ai1r + y_ai2r + y_ai3r)
            + y_aiir
            + (R + h_yr + h_sr) * inputs["theta_sh"]
            + M_ar * L_r**2 * 0 / (2 * E * I)
        )
        y_allow_r = L_r / 100 * inputs["y_allow_pcent"]

        # Torsional deformation of rotor
        J_dr = 0.5 * np.pi * ((R + h_yr + h_sr) ** 4 - R_sh**4)
        J_cylr = 0.5 * np.pi * ((R + h_yr + h_sr) ** 4 - R**4)
        twist_r = 180 / np.pi * inputs["rated_torque"] / G * (t_r / J_dr + (L_r - t_r) / J_cylr)
        Structural_mass_rotor = (
            rho_Fes
            * np.pi
            * (((R + h_yr + h_sr) ** 2 - (R_sh) ** 2) * t_r + ((R + h_yr + h_sr) ** 2 - (R + h_yr) ** 2) * len_s)
        )
        TC1 = inputs["rated_torque"] / (2 * np.pi * sigma)
        TC2r = (R + (h_yr + h_sr)) ** 2 * L_r

        ######################## Stator inactive (structural) design ###################################
        # Radial deformation of Stator
        L_stator = len_s + t_s + 0.1
        R_stator = rad_ag - len_ag - h_t - h_ys - h_ss
        constants_x_0 = shell_constant(R_stator, t_s, L_stator, 0, E, v)
        constants_x_L = shell_constant(R_stator, t_s, L_stator, L_stator, E, v)
        f_d_denom1 = (
            R_stator / (E * ((R_stator) ** 2 - (R_no) ** 2)) * ((1 - v) * R_stator**2 + (1 + v) * (R_no) ** 2)
        )
        f_d_denom2 = (
            t_s
            / (2 * constants_x_0[0] * (constants_x_0[1]) ** 3)
            * (
                constants_x_0[2] / (2 * constants_x_0[3]) * constants_x_0[4]
                - constants_x_0[5] / constants_x_0[3] * constants_x_0[6]
                - 0.5 * constants_x_0[7]
            )
        )
        f = q * (R_stator) ** 2 * t_s / (E * (h_ys + h_ss) * (f_d_denom1 + f_d_denom2))
        # TODO: Adds y_bd twice?
        u_as = (
            (q * (R_stator) ** 2) / (E * (h_ys + h_ss))
            - f
            * 0
            / (constants_x_L[0] * (constants_x_L[1]) ** 3)
            * (
                (
                    constants_x_L[2] / (2 * constants_x_L[3]) * constants_x_L[4]
                    - constants_x_L[5] / constants_x_L[3] * constants_x_L[6]
                    - 1 / 2 * constants_x_L[7]
                )
            )
            + y_bd
        )
        u_as = np.abs(u_as + y_bd)
        u_allow_s = 2 * rad_ag / 1000 * inputs["u_allow_pcent"] / 100

        # axial deformation of stator
        W_back_iron = plate_constant(R_stator + h_ss + h_ys + h_t, R_no, E, v, 0.5 * h_ys + h_ss + R_stator, t_s)
        W_ssteel = plate_constant(R_stator + h_ss + h_ys + h_t, R_no, E, v, R_stator + h_ss * 0.5, t_s)
        W_active = plate_constant(R_stator + h_ss + h_ys + h_t, R_no, E, v, R_stator + h_ss + h_ys + h_t * 0.5, t_s)
        W_is = rho_Fe * gravity * np.sin(phi) * (L_stator - t_s) * h_ys
        y_ai1s = (
            -W_is
            * (0.5 * h_ys + R_stator) ** 4
            / (R_no * W_back_iron[0])
            * (W_back_iron[1] * W_back_iron[4] / W_back_iron[3] - W_back_iron[2])
        )
        W_ss = rho_Fes * gravity * np.sin(phi) * (L_stator - t_s) * h_ss
        y_ai2s = (
            -W_ss
            * (h_ss * 0.5 + h_ys + R_stator) ** 4
            / (R_no * W_ssteel[0])
            * (W_ssteel[1] * W_ssteel[4] / W_ssteel[3] - W_ssteel[2])
        )
        W_cu = np.sin(phi) * Mass_tooth_stator / (2 * np.pi * (R_stator + h_ss + h_ys + h_t * 0.5))
        y_ai3s = (
            -W_cu
            * (R_stator + h_ss + h_ys + h_t * 0.5) ** 4
            / (R_no * W_active[0])
            * (W_active[1] * W_active[4] / W_active[3] - W_active[2])
        )
        w_disc_s = rho_Fes * gravity * np.sin(phi) * t_s
        a_ii = R_stator + h_ss + h_ys + h_t
        r_oii = R_no
        M_rb = (
            -w_disc_s
            * a_ii**2
            / W_ssteel[5]
            * (W_ssteel[6] * 0.5 / (a_ii * R_no) * (a_ii**2 - r_oii**2) - W_ssteel[8])
        )
        Q_b = w_disc_s * 0.5 / R_no * (a_ii**2 - r_oii**2)
        y_aiis = (
            M_rb * a_ii**2 / W_ssteel[0] * W_ssteel[1]
            + Q_b * a_ii**3 / W_ssteel[0] * W_ssteel[2]
            - w_disc_s * a_ii**4 / W_ssteel[0] * W_ssteel[7]
        )
        I = np.pi * 0.25 * (R_stator**4 - (R_no) ** 4)
        F_ecc = q * 2 * np.pi * K_rad * rad_ag**2
        M_as = F_ecc * L_stator * 0.5

        y_as = np.abs(
            y_ai1s + y_ai2s + y_ai3s + y_aiis + (R_stator + h_ys + h_ss + h_t) * inputs["theta_bd"]
        ) + M_as * L_stator**2 * 0 / (2 * E * I)
        y_allow_s = L_stator * inputs["y_allow_pcent"] / 100

        # Torsional deformation of stator
        J_ds = 0.5 * np.pi * ((R_stator + h_ys + h_ss + h_t) ** 4 - R_no**4)
        J_cyls = 0.5 * np.pi * ((R_stator + h_ys + h_ss + h_t) ** 4 - R_stator**4)
        twist_s = 180.0 / np.pi * inputs["rated_torque"] / G * (t_s / J_ds + (L_stator - t_s) / J_cyls)

        Structural_mass_stator = rho_Fes * (
            np.pi * ((R_stator + h_ys + h_ss + h_t) ** 2 - (R_no) ** 2) * t_s
            + np.pi * ((R_stator + h_ss) ** 2 - R_stator**2) * len_s
        )
        TC2s = (R_stator + h_ys + h_ss + h_t) ** 2 * L_stator

        ######################## Outputs ###################################

        outputs["K_rad"] = K_rad
        outputs["len_ag"] = len_ag
        outputs["tau_p"] = tau_p
        outputs["S"] = S
        outputs["tau_s"] = tau_s
        outputs["b_m"] = b_m
        outputs["f"] = freq
        outputs["B_pm1"] = B_pm1
        outputs["B_g"] = B_g
        outputs["B_symax"] = B_symax
        outputs["B_rymax"] = B_rymax
        outputs["b_t"] = b_t
        outputs["q"] = q
        outputs["N_s"] = N_s[-1]
        outputs["A_Cuscalc"] = A_Cuscalc
        outputs["b_s"] = b_s
        outputs["L_s"] = L_s
        outputs["J_s"] = J_s
        outputs["Slot_aspect_ratio"] = h_s / b_s
        outputs["I_s"] = I_s
        outputs["A_1"] = A_1
        outputs["J_actual"] = J_actual
        outputs["R_s"] = R_s
        outputs["B_smax"] = B_smax[-1]
        outputs["h_t"] = h_t
        outputs["Copper"] = Copper
        outputs["Iron"] = Iron
        outputs["mass_PM"] = mass_PM
        outputs["T_e"] = T_e
        outputs["Mass_tooth_stator"] = Mass_tooth_stator
        outputs["Mass_yoke_rotor"] = Mass_yoke_rotor
        outputs["Mass_yoke_stator"] = Mass_yoke_stator
        outputs["R_out"] = R_out
        outputs["Losses"] = Losses
        outputs["eandm_efficiency"] = np.maximum(eps, gen_eff)
        outputs["u_ar"] = u_ar
        outputs["u_allow_r"] = u_allow_r
        outputs["y_ar"] = y_ar
        outputs["y_allow_r"] = y_allow_r
        outputs["twist_r"] = twist_r
        outputs["Structural_mass_rotor"] = Structural_mass_rotor
        outputs["TC1"] = TC1
        outputs["TC2r"] = TC2r
        outputs["u_as"] = u_as
        outputs["u_allow_s"] = u_allow_s
        outputs["y_as"] = y_as
        outputs["y_allow_s"] = y_allow_s
        outputs["twist_s"] = twist_s
        outputs["Structural_mass_stator"] = Structural_mass_stator
        outputs["TC2s"] = TC2s
        outputs["Structural_mass"] = outputs["Structural_mass_rotor"] + outputs["Structural_mass_stator"]
        outputs["stator_mass"] = Stator + outputs["Structural_mass_stator"]
        outputs["rotor_mass"] = Rotor + outputs["Structural_mass_rotor"]
        outputs["generator_mass"] = Stator + Rotor + outputs["Structural_mass"]


# ----------------------------------------------------------------------------------------
class PMSG_Disc(GeneratorBase):
    """
    Estimates overall mass dimensions and Efficiency of PMSG-disc rotor generator.

    Parameters
    ----------
    tau_p : float, [m]
        Pole pitch self.tau_p
    t_d : float, [m]
        disc thickness
    h_yr : float, [m]
        rotor yoke height
    h_ys : float, [m]
        Yoke height

    Returns
    -------
    B_tmax : float, [T]
        Peak Teeth flux density
    B_smax : float, [T]
        Peak Stator Yoke flux density B_ymax
    B_symax : float, [T]
        Peak Stator Yoke flux density B_ymax
    E_p : float
        Stator phase voltage

    """

    def initialize(self):
        super(PMSG_Disc, self).initialize()

    def setup(self):
        super(PMSG_Disc, self).setup()
        self.add_input("tau_p", val=0.0, units="m")
        self.add_input("t_d", val=0.0, units="m")
        self.add_input("h_yr", val=0.0, units="m")
        self.add_input("h_ys", val=0.0, units="m")

        self.add_output("B_tmax", val=0.0, units="T")
        self.add_output("B_smax", val=0.0, units="T")
        self.add_output("B_symax", val=0.0, units="T")
        self.add_output("E_p", val=np.zeros(self.options["n_pc"]))

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        rad_ag = inputs["rad_ag"]
        len_s = inputs["len_s"]
        h_s = inputs["h_s"]
        tau_p = inputs["tau_p"]
        h_m = inputs["h_m"]
        h_ys = inputs["h_ys"]
        h_yr = inputs["h_yr"]
        machine_rating = inputs["machine_rating"]
        shaft_rpm = inputs["shaft_rpm"]
        Torque = inputs["rated_torque"]

        b_st = inputs["b_st"]
        d_s = inputs["d_s"]
        t_ws = inputs["t_ws"]
        n_s = inputs["n_s"]
        t_d = inputs["t_d"]

        R_sh = 0.5 * inputs["D_shaft"]
        rho_Fe = inputs["rho_Fe"]
        rho_Copper = inputs["rho_Copper"]
        rho_Fes = inputs["rho_Fes"]
        rho_PM = inputs["rho_PM"]

        # Grab constant values
        B_r = inputs["B_r"]
        E = inputs["E"]
        P_Fe0e = inputs["P_Fe0e"]
        P_Fe0h = inputs["P_Fe0h"]
        S_N = inputs["S_N"]
        alpha_p = inputs["alpha_p"]
        b_r_tau_r = inputs["b_r_tau_r"]
        b_ro = inputs["b_ro"]
        b_s_tau_s = inputs["b_s_tau_s"]
        b_so = inputs["b_so"]
        cofi = inputs["cofi"]
        freq = inputs["freq"]
        h_i = inputs["h_i"]
        h_sy0 = inputs["h_sy0"]
        h_w = inputs["h_w"]
        k_fes = inputs["k_fes"]
        k_fillr = inputs["k_fillr"]
        k_fills = inputs["k_fills"]
        k_s = inputs["k_s"]
        m = discrete_inputs["m"]
        mu_0 = inputs["mu_0"]
        mu_r = inputs["mu_r"]
        p = inputs["p"]
        phi = inputs["phi"]
        q1 = discrete_inputs["q1"]
        ratio_mw2pp = inputs["ratio_mw2pp"]
        resist_Cu = inputs["resist_Cu"]
        sigma = inputs["sigma"]
        v = inputs["v"]
        y_tau_p = inputs["y_tau_p"]
        y_tau_pr = inputs["y_tau_pr"]

        """
        # Assign values to universal constants
        B_r    = 1.2                 # remnant flux density (Tesla = kg / (s^2 A))
        E      = 2e11                # N / m^2 young's modulus
        sigma  = 40000.0             # shear stress assumed
        ratio_mw2pp  = 0.7           # ratio of magnet width to pole pitch(bm / self.tau_p)
        mu_0   = np.pi * 4e-7           # permeability of free space in m * kg / (s**2 * A**2)
        mu_r   = 1.06                # relative permeability (probably for neodymium magnets, often given as 1.05 - GNS)
        phi    = np.deg2rad(90)         # tilt angle (rotor tilt -90 degrees during transportation)
        cofi   = 0.85                # power factor

        # Assign values to design constants
        h_w     = 0.005              # wedge height
        y_tau_p = 1.0                # coil span to pole pitch
        m       = 3                  # no of phases
        q1      = 1                  # no of slots per pole per phase
        b_s_tau_s = 0.45             # slot width / slot pitch ratio
        k_fills = 0.65                # Slot fill factor
        P_Fe0h = 4.0                 # specific hysteresis losses W / kg @ 1.5 T
        P_Fe0e = 1.0                 # specific hysteresis losses W / kg @ 1.5 T
        resist_Cu = 1.8e-8 * 1.4        # resistivity of copper
        b_so  =  0.004               # stator slot opening
        k_fes = 0.9                  # useful iron stack length
        #T =   Torque
        v = 0.3                      # poisson's ratio
        """

        # back iron thickness for rotor and stator
        t_s = h_ys
        t = h_yr

        # Aspect ratio
        K_rad = len_s / (2 * rad_ag)  # aspect ratio

        ###################################################### Electromagnetic design#############################################

        dia_ag = 2 * rad_ag  # air gap diameter
        len_ag = 0.001 * dia_ag  # air gap length
        b_m = ratio_mw2pp * tau_p  # magnet width
        l_u = k_fes * len_s  # useful iron stack length
        l_e = len_s + 2 * 0.001 * rad_ag  # equivalent core length
        r_r = rad_ag - len_ag  # rotor radius
        p = np.round(np.pi * rad_ag / tau_p)  # pole pairs   Eq.(11)
        f = p * shaft_rpm / 60.0  # rpm to frequency (Hz)
        S = 2 * p * q1 * m  # Stator slots Eq.(12)
        N_conductors = S * 2
        N_s = N_conductors / 2 / m  # Stator turns per phase
        tau_s = np.pi * dia_ag / S  # slot pitch  Eq.(13)
        b_s = b_s_tau_s * tau_s  # slot width
        b_t = tau_s - b_s  # tooth width  Eq.(14)
        Slot_aspect_ratio = h_s / b_s
        alpha_p = np.pi / 2 * 0.7

        # Calculating Carter factor for stator and effective air gap length
        gamma = (
            4
            / np.pi
            * (
                b_so / 2 / (len_ag + h_m / mu_r) * np.arctan(b_so / 2 / (len_ag + h_m / mu_r))
                - np.log(np.sqrt(1 + (b_so / 2 / (len_ag + h_m / mu_r)) ** 2))
            )
        )
        k_C = tau_s / (tau_s - gamma * (len_ag + h_m / mu_r))  # carter coefficient
        g_eff = k_C * (len_ag + h_m / mu_r)

        # angular frequency in radians / sec
        om_m = 2 * np.pi * (shaft_rpm / 60.0)  # rpm to rad/s
        om_e = p * om_m / 2

        # Calculating magnetic loading
        B_pm1 = B_r * h_m / mu_r / g_eff
        B_g = B_r * h_m / mu_r / g_eff * (4.0 / np.pi) * np.sin(alpha_p)
        B_symax = B_g * b_m * l_e / (2 * h_ys * l_u)
        B_rymax = B_g * b_m * l_e / (2 * h_yr * len_s)
        B_tmax = B_g * tau_s / b_t

        k_wd = np.sin(np.pi / 6) / q1 / np.sin(np.pi / 6 / q1)  # winding factor
        L_t = len_s + 2 * tau_p

        # Stator winding length, cross-section and resistance
        l_Cus = 2 * N_s * (2 * tau_p + L_t)
        A_s = b_s * (h_s - h_w) * q1 * p  # m^2
        A_scalc = b_s * 1e3 * (h_s - h_w) * 1e3 * q1 * p  # mm^2
        A_Cus = A_s * k_fills / N_s
        A_Cuscalc = A_scalc * k_fills / N_s
        R_s = l_Cus * resist_Cu / A_Cus

        # Calculating leakage inductance in stator
        L_m = 2 * mu_0 * N_s**2 / p * m * k_wd**2 * tau_p * L_t / np.pi**2 / g_eff
        L_ssigmas = (
            2 * mu_0 * N_s**2 / p / q1 * len_s * ((h_s - h_w) / (3 * b_s) + h_w / b_so)
        )  # slot        leakage inductance
        L_ssigmaew = (
            2 * mu_0 * N_s**2 / p / q1 * len_s * 0.34 * len_ag * (l_e - 0.64 * tau_p * y_tau_p) / len_s
        )  # end winding leakage inductance
        L_ssigmag = (
            2 * mu_0 * N_s**2 / p / q1 * len_s * (5 * (len_ag * k_C / b_so) / (5 + 4 * (len_ag * k_C / b_so)))
        )  # tooth tip   leakage inductance
        L_ssigma = L_ssigmas + L_ssigmaew + L_ssigmag
        L_s = L_m + L_ssigma

        # Calculating no-load voltage induced in the stator and stator current
        E_p = np.sqrt(2) * N_s * L_t * rad_ag * k_wd * om_m * B_g

        Z = machine_rating / (m * E_p)
        G = np.maximum(0.0, E_p**2 - (om_e * L_s * Z) ** 2)

        # Calculating stator current and electrical loading
        I_s = np.sqrt(Z**2 + (((E_p - G**0.5) / (om_e * L_s) ** 2) ** 2))
        B_smax = np.sqrt(2) * I_s * mu_0 / g_eff
        J_s = I_s / A_Cuscalc
        A_1 = 6 * N_s * I_s / (np.pi * dia_ag)
        I_snom = machine_rating / (m * E_p * cofi)  # rated current
        I_qnom = machine_rating / (m * E_p)
        X_snom = om_e * (L_m + L_ssigma)

        # Calculating electromagnetically active mass

        V_Cus = m * l_Cus * A_Cus  # copper volume
        V_Fest = L_t * 2 * p * q1 * m * b_t * h_s  # volume of iron in stator tooth
        V_Fesy = L_t * np.pi * ((rad_ag + h_s + h_ys) ** 2 - (rad_ag + h_s) ** 2)  # volume of iron in stator yoke
        V_Fery = L_t * np.pi * ((r_r - h_m) ** 2 - (r_r - h_m - h_yr) ** 2)  # volume of iron in rotor yoke
        Copper = V_Cus * rho_Copper
        M_Fest = V_Fest * rho_Fe  # mass of stator tooth
        M_Fesy = V_Fesy * rho_Fe  # mass of stator yoke
        M_Fery = V_Fery * rho_Fe  # mass of rotor yoke
        Iron = M_Fest + M_Fesy + M_Fery

        # Calculating losses
        # 1.Copper losses
        K_R = 1.2  # Skin effect correction co - efficient
        P_Cu = m * I_snom**2 * R_s * K_R

        # Iron Losses ( from Hysteresis and eddy currents)
        P_Hyys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator yoke
        P_Ftys = (
            M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy       losses in stator yoke
        P_Fesynom = P_Hyys + P_Ftys
        P_Hyd = M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator teeth
        P_Ftd = (
            M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy       losses in stator teeth
        P_Festnom = P_Hyd + P_Ftd
        P_ad = 0.2 * (P_Hyys + P_Ftys + P_Hyd + P_Ftd)  # additional stray losses due to leakage flux
        pFtm = 300  # specific magnet loss
        P_Ftm = pFtm * 2 * p * b_m * len_s  # magnet losses

        Losses = P_Cu + P_Festnom + P_Fesynom + P_ad + P_Ftm
        gen_eff = machine_rating / (machine_rating + Losses)

        ################################################## Structural  Design ############################################################

        ## Structural deflection calculations
        # rotor structure
        R = rad_ag - len_ag - h_m - 0.5 * t  # mean radius of the rotor rim
        # l       = L_t  using L_t everywhere now
        b = R_sh  # Shaft radius (not used)
        R_b = R - 0.5 * t  # Inner radius of the rotor
        R_a = R + 0.5 * h_yr  # Outer radius of rotor yoke
        a = R - 0.5 * t  # same as R_b
        a_1 = R_b  # same as R_b, a
        c = R / 500
        u_allow_r = c / 20  # allowable radial deflection
        y_allow = 2 * L_t / 100  # allowable axial deflection
        R_1 = R - 0.5 * t  # inner radius of rotor cylinder # same as R_b, a, a_1 (not used)
        K = 4 * (np.sin(ratio_mw2pp * np.pi / 2)) / np.pi  # (not used)
        q3 = B_g**2 / (2 * mu_0)  # normal component of Maxwell's stress

        mass_PM = 2 * np.pi * (R + 0.5 * t) * L_t * h_m * ratio_mw2pp * rho_PM  # magnet mass
        mass_st_lam = rho_Fe * 2 * np.pi * R * L_t * h_yr  # mass of rotor yoke steel

        # Calculation of radial deflection of rotor
        # cylindrical shell function and circular plate parameters for disc rotor based on Table 11.2 Roark's formulas
        # lamb, C* and F* parameters are from Appendix A of McDonald

        lamb = (3 * (1 - v**2) / R_a**2 / h_yr**2) ** 0.25  # m^-1
        x1 = lamb * L_t  # no units

        # ----------------

        C_2 = chsPshc(x1)
        C_4 = chsMshc(x1)
        C_13 = chsMshc(x1)  # (not used)
        C_a2 = chsPshc(x1 * 0.5)
        F_2_x0 = chsPshc(lamb * 0)
        F_2_ls2 = chsPshc(x1 / 2)

        F_a4_x0 = chsMshc(lamb * (0))
        Fa4arg = np.pi / 180 * lamb * (0.5 * len_s - a)
        F_a4_ls2 = chsMshc(Fa4arg)

        # print('pmsg_disc: F_a4_ls2, Fa4arg, lamb, len_s, a ', F_a4_ls2, Fa4arg, lamb, len_s, a)
        # if np.isnan(F_a4_ls2):
        #    sys.stderr.write('*** pmsg_discSE error: F_a4_ls2 is nan\n')

        # C_2 = np.cosh(x1) * np.sin(x1) + np.sinh(x1) * np.cos(x1)
        C_3 = np.sinh(x1) * np.sin(x1)
        # C_4 = np.cosh(x1) * np.sin(x1) - np.sinh(x1) * np.cos(x1)
        C_11 = (np.sinh(x1)) ** 2 - (np.sin(x1)) ** 2
        # C_13 = np.cosh(x1) * np.sinh(x1) - np.cos(x1) * np.sin(x1) # (not used)
        C_14 = np.sinh(x1) ** 2 + np.sin(x1) ** 2  # (not used)
        C_a1 = np.cosh(x1 * 0.5) * np.cos(x1 * 0.5)
        # C_a2 = np.cosh(x1 * 0.5) * np.sin(x1 * 0.5) + np.sinh(x1 * 0.5) * np.cos(x1 * 0.5)
        F_1_x0 = np.cosh(lamb * 0) * np.cos(lamb * 0)
        F_1_ls2 = np.cosh(lamb * 0.5 * len_s) * np.cos(lamb * 0.5 * len_s)
        # F_2_x0  = np.cosh(lamb * 0) * np.sin(lamb * 0) + np.sinh(lamb * 0) * np.cos(lamb * 0)
        # F_2_ls2 = np.cosh(x1 / 2)   * np.sin(x1 / 2)   + np.sinh(x1 / 2)   * np.cos(x1 / 2)

        if len_s < 2 * a:
            a = len_s / 2
        else:
            a = len_s * 0.5 - 1

        # F_a4_x0 = np.cosh(lamb * (0)) * np.sin(lamb * (0)) \
        #        - np.sinh(lamb * (0)) * np.cos(lamb * (0))
        # F_a4_ls2 = np.cosh(np.pi / 180 * lamb * (0.5 * len_s - a)) * np.sin(np.pi / 180 * lamb * (0.5 * len_s - a)) \
        #         - np.sinh(np.pi / 180 * lamb * (0.5 * len_s - a)) * np.cos(np.pi / 180 * lamb * (0.5 * len_s - a))
        """
        Where did the np.pi/180 factor (conversion to radians) come from?
          lamb is m^-1
          0.5*len_s - a is m
        """

        # ----------------

        D_r = E * h_yr**3 / (12 * (1 - v**2))
        D_ax = E * t_d**3 / (12 * (1 - v**2))

        # Radial deflection analytical model from McDonald's thesis defined in parts
        Part_1 = R_b * ((1 - v) * R_b**2 + (1 + v) * R_sh**2) / (R_b**2 - R_sh**2) / E
        Part_2 = (C_2 * C_a2 - 2 * C_3 * C_a1) / 2 / C_11
        Part_3 = (C_3 * C_a2 - C_4 * C_a1) / C_11
        Part_4 = 0.25 / D_r / lamb**3
        Part_5 = q3 * R_b**2 / (E * (R_a - R_b))
        f_d = Part_5 / (Part_1 - t_d * (Part_4 * Part_2 * F_2_ls2 - Part_3 * 2 * Part_4 * F_1_ls2 - Part_4 * F_a4_ls2))
        fr = f_d * t_d
        u_ar = abs(
            Part_5
            + fr
            / (2 * D_r * lamb**3)
            * (
                (-F_1_x0 / C_11) * (C_3 * C_a2 - C_4 * C_a1)
                + (F_2_x0 / 2 / C_11) * (C_2 * C_a2 - 2 * C_3 * C_a1)
                - F_a4_x0 / 2
            )
        )

        # Calculation of Axial deflection of rotor
        W = (
            0.5 * gravity * np.sin(phi) * ((L_t - t_d) * h_yr * rho_Fes)
        )  # uniform annular line load acting on rotor cylinder assumed as an annular plate
        w = rho_Fes * gravity * np.sin(phi) * t_d  # disc assumed as plate with a uniformly distributed pressure between
        a_i = R_sh

        # Flat circular plate constants according to Roark's table 11.2
        C_2p = 0.25 * (1 - (((R_sh / R) ** 2) * (1 + (2 * np.log(R / R_sh)))))
        C_3p = (R_sh / 4 / R) * ((1 + (R_sh / R) ** 2) * np.log(R / R_sh) + (R_sh / R) ** 2 - 1)
        C_6 = (R_sh / 4 / R_a) * ((R_sh / R_a) ** 2 - 1 + 2 * np.log(R_a / R_sh))
        C_5 = 0.5 * (1 - (R_sh / R) ** 2)
        C_8 = 0.5 * (1 + v + (1 - v) * ((R_sh / R) ** 2))
        C_9 = (R_sh / R) * (0.5 * (1 + v) * np.log(R / R_sh) + (1 - v) / 4 * (1 - (R_sh / R) ** 2))

        # Flat circular plate loading constants
        L_11 = (
            1
            + 4 * (R_sh / a_1) ** 2
            - 5 * (R_sh / a_1) ** 4
            - 4 * ((R_sh / a_1) ** 2) * np.log(a_1 / R_sh) * (2 + (R_sh / a_1) ** 2)
        ) / 64
        L_14 = (1 - (R_sh / R_b) ** 4 - 4 * (R_sh / R_b) ** 2 * np.log(R_b / R_sh)) / 16
        y_ai = (
            -W * (a_1**3) * (C_2p * (C_6 * a_1 / R_sh - C_6) / C_5 - a_1 * C_3p / R_sh + C_3p) / D_ax
        )  # Axial deflection of plate due to deflection of an annular plate with a uniform annular line load

        # Axial Deflection due to uniformaly distributed pressure load
        M_rb = -w * R**2 * (C_6 * (R**2 - R_sh**2) * 0.5 / R / R_sh - L_14) / C_5
        Q_b = w * 0.5 * (R**2 - R_sh**2) / R_sh
        y_aii = M_rb * R_a**2 * C_2p / D_ax + Q_b * R_a**3 * C_3p / D_ax - w * R_a**4 * L_11 / D_ax

        y_ar = abs(y_ai + y_aii)

        z_allow_r = np.deg2rad(0.05 * R)  # allowable torsional deflection of rotor

        # stator structure deflection calculation

        R_out = R / 0.995 + h_s + h_ys

        a_s = (b_st * d_s) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws))  # cross-sectional area of stator armms
        A_st = L_t * t_s  # cross-sectional area of rotor cylinder
        N_st = np.round(n_s)
        theta_s = np.pi * 1 / N_st  # half angle between spokes
        I_st = L_t * t_s**3 / 12  # second moment of area of stator cylinder
        I_arm_axi_s = (
            (b_st * d_s**3) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws) ** 3)
        ) / 12  # second moment of area of stator arm
        I_arm_tor_s = (
            (d_s * b_st**3) - ((d_s - 2 * t_ws) * (b_st - 2 * t_ws) ** 3)
        ) / 12  # second moment of area of rotot arm w.r.t torsion
        R_st = rad_ag + h_s + h_ys * 0.5
        k_2 = np.sqrt(I_st / A_st)  # radius of gyration
        b_allow_s = 2 * np.pi * R_sh / N_st
        m2 = (k_2 / R_st) ** 2
        c1 = R_st / 500
        R_1s = R_st - t_s * 0.5
        d_se = dia_ag + 2 * (h_ys + h_s + h_w)  # stator outer diameter

        # Calculation of radial deflection of stator
        Numers = R_st**3 * (
            (0.25 * (np.sin(theta_s) - (theta_s * np.cos(theta_s))) / (np.sin(theta_s)) ** 2)
            - (0.5 / np.sin(theta_s))
            + (0.5 / theta_s)
        )
        Povs = ((theta_s / (np.sin(theta_s)) ** 2) + 1 / np.tan(theta_s)) * (
            (0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st)
        )
        Qovs = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
        Lovs = (R_1s - R_sh) * 0.5 / a_s
        Denoms = I_st * (Povs - Qovs + Lovs)

        u_as = (q3 * R_st**2 / E / t_s) * (1 + Numers / Denoms)

        # Calculation of axial deflection of stator
        mass_st_lam_s = M_Fest + np.pi * L_t * rho_Fe * ((R_st + 0.5 * h_ys) ** 2 - (R_st - 0.5 * h_ys) ** 2)
        W_is = (
            0.5 * gravity * np.sin(phi) * (rho_Fes * L_t * d_s**2)
        )  # length of stator arm beam at which self-weight acts
        W_iis = (
            gravity * np.sin(phi) * (mass_st_lam_s + V_Cus * rho_Copper) / 2 / N_st
        )  # weight of stator cylinder and teeth
        w_s = rho_Fes * gravity * np.sin(phi) * a_s * N_st  # uniformly distributed load of the arms

        l_is = R_st - R_sh  # distance at which the weight of the stator cylinder acts
        l_iis = l_is  # distance at which the weight of the stator cylinder acts
        l_iiis = l_is  # distance at which the weight of the stator cylinder acts
        u_allow_s = c1 / 20

        X_comp1 = (
            W_is * l_is**3 / 12 / E / I_arm_axi_s
        )  # deflection component due to stator arm beam at which self-weight acts
        X_comp2 = W_iis * l_iis**4 / 24 / E / I_arm_axi_s  # deflection component due to 1/nth of stator cylinder
        X_comp3 = w_s * l_iiis**4 / 24 / E / I_arm_axi_s  # deflection component due to weight of arms
        y_as = X_comp1 + X_comp2 + X_comp3  # axial deflection

        # Stator circumferential deflection
        z_allow_s = np.deg2rad(0.05 * R_st)  # allowable torsional deflection
        z_as = (
            2 * np.pi * (R_st + 0.5 * t_s) * L_t / (2 * N_st) * sigma * (l_is + 0.5 * t_s) ** 3 / (3 * E * I_arm_tor_s)
        )

        mass_stru_steel = 2 * (N_st * (R_1s - R_sh) * a_s * rho_Fes)

        TC1 = Torque * 1.0 / (2 * np.pi * sigma)  # Torque / shear stress
        TC2r = R**2 * L_t  # Evaluating Torque constraint for rotor
        TC2s = R_st**2 * L_t  # Evaluating Torque constraint for stator
        Structural_mass = mass_stru_steel + (np.pi * (R**2 - R_sh**2) * t_d * rho_Fes)

        Mass = Structural_mass + Iron + Copper + mass_PM

        outputs["B_tmax"] = B_tmax
        outputs["B_rymax"] = B_rymax
        outputs["B_symax"] = B_symax
        outputs["B_smax"] = B_smax[-1]
        outputs["B_pm1"] = B_pm1
        outputs["B_g"] = B_g
        outputs["N_s"] = N_s
        outputs["b_s"] = b_s

        outputs["b_t"] = b_t
        outputs["A_Cuscalc"] = A_Cuscalc
        outputs["b_m"] = b_m
        outputs["E_p"] = E_p
        outputs["f"] = f

        outputs["I_s"] = I_s
        outputs["R_s"] = R_s
        outputs["L_s"] = L_s
        outputs["A_1"] = A_1
        outputs["J_s"] = J_s
        outputs["Losses"] = Losses

        outputs["K_rad"] = K_rad
        outputs["eandm_efficiency"] = np.maximum(eps, gen_eff)
        outputs["S"] = S
        outputs["Slot_aspect_ratio"] = Slot_aspect_ratio
        outputs["Copper"] = Copper
        outputs["Iron"] = Iron
        outputs["u_ar"] = u_ar
        outputs["y_ar"] = y_ar

        outputs["u_as"] = u_as
        outputs["y_as"] = y_as
        outputs["z_as"] = z_as
        outputs["u_allow_r"] = u_allow_r
        outputs["u_allow_s"] = u_allow_s

        outputs["y_allow_r"] = outputs["y_allow_s"] = y_allow
        outputs["z_allow_s"] = z_allow_s
        outputs["z_allow_r"] = z_allow_r
        outputs["b_allow_s"] = b_allow_s
        outputs["TC1"] = TC1

        outputs["TC2r"] = TC2r
        outputs["TC2s"] = TC2s
        outputs["R_out"] = R_out
        outputs["Structural_mass"] = Structural_mass
        outputs["generator_mass"] = Mass
        outputs["mass_PM"] = mass_PM


# ----------------------------------------------------------------------------------------
class PMSG_Arms(GeneratorBase):
    """
    Estimates overall mass dimensions and Efficiency of PMSG-disc rotor generator.

    Parameters
    ----------
    b_arm : float, [m]
        arm width
    tau_p : float, [m]
        Pole pitch self.tau_p
    h_yr : float, [m]
        rotor yoke height
    h_ys : float, [m]
        Yoke height

    Returns
    -------
    B_tmax : float, [T]
        Peak Teeth flux density
    B_smax : float, [T]
        Peak Stator Yoke flux density B_ymax
    B_symax : float, [T]
        Peak Stator Yoke flux density B_ymax
    E_p : float
        Stator phase voltage

    """

    def initialize(self):
        super(PMSG_Arms, self).initialize()

    def setup(self):
        super(PMSG_Arms, self).setup()
        self.add_input("b_arm", val=0.0, units="m")
        self.add_input("tau_p", val=0.0, units="m")
        self.add_input("h_yr", val=0.0, units="m")
        self.add_input("h_ys", val=0.0, units="m")

        self.add_output("B_tmax", val=0.0, units="T")
        self.add_output("B_smax", val=0.0, units="T")
        self.add_output("B_symax", val=0.0, units="T")
        self.add_output("E_p", val=np.zeros(self.options["n_pc"]))

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        # r_s               = inputs['r_s']
        rad_ag = inputs["rad_ag"]
        len_s = inputs["len_s"]
        h_s = inputs["h_s"]
        tau_p = inputs["tau_p"]
        h_m = inputs["h_m"]
        h_ys = inputs["h_ys"]
        h_yr = inputs["h_yr"]
        machine_rating = inputs["machine_rating"]
        shaft_rpm = inputs["shaft_rpm"]
        Torque = inputs["rated_torque"]

        b_st = inputs["b_st"]
        d_s = inputs["d_s"]
        t_ws = inputs["t_ws"]
        n_r = inputs["n_r"]
        n_s = inputs["n_s"]
        b_r = inputs["b_arm"]
        d_r = inputs["d_r"]
        t_wr = inputs["t_wr"]

        R_sh = 0.5 * inputs["D_shaft"]
        rho_Fe = inputs["rho_Fe"]
        rho_Copper = inputs["rho_Copper"]
        rho_Fes = inputs["rho_Fes"]
        rho_PM = inputs["rho_PM"]

        # Grab constant values
        B_r = inputs["B_r"]
        E = inputs["E"]
        P_Fe0e = inputs["P_Fe0e"]
        P_Fe0h = inputs["P_Fe0h"]
        S_N = inputs["S_N"]
        alpha_p = inputs["alpha_p"]
        b_r_tau_r = inputs["b_r_tau_r"]
        b_ro = inputs["b_ro"]
        b_s_tau_s = inputs["b_s_tau_s"]
        b_so = inputs["b_so"]
        cofi = inputs["cofi"]
        freq = inputs["freq"]
        h_i = inputs["h_i"]
        h_sy0 = inputs["h_sy0"]
        h_w = inputs["h_w"]
        k_fes = inputs["k_fes"]
        k_fillr = inputs["k_fillr"]
        k_fills = inputs["k_fills"]
        k_s = inputs["k_s"]
        m = discrete_inputs["m"]
        mu_0 = inputs["mu_0"]
        mu_r = inputs["mu_r"]
        p = inputs["p"]
        phi = inputs["phi"]
        q1 = discrete_inputs["q1"]
        ratio_mw2pp = inputs["ratio_mw2pp"]
        resist_Cu = inputs["resist_Cu"]
        sigma = inputs["sigma"]
        v = inputs["v"]
        y_tau_p = inputs["y_tau_p"]
        y_tau_pr = inputs["y_tau_pr"]

        """
        # Assign values to universal constants
        B_r    = 1.2                 # Tesla remnant flux density
        E      = 2e11                # N / m^2 young's modulus
        sigma  = 40e3                # shear stress assumed (yield strength of ?? steel, in psi - GNS)
        ratio_mw2pp  = 0.7                 # ratio of magnet width to pole pitch(bm / tau_p)
        mu_0   = np.pi * 4e-7           # permeability of free space in m * kg / (s**2 * A**2)
        mu_r   = 1.06                # relative permeability (probably for neodymium magnets, often given as 1.05 - GNS)
        phi    = np.deg2rad(90)         # tilt angle (rotor tilt -90 degrees during transportation)
        cofi   = 0.85                # power factor

        # Assign values to design constants
        h_w       = 0.005            # Slot wedge height
        h_i       = 0.001            # coil insulation thickness
        y_tau_p   = 1                # Coil span to pole pitch
        m         = 3                # no of phases
        q1        = 1                # no of slots per pole per phase
        b_s_tau_s = 0.45             # slot width to slot pitch ratio
        k_fills    = 0.65             # Slot fill factor
        P_Fe0h    = 4                # specific hysteresis losses W / kg @ 1.5 T
        P_Fe0e    = 1                # specific eddy losses W / kg @ 1.5 T
        resist_Cu    = 1.8e-8 * 1.4     # Copper resisitivty
        k_fes     = 0.9              # Stator iron fill factor per Grauers
        b_so      = 0.004            # Slot opening
        alpha_p   = np.pi / 2 * 0.7
        """

        # back iron thickness for rotor and stator
        t_s = h_ys
        t = h_yr

        ###################################################### Electromagnetic design#############################################

        K_rad = len_s / (2 * rad_ag)  # Aspect ratio
        # T     = Torque                    # rated torque
        l_u = k_fes * len_s  # useful iron stack length
        We = tau_p
        l_b = 2 * tau_p  # end winding length
        l_e = len_s + 2 * 0.001 * rad_ag  # equivalent core length
        b_m = 0.7 * tau_p  # magnet width

        # Calculating air gap length
        dia_ag = 2 * rad_ag  # air gap diameter
        len_ag = 0.001 * dia_ag  # air gap length
        r_m = rad_ag + h_ys + h_s  # magnet radius
        r_r = rad_ag - len_ag  # rotor radius

        p = np.round(np.pi * dia_ag / (2 * tau_p))  # pole pairs
        f = shaft_rpm * p / 60.0  # outout frequency rpm to Hz
        S = 2 * p * q1 * m  # Stator slots
        N_conductors = S * 2
        N_s = N_conductors / (2 * m)  # Stator turns per phase
        tau_s = np.pi * dia_ag / S  # Stator slot pitch
        b_s = b_s_tau_s * tau_s  # slot width
        b_t = tau_s - b_s  # tooth width
        Slot_aspect_ratio = h_s / b_s

        # Calculating Carter factor for stator and effective air gap length
        ahm = len_ag + h_m / mu_r
        ba = b_so / (2 * ahm)
        gamma = 4 / np.pi * (ba * np.arctan(ba) - np.log(np.sqrt(1 + ba**2)))
        k_C = tau_s / (tau_s - gamma * ahm)  # carter coefficient
        g_eff = k_C * ahm

        # angular frequency in radians
        om_m = 2 * np.pi * shaft_rpm / 60.0  # rpm to radians per second
        om_e = p * om_m / 2  # electrical output frequency (Hz)

        # Calculating magnetic loading
        B_pm1 = B_r * h_m / mu_r / g_eff
        B_g = B_r * h_m / mu_r / g_eff * (4 / np.pi) * np.sin(alpha_p)
        B_symax = B_g * b_m * l_e / (2 * h_ys * l_u)
        B_rymax = B_g * b_m * l_e / (2 * h_yr * len_s)
        B_tmax = B_g * tau_s / b_t

        # Calculating winding factor
        k_wd = np.sin(np.pi / 6) / q1 / np.sin(np.pi / 6 / q1)

        L_t = len_s + 2 * tau_p  # overall stator len w/end windings - should be tau_s???

        # l = L_t                          # length - now using L_t everywhere

        # Stator winding length, cross-section and resistance
        l_Cus = 2 * N_s * (2 * tau_p + L_t)
        A_s = b_s * (h_s - h_w) * q1 * p
        A_scalc = b_s * 1000 * (h_s - h_w) * 1000 * q1 * p
        A_Cus = A_s * k_fills / N_s
        A_Cuscalc = A_scalc * k_fills / N_s
        R_s = l_Cus * resist_Cu / A_Cus

        # Calculating leakage inductance in  stator
        L_m = 2 * mu_0 * N_s**2 / p * m * k_wd**2 * tau_p * L_t / np.pi**2 / g_eff
        L_ssigmas = (
            2 * mu_0 * N_s**2 / p / q1 * len_s * ((h_s - h_w) / (3 * b_s) + h_w / b_so)
        )  # slot leakage inductance
        L_ssigmaew = (
            2 * mu_0 * N_s**2 / p / q1 * len_s * 0.34 * len_ag * (l_e - 0.64 * tau_p * y_tau_p) / len_s
        )  # end winding leakage inductance
        L_ssigmag = (
            2 * mu_0 * N_s**2 / p / q1 * len_s * (5 * (len_ag * k_C / b_so) / (5 + 4 * (len_ag * k_C / b_so)))
        )  # tooth tip leakage inductance
        L_ssigma = L_ssigmas + L_ssigmaew + L_ssigmag
        L_s = L_m + L_ssigma

        # Calculating no-load voltage induced in the stator
        E_p = 2 * N_s * L_t * rad_ag * k_wd * om_m * B_g / np.sqrt(2)

        Z = machine_rating / (m * E_p)
        G = np.maximum(0.0, E_p**2 - (om_e * L_s * Z) ** 2)

        # Calculating stator current and electrical loading
        is2 = Z**2 + (((E_p - G**0.5) / (om_e * L_s) ** 2) ** 2)
        I_s = np.sqrt(Z**2 + (((E_p - G**0.5) / (om_e * L_s) ** 2) ** 2))
        J_s = I_s / A_Cuscalc
        A_1 = 6 * N_s * I_s / (np.pi * dia_ag)
        I_snom = machine_rating / (m * E_p * cofi)  # rated current
        I_qnom = machine_rating / (m * E_p)
        X_snom = om_e * (L_m + L_ssigma)

        B_smax = np.sqrt(2) * I_s * mu_0 / g_eff

        # Calculating Electromagnetically active mass

        V_Cus = m * l_Cus * A_Cus  # copper volume
        V_Fest = L_t * 2 * p * q1 * m * b_t * h_s  # volume of iron in stator tooth

        V_Fesy = L_t * np.pi * ((rad_ag + h_s + h_ys) ** 2 - (rad_ag + h_s) ** 2)  # volume of iron in stator yoke
        V_Fery = L_t * np.pi * ((r_r - h_m) ** 2 - (r_r - h_m - h_yr) ** 2)
        Copper = V_Cus * rho_Copper

        M_Fest = V_Fest * rho_Fe  # Mass of stator tooth
        M_Fesy = V_Fesy * rho_Fe  # Mass of stator yoke
        M_Fery = V_Fery * rho_Fe  # Mass of rotor yoke
        Iron = M_Fest + M_Fesy + M_Fery

        # Calculating Losses
        ##1. Copper Losses

        K_R = 1.2  # Skin effect correction co-efficient
        P_Cu = m * I_snom**2 * R_s * K_R

        # Iron Losses ( from Hysteresis and eddy currents)
        P_Hyys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator yoke
        P_Ftys = (
            M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy       losses in stator yoke
        P_Fesynom = P_Hyys + P_Ftys

        P_Hyd = M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator teeth
        P_Ftd = (
            M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy       losses in stator teeth
        P_Festnom = P_Hyd + P_Ftd

        # additional stray losses due to leakage flux
        P_ad = 0.2 * (P_Hyys + P_Ftys + P_Hyd + P_Ftd)
        pFtm = 300  # specific magnet loss
        P_Ftm = pFtm * 2 * p * b_m * len_s

        Losses = P_Cu + P_Festnom + P_Fesynom + P_ad + P_Ftm
        gen_eff = machine_rating / (machine_rating + Losses)

        #################################################### Structural  Design ############################################################

        ## Deflection Calculations ##
        # rotor structure calculations

        a_r = (b_r * d_r) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr))  # cross-sectional area of rotor arms
        A_r = L_t * t  # cross-sectional area of rotor cylinder
        N_r = np.round(n_r)  # rotor arms
        theta_r = np.pi * 1 / N_r  # half angle between spokes
        I_r = L_t * t**3 / 12  # second moment of area of rotor cylinder
        I_arm_axi_r = (
            (b_r * d_r**3) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr) ** 3)
        ) / 12  # second moment of area of rotor arm
        I_arm_tor_r = (
            (d_r * b_r**3) - ((d_r - 2 * t_wr) * (b_r - 2 * t_wr) ** 3)
        ) / 12  # second moment of area of rotot arm w.r.t torsion
        R = rad_ag - len_ag - h_m - 0.5 * t  # Rotor mean radius
        c = R / 500
        u_allow_r = c / 20  # allowable radial deflection
        R_1 = R - t * 0.5  # inner radius of rotor cylinder
        k_1 = np.sqrt(I_r / A_r)  # radius of gyration
        m1 = (k_1 / R) ** 2
        l_ir = R  # length of rotor arm beam at which rotor cylinder acts
        l_iir = R_1

        b_allow_r = 2 * np.pi * R_sh / N_r  # allowable circumferential arm dimension for rotor
        q3 = B_g**2 / 2 / mu_0  # normal component of Maxwell stress
        mass_PM = 2 * np.pi * (R + 0.5 * t) * L_t * h_m * ratio_mw2pp * rho_PM  # magnet mass

        # Calculating radial deflection of the rotor
        Numer = R**3 * (
            (0.25 * (np.sin(theta_r) - (theta_r * np.cos(theta_r))) / (np.sin(theta_r)) ** 2)
            - (0.5 / np.sin(theta_r))
            + (0.5 / theta_r)
        )
        Pov = ((theta_r / (np.sin(theta_r)) ** 2) + 1 / np.tan(theta_r)) * ((0.25 * R / A_r) + (0.25 * R**3 / I_r))
        Qov = R**3 / (2 * I_r * theta_r * (m1 + 1))
        Lov = (R_1 - R_sh) / a_r
        Denom = I_r * (Pov - Qov + Lov)  # radial deflection % rotor
        u_ar = (q3 * R**2 / E / t) * (1 + Numer / Denom)

        # Calculating axial deflection of the rotor under its own weight

        w_r = rho_Fes * gravity * np.sin(phi) * a_r * N_r  # uniformly distributed load of the weight of the rotor arm
        mass_st_lam = rho_Fe * 2 * np.pi * R * L_t * h_yr  # mass of rotor yoke steel
        W = gravity * np.sin(phi) * (mass_st_lam / N_r + mass_PM / N_r)  # weight of 1/nth of rotor cylinder

        y_a1 = W * l_ir**3 / 12 / E / I_arm_axi_r  # deflection from weight component of back iron
        y_a2 = w_r * l_iir**4 / 24 / E / I_arm_axi_r  # deflection from weight component of the arms
        y_ar = y_a1 + y_a2  # axial deflection

        y_allow = 2 * L_t / 100  # allowable axial deflection

        # Calculating # circumferential deflection of the rotor

        z_allow_r = np.deg2rad(0.05 * R)  # allowable torsional deflection
        z_ar = (
            (2 * np.pi * (R - 0.5 * t) * L_t / N_r) * sigma * (l_ir - 0.5 * t) ** 3 / 3 / E / I_arm_tor_r
        )  # circumferential deflection

        val_str_rotor = mass_PM + (mass_st_lam + (N_r * (R_1 - R_sh) * a_r * rho_Fes))  # rotor mass

        # stator structure deflection calculation
        a_s = (b_st * d_s) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws))  # cross-sectional area of stator armms
        A_st = L_t * t_s  # cross-sectional area of stator cylinder
        N_st = np.round(n_s)  # stator arms
        theta_s = np.pi * 1 / N_st  # half angle between spokes
        I_st = L_t * t_s**3 / 12  # second moment of area of stator cylinder
        k_2 = np.sqrt(I_st / A_st)  # radius of gyration
        I_arm_axi_s = (
            (b_st * d_s**3) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws) ** 3)
        ) / 12  # second moment of area of stator arm
        I_arm_tor_s = (
            (d_s * b_st**3) - ((d_s - 2 * t_ws) * (b_st - 2 * t_ws) ** 3)
        ) / 12  # second moment of area of rotot arm w.r.t torsion
        R_st = rad_ag + h_s + h_ys * 0.5  # stator cylinder mean radius
        R_1s = R_st - t_s * 0.5  # inner radius of stator cylinder, m
        m2 = (k_2 / R_st) ** 2
        d_se = dia_ag + 2 * (h_ys + h_s + h_w)  # stator outer diameter

        # allowable radial deflection of stator
        c1 = R_st / 500
        u_allow_s = c1 / 20
        R_out = R / 0.995 + h_s + h_ys
        l_is = R_st - R_sh  # distance at which the weight of the stator cylinder acts
        l_iis = l_is  # distance at which the weight of the stator cylinder acts
        l_iiis = l_is  # distance at which the weight of the stator cylinder acts

        mass_st_lam_s = M_Fest + np.pi * L_t * rho_Fe * ((R_st + 0.5 * h_ys) ** 2 - (R_st - 0.5 * h_ys) ** 2)
        W_is = (
            0.5 * gravity * np.sin(phi) * (rho_Fes * L_t * d_s**2)
        )  # length of stator arm beam at which self-weight acts
        W_iis = (
            gravity * np.sin(phi) * (mass_st_lam_s + V_Cus * rho_Copper) / 2 / N_st
        )  # weight of stator cylinder and teeth
        w_s = rho_Fes * gravity * np.sin(phi) * a_s * N_st  # uniformly distributed load of the arms

        mass_stru_steel = 2 * (N_st * (R_1s - R_sh) * a_s * rho_Fes)  # Structural mass of stator arms

        # Calculating radial deflection of the stator

        Numers = R_st**3 * (
            (0.25 * (np.sin(theta_s) - (theta_s * np.cos(theta_s))) / (np.sin(theta_s)) ** 2)
            - (0.5 / np.sin(theta_s))
            + (0.5 / theta_s)
        )
        Povs = ((theta_s / (np.sin(theta_s)) ** 2) + 1 / np.tan(theta_s)) * (
            (0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st)
        )
        Qovs = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
        Lovs = (R_1s - R_sh) * 0.5 / a_s
        Denoms = I_st * (Povs - Qovs + Lovs)
        u_as = (q3 * R_st**2 / E / t_s) * (1 + Numers / Denoms)

        # Calculating axial deflection of the stator
        X_comp1 = (
            W_is * l_is**3 / 12 / E / I_arm_axi_s
        )  # deflection component due to stator arm beam at which self-weight acts
        X_comp2 = W_iis * l_iis**4 / 24 / E / I_arm_axi_s  # deflection component due to 1 / nth of stator cylinder
        X_comp3 = w_s * l_iiis**4 / 24 / E / I_arm_axi_s  # deflection component due to weight of arms
        y_as = X_comp1 + X_comp2 + X_comp3  # axial deflection

        # Calculating circumferential deflection of the stator
        z_as = 2 * np.pi * (R_st + 0.5 * t_s) * L_t / (2 * N_st) * sigma * (l_is + 0.5 * t_s) ** 3 / 3 / E / I_arm_tor_s
        z_allow_s = np.deg2rad(0.05 * R_st)  # allowable torsional deflection
        b_allow_s = 2 * np.pi * R_sh / N_st  # allowable circumferential arm dimension

        val_str_stator = mass_stru_steel + mass_st_lam_s
        val_str_mass = val_str_rotor + val_str_stator

        TC1 = Torque / (2 * np.pi * sigma)  # Desired shear stress
        TC2r = R**2 * L_t  # Evaluating Torque constraint for rotor
        TC2s = R_st**2 * L_t  # Evaluating Torque constraint for stator

        Structural_mass = mass_stru_steel + (N_r * (R_1 - R_sh) * a_r * rho_Fes)
        Stator = mass_st_lam_s + mass_stru_steel + Copper
        Rotor = ((2 * np.pi * t * L_t * R * rho_Fe) + (N_r * (R_1 - R_sh) * a_r * rho_Fes)) + mass_PM
        Mass = Stator + Rotor

        outputs["B_tmax"] = B_tmax
        outputs["B_rymax"] = B_rymax
        outputs["B_symax"] = B_symax
        outputs["B_smax"] = B_smax[-1]
        outputs["B_pm1"] = B_pm1
        outputs["B_g"] = B_g
        outputs["N_s"] = N_s
        outputs["b_s"] = b_s
        outputs["b_t"] = b_t
        outputs["A_Cuscalc"] = A_Cuscalc
        outputs["b_m"] = b_m
        outputs["E_p"] = E_p
        outputs["f"] = f
        outputs["I_s"] = I_s
        outputs["R_s"] = R_s
        outputs["L_s"] = L_s
        outputs["A_1"] = A_1
        outputs["J_s"] = J_s
        outputs["Losses"] = Losses
        outputs["K_rad"] = K_rad
        outputs["eandm_efficiency"] = np.maximum(eps, gen_eff)
        outputs["S"] = S
        outputs["Slot_aspect_ratio"] = Slot_aspect_ratio
        outputs["Copper"] = Copper
        outputs["Iron"] = Iron
        outputs["u_ar"] = u_ar
        outputs["y_ar"] = y_ar
        outputs["z_ar"] = z_ar
        outputs["u_as"] = u_as
        outputs["y_as"] = y_as
        outputs["z_as"] = z_as
        outputs["u_allow_r"] = u_allow_r
        outputs["u_allow_s"] = u_allow_s
        outputs["y_allow_r"] = outputs["y_allow_s"] = y_allow
        outputs["z_allow_s"] = z_allow_s
        outputs["z_allow_r"] = z_allow_r
        outputs["b_allow_s"] = b_allow_s
        outputs["b_allow_r"] = b_allow_r
        outputs["TC1"] = TC1
        outputs["TC2r"] = TC2r
        outputs["TC2s"] = TC2s
        outputs["R_out"] = R_out
        outputs["Structural_mass"] = Structural_mass
        outputs["generator_mass"] = Mass
        outputs["mass_PM"] = mass_PM


# ----------------------------------------------------------------------------------------
class DFIG(GeneratorBase):
    """
    Estimates overall mass, dimensions and Efficiency of DFIG generator.

    Parameters
    ----------
    S_Nmax : float
        Max rated Slip
    B_symax : float, [T]
        Peak Stator Yoke flux density B_ymax

    Returns
    -------
    N_r : float
        Rotor turns
    L_r : float
        Rotor inductance
    h_yr : float
        rotor yoke height
    h_ys : float
        Stator Yoke height
    tau_p : float
        Pole pitch
    Current_ratio : float
        Rotor current ratio
    E_p : float
        Stator phase voltage

    """

    def initialize(self):
        super(DFIG, self).initialize()

    def setup(self):
        super(DFIG, self).setup()
        n_pc = self.options["n_pc"]

        self.add_input("S_Nmax", val=0.0)
        self.add_input("B_symax", val=0.0, units="T")

        self.add_output("N_r", val=0.0)
        self.add_output("L_r", val=0.0)
        self.add_output("h_yr", val=0.0)
        self.add_output("h_ys", val=0.0)
        self.add_output("tau_p", val=0.0)
        self.add_output("Current_ratio", val=np.zeros(n_pc))
        self.add_output("E_p", val=np.zeros(n_pc))

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        rad_ag = inputs["rad_ag"]
        len_s = inputs["len_s"]
        h_s = inputs["h_s"]
        h_0 = inputs["h_0"]
        I_0 = inputs["I_0"]

        machine_rating = inputs["machine_rating"]
        shaft_rpm = inputs["shaft_rpm"]

        rho_Fe = inputs["rho_Fe"]
        rho_Copper = inputs["rho_Copper"]

        B_symax = inputs["B_symax"]
        S_Nmax = inputs["S_Nmax"]

        # Grab constant values
        B_r = inputs["B_r"]
        E = inputs["E"]
        P_Fe0e = inputs["P_Fe0e"]
        P_Fe0h = inputs["P_Fe0h"]
        S_N = inputs["S_N"]
        alpha_p = inputs["alpha_p"]
        b_r_tau_r = inputs["b_r_tau_r"]
        b_ro = inputs["b_ro"]
        b_s_tau_s = inputs["b_s_tau_s"]
        b_so = inputs["b_so"]
        cofi = inputs["cofi"]
        freq = inputs["freq"]
        h_i = inputs["h_i"]
        h_sy0 = inputs["h_sy0"]
        h_w = inputs["h_w"]
        k_fes = inputs["k_fes"]
        k_fillr = inputs["k_fillr"]
        k_s = inputs["k_s"]
        m = discrete_inputs["m"]
        mu_0 = inputs["mu_0"]
        mu_r = inputs["mu_r"]
        p = inputs["p"]
        phi = inputs["phi"]
        q1 = discrete_inputs["q1"]
        q2 = q1 - 1  # Rotor  slots per pole per phase
        ratio_mw2pp = inputs["ratio_mw2pp"]
        resist_Cu = inputs["resist_Cu"]
        sigma = inputs["sigma"]
        v = inputs["v"]
        y_tau_p = inputs["y_tau_p"]
        y_tau_pr = inputs["y_tau_pr"]

        """
        #Assign values to universal constants
        sigma       = 21.5e3           # shear stress in psi (what material? Al, brass, Cu?) ~148e6 Pa
        mu_0        = np.pi * 4e-7        # permeability of free space in m * kg / (s**2 * A**2)
        cofi        = 0.9              # power factor
        h_w         = 0.005            # wedge height
        m           = 3                # Number of phases
        resist_Cu   = 1.8e-8 * 1.4     # copper resisitivity
        h_sy0       = 0

        #Assign values to design constants
        b_so = 0.004                    # Stator slot opening width
        b_ro = 0.004                    # Rotor  slot opening width
        q1 = 5                          # Stator slots per pole per phase
        b_s_tau_s = 0.45                # Stator slot-width / slot-pitch ratio
        b_r_tau_r = 0.45                # Rotor  slot-width / slot-pitch ratio
        y_tau_p  = 12. / 15             # Stator coil span to pole pitch
        y_tau_pr = 10. / 12             # Rotor  coil span to pole pitch

        p = 3                           # pole pairs
        freq = 60                       # grid frequency in Hz
        k_fillr = 0.55                  # Rotor Slot fill factor
        P_Fe0h = 4                      # specific hysteresis losses W / kg @ 1.5 T
        P_Fe0e = 1                      # specific eddy losses W / kg @ 1.5 T
        """

        K_rs = 1 / (-1 * S_Nmax)  # Winding turns ratio between rotor and Stator
        I_SN = machine_rating / (np.sqrt(3) * 3000)  # Rated current
        I_SN_r = I_SN / K_rs  # Stator rated current reduced to rotor

        # Calculating winding factor for stator and rotor

        k_y1 = np.sin(np.pi / 2 * y_tau_p)  # winding chording factor
        k_q1 = np.sin(np.pi / 6) / (q1 * np.sin(np.pi / (6 * q1)))  # winding zone factor
        k_y2 = np.sin(np.pi / 2 * y_tau_pr)  # winding chording factor
        k_q2 = np.sin(np.pi / 6) / (q2 * np.sin(np.pi / (6 * q2)))  # winding zone factor
        k_wd1 = k_y1 * k_q1  # Stator winding factor
        k_wd2 = k_q2 * k_y2  # Rotor winding factor

        ag_dia = 2 * rad_ag  # air gap diameter
        ag_len = (0.1 + 0.012 * machine_rating ** (1.0 / 3)) * 0.001  # air gap length in m
        K_rad = len_s / ag_dia  # Aspect ratio
        rad_r = rad_ag - ag_len  # rotor radius  (was r_r)
        tau_p = np.pi * ag_dia / (2 * p)  # pole pitch

        S = 2 * p * q1 * m  # Stator slots
        N_slots_pp = S / (m * p * 2)  # Number of stator slots per pole per phase
        n = S / 2 * p / q1  # no of slots per pole per phase
        tau_s = tau_p / (m * q1)  # Stator slot pitch
        b_s = b_s_tau_s * tau_s  # Stator slot width
        b_t = tau_s - b_s  # Stator tooth width

        Q_r = 2 * p * m * q2  # Rotor slots
        tau_r = np.pi * (ag_dia - 2 * ag_len) / Q_r  # Rotor slot pitch
        b_r = b_r_tau_r * tau_r  # Rotor slot width
        b_tr = tau_r - b_r  # Rotor tooth width

        # Calculating equivalent slot openings

        mu_rs = 0.005
        mu_rr = 0.005
        W_s = (b_s / mu_rs) * 1e-3  # Stator, in m
        W_r = (b_r / mu_rr) * 1e-3  # Rotor,  in m

        Slot_aspect_ratio1 = h_s / b_s
        Slot_aspect_ratio2 = h_0 / b_r

        # Calculating Carter factor for stator,rotor and effective air gap length

        gamma_s = (2 * W_s / ag_len) ** 2 / (5 + 2 * W_s / ag_len)
        K_Cs = tau_s / (tau_s - ag_len * gamma_s * 0.5)  # page 3 - 13
        gamma_r = (2 * W_r / ag_len) ** 2 / (5 + 2 * W_r / ag_len)
        K_Cr = tau_r / (tau_r - ag_len * gamma_r * 0.5)  # page 3 - 13
        K_C = K_Cs * K_Cr
        g_eff = K_C * ag_len

        om_m = 2 * np.pi * shaft_rpm / 60  # mechanical frequency
        om_e = p * om_m  # electrical frequency
        f = shaft_rpm * p / 60  # generator output freq
        K_s = 0.3  # saturation factor for Iron
        n_c1 = 2  # number of conductors per coil
        a1 = 2  # number of parallel paths
        N_s = np.round(2 * p * N_slots_pp * n_c1 / a1)  # Stator winding turns per phase
        N_r = np.round(N_s * k_wd1 * K_rs / k_wd2)  # Rotor winding turns per phase
        n_c2 = N_r / (Q_r / m)  # rotor turns per coil

        # Calculating peak flux densities and back iron thickness

        B_g1 = mu_0 * 3 * N_r * I_0 * np.sqrt(2) * k_y2 * k_q2 / (np.pi * p * g_eff * (1 + K_s))
        B_g = B_g1 * K_C
        h_ys = B_g * tau_p / (B_symax * np.pi)
        B_rymax = B_symax
        h_yr = h_ys
        B_tsmax = B_g * tau_s / b_t

        d_se = ag_dia + 2 * (h_ys + h_s + h_w)  # stator outer diameter
        D_ratio = d_se / ag_dia  # Diameter ratio

        # Stator slot fill factor
        if ag_dia > 2:
            k_fills = 0.65
        else:
            k_fills = 0.4

        # Stator winding calculation

        # End connection length for stator winding coils

        l_fs = 2 * (0.015 + y_tau_p * tau_p / (2 * np.cos(np.deg2rad(40)))) + np.pi * h_s  # added radians() 2019 09 11

        l_Cus = 2 * N_s * (l_fs + len_s) / a1  # Length of Stator winding

        # Conductor cross-section
        A_s = b_s * (h_s - h_w)
        A_scalc = b_s * 1000 * (h_s - h_w) * 1000
        A_Cus = A_s * q1 * p * k_fills / N_s
        A_Cuscalc = A_scalc * q1 * p * k_fills / N_s

        # Stator winding resistance

        R_s = l_Cus * resist_Cu / A_Cus
        tau_r_min = np.pi * (ag_dia - 2 * (ag_len + h_0)) / Q_r

        # Peak magnetic loading on the rotor tooth

        b_trmin = tau_r_min - b_r_tau_r * tau_r_min
        B_trmax = B_g * tau_r / b_trmin

        # Calculating leakage inductance in  stator

        K_01 = 1 - 0.033 * (W_s**2 / ag_len / tau_s)
        sigma_ds = 0.0042

        L_ssigmas = (2 * mu_0 * len_s * n_c1**2 * S / m / a1**2) * (
            (h_s - h_w) / (3 * b_s) + h_w / b_so
        )  # slot leakage inductance
        L_ssigmaew = (
            (2 * mu_0 * len_s * n_c1**2 * S / m / a1**2) * 0.34 * q1 * (l_fs - 0.64 * tau_p * y_tau_p) / len_s
        )  # end winding leakage inductance
        L_ssigmag = (2 * mu_0 * len_s * n_c1**2 * S / m / a1**2) * (
            0.9 * tau_s * q1 * k_wd1 * K_01 * sigma_ds / g_eff
        )  # tooth tip leakage inductance
        L_ssigma = L_ssigmas + L_ssigmaew + L_ssigmag  # stator leakage inductance
        L_sm = 6 * mu_0 * len_s * tau_p * (k_wd1 * N_s) ** 2 / (np.pi**2 * (p) * g_eff * (1 + K_s))
        L_s = L_ssigmas + L_ssigmaew + L_ssigmag  # stator  inductance

        # Calculating leakage inductance in  rotor

        K_02 = 1 - 0.033 * (W_r**2 / ag_len / tau_r)
        sigma_dr = 0.0062

        l_fr = (0.015 + y_tau_pr * tau_r / 2 / np.cos(np.deg2rad(40))) + np.pi * h_0  # Rotor end connection length
        L_rsl = (mu_0 * len_s * (2 * n_c2) ** 2 * Q_r / m) * (
            (h_0 - h_w) / (3 * b_r) + h_w / b_ro
        )  # slot leakage inductance
        L_rel = (
            (mu_0 * len_s * (2 * n_c2) ** 2 * Q_r / m) * 0.34 * q2 * (l_fr - 0.64 * tau_r * y_tau_pr) / len_s
        )  # end winding leakage inductance
        L_rtl = (mu_0 * len_s * (2 * n_c2) ** 2 * Q_r / m) * (
            0.9 * tau_s * q2 * k_wd2 * K_02 * sigma_dr / g_eff
        )  # tooth tip leakage inductance
        L_r = (L_rsl + L_rtl + L_rel) / K_rs**2  # rotor leakage inductance
        sigma1 = 1 - (L_sm**2 / L_s / L_r)

        # Rotor Field winding

        # conductor cross-section
        diff = h_0 - h_w
        A_Cur = k_fillr * p * q2 * b_r * diff / N_r
        A_Curcalc = A_Cur * 1e6

        L_cur = 2 * N_r * (l_fr + len_s)  # rotor winding length
        Resist_r = resist_Cu * L_cur / A_Cur  # Rotor resistance
        R_R = Resist_r / K_rs**2  # Equivalent rotor resistance reduced to stator

        om_s = shaft_rpm * 2 * np.pi / 60  # synchronous speed in rad / s
        P_e = machine_rating / (1 - S_Nmax)  # Air gap power

        # Calculating No-load voltage
        E_p = om_s * N_s * k_wd1 * rad_ag * len_s * B_g1 * np.sqrt(2)
        I_r = P_e / m / E_p  # rotor active current
        I_sm = E_p / (2 * np.pi * freq * (L_s + L_sm))  # stator reactive current
        I_s = np.sqrt(I_r**2 + I_sm**2)  # Stator current
        I_srated = machine_rating / 3 / K_rs / E_p  # Rated current

        # Calculating winding current densities and specific current loading

        J_s = I_s / A_Cuscalc
        J_r = I_r / A_Curcalc
        A_1 = 2 * m * N_s * I_s / (np.pi * 2 * rad_ag)
        Current_ratio = I_0 / I_srated  # Ratio of magnetization current to rated current

        # Calculating masses of the electromagnetically active materials

        V_Cuss = m * l_Cus * A_Cus
        V_Cusr = m * L_cur * A_Cur
        V_Fest = len_s * np.pi * ((rad_ag + h_s) ** 2 - rad_ag**2) - (2 * m * q1 * p * b_s * h_s * len_s)
        V_Fesy = len_s * np.pi * ((rad_ag + h_s + h_ys) ** 2 - (rad_ag + h_s) ** 2)
        V_Fert = len_s * np.pi * (rad_r**2 - (rad_r - h_0) ** 2) - 2 * m * q2 * p * b_r * h_0 * len_s
        V_Fery = len_s * np.pi * ((rad_r - h_0) ** 2 - (rad_r - h_0 - h_yr) ** 2)
        Copper = (V_Cuss + V_Cusr) * rho_Copper
        M_Fest = V_Fest * rho_Fe
        M_Fesy = V_Fesy * rho_Fe
        M_Fert = V_Fert * rho_Fe
        M_Fery = V_Fery * rho_Fe
        Iron = M_Fest + M_Fesy + M_Fert + M_Fery
        M_gen = (Copper) + (Iron)

        # K_gen = Cu * C_Cu + (Iron) * C_Fe #%M_pm * K_pm

        L_tot = len_s
        Structural_mass = 0.0002 * M_gen**2 + 0.6457 * M_gen + 645.24
        Mass = M_gen + Structural_mass

        # Calculating Losses and efficiency
        # 1. Copper losses

        K_R = 1.2  # skin effect correction coefficient
        P_Cuss = m * I_s**2 * R_s * K_R  # Copper loss - stator
        P_Cusr = m * I_r**2 * R_R  # Copper loss - rotor
        P_Cusnom = P_Cuss + P_Cusr  # Copper loss - total

        # Iron Losses ( from Hysteresis and eddy currents)
        P_Hyys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator yoke
        P_Ftys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)  # Eddy losses in stator yoke
        P_Hyd = M_Fest * (B_tsmax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator teeth
        P_Ftd = M_Fest * (B_tsmax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)  # Eddy losses in stator teeth
        P_Hyyr = (
            M_Fery * (B_rymax / 1.5) ** 2 * (P_Fe0h * abs(S_Nmax) * om_e / (2 * np.pi * 60))
        )  # Hysteresis losses in rotor yoke
        P_Ftyr = (
            M_Fery * (B_rymax / 1.5) ** 2 * (P_Fe0e * (abs(S_Nmax) * om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy losses in rotor yoke
        P_Hydr = (
            M_Fert * (B_trmax / 1.5) ** 2 * (P_Fe0h * abs(S_Nmax) * om_e / (2 * np.pi * 60))
        )  # Hysteresis losses in rotor teeth
        P_Ftdr = (
            M_Fert * (B_trmax / 1.5) ** 2 * (P_Fe0e * (abs(S_Nmax) * om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy losses in rotor teeth
        P_add = 0.5 * machine_rating / 100  # additional losses
        P_Fesnom = P_Hyys + P_Ftys + P_Hyd + P_Ftd + P_Hyyr + P_Ftyr + P_Hydr + P_Ftdr  # Total iron loss
        delta_v = 1  # allowable brush voltage drop
        p_b = 3 * delta_v * I_r  # Brush loss

        Losses = P_Cusnom + P_Fesnom + p_b + P_add
        gen_eff = (P_e - Losses) / P_e

        # Calculating stator winding current density
        J_s = I_s / A_Cuscalc

        # Calculating  electromagnetic torque
        T_e = p * (machine_rating * 1.01) / (2 * np.pi * freq * (1 - S_Nmax))

        # Calculating for tangential stress constraints

        TC1 = T_e / (2 * np.pi * sigma)
        TC2r = rad_ag**2 * len_s

        r_out = d_se * 0.5
        outputs["R_out"] = r_out
        outputs["B_g"] = B_g
        outputs["B_g1"] = B_g1
        outputs["B_rymax"] = B_rymax
        outputs["B_tsmax"] = B_tsmax
        outputs["B_trmax"] = B_trmax

        outputs["N_s"] = N_s
        outputs["S"] = S
        outputs["h_ys"] = h_ys
        outputs["b_s"] = b_s
        outputs["b_t"] = b_t

        outputs["D_ratio"] = D_ratio
        outputs["A_Cuscalc"] = A_Cuscalc
        outputs["Slot_aspect_ratio1"] = Slot_aspect_ratio1
        outputs["h_yr"] = h_yr

        outputs["tau_p"] = tau_p
        outputs["Q_r"] = Q_r
        outputs["N_r"] = N_r
        outputs["b_r"] = b_r
        outputs["b_trmin"] = b_trmin

        outputs["b_tr"] = b_tr
        outputs["A_Curcalc"] = A_Curcalc
        outputs["Slot_aspect_ratio2"] = Slot_aspect_ratio2
        outputs["E_p"] = E_p
        outputs["f"] = f

        outputs["I_s"] = I_s
        outputs["A_1"] = A_1
        outputs["J_s"] = J_s
        outputs["J_r"] = J_r
        outputs["R_s"] = R_s
        outputs["R_R"] = R_R

        outputs["L_r"] = L_r
        outputs["L_s"] = L_s
        outputs["L_sm"] = L_sm
        outputs["generator_mass"] = Mass
        outputs["K_rad"] = K_rad
        outputs["Losses"] = Losses

        outputs["eandm_efficiency"] = np.maximum(eps, gen_eff)
        outputs["Copper"] = Copper
        outputs["Iron"] = Iron
        outputs["Structural_mass"] = Structural_mass
        outputs["TC1"] = TC1
        outputs["TC2r"] = TC2r

        outputs["Current_ratio"] = Current_ratio


# ----------------------------------------------------------------------------------------
class SCIG(GeneratorBase):
    """
    Estimates overall mass dimensions and Efficiency of Squirrel cage Induction generator.

    Parameters
    ----------
    B_symax : float, [T]
        Peak Stator Yoke flux density B_ymax

    Returns
    -------
    h_yr : float
        rotor yoke height
    h_ys : float
        Stator Yoke height
    tau_p : float
        Pole pitch
    D_ratio_UL : float
        Dia ratio upper limit
    D_ratio_LL : float
        Dia ratio Lower limit
    K_rad_UL : float
        Aspect ratio upper limit
    K_rad_LL : float
        Aspect ratio Lower limit
    rad_r : float
        rotor radius
    A_bar : float
        Rotor Conductor cross-section mm^2
    E_p : float
        Stator phase voltage

    """

    def initialize(self):
        super(SCIG, self).initialize()

    def setup(self):
        super(SCIG, self).setup()
        self.add_input("B_symax", val=0.0, units="T")

        self.add_output("h_yr", val=0.0)
        self.add_output("h_ys", val=0.0)
        self.add_output("tau_p", val=0.0)
        self.add_output("D_ratio_UL", val=0.0)
        self.add_output("D_ratio_LL", val=0.0)
        self.add_output("K_rad_UL", val=0.0)
        self.add_output("K_rad_LL", val=0.0)
        self.add_output("rad_r", val=0.0)
        self.add_output("A_bar", val=0.0)
        self.add_output("E_p", val=np.zeros(self.options["n_pc"]))

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        rad_ag = inputs["rad_ag"]
        len_s = inputs["len_s"]
        h_s = inputs["h_s"]
        h_0 = inputs["h_0"]
        machine_rating = inputs["machine_rating"]
        shaft_rpm = inputs["shaft_rpm"]
        I_0 = inputs["I_0"]
        rho_Fe = inputs["rho_Fe"]
        rho_Copper = inputs["rho_Copper"]
        B_symax = inputs["B_symax"]

        # Grab constant values
        B_r = inputs["B_r"]
        E = inputs["E"]
        P_Fe0e = inputs["P_Fe0e"]
        P_Fe0h = inputs["P_Fe0h"]
        S_N = inputs["S_N"]
        alpha_p = inputs["alpha_p"]
        b_r_tau_r = inputs["b_r_tau_r"]
        b_ro = inputs["b_ro"]
        b_s_tau_s = inputs["b_s_tau_s"]
        b_so = inputs["b_so"]
        cofi = inputs["cofi"]
        freq = inputs["freq"]
        h_i = inputs["h_i"]
        h_sy0 = inputs["h_sy0"]
        h_w = inputs["h_w"]
        k_fes = inputs["k_fes"]
        k_fillr = inputs["k_fillr"]
        k_s = inputs["k_s"]
        m = discrete_inputs["m"]
        mu_0 = inputs["mu_0"]
        mu_r = inputs["mu_r"]
        p = inputs["p"]
        phi = inputs["phi"]
        q1 = discrete_inputs["q1"]
        q2 = discrete_inputs["q2"]
        ratio_mw2pp = inputs["ratio_mw2pp"]
        resist_Cu = inputs["resist_Cu"]
        sigma = inputs["sigma"]
        v = inputs["v"]
        y_tau_p = inputs["y_tau_p"]
        y_tau_pr = inputs["y_tau_pr"]

        """
        # Assign values to universal constants
        sigma = 21.5e3            # shear stress (psi) what material?
        mu_0  = np.pi*4e-7           # permeability of free space in m * kg / (s**2 * A**2)
        cofi  = 0.9               # power factor
        h_w   = 0.005             # wedge height
        m     = 3                 # Number of phases
        resist_Cu = 1.8e-8 * 1.4  # Copper resistivity

        #Assign values to design constants
        b_so      = 0.004                     # Stator slot opening width
        b_ro      = 0.004                     # Rotor  slot opening width
        q1        = 6                         # Stator slots per pole per phase
        q2        = 4                         # Rotor  slots per pole per phase
        b_s_tau_s = 0.45                      # Stator Slot width/Slot pitch ratio
        b_r_tau_r = 0.45                      # Rotor Slot width/Slot pitch ratio
        y_tau_p   = 12./15                    # Coil span/pole pitch

        p         = 3                         # number of pole pairs
        freq      = 60                        # frequency in Hz
        k_fillr   = 0.7                       # Rotor  slot fill factor
        P_Fe0h    = 4                         # specific hysteresis losses W / kg @ 1.5 T @50 Hz
        P_Fe0e    = 1                         # specific eddy losses W / kg @ 1.5 T @50 Hz

        S_N       = -0.002                    # Slip
        """

        n_1 = shaft_rpm / (1 - S_N)  # actual rotor speed (rpm)

        # Calculating winding factor

        k_y1 = np.sin(np.pi / 2 * y_tau_p)  # winding chording factor
        k_q1 = np.sin(np.pi / 6) / (q1 * np.sin(np.pi / (6 * q1)))  # zone factor
        k_wd = k_y1 * k_q1  # winding factor

        # Calculating air gap length
        ag_dia = 2 * rad_ag  # air gap diameter
        ag_len = (0.1 + 0.012 * machine_rating ** (1.0 / 3)) * 0.001  # air gap length in m
        K_rad = len_s / ag_dia  # Aspect ratio
        K_rad_LL = 0.5  # lower limit on aspect ratio
        K_rad_UL = 1.5  # upper limit on aspect ratio
        rad_r = rad_ag - ag_len  # rotor radius

        tau_p = np.pi * ag_dia / (2 * p)  # pole pitch
        S = 2 * p * q1 * m  # Stator slots
        N_slots_pp = S / (m * p * 2)  # Number of stator slots per pole per phase
        tau_s = tau_p / (m * q1)  # Stator slot pitch

        b_s = b_s_tau_s * tau_s  # Stator slot width
        b_t = tau_s - b_s  # Stator tooth width

        Q_r = 2 * p * m * q2  # Rotor slots
        tau_r = np.pi * (ag_dia - 2 * ag_len) / Q_r  # Rotor slot pitch
        b_r = b_r_tau_r * tau_r  # Rotor slot width
        b_tr = tau_r - b_r  # Rotor tooth width
        tau_r_min = np.pi * (ag_dia - 2 * (ag_len + h_0)) / Q_r
        b_trmin = tau_r_min - b_r_tau_r * tau_r_min  # minumum rotor tooth width

        # Calculating equivalent slot openings
        mu_rs = 0.005
        mu_rr = 0.005
        W_s = (b_s / mu_rs) * 1e-3  # Stator, in m
        W_r = (b_r / mu_rr) * 1e-3  # Rotor,  in m

        Slot_aspect_ratio1 = h_s / b_s  # Stator slot aspect ratio
        Slot_aspect_ratio2 = h_0 / b_r  # Rotor slot aspect ratio

        # Calculating Carter factor for stator,rotor and effective air gap length
        """
        gamma_s = (2 * W_s / ag_len)**2 / (5 + 2 * W_s / ag_len)
        K_Cs    = tau_s / (tau_s - ag_len * gamma_s * 0.5) # page 3-13 Boldea Induction machines Chapter 3
        gamma_r = (2 * W_r / ag_len)**2 / (5 + 2 * W_r / ag_len)
        K_Cr    = tau_r / (tau_r - ag_len * gamma_r * 0.5) # page 3-13 Boldea Induction machines Chapter 3
        """

        K_Cs = carterFactor(ag_len, W_s, tau_s)
        K_Cr = carterFactor(ag_len, W_r, tau_r)
        K_C = K_Cs * K_Cr
        g_eff = K_C * ag_len

        om_m = 2 * np.pi * shaft_rpm / 60  # mechanical frequency
        om_e = p * om_m  # electrical frequency
        f = shaft_rpm * p / 60  # generator output freq
        K_s = 0.3  # saturation factor for Iron
        n_c = 2  # number of conductors per coil
        a1 = 2  # number of parallel paths

        # Calculating stator winding turns
        N_s = np.round(2 * p * N_slots_pp * n_c / a1)

        # Calculating Peak flux densities
        B_g1 = mu_0 * 3 * N_s * I_0 * np.sqrt(2) * k_y1 * k_q1 / (np.pi * p * g_eff * (1 + K_s))
        B_g = B_g1 * K_C
        B_rymax = B_symax

        # calculating back iron thickness
        h_ys = B_g * tau_p / (B_symax * np.pi)
        h_yr = h_ys

        d_se = ag_dia + 2 * (h_ys + h_s + h_w)  # stator outer diameter
        D_ratio = d_se / ag_dia  # Diameter ratio

        # limits for Diameter ratio depending on pole pair
        if 2 * p == 2:
            D_ratio_LL = 1.65
            D_ratio_UL = 1.69
        elif 2 * p == 4:
            D_ratio_LL = 1.46
            D_ratio_UL = 1.49
        elif 2 * p == 6:
            D_ratio_LL = 1.37
            D_ratio_UL = 1.4
        elif 2 * p == 8:
            D_ratio_LL = 1.27
            D_ratio_UL = 1.3
        else:
            D_ratio_LL = 1.2
            D_ratio_UL = 1.24

        # Stator slot fill factor
        if ag_dia > 2:
            k_fills = 0.65
        else:
            k_fills = 0.4

        # Stator winding length and cross-section
        l_fs = 2 * (0.015 + y_tau_p * tau_p / 2 / np.cos(np.deg2rad(40))) + np.pi * h_s  # end connection
        l_Cus = 2 * N_s * (l_fs + len_s) / a1  # shortpitch
        A_s = b_s * (h_s - h_w)  # Slot area
        A_scalc = b_s * 1000 * (h_s - h_w) * 1000  # Conductor cross-section (mm^2)
        A_Cus = A_s * q1 * p * k_fills / N_s  # Conductor cross-section (m^2)
        A_Cuscalc = A_scalc * q1 * p * k_fills / N_s

        # Stator winding resistance
        R_s = l_Cus * resist_Cu / A_Cus

        # Calculating no-load voltage
        om_s = shaft_rpm * 2 * np.pi / 60  # rated angular frequency
        P_e = machine_rating / (1 - S_N)  # Electrical power
        E_p = om_s * N_s * k_wd * rad_ag * len_s * B_g1 * np.sqrt(2)

        S_GN = (1.0 - S_N) * machine_rating  # same as P_e?
        T_e = p * S_GN / (2 * np.pi * freq * (1 - S_N))
        I_srated = machine_rating / (3 * E_p * cofi)

        # Rotor design
        diff = h_0 - h_w
        A_bar = b_r * diff  # bar cross section
        Beta_skin = np.sqrt(np.pi * mu_0 * freq / 2 / resist_Cu)  # coefficient for skin effect correction
        k_rm = Beta_skin * h_0  # coefficient for skin effect correction
        J_b = 6e06  # Bar current density
        K_i = 0.864
        I_b = 2 * m * N_s * k_wd * I_srated / Q_r  # bar current

        # Calculating bar resistance

        R_rb = resist_Cu * k_rm * len_s / A_bar
        I_er = I_b / (2 * np.sin(np.pi * p / Q_r))  # End ring current
        J_er = 0.8 * J_b  # End ring current density
        A_er = I_er / J_er  # End ring cross-section
        b = h_0  # End ring dimension
        a = A_er / b  # End ring dimension
        D_er = (rad_ag * 2 - 2 * ag_len) - 0.003  # End ring diameter
        l_er = np.pi * (D_er - b) / Q_r  # End ring segment length
        if debug:
            sys.stderr.write("l_er {:.4f} A_er {:.4f} D_er {:.4f}\n".format(l_er[0], A_er[0], D_er[0]))

        # Calculating end ring resistance
        R_re = resist_Cu * l_er / (2 * A_er * (np.sin(np.pi * p / Q_r)) ** 2)

        # Calculating equivalent rotor resistance
        if debug:
            sys.stderr.write("R_rb {:.3e} R_re {:.3e} k_wd {:.4f} N_s {} Q_r {}\n".format(R_rb, R_re, k_wd, N_s, Q_r))
        R_R = (R_rb + R_re) * 4 * m * (k_wd * N_s) ** 2 / Q_r

        # Calculating Rotor and Stator teeth flux density
        B_trmax = B_g * tau_r / b_trmin
        B_tsmax = B_g * tau_s / b_t

        # Calculating Equivalent core lengths
        l_r = len_s + 4 * ag_len  # for axial cooling
        l_se = len_s + (2 / 3) * ag_len
        K_fe = 0.95  # Iron factor
        L_e = l_se * K_fe  # radial cooling

        # Calculating leakage inductance in  stator
        if debug:
            sys.stderr.write("b_s {:.3e} b_so {:.3e}\n".format(b_s[0], b_so[0]))
        L_ssigmas = (
            2 * mu_0 * len_s * N_s**2 / p / q1 * ((h_s - h_w) / (3 * b_s) + h_w / b_so)
        )  # slot        leakage inductance
        L_ssigmaew = (
            2 * mu_0 * len_s * N_s**2 / p / q1 * 0.34 * q1 * (l_fs - 0.64 * tau_p * y_tau_p) / len_s
        )  # end winding leakage inductance
        L_ssigmag = (
            2 * mu_0 * len_s * N_s**2 / p / q1 * (5 * (ag_len * K_C / b_so) / (5 + 4 * (ag_len * K_C / b_so)))
        )  # tooth tip   leakage inductance
        L_s = L_ssigmas + L_ssigmaew + L_ssigmag  # stator      leakage inductance
        L_sm = 6 * mu_0 * len_s * tau_p * (k_wd * N_s) ** 2 / (np.pi**2 * p * g_eff * (1 + K_s))

        # Calculating leakage inductance in  rotor
        lambda_ei = 2.3 * D_er / (4 * Q_r * len_s * (np.sin(np.pi * p / Q_r) ** 2)) * np.log(4.7 * ag_dia / (a + 2 * b))
        lambda_b = h_0 / (3 * b_r) + h_w / b_ro
        L_i = np.pi * ag_dia / Q_r

        L_rsl = mu_0 * len_s * ((h_0 - h_w) / (3 * b_r) + h_w / b_ro)  # slot        leakage inductance
        L_rel = mu_0 * (len_s * lambda_b + 2 * lambda_ei * L_i)  # end winding leakage inductance
        L_rtl = mu_0 * len_s * (0.9 * tau_r * 0.09 / g_eff)  # tooth tip   leakage inductance
        L_rsigma = (L_rsl + L_rtl + L_rel) * 4 * m * (k_wd * N_s) ** 2 / Q_r  # rotor       leakage inductance

        # Calculating rotor current
        if debug:
            sys.stderr.write(
                "S_N {} P_e {:.1f} m {} R_R {:.4f} = {:.1f}\n".format(S_N, P_e, m, R_R, -S_N * P_e / m / R_R)
            )
        I_r = np.sqrt(-S_N * P_e / m / R_R)

        I_sm = E_p / (2 * np.pi * freq * L_sm)
        # Calculating stator currents and specific current loading
        I_s = np.sqrt((I_r**2 + I_sm**2))

        A_1 = 2 * m * N_s * I_s / (np.pi * 2 * rad_ag)

        # Calculating masses of the electromagnetically active materials

        V_Cuss = m * l_Cus * A_Cus  # Volume of copper in stator
        V_Cusr = Q_r * len_s * A_bar + np.pi * (D_er * A_er - A_er * b)  # Volume of copper in rotor
        V_Fest = (
            len_s * np.pi * ((rad_ag + h_s) ** 2 - rad_ag**2) - 2 * m * q1 * p * b_s * h_s * len_s
        )  # Volume of iron in stator teeth
        V_Fesy = len_s * np.pi * ((rad_ag + h_s + h_ys) ** 2 - (rad_ag + h_s) ** 2)  # Volume of iron in stator yoke
        rad_r = rad_ag - ag_len  # rotor radius

        V_Fert = (
            np.pi * len_s * (rad_r**2 - (rad_r - h_0) ** 2) - 2 * m * q2 * p * b_r * h_0 * len_s
        )  # Volume of iron in rotor teeth
        V_Fery = np.pi * len_s * ((rad_r - h_0) ** 2 - (rad_r - h_0 - h_yr) ** 2)  # Volume of iron in rotor yoke
        Copper = (V_Cuss + V_Cusr)[-1] * rho_Copper  # Mass of Copper
        M_Fest = V_Fest * rho_Fe  # Mass of stator teeth
        M_Fesy = V_Fesy * rho_Fe  # Mass of stator yoke
        M_Fert = V_Fert * rho_Fe  # Mass of rotor tooth
        M_Fery = V_Fery * rho_Fe  # Mass of rotor yoke
        Iron = M_Fest + M_Fesy + M_Fert + M_Fery

        Active_mass = Copper + Iron
        L_tot = len_s
        Structural_mass = 0.0001 * Active_mass**2 + 0.8841 * Active_mass - 132.5
        Mass = Active_mass + Structural_mass

        # Calculating Losses and efficiency

        # 1. Copper losses

        K_R = 1.2  # skin effect correction coefficient
        P_Cuss = m * I_s**2 * R_s * K_R  # Copper loss - stator
        P_Cusr = m * I_r**2 * R_R  # Copper loss - rotor
        P_Cusnom = P_Cuss + P_Cusr  # Copper loss - total

        # Iron Losses ( from Hysteresis and eddy currents)
        P_Hyys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator yoke
        P_Ftys = (
            M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy       losses in stator yoke
        P_Hyd = M_Fest * (B_tsmax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator tooth
        P_Ftd = (
            M_Fest * (B_tsmax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy       losses in stator tooth
        P_Hyyr = (
            M_Fery * (B_rymax / 1.5) ** 2 * (P_Fe0h * abs(S_N) * om_e / (2 * np.pi * 60))
        )  # Hysteresis losses in rotor yoke
        P_Ftyr = (
            M_Fery * (B_rymax / 1.5) ** 2 * (P_Fe0e * (abs(S_N) * om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy       losses in rotor yoke
        P_Hydr = (
            M_Fert * (B_trmax / 1.5) ** 2 * (P_Fe0h * abs(S_N) * om_e / (2 * np.pi * 60))
        )  # Hysteresis losses in rotor tooth
        P_Ftdr = (
            M_Fert * (B_trmax / 1.5) ** 2 * (P_Fe0e * (abs(S_N) * om_e / (2 * np.pi * 60)) ** 2)
        )  # Eddy       losses in rotor tooth

        # Calculating Additional losses
        P_add = 0.5 * machine_rating / 100
        P_Fesnom = P_Hyys + P_Ftys + P_Hyd + P_Ftd + P_Hyyr + P_Ftyr + P_Hydr + P_Ftdr
        Losses = P_Cusnom + P_Fesnom + P_add
        gen_eff = (P_e - Losses) / P_e

        # Calculating current densities in the stator and rotor
        J_s = I_s / A_Cuscalc
        J_r = I_r / A_bar / 1e6

        # Calculating Tangential stress constraints
        TC1 = T_e / (2 * np.pi * sigma)
        TC2r = rad_ag**2 * len_s

        # Calculating mass moments of inertia and center of mass
        r_out = d_se * 0.5
        outputs["R_out"] = r_out
        outputs["B_tsmax"] = B_tsmax
        outputs["B_trmax"] = B_trmax
        outputs["B_rymax"] = B_rymax
        outputs["B_g"] = B_g
        outputs["B_g1"] = B_g1
        outputs["N_s"] = N_s
        outputs["S"] = S
        outputs["h_ys"] = h_ys
        outputs["b_s"] = b_s
        outputs["b_t"] = b_t
        outputs["D_ratio"] = D_ratio
        outputs["D_ratio_UL"] = D_ratio_UL
        outputs["D_ratio_LL"] = D_ratio_LL
        outputs["A_Cuscalc"] = A_Cuscalc
        outputs["Slot_aspect_ratio1"] = Slot_aspect_ratio1
        outputs["h_yr"] = h_yr
        outputs["tau_p"] = tau_p
        outputs["Q_r"] = Q_r
        outputs["b_r"] = b_r
        outputs["b_trmin"] = b_trmin
        outputs["b_tr"] = b_tr
        outputs["rad_r"] = rad_r
        outputs["A_bar"] = A_bar
        outputs["Slot_aspect_ratio2"] = Slot_aspect_ratio2
        outputs["E_p"] = E_p
        outputs["f"] = f
        outputs["I_s"] = I_s
        outputs["A_1"] = A_1
        outputs["J_s"] = J_s
        outputs["J_r"] = J_r
        outputs["R_s"] = R_s
        outputs["R_R"] = R_R[-1]
        outputs["L_s"] = L_s
        outputs["L_sm"] = L_sm
        outputs["generator_mass"] = Mass
        outputs["K_rad"] = K_rad
        outputs["K_rad_UL"] = K_rad_UL
        outputs["K_rad_LL"] = K_rad_LL
        outputs["Losses"] = Losses
        outputs["eandm_efficiency"] = np.maximum(eps, gen_eff)
        outputs["Copper"] = Copper
        outputs["Iron"] = Iron
        outputs["Structural_mass"] = Structural_mass
        outputs["TC1"] = TC1
        outputs["TC2r"] = TC2r


# ----------------------------------------------------------------------------------------
class EESG(GeneratorBase):
    """
    Estimates overall mass dimensions and Efficiency of Electrically Excited Synchronous generator.

    Parameters
    ----------
    I_f : float, [A]
        Excitation current
    N_f : float
        field turns
    b_arm : float, [m]
        arm width
    h_yr : float, [m]
        rotor yoke height
    h_ys : float, [m]
        Yoke height
    tau_p : float, [m]
        Pole pitch self.tau_p

    Returns
    -------
    n_brushes : float
        number of brushes
    h_p : float, [m]
        Pole height
    b_p : float, [m]
        Pole width
    L_m : float, [H]
        Stator synchronising inductance
    R_r : float, [ohm]
        Rotor resistance
    B_tmax : float, [T]
        Peak Teeth flux density
    B_gfm : float, [T]
        Average air gap flux density B_g
    B_pc : float, [T]
        Pole core flux density
    B_symax : float, [T]
        Peak Stator Yoke flux density B_ymax
    E_s : float, [V]
        Stator phase voltage
    J_f : float, [A*m**-2]
        rotor Current density
    Power_ratio : float
        Power_ratio
    Load_mmf_ratio : float
        mmf_ratio
    """

    def initialize(self):
        super(EESG, self).initialize()

    def setup(self):
        super(EESG, self).setup()

        self.add_input("I_f", val=0.0, units="A")
        self.add_input("N_f", val=0.0)
        self.add_input("b_arm", val=0.0, units="m")
        self.add_input("h_yr", val=0.0, units="m")
        self.add_input("h_ys", val=0.0, units="m")
        self.add_input("tau_p", val=0.0, units="m")

        self.add_output("n_brushes", val=0.0)
        self.add_output("h_p", val=0.0, units="m")
        self.add_output("b_p", val=0.0, units="m")
        self.add_output("L_m", val=0.0, units="H")
        self.add_output("R_r", val=0.0, units="ohm")
        self.add_output("B_tmax", val=0.0, units="T")
        self.add_output("B_gfm", val=0.0, units="T")
        self.add_output("B_pc", val=0.0, units="T")
        self.add_output("B_symax", val=0.0, units="T")
        self.add_output("E_s", val=np.zeros(self.options["n_pc"]), units="V")
        self.add_output("J_f", val=0.0, units="A*m**-2")
        self.add_output("Power_ratio", val=0.0)
        self.add_output("Load_mmf_ratio", val=0.0)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        rad_ag = inputs["rad_ag"]
        len_s = inputs["len_s"]
        h_s = inputs["h_s"]
        tau_p = inputs["tau_p"]
        N_f = inputs["N_f"]
        I_f = inputs["I_f"]
        h_ys = inputs["h_ys"]
        h_yr = inputs["h_yr"]
        machine_rating = inputs["machine_rating"]
        shaft_rpm = inputs["shaft_rpm"]
        Torque = inputs["rated_torque"]

        b_st = inputs["b_st"]
        d_s = inputs["d_s"]
        t_ws = inputs["t_ws"]
        n_r = inputs["n_r"]
        n_s = inputs["n_s"]
        b_r = inputs["b_arm"]
        d_r = inputs["d_r"]
        t_wr = inputs["t_wr"]

        R_sh = 0.5 * inputs["D_shaft"]
        rho_Fe = inputs["rho_Fe"]
        rho_Copper = inputs["rho_Copper"]
        rho_Fes = inputs["rho_Fes"]

        # Grab constant values
        B_r = inputs["B_r"]
        E = inputs["E"]
        P_Fe0e = inputs["P_Fe0e"]
        P_Fe0h = inputs["P_Fe0h"]
        S_N = inputs["S_N"]
        alpha_p = inputs["alpha_p"]
        b_r_tau_r = inputs["b_r_tau_r"]
        b_ro = inputs["b_ro"]
        b_s_tau_s = inputs["b_s_tau_s"]
        b_so = inputs["b_so"]
        cofi = inputs["cofi"]
        freq = inputs["freq"]
        h_i = inputs["h_i"]
        h_sy0 = inputs["h_sy0"]
        h_w = inputs["h_w"]
        k_fes = inputs["k_fes"]
        k_fillr = inputs["k_fillr"]
        k_fills = inputs["k_fills"]
        k_s = inputs["k_s"]
        m = discrete_inputs["m"]
        mu_0 = inputs["mu_0"]
        mu_r = inputs["mu_r"]
        p = inputs["p"]
        phi = inputs["phi"]
        q1 = discrete_inputs["q1"]
        q2 = discrete_inputs["q2"]
        ratio_mw2pp = inputs["ratio_mw2pp"]
        resist_Cu = inputs["resist_Cu"]
        sigma = inputs["sigma"]
        v = inputs["v"]
        y_tau_p = inputs["y_tau_p"]
        y_tau_pr = inputs["y_tau_pr"]

        """
        # Assign values to universal constants
        E      = 2e11                # N / m^2 young's modulus
        sigma  = 48.373e3            # shear stress of steel in psi (~333 MPa)
        mu_0   = np.pi * 4e-7           # permeability of free space in m * kg / (s**2 * A**2)
        phi    = np.deg2rad(90)

        # Assign values to design constants
        h_w       = 0.005
        b_so      = 0.004              # Stator slot opening
        m         = 3                  # number of phases
        q1        = 2                  # no of stator slots per pole per phase
        b_s_tau_s = 0.45               # ratio of slot width to slot pitch
        P_Fe0h    = 4                  # specific hysteresis losses W / kg @ 1.5 T @50 Hz
        P_Fe0e    = 1                  # specific eddy losses W / kg @ 1.5 T @50 Hz
        resist_Cu    = 1.8e-8 * 1.4       # resisitivity of copper # ohm-meter  (Why the 1.4 factor?)
        k_fes     = 0.9                # iron fill factor (not used)
        y_tau_p   = 1                  # coil span / pole pitch fullpitch
        k_fillr   = 0.7                # rotor slot fill factor
        k_s       = 0.2                # magnetic saturation factor for iron
        cofi   = 0.85               # power factor
        """
        T = Torque

        # back iron thickness for rotor and stator
        t_s = h_ys
        t = h_yr

        # Aspect ratio
        K_rad = len_s / (2 * rad_ag)

        ###################################################### Electromagnetic design#############################################

        alpha_p = np.pi / 2 * 0.7  # (not used)
        dia = 2 * rad_ag  # air gap diameter

        # air gap length and minimum values
        g = 0.001 * dia
        if g < 0.005:
            g = 0.005

        r_r = rad_ag - g  # rotor radius
        d_se = dia + 2 * h_s + 2 * h_ys  # stator outer diameter (not used)
        p = np.round(np.pi * dia / (2 * tau_p))  # number of pole pairs
        S = 2 * p * q1 * m  # number of slots of stator phase winding
        N_conductors = S * 2
        N_s = N_conductors / 2 / m  # Stator turns per phase
        alpha = 180 / S / p  # electrical angle (not used)

        tau_s = np.pi * dia / S  # slot pitch
        h_ps = 0.1 * tau_p  # height of pole shoe
        b_pc = 0.4 * tau_p  # width  of pole core
        h_pc = 0.6 * tau_p  # height of pole core
        h_p = 0.7 * tau_p  # pole height
        b_p = h_p
        b_s = tau_s * b_s_tau_s  # slot width
        Slot_aspect_ratio = h_s / b_s
        b_t = tau_s - b_s  # tooth width

        # Calculating Carter factor and effective air gap
        g_a = g
        K_C1 = (tau_s + 10 * g_a) / (tau_s - b_s + 10 * g_a)  # salient pole rotor
        g_1 = K_C1 * g

        # calculating angular frequency
        om_m = 2 * np.pi * shaft_rpm / 60
        om_e = 60
        f = shaft_rpm * p / 60

        # Slot fill factor according to air gap radius

        if 2 * rad_ag > 2:
            K_fills = 0.65
        else:
            K_fills = 0.4

        # Calculating Stator winding factor

        k_y1 = np.sin(y_tau_p * np.pi / 2)  # chording factor
        k_q1 = np.sin(np.pi / 6) / q1 / np.sin(np.pi / 6 / q1)  # winding zone factor
        k_wd = k_y1 * k_q1

        # Calculating stator winding conductor length, cross-section and resistance

        shortpitch = 0
        l_Cus = 2 * N_s * (2 * (tau_p - shortpitch / m / q1) + len_s)  # length of winding
        A_s = b_s * (h_s - h_w)
        A_scalc = b_s * 1000 * (h_s - h_w) * 1000  # cross section in mm^2
        A_Cus = A_s * q1 * p * K_fills / N_s
        A_Cuscalc = A_scalc * q1 * p * K_fills / N_s
        R_s = l_Cus * resist_Cu / A_Cus

        # field winding design, conductor length, cross-section and resistance

        N_f = np.round(N_f)  # rounding the field winding turns to the nearest integer
        I_srated = machine_rating / (np.sqrt(3) * 5000 * cofi)
        l_pole = len_s - 0.050 + 0.120  # 50mm smaller than stator and 120mm longer to accommodate end stack
        K_fe = 0.95
        l_pfe = l_pole * K_fe
        l_Cur = 4 * p * N_f * (l_pfe + b_pc + np.pi / 4 * (np.pi * (r_r - h_pc - h_ps) / p - b_pc))
        A_Cur = k_fillr * h_pc * 0.5 / N_f * (np.pi * (r_r - h_pc - h_ps) / p - b_pc)
        A_Curcalc = k_fillr * h_pc * 1000 * 0.5 / N_f * (np.pi * (r_r - h_pc - h_ps) / p - b_pc) * 1000
        Slot_Area = A_Cur * 2 * N_f / k_fillr  # (not used)
        R_r = resist_Cu * l_Cur / A_Cur  # ohms

        # field winding current density

        J_f = I_f / A_Curcalc

        # calculating air flux density

        B_gfm = mu_0 * N_f * I_f / (g_1 * (1 + k_s))  # No-load air gap flux density

        B_g = B_gfm * 4 * np.sin(0.5 * b_p * np.pi / tau_p) / np.pi  # fundamental component
        B_symax = tau_p * B_g / np.pi / h_ys  # stator yoke flux density
        L_fg = (
            2
            * mu_0
            * p
            * len_s
            * 4
            * N_f**2
            * ((h_ps / (tau_p - b_p)) + (h_pc / (3 * np.pi * (r_r - h_pc - h_ps) / p - b_pc)))
        )  #  (not used)

        # calculating no-load voltage and stator current

        E_s = 2 * N_s * len_s * rad_ag * k_wd * om_m * B_g / np.sqrt(2)  # no-load voltage
        # I_s = (E_s - (E_s**2 - 4 * R_s * machine_rating / m)**0.5) / (2 * R_s)
        erm = np.maximum(0.0, E_s**2 - 4 * R_s * machine_rating / m)
        I_s = (E_s - erm**0.5) / (2 * R_s)

        # Calculating stator winding current density and specific current loading

        A_1 = 6 * N_s * I_s / (np.pi * dia)
        J_s = I_s / A_Cuscalc

        # Calculating magnetic loading in other parts of the machine

        delta_m = 0  # Initialising load angle

        # peak flux density in pole core, rotor yoke and stator teeth

        B_pc = (1 / b_pc) * (
            (2 * tau_p / np.pi) * B_g * np.cos(delta_m)
            + (2 * mu_0 * I_f * N_f * ((2 * h_ps / (tau_p - b_p)) + (h_pc / (tau_p - b_pc))))
        )
        B_rymax = 0.5 * b_pc * B_pc / h_yr
        B_tmax = (B_gfm + B_g) * tau_s * 0.5 / b_t

        # Calculating leakage inductances in the stator

        L_ssigmas = (
            2 * mu_0 * len_s * N_s**2 / p / q1 * ((h_s - h_w) / (3 * b_s) + h_w / b_so)
        )  # slot leakage inductance
        L_ssigmaew = mu_0 * 1.2 * N_s**2 / p * 1.2 * (2 / 3 * tau_p + 0.01)  # end winding leakage inductance
        L_ssigmag = (
            2 * mu_0 * len_s * N_s**2 / p / q1 * (5 * (g / b_so) / (5 + 4 * (g / b_so)))
        )  # tooth tip leakage inductance
        L_ssigma = L_ssigmas + L_ssigmaew + L_ssigmag  # stator leakage inductance

        # Calculating effective air gap

        """
        What is the source of this function that combines 1st and 13th powers? Very suspicious...
        Inputs appear to be in the range of 0.45 to 2.2, so outputs are 180 to 178000

        Equations given without reference in:
        H. Polinder, J. G. Slootweg . “Design optimization of a synchronous generator for a direct-drive wind turbine,”
        (paper presented at the European Wind Energy Conference, Copenhagen, Denmark, July2–6, 2001

        def airGapFn(B, fact):
            val = 400 * B + 7 * B**13
            ans = val * fact
            sys.stderr.write('aGF: B {} val {} ans {}\n'.format(B, val, ans))
            return val

        At_t =  h_s           * airGapFn(B_tmax, h_s)
        At_sy = tau_p / 2     * airGapFn(B_symax, tau_p/2)
        At_pc = (h_pc + h_ps) * airGapFn(B_pc, h_pc + h_ps)
        At_ry = tau_p / 2     * airGapFn(B_rymax, tau_p/2)
        """
        At_g = g_1 * B_gfm / mu_0
        At_t = h_s * (400 * B_tmax + 7 * B_tmax**13)
        At_sy = tau_p * 0.5 * (400 * B_symax + 7 * B_symax**13)
        At_pc = (h_pc + h_ps) * (400 * B_pc + 7 * B_pc**13)
        At_ry = tau_p * 0.5 * (400 * B_rymax + 7 * B_rymax**13)
        g_eff = (At_g + At_t + At_sy + At_pc + At_ry) * g_1 / At_g

        L_m = 6 * k_wd**2 * N_s**2 * mu_0 * rad_ag * len_s / np.pi / g_eff / p**2
        B_r1 = (mu_0 * I_f * N_f * 4 * np.sin(0.5 * (b_p / tau_p) * np.pi)) / g_eff / np.pi  # (not used)

        # Calculating direct axis and quadrature axes inductances
        L_dm = (b_p / tau_p + (1 / np.pi) * np.sin(np.pi * b_p / tau_p)) * L_m
        L_qm = (
            b_p / tau_p - (1 / np.pi) * np.sin(np.pi * b_p / tau_p) + 2 / (3 * np.pi) * np.cos(b_p * np.pi / 2 * tau_p)
        ) * L_m

        # Calculating actual load angle

        delta_m = np.arctan(om_e * L_qm * I_s / E_s)
        L_d = L_dm + L_ssigma  # (not used)
        L_q = L_qm + L_ssigma  # (not used)
        I_sd = I_s * np.sin(delta_m)
        I_sq = I_s * np.cos(delta_m)

        # induced voltage

        E_p = om_e * L_dm * I_sd + np.sqrt(E_s**2 - (om_e * L_qm * I_sq) ** 2)  # (not used)
        # M_sf = mu_0 * 8*rad_ag * len_s * k_wd * N_s * N_f * np.sin(0.5 * b_p / tau_p * np.pi) / (p * g_eff * np.pi)
        # I_f1 = np.sqrt(2) * (E_p) / (om_e * M_sf)
        # I_f2 = (E_p / E_s) * B_g * g_eff * np.pi / (4 * N_f * mu_0 * np.sin(np.pi * b_p / 2/tau_p))
        # phi_max_stator = k_wd * N_s * np.pi * rad_ag * len_s * 2*mu_0 * N_f * I_f * 4*np.sin(0.5 * b_p / tau_p / np.pi) / (p * np.pi * g_eff * np.pi)
        # M_sf = mu_0 * 8*rad_ag * len_s * k_wd * N_s * N_f * np.sin(0.5 * b_p / tau_p / np.pi) / (p * g_eff * np.pi)

        L_tot = len_s + 2 * tau_p

        # Excitation power
        V_fn = 500
        Power_excitation = V_fn * 2 * I_f  # total rated power in excitation winding
        Power_ratio = Power_excitation * 100 / machine_rating

        # Calculating Electromagnetically Active mass
        L_tot = len_s + 2 * tau_p  # (not used)
        V_Cuss = m * l_Cus * A_Cus  # volume of copper in stator
        V_Cusr = l_Cur * A_Cur  # volume of copper in rotor
        V_Fest = (
            len_s * np.pi * ((rad_ag + h_s) ** 2 - rad_ag**2) - 2 * m * q1 * p * b_s * h_s * len_s
        )  # volume of iron in stator tooth
        V_Fesy = len_s * np.pi * ((rad_ag + h_s + h_ys) ** 2 - (rad_ag + h_s) ** 2)  # volume of iron in stator yoke
        V_Fert = l_pfe * 2 * p * (h_pc * b_pc + b_p * h_ps)  # volume of iron in rotor pole
        V_Fery = (
            l_pfe * np.pi * ((r_r - h_ps - h_pc) ** 2 - (r_r - h_ps - h_pc - h_yr) ** 2)
        )  # volume of iron in rotor yoke

        Copper = (V_Cuss + V_Cusr) * rho_Copper
        M_Fest = V_Fest * rho_Fe
        M_Fesy = V_Fesy * rho_Fe
        M_Fert = V_Fert * rho_Fe
        M_Fery = V_Fery * rho_Fe
        Iron = M_Fest + M_Fesy + M_Fert + M_Fery

        I_snom = machine_rating / (3 * E_s * cofi)

        ## Optional## Calculating mmf ratio
        F_1no_load = 3 * 2**0.5 * N_s * k_wd * I_s / (np.pi * p)  # (not used)
        Nf_If_no_load = N_f * I_f
        F_1_rated = (3 * 2**0.5 * N_s * k_wd * I_srated) / (np.pi * p)
        Nf_If_rated = 2 * Nf_If_no_load
        Load_mmf_ratio = Nf_If_rated / F_1_rated

        ## Calculating losses
        # 1. Copper losses
        K_R = 1.2  # skin effect correction coefficient
        P_Cuss = m * I_snom**2 * R_s * K_R
        P_Cusr = I_f**2 * R_r
        P_Cusnom_total = P_Cuss + P_Cusr  # Watts

        # 2. Iron losses ( Hysteresis and Eddy currents)
        P_Hyys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator yoke
        P_Ftys = M_Fesy * (B_symax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)  # Eddy losses in stator yoke
        P_Fesynom = P_Hyys + P_Ftys
        P_Hyd = M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0h * om_e / (2 * np.pi * 60))  # Hysteresis losses in stator teeth
        P_Ftd = M_Fest * (B_tmax / 1.5) ** 2 * (P_Fe0e * (om_e / (2 * np.pi * 60)) ** 2)  # Eddy losses in stator teeth
        P_Festnom = P_Hyd + P_Ftd

        # brushes
        delta_v = 1
        n_brushes = I_f * 2 / 120

        if n_brushes < 0.5:
            n_brushes = 1
        else:
            n_brushes = np.round(n_brushes)

        # 3. brush losses

        p_b = 2 * delta_v * I_f
        Losses = P_Cusnom_total + P_Festnom + P_Fesynom + p_b
        gen_eff = machine_rating / (Losses + machine_rating)

        ################################################## Structural  Design ########################################################

        ## Structural deflection calculations

        # rotor structure

        q3 = B_g**2 / 2 / mu_0  # normal component of Maxwell's stress
        # l           = l_s                        # l - stator core length - now using l_s everywhere
        l_b = 2 * tau_p  # end winding length # (not used)
        l_e = len_s + 2 * 0.001 * rad_ag  # equivalent core length # (not used)
        a_r = (b_r * d_r) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr))  # cross-sectional area of rotor armms
        A_r = len_s * t  # cross-sectional area of rotor cylinder
        N_r = np.round(n_r)
        theta_r = np.pi / N_r  # half angle between spokes
        I_r = len_s * t**3 / 12  # second moment of area of rotor cylinder
        I_arm_axi_r = (
            (b_r * d_r**3) - ((b_r - 2 * t_wr) * (d_r - 2 * t_wr) ** 3)
        ) / 12  # second moment of area of rotor arm
        I_arm_tor_r = (
            (d_r * b_r**3) - ((d_r - 2 * t_wr) * (b_r - 2 * t_wr) ** 3)
        ) / 12  # second moment of area of rotot arm w.r.t torsion
        R = r_r - h_ps - h_pc - 0.5 * h_yr
        R_1 = R - h_yr * 0.5  # inner radius of rotor cylinder
        k_1 = np.sqrt(I_r / A_r)  # radius of gyration
        m1 = (k_1 / R) ** 2
        c = R / 500  # (not used)

        u_allow_r = R / 10000  # allowable radial deflection
        b_allow_r = 2 * np.pi * R_sh / N_r  # allowable circumferential arm dimension

        # Calculating radial deflection of rotor structure according to Mc Donald's
        Numer = R**3 * (
            (0.25 * (np.sin(theta_r) - (theta_r * np.cos(theta_r))) / (np.sin(theta_r)) ** 2)
            - (0.5 / np.sin(theta_r))
            + (0.5 / theta_r)
        )
        Pov = ((theta_r / (np.sin(theta_r)) ** 2) + 1 / np.tan(theta_r)) * ((0.25 * R / A_r) + (0.25 * R**3 / I_r))
        Qov = R**3 / (2 * I_r * theta_r * (m1 + 1))
        Lov = (R_1 - R_sh) / a_r
        Denom = I_r * (Pov - Qov + Lov)  # radial deflection % rotor
        u_ar = (q3 * R**2 / E / h_yr) * (1 + Numer / Denom)

        # Calculating axial deflection of rotor structure

        w_r = rho_Fes * gravity * np.sin(phi) * a_r * N_r
        mass_st_lam = rho_Fe * 2 * np.pi * (R + 0.5 * h_yr) * len_s * h_yr  # mass of rotor yoke steel
        W = gravity * np.sin(phi) * (mass_st_lam + (V_Cusr * rho_Copper) + M_Fert) / N_r  # weight of rotor cylinder
        l_ir = R  # length of rotor arm beam at which rotor cylinder acts
        l_iir = R_1

        y_ar = (W * l_ir**3 / 12 / E / I_arm_axi_r) + (w_r * l_iir**4 / 24 / E / I_arm_axi_r)  # axial deflection

        # Calculating torsional deflection of rotor structure

        z_allow_r = np.deg2rad(0.05 * R)  # allowable torsional deflection
        z_ar = (
            (2 * np.pi * (R - 0.5 * h_yr) * len_s / N_r) * sigma * (l_ir - 0.5 * h_yr) ** 3 / (3 * E * I_arm_tor_r)
        )  # circumferential deflection

        # STATOR structure

        A_st = len_s * t_s
        a_s = (b_st * d_s) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws))
        N_st = np.round(n_s)
        theta_s = np.pi / N_st
        I_st = len_s * t_s**3 / 12
        I_arm_axi_s = (
            (b_st * d_s**3) - ((b_st - 2 * t_ws) * (d_s - 2 * t_ws) ** 3)
        ) / 12  # second moment of area of stator arm
        I_arm_tor_s = (
            (d_s * b_st**3) - ((d_s - 2 * t_ws) * (b_st - 2 * t_ws) ** 3)
        ) / 12  # second moment of area of rotot arm w.r.t torsion
        R_st = rad_ag + h_s + h_ys * 0.5
        R_1s = R_st - h_ys * 0.5
        k_2 = np.sqrt(I_st / A_st)
        m2 = (k_2 / R_st) ** 2

        # allowable deflections

        b_allow_s = 2 * np.pi * R_sh / N_st
        u_allow_s = R_st / 10000
        y_allow = 2 * len_s / 100  # allowable axial     deflection
        z_allow_s = np.deg2rad(0.05 * R_st)  # allowable torsional deflection

        # Calculating radial deflection according to McDonald's

        Numers = R_st**3 * (
            (0.25 * (np.sin(theta_s) - (theta_s * np.cos(theta_s))) / (np.sin(theta_s)) ** 2)
            - (0.5 / np.sin(theta_s))
            + (0.5 / theta_s)
        )
        Povs = ((theta_s / (np.sin(theta_s)) ** 2) + 1 / np.tan(theta_s)) * (
            (0.25 * R_st / A_st) + (0.25 * R_st**3 / I_st)
        )
        Qovs = R_st**3 / (2 * I_st * theta_s * (m2 + 1))
        Lovs = (R_1s - R_sh) * 0.5 / a_s
        Denoms = I_st * (Povs - Qovs + Lovs)
        R_out = R / 0.995 + h_s + h_ys
        u_as = (q3 * R_st**2 / E / t_s) * (1 + Numers / Denoms)

        # Calculating axial deflection according to McDonald

        l_is = R_st - R_sh
        l_iis = l_is
        l_iiis = l_is  # length of rotor arm beam at which self-weight acts
        mass_st_lam_s = M_Fest + np.pi * len_s * rho_Fe * ((R_st + 0.5 * h_ys) ** 2 - (R_st - 0.5 * h_ys) ** 2)
        W_is = gravity * np.sin(phi) * (rho_Fes * len_s * d_s**2 * 0.5)  # weight of rotor cylinder
        W_iis = gravity * np.sin(phi) * (V_Cuss * rho_Copper + mass_st_lam_s) / 2 / N_st
        w_s = rho_Fes * gravity * np.sin(phi) * a_s * N_st

        X_comp1 = W_is * l_is**3 / (12 * E * I_arm_axi_s)
        X_comp2 = W_iis * l_iis**4 / (24 * E * I_arm_axi_s)
        X_comp3 = w_s * l_iiis**4 / (24 * E * I_arm_axi_s)

        y_as = X_comp1 + X_comp2 + X_comp3  # axial deflection

        # Calculating torsional deflection

        z_as = (
            2
            * np.pi
            * (R_st + 0.5 * t_s)
            * len_s
            / (2 * N_st)
            * sigma
            * (l_is + 0.5 * t_s) ** 3
            / (3 * E * I_arm_tor_s)
        )

        # tangential stress constraints

        TC1 = T / (2 * np.pi * sigma)
        TC2r = R**2 * len_s
        TC2s = R_st**2 * len_s

        # Calculating inactive mass and total mass

        mass_stru_steel = 2 * N_st * (R_1s - R_sh) * a_s * rho_Fes
        Structural_mass = mass_stru_steel + (N_r * (R_1 - R_sh) * a_r * rho_Fes)
        Mass = Copper + Iron + Structural_mass

        outputs["B_symax"] = B_symax
        outputs["B_tmax"] = B_tmax
        outputs["B_rymax"] = B_rymax
        outputs["B_gfm"] = B_gfm
        outputs["B_g"] = B_g
        outputs["B_pc"] = B_pc
        outputs["N_s"] = N_s
        outputs["b_s"] = b_s

        outputs["b_t"] = b_t
        outputs["A_Cuscalc"] = A_Cuscalc
        outputs["A_Curcalc"] = A_Curcalc
        outputs["b_p"] = b_p
        outputs["h_p"] = h_p
        outputs["E_s"] = E_s
        outputs["f"] = f

        outputs["I_s"] = I_s
        outputs["R_s"] = R_s
        outputs["L_m"] = L_m
        outputs["A_1"] = A_1
        outputs["J_s"] = J_s
        outputs["R_r"] = R_r
        outputs["Losses"] = Losses

        outputs["Load_mmf_ratio"] = Load_mmf_ratio
        outputs["Power_ratio"] = Power_ratio
        outputs["n_brushes"] = n_brushes
        outputs["J_f"] = J_f
        outputs["K_rad"] = K_rad
        outputs["eandm_efficiency"] = np.maximum(eps, gen_eff)
        outputs["S"] = S

        outputs["Slot_aspect_ratio"] = Slot_aspect_ratio
        outputs["Copper"] = Copper
        outputs["Iron"] = Iron
        outputs["u_ar"] = u_ar
        outputs["y_ar"] = y_ar

        outputs["z_ar"] = z_ar
        outputs["u_as"] = u_as
        outputs["y_as"] = y_as
        outputs["z_as"] = z_as
        outputs["u_allow_r"] = u_allow_r
        outputs["u_allow_s"] = u_allow_s

        outputs["y_allow_r"] = outputs["y_allow_s"] = y_allow
        outputs["z_allow_s"] = z_allow_s
        outputs["z_allow_r"] = z_allow_r
        outputs["b_allow_s"] = b_allow_s
        outputs["b_allow_r"] = b_allow_r
        outputs["TC1"] = TC1

        outputs["TC2r"] = TC2r
        outputs["TC2s"] = TC2s
        outputs["R_out"] = R_out
        outputs["Structural_mass"] = Structural_mass
        outputs["generator_mass"] = Mass


# ----------------------------------------------------------------------------------------
