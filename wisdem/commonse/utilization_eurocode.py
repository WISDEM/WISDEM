import numpy as np

from wisdem.commonse.constants import eps
from wisdem.commonse.utilities import smooth_max, smooth_min, cubic_spline_eval
from scipy.optimize import fsolve


def hoopStressEurocode(d, t, L_reinforced, hoop):
    """default method for scaling hoop stress using Eurocode method
    GB 06/21/2018: Ansys comparisons for submerged case suggests this over-compensates for stiffener
                   I'm not even sure the Eurocode is implemented correctly here.  Suggest using the standard
                   hoop stress expression above or API's handling of ring stiffeners below.
    """

    r = d / 2.0 - t / 2.0  # radius of cylinder middle surface
    omega = L_reinforced / np.sqrt(r * t)

    C_theta = 1.5  # clamped-clamped
    k_w = 0.46 * (1.0 + 0.1 * np.sqrt(C_theta / omega * r / t))
    k_w, _, _ = smooth_max(k_w, 0.65)
    k_w, _, _ = smooth_min(k_w, 1.0)
    return k_w * hoop

def YieldCriterionEurocode(sigma_x, sigma_z, tau, gamma, sigma_y):
    """Yield criterion from EN1993 1-1 (6.1)
    """

    a = (sigma_x*gamma/sigma_y)**2
    b = (sigma_z*gamma/sigma_y)**2
    c = (sigma_x*gamma/sigma_y)*(sigma_z*gamma/sigma_y)
    d = 3*(tau*gamma/sigma_y)**2

    # stress margin   
    stress_utilization = a+b-c+d

    return stress_utilization


def bucklingGL(d, t, Fz, Myy, tower_height, E, sigma_y, gamma_f=1.2, gamma_b=1.1):
    # other factors
    alpha = 0.21  # buckling imperfection factor
    beta = 1.0  # bending coefficient
    sk_factor = 2.0  # fixed-free
    tower_height = tower_height * sk_factor

    # geometry
    A = np.pi * d * t
    I = np.pi * (d / 2.0) ** 3 * t
    Wp = I / (d / 2.0)

    # applied loads
    Nd = -Fz * gamma_f
    Md = Myy * gamma_f

    # plastic resistance
    Np = A * sigma_y / gamma_b
    Mp = Wp * sigma_y / gamma_b

    # factors
    Ne = np.pi**2 * (E * I) / (1.1 * tower_height**2)
    lambda_bar = np.sqrt(Np * gamma_b / Ne)
    phi = 0.5 * (1 + alpha * (lambda_bar - 0.2) + lambda_bar**2)
    kappa = np.ones(A.shape)
    idx = lambda_bar > 0.2
    kappa[idx] = 1.0 / (phi[idx] + np.sqrt(phi[idx] ** 2 - lambda_bar[idx] ** 2))
    delta_n = 0.25 * kappa * lambda_bar**2
    delta_n = np.minimum(delta_n, 0.1)

    GL_utilization = Nd / (kappa * Np) + beta * Md / Mp + delta_n  # this is utilization must be <1

    return GL_utilization


def shellBucklingEurocode(d, t, sigma_z, sigma_t, tau_zt, L_reinforced, E, sigma_y, gamma_f=1.2, gamma_b=1.1):
    """
    Estimate shell buckling utilization along tower.

    Arguments:
    npt - number of locations at each node at which stress is evaluated.
    sigma_z - axial stress at npt*node locations.  must be in order
                  [(node1_pts1-npt), (node2_pts1-npt), ...]
    sigma_t - azimuthal stress given at npt*node locations
    tau_zt - shear stress (z, theta) at npt*node locations
    E - modulus of elasticity
    sigma_y - yield stress
    L_reinforced - reinforcement length - structure is re-discretized with this spacing
    gamma_f - safety factor for stresses
    gamma_b - safety factor for buckling

    Returns:
    z
    EU_utilization: - array of shell buckling utilizations evaluted at (z[0] at npt locations, \n
                      z[0]+L_reinforced at npt locations, ...). \n
                      Each utilization must be < 1 to avoid failure.
    """

    n = len(t)
    EU_utilization = np.zeros(n)
    sigma_z_sh = np.zeros(n)
    sigma_t_sh = np.zeros(n)
    tau_zt_sh = np.zeros(n)

    for i in range(n):
        h = L_reinforced[i]

        r1 = d[i] / 2.0 - t[i] / 2.0
        r2 = d[i + 1] / 2.0 - t[i] / 2.0

        sigma_z_shell = sigma_z[i]
        sigma_t_shell = sigma_t[i]
        tau_zt_shell = tau_zt[i]

        # TODO: the following is non-smooth, although in general its probably OK
        # change to magnitudes and add safety factor
        sigma_z_shell = gamma_f * abs(sigma_z_shell)
        sigma_t_shell = gamma_f * abs(sigma_t_shell)
        tau_zt_shell = gamma_f * abs(tau_zt_shell)

        EU_utilization[i] = _shellBucklingOneSection(
            h, r1, r2, t[i], gamma_b, sigma_z_shell, sigma_t_shell, tau_zt_shell, E[i], sigma_y[i]
        )

        # make them into vectors
        sigma_z_sh[i] = sigma_z_shell
        sigma_t_sh[i] = sigma_t_shell
        tau_zt_sh[i] = tau_zt_shell

    return EU_utilization  # this is utilization must be <1


def _cxsmooth(omega, rovert):
    Cxb = 6.0  # clamped-clamped
    constant = 1 + 1.83 / 1.7 - 2.07 / 1.7**2

    ptL1 = 1.7 - 0.25
    ptR1 = 1.7 + 0.25

    ptL2 = 0.5 * rovert - 1.0
    ptR2 = 0.5 * rovert + 1.0

    ptL3 = (0.5 + Cxb) * rovert - 1.0
    ptR3 = (0.5 + Cxb) * rovert + 1.0

    if omega < ptL1:
        Cx = constant - 1.83 / omega + 2.07 / omega**2

    elif omega >= ptL1 and omega <= ptR1:
        fL = constant - 1.83 / ptL1 + 2.07 / ptL1**2
        fR = 1.0
        gL = 1.83 / ptL1**2 - 4.14 / ptL1**3
        gR = 0.0
        Cx = cubic_spline_eval(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        Cx = 1.0

    elif omega >= ptL2 and omega <= ptR2:
        fL = 1.0
        fR = 1 + 0.2 / Cxb * (1 - 2.0 * ptR2 / rovert)
        gL = 0.0
        gR = -0.4 / Cxb / rovert
        Cx = cubic_spline_eval(ptL2, ptR2, fL, fR, gL, gR, omega)

    elif omega > ptR2 and omega < ptL3:
        Cx = 1 + 0.2 / Cxb * (1 - 2.0 * omega / rovert)

    elif omega >= ptL3 and omega <= ptR3:
        fL = 1 + 0.2 / Cxb * (1 - 2.0 * ptL3 / rovert)
        fR = 0.6
        gL = -0.4 / Cxb / rovert
        gR = 0.0
        Cx = cubic_spline_eval(ptL3, ptR3, fL, fR, gL, gR, omega)

    else:
        Cx = 0.6

    return Cx


def _sigmasmooth(omega, E, rovert):
    Ctheta = 1.5  # clamped-clamped

    ptL = 1.63 * rovert * Ctheta - 1
    ptR = 1.63 * rovert * Ctheta + 1

    if omega < 20.0 * Ctheta:
        offset = 10.0 / (20 * Ctheta) ** 2 - 5 / (20 * Ctheta) ** 3
        Cthetas = 1.5 + 10.0 / omega**2 - 5 / omega**3 - offset
        sigma = 0.92 * E * Cthetas / omega / rovert

    elif omega >= 20.0 * Ctheta and omega < ptL:
        sigma = 0.92 * E * Ctheta / omega / rovert

    elif omega >= ptL and omega <= ptR:
        alpha1 = 0.92 / 1.63 - 2.03 / 1.63**4

        fL = 0.92 * E * Ctheta / ptL / rovert
        fR = E * (1.0 / rovert) ** 2 * (alpha1 + 2.03 * (Ctheta / ptR * rovert) ** 4)
        gL = -0.92 * E * Ctheta / rovert / ptL**2
        gR = -E * (1.0 / rovert) * 2.03 * 4 * (Ctheta / ptR * rovert) ** 3 * Ctheta / ptR**2

        sigma = cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, omega)

    else:
        alpha1 = 0.92 / 1.63 - 2.03 / 1.63**4
        sigma = E * (1.0 / rovert) ** 2 * (alpha1 + 2.03 * (Ctheta / omega * rovert) ** 4)

    return sigma


def _tausmooth(omega, rovert):
    ptL1 = 9
    ptR1 = 11

    ptL2 = 8.7 * rovert - 1
    ptR2 = 8.7 * rovert + 1

    if omega < ptL1:
        C_tau = np.sqrt(1.0 + 42.0 / omega**3 - 42.0 / 10**3)

    elif omega >= ptL1 and omega <= ptR1:
        fL = np.sqrt(1.0 + 42.0 / ptL1**3 - 42.0 / 10**3)
        fR = 1.0
        gL = -63.0 / ptL1**4 / fL
        gR = 0.0
        C_tau = cubic_spline_eval(ptL1, ptR1, fL, fR, gL, gR, omega)

    elif omega > ptR1 and omega < ptL2:
        C_tau = 1.0

    elif omega >= ptL2 and omega <= ptR2:
        fL = 1.0
        fR = 1.0 / 3.0 * np.sqrt(ptR2 / rovert) + 1 - np.sqrt(8.7) / 3
        gL = 0.0
        gR = 1.0 / 6 / np.sqrt(ptR2 * rovert)
        C_tau = cubic_spline_eval(ptL2, ptR2, fL, fR, gL, gR, omega)

    else:
        C_tau = 1.0 / 3.0 * np.sqrt(omega / rovert) + 1 - np.sqrt(8.7) / 3

    return C_tau


def _shellBucklingOneSection(h, r1, r2, t, gamma_b, sigma_z, sigma_t, tau_zt, E, sigma_y):
    """
    Estimate shell buckling for one tapered cylindrical shell section.

    Arguments:
    h - height of conical section
    r1 - radius at bottom
    r2 - radius at top
    t - shell thickness
    E - modulus of elasticity
    sigma_y - yield stress
    gamma_b - buckling reduction safety factor
    sigma_z - axial stress component
    sigma_t - azimuthal stress component
    tau_zt - shear stress component (z, theta)

    Returns:
    EU_utilization, shell buckling utilization which must be < 1 to avoid failure

    """

    # NOTE: definition of r1, r2 switched from Eurocode document to be consistent with FEM.

    # ----- geometric parameters --------
    beta = np.arctan2(r1 - r2, h)
    L = h / np.cos(beta)

    # ------------- axial stress -------------
    # length parameter
    le = L
    re = 0.5 * (r1 + r2) / np.cos(beta)
    omega = le / np.sqrt(re * t)
    rovert = re / t

    # compute Cx
    Cx = _cxsmooth(omega, rovert)

    # if omega <= 1.7:
    #     Cx = 1.36 - 1.83/omega + 2.07/omega/omega
    # elif omega > 0.5*rovert:
    #     Cxb = 6.0  # clamped-clamped
    #     Cx = max(0.6, 1 + 0.2/Cxb*(1-2.0*omega/rovert))
    # else:
    #     Cx = 1.0

    # critical axial buckling stress
    sigma_z_Rcr = 0.605 * E * Cx / rovert

    # compute buckling reduction factors
    lambda_z0 = 0.2
    beta_z = 0.6
    eta_z = 1.0
    Q = 25.0  # quality parameter - high
    lambda_z = np.sqrt(sigma_y / sigma_z_Rcr)
    delta_wk = 1.0 / Q * np.sqrt(rovert) * t
    alpha_z = 0.62 / (1 + 1.91 * (delta_wk / t) ** 1.44)

    chi_z = _buckling_reduction_factor(alpha_z, beta_z, eta_z, lambda_z0, lambda_z)

    # design buckling stress
    sigma_z_Rk = chi_z * sigma_y
    sigma_z_Rd = sigma_z_Rk / gamma_b

    # ---------------- hoop stress ------------------

    # length parameter
    le = L
    re = 0.5 * (r1 + r2) / (np.cos(beta))
    omega = le / np.sqrt(re * t)
    rovert = re / t

    # Ctheta = 1.5  # clamped-clamped
    # CthetaS = 1.5 + 10.0/omega**2 - 5.0/omega**3

    # # critical hoop buckling stress
    # if (omega/Ctheta < 20.0):
    #     sigma_t_Rcr = 0.92*E*CthetaS/omega/rovert
    # elif (omega/Ctheta > 1.63*rovert):
    #     sigma_t_Rcr = E*(1.0/rovert)**2*(0.275 + 2.03*(Ctheta/omega*rovert)**4)
    # else:
    #     sigma_t_Rcr = 0.92*E*Ctheta/omega/rovert

    sigma_t_Rcr = np.maximum(eps, _sigmasmooth(omega, E, rovert))

    # buckling reduction factor
    alpha_t = 0.65  # high fabrication quality
    lambda_t0 = 0.4
    beta_t = 0.6
    eta_t = 1.0
    lambda_t = np.sqrt(sigma_y / sigma_t_Rcr)

    chi_theta = _buckling_reduction_factor(alpha_t, beta_t, eta_t, lambda_t0, lambda_t)

    sigma_t_Rk = chi_theta * sigma_y
    sigma_t_Rd = sigma_t_Rk / gamma_b

    # ----------------- shear stress ----------------------

    # length parameter
    le = h
    rho = np.sqrt((r1 + r2) / (2.0 * r2))
    re = (1.0 + rho - 1.0 / rho) * r2 * np.cos(beta)
    omega = le / np.sqrt(re * t)
    rovert = re / t

    # if (omega < 10):
    #     C_tau = np.sqrt(1.0 + 42.0/omega**3)
    # elif (omega > 8.7*rovert):
    #     C_tau = 1.0/3.0*np.sqrt(omega/rovert)
    # else:
    #     C_tau = 1.0
    C_tau = _tausmooth(omega, rovert)

    tau_zt_Rcr = 0.75 * E * C_tau * np.sqrt(1.0 / omega) / rovert

    # reduction factor
    alpha_tau = 0.65  # high fabrifaction quality
    beta_tau = 0.6
    lambda_tau0 = 0.4
    eta_tau = 1.0
    lambda_tau = np.sqrt(sigma_y / np.sqrt(3) / tau_zt_Rcr)

    chi_tau = _buckling_reduction_factor(alpha_tau, beta_tau, eta_tau, lambda_tau0, lambda_tau)

    tau_zt_Rk = chi_tau * sigma_y / np.sqrt(3)
    tau_zt_Rd = tau_zt_Rk / gamma_b

    # buckling interaction parameters

    k_z = 1.25 + 0.75 * chi_z
    k_theta = 1.25 + 0.75 * chi_theta
    k_tau = 1.75 + 0.25 * chi_tau
    k_i = (chi_z * chi_theta) ** 2

    # shell buckling utilization

    utilization = (
        (sigma_z / sigma_z_Rd) ** k_z
        + (sigma_t / sigma_t_Rd) ** k_theta
        - k_i * (sigma_z * sigma_t / sigma_z_Rd / sigma_t_Rd)
        + (tau_zt / tau_zt_Rd) ** k_tau
    )

    return utilization  # this is utilization must be <1


def _buckling_reduction_factor(alpha, beta, eta, lambda_0, lambda_bar):
    """
    Computes a buckling reduction factor used in Eurocode shell buckling formula.
    """

    lambda_p = np.sqrt(alpha / (1.0 - beta))

    ptL = 0.9 * lambda_0
    ptR = 1.1 * lambda_0

    if lambda_bar < ptL:
        chi = 1.0

    elif lambda_bar >= ptL and lambda_bar <= ptR:  # cubic spline section
        fracR = (ptR - lambda_0) / (lambda_p - lambda_0)
        fL = 1.0
        fR = 1 - beta * fracR**eta
        gL = 0.0
        gR = -beta * eta * fracR ** (eta - 1) / (lambda_p - lambda_0)

        chi = cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, lambda_bar)

    elif lambda_bar > ptR and lambda_bar < lambda_p:
        chi = 1.0 - beta * ((lambda_bar - lambda_0) / (lambda_p - lambda_0)) ** eta

    else:
        chi = alpha / lambda_bar**2

    # if (lambda_bar <= lambda_0):
    #     chi = 1.0
    # elif (lambda_bar >= lambda_p):
    #     chi = alpha/lambda_bar**2
    # else:
    #     chi = 1.0 - beta*((lambda_bar-lambda_0)/(lambda_p-lambda_0))**eta

    return chi


class memberBuckling:
    """
    class for calculating member buckling based on EN 1993 1-1

    """

    def __init__(
        self,
        a,
        b,
        l,
        A,
        Ixx,
        Iyy,
        Fz,
        Mx,
        My,
        I_T,
        E=200e9,
        G=79.3e9,
        sigma_y=345e6,
        gamma_m=1.1,
        kw=0.5,
        Iw=0,
        k=0.5,
        **kwargs,
    ):
        """
        create an instance of memberBuckling

        Args:
            l   (np.array): section length
            A   (np.array): Areas of rectangular member section
            Ixx (np.array): Moment of inertia x
            Iyy (np.array): Moment of inertia y
            IT (float): St Venent torsional constant
            kw (float): factor allowing for end constraint, default to full fixity 0.5
            E   (float): Isotropic Young's modulus [Pa]. Defaults to 200e9.
            G   (float, optional): Shear modulus [Pa]. Defaults to 79.3e9.
            sigma_y (float): Yield strength [Pa]. Defaults to 345e6.
            gamma_m (float, optional): Partial factor for resistance of members to instability. Defaults to 1.1 by DNVGL.
        """
        self.a = a
        self.b = b
        self.A = A
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Fz = Fz
        self.Mx = Mx
        self.My = My
        self.E = E
        self.l = l
        self.I_T = I_T
        self.Iw = Iw
        self.k = k
        self.kw = kw
        self.G = G
        self.sigma_y = sigma_y
        self.gamma_m = gamma_m

        self.k = 0.5 # factor due to end constraint
        # The design mormal force is compression
        self.Ned = Fz # Might need to do Fz[:-1]-Fz[1:] as axial loads

        # Elastic section modulus =  ratio of the second moment of area and the distance from the neutral axis
        # to any given fiber
        self.Welx = self.Ixx/(self.a/2)
        self.Wely = self.Iyy/(self.b/2)

        # gyration radius
        self.ix = np.sqrt((self.Ixx/self.A))
        self.iy = np.sqrt((self.Iyy/self.A))

        # distance to shear center
        self.x0 = 0
        self.y0 = 0

        self.elastic_section_modulus()
        self.critical_loads()
        self.characteristic_resistance()
        self.reduction_factor()
        self.interaction_factor()
        

    def critical_loads(self):

        self.Ncrx = np.pi**2*self.E*self.Ixx/(self.k*self.l)
        self.Ncry = np.pi**2*self.E*self.Iyy/(self.k*self.l)
        # NcrT: elastic torsional-flexural buckling force
        # Reference: https://help.scia.net/19.1/en/pvt/steelcodechecktb/en/torsional_flexural_buckling.htm
        self.i02 = self.ix**2+self.iy**2+self.x0**2+self.y0**2
        self.NcrT = 1.0/self.i02*(self.G*self.I_T+np.pi**2*self.E*self.Iw/(self.k*self.l)**2)
        fak = fsolve(self.NcrTF_eqn, self.NcrT, xtol=1e-4, maxfev=50)
        self.NcrTF = fak

        # I_T is the St Venent torsinal constant
        # For rectangular thin-walled with uniform thickness I_T = 2*t**2*(a-t)**2*(b-t)**2/(a*t+b*t-2*t**2)
        # Iw is the warping constant
        # For rectangular, Iw is usually zero
        # kw is factor allowing for end constraint, refers to end support warping, 1 for no fixity and 0.5 for full fixity
        self.Mcrx = self.Ncrx*np.sqrt(self.k/self.kw*self.Iw/self.Ixx+1/self.Ncrx*self.G*self.I_T/self.Ixx)
        self.Mcry = self.Ncry*np.sqrt(self.k/self.kw*self.Iw/self.Iyy+1/self.Ncry*self.G*self.I_T/self.Iyy)

    def NcrTF_eqn(self, Nload):
        return self.i02*(Nload-self.Ncrx)*(Nload-self.NcrT) - Nload**2*self.y0**2*(Nload-self.Ncrx) - Nload**2*self.x0**2*(Nload-self.Ncry)

    def elastic_section_modulus(self):
        self.Wx = self.Ixx*2/self.b
        self.Wy = self.Iyy*2/self.a

    def characteristic_resistance(self):
        self.Nrk = self.A*self.sigma_y
        self.Mxrk = self.Ixx*self.sigma_y
        self.Myrk = self.Iyy*self.sigma_y

    def interaction_factor(self):
        alpha_LT_x = np.max(1-self.I_T/self.Ixx, 0)
        alpha_LT_y = np.max(1-self.I_T/self.Iyy, 0)

        # Equivalent uniform moment, EN 1993 1-1 Table A.2
        # Psi_x = self.Mx[1:]/self.Mx[:-1]
        # Psi_y = self.My[1:]/self.My[:-1]
        Psi_x = np.ones(len(self.a))
        Psi_y = np.ones_like(Psi_x)

        cmx0 = 0.79+0.21*Psi_x+0.36*(Psi_x-0.33)*self.Ned/self.Ncrx
        cmy0 = 0.79+0.21*Psi_y+0.36*(Psi_y-0.33)*self.Ned/self.Ncry

        mux = (1-self.Ned/self.Ncrx)/(1-self.Chi_x*self.Ned/self.Ncrx)
        muy = (1-self.Ned/self.Ncry)/(1-self.Chi_y*self.Ned/self.Ncry)


        kc_x = 1/(1.33-0.33*Psi_x)
        kc_y = 1/(1.33-0.33*Psi_y)

        C1_x = kc_x**(-2) # is a factor depending on the loading and end conditions
        C1_y = kc_y**(-2) # is a factor depending on the loading and end conditions

        lambda_crit_x = 0.2*np.sqrt(C1_x)*((1-self.Ned/self.Ncry)*(1-self.Ned/self.NcrT))**0.25
        lambda_crit_y = 0.2*np.sqrt(C1_x)*((1-self.Ned/self.Ncrx)*(1-self.Ned/self.NcrT))**0.25

        lambda_0_x =  0.2*((1-self.Ned/self.Ncry)*(1-self.Ned/self.NcrT))**0.25
        lambda_0_y =  0.2*((1-self.Ned/self.Ncrx)*(1-self.Ned/self.NcrT))**0.25

        # Take the moments directly as the maximum design values of bending moments
        Mxed = self.Mx
        Myed = self.My

        esp_x = Mxed/self.Ned*self.A/self.Welx
        esp_y = Myed/self.Ned*self.A/self.Wely

        self.kxx = np.zeros_like(self.Ned)
        self.kyy = np.zeros_like(self.kxx)
        self.kxy = np.zeros_like(self.kxx)
        self.kyx = np.zeros_like(self.kxx)


        for i in range(len(cmx0)):

            if lambda_0_x[i] <= lambda_crit_x[i]:
                cmx = cmx0[i]
                cmy = cmy0[i]
                cmLT=1.0

            else:
                cmx = cmx0[i]+(1-cmx0[i])*np.sqrt(esp_x[i])*alpha_LT_x[i]/(1+np.sqrt(esp_x[i])*alpha_LT_x[i])
                cmy = cmy0[i]
                cmLT = cmy**2*alpha_LT_x[i]/(np.sqrt((1-self.Ned[i]/self.Ncry[i])*(1-self.Ned[i]/self.NcrTF[i])))

            self.kxx[i] = cmx*cmLT*mux[i]/(1-self.Ned[i]/self.Ncrx[i])
            self.kxy[i] = cmy*mux[i]/(1-self.Ned[i]/self.Ncry[i])

            if lambda_0_y[i] <= lambda_crit_y[i]:
                cmx = cmx0[i]
                cmy = cmy0[i]
                cmLT=1.0

            else:
                cmx = cmx0[i]
                cmy = cmy0[i]+(1-cmy0[i])*np.sqrt(esp_y[i])*alpha_LT_y[i]/(1+np.sqrt(esp_y[i])*alpha_LT_y[i])
                cmLT = cmy**2*alpha_LT_y[i]/(np.sqrt((1-self.Ned[i]/self.Ncrx[i])*(1-self.Ned[i]/self.NcrTF[i])))

            self.kyx[i] = cmx*cmLT*muy[i]/(1-self.Ned[i]/self.Ncrx[i])
            self.kyy[i] = cmy*muy[i]/(1-self.Ned[i]/self.Ncry[i])

    def _reduction_factor(self, lambda_slender, Psi, alpha_imperf=0.49):
        """
        Compute the reduction factor due to flexural buckling or lateral torsion buckling

        Args:
            lambda_slender (float): Slenderness due to flexural buckling or lateral torsion
            alpha_imperf (float): Imperfection factor, default 0.49, EN1993 1-1 Table 6.1 and 6.2
        retuun:
            reduction factors
        """

        Phi = 0.5*(1+alpha_imperf*(lambda_slender-0.2)+lambda_slender**2)
        Chi = 1/(Phi+np.sqrt(Phi**2-lambda_slender**2))

        # kc is the correction factor
        # Taking into account the moment distribution between the lateral restraints of members
        # See Table 6.6 in Eurocode 
        # the reduction factor may be modified as follows
        # Only consider linear variation
        kc = 1/(1.33-0.33*Psi)

        f = 1.0 - 0.5*(1-kc)*(1-2.0*(lambda_slender-0.8)**2)

        return Chi
    
    def reduction_factor(self, alpha_imperf=0.49):

        # Reduction factors
        # Because the moments are on the sections, there is not end moment compute Psi
        # Currecntly assume uniform distribution, so Psi=1
        # Psi_x = self.Mx[1:]/self.Mx[:-1]
        # Psi_y = self.My[1:]/self.My[:-1]
        Psi_x = np.ones(len(self.a))
        Psi_y = np.ones_like(Psi_x)
        lambda_slender_x, lambda_slender_y = self.flexural_slender()
        self.Chi_x = self._reduction_factor(lambda_slender_x, Psi=Psi_x, alpha_imperf=alpha_imperf)
        self.Chi_y = self._reduction_factor(lambda_slender_y, Psi=Psi_y, alpha_imperf=alpha_imperf)
        lambda_slender_x, lambda_slender_y = self.lateral_torsion_slender()
        self.Chi_LT_x = self._reduction_factor(lambda_slender_x, Psi=Psi_x, alpha_imperf=alpha_imperf)
        self.Chi_LT_y = self._reduction_factor(lambda_slender_y, Psi=Psi_y, alpha_imperf=alpha_imperf)


    def utilization(self, Ned, N_rk, Mx_ed, Mx_rk, My_ed, My_rk):
        """
        Compute the buckling constraints

        Args:
            Ned (np.array): Design values of the compression force and the maximum moments about the x-x and y-y axes
            N_rk (float): Characteristic resistance to normal force of the critical cross sections
            Mx_ed (float):  Design bending moment
            Mx_rk (float): Characteristic moment resistance of the critical cross sections
            My_ed (float): Design bending moment
            My_rk (float): Characteristic moment resistance of the critical cross sections
        """
        # Ignore the shift of the centroidal axes
        # The subscript of force and moment are modified from EN 1993 definition to be consistent within
        # WISDEM

        a = Ned*self.gamma_m/self.Chi_x/N_rk
        b = self.kxx*Mx_ed*self.gamma_m/self.Chi_LT_x/Mx_rk
        c = self.kxy*My_ed*self.gamma_m/self.Chi_LT_x/My_rk

        utilization_x = a + b + c

        a = Ned*self.gamma_m/self.Chi_y/N_rk
        b = self.kyx*Mx_ed*self.gamma_m/self.Chi_LT_y/Mx_rk
        c = self.kyy*My_ed*self.gamma_m/self.Chi_LT_y/My_rk

        utilization_y = a + b + c

        return utilization_x, utilization_y
    
    def run_buckling_checks(self, Fz, Mxx, Myy):

        Ned = Fz
        N_rk = self.A*self.sigma_y/self.gamma_m
        Mx_ed = Mxx
        My_ed = Myy
        Mx_rk = self.Welx*self.sigma_y/self.gamma_m
        My_rk = self.Wely*self.sigma_y/self.gamma_m

        utilization_x, utilization_y = self.utilization(Ned, N_rk, Mx_ed, Mx_rk, My_ed, My_rk)

        return np.maximum(utilization_x, utilization_y)

    
    def flexural_slender(self):

        lambda_slender_x = np.sqrt(self.A*self.sigma_y/self.Ncrx)
        lambda_slender_y = np.sqrt(self.A*self.sigma_y/self.Ncry)
        
        return lambda_slender_x, lambda_slender_y
    
    def lateral_torsion_slender(self):

        lambda_slender_x = np.sqrt(self.Wx*self.sigma_y/self.Mcrx)
        lambda_slender_y = np.sqrt(self.Wy*self.sigma_y/self.Mcry)

        return lambda_slender_x, lambda_slender_y
    

