import numpy as np
from wisdem.commonse.constants import eps
from wisdem.commonse.utilities import smooth_max, smooth_min, cubic_spline_eval


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
    Ne = np.pi ** 2 * (E * I) / (1.1 * tower_height ** 2)
    lambda_bar = np.sqrt(Np * gamma_b / Ne)
    phi = 0.5 * (1 + alpha * (lambda_bar - 0.2) + lambda_bar ** 2)
    kappa = np.ones(A.shape)
    idx = lambda_bar > 0.2
    kappa[idx] = 1.0 / (phi[idx] + np.sqrt(phi[idx] ** 2 - lambda_bar[idx] ** 2))
    delta_n = 0.25 * kappa * lambda_bar ** 2
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
    constant = 1 + 1.83 / 1.7 - 2.07 / 1.7 ** 2

    ptL1 = 1.7 - 0.25
    ptR1 = 1.7 + 0.25

    ptL2 = 0.5 * rovert - 1.0
    ptR2 = 0.5 * rovert + 1.0

    ptL3 = (0.5 + Cxb) * rovert - 1.0
    ptR3 = (0.5 + Cxb) * rovert + 1.0

    if omega < ptL1:
        Cx = constant - 1.83 / omega + 2.07 / omega ** 2

    elif omega >= ptL1 and omega <= ptR1:

        fL = constant - 1.83 / ptL1 + 2.07 / ptL1 ** 2
        fR = 1.0
        gL = 1.83 / ptL1 ** 2 - 4.14 / ptL1 ** 3
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
        Cthetas = 1.5 + 10.0 / omega ** 2 - 5 / omega ** 3 - offset
        sigma = 0.92 * E * Cthetas / omega / rovert

    elif omega >= 20.0 * Ctheta and omega < ptL:

        sigma = 0.92 * E * Ctheta / omega / rovert

    elif omega >= ptL and omega <= ptR:

        alpha1 = 0.92 / 1.63 - 2.03 / 1.63 ** 4

        fL = 0.92 * E * Ctheta / ptL / rovert
        fR = E * (1.0 / rovert) ** 2 * (alpha1 + 2.03 * (Ctheta / ptR * rovert) ** 4)
        gL = -0.92 * E * Ctheta / rovert / ptL ** 2
        gR = -E * (1.0 / rovert) * 2.03 * 4 * (Ctheta / ptR * rovert) ** 3 * Ctheta / ptR ** 2

        sigma = cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, omega)

    else:

        alpha1 = 0.92 / 1.63 - 2.03 / 1.63 ** 4
        sigma = E * (1.0 / rovert) ** 2 * (alpha1 + 2.03 * (Ctheta / omega * rovert) ** 4)

    return sigma


def _tausmooth(omega, rovert):

    ptL1 = 9
    ptR1 = 11

    ptL2 = 8.7 * rovert - 1
    ptR2 = 8.7 * rovert + 1

    if omega < ptL1:
        C_tau = np.sqrt(1.0 + 42.0 / omega ** 3 - 42.0 / 10 ** 3)

    elif omega >= ptL1 and omega <= ptR1:
        fL = np.sqrt(1.0 + 42.0 / ptL1 ** 3 - 42.0 / 10 ** 3)
        fR = 1.0
        gL = -63.0 / ptL1 ** 4 / fL
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
        fR = 1 - beta * fracR ** eta
        gL = 0.0
        gR = -beta * eta * fracR ** (eta - 1) / (lambda_p - lambda_0)

        chi = cubic_spline_eval(ptL, ptR, fL, fR, gL, gR, lambda_bar)

    elif lambda_bar > ptR and lambda_bar < lambda_p:
        chi = 1.0 - beta * ((lambda_bar - lambda_0) / (lambda_p - lambda_0)) ** eta

    else:
        chi = alpha / lambda_bar ** 2

    # if (lambda_bar <= lambda_0):
    #     chi = 1.0
    # elif (lambda_bar >= lambda_p):
    #     chi = alpha/lambda_bar**2
    # else:
    #     chi = 1.0 - beta*((lambda_bar-lambda_0)/(lambda_p-lambda_0))**eta

    return chi
