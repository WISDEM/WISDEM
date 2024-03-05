import numpy as np
from scipy.optimize import brentq, minimize_scalar

from wisdem.commonse.constants import gravity


def _TBeamProperties(h_web, t_web, w_flange, t_flange):
    """Computes T-cross section area, CG, and moments of inertia
    See: http://www.amesweb.info/SectionalPropertiesTabs/SectionalPropertiesTbeam.aspx

    INPUTS:
    ----------
    h_web    : float (scalar/vector),  web (T-base) height
    t_web    : float (scalar/vector),  web (T-base) thickness
    w_flange : float (scalar/vector),  flange (T-top) width/height
    t_flange : float (scalar/vector),  flange (T-top) thickness

    OUTPUTS:
    -------
    area : float (scalar/vector),  T-cross sectional area
    y_cg : float (scalar/vector),  Position of CG along y-axis (extending from base up through the T)
    Ixx  : float (scalar/vector),  Moment of intertia around axis parallel to flange, through y_cg
    Iyy  : float (scalar/vector),  Moment of intertia around y-axis
    """
    # Area of T cross section is sum of the two rectangles
    area_web = h_web * t_web
    area_flange = w_flange * t_flange
    area = area_web + area_flange
    # Y-position of the center of mass (Yna) measured from the base
    y_cg = ((h_web + 0.5 * t_flange) * area_flange + 0.5 * h_web * area_web) / area
    # Moments of inertia: y-axis runs through base (spinning top),
    # x-axis runs parallel to flange through cg
    Iyy = (area_web * t_web**2 + area_flange * w_flange**2) / 12.0
    Ixx = (
        (area_web * h_web**2 + area_flange * t_flange**2) / 12.0
        + area_web * (y_cg - 0.5 * h_web) ** 2
        + area_flange * (h_web + 0.5 * t_flange - y_cg) ** 2
    )
    return area, y_cg, Ixx, Iyy


def _IBeamProperties(h_web, t_web, w_flange, t_flange, w_base, t_base):
    """Computes uneven I-cross section area, CG
    See: http://www.amesweb.info/SectionalPropertiesTabs/SectionalPropertiesTbeam.aspx

    INPUTS:
    ----------
    h_web    : float (scalar/vector),  web (I-stem) height
    t_web    : float (scalar/vector),  web (I-stem) thickness
    w_flange : float (scalar/vector),  flange (I-top) width/height
    t_flange : float (scalar/vector),  flange (I-top) thickness
    w_base   : float (scalar/vector),  base (I-bottom) width/height
    t_base   : float (scalar/vector),  base (I-bottom) thickness

    OUTPUTS:
    -------
    area : float (scalar/vector),  T-cross sectional area
    y_cg : float (scalar/vector),  Position of CG along y-axis (extending from base up through the T)
    """
    # Area of T cross section is sum of the two rectangles
    area_web = h_web * t_web
    area_flange = w_flange * t_flange
    area_base = w_base * t_base
    area = area_web + area_flange + area_base
    # Y-position of the center of mass (Yna) measured from the base
    y_cg = (
        (t_base + h_web + 0.5 * t_flange) * area_flange + (t_base + 0.5 * h_web) * area_web + 0.5 * t_base * area_base
    ) / area
    return area, y_cg


def _compute_applied_axial(R_od, t_wall, m_stack, section_mass):
    """Compute axial stress for spar from z-axis loading

    INPUTS:
    ----------
    params       : dictionary of input parameters
    section_mass : float (scalar/vector),  mass of each spar section as axial loading increases with spar depth

    OUTPUTS:
    -------
    stress   : float (scalar/vector),  axial stress
    """

    R = R_od - 0.5 * t_wall
    # Add in weight of sections above it
    axial_load = m_stack + np.r_[0.0, np.cumsum(section_mass[:-1])]
    # Divide by shell cross sectional area to get stress
    return gravity * axial_load / (2.0 * np.pi * R * t_wall)


def _compute_applied_hoop(pressure, R_od, t_wall):
    """Compute hoop stress WITHOUT accounting for stiffener rings

    INPUTS:
    ----------
    pressure : float (scalar/vector),  radial (hydrostatic) pressure
    R_od     : float (scalar/vector),  radius to outer wall of shell
    t_wall   : float (scalar/vector),  shell wall thickness

    OUTPUTS:
    -------
    stress   : float (scalar/vector),  hoop stress with no stiffeners
    """
    return pressure * R_od / t_wall


def _compute_stiffener_factors(
    pressure, axial_stress, R_od, t_wall, h_web, t_web, w_flange, t_flange, L_stiffener, E, nu
):
    """Compute modifiers to stress due to presence of stiffener rings.

    INPUTS:
    ----------
    params       : dictionary of input parameters
    pressure     : float (scalar/vector),  radial (hydrostatic) pressure
    axial_stress : float (scalar/vector),  axial loading (z-axis) stress

    OUTPUTS:
    -------
    stiffener_factor_KthL : float (scalar/vector),  Stress modifier from stiffeners for local buckling from axial loads
    stiffener_factor_KthG : float (scalar/vector),  Stress modifier from stiffeners for general buckling from external pressure
    """

    # Geometry computations
    R_flange = R_od - h_web  # Should have "- t_wall", but not in appendix B
    area_stiff, y_cg, Ixx, Iyy = _TBeamProperties(h_web, t_web, w_flange, t_flange)
    t_stiff = area_stiff / h_web  # effective thickness(width) of stiffener section

    # Compute hoop stress modifiers accounting for stiffener rings
    # This has to be done at midpoint between stiffeners and at stiffener location itself
    # Compute beta (just a local term used here)
    D = E * t_wall**3 / (12.0 * (1 - nu * nu))
    beta = (0.25 * E * t_wall / R_od**2 / D) ** 0.25
    # Compute psi-factor (just a local term used here)
    u = np.minimum(0.5 * beta * L_stiffener, 30.0)
    psi_k = 2.0 * (np.sin(u) * np.cosh(u) + np.cos(u) * np.sinh(u)) / (np.sinh(2 * u) + np.sin(2 * u))

    # Compute a couple of other local terms
    u = np.minimum(beta * L_stiffener, 30.0)
    k_t = 8 * beta**3 * D * (np.cosh(u) - np.cos(u)) / (np.sinh(u) + np.sin(u))
    k_d = E * t_stiff * (R_od**2 - R_flange**2) / R_od / ((1 + nu) * R_od**2 + (1 - nu) * R_flange**2)

    # Pressure from axial load
    pressure_sigma = pressure - nu * axial_stress * t_wall / R_od

    # Compute the correction to hoop stress due to the presesnce of ring stiffeners
    stiffener_factor_KthL = 1 - psi_k * (pressure_sigma / pressure) * (k_d / (k_d + k_t))
    stiffener_factor_KthG = 1 - (pressure_sigma / pressure) * (k_d / (k_d + k_t))
    return stiffener_factor_KthL, stiffener_factor_KthG


def _compute_elastic_stress_limits(
    R_od, t_wall, h_section, h_web, t_web, w_flange, t_flange, L_stiffener, E, nu, KthG, loading="hydrostatic"
):
    """Compute modifiers to stress due to presence of stiffener rings.

    INPUTS:
    ----------
    params  : dictionary of input parameters
    KthG    : float (scalar/vector),  Stress modifier from stiffeners for general buckling from external pressure
    loading : string (hydrostatic/radial), Parameter that determines a coefficient- is only included for unit testing
              consistency with API 2U Appdx B and should not be used in practice

    OUTPUTS:
    -------
    elastic_axial_local_FxeL    : float (scalar/vector),  Elastic stress limit for local buckling from axial loads
    elastic_extern_local_FreL   : float (scalar/vector),  Elastic stress limit for local buckling from external pressure loads
    elastic_axial_general_FxeG  : float (scalar/vector),  Elastic stress limit for general instability from axial loads
    elastic_extern_general_FreG : float (scalar/vector),  Elastic stress limit for general instability from external pressure loads
    """

    # Geometry computations
    nsections = R_od.size
    area_stiff, y_cg, Ixx, Iyy = _TBeamProperties(h_web, t_web, w_flange, t_flange)
    area_stiff_bar = area_stiff / L_stiffener / t_wall
    R = R_od - 0.5 * t_wall

    # 1. Local shell mode buckling from axial loads
    # Compute a few parameters that define the curvature of the geometry
    m_x = L_stiffener / np.sqrt(R * t_wall)
    z_x = m_x**2 * np.sqrt(1 - nu**2)
    z_m = 12.0 * z_x**2 / np.pi**4
    # Imperfection factor- empirical fit that converts theory to reality
    a_xL = 9.0 * (300.0 + (2 * R / t_wall)) ** (-0.4)
    # Calculate buckling coefficient
    C_xL = np.sqrt(1 + 150.0 * a_xL**2 * m_x**4 / (2 * R / t_wall))
    # Calculate elastic and inelastic final limits
    elastic_axial_local_FxeL = C_xL * np.pi**2 * E * (t_wall / L_stiffener) ** 2 / 12.0 / (1 - nu**2)

    # 2. Local shell mode buckling from external (pressure) loads
    # Imperfection factor- empirical fit that converts theory to reality
    a_thL = np.ones(m_x.shape)
    a_thL[m_x > 5.0] = 0.8
    # Find the buckling mode- closest integer that is root of solved equation
    n = np.zeros((nsections,))
    maxn = 50
    for k in range(nsections):
        c = L_stiffener[k] / np.pi / R[k]
        myfun = lambda x: ((c * x) ** 2 * (1 + (c * x) ** 2) ** 4 / (2 + 3 * (c * x) ** 2) - z_m[k])
        try:
            n[k] = brentq(myfun, 0, maxn, disp=False)
        except:
            n[k] = maxn
    # Calculate beta (local term)
    beta = np.round(n) * L_stiffener / np.pi / R
    # Calculate buckling coefficient
    C_thL = a_thL * (
        (1 + beta**2) ** 2 / (0.5 + beta**2) + 0.112 * m_x**4 / (1 + beta**2) ** 2 / (0.5 + beta**2)
    )
    # Calculate elastic and inelastic final limits
    elastic_extern_local_FreL = C_thL * np.pi**2 * E * (t_wall / L_stiffener) ** 2 / 12.0 / (1 - nu**2)

    # 3. General instability buckling from axial loads
    # Compute imperfection factor
    a_x = 0.85 / (1 + 0.0025 * 2 * R / t_wall)
    a_xG = a_x
    a_xG[area_stiff_bar >= 0.2] = 0.72
    ind = np.logical_and(area_stiff_bar < 0.06, area_stiff_bar < 0.2)
    a_xG[ind] = (3.6 - 5.0 * a_x[ind]) * area_stiff_bar[ind]
    # Calculate elastic and inelastic final limits
    elastic_axial_general_FxeG = 0.605 * a_xG * E * t_wall / R * np.sqrt(1 + area_stiff_bar)

    # 4. General instability buckling from external loads
    # Distance from shell centerline to stiffener cg
    z_r = -(y_cg + 0.5 * t_wall)
    # Imperfection factor
    a_thG = 0.8
    # Effective shell width if the outer shell and the T-ring stiffener were to be combined to make an uneven I-beam
    L_shell_effective = 1.1 * np.sqrt(2.0 * R * t_wall) + t_web
    L_shell_effective[m_x <= 1.56] = L_stiffener[m_x <= 1.56]
    # Get properties of this effective uneven I-beam
    _, yna_eff = _IBeamProperties(h_web, t_web, w_flange, t_flange, L_shell_effective, t_wall)
    Rc = R_od - yna_eff
    # Compute effective shell moment of inertia based on Ir - I of stiffener
    Ier = (
        Ixx
        + area_stiff * z_r**2 * L_shell_effective * t_wall / (area_stiff + L_shell_effective * t_wall)
        + L_shell_effective * t_wall**3 / 12.0
    )
    # Lambda- a local constant
    lambda_G = np.pi * R / h_section
    # Coefficient factor listed as 'k' in peG equation
    coeff = 0.5 if loading in ["hydro", "h", "hydrostatic", "static"] else 0.0
    # Compute pressure leading to elastic failure
    n = np.zeros(R_od.shape)
    pressure_failure_peG = np.zeros(R_od.shape)
    for k in range(nsections):
        peG = lambda x: (
            E[k]
            * lambda_G[k] ** 4
            * t_wall[k]
            / R[k]
            / (x**2 + 0.0 * lambda_G[k] ** 2 - 1)
            / (x**2 + lambda_G[k] ** 2) ** 2
            + E[k] * Ier[k] * (x**2 - 1) / L_stiffener[k] / Rc[k] ** 2 / R_od[k]
        )
        minout = minimize_scalar(peG, bounds=(2.0, 15.0), method="bounded", options={"disp": False})
        n[k] = minout.x
        pressure_failure_peG[k] = peG(n[k])
    # Calculate elastic and inelastic final limits
    elastic_extern_general_FreG = a_thG * pressure_failure_peG * R_od * KthG / t_wall

    return elastic_axial_local_FxeL, elastic_extern_local_FreL, elastic_axial_general_FxeG, elastic_extern_general_FreG


def _plasticityRF(Felastic, yield_stress):
    """Computes plasticity reduction factor for elastic stresses near the yield stress to obtain an inelastic stress
    This is defined in Section 5 of API Bulletin 2U

    INPUTS:
    ----------
    Felastic     : float (scalar/vector),  elastic stress
    yield_stress : float (scalar/vector),  yield stress

    OUTPUTS:
    -------
    Finelastic   : float (scalar/vector),  modified (in)elastic stress
    """
    Fratio = np.array(yield_stress / Felastic)
    eta = Fratio * (1.0 + 3.75 * Fratio**2) ** (-0.25)
    Finelastic = np.array(Felastic)
    Finelastic[Felastic > 0.5 * yield_stress] *= eta[Felastic > 0.5 * yield_stress]
    return Finelastic


def _safety_factor(Ficj, yield_stress):
    """Use the inelastic limits and yield stress to compute required safety factors
    This is defined in Section 9 of API Bulletin 2U

    INPUTS:
    ----------
    Ficj          : float (scalar/vector),  inelastic stress
    yield_stress  : float (scalar/vector),  yield stress

    OUTPUTS:
    -------
    safety_factor : float (scalar/vector),  margin applied to inelastic stress limits
    """
    # Partial safety factor, psi
    psi = np.array(1.4 - 0.4 * Ficj / yield_stress)
    psi[Ficj <= 0.5 * yield_stress] = 1.2
    psi[Ficj >= yield_stress] = 1.0
    # Final safety factor is 25% higher to give a margin
    return 1.25 * psi


def shellBuckling_withStiffeners(
    P, sigma_ax, R_od, t_wall, h_section, h_web, t_web, w_flange, t_flange, L_stiffener, E, nu, sigma_y, loading="hydro"
):

    # APPLIED STRESSES (Section 11 of API Bulletin 2U)
    stiffener_factor_KthL, stiffener_factor_KthG = _compute_stiffener_factors(
        P, sigma_ax, R_od, t_wall, h_web, t_web, w_flange, t_flange, L_stiffener, E, nu
    )
    hoop_stress_nostiff = _compute_applied_hoop(P, R_od, t_wall)
    hoop_stress_between = hoop_stress_nostiff * stiffener_factor_KthL
    hoop_stress_atring = hoop_stress_nostiff * stiffener_factor_KthG

    # BUCKLING FAILURE STRESSES (Section 4 of API Bulletin 2U)
    (
        elastic_axial_local_FxeL,
        elastic_extern_local_FreL,
        elastic_axial_general_FxeG,
        elastic_extern_general_FreG,
    ) = _compute_elastic_stress_limits(
        R_od,
        t_wall,
        h_section,
        h_web,
        t_web,
        w_flange,
        t_flange,
        L_stiffener,
        E,
        nu,
        stiffener_factor_KthG,
        loading=loading,
    )
    inelastic_axial_local_FxcL = _plasticityRF(elastic_axial_local_FxeL, sigma_y)
    inelastic_axial_general_FxcG = _plasticityRF(elastic_axial_general_FxeG, sigma_y)
    inelastic_extern_local_FrcL = _plasticityRF(elastic_extern_local_FreL, sigma_y)
    inelastic_extern_general_FrcG = _plasticityRF(elastic_extern_general_FreG, sigma_y)

    # COMBINE AXIAL AND HOOP (EXTERNAL PRESSURE) LOADS TO FIND DESIGN LIMITS
    # (Section 6 of API Bulletin 2U)
    load_per_length_Nph = sigma_ax * t_wall
    load_per_length_Nth = hoop_stress_nostiff * t_wall
    load_ratio_k = load_per_length_Nph / load_per_length_Nth

    def solveFthFph(Fxci, Frci, Kth):
        Fphci = np.zeros(Fxci.shape)
        Fthci = np.zeros(Fxci.shape)
        Kph = 1.0
        c1 = (Fxci + Frci) / sigma_y - 1.0
        c2 = load_ratio_k * Kph / Kth
        for k in range(Fxci.size):
            try:
                Fthci[k] = brentq(
                    lambda x: (c2[k] * x / Fxci[k]) ** 2
                    - c1[k] * (c2[k] * x / Fxci[k]) * (x / Frci[k])
                    + (x / Frci[k]) ** 2
                    - 1.0,
                    0,
                    Fxci[k] + Frci[k],
                    maxiter=20,
                    disp=False,
                )
            except:
                Fthci[k] = Fxci[k] + Frci[k]
            Fphci[k] = c2[k] * Fthci[k]
        return Fphci, Fthci

    inelastic_local_FphcL, inelastic_local_FthcL = solveFthFph(
        inelastic_axial_local_FxcL, inelastic_extern_local_FrcL, stiffener_factor_KthL
    )
    inelastic_general_FphcG, inelastic_general_FthcG = solveFthFph(
        inelastic_axial_general_FxcG, inelastic_extern_general_FrcG, stiffener_factor_KthG
    )

    # Use the inelastic limits and yield stress to compute required safety factors
    # and adjust the limits accordingly
    axial_limit_local_FaL = inelastic_local_FphcL / _safety_factor(inelastic_local_FphcL, sigma_y)
    extern_limit_local_FthL = inelastic_local_FthcL / _safety_factor(inelastic_local_FthcL, sigma_y)
    axial_limit_general_FaG = inelastic_general_FphcG / _safety_factor(inelastic_general_FphcG, sigma_y)
    extern_limit_general_FthG = inelastic_general_FthcG / _safety_factor(inelastic_general_FthcG, sigma_y)

    # Compare limits to applied stresses and use this ratio as a design constraint
    # (Section 9 "Allowable Stresses" of API Bulletin 2U)
    # These values must be <= 1.0
    axial_local_api = sigma_ax / axial_limit_local_FaL
    axial_general_api = sigma_ax / axial_limit_general_FaG
    external_local_api = hoop_stress_between / extern_limit_local_FthL
    external_general_api = hoop_stress_between / extern_limit_general_FthG

    # Compute unification ratios without safety factors in case we want to apply our own later
    axial_local_raw = sigma_ax / inelastic_local_FphcL
    axial_general_raw = sigma_ax / inelastic_general_FphcG
    external_local_raw = hoop_stress_between / inelastic_local_FthcL
    external_general_raw = hoop_stress_between / inelastic_general_FthcG

    return (
        axial_local_api,
        axial_general_api,
        external_local_api,
        external_general_api,
        axial_local_raw,
        axial_general_raw,
        external_local_raw,
        external_general_raw,
    )
