# -------------------------------------------------------------------------
#
# Author:      RRD
#
# Created:     24/10/2012
# Copyright:   (c) rdamiani 2012
# Licence:     <your licence>
# -------------------------------------------------------------------------


import numpy as np


def frustum(Db, Dt, H):
    """This function returns a frustum's volume and center of mass, CM

    INPUT:
    Parameters
    ----------
    Db : float,        base diameter
    Dt : float,        top diameter
    H : float,         height

    OUTPUTs:
    -------
    vol : float,        volume
    cm : float,        geometric centroid relative to bottom (center of mass if uniform density)

    """
    vol = frustumVol(Db, Dt, H, diamFlag=True)
    cm = frustumCG(Db, Dt, H, diamFlag=True)
    # vol = np.pi/12*H * (Db**2 + Dt**2 + Db * Dt)
    # cm = H/4 * (Db**2 + 3*Dt**2 + 2*Db*Dt) / (Db**2 + Dt**2 + Db*Dt)
    return vol, cm


def frustumVol(rb_0, rt_0, h, diamFlag=False):
    """This function returns a frustum's volume with radii or diameter inputs.

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    vol : float (scalar/vector), volume
    """
    if diamFlag:
        # Convert diameters to radii
        rb, rt = 0.5 * rb_0, 0.5 * rt_0
    else:
        rb, rt = rb_0, rt_0
    return np.pi * (h / 3.0) * (rb * rb + rt * rt + rb * rt)


def frustumCG(rb_0, rt_0, h, diamFlag=False):
    """This function returns a frustum's center of mass/gravity (centroid) with radii or diameter inputs.
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (centroid)
    """
    if diamFlag:
        # Convert diameters to radii
        rb, rt = 0.5 * rb_0, 0.5 * rt_0
    else:
        rb, rt = rb_0, rt_0
    return 0.25 * h * (rb ** 2 + 2.0 * rb * rt + 3.0 * rt ** 2) / (rb ** 2 + rb * rt + rt ** 2)


def frustumIzz(rb_0, rt_0, h, diamFlag=False):
    """This function returns a frustum's mass-moment of inertia (divided by density) about the
    central (axial) z-axis with radii or diameter inputs.
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    Izz : float (scalar/vector),  Moment of inertia about z-axis
    """
    if diamFlag:
        # Convert diameters to radii
        rb, rt = 0.5 * rb_0, 0.5 * rt_0
    else:
        rb, rt = rb_0, rt_0
    # Integrate 2*pi*r*r^2 dr dz from r=0 to r(z), z=0 to h
    # Also equals 0.3*Vol * (rt**5.0 - rb**5.0) / (rt**3.0 - rb**3.0)
    # Also equals (0.1*np.pi*h * (rt**5.0 - rb**5.0) / (rt - rb) )
    return 0.1 * np.pi * h * (rt ** 4.0 + rb * rt ** 3 + rb ** 2 * rt ** 2 + rb ** 3 * rt + rb ** 4.0)


def frustumIxx(rb_0, rt_0, h, diamFlag=False):
    """This function returns a frustum's mass-moment of inertia (divided by density) about the
    transverse x/y-axis passing through the center of mass with radii or diameter inputs.
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    Ixx=Iyy : float (scalar/vector),  Moment of inertia about x/y-axis through center of mass (principle axes)
    """
    if diamFlag:
        # Convert diameters to radii
        rb, rt = 0.5 * rb_0, 0.5 * rt_0
    else:
        rb, rt = rb_0, rt_0
    # Integrate pi*r(z)^4/4 + pi*r(z)^2*(z-z_cg)^2 dz from z=0 to h
    A = 0.5 * frustumIzz(rb_0, rt_0, h)
    B = (
        np.pi
        * h ** 3
        / 80.0
        * (
            (rb ** 4 + 4.0 * rb ** 3 * rt + 10.0 * rb ** 2 * rt ** 2 + 4.0 * rb * rt ** 3 + rt ** 4)
            / (rb ** 2 + rb * rt + rt ** 2)
        )
    )
    return A + B


def frustumShellVol(rb_0, rt_0, t, h, diamFlag=False):
    """This function returns a frustum shell's volume (for computing mass with density) with radii or diameter inputs.
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    t  : float (scalar/vector),  thickness
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    if diamFlag:
        # Convert diameters to radii
        rb, rt = 0.5 * rb_0, 0.5 * rt_0
    else:
        rb, rt = rb_0, rt_0
    # Integrate 2*pi*r*dr*dz from r=ri(z) to ro(z), z=0 to h
    rb_o = rb
    rb_i = rb - t
    rt_o = rt
    rt_i = rt - t
    # ( (np.pi*h/3.0) * ( (rb_o**2 + rb_o*rt_o + rt_o**2) - (rb_i**2 + rb_i*rt_i + rt_i**2) ) )
    return frustumVol(rb_o, rt_o, h) - frustumVol(rb_i, rt_i, h)


def frustumShellCG(rb_0, rt_0, t, h, diamFlag=False):
    """This function returns a frustum's center of mass/gravity (centroid) with radii or diameter inputs.
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    t  : float (scalar/vector),  thickness
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    if diamFlag:
        # Convert diameters to radii
        rb, rt = 0.5 * rb_0, 0.5 * rt_0
    else:
        rb, rt = rb_0, rt_0
    # Integrate 2*pi*r*z*dr*dz/V from r=ri(z) to ro(z), z=0 to h
    rb_o = rb
    rb_i = rb - t
    rt_o = rt
    rt_i = rt - t
    A = (rb_o ** 2 + 2.0 * rb_o * rt_o + 3.0 * rt_o ** 2) - (rb_i ** 2 + 2.0 * rb_i * rt_i + 3.0 * rt_i ** 2)
    B = (rb_o ** 2 + rb_o * rt_o + rt_o ** 2) - (rb_i ** 2 + rb_i * rt_i + rt_i ** 2)
    return h * A / 4.0 / B


def frustumShellIzz(rb_0, rt_0, t, h, diamFlag=False):
    """This function returns a frustum's mass-moment of inertia (divided by density) about the
    central (axial) z-axis with radii or diameter inputs.
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    t  : float (scalar/vector),  thickness
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    Izz : float (scalar/vector),  Moment of inertia about z-axis
    """
    if diamFlag:
        # Convert diameters to radii
        rb, rt = 0.5 * rb_0, 0.5 * rt_0
    else:
        rb, rt = rb_0, rt_0
    # Integrate 2*pi*r*dr*dz from r=ri(z) to ro(z), z=0 to h
    rb_o = rb
    rb_i = rb - t
    rt_o = rt
    rt_i = rt - t
    return frustumIzz(rb_o, rt_o, h) - frustumIzz(rb_i, rt_i, h)


def frustumShellIxx(rb_0, rt_0, t, h, diamFlag=False):
    """This function returns a frustum's mass-moment of inertia (divided by density) about the
    transverse x/y-axis passing through the center of mass with radii or diameter inputs.
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    rb : float (scalar/vector),  base radius
    rt : float (scalar/vector),  top radius
    t  : float (scalar/vector),  thickness
    h  : float (scalar/vector),  height
    diamFlag : boolean, True if rb and rt are entered as diameters

    OUTPUTs:
    -------
    Ixx=Iyy : float (scalar/vector),  Moment of inertia about x/y-axis through center of mass (principle axes)
    """
    if diamFlag:
        # Convert diameters to radii
        rb, rt = 0.5 * rb_0, 0.5 * rt_0
    else:
        rb, rt = rb_0, rt_0
    # Integrate 2*pi*r*dr*dz from r=ri(z) to ro(z), z=0 to h
    rb_o = rb
    rb_i = rb - t
    rt_o = rt
    rt_i = rt - t
    return frustumIxx(rb_o, rt_o, h) - frustumIxx(rb_i, rt_i, h)


if __name__ == "__main__":
    Db = 6.5
    Dt = 4.0
    H = 120.0

    print("From commonse.Frustum: Sample Volume and CM of FRUSTUM=" + 4 * "{:8.4f}, ").format(
        *frustum(Db, Dt, H)[0].flatten()
    )


def main():
    pass
