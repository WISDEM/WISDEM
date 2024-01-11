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

def RectangularFrustumVol(ab, bb, at, bt, h):
    """This function returns a rectangular frustum's volume with side length inputs.

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    vol : float (scalar/vector), volume
    """

    Abottom = ab * bb
    Atop = at * bt

    V = GeneralFrustumVol(Ab=Abottom, At=Atop, h=h)
    
    return V

def GeneralFrustumVol(Ab, At, h):
    """This function returns a frustum's volume with top and base areas, and height

    INPUTS:
    Parameters
    ----------
    Ab : float (scalar/vector),  base area
    At : float (scalar/vector),  top area
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    vol : float (scalar/vector), volume
    """

    Atop = At
    Abottom = Ab

    V = h / 3.0 * (Atop + Abottom + np.sqrt(Atop * Abottom))
    
    return V

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
    return 0.25 * h * (rb**2 + 2.0 * rb * rt + 3.0 * rt**2) / (rb**2 + rb * rt + rt**2)

def RectangularFrustumCG(ab, bb, at, bt, h):
    """This function returns a rectangular frustum's center of mass/gravity (centroid) with side lengths and height
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (centroid)
    """
    cg = GeneralFrustumCG(Ab = ab*bb, At = at*bt, h = h)

    return cg


def GeneralFrustumCG(Ab, At, h):
    """This function returns a frustum's center of mass/gravity (centroid) with base and top area inputs.
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    Ab : float (scalar/vector),  base area
    At : float (scalar/vector),  top area
    h  : float (scalar/vector),  height
    V  : float (scalar/vector),  Volume

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (centroid)
    """
    cg = h * (Ab + 2*np.sqrt(Ab*At) + 3*At) / (4 * (Ab + np.sqrt(Ab * At) + At))

    return cg

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
    return 0.1 * np.pi * h * (rt**4.0 + rb * rt**3 + rb**2 * rt**2 + rb**3 * rt + rb**4.0)

def RectangularFrustumIzz(ab, bb, at, bt, h):
    """This function returns a rectangular frustum's mass-moment of inertia (divided by density) about the
    central (axial) z-axis with side lengths and height
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    Izz : float (scalar/vector),  Moment of inertia about z-axis
    """

    # Use calculations from RAFT

    x2 = (1/12) * ( (at-ab)**3*h*(bt/5 + bb/20) + (at-ab)**2*ab*h*(3*bt/4 + bb/4) + \
                    (at-ab)*ab**2*h*(bt + bb/2) + ab**3*h*(bt/2 + bb/2) )

    y2 = (1/12) * ( (bt-bb)**3*h*(at/5 + ab/20) + (bt-bb)**2*bb*h*(3*at/4 + ab/4) + \
                        (bt-bb)*bb**2*h*(at + ab/2) + bb**3*h*(at/2 + ab/2) )
    return x2+y2


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
        * h**3
        / 80.0
        * (
            (rb**4 + 4.0 * rb**3 * rt + 10.0 * rb**2 * rt**2 + 4.0 * rb * rt**3 + rt**4)
            / (rb**2 + rb * rt + rt**2)
        )
    )
    return A + B


def RectangularFrustumIxx(ab, bb, at, bt, h):
    """This function returns a rectangular frustum's mass-moment of inertia (divided by density) about the
    central (axial) z-axis with side lengths and height
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    Izz : float (scalar/vector),  Moment of inertia about z-axis
    """

    # Use calculations from RAFT

    y2 = (1/12) * ( (bt-bb)**3*h*(at/5 + ab/20) + (bt-bb)**2*bb*h*(3*at/4 + ab/4) + \
                        (bt-bb)*bb**2*h*(at + ab/2) + bb**3*h*(at/2 + ab/2) )
    
    z2 = ( bt*at/5 + bb*at/20 + ab*bt/20 + bb*ab*(1/30) ) * h**3

    return y2 + z2

def RectangularFrustumIyy(ab, bb, at, bt, h):
    """This function returns a rectangular frustum's mass-moment of inertia (divided by density) about the
    central (axial) z-axis with side lengths and height
    NOTE: This is for a SOLID frustum, not a shell

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height

    OUTPUTs:
    -------
    Izz : float (scalar/vector),  Moment of inertia about z-axis
    """

    # Use calculations from RAFT

    x2 = (1/12) * ( (at-ab)**3*h*(bt/5 + bb/20) + (at-ab)**2*ab*h*(3*bt/4 + bb/4) + \
                    (at-ab)*ab**2*h*(bt + bb/2) + ab**3*h*(bt/2 + bb/2) )
   
    z2 = ( bt*at/5 + bb*at/20 + ab*bt/20 + bb*ab*(1/30) ) * h**3

    return x2 + z2


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

def RectangularFrustumShellVol(ab, bb, at, bt, t, h):
    """This function returns a rectangular frustum shell's volume (for computing mass with density) with side lengths.
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height
    t  : float (scalar/vector),  thickness

    OUTPUTs:
    -------
    vol : float (scalar/vector),  shell volume
    """

    ab_o = ab
    bb_o = bb
    at_o = at
    bt_o = bt
    ab_i = ab_o - 2*t
    bb_i = bb_o - 2*t
    at_i = at_o - 2*t
    bt_i = bt_o - 2*t

    if isinstance(ab_i, float):
        assert ab_i > 0.0, "Thickness is larger than inner side length ab"
        assert bb_i > 0.0, "Thickness is larger than inner side length bb"
        assert at_i > 0.0, "Thickness is larger than inner side length at"
        assert bt_i > 0.0, "Thickness is larger than inner side length bt"
    else:
        assert (ab_i > 0.0).all(), "Thickness is larger than inner side length ab"
        assert (bb_i > 0.0).all(), "Thickness is larger than inner side length bb"
        assert (at_i > 0.0).all(), "Thickness is larger than inner side length at"
        assert (bt_i > 0.0).all(), "Thickness is larger than inner side length bt"

    return RectangularFrustumVol(ab_o, bb_o, at_o, bt_o, h) - RectangularFrustumVol(ab_i, bb_i, at_i, bt_i, h)


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
    A = (rb_o**2 + 2.0 * rb_o * rt_o + 3.0 * rt_o**2) - (rb_i**2 + 2.0 * rb_i * rt_i + 3.0 * rt_i**2)
    B = (rb_o**2 + rb_o * rt_o + rt_o**2) - (rb_i**2 + rb_i * rt_i + rt_i**2)
    return h * A / 4.0 / B

def RectangularFrustumShellCG(ab, bb, at, bt, t, h):
    """This function returns a rectangular frustum's center of mass/gravity (centroid) with side lengths and height
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height
    t  : float (scalar/vector),  thickness

    OUTPUTs:
    -------
    cg : float (scalar/vector),  center of mass/gravity (ventroid)
    """
    z_integral = 2.0/3.0*(at+bt+0.5*ab+0.5*bb)*h**2*t - 2.0*t**2*h**2
    return z_integral/RectangularFrustumShellVol(ab, bb, at, bt, t, h)


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

def RectangularFrustumShellIzz(ab, bb, at, bt, t, h):
    """This function returns a rectangular frustum's mass-moment of inertia (divided by density) about the
    central (axial) z-axis with side lengths
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height
    t  : float (scalar/vector),  thickness

    OUTPUTs:
    -------
    Izz : float (scalar/vector),  Moment of inertia about z-axis
    """
    ab_o = ab
    bb_o = bb
    at_o = at
    bt_o = bt
    ab_i = ab_o - 2*t
    bb_i = bb_o - 2*t
    at_i = at_o - 2*t
    bt_i = bt_o - 2*t

    return RectangularFrustumIzz(ab_o, bb_o, at_o, bt_o, h) - RectangularFrustumIzz(ab_i, bb_i, at_i, bt_i, h)


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

def RectangularFrustumShellIxx(ab, bb, at, bt, t, h):
    """This function returns a rectangular frustum's mass-moment of inertia (divided by density) about the
    central (axial) z-axis with side lengths and height
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height
    t  : float (scalar/vector),  thickness

    OUTPUTs:
    -------
    Ixx : float (scalar/vector),  Moment of inertia about z-axis
    """
    ab_o = ab
    bb_o = bb
    at_o = at
    bt_o = bt
    ab_i = ab_o - 2*t
    bb_i = bb_o - 2*t
    at_i = at_o - 2*t
    bt_i = bt_o - 2*t

    return RectangularFrustumIxx(ab_o, bb_o, at_o, bt_o, h) - RectangularFrustumIxx(ab_i, bb_i, at_i, bt_i, h)

def RectangularFrustumShellIyy(ab, bb, at, bt, t, h):
    """This function returns a rectangular frustum's mass-moment of inertia (divided by density) about the
    central (axial) z-axis with side lengths and height
    NOTE: This is for a frustum SHELL, not a solid

    INPUTS:
    Parameters
    ----------
    ab : float (scalar/vector),  base side length a
    bb : float (scalar/vector),  base side length b
    at : float (scalar/vector),  top side length a
    bt : float (scalar/vector),  top side length b
    h  : float (scalar/vector),  height
    t  : float (scalar/vector),  thickness

    OUTPUTs:
    -------
    Izz : float (scalar/vector),  Moment of inertia about z-axis
    """
    ab_o = ab
    bb_o = bb
    at_o = at
    bt_o = bt
    ab_i = ab_o - 2*t
    bb_i = bb_o - 2*t
    at_i = at_o - 2*t
    bt_i = bt_o - 2*t

    return RectangularFrustumIyy(ab_o, bb_o, at_o, bt_o, h) - RectangularFrustumIyy(ab_i, bb_i, at_i, bt_i, h)


if __name__ == "__main__":
    Db = 6.5
    Dt = 4.0
    H = 120.0

    print("From commonse.Frustum: Sample Volume and CM of FRUSTUM=" + 4 * "{:8.4f}, ").format(
        *frustum(Db, Dt, H)[0].flatten()
    )


def main():
    pass
