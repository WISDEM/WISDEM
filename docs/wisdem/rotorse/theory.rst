.. _theory:

Theory
------

Most of the underlying theory is detailed in the stand-alone codes: `AirfoilPreppy <https://github.com/WISDEM/AirfoilPreppy>`_, `CCBlade <https://github.com/WISDEM/CCBlade>`_, `pBEAM <https://github.com/WISDEM/pBEAM>`_, `PreComp <wind.nrel.gov/designcodes/preprocessors/precomp/>`_, and `CurveFEM <http://www1.pacific.edu/~slarwood/dissertation_page.html>`_.  Some additional detail is provided in the following sections.

Aerodynamics
============

.. currentmodule:: rotorse.rotoraero

The :mod:`rotorse.rotoraero` module allows for fixed/variable speed and fixed/variable pitch machines.  Any aerodynamic tool can be used with this module as long as it implements :class:`AeroBase`.  For variable-speed machines the target tip-speed ratio for Region 2 is taken as an input rather than solved internally.  This allows for it to be specified or solved as part of an external optimization process.  There are various potential methods for numerically generating the power curve.  The methodology used here aims for efficiency, with the assumption that the aerodynamic code could potentially be computationally expensive.  For a variable-speed turbine, the aerodynamics code is run at a set number of points from cut-in to cut-out (defaults to 20).  A maximum rotation speed is applied (in a smooth manner so that the result is still continuously differentiable) to the set of run conditions. After running the aerodynamics code, a spline (Akima) is fit between the power to represent the unregulated power curve.  The variation in the unregulated power curve should be smooth and easily represented by a spline, so this approach allows us to specify the unregulated power for any wind speed but only requires running the aerodynamics code at a smaller number of points.  A drivetrain efficiency function is then applied to get the mechanical power, where any function can be applied as long as it is of the form

.. math::
    P = f(P_{aero}, Q_{aero}, T_{aero}, P_{rated})

Next, the rated speed must be determined.  This is done using an internal 1D root-solver (Brent's method).  Because we can evaluate the power from a spline, rather than repeatedly running the aerodynamics code, this process is very efficient.  Once the rated speed is determined, the power curve is truncated at its rated power for higher wind speeds.  (Physically this is accomplished through pitch control, generally pitch toward feather, but the actual mechanism is irrelevant for the purposes of this process).  This process also automatically allows for a Region 2.5 through the application of a maximum rotation speed.

A typical power curve for a variable-speed, variable-pitch turbine is shown below. Region 1 has no power generation as it occurs below the cut-in speed. In Region 2, variable-speed turbines operate at the specified tip-speed ratio until either rated power or the maximum rotation speed is reached. If the maximum rotor speed is reached, a Region 2.5 handles the transition intro Region 3.  Blade pitch is varied in Region 3 so that rated power is maintained in Region 3.

.. figure:: /images/pc.*
    :width: 3.5in
    :align: center




.. The powercurve() method does not actually need to compute the required rotation speed / pitch setting for operation in Region 3.  This is beneficial because the computation of the control-settings is non-negligible for variable-pitch machines.  For fixed-pitch machines the required control setting can be determined at any location of the power curve without additional calls to the :class:`BladeAero <twister.turbine.bladeaero.BladeAero>` object.  However, for variable-pitch machines, in Region 3 additional calls are necessary in order to determine the appropriate pitch setting to maintain rated power.  It is assumed that pitch is toward feather. Although they are not needed for the power curve, the control settings are needed for many of the structural analyses (e.g.  distributedAeroLoads(Uinf, pitch) for use in estimating deflection, etc.).  Fortunately, the structural analysis is generally evaluated at far fewer wind speeds than the aerodynamic analysis is.




Annual energy production (AEP) can be computed with any arbitrary wind speed distribution, though convenience methods are provided for Rayleigh and Weibull distribution.  Losses caused by wake interference from other turbines in the wind farm and losses caused by electrical grid unavailability are estimated simply through a total loss factor.  The annual energy production (in kWh) is calculated as

.. math::
    AEP = 8.76\ loss \int_{V_{in}}^{V_{out}} P(V) f(V) dV = 8.76\ loss \int_{V_{in}}^{V_{out}} P(V) dF(V)

where P is in Watts, f(V) is a probability density function for the site, and F(V) is the corresponding cumulative distribution function.

Structures
==========

.. currentmodule:: rotorse.rotor

The :mod:`rotorse.rotor` module uses cross-sectional composite analysis codes that implement :class:`BeamPropertiesBase` (optionally), and structural analysis codes that implement :class:`StrucBase`.  The :class:`PreCompSections` class provides a concrete implementation of :class:`BeamPropertiesBase`.  It links a description of the blade geometry and section composite layup with an existing NWTC code PreComp :cite:`Bir2005`. PreComp uses modified classic laminate theory combined with a shear-flow approach, to estimate equivalent sectional inertial and stiffness properties of composite blades. PreComp requires the geometric description of the blade (chord, twist, section profile shapes, web locations), along with the internal structural layup (laminate schedule, orientation of fibers, laminate material properties). It allows for high-flexibility in the specification of the composite layup both spanwise and chordwise. The underlying code PreComp is written in Fortran and is linked to this class with f2py.


A panel buckling calculation is added to augment the sectional analysis. The constitutive equations for a laminate sequence can be expressed as



.. math::
    :label: constitutive

    \left[\begin{matrix} N \\
    M \end{matrix}\right] = \left[\begin{matrix} A & B \\
    B & D \end{matrix}\right] \left[\begin{matrix} \epsilon^0 \\
    k \end{matrix}\right]

where N and M are the average forces and moments of the laminate per unit length, and :math:`\epsilon^0` and :math:`k` are the mid-plane strains and curvature (see :cite:`Halpin1992`).  The D matrix is a :math:`3 \times 3` matrix of the form (while wind turbine blade cross-sections are not always precisely specially orthotropic they are well approximate as such).

.. math::

    \left[
    \begin{array}{ccc}
        D_{11} & D_{12} & 0 \\
        D_{12} & D_{22} & 0 \\
        0 & 0 & D_{66}
    \end{array}
    \right]


The critical buckling load for long (length greater than twice the width) simply supported panels at a given section is estimated as :cite:`Johnson1994`

.. math::
    N_{cr} = 2 \left(\frac{\pi}{w}\right)^2 \left[  \sqrt{D_{11} D_{22}} + D_{12} + 2 D_{66}\right]

where :math:`w` is the panel width.  If we denote the matrix in the constitutive equation (Equation :eq:`constitutive`) as :math:`S` and its inverse as :math:`S^*`, then :math:`\epsilon_{zz} \approx S^*_{11}N_z`.  This expression ignores laminate shear and bending moment effects (the latter would be zero for a symmetric laminate), a good approximation for slender turbine blades.  At the same time, an effective smeared modulus of elasticity can be computed by integrating across the laminate stack

.. math:: E_{zz} = \frac{1}{\epsilon_{zz} h} \int_{-h/2}^{h/2} \sigma_{zz} dh = \frac{N_z}{ \epsilon_{zz} h}

where :math:`N_z` in this equation is the average force per unit length of the laminate.  Combining these equations yields an estimate for the effective axial modulus of elasticity

.. math:: E_{zz} = \frac{1}{S^*_{11} h}

The critical strain can then be computed as

.. math::
    \epsilon_b = - \frac{N_{cr}}{h \ E_{zz}}

where the negative sign accounts for the fact that the strain is compressive in buckling.



The :class:`RotorWithpBEAM` class provides a concrete implementation of :class:`StrucBase`.  Most of the methodology is implemented using the beam finite element code, called `pBEAM <https://github.com/WISDEM/pBEAM>`_ (polynomial beam element analysis module), which was developed specifically for this application.  The finite element code pBEAM operates about the elastic center of a structure and in principal axes in order to remove cross-coupling terms. Thus, the computed flapwise, edgewise, and coupled stiffness properties from the :class:`BeamPropertiesBase` objects are translated to the elastic center and rotated to principal axes as described by :cite:`Hansen2008`. Similarly, input flapwise and edgewise loads are rotated to the principal axes, and output deflections are rotated back to the flapwise and edgewise axes.

.. math::
    :label: strain

    \epsilon(x,y) = \frac{M_1}{[EI]_1} y - \frac{M_2}{[EI]_2} x + \frac{N}{[EA]}

A simple fatigue calculation is also included in this component.  Damage equivalent moments are supplied by a user.  These should be computed using a full lifecycle analysis using an aeroelastic tool like FAST.  From Equation :eq:`strain` A corresponding strain can be computed.  An S-N fatigue life curve is of the form (here written in terms of strain)

.. math::
    \epsilon = \epsilon_{max} N_f^\frac{-1}{m}

If we arrange to solve in terms of the number of cycles for a given level of strain, and adding in a safety factor, we have

.. math::
    N_f = \left(\frac{\epsilon_{max}}{\eta \epsilon}\right)^m

where :math:`\eta` is a safety factor, and m is the slope parameter found from viewing the S-N data on a log-log plot (generally around 10 for glass-reinforced composites :cite:`Mandell1997`).  Then the damage is the number of cycles corresponding to the damage equivalent moments divided by the number of cycles to failure

.. math::
    damage = \frac{N}{N_f}



.. only:: html

    :bib:`Bibliography`

.. bibliography:: references.bib
    :style: unsrt