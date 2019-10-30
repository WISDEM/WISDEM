.. module:: wisdem.pBeam._pBEAM

.. _documentation-label:

Module Documentation
--------------------

This documentation only details the outward facing classes that are available through the Python module.  Other classes and methods are available through the C++ implementation; however, they primarily encapsulate implementation details not necessary for external use.  For details on use through C++, refer to the source code and examples in the test suite.  The main class in the pBEAM module is :ref:`beam-label`.  All other classes are only helper objects used as inputs to the Beam object.

.. only:: latex

    The HTML version of this documentation is better suited to view the code documentation and contains details on the :ref:`methods <beam-methods-label>` contained in the Beam class as well as hyperlinks to the source code.


.. _beam-label:

Beam
^^^^
Beam is the main class for pBEAM and defines the complete beam object.  Three constructors are provided: a convenience constructor for a beam with cylindrical shell sections and isotropic material (e.g., a wind turbine tower), a constructor for general sections that very linearly between sections, and a constructor where properties vary as polynomials across elements.

.. rubric:: Class Summary:

.. _beam-tower-label:

.. class:: Beam(nodes, z, d, t, loads, mat, tip, base)

    A convenience constructor for a beam with cylindrical shell sections and isotropic material (e.g., a wind turbine tower)

    Parameters
    ----------
    nodes : int
        number of nodes
    z : ndarray [nodes] (m)
        location of beam sections starting from base and ending at tip
    d : ndarray [nodes] (m)
        diameter of beam at each z-location
    t : ndarray [nodes] (m)
        shell thickness of beam at each z-location
    loads : :ref:`loads-label`
        loads along beam
    material : :ref:`material-label`
        isotropic material properties
    tip : :ref:`tip-label`
        properties of offset tip mass
    base : :ref:`base-label`
        properties of base stiffness boundary condition

.. _beam-linear-label:

.. class:: Beam(section, loads, tip, base)

    A beam with general section properties with linear variation between nodes.

    Parameters
    ----------
    section : :ref:`section-data-label`
        section data (inertial and stiffness properties) along beam
    loads : :ref:`loads-label`
        loads along beam
    tip : :ref:`tip-label`
        properties of offset tip mass
    base : :ref:`base-label`
        properties of base stiffness boundary condition


.. class:: Beam(section, loads, tip, base, flag)

    A beam with general section properties with polynomial variation between nodes.

    Parameters
    ----------
    section : :ref:`polysection-data-label`
        polynomial section data (inertial and stiffness properties) along beam
    loads : :ref:`polyloads-label`
        polynomial loads along beam
    tip : :ref:`tip-label`
        properties of offset tip mass
    base : :ref:`base-label`
        properties of base stiffness boundary condition
    flag : int
        can be set to any value, just a flag to allow use of polynomial variation rather
        than the overloaded constructor above that uses linear variation


.. _beam-methods-label:

.. only:: latex

    TABLE CAPTION:: Methods available for Beam objects.

.. only:: html

    .. rubric:: Methods

=============================================================== =======================================================
:doc:`mass() <methods/mass>`                                    mass of beam
:doc:`naturalFrequencies(n) <methods/freq>`                     first n natural frequencies
:doc:`naturalFrequenciesAndEigenvectors(n) <methods/freqvec>`   first n natural frequencies and eigenvectors
:doc:`displacement() <methods/disp>`                            6 DOF displacement at each node
:doc:`criticalBucklingLoads() <methods/buckling>`               global minimum critical axial buckling loads
:doc:`axialStrain(n, x, y, z) <methods/strain>`                 axial strain along beam
:doc:`outOfPlaneMomentOfInertia() <methods/inertia>`            out of plane moment of inertia
=============================================================== =======================================================


.. _section-data-label:

SectionData
^^^^^^^^^^^
SectionData is a C++ struct that defines the section properties along the beam, assuming a linear variation in properties between the defined sections.  For polynomial variation, see :ref:`polysection-data-label`.

.. rubric:: Class Summary:

.. _section-data-linear-label:

.. class:: SectionData(n, z, EA, EIxx, EIyy, GJ, rhoA, rhoJ)

    Section data defined along the structure from base to tip.

    Parameters
    ----------
    n : int
        number of sections where data is defined (nodes)
    z : ndarray [n] (m)
        location of beam sections starting from base and ending at tip
    EA : ndarray [n] (N)
        axial stiffness at each section
    EIxx : ndarray [n] (N*m**2)
        bending stiffness about +x-axis
    EIyy : ndarray [n] (N*m**2)
        bending stiffness about +y-axis
    GJ : ndarray [n] (N*m**2)
        torsional stiffness about +z-axis
    rhoA : ndarray [n] (kg/m)
        mass per unit length
    rhoJ : ndarray [n] (kg*m)
        polar mass moment of inertia per unit length

    Notes
    -----

    All parameters must be specified about the elastic center and in principal axis
    (i.e., EIxy, Sx, and Sy are all zero). Linear variation in properties between sections is assumed.

.. _polysection-data-label:

PolySectionData
^^^^^^^^^^^^^^^
PolySectionData is a C++ struct that defines the section properties along the beam, and allows for polynomial variation of properties between the defined sections.  For linear variation, :ref:`section-data-label` is simpler to work with.

.. rubric:: Class Summary:

.. class:: PolySectionData(nodes, z, nA, nI, EA, EIxx, EIyy, GJ, rhoA, rhoJ)

    Polynomial section data defined along the structure from base to tip.

    Parameters
    ----------
    n : int
        number of sections where data is defined (nodes)
    z : ndarray [n] (m)
        location of beam sections starting from base and ending at tip
    nA : ndarray(int) [n - 1]
        nA[i] is the order of the polynomial describing structural properties between nodes i and i + 1 for
        properties that are area dependent (EA, rhoA)
    nI : ndarray(int) [n - 1]
        nI[i] is the order of the polynomial describing structural properties between nodes i and i + 1 for
        properties that are moment of inertia dependent (EIxx, EIyy, GJ, rhoJ)
    EA : list(ndarray) [n - 1] (N)
        EA[i] is a polynomial of length nA[i] that describes the axial stiffness between nodes i and i + 1
    EIxx : list(ndarray) [n - 1] (N*m**2)
        EIxx[i] is a polynomial of length nI[i] that describes the bending stiffness about +x-axis between nodes i and i + 1
    EIyy : list(ndarray) [n - 1] (N*m**2)
        EIyy[i] is a polynomial of length nI[i] that describes the bending stiffness about +y-axis between nodes i and i + 1
    GJ : list(ndarray) [n - 1] (N*m**2)
        GJ[i] is a polynomial of length nI[i] that describes the torsional stiffness about +z-axis between nodes i and i + 1
    rhoA : list(ndarray) [n - 1] (kg/m)
        rhoA[i] is a polynomial of length nA[i] that describes the mass per unit length between nodes i and i + 1
    rhoJ : list(ndarray) [n - 1] (kg*m)
        rhoJ[i] is a polynomial of length nI[i] that describes the polar mass moment of inertia per unit length between nodes i and i + 1

    Notes
    -----

    All parameters must be specified about the elastic center and in principal axis
    (i.e., EIxy, Sx, and Sy are all zero).  Polynomials are expressed as:

    EA[i] = [5.0, 3.0, 2.0] which means :math:`EA[i] = 5.0\eta^2 + 3.0\eta + 2.0`

    where :math:`\eta` is a normalized coordinate s.t. it equals 0 at the base
    of the given element and 1 at the top of the element.
    The numpy polynomial class (numpy.poly1d) is useful for multiplying polynomials, etc.

.. _loads-label:

Loads
^^^^^
Loads is a C++ struct that defines the applied loads along the beam.  Three constructors are provided: beams with no external loads, beams with only distributed loads, and beams with distributed loads and point forces/moments.  Distributed loads are assumed to vary lineary between sections.  For polynomial variation in distributed loads, see :ref:`polyloads-label`.

.. rubric:: Class Summary:

.. class:: Loads()

    No applied external loads.

.. _loads-distributed-label:

.. class:: Loads(n, Px, Py, Pz)

    Distributed loads along beam from base to tip.

    Parameters
    ----------
    n : int
        number of sections where forces are defined (nodes)
    Px : ndarray [n] (N/m)
        force per unit length along beam in the x-direction
    Py : ndarray [n] (N/m)
        force per unit length along beam in the y-direction
    Pz : ndarray [n] (N/m)
        force per unit length along beam in the z-direction

    Notes
    -----
    Loads must be given at the corresponding z locations defined
    in :ref:`section-data-label` or :ref:`polysection-data-label`.


.. class:: Loads(n, Px, Py, Pz, Fx, Fy, Fz, Mx, My, Mz)

    Distributed loads and applied point forces/moments along beam
    from base to tip.

    Parameters
    ----------
    n : int
        number of sections where forces are defined (nodes)
    Px : ndarray [n] (N/m)
        force per unit length along beam in the x-direction
    Py : ndarray [n] (N/m)
        force per unit length along beam in the y-direction
    Pz : ndarray [n] (N/m)
        force per unit length along beam in the z-direction
    Fx : ndarray [n] (N)
        point forces in the x-direction
    Fy : ndarray [n] (N)
        point forces in the y-direction
    Fz : ndarray [n] (N)
        point forces in the z-direction
    Mx : ndarray [n] (N*m)
        point moments in the x-direction
    My : ndarray [n] (N*m)
        point moments in the y-direction
    Mz : ndarray [n] (N*m)
        point moments in the z-direction

    Notes
    -----
    Loads must be given at the corresponding z locations defined
    in :ref:`section-data-label` or :ref:`polysection-data-label`.

.. _polyloads-label:

PolyLoads
^^^^^^^^^
PolyLoads is a C++ struct that defines the applied loads along the beam and allows for polynomial variation of loads between the defined sections.  For linear variation in distributed loads, :ref:`loads-label` is simpler to work with.

.. rubric:: Class Summary:

.. class:: PolyLoads(n, nP, Px, Py, Pz, Fx, Fy, Fz, Mx, My, Mz)

    Polynomial variation in distributed loads, and point forces/moments defined along the structure from base to tip.

    Parameters
    ----------
    n : int
        number of sections where loads are defined (nodes)
    nP : ndarray(int) [n - 1]
        nP[i] is the order of the polynomial describing the distributed load between nodes i and i + 1
    Px : list(ndarray) [n - 1] (N)
        Px[i] is a polynomial of length nP[i] that describes the force per unit length
        in the +x-direction between nodes i and i + 1
    Py : list(ndarray) [n - 1] (N)
        Py[i] is a polynomial of length nP[i] that describes the force per unit length
        in the +y-direction between nodes i and i + 1
    Pz : list(ndarray) [n - 1] (N)
        Pz[i] is a polynomial of length nP[i] that describes the force per unit length
        in the +z-direction between nodes i and i + 1
    Fx : ndarray [n] (N)
        point forces in the x-direction
    Fy : ndarray [n] (N)
        point forces in the y-direction
    Fz : ndarray [n] (N)
        point forces in the z-direction
    Mx : ndarray [n] (N*m)
        point moments in the x-direction
    My : ndarray [n] (N*m)
        point moments in the y-direction
    Mz : ndarray [n] (N*m)
        point moments in the z-direction

    Notes
    -----
    Polynomials are expressed as:

    Px[i] = [5.0, 3.0, 2.0], which means :math:`Px[i] = 5.0\eta^2 + 3.0\eta + 2.0`

    where :math:`\eta` is a normalized coordinate s.t. it equals 0 at the base
    of the given element and 1 at the top of the element.
    The numpy polynomial class (numpy.poly1d) is useful for multiplying polynomials, etc.

    Loads must be given at the corresponding z locations defined in
    :ref:`section-data-label` or :ref:`polysection-data-label`

.. _base-label:

BaseData
^^^^^^^^
BaseData is a C++ struct that defines the stiffness properties of the base of the beam.  Two constructors are available: a convenience constructor for a free-end, and a general constructor for specifying the equivalent spring stiffness in all 6 DOF.

.. rubric:: Class Summary:

.. class:: BaseData()

    A free-end.  External spring stiffness is zero in all directions.

.. class:: BaseData(k, infinity)

    A base with equivalent external spring stiffness applied.

    Parameters
    ----------
    k : ndarray [6] (N/m)
        stifness at base [k_xx, k_txtx, k_yy, k_tyty, k_zz, k_tztz]
        where tx is the rotational direction theta_x and so forth
    infinity : float (N/m)
        a value that represents infinity (can be any arbitrary float but it is convenient to use
        Python's ``float('inf')``).  used to denote infinitely rigid directions.

.. _tip-label:

TipData
^^^^^^^
TipData is a C++ struct that defines the properties of the offset tip mass.  Two constructors are available: a convenience constructor for a beam with no offset tip mass, and a general constructor for specifying the properties of the offset tip mass.

.. rubric:: Class Summary:

.. class:: TipData()

    No offset tip mass.

.. class:: TipData(m, cm, I, F, M)

    Used to model an offset tip mass (e.g., rotor/nacelle/assembly on top of a wind turbine tower)

    Parameters
    ----------
    m : float (kg)
        mass of object
    cm : ndarray [3] (m)
        location of object's center of mass relative to beam tip [x, y, z]
    I : ndarray [6] (m^4)
        area moment of inertia of object about beam tip [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    F : ndarray [3] (N)
        applied force from the object onto the beam tip [Fx, Fy, Fz]
    M : ndarray [3] (N*m)
        applied moment from the object onto the beam tip [Mx, My, Mz]

.. _material-label:

Material
^^^^^^^^
Material is a C++ struct that defines the material properties for an isotropic material.

.. rubric:: Class Summary:

.. class:: Material(E, G, rho)

    Material properties for an isotropic material.

    Parameters
    ----------
    E : float (N/m**2)
        modulus of elasticity
    G : float (N/m**2)
        shear modulus
    rho : float
        mass density (kg/m**3)

