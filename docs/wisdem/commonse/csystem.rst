.. _csystem-label:

Coordinate System
=================

.. currentmodule:: wisdem.commonse.csystem

This module defines coordinate systems for horizontal axis wind turbines and provides convenience methods for transforming vectors between the various coordinate systems.
The supplied transformation methods are for *rotation only* and do not account for any offsets that may be necessary depending on the vector quantity (e.g., transfer of forces between coordinate system does not depend on the location where the force is defined, but position, velocity, moments, etc. do).
In other words the vectors are treated as directions only and are independent of the defined position.
How the vector should transform based on position is not generalizable and depends on the quantity of interest.

All coordinate systems obey the right-hand rule, :math:`x \times y = z`, and all angles must be input in **degrees**.
The turbine can be either an upwind or downwind configuration, but in either case it is assumed that that the blades rotate in the **clockwise** direction when looking downwind (more specifically the rotor is assumed to rotate about the :math:`+x_h` axis in :numref:`Figure %s <yaw-hub-fig>`.
The vectors allow for elementary operations (+, -, \*, /, +=, -=, \*=, /=) between other vectors of the same type, or with scalars (e.g., force_total = force1 + force2).


.. autoclass:: DirectionVector

.. _inertial_wind_coord:

Inertial and Wind-aligned
-------------------------

.. _inertial-wind-fig:

.. figure:: /images/commonse/inertial_wind.*
    :width: 5.5in
    :align: center

    Inertial and Wind-aligned axes.

:numref:`Figure %s <inertial-wind-fig>` defines the transformation between the inertial and wind-aligned coordinate systems.
The two coordinate systems share a common origin, and a common z-direction.
The wind angle :math:`\beta` is positive for rotation about the +z axis.
The direction of wave loads are defined similarly to the wind loads, but there is no wave-aligned coordinate system.


*Inertial coordinate system*

    **origin**: center of the tower base (ground-level or sea-bed level)

    **x-axis**: any direction as long as used consistently, but convenient to be in primary wind direction

    **y-axis**: follows from the right-hand rule

    **z-axis**: up the tower (opposite to gravity vector)



*Wind-aligned coordinate system*

    **origin**: center of the tower base (ground-level or sea-bed level)

    **x-axis**: in direction of the wind

    **y-axis**: follows from the right-hand rule

    **z-axis**: up the tower (opposite to gravity vector), coincident with inertial z-axis


.. only:: latex

    TABLE CAPTION:: Inertial-Wind conversion methods

    .. autosummary::

        ~DirectionVector.inertialToWind
        ~DirectionVector.windToInertial


.. only:: html

    .. rubric:: Associated Methods

    .. autosummary::

        ~DirectionVector.inertialToWind
        ~DirectionVector.windToInertial


.. autogenerate
    .. autosummary::
        :toctree: generated

        ~DirectionVector.inertialToWind
        ~DirectionVector.windToInertial




.. _wind_yaw_coord:

Wind-aligned and Yaw-aligned
----------------------------

.. _wind-yaw-fig:

.. figure:: /images/commonse/wind_yaw.*
    :width: 6.5in
    :align: center

    Wind-aligned and yaw-aligned axes.
    :math:`\Psi` is the rotor yaw angle.

:numref:`Figure %s <wind-yaw-fig>` defines the transformation between the wind-aligned and yaw-aligned coordinate systems.
The two coordinate systems are offset by the height :math:`h_t` along the common z-axis.
The yaw angle :math:`\Psi` is positive when rotating about the +z axis, and should be between -180 and +180 degrees.

.. For a downwind machine the yaw angle will be near +/-180 degrees, and the :math:`\hat{x}_y` and :math:`\hat{x}_w` will point in nominally opposite directions.

*Yaw-aligned coordinate system*

    **origin**: Tower top (center of the yaw bearing system)

    **x-axis**: along projection of rotor shaft in horizontal plane (aligned with rotor shaft for zero tilt angle).
    The positive direction is defined such that the x-axis points downwind at its design operating orientation (i.e., at zero yaw :math:`x_y` is the same direction as :math:`x_w`).
    Thus, for a downwind machine the :math:`x_y` axis would still be downwind at zero yaw, but in terms of nacelle orientation it would point from the back of the nacelle toward the hub.

    **y-axis**: follows from the right-hand rule

    **z-axis**: points up the tower (opposite to gravity vector), coincident with wind-aligned z-axis



.. only:: latex

    TABLE CAPTION:: Wind-Yaw conversion methods

    .. autosummary::

        ~DirectionVector.windToYaw
        ~DirectionVector.yawToWind


.. only:: html

    .. rubric:: Associated Methods

    .. autosummary::

        ~DirectionVector.windToYaw
        ~DirectionVector.yawToWind

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~DirectionVector.windToYaw
        ~DirectionVector.yawToWind



.. _yaw_hub_coord:

Yaw-aligned and Hub-aligned
-----------------------------

.. _yaw-hub-fig:

.. figure:: /images/commonse/yaw_hub.*
    :width: 3.5in
    :align: center

    Yaw-aligned and hub-aligned axes.
    :math:`\Theta` is the rotor tilt angle.

:numref:`Figure %s <yaw-hub-fig>` defines the transformation between the yaw-aligned and hub-aligned coordinate systems.
The two coordinate systems share a common y axis.
The tilt angle :math:`\Theta` is positive when rotating about the +y axis, which tilts the rotor up for an upwind machine (tilts the rotor down for a downwind machine).

*Hub-aligned coordinate system*

    **origin**: center of the rotor.

    **x-axis**: along the rotor shaft toward the nominal downwind direction (aligned with :math:`x_y` for zero tilt)

    **y-axis**: coincident with yaw-aligned y-axis

    **z-axis**: right-hand rule (vertical if zero tilt)




.. only:: latex

    TABLE CAPTION:: Yaw-Hub conversion methods

    .. autosummary::

        ~DirectionVector.yawToHub
        ~DirectionVector.hubToYaw


.. only:: html

    .. rubric:: Associated Methods

    .. autosummary::

        ~DirectionVector.yawToHub
        ~DirectionVector.hubToYaw

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~DirectionVector.yawToHub
        ~DirectionVector.hubToYaw


.. _hub_azimuth_coord:

Hub-aligned and Azimuth-aligned
---------------------------------

.. _hub-azimuth-fig:

.. figure:: /images/commonse/hub_azimuth.*
    :width: 4.5in
    :align: center

    Hub-aligned and azimuth-aligned axes.
    :math:`\Lambda` is the (local) blade azimuth angle.

:numref:`Figure %s <hub-azimuth-fig>` defines the transformation between the hub-aligned and azimuth-aligned coordinate systems.
The two coordinate systems share a common x-axis.
The azimuth angle :math:`\Lambda` is positive when rotating about the +x axis.
The blade can employ a variable azimuth angle along the blade axis, to allow for swept blades.

*Azimuth-aligned coordinate system*

    A rotating coordinate system---about the :math:`x_h` axis.
    The coordinate-system is locally-defined for the case of a variable-swept blade.

    **origin**: blade pitch axis, local to the blade section

    **x-axis**: aligned with the hub-aligned x-axis

    **y-axis**: right-hand rule

    **z-axis**: along projection of blade from root to tip in the :math:`y_h` - :math:`z_h` plane (aligned with blade only for zero precone)


.. only:: latex

    TABLE CAPTION:: Hub-Azimuth conversion methods

    .. autosummary::

        ~DirectionVector.hubToAzimuth
        ~DirectionVector.azimuthToHub


.. only:: html

    .. rubric:: Associated Methods

    .. autosummary::

        ~DirectionVector.hubToAzimuth
        ~DirectionVector.azimuthToHub

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~DirectionVector.hubToAzimuth
        ~DirectionVector.azimuthToHub




.. _azimuth_blade_coord:

Azimuth-aligned and Blade-aligned
---------------------------------

.. _azimuth-blade-fig:

.. figure:: /images/commonse/azimuth_blade.*
    :width: 3.5in
    :align: center

    Azimuth-aligned and blade-aligned axes.
    :math:`\Phi` is the (local) blade precone angle.


:numref:`Figure %s <azimuth-blade-fig>` defines the transformation between the azimuth-aligned and blade-aligned coordinate systems.
The :math:`y_b` and :math:`y_z` axes are in the same direction.
The two coordinate systems rotate together such that the :math:`x_b` - :math:`z_b` plane is always coplanar with the :math:`x_z` - :math:`z_z` plane.
The precone angle :math:`\Phi` is positive when rotating about the -:math:`y_z` axis, and causes the blades to tilt away from the nacelle/tower for a downwind machine (tilts toward tower for upwind machine).
The blade can employ a variable precone angle along the blade axis.
The blade-aligned coordinate system is considered local to a section of the blade.

.. _blade_coord:

*Blade-aligned coordinate system*

    A rotating coordinate system that rotates with the azimuth-aligned coordinate system.
    The coordinate-system is locally-defined along the blade radius. The direction of blade rotation is in the negative y-axis.
    A force in the x-axis would be a flapwise shear, and a force in the y-axis would be a lead-lag shear.

    **origin**: blade pitch axis, local to the blade section

    **x-axis**: follows from the right-hand rule (in nominal downwind direction)

    **y-axis**: opposite to rotation direction, positive from section leading edge to trailing edge (for no twist)

    **z-axis**: along the blade pitch axis in increasing radius



.. only:: latex

    TABLE CAPTION:: Azimuth-Blade conversion methods

    .. autosummary::

        ~DirectionVector.azimuthToBlade
        ~DirectionVector.bladeToAzimuth


.. only:: html

    .. rubric:: Associated Methods

    .. autosummary::

        ~DirectionVector.azimuthToBlade
        ~DirectionVector.bladeToAzimuth

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~DirectionVector.azimuthToBlade
        ~DirectionVector.bladeToAzimuth




.. _blade_airfoil_coord:

Blade-aligned and Airfoil-aligned
---------------------------------

.. _blade-airfoil-fig:

.. figure:: /images/commonse/blade_airfoil.*
    :width: 6in
    :align: center

    Blade-aligned and airfoil-aligned coordinate systems.
    :math:`\theta` is the airfoil twist + pitch angle.
    For convenience the local wind vector and angle of attack is shown.

:numref:`Figure %s <blade-airfoil-fig>` defines the transformation between the blade-aligned and airfoil-aligned coordinate systems.
The :math:`z_b` and :math:`z_a` axes are in the same direction.
The twist angle :math:`\theta` is positive when rotating about the -:math:`z_a` axis, and causes the angle of attack to decrease.

*Airfoil-aligned coordinate system*

    A force in the x-axis would be a flatwise shear, and a force in the y-axis would be an edgewise shear.

    **origin**: blade pitch axis, local to the blade section

    **x-axis**: follows from the right-hand rule

    **y-axis**: along chord line in direction of trailing edge

    **z-axis**: along the blade pitch axis in increasing radius, same as :math:`z_b` (into the page in above figure)


.. only:: latex

    TABLE CAPTION:: Blade-Airfoil conversion methods

    .. autosummary::

        ~DirectionVector.bladeToAirfoil
        ~DirectionVector.airfoilToBlade


.. only:: html

    .. rubric:: Associated Methods

    .. autosummary::

        ~DirectionVector.bladeToAirfoil
        ~DirectionVector.airfoilToBlade

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~DirectionVector.bladeToAirfoil
        ~DirectionVector.airfoilToBlade





.. _airfoil_profile_coord:

Airfoil-aligned and Profile
---------------------------

.. _airfoil-profile-fig:

.. figure:: /images/commonse/airfoil_profile.*
    :width: 6in
    :align: center

    Airfoil-aligned and profile coordinate systems.

:numref:`Figure %s <airfoil-profile-fig>` defines the transformation between the airfoil-aligned and profile coordinate systems.
The profile coordinate system is generally used only to define airfoil profile data.

*Profile coordinate system*

    **origin**: airfoil noise

    **x-axis**: positive from nose to trailing edge along chord line

    **y-axis**: orthogonal to x-axis, positive from lower to upper surface

    **z-axis**: n/a (profile is a 2-dimensional coordinate system)


.. only:: latex

    TABLE CAPTION:: Airfoil-Profile conversion methods

    .. autosummary::

        ~DirectionVector.airfoilToProfile
        ~DirectionVector.profileToAirfoil


.. only:: html

    .. rubric:: Associated Methods

    .. autosummary::

        ~DirectionVector.airfoilToProfile
        ~DirectionVector.profileToAirfoil

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~DirectionVector.airfoilToProfile
        ~DirectionVector.profileToAirfoil
