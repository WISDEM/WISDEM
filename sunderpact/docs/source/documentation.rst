Module Documentation
--------------------

.. only:: latex

    An HTML version of this documentation is available which is better formatted for reading the code documentation and contains hyperlinks to the source code.

Turbine component sizing models for hub and drivetrain components are described along with mass-cost models for the full set of turbine components from the rotor to tower and foundation.

SubComponent
^^^^^^^^^^^^

The component objects in the Sunderland-WindPACT model set need only implement the __init__ method and set mass and mass property attributes.  The following attributes are present: mass, array of center of mass on an arbitrary axis, mass moments of inertia, and key dimensions if applicable for diameter, depth, length, height and width.  The interface contains no methods that must be implemented.



Hub System Physical Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

Hub system components models inherit from SubComponent as described above and contains additional functionality as described to compute the properties for SubComponent.

.. _Hub-class-label:

Hub
"""

.. currentmodule:: sunderpact.src.hubsystem

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: Hub
        :members:

.. only:: html

    .. autoclass:: Hub

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~Hub.__init__
            ~Hub.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~Hub.__init__
        ~Hub.update_mass


.. _PitchSystem-class-label:

PitchSystem
""""""""""""

.. currentmodule:: sunderpact.src.hubsystem

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: PitchSystem
        :members:

.. only:: html

    .. autoclass:: PitchSystem

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~PitchSystem.__init__
            ~PitchSystem.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~PitchSystem.__init__
        ~PitchSystem.update_mass


.. _Spinner-class-label:

Spinner
""""""""""""

.. currentmodule:: sunderpact.src.hubsystem

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: Spinner
        :members:

.. only:: html

    .. autoclass:: Spinner

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~Spinner.__init__
            ~Spinner.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~Spinner.__init__
        ~Spinner.update_mass


.. _HubSystem-class-label:

HubSystem
""""""""""""

.. currentmodule:: sunderpact.src.hubsystem

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: HubSystem
        :members:

.. only:: html

    .. autoclass:: HubSystem

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~HubSystem.__init__
            ~HubSystem.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~HubSystem.__init__
        ~HubSystem.update_mass





Nacelle System Physical Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Nacelle System Component models inherit from SubComponent as described above and contain additional functionality as described to compute the properties for SubComponent.

.. _LowSpeedShaft-class-label:

LowSpeedShaft
""""""""""""

.. currentmodule:: sunderpact.src.nacellesystem

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: LowSpeedShaft
        :members:

.. only:: html

    .. autoclass:: LowSpeedShaft

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~LowSpeedShaft.__init__
            ~LowSpeedShaft.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~LowSpeedShaft.__init__
        ~LowSpeedShaft.update_mass

.. _MainBearings-class-label:

MainBearings
""""""""""""

.. currentmodule:: sunderpact.src.nacellesystem 

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: MainBearings
        :members:

.. only:: html

    .. autoclass:: MainBearings

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~MainBearings.__init__
            ~MainBearings.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~MainBearings.__init__
        ~MainBearings.update_mass


.. _Gearbox-class-label:

Gearbox
""""""""""""

.. currentmodule:: sunderpact.src.nacellesystem 

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: Gearbox
        :members:

.. only:: html

    .. autoclass:: Gearbox

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~Gearbox.__init__
            ~Gearbox.update_mass
            ~Gearbox.getStageMass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~Gearbox.__init__
        ~Gearbox.update_mass
        ~Gearbox.getStageMass


.. _HighSpeedSide-class-label:

HighSpeedSide
""""""""""""

.. currentmodule:: sunderpact.src.nacellesystem 

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: HighSpeedSide
        :members:

.. only:: html

    .. autoclass:: HighSpeedSide

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~HighSpeedSide.__init__
            ~HighSpeedSide.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~HighSpeedSide.__init__
        ~HighSpeedSide.update_mass


.. _Generator-class-label:

Generator
""""""""""""

.. currentmodule:: sunderpact.src.nacellesystem 

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: Generator
        :members:

.. only:: html

    .. autoclass:: Generator

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~Generator.__init__
            ~Generator.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~Generator.__init__
        ~Generator.update_mass


.. _Bedplate-class-label:

Bedplate
""""""""""""

.. currentmodule:: sunderpact.src.nacellesystem 

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: Bedplate
        :members:

.. only:: html

    .. autoclass:: Bedplate

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~Bedplate.__init__
            ~Bedplate.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~Bedplate.__init__
        ~Bedplate.update_mass


.. _YawSystem-class-label:

YawSystem
""""""""""""

.. currentmodule:: sunderpact.src.nacellesystem 

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: YawSystem
        :members:

.. only:: html

    .. autoclass:: YawSystem

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~YawSystem.__init__
            ~YawSystem.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~YawSystem.__init__
        ~YawSystem.update_mass


.. _NacelleSystem-class-label:

NacelleSystem
""""""""""""

.. currentmodule:: sunderpact.src.nacellesystem 

.. rubric:: Class Summary:

.. only:: latex

    .. autoclass:: NacelleSystem
        :members:

.. only:: html

    .. autoclass:: NacelleSystem

        .. rubric:: Methods
        .. autosummary::
            :nosignatures:

            ~NacelleSystem.__init__
            ~NacelleSystem.update_mass

.. autogenerate
    .. autosummary::
        :toctree: generated

        ~NacelleSystem.__init__
        ~NacelleSystem.update_mass

