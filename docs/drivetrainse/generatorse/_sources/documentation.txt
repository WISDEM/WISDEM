.. _documentation-label:

.. currentmodule:: GeneratorSE

Documentation
--------------

.. only:: latex

    An HTML version of this documentation is available which is better formatted for reading the code documentation and contains hyperlinks to the source code.


Sizing models for the different generator modules are described along with mass, cost and efficiency as objective functions .

Documentation for GeneratorSE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inputs and outputs are defined for GeneratorSE.PMSG_arms :

.. literalinclude:: ../src/generatorse/PMSG_arms.py
    :language: python
    :start-after: PMSG(Component)
    :end-before: def configure(self)
    :prepend: class Drive_PMSG_arms(Assembly):

.. module:: generatorse.PMSG_arms
.. class:: generatorse.PMSG_arms.Drive_PMSG_arms

The following inputs and outputs are defined for PMSG_disc :

.. literalinclude:: ../src/generatorse/PMSG_disc.py
    :language: python
    :start-after: PMSG(Component)
    :end-before: def configure(self)
    :prepend: class Drive_PMSG_disc(Assembly):

.. module:: generatorse.PMSG_disc
.. class:: generatorse.PMSG_disc.Drive_PMSG_disc

The following inputs and outputs are defined for EESG :

.. literalinclude:: ../src/generatorse/EESG.py
    :language: python
    :start-after: EESG(Component)
    :end-before: def configure(self)
    :prepend: class Drive_EESG(Assembly):

.. module:: generatorse.EESG
.. class:: generatorse.EESG.Drive_EESG


The following inputs and outputs are defined for SCIG :

.. literalinclude:: ../src/generatorse/SCIG.py
    :language: python
    :start-after: SCIG(Component)
    :end-before: def configure(self)
    :prepend: class Drive_SCIG(Assembly):

.. module:: generatorse.SCIG
.. class:: generatorse.SCIG.Drive_SCIG

The following inputs and outputs are defined for DFIG :

.. literalinclude:: ../src/generatorse/DFIG.py
    :language: python
    :start-after: DFIG(Component)
    :end-before: def configure(self)
    :prepend: class Drive_DFIG(Assembly):

.. module:: generatorse.DFIG
.. class:: generatorse.DFIG.Drive_DFIG




