.. _tutorial-label:

.. currentmodule:: rotorse.rotoraero

Tutorial
--------

You can refer to classes (:class:`AeroBase`), or methods (:meth:`CCAirfoil.initFromAerodynFile`) assuming you have set the module with ..currentmodule or ..module.  You can also refer to modules :mod:`rotorse.rotoraero`.

You might want to include a figure (:num:`Figure #somelabel-fig`)

.. _somelabel-fig:

.. figure:: /images/figurename.*
    :height: 4in
    :align: center

    Caption goes here

You can also include code from an example.  This is preferable to writing the actual code here in this page, because then you can run the example and test it.

.. literalinclude:: ../src/rotorse/rotoraerodefaults.py
    :start-after: # --- rotor geometry
    :end-before: # ---

Print out some results from the code

>>> CP = [0.48329808]
>>> CT = [0.7772276]
>>> CQ = [0.06401299]
