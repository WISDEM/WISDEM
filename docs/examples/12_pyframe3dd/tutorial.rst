.. _pyframe3dd-tutorial-label:


12. pyFrame3DD Example
========================

This document walks through the pyFrame3dd usage that matches the `Pyramid Frame (B) example <http://frame3dd.sourceforge.net/>`_ provided by Frame3DD.

.. contents:: pyFrame3dd Analysis Steps
   :depth: 2

Geometry
--------

Setting the geometry of the structure involves specifying node locations, element cross-section properties, and boundary conditions.  The inputs can be of any units that the user desires, as long as they are self-consistent across all of the entries.

The node locations are specified by:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 0 ---
    :end-before: # 0 ---

The boundary conditions are specified by listing which node degrees of freedom (DOF) have reactions, or are fixed:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 1 ---
    :end-before: # 1 ---


The element cross sections include area, "shear area", moments of inertia, Young's and shear moduli, and material density:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 2 ---
    :end-before: # 2 ---


The final geometry element specifies some modeling parameters for Frame3DD:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 3 ---
    :end-before: # 3 ---


Now we can create a full pyFrame3DD Frame object that stores the geometry:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 4 ---
    :end-before: # 4 ---


Loading
--------

Frame3DD can assess many different load cases on the same structure simultaneously.  It can also handle many different types of loading, only a few of which are featured in this example.

The first load case uses the standard static gravity load and a point force acting in all directions (with no moment).  Note that pyFrame3DD initializes all load objects through the gravity static load call.  Then the load case must be added to the Frame object:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 5 ---
    :end-before: # 5 ---

The second load case starts with the same gravity static load, but then adds in uniform and trapezoidally distributed loads along specific elements.  Note that element loads are specified in the element coordinate system, not the global coordinate system of the nodes.  Finally, a temperature load is also added:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 6 ---
    :end-before: # 6 ---

The final load case features internal loads, although this is not a feature that is used within WISDEM:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 7 ---
    :end-before: # 7 ---

Modal Analysis
--------------

Frame3DD includes extensive modal analysis options and outputs.  These are set in pyFrame3DD via the following call:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 8 ---
    :end-before: # 8 ---

Added Mass
----------

An extra feature of pyFrame3DD is the ability to include extra mass in both the load and modal calculations.  WISDEM takes full advantage of this feature, especially in the support structure analyses.  However, this means care must be taken to only add nodal mass once all of the load cases have already been added to the Frame object:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 9 ---
    :end-before: # 9 ---


Simulation
----------

Running the Frame object with its load cases is a simple call:

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 10 ---
    :end-before: # 10 ---


Output
------

Analysis output is available using the same keywords as the Frame3DD manual, and is all available as Numpy arrays.  Outputs of interest and interrogation of the data is user and application specific.  In this example, the full data structure is simply printed to the screen.

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 11 ---
    :end-before: # 11 ---


Lastly, we can also output a Frame3DD file using a built-in method.

.. literalinclude:: /../examples/12_pyframe3dd/exB.py
    :start-after: # 12 ---
    :end-before: # 12 ---
