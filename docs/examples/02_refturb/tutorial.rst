.. _refturb_tutorial-label:

2. Running the NREL 5-MW and IEA Wind 15-MW Reference Wind Turbines
-------------------------------------------------------------------

The next example involves running two reference turbine examples, the NREL 5-MW land-based and IEA Wind 15-MW offshore fixed-bottom turbines.  There are multiple ways to run WISDEM, each is perfectly valid and users should adopt whatever approach they are most comfortable with.

Calling WISDEM
**************

Option 1. Use the GUI
=====================

Installing WISDEM creates a :code:`wisdem` command that should be available at the command prompt (a terminal prompt on Linux / Mac machines or the Anaconda PowerShell prompt for Windows).  Just typing :code:`wisdem` at the command line opens the GUI.  From there, you can use the dialogue menus to load in the yaml input files and run WISDEM.

Option 2. Pass YAML-files directly to the ``wisdem`` command
==============================================================

Installing WISDEM creates a :code:`wisdem` command that should be available at the command prompt (a terminal prompt on Linux / Mac machines or the Anaconda PowerShell prompt for Windows).  You can pass that command the three input yaml-files in order to run WISDEM.

.. code-block:: bash

    $ wisdem nrel5mw.yaml modeling_options.yaml analysis_options.yaml

Alternatively, you can create a summary WISDEM file that points to each file,

.. code-block:: bash

    $ wisdem nrel5mw_driver.yaml

Where the contents of ``nrel5mw_driver.yaml`` are,

.. literalinclude:: /../examples/02_reference_turbines/nrel5mw_driver.yaml
    :language: yaml

Note that to run the IEA Wind 15-MW reference wind turbine, simply substitute the file, ``IEA-15-240-RWT.yaml``, in as the geometry file.  The ``modeling_options.yaml`` and ``analysis_options.yaml`` file can remain the same.

Option 3. Call WISDEM from a Python Script
==========================================

For those users who are comfortable with Python scripting, the WISDEM yaml input files can also be passed as path names to the main WISDEM function in this way,

.. code:: bash

    $ python nrel5mw_driver.py

Where the contents of ``nrel5mw_driver.py`` are,

.. literalinclude:: /../examples/02_reference_turbines/nrel5mw_driver.py
    :start-after: #!
    :end-before: # end

Screen Output
*************

Successfully running WISDEM should show the following screen output for the NREL 5-MW reference wind turbine (note that exact number values will vary based on WISDEM version and computer):

.. code:: console

    =========
    wt.wt_rna
    =========
    NL: NLBGS 1 ; 4.0637118e+11 1
    NL: NLBGS 2 ; 5128312.48 1.26197741e-05
    NL: NLBGS 3 ; 0 0
    NL: NLBGS Converged
    ########################################
    Objectives
    Turbine AEP: 23.9127972556 GWh
    Blade Mass:  16419.8666989212 kg
    LCOE:        51.6906874361 USD/MWh
    Tip Defl.:   4.7295143980 m
    ########################################
    Completed in, 21.37621808052063 seconds
    blade mass: [16419.86669892]
    blade moments of inertia: [35590815.60501075 17795407.80250537 17795407.80250537 0. 0. 0.]
    BRFM: [-16256766.58646931]
    hub forces: [1171726.2468185     9709.60828863    9930.09366145]
    hub moments: [11075137.65779509  1317752.66701219  1200194.31654923]

And this output for the IEA Wind 15-MW reference wind turbine:

.. code:: console

    =========
    wt.wt_rna
    =========
    NL: NLBGS 1 ; 4.89089054e+11 1
    NL: NLBGS 2 ; 13008245.1 2.65968846e-05
    NL: NLBGS 3 ; 301966.026 6.17404998e-07
    NL: NLBGS 4 ; 7593.53109 1.5525866e-08
    NL: NLBGS 5 ; 190.680588 3.89868853e-10
    NL: NLBGS Converged
    ORBIT library intialized at '/Users/gbarter/devel/WISDEM/wisdem/library'
    ########################################
    Objectives
    Turbine AEP: 77.9037579237 GWh
    Blade Mass:  68206.4068005262 kg
    LCOE:        85.9788523601 USD/MWh
    Tip Defl.:   25.8318262264 m
    ########################################
    Completed in, 28.88100290298462 seconds

Some helpful summary information is printed to the screen.  More detailed output can be found in the ``outputs`` directory.  This creates output files that can be read-in by Matlab, Numpy, Python pickle-package, and Excel.  These files have the complete list of all WISDEM variables (with extended naming based on their OpenMDAO Group hierarchy) and the associated values.  An output yaml-file is also written, in case any input values were altered in the course of the analysis.

.. code:: bash

    $ ls -1 outputs

    refturb_output.mat
    refturb_output.npz
    refturb_output.pkl
    refturb_output.xlsx
    refturb_output.yaml
    refturb_output-modeling.yaml
    refturb_output-analysis.yaml


+-----------+-------------------------+
| Extension | Description             |
+===========+=========================+
| ``.mat``  | MatLab output format    |
+-----------+-------------------------+
| ``.npz``  | Archive of NumPy arrays |
+-----------+-------------------------+
| ``.pkl``  | Python Pickle format    |
+-----------+-------------------------+
| ``.xlsx`` | Microsoft Excel format  |
+-----------+-------------------------+
| ``.yaml`` | YAML format             |
+-----------+-------------------------+

As an example, the ``sample_plot.py`` script plots Axial Induction versus Blade Nondimensional Span by extracting the values from the Python pickle file.  The script content is:


.. literalinclude:: ../../../examples/02_reference_turbines/sample_plot.py
    :language: python

This script generates the following plot:

.. figure:: /images/yaml/first_steps_first_plot.png
