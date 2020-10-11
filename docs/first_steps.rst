WISDEM First Steps
==================

This document gives a step-by-step introduction to running WISDEM

Files required to run WISDEM
----------------------------

In the following table, each type of file is listed as a row with columns describing the type of file, a link to a working example of that file type, and a link to documentation of that file type.

+---------------------------+---------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+
| Description               | Suggested Default to Modify                                                                                                     | Where to learn more                                                          |
+===========================+=================================================================================================================================+==============================================================================+
| Geometry of the turbine   | `nrel5mw.yaml <https://github.com/WISDEM/WISDEM/blob/develop/examples/reference_turbines_lcoe/nrel5mw.yaml>`_                   | `WindIO docs <https://windio.readthedocs.io/en/latest/source/turbine.html>`_ |
+---------------------------+---------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+
| Modeling options          | `modeling_options.yaml <https://github.com/WISDEM/WISDEM/blob/develop/examples/reference_turbines_lcoe/modeling_options.yaml>`_ | TODO: Add docs here                                                          |
+---------------------------+---------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+
| Analysis options          | `analysis_options.yaml <https://github.com/WISDEM/WISDEM/blob/develop/examples/reference_turbines_lcoe/analysis_options.yaml>`_ | TODO: Add docs here                                                          |
+---------------------------+---------------------------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------+

One of the methods you can use to setup your own run runs is to copy our example files from reference turbines and modify them to suit your needs. Follow these steps for this method:

1. Create a new folder for your new files.
2. Make copies of each of the files in the table above and place them into your new folder.
3. Edit each of the files using either using a text editor (such as Visual Studio Code) or the WISDEM GUI.

The rest of this tutorial focuses on the WISDEM GUI.

Using the WISDEM GUI
--------------------

Loading the files
~~~~~~~~~~~~~~~~~

The WISDEM GUI is laid out from left to right to edit the geometry, modeling, and analysis files. The upper part of the interface shows a status bar and at the right side there is a `Run WISDEM` button.

Launch the GUI by activating your WISDEM virtual environment and typing the command `wisdem` on the command line. A window similar to the following will appear:

.. figure:: /images/yaml/wisdem_gui_step_01.png

First, load a geometry file. The `nrel5mw.yaml` file is loaded in the following figure. Load this file by clicking the `Select geometry YAML` button and selecting your copy of `nrel5mw.yaml`.

.. figure:: /images/yaml/wisdem_gui_step_02.png

Similarly, open your copies of the `modeling_options.yaml` as in the following figure:

.. figure:: /images/yaml/wisdem_gui_step_03.png

Finally, open the `analysis_options.yaml` as seen in this figure:

.. figure:: /images/yaml/wisdem_gui_step_04.png

Running WISDEM
~~~~~~~~~~~~~~

In the GUI, click on the `Run WISDEM` button. The following dialog box will appear

.. figure:: /images/yaml/wisdem_gui_step_05.png

When you see this dialog box, the GUI has written the YAML files. WISDEM takes a long time to run, so you are asked to confirm that you want to execute the run of WISDEM. Click `OK` to continue. Once you click `OK`, the GUI will stop responding while WISDEM executes. Watch the command line window for messages as WISDEM executes. When WISDEM has finished, you will see the following message:

.. figure:: /images/yaml/wisdem_gui_step_06.png
