11. Airfoil Polar Preparation Example
---------------------------------------

.. currentmodule:: wisdem.airfoilprep

AirfoilPrep.py can be accessed either through the :ref:`command line <command-line-usage-label>` or through :ref:`Python <python-usage-label>`.
The command-line interface is the simplest but provides only a limited number of options.
The Python interface is useful for more advanced preprocessing and for integration with other codes.

.. _command-line-usage-label:

Command-Line Usage
^^^^^^^^^^^^^^^^^^
From the terminal, to see the options, invoke help:

.. code-block:: bash

   $ python airfoilprep.py -h

When using the command-line options, all files must be `AeroDyn <https://openfast.readthedocs.io/en/master/source/user/aerodyn/index.html>`_ formatted files.
The command line provides three main methods for working with files directly: 3-D stall corrections, high angle of attack extrapolation, and a blending operation.
In all cases, you first specify the name (and path if necessary) of the file you want to work with:

.. code-block:: bash

   $ python airfoilprep.py airfoil.dat

The following optional arguments are available

.. only:: latex

    TABLE CAPTION:: Available flags for using AirfoilPrep.py in command-line mode.

.. only:: html

    .. rubric:: Available Flags

================== =================  ============================================================
flag                arguments          description
================== =================  ============================================================
:code:`-h`                             display help
:code:`--stall3D`   r/R c/r tsr        3-D rotational corrections near stall
:code:`--extrap`    cdmax              high angle of attack extrapolation
:code:`--blend`     other weight       blend with other file using specified weight
:code:`--out`       outfile            specify a different name for output file
:code:`--plot`                         plot data (for diagnostic purposes) using matplotlib
:code:`--common`                       output airfoil data using a common set of angles of attack
================== =================  ============================================================

Stall Corrections
"""""""""""""""""

The first method available from the command line is :code:`--stall3D`, which reads the file, applies rotational corrections, and then writes the data to a separate file.
This argument must specify the parameters used for the correction in the format :code:`--stall3D r/R c/r tsr`, where :code:`r/R` is the local radius normalized by the rotor radius, :code:`c/r` is the local chord normalized by the local radius, and :code:`tsr` is the local tip-speed ratio.
For example, if :code:`airfoil.dat` contained 2-D data with :code:`r/R=0.5`, :code:`c/r=0.15`, :code:`tsr=5.0`, then we would apply rotational corrections to the airfoil using:

.. code-block:: bash

   $ python airfoilprep.py airfoil.dat --stall3D 0.5 0.15 5.0

By default the output file will append _3D to the name.
In the above example, the output file would be :code:`airfoil_3D.dat`.
However, this can be overridden with the :code:`--out` option.
To output to a file at :code:`/Users/Me/Airfoils/my_new_airfoil.dat`

.. _convert-cmdline-label:

.. code-block:: bash

   $ python airfoilprep.py airfoil.dat --stall3D 0.5 0.15 5.0 \
   > --out /Users/Me/Airfoils/my_new_airfoil.dat

Optionally, you can also plot the results (`matplotlib <http://matplotlib.org>`_ must be installed) with the :code:`--plot` flag.
For example,

.. code-block:: bash

   $ python airfoilprep.py DU21_A17.dat --stall3D 0.2 0.3 5.0 --plot

displays :numref:`Figure %s <stall-fig>` (only one Reynolds number shown) along with producing the output file.

.. _stall-fig:

.. figure:: /images/airfoilprep/stall3d.*
    :width: 6in

    Lift and drag coefficient with 3-D stall corrections applied.

.. _common-cmdline-label:

AirfoilPrep.py can utilize data for which every Reynolds number uses a different set of angles of attack.
However, some codes need data on a uniform grid of Reynolds number and angle of attack.
To output the data on a common set of angles of attack, use the :code:`--common` flag.

.. code-block:: bash

   $ python airfoilprep.py airfoil.dat --stall3D 0.5 0.15 5.0 --common


Angle of Attack Extrapolation
"""""""""""""""""""""""""""""

The second method available from the command line is :code:`--extrap`, which reads the file, applies high angle of attack extrapolations, and then writes the data to a separate file.
This argument must specify the maximum drag coefficient to use in the extrapolation across the full +/- 180-degree range :code:`--extrap cdmax`.
For example, if :code:`airfoil_3D.dat` contained 3D stall corrected data and :code:`cdmax=1.3`, then we could extrapolate the airfoil using:

.. _extrap-cmdline-label:

.. code-block:: bash

   $ python airfoilprep.py airfoil_3D.dat --extrap 1.3

By default the output file will append _extrap to the name.
In the above example, the output file would be :code:`airfoil_3D_extrap.dat`.
However, this can also be overridden with the :code:`--out` flag.
The :code:`--common` flag is also useful here if a common set of angles of attack is needed.

The output can be plotted with the --plot flag.
The command

.. code-block:: bash

   $ python airfoilprep.py DU21_A17_3D.dat --extrap 1.3 --plot

displays :numref:`Figure %s <extrap-fig>` (only one Reynolds number shown) along with producing the output file.

.. _extrap-fig:

.. figure:: /images/airfoilprep/extrap.*
    :width: 6in

    Airfoil data extrapolated to high angles of attack.

Blending
""""""""

The final capability accessible from the command line is blending of airfoils.
This is invoked through :code:`--blend filename weight`, where :code:`filename` is the name (and path if necessary) of a second file to blend with, and :code:`weight` is the weighting used in the blending.
The weight ranges on a scale of 0 to 1 where 0 returns the first airfoil and 1 the second airfoil.
For example, the following command blends airfoil1.dat with airfoil2.dat with a weighting of 0.3 (conceptually the new airfoil would equal 0.7*airfoil1.dat + 0.3*airfoil2.dat).

.. _blend-cmdline-label:

.. code-block:: bash

   $ python airfoilprep.py airfoil1.dat --blend airfoil2.dat 0.3

By default, the output file appends the names of the two files with a '+' sign, then appends the weighting using '_blend' and the value for the weight.
In this example, the output file would be :code:`airfoil1+airfoil2_blend0.3.dat`.
Just like the previous case, the name of the output file can be overridden by using the :code:`--out` flag.
The :code:`--common` flag is also useful here if a common set of angles of attack is needed.
This data can also be plotted, but only the blended airfoil data will be shown.
Direct comparison to the original data is not always possible, because the blend method allows for the specified airfoils to be defined at different Reynolds numbers.
Blending first occurs across Reynolds numbers and then across angle of attack.

.. _python-usage-label:

Python Usage
^^^^^^^^^^^^

The Python interface allows for more flexible usage or integration with other programs.
Descriptions of the interfaces for the classes contained in the module are contained in :ref:`interfaces-label`.

This complete example script can be found at ``WISDEM/examples/11_airfoilprep/example.py``.

Airfoils can be created from `AeroDyn <https://openfast.readthedocs.io/en/master/source/user/aerodyn/index.html>`_ formatted files,

.. literalinclude:: ../../../examples/11_airfoilprep/example.py
    :start-after: # Imports
    :end-before: # ------

or they can be created directly from airfoil data.

.. literalinclude:: ../../../examples/11_airfoilprep/example.py
    :start-after: # first polar
    :end-before: # ------



Blending is easily accomplished just like in the :ref:`command-line interface <blend-cmdline-label>`.
There is no requirement that the two airfoils share a common set of angles of attack.

.. literalinclude:: ../../../examples/11_airfoilprep/example.py
    :start-after: # read in
    :end-before: # ------


Applying 3-D corrections and high alpha extensions directly in Python, allows for a few additional options as compared to the command-line version.
The following example performs the same 3-D correction as in the :ref:`command-line version <convert-cmdline-label>`, followed by an alternative 3-D correction that utilizes some of the optional inputs.
See :py:meth:`correction3D <Airfoil.correction3D>` for more details on the optional parameters.

.. literalinclude:: ../../../examples/11_airfoilprep/example.py
    :start-after: # apply 3D corrections as desired
    :end-before: # ------


The airfoil data can be extended to high angles of attack using the :py:meth:`extrapolate <Airfoil.correction3D>`
method.
Just like the previous method, a few optional parameters are available through the Python interface.
The following example performs the same extrapolation as in the :ref:`command-line version <extrap-cmdline-label>`, followed by an alternative extrapolation that utilizes some of the optional inputs.

.. literalinclude:: ../../../examples/11_airfoilprep/example.py
    :start-after: # extend airfoil
    :end-before: # ------


Some codes need to use the same set of angles of attack data for every Reynolds number defined in the airfoil.
The following example performs the same method as in the :ref:`command-line version <common-cmdline-label>` followed by an alternate approach where the user can specify the set of angles of attack to use.

.. literalinclude:: ../../../examples/11_airfoilprep/example.py
    :start-after: # create new
    :end-before: # ------



For direct access to the underlying data in a grid format (if not already a grid, it is interpolated to a grid first), use the :py:meth:`createDataGrid <Airfoil.createDataGrid>` method as follows:

.. literalinclude:: ../../../examples/11_airfoilprep/example.py
    :start-after: # extract
    :end-before: # ------


Finally, writing AeroDyn formatted files is straightforward.

.. literalinclude:: ../../../examples/11_airfoilprep/example.py
    :start-after: # write
    :end-before: # ------
