.. _tutorial-label:

.. currentmodule:: wisdem.aeroelasticse

Tutorial
========

The following examples are called as test cases in the modules  (:mod:`AeroelasticSE.runFAST`,  :mod:`AeroelasticSE.runTurbSim`,  :mod:`AeroelasticSE.FusedFAST`, :mod:`AeroelasticSE.FAST_component`, :mod:`AeroelasticSE.iecApp`).

For all but the VariableTree based approach, FAST is run via "template" input files, whereby the user sets up a working FAST
input deck, and the python wrapper parses and manipulates the data in these files.  

For the most part, this is all accomplished by python dictionaries, easily modified to set parameters in a custom fashion.  The exceptions
to this rule regard file names.  The need to accomodate cross-platform file names necessitates a slightly more careful treatment.  For example,
files written as `AFDIR\afname.dat` need to be correctly parsed such that on linux and mac they end up as `AFDIR/afname.dat`.

But for the most part the interface is by setting up a working FAST input set, then addressing the variables of the input
programmically in python via dictionaries.  Some of the several scenarios are described below.


running FAST one time
---------------------

The FAST wrapper implemented in :mod:`runFAST.py` adopts a "template-based" approach to running FAST.  That is,
the user is expected to have access to a previously prepared directory containing working FAST input files (e.g., such that
FAST could be run from the command line in that directory).  Then, to use the WISDEM FAST wrapper, the user
just needs to point WISDEM to that directory, and to the FAST executable, and to the particular "main" FAST
input file (e.g., the ".fst" file).  The code then parses the input files into python dictionaries.  The python user
drives FAST by modifying the contents of these input files, then allowing the wrapper to rewrite these input files
and run FAST.  Everything except the modification of the dictionary is hidden from the user, resulting in a 
`pythonic` interface.

(Note: a "Variable-tree" based approach, in which the entire turbine
is represented as an openMDAO ``VariableTree`` object, is described in :mod:`AeroelasticSE.FSTTemplate_runner`.)

The following code, found at the bottom of :mod:`runFAST.py`, illustrates to basic process of running FAST via
the template based approach of :mod:`runFAST.py`.  

.. literalinclude:: ../../../wisdem/aeroelasticse/runFAST.py
    :start-after: def example
    :end-before: def turbsim_example

To run it successfully, you should edit the file (or copy the example to a new file)
and change the ``.fst_exe``, ``.fst_dir``, ``.fst_file``, and ``.run_dir`` fields of
the ``runFast`` object
to reflect the location of your FAST executable and template input files.

Then run::

  python runFAST.py

This will print something like:

>>> max power
>>> 11840.0

Run::

  python runFAST.py -t

to run TurbSim first, then FAST.


running TurbSim one time
------------------------

There is a similar example for TurbSim in :mod:`TurbSim.py`:

.. literalinclude:: ../../../wisdem/aeroelasticse/runTurbSim.py
    :start-after: if __name__==

Again note, the user is required to specify
``.ts_exe``, ``.ts_dir``, ``.ts_file``, and ``.run_dir``
fields for the ``TurbSim`` object. 

running FusedFAST
-----------------

:mod:`FusedFAST` uses the :mod:`fusedwind` framework.  The input to the code is in the form of
``GenericRunCase`` objects.  These represent inputs in a simple, generic way that is independent of any particular
turbine code.  The ``FusedFAST`` object translates these to a form required by ``FAST`` and then runs
``FAST``.

.. literalinclude:: ../../../wisdem/aeroelasticse/FusedFAST.py
    :start-after: def openFAST_test

running FAST_component
----------------------

The ``FAST_component`` presents ``runFAST`` as an openMDAO ``Component``.
It forms a base class for FAST wrappers that utilize the openMDAO input and output
conventions, while still utilizing a template input file-based approach to FAST. 

.. literalinclude:: ../../../wisdem/aeroelasticse/FAST_component.py
    :start-after: def FAST_component_test
    :end-before: end FAST_component_test


running FAST_iter_component
---------------------------

The ``FAST_iter_component`` allows FAST to be run multiple times  (e.g. for making a power curve)
in a very simple manner.

.. literalinclude:: ../../../wisdem/aeroelasticse/FAST_component.py
    :start-after: def FAST_iter_component_test
    :end-before: end FAST_iter_component_test



running iecApp
--------------

:mod:`iecApp` uses the :mod:`fusedwind` framework to run large numbers of FAST cases for, e.g., IEC standards
analysis. The input to the code is a text file containing a table of
run cases.  

To run this code, first prepare a file of cases, e.g. ``some_cases.txt`` and a file of control input, e.g.  ``runbatch-control``.
Examples of these::

    (openmdao-0.9.5)stc-24038s:AeroelasticSE pgraf$ more some_cases.txt
    AnalTime Vhub Hs Tp WaveDir Prob
    3.00e+00 1.00e+01 1.59e+00 1.14e+01 -1.01e+00  1.70e-02
    3.00e+00 1.02e+01 9.49e-01 8.37e+00 -9.48e-01  4.88e-03
    3.00e+00 9.95e+00 1.81e+00 1.14e+01 -4.38e-01  1.78e-02
    3.00e+00 8.87e+00 1.49e+00 1.37e+01 2.29e-03  7.63e-03
    3.00e+00 9.05e+00 1.25e+00 1.09e+01 -1.71e+00  5.62e-03
    # in reality this file might have thousands of cases...

    (openmdao-0.9.5)stc-24038s:AeroelasticSE pgraf$ more runbatch-control.txt
    # key = value file of locations of various files to read and
    # to write
    # and for output keys.
    # and misc. control functionality    
    main_output_file = "runbatch.out"
    output_keys =  RootMxc1  RootMyc1
    output_operations = rf np.std max
    ts_exe = "/Users/pgraf/opt/windcode-7.31.13/TurbSim/build/TurbSim_glin64"
    ts_dir = "/Users/pgraf/work/wese/fatigue12-13/from_gordie/SparFAST3.orig/TurbSim"
    ts_file = "TurbSim.inp"
    fst_exe = "/Users/pgraf/opt/windcode-7.31.13/build/FAST_glin64"
    fst_dir = "/Users/pgraf/work/wese/fatigue12-13/from_gordie/SparFAST3.almostorig"
    fst_file = "NRELOffshrBsline5MW_Floating_OC3Hywind.fst"
    run_dir = "all_runs"

The cases file is a simple table of FAST input variable to change.  These are translated to FAST "language" (e.g. 
``AnalTime`` becomes ``Tmax``) by the software.  The keys in the control file are all self-explanatory except for
``output_operations``.  These are the names of python functions that will be applied to the output array to reduce the
time sequence for each sensor to a single scalar.  These can be arbitrary python ``<module>.<function_name>`` directives.  So,
for example if you had a custom postprocessor function ``reduce_data`` in module ``mystudy`` you would use::

    output_operations = mystudy.reduce_data

Here the function ``reduce_data`` accepts a numpy array and returns a scalar.
Run::

    python iecApp.py -i some_cases.txt -f runbatch-control.txt

Hopefully you will see FAST running many times, resulting in stdout like::

    RUNS ARE DONE:
    collecting output from copied-back files (not from case recorder), see runbatch.out
    processing case <fusedwind.runSuite.runCase.GenericRunCase object at 0x103e1af50>
    collecting from  /Users/pgraf/work/wese/AeroelasticSE-1_3_14/wisdem/aeroelasticse/all_runs/raw_casesAna.3.0Wav.-1.0Hs.1.6Vhu.10.0Tp.11.4Pro.0.0
    collecting from  /Users/pgraf/work/wese/AeroelasticSE-1_3_14/wisdem/aeroelasticse/all_runs/raw_casesAna.3.0Wav.-0.9Hs.0.9Vhu.10.2Tp.8.4Pro.0.0
    collecting from  /Users/pgraf/work/wese/AeroelasticSE-1_3_14/wisdem/aeroelasticse/all_runs/raw_casesAna.3.0Wav.-0.4Hs.1.8Vhu.10.0Tp.11.4Pro.0.0
    collecting from  /Users/pgraf/work/wese/AeroelasticSE-1_3_14/wisdem/aeroelasticse/all_runs/raw_casesAna.3.0Wav.0.0Hs.1.5Vhu.8.9Tp.13.7Pro.0.0
    collecting from  /Users/pgraf/work/wese/AeroelasticSE-1_3_14/wisdem/aeroelasticse/all_runs/raw_casesAna.3.0Wav.-1.7Hs.1.3Vhu.9.1Tp.10.9Pro.0.0

and a file ``runbatch.out`` with the output in the form of a space separated text file table.


running FSTVT_runner
----------------------

An example use of the variable tree version of the FAST aeroelastic code wrapper is provided in FSTVT_runner.py.  The code sets up a FAST variable tree from a set of template files for the NREL 5 MW fast turbine; it then updates the environmental variables for wind conditions and runs through cases for those conditions.  First we import the necessary files and set up our case FAST case iterator class.


.. literalinclude:: ../../../wisdem/aeroelasticse/FAST_VT/FSTVT_runner.py
    :start-after: # 1 ---
    :end-before: # 1 ---

Then we create the cases from FUSED-Wind for the environmental conditions of interest.

.. literalinclude:: ../../../wisdem/aeroelasticse/FAST_VT/FSTVT_runner.py
    :start-after: # 2 ---
    :end-before: # 2 ---

We then run those cases.


.. literalinclude:: ../../../wisdem/aeroelasticse/FAST_VT/FSTVT_runner.py
    :start-after: # 3 ---
    :end-before: # 3 ---
