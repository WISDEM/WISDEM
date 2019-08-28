Installation
------------

.. admonition:: Prerequisites
   :class: warning

	General: NumPy, SciPy, Swig, pyWin32, MatlPlotLib, Lxml, OpenMDAO

	Wind Plant Framework: FUSED-Wind (Framework for Unified Systems Engineering and Design of Wind Plants)

	Supporting python packages: Pandas, Algopy, Zope.interface, Sphinx, Xlrd, PyOpt, py2exe, Pyzmq, Sphinxcontrib-bibtex, Sphinxcontrib-zopeext, Numpydoc, Ipython

Clone the repository at `<https://github.com/WISDEM/AeroelasticSE>`_ or download the releases and uncompress/unpack from the website link at the bottom the `AeroelasticSE site<http://nwtc.nrel.gov/AeroelasticSE>`_.

Install the AeroelasticSE plugin with the following command from an activated OpenMDAO environemnt.

.. code-block:: bash

   $ plugin install

To check if installation was successful try to import the module from an activated OpenMDAO environment.

.. code-block:: bash

    $ python -c "import AeroelasticSE"

Then run the unit tests for the various FAST wrappers.
NOTE: these will require the user to correctly set the location of FAST and its input files.
Currently this is done by editing the source files themselves.  (The tests are in the
source modules).  

.. code-block:: bash

  $ python runFAST.py --help
  $ python runFAST.py
  $ python runFAST.py -t
  $ python FAST_component.py
  $ python FusedFAST.py 
  $ python iecApp.py -i some_cases.txt -f runbatch-control.txt
  
Please see :ref:`aeroelasticse`.
