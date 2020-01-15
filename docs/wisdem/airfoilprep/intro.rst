Introduction
------------

AirfoilPrep.py (pronounced Airfoil Preppy) provides functionality to preprocess aerodynamic airfoil data.  Essentially, the module is an object oriented version of the `AirfoilPrep spreadsheet <http://wind.nrel.gov/designcodes/preprocessors/airfoilprep/>`_ with additional functionality and is written in the Python language.  The intent is to provide the functionality of the AirfoilPrep spreadsheet, but in an easy-to-use format both for stand-alone preprocessing through scripting and for direct implementation within other codes such as blade element momentum methods.

AirfoilPrep.py allows the user to read in two-dimensional (2-D) aerodynamic airfoil data (i.e., from wind tunnel data or numerical simulation), apply three-dimensional (3-D) rotation corrections for wind turbine applications, and extend the data to very large angles of attack.  Airfoil data can also be blended together to define intermediate sections between linearly lofted sections.  Capabilities unique to the Python version include the ability to read and write to AeroDyn format files directly.  The only feature that is contained in the spreadsheet version but is currently missing in AirfoilPrep.py, is handling of pitching moment coefficients.

This document discusses installation, usage, and documentation of the module.  Because the theory is simplistic, only a brief overview is provided in the documentation section with corresponding references that contain further detail.
