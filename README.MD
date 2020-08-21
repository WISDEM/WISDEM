# pCrunch
A collection of tools to ease the process of parsing large amounts of OpenFAST output data, loads analysis, etc... A bit of a spin-off of the MCRUNCH library, mostly python based, hence the name...

You will need to check out the [ROSCO_toolbox](https://github.com/NREL/ROSCO_toolbox/) to run these. If you don't care about any of the controller source code or actually using the ROSCO_toolbox, you can simply do:
```
pip install rosco-toolbox
```
some of the functionalities of this toolbox involves wisdem, but the primary functions do not. If you think you will find yourself needing [wisdem](https://github.com/wisdem/wisdem), you can simply use
```
conda install -c nikhar-abbas rosco-toolbox
```
to get both wisdem and the rosco_toolbox. Using conda does have its pitfalls, in that the installed packages are not installed in "develop" mode (which I guess is obvious if you don't clone the source-code anyways...).

Both wisdem and the rosco toolbox can, of course, be cloned and installed per their respective individual methods as well. This is encouraged if you plan to dig into the source code of either. 

NOTE:
Any spectral analysis will need the IEAOntology4All branch of [wisdem](https://github.com/wisdem/wisdem). 

## Some notes on structure
The primary file breakdown for the python tools is fairly simple and serve two primary functions: processing, and analysis. Unsurprisingly, these are separated into `processing.py` and `analysis.py`.The files themselves contain a few different classes that can stand alone (admittedly, some functions don't need to be in a class at all). 

Additionally, there is a file called `CaseGen_Control.py`. This leverages some WISDEM tools to aid in running parameter studies for different control parameters. The scripts in the runBatch and postProcessing folder soffer some examples on easy ways to run and process large OpenFAST runs (e.g. DLCs). The HPC_tools folder simply has some scripts to make using eagle just a little bit easier.   

## Examples
Take a look at batch_processing.ipynb for a detailed breakdown of some of the processes and discussion of some of the details.

A scripted implementation of a design comparison is provided postProcessing/post_DLCcomparison.py for a (hopefully) straight forward start to your own analysis. (This might be out of date now...)

## Contributing
Please do! This should hopefully be a tool that many people can use and leverage regularly. I know that pretty much all of this could be improved, cleaned up, etc. Ideally, this is a fluid project that a few primary people are maintaining. 

If you do contribute, please try to maintain the established conventions for commenting these codes (initial description of parameter and return values, reasonable in-line comments). 

## And finally...
Enjoy.

