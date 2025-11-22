# WISDEM&reg;

[![Actions Status](https://github.com/WISDEM/WISDEM/workflows/CI_WISDEM/badge.svg?branch=develop)](https://github.com/WISDEM/WISDEM/actions)
[![Coverage Status](https://coveralls.io/repos/github/WISDEM/WISDEM/badge.svg?branch=develop)](https://coveralls.io/github/WISDEM/WISDEM?branch=develop)
[![Documentation Status](https://readthedocs.org/projects/wisdem/badge/?version=master)](https://wisdem.readthedocs.io/en/master/?badge=master)


The Wind-Plant Integrated System Design and Engineering Model (WISDEM&reg;) is a set of models for assessing overall wind plant cost of energy (COE). The models use wind turbine and plant cost and energy production as well as financial models to estimate COE and other wind plant system attributes. WISDEM&reg; is accessed through Python, is built using [OpenMDAO](https://openmdao.org/), and uses several sub-models that are also implemented within OpenMDAO. These sub-models can be used independently but they are required to use the overall WISDEM&reg; turbine design capability. Please install all of the pre-requisites prior to installing WISDEM&reg; by following the directions below. For additional information about the NWTC effort in systems engineering that supports WISDEM&reg; development, please visit the official [NREL systems engineering for wind energy website](https://www.nrel.gov/wind/systems-engineering.html).

Author: [NREL WISDEM Team](mailto:systems.engineering@nrel.gov)

## Part of the WETO Stack

WISDEM is primarily developed with the support of the U.S. Department of Energy and is part of the [WETO Software Stack](https://nrel.github.io/WETOStack). For more information and other integrated modeling software, see:
- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [Systems Engineering Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#systems-engineering)

## Documentation

See local documentation in the `docs`-directory or access the online version at <https://wisdem.readthedocs.io/en/master/>

## Packages

WISDEM&reg; is a family of modules.  The core modules are:

* _CommonSE_ includes several libraries shared among modules
* _FloatingSE_ works with the floating platforms
* _DrivetrainSE_ sizes the drivetrain and generator systems (formerly DriveSE and GeneratorSE)
* _TowerSE_ is a tool for tower (and monopile) design
* _RotorSE_ is a tool for rotor design
* _NREL CSM_ is the regression-based turbine mass, cost, and performance model
* _ORBIT_ is the process-based balance of systems cost model for offshore plants
* _LandBOSSE_ is the process-based balance of systems cost model for land-based plants
* _Plant_FinanceSE_ runs the financial analysis of a wind plant

The core modules draw upon some utility packages, which are typically compiled code with python wrappers:

* _Airfoil Preppy_ is a tool to handle airfoil polar data
* _CCBlade_ is the BEM module of WISDEM
* _pyFrame3DD_ brings libraries to handle various coordinate transformations
* _MoorPy_ is a quasi-static mooring line model
* [_pyOptSparse_](https://github.com/mdolab/pyoptsparse) provides some additional optimization algorithms to OpenMDAO

## Installation

Installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WISDEM&reg; requires [Anaconda 64-bit](https://www.anaconda.com/distribution/).  However, the `conda` command has begun to show its age and we now recommend the one-for-one replacement with the [Miniforge3 distribution](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3), which is much more lightweight and more easily solves for the WISDEM package dependencies.


### Installation as a "library"

1. Create a conda environment with your preferred name (`wisdem-env` in the following example) and favorite, approved Python version:

   ```console
   conda create -n wisdem-env python=3.13 -y
   ```

2. Activate the environment:

   ```console
   conda activate wisdem-env
   ```

3. Install WISDEM via a `conda` or `pip`. We highly recommend via conda.

    ```console
    conda install wisdem
    ```

    or

    ```console
    pip install wisdem
    ```

To use WISDEM's modules as a library for incorporation into other scripts or tools, WISDEM is available via `conda install wisdem` or `pip install wisdem`, assuming that you have already setup your python environment.  Note that on Windows platforms, we suggest using `conda` exclusively.

### Installation for direct use or development

These instructions are for interaction with WISDEM directly, the use of its examples, and the direct inspection of its source code.

The installation instructions below use the environment name, "wisdem-env," but any name is acceptable. Below are a series of considerations:
- For those working behind company firewalls, you may have to change the conda authentication with `conda config --set ssl_verify no`.
- Proxy servers can also be set with `conda config --set proxy_servers.http http://id:pw@address:port` and `conda config --set proxy_servers.https https://id:pw@address:port`.
- To setup an environment based on a different Github branch of WISDEM, simply substitute the branch name for `master` in the setup line.

> **Note**
> For Windows users, we recommend installing `git` and the `m2w64` packages in separate environments as some of the
> libraries appear to conflict such that WISDEM cannot be successfully built from source.  The `git` package is best
> installed in the `base` environment.

#### Direct use

We still highly recommend users use `conda install wisdem` into an environment, but if there is a reason that is not
desired, please use the following instructions.

Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)

> [!IMPORTANT]
> In the `environment.yaml` please uncomment out the OS-dependent dependencies at the top

1. Install `git` if you don't already have it:

    ```console
    conda install git
    ```

2. Clone the repository and enter it:

    ```console
    git clone https://github.com/WISDEM/WISDEM.git
    cd WISDEM
    ```

3. Checkout the desired branch, if necessary:

    ```console
    git checkout <branch>
    ```

4. Create and activate your `wisdem-env` environment, substituting "wisdem-env" with a different desired name.

    ```console
    conda env create --name wisdem-env -f environment.yml
    conda activate wisdem-env
    ```

5. Install WISDEM.

    ```console
    pip install --no-deps . -v
    ```

#### Development

In order to directly use the examples in the repository and peek at the code when necessary, we recommend all users install WISDEM in *developer / editable* mode using the instructions here.  If you really just want to use WISDEM as a library and lean on the documentation, you can always do `conda install wisdem` and be done.  Note the differences between Windows and Mac/Linux build systems.

> [!IMPORTANT]
> In the `environment_dev.yaml` please uncomment out the OS-dependent dependencies at the top
>
> For Linux, we recommend using the native compilers (for example, gcc and gfortran in the default GNU suite).

Please follow steps 1-3 in the Direct Use section above, replacing steps 4 & 5 with the following to ensure the development
dependencies for building, testing, and documentation are also installed:

4. Create and activate your `wisdem-env` environment, substituting "wisdem-env" with a different desired name.

    ```console
    conda env create --name wisdem-env -f environment_dev.yml
    conda activate wisdem-env
    ```

5. Install WISDEM. Please note the `-e` (editable) flag used to ensure your code changes are registered dynamically every
   time you save modifications.

    ```console
    pip install --no-deps -e . -v
    ```

## Run Unit Tests

Each package has its own set of unit tests, and the project runs the examples as integration tests. These can be run in batch with the following command:

```console
pytest
```

Users can add either the `--unit` or `--integration` flags if they would like to skip running
the examples or just run the examples. Otherwise, all tests will be run.

> [!Note]
> Legacy users can continue to run `python test/test_all.py` to run the scipts, though it is recommend to adopt the
> simpler `pytest` call. In a future version, `test_all.py` will be removed.

## Feedback

For software issues please use <https://github.com/WISDEM/WISDEM/issues>.  For functionality and theory related questions and comments please use the NWTC forum for [Systems Engineering Software Questions](https://wind.nrel.gov/forum/wind/viewtopic.php?f=34&t=1002).
