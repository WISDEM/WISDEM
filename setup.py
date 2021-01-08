#!/usr/bin/env python

import os
import sys
import glob
import platform

from setuptools import find_packages

from numpy.distutils.core import Extension, setup

os.environ["NPY_DISTUTILS_APPEND_FLAGS"] = "1"

# CFLAGS for pyMAP
if platform.system() == "Windows":  # For Anaconda
    pymapArgs = ["-O1", "-m64", "-fPIC", "-std=c99", "-DCMINPACK_NO_DLL"]
elif sys.platform == "cygwin":
    pymapArgs = ["-O1", "-m64", "-fPIC", "-std=c99"]
elif platform.system() == "Darwin":
    pymapArgs = ["-O1", "-m64", "-fno-omit-frame-pointer", "-fPIC"]  # , '-std=c99']
else:
    # pymapArgs = ['-O1', '-m64', '-fPIC', '-std=c99', '-D WITH_LAPACK']
    pymapArgs = ["-O1", "-m64", "-fPIC", "-std=c99"]

# All the extensions
bemExt = Extension(
    "wisdem.ccblade._bem",
    sources=[os.path.join("wisdem", "ccblade", "src", "bem.f90")],
    extra_compile_args=["-O2", "-fPIC"],
)
pyframeExt = Extension(
    "wisdem.pyframe3dd._pyframe3dd", sources=glob.glob(os.path.join("wisdem", "pyframe3dd", "src", "*.c"))
)
precompExt = Extension(
    "wisdem.rotorse._precomp",
    sources=[os.path.join("wisdem", "rotorse", "PreCompPy.f90")],
    extra_compile_args=["-O2", "-fPIC"],
)
pymapExt = Extension(
    "wisdem.pymap._libmap",
    sources=glob.glob(os.path.join("wisdem", "pymap", "**", "*.c"), recursive=True)
    + glob.glob(os.path.join("wisdem", "pymap", "**", "*.cc"), recursive=True),
    extra_compile_args=pymapArgs,
    include_dirs=[os.path.join("wisdem", "include", "lapack")],
)

# Top-level setup
setup(
    name="WISDEM",
    version="3.2.0",
    description="Wind-Plant Integrated System Design & Engineering Model",
    long_description="""WISDEM is a Python package for conducting multidisciplinary analysis and
    optimization of wind turbines and plants.  It is built on top of NASA's OpenMDAO library.""",
    url="https://github.com/WISDEM/WISDEM",
    author="NREL WISDEM Team",
    author_email="systems.engineering@nrel.gov",
    install_requires=[
        "jsonschema",
        "marmot-agents",
        "numpy",
        "openmdao>=3.4",
        "openpyxl",
        "pandas",
        "pyside2",
        "pytest",
        "pyyaml",
        "scipy",
        "simpy",
        "sortedcontainers",
    ],
    python_requires=">=3.7",
    package_data={"": ["*.yaml", "*.xlsx"], "wisdem": ["*.txt"]},
    # package_dir      = {'': 'wisdem'},
    packages=find_packages(exclude=["docs", "tests", "ext"]),
    license="Apache License, Version 2.0",
    ext_modules=[bemExt, pyframeExt, precompExt, pymapExt],
    entry_points={
        "console_scripts": [
            "wisdem=wisdem.main:wisdem_cmd",
            "compare_designs=wisdem.postprocessing.compare_designs:main",
        ],
    },
    zip_safe=False,
)
