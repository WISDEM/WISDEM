#!/usr/bin/env python
# encoding: utf-8

# setup.py
# only if building in place: ``python setup.py build_ext --inplace``
import os
import re
import shutil
import platform
import subprocess

import setuptools


def run_meson_build(staging_dir):
    prefix = os.path.join(os.getcwd(), staging_dir)
    purelibdir = "."

    # check if meson extra args are specified
    meson_args = ""
    if "MESON_ARGS" in os.environ:
        meson_args = os.environ["MESON_ARGS"]

    if platform.system() == "Windows":
        if not "FC" in os.environ:
            os.environ["FC"] = "gfortran"
        if not "CC" in os.environ:
            os.environ["CC"] = "gcc"

    # configure
    meson_path = shutil.which("meson")
    if meson_path is None:
        raise OSError("The meson command cannot be found on the system")

    meson_call = (
        f"{meson_path} setup {staging_dir} --prefix={prefix} "
        + f"-Dpython.purelibdir={purelibdir} -Dpython.platlibdir={purelibdir} {meson_args}"
    )
    sysargs = meson_call.split(" ")
    sysargs = [arg for arg in sysargs if arg != ""]
    print(sysargs)
    p1 = subprocess.run(sysargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.makedirs(staging_dir, exist_ok=True)
    setup_log = os.path.join(staging_dir, "setup.log")
    with open(setup_log, "wb") as f:
        f.write(p1.stdout)
    if p1.returncode != 0:
        with open(setup_log, "r") as f:
            print(f.read())
        raise OSError(sysargs, f"The meson setup command failed! Check the log at {setup_log} for more information.")

    # build
    meson_call = f"{meson_path} compile -vC {staging_dir}"
    sysargs = meson_call.split(" ")
    sysargs = [arg for arg in sysargs if arg != ""]
    print(sysargs)
    p2 = subprocess.run(sysargs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    compile_log = os.path.join(staging_dir, "compile.log")
    with open(compile_log, "wb") as f:
        f.write(p2.stdout)
    if p2.returncode != 0:
        with open(compile_log, "r") as f:
            print(f.read())
        raise OSError(
            sysargs, f"The meson compile command failed! Check the log at {compile_log} for more information."
        )


def copy_shared_libraries():
    build_path = os.path.join(staging_dir, "wisdem")
    for root, _dirs, files in os.walk(build_path):
        for file in files:
            # move wisdem libraries to just under staging_dir
            if file.endswith((".so", ".lib", ".pyd", ".pdb", ".dylib", ".dll")):
                if ".so.p" in root or ".pyd.p" in root:  # excludes intermediate object files
                    continue
                file_path = os.path.join(root, file)
                new_path = str(file_path)
                match = re.search(staging_dir, new_path)
                new_path = new_path[match.span()[1] + 1 :]
                print(f"Copying build file {file_path} -> {new_path}")
                shutil.copy(file_path, new_path)


if __name__ == "__main__":
    # This is where the meson build system will install to, it is then
    # used as the sources for setuptools
    staging_dir = "meson_build"

    # this keeps the meson build system from running more than once
    if "dist" not in str(os.path.abspath(__file__)):
        cwd = os.getcwd()
        run_meson_build(staging_dir)
        os.chdir(cwd)
        copy_shared_libraries()

    # docs_require = ""
    # req_txt = os.path.join("doc", "requirements.txt")
    # if os.path.isfile(req_txt):
    #    with open(req_txt) as f:
    #        docs_require = f.read().splitlines()

    init_file = os.path.join("wisdem", "__init__.py")
    # __version__ = re.findall(
    #    r"""__version__ = ["']+([0-9\.]*)["']+""",
    #    open(init_file).read(),
    # )[0]

    setuptools.setup(
        name="WISDEM",
        version="3.8",
        description="Wind-Plant Integrated System Design & Engineering Model",
        long_description="""WISDEM is a Python package for conducting multidisciplinary analysis and
        optimization of wind turbines and plants.  It is built on top of NASA's OpenMDAO library.""",
        url="https://github.com/WISDEM/WISDEM",
        author="NREL WISDEM Team",
        author_email="systems.engineering@nrel.gov",
        install_requires=[
            "dearpygui",
            "jsonschema",
            "marmot-agents>=0.2.5",
            "numpy",
            "openmdao>=3.18",
            "openpyxl",
            "nlopt",
            "pandas",
            "pydoe2",
            "python-benedict",
            "pyyaml",
            "ruamel.yaml",
            "scipy",
            "simpy",
            "sortedcontainers",
            "statsmodels",
        ],
        extras_require={
            "testing": ["pytest"],
        },
        python_requires=">=3.8",
        package_data={"": ["*.yaml", "*.xlsx", "*.txt", "*.so", "*.lib", "*.pyd", "*.pdb", "*.dylib", "*.dll"]},
        # package_dir      = {'': 'wisdem'},
        packages=setuptools.find_packages(exclude=["docs", "tests", "ext"]),
        license="Apache License, Version 2.0",
        entry_points={
            "console_scripts": [
                "wisdem=wisdem.main:wisdem_cmd",
                "compare_designs=wisdem.postprocessing.compare_designs:main",
            ],
        },
        zip_safe=False,
    )

# os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'

# bemExt = Extension(
#    "wisdem.ccblade._bem",
#    sources=[os.path.join("wisdem", "ccblade", "src", "bem.f90")],
#    extra_compile_args=["-O2", "-fPIC", "-std=c11"],
# )
# pyframeExt = Extension(
#    "wisdem.pyframe3dd._pyframe3dd",
#    sources=glob.glob(os.path.join("wisdem", "pyframe3dd", "src", "*.c")),
#    extra_compile_args=["-O2", "-fPIC", "-std=c11"],
# )
# precompExt = Extension(
#    "wisdem.rotorse._precomp",
#    sources=[os.path.join("wisdem", "rotorse", "PreCompPy.f90")],
#    extra_compile_args=["-O2", "-fPIC", "-std=c11"],
# )

# Top-level setup
#    ext_modules=[bemExt, pyframeExt, precompExt],
