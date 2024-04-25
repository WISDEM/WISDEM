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

#######
# This forces wheels to be platform specific
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""
    def has_ext_modules(foo):
        return True
#######
    
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

    meson_call = [meson_path, "setup", staging_dir, "--wipe",
                  f"--prefix={prefix}", f"-Dpython.purelibdir={purelibdir}",
                  f"-Dpython.platlibdir={purelibdir}"] + meson_args.split()
    meson_call = [m for m in meson_call if m.strip() != ""]
    print(meson_call)
    p1 = subprocess.run(meson_call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.makedirs(staging_dir, exist_ok=True)
    setup_log = os.path.join(staging_dir, "setup.log")
    with open(setup_log, "wb") as f:
        f.write(p1.stdout)
    if p1.returncode != 0:
        with open(setup_log, "r") as f:
            print(f.read())
        raise OSError(meson_call, f"The meson setup command failed! Check the log at {setup_log} for more information.")

    # build
    meson_call = [meson_path, "compile", "-vC", staging_dir]
    meson_call = [m for m in meson_call if m != ""]
    print(meson_call)
    p2 = subprocess.run(meson_call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    compile_log = os.path.join(staging_dir, "compile.log")
    with open(compile_log, "wb") as f:
        f.write(p2.stdout)
    if p2.returncode != 0:
        with open(compile_log, "r") as f:
            print(f.read())
        raise OSError(meson_call, f"The meson compile command failed! Check the log at {compile_log} for more information.")


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

    #init_file = os.path.join("wisdem", "__init__.py")
    # __version__ = re.findall(
    #    r"""__version__ = ["']+([0-9\.]*)["']+""",
    #    open(init_file).read(),
    # )[0]

    setuptools.setup(cmdclass={'bdist_wheel': bdist_wheel}, distclass=BinaryDistribution)

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
