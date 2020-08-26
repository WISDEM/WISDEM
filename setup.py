import os
import sys
import platform
import multiprocessing
from distutils.core import run_setup
from setuptools import find_packages
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.core import setup, Extension
from io import open

# Global constants
ncpus = multiprocessing.cpu_count()
this_directory = os.path.abspath(os.path.dirname(__file__))

# For the CMake Extensions
class CMakeExtension(Extension):

    def __init__(self, name, sourcedir='', **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuildExt(build_ext):
    
    def copy_extensions_to_source(self):
        newext = []
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension): continue
            newext.append( ext )
        self.extensions = newext
        super().copy_extensions_to_source()
    
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            # Ensure that CMake is present and working
            try:
                self.spawn(['cmake', '--version'])
            except OSError:
                raise RuntimeError('Cannot find CMake executable')

            localdir = os.path.join(this_directory, 'local')

            cmake_args = ['-DBUILD_SHARED_LIBS=OFF',
                          '-DCMAKE_INSTALL_PREFIX=' + localdir]
            
            if platform.system() == 'Windows':
                cmake_args += ['-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE']
                
                if self.compiler.compiler_type == 'msvc':
                    cmake_args += ['-DCMAKE_GENERATOR_PLATFORM=x64']
                else:
                    cmake_args += ['-G', 'MinGW Makefiles']

            self.build_temp += '_'+ext.name
            os.makedirs(localdir, exist_ok=True)
            # Need fresh build directory for CMake
            os.makedirs(self.build_temp, exist_ok=True)

            self.spawn(['cmake', '-S', ext.sourcedir, '-B', self.build_temp] + cmake_args)
            self.spawn(['cmake', '--build', self.build_temp, '-j', str(ncpus), '--target', 'install', '--config', 'Release'])

        else:
            super().build_extension(ext)


# All of the extensions
fastExt    = CMakeExtension('openfast','OpenFAST')
roscoExt   = CMakeExtension('rosco','ROSCO')
            
# Setup content
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

CLASSIFIERS = '''
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
'''

weis_pkgs       = find_packages()

# Install the python sub-packages
print(sys.argv)
for pkg in ['WISDEM','ROSCO_toolbox','pCrunch']:
    os.chdir(pkg)
    run_setup('setup.py', script_args=sys.argv[1:], stop_after='run')
    # subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])  # This option runs `pip install -e .` on each package
    os.chdir('..')

# Now install WEIS and the Fortran packages
metadata = dict(
    name                          = 'WEIS',
    version                       = '0.0.1',
    description                   = 'Wind Energy with Integrated Servo-control',
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    author                        = 'NREL',
    url                           = 'https://github.com/WISDEM/WEIS',
    install_requires              = ['openmdao>=3.2','numpy','scipy','nlopt','dill','smt'],
    classifiers                   = [_f for _f in CLASSIFIERS.split('\n') if _f],
    packages                      = weis_pkgs,
    python_requires               = '>=3.6',
    license                       = 'Apache License, Version 2.0',
    ext_modules                   = [roscoExt, fastExt],
    cmdclass                      = {'build_ext': CMakeBuildExt},
    zip_safe                      = False,
)

setup(**metadata)
