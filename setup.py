import os
import sys
import platform
import glob
import multiprocessing
from setuptools import find_packages
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.core import setup, Extension
from io import open

# Global constants
ncpus = multiprocessing.cpu_count()
os.environ['NPY_DISTUTILS_APPEND_FLAGS'] = '1'
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

            cmake_args = ['-DBUILD_SHARED_LIBS=ON',
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
bemExt     = Extension('wisdem.ccblade._bem',
                       sources=[os.path.join('WISDEM','wisdem','ccblade','src','bem.f90')],
                       extra_compile_args=['-O2','-fPIC'])
pyframeExt = Extension('wisdem.pyframe3dd._pyframe3dd',
                       sources=glob.glob(os.path.join('WISDEM','wisdem','pyframe3dd','src','*.c')) )
precompExt = Extension('wisdem.rotorse._precomp',
                       sources=[os.path.join('WISDEM','wisdem','rotorse','PreCompPy.f90')],
                       extra_compile_args=['-O2','-fPIC'])

if platform.system() == 'Windows': # For Anaconda
    pymapArgs = ['-O1', '-m64', '-fPIC', '-std=c99','-DCMINPACK_NO_DLL']
elif sys.platform == 'cygwin':
    pymapArgs = ['-O1', '-m64', '-fPIC', '-std=c99']
elif platform.system() == 'Darwin':
    pymapArgs = ['-O1', '-m64', '-fno-omit-frame-pointer', '-fPIC']
else:
    pymapArgs = ['-O1', '-m64', '-fPIC', '-std=c99']
    
pymapExt   = Extension('wisdem.pymap._libmap',
                       sources = (glob.glob(os.path.join('WISDEM','wisdem','pymap','**','*.c'), recursive=True) +
                                  glob.glob(os.path.join('WISDEM','wisdem','pymap','**','*.cc'), recursive=True)),
                       extra_compile_args=pymapArgs,
                       include_dirs=[os.path.join('WISDEM','wisdem','include','lapack')])
            
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
wisdem_pkgs     = find_packages(where='WISDEM',exclude=['docs', 'tests', '*.test.*', 'ext'])
roscotools_pkgs = find_packages(where='ROSCO_Toolbox')
pcrunch_pkgs    = find_packages(where='pCrunch')

metadata = dict(
    name                          = 'WEIS',
    version                       = '0.0.1',
    description                   = 'Wind Energy with Integrated Servo-control',
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    author                        = 'NREL',
    url                           = 'https://github.com/WISDEM/WEIS',
    install_requires              = ['openmdao>=3.2','numpy','scipy','pandas','simpy','marmot-agents','nlopt','dill','smt'],
    classifiers                   = [_f for _f in CLASSIFIERS.split('\n') if _f],
    package_dir                   = {'wisdem':'WISDEM/wisdem',
                                     # 'ROSCO_toolbox.ROSCO_toolbox':'ROSCO_toolbox',
                                     # 'pCrunch.pCrunch':'pCrunch',
                                     }, # weis doesn't need special directions
    packages                      = weis_pkgs + wisdem_pkgs + roscotools_pkgs + pcrunch_pkgs,
    python_requires               = '>=3.6',
    license                       = 'Apache License, Version 2.0',
    ext_modules                   = [bemExt, pyframeExt, precompExt, pymapExt, roscoExt, fastExt],
    cmdclass                      = {'build_ext': CMakeBuildExt},
    zip_safe                      = False,
)

setup(**metadata)
