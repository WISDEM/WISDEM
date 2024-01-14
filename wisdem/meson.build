# NumPy include directory - needed in all submodules
incdir_numpy = get_option('incdir_numpy')
if incdir_numpy == ''
    incdir_numpy = run_command(py3,
      [
        '-c',
        'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
      ],
      check: true
    ).stdout().strip()
endif
# this creates a raw string which is useful for Windows use of '\' for paths
incdir_numpy = '''@0@'''.format(incdir_numpy)

#incdir_f2py = run_command(py3,
#    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
#    check : true
#).stdout().strip()

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

inc_np = include_directories(incdir_numpy, incdir_f2py)

python_sources = [
  '__init__.py',
  'main.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem'
)

subdir('orbit/phases/install/$file')
subdir('ccblade')
subdir('commonse')
subdir('drivetrainse')
subdir('fixed_bottomse')
subdir('floatingse')
subdir('glue_code')
subdir('h2')
subdir('inputs')
subdir('landbosse')
subdir('library')
subdir('nrelcsm')
subdir('optimization_drivers')
subdir('orbit')
subdir('plant_financese')
subdir('postprocessing')
subdir('pyframe3dd')
subdir('rotorse')
subdir('test')
subdir('towerse')
__init__.py
__pycache__
orbit/phases/install/$file
ccblade
commonse
drivetrainse
fixed_bottomse
floatingse
glue_code
h2
inputs
landbosse
library
main.py
meson.build
nrelcsm
optimization_drivers
orbit
plant_financese
postprocessing
pyframe3dd
rotorse
test
towerse
__init__.py
__pycache__
airfoilprep
ccblade
commonse
drivetrainse
fixed_bottomse
floatingse
glue_code
h2
inputs
landbosse
library
main.py
meson.build
meson.build-s
nrelcsm
optimization_drivers
orbit
plant_financese
postprocessing
pyframe3dd
rotorse
test
towerse
