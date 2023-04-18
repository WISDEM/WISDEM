# Much of this is from SciPy

project(
  'wisdem',
  'c',
# unnecessary metadata commented out until Meson supports PEP517 and installation with pip
  license: 'Apache',
  meson_version: '>= 0.60',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c11',
  ],
)

cc = meson.get_compiler('c')
add_languages('fortran', native: false)
fc = meson.get_compiler('fortran')
is_windows = host_machine.system() == 'windows'

# We need -lm for all C code (assuming it uses math functions, which is safe to
# assume for SciPy). For C++ it isn't needed, because libstdc++/libc++ is
# guaranteed to depend on it. For Fortran code, Meson already adds `-lm`.
m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : 'c')
endif

# https://mesonbuild.com/Python-module.html
# Here we differentiate from the python used by meson, py3_command, and that python target, py3_target. This is useful
# when cross compiling like on conda-forge
py_mod = import('python')
if get_option('python_target') != ''
  py3 = py_mod.find_installation(get_option('python_target'))
else
  py3 = py_mod.find_installation('python')
endif
py3_dep = py3.dependency()

message(py3.path())
message(py3.get_install_dir())

subdir('wisdem')
