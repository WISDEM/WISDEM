
sources = [
  'src/common.h',
  'src/py_structs.h',
  'src/HPGutil.h',
  'src/HPGutil.c',
  'src/coordtrans.h',
  'src/coordtrans.c',
  'src/NRutil.h',
  'src/NRutil.c',
  'src/py_io.h',
  'src/py_io.c',
  'src/py_eig.h',
  'src/py_eig.c',
  'src/py_HPGmatrix.h',
  'src/py_HPGmatrix.c',
  'src/py_frame3dd.h',
  'src/py_frame3dd.c',
  'src/py_main.c',
]

# We need to build a shared library NOT A PYTHON EXTENSION
# The ctypes wrapper handles the extension part.
# If the interface was done purely in C, then need the python extension.
shared_library('_pyframe3dd',
               sources,
               name_prefix: '',
               include_directories: ['src','src/microstran'],
               install_dir: py3.get_install_dir() / 'wisdem/pyframe3dd',
               install : true)


python_sources = [
  '__init__.py',
  'pyframe3dd.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/pyframe3dd'
)
