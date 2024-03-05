
sources = ['src/py_main.c',
     'src/py_structs.h',
     'src/py_io.c',
     'src/py_io.h',
     'src/py_frame3dd.c',
     'src/py_frame3dd.h',
     'src/py_eig.c',
     'src/py_eig.h',
     'src/py_HPGmatrix.c',
     'src/py_HPGmatrix.h',
     'src/coordtrans.c',
     'src/coordtrans.h',
     'src/HPGutil.c',
     'src/HPGutil.h',
     'src/NRutil.c',
     'src/NRutil.h']

# We need to build a shared library NOT A PYTHON EXTENSION
# The ctypes wrapper handles the extension part.
# If the interface was done purely in C, then need the python extension.
shared_library('_pyframe3dd', sources,
     name_prefix: '',
     include_directories: 'src',
     dependencies : py3_dep,
     install : true)
