precomp_source = custom_target('precompmodule.c',
                            input : ['src/PreCompPy.f90'],
                            output : ['_precompmodule.c', '_precomp-f2pywrappers2.f90'],
                            command: [py3, '-m', 'numpy.f2py',
			    '@INPUT@', '-m', '_precomp', '--lower', '--build-dir', 'wisdem/rotorse']
                            )

py3.extension_module('_precomp',
                    'src/PreCompPy.f90',
                    precomp_source,
                    fortranobject_c,
                    include_directories: inc_np,
                    dependencies : py3_dep,
                    subdir: 'wisdem/rotorse',
		    link_language: 'fortran',
                    install : true)

python_sources = [
  '__init__.py',
  'blade_cost.py',
  'parametrize_rotor.py',
  'precomp.py',
  'rail_transport.py',
  'rotor.py',
  'rotor_elasticity.py',
  'rotor_power.py',
  'rotor_structure.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/rotorse'
)

subdir('geometry_tools')

