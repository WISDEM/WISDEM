bem_source = custom_target('bemmodule.c',
                            input : ['src/bem.f90'],
                            output : ['_bemmodule.c'],
                            command: [py3, '-m', 'numpy.f2py',
			    '@INPUT@', '-m', '_bem', '--lower', '--build-dir', 'wisdem/ccblade']
                            )

py3.extension_module('_bem',
                     'src/bem.f90',
                     bem_source,
                     dependencies : fortranobject_dep,
                     subdir: 'wisdem/ccblade',
		     link_language: 'fortran',
                     install : true)


python_sources = [
  '__init__.py',
  'Polar.py',
  'ccblade.py',
  'ccblade_component.py',
  'polar_file.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/ccblade'
)
