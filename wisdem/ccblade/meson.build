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
