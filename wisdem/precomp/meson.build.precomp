precomp_source = custom_target('precompmodule.c',
                            input : ['src/PreCompPy.f90'],
                            output : ['_precompmodule.c', '_precomp-f2pywrappers2.f90'],
                            command: [py3, '-m', 'numpy.f2py',
			    '@INPUT@', '-m', '_precomp', '--lower', '--build-dir', 'wisdem/precomp']
                            )

py3.extension_module('_precomp',
                     'src/PreCompPy.f90',
                     precomp_source,
                     dependencies : fortranobject_dep,
                     subdir: 'wisdem/precomp',
		     link_language: 'fortran',
                     install : true)
