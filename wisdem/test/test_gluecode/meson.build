
python_sources = [
  '__init__.py',
  'test_drivers.py',
  'test_gc_loadinputs.py',
  'test_gc_modified_yaml.py',
  'test_gc_yaml_floating.py',
  'test_gluecode.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/test/test/test_gluecode'
)


data_files = [
  'analysis_options_COBYLA.yaml',
  'analysis_options_GA.yaml',
  'analysis_options_LD_SLSQP.yaml',
  'analysis_options_Nelder-Mead.yaml',
  'modified_modeling_options.yaml',
  ]

py3.install_sources(
  data_files,
  subdir: 'wisdem/test/test_gluecode',
)

