
python_sources = [
  '__init__.py',
  'compare_designs.py',
  'wisdem_get.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/postprocessing'
)

data_files = [
  'default_analysis_options.yaml',
  'default_modeling_options.yaml',
  ]

install_data(
  sources: data_files,
  install_dir: 'wisdem/postprocessing',
)
