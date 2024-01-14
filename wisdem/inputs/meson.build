
python_sources = [
  '__init__.py',
  'gui.py',
  'schema2rst.py',
  'validation.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/inputs'
)

data_files = [
  'analysis_schema.yaml',
  'geometry_schema.yaml',
  'modeling_schema.yaml',
  ]

install_data(
  sources: data_files,
  install_dir: 'wisdem/inputs',
)
