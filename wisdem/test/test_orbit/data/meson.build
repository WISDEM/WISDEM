
python_sources = [
  '__init__.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/test/test_orbit/data'
)

data_files = [
  'analysis_weather.csv',
  ]

install_data(
  sources: data_files,
  install_dir: 'wisdem/test/test_orbit/data',
)

subdir('library')
