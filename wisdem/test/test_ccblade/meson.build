
python_sources = [
  '__init__.py',
  'test_ccblade.py',
  'test_gradients.py',
  'test_om_gradients.py',
  'test_polar.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/test/test_ccblade'
)

data_files = [
  'smaller_dataset.npz'
  ]

install_data(
  sources: data_files,
  install_dir: 'wisdem/test/test_ccblade',
)

