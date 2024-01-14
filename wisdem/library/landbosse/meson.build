
python_sources = [
  '__init__.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/library/landbosse'
)

data_files = [
  'Vestas_V47_public.xlsx',
  'clipper_25_public.xlsx',
  'ge15_expected_validation.xlsx',
  'ge15_public.xlsx',
  'ge15_public_dist.xlsx',
  'iea36_120_project_data.xlsx',
  'iea36_85_project_data.xlsx',
  'northwind_100_public.xlsx',
  ]

install_data(
  sources: data_files,
  install_dir: 'wisdem/library/landbosse'
)
