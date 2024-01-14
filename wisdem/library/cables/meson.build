
python_sources = [
  '__init__.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/library/cables'
)

data_files = [
  'XLPE_1000m_220kV.yaml',
  'XLPE_185mm_66kV.yaml',
  'XLPE_400mm_33kV.yaml',
  'XLPE_500mm_132kV.yaml',
  'XLPE_630mm_33kV.yaml',
  'XLPE_630mm_66kV.yaml',
  'dudgeon_array.csv',
  'dudgeon_array_no_data.csv',
  'dudgeon_custom.csv',
  'dudgeon_custom_to_update.csv',
  'dudgeon_distance_based.csv',
  ]

install_data(
  sources: data_files,
  install_dir: 'wisdem/library/cables'
)
