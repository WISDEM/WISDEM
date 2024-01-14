
python_sources = [
  '__init__.py',
  'test_BladeJointSizing.py',
  'test_blade_cost.py',
  'test_rail_transport.py',
  'test_rotor_power.py',
  'test_rotor_structure.py',
]

py3.install_sources(
  python_sources,
  subdir: 'wisdem/test/test_rotorse'
)

data_files = [
  'Cp_Ct_Cq.txt',
  'debug.npz',
  'nrel5mw_test.npz',
  'nrel5mw_test2.npz',
  'rail.npz',
  'regulation.npz',
  ]

install_data(
  sources: data_files,
  install_dir: 'wisdem/test/test/test_rotorse',
)



