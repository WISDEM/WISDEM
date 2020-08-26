from weis.aeroelasticse.Turbsim_mdao.turbsim_writer import TurbsimBuilder
from weis.aeroelasticse.Turbsim_mdao.turbsim_wrapper import Turbsim_wrapper
from weis.aeroelasticse.Turbsim_mdao.turbsim_reader import turbsimReader

reader = turbsimReader()
writer = TurbsimBuilder()
wrapper = Turbsim_wrapper()

reader.read_input_file('TurbsimInputFiles/test01.inp')

writer.turbsim_vt = reader.turbsim_vt
writer.turbsim_vt.spatialcoherance.CohExp = 0.1
writer.turbulence_template = 'TurbsimInputFiles/turbulence_user.inp'
writer.run_dir = 'test'
writer.execute()

wrapper.run_dir = writer.run_dir
wrapper.turbsim_exe = '/Users/jquick/SE/TurbSim/bin/TurbSim_glin64'
wrapper.execute()
