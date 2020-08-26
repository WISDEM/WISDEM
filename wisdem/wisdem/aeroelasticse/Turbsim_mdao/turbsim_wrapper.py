import os
import subprocess
class Turbsim_wrapper(object):
    def __init__(self):
        self.turbsim_exe = 'turbsim'
        #self.turbsim_input = './turbsim_default.in'
        self.turbsim_input = "/Users/pgraf/work/wese/turbsim_studies/pghacking/TurbSim.inp"
        self.run_dir = '.'
        self.debug_level = 2

    def execute(self):
        exec_string = [self.turbsim_exe, self.turbsim_input]
        olddir = os.getcwd()
        os.chdir(self.run_dir)

        if self.debug_level > 0:
            print ("EXECUTING TurbSim")
            print ("Executable: \t", self.turbsim_exe )
            print ("Run directory: \t", self.run_dir)
            print ("Input file: \t", self.turbsim_input)
            print ("Exec string: \t", exec_string)

        if self.debug_level > 1:
            subprocess.call(exec_string)
        else:
            FNULL = open(os.devnull, 'w')
            subprocess.call(exec_string, stdout=FNULL, stderr=subprocess.STDOUT)

        if self.debug_level > 0:
            print ("COMPLETE TurbSim")

        os.chdir(olddir)
if __name__=='__main__':
    wrapper = Turbsim_wrapper()
    wrapper.turbsim_exe = '/Users/jquick/TurbSim/bin/TurbSim_glin64'
    wrapper.execute()
