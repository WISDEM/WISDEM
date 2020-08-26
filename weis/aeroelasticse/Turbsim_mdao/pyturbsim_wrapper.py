import numpy as np
import os

import pyts
import pyts.runInput.main as ptsm
import pyts.io.input as ptsin
from pyts.base import tsGrid
from pyts.phaseModels.main import Rinker, Uniform


from weis.aeroelasticse.CaseGen_General import CaseGen_General, save_case_matrix

# class pyTurbsim_wrapper():
#     """ wrapper for py turbsim so we can mostly just plug it in """
#     def __init__(self):
#         import pyts
#         import pyts.runInput.main as ptsm
#         import pyts.io.input as ptsin
#         from pyts.base import tsGrid
#         from pyts.phaseModels.main import Rinker, Uniform

#         pass

#     def execute(self, IEC_WindType, U, rho):  ### Note rho is a new param.  maybe a dict as input would be better,
#         ### but that would mean changing pyIECWind, or at least adding a new function like "execute_dict()"
#         pass


class pyTurbsim_wrapper():
    """ a component to run TurbSim.
        will try to make it aware of whether the wind file already exists"""

    def create_tsr(self, rho, U, tmax, dt):
        ######## NOT CALLED ##############
        tsr = api.tsrun()
        tsr.grid = api.tsGrid(center=10, ny=self.ny, nz=self.nz,
                              height=10, width=10, time_sec=tmax, dt=dt)
        tsr.prof = api.profModels.pl(U, 90)

        tsr.spec = api.specModels.tidal(self.ustar, 10)

        #tsr.cohere = api.cohereModels.nwtc()
        tsr.cohere = pyts.cohereModels.main.none()

        tsr.stress = api.stressModels.uniform(0.0, 0.0, 0.0)

        tsr.cohere = pyts.cohereModels.main.none()
        tsr.stress = pyts.stressModels.main.uniform(0,0,0)

        from pyts.phaseModels.main import Rinker, Uniform
        tsr.phase = Rinker(rho, self.mu)
        #tsr.phase = Uniform()
        # tsr.stress=np.zeros(tsr.grid.shape,dtype='float32')

        self.tsr = tsr
        return tsr
        #######################################


    def __init__(self, filedict, case, case_name):

        self.overwrite = True

        self.case = case
        self.case_name = case_name

#        self.rawts.ts_exe = filedict['ts_exe']
        self.ts_dir = filedict['ts_dir']
        self.ts_file = filedict['ts_file']

#        self.ustar = .8
##        self.U = 17.
#        self.ny = 15
#        self.nz = 15
#        self.dt = 0.05  # these (should/have in past) come from TurbSim template file
#        self.mid = self.nz/2

##        self.rho = 0.9999
        self.mu = np.pi

###        np.random.seed(1)  ## probably don't want this in this context

#        self.rawts.run_name = self.run_name

        try:
            self.basedir = os.path.join(os.getcwd(),"allts_runs")
        except:
            pass
        
        if 'run_dir' in filedict:
            try:
                self.basedir = os.path.join(os.getcwd(),filedict['run_dir'])
            except:
                self.basedir = filedict['run_dir']
        if (not os.path.exists(self.basedir)):
            try:
                os.mkdir(self.basedir)
            except:
                pass
                # print("ok, %s exists after all" % self.basedir)

    def add_phase_dist(self, tsr, rho, tmax):
        tsr.cohere = pyts.cohereModels.main.none()
        #tsr.cohere = pyts.cohereModels.main.nwtc()
        #tsr.stress = pyts.stressModels.main.uniform(0,0,0)
    ####    tsr.phase = Rinker(rho, self.mu)  ### pgraf turned it off for testing!
    ####    cg = tsr.grid
    ###    tsr.grid = tsGrid(center=cg.center, ny=cg.n_y, nz=cg.n_z,
    ###                      height=cg.height, width=cg.width,
    ###                      time_sec=tmax, dt=cg.dt)

        return tsr

    def execute(self):
        case = self.case
        case_name = self.case_name
        # print("CASE", case, case_name)
        ws=case['Vhub']
        rho = case['Rho']   #case.fst_params['rho'] ####### TODO: how does this get here?
        rs = case['RandSeed1'] if 'RandSeed1' in case else None
        tmax = 2  ## should not be hard default ##
        if ('TMax' in case):  ## Note, this gets set via "AnalTime" in input files--FAST peculiarity ? ##
            tmax = case['TMax']

        # run TurbSim to generate the wind:
        run_dir = os.path.join(self.basedir, case_name)
        if (not os.path.exists(run_dir)):
            try:
                os.mkdir(run_dir)
            except:
                print("%s exists after all" % run_dir)

#        tsdict = dict({"URef": ws, "AnalysisTime":tmax, "UsableTime":tmax}.items() + case.items())
        tsdict = dict({"URef": ws,  "UsableTime":tmax}.items() + case.items())
#        tsoutname = self.ts_file.replace("inp", "wnd")  #self.rawts.ts_file.replace("inp", "wnd")
        # tsoutname = self.ts_file.replace("inp", "bts")  #self.rawts.ts_file.replace("inp", "wnd")
        tsoutname = case_name + '.bts'
        tsoutname = os.path.join(run_dir, tsoutname)
        tssumname = case_name + '.sum'
        # tssumname = self.ts_file.replace("inp", "sum")
        # print("running TurbSim in dir for case:" , run_dir, case, tsdict)

        if self.overwrite or (not self.overwrite and not os.path.exists(tsoutname)):

            tsinput = ptsin.read(os.path.join(self.ts_dir, self.ts_file))
            for key in tsdict:
                tsinput[key] = tsdict[key]
            tsr = ptsm.cfg2tsrun(tsinput)

            ### the random seed:
            ### bug somewhere in complicated pyts use of numpy (only when called inside multiprocessing)
            ### Success via cutting out the middle man!:
            if rs is None:
                tsr.randgen.seed(tsr.RandSeed)
            else:
                tsr.randgen.seed(rs)  ## this does nothing!
                np.random.seed(rs)  ### this does the trick!

            ###tsr = self.add_phase_dist(tsr, rho, tmax)
            tsr.cohere = pyts.cohereModels.main.nwtc()
            tsr.stress = pyts.stressModels.main.uniform(0,0,0)
            tsr.phase = Rinker(rho, np.pi)
            cg = tsr.grid
            tsr.grid = tsGrid(center=cg.center, ny=cg.n_y, nz=cg.n_z,
                         height=cg.height, width=cg.width,
                         time_sec=tmax, dt=cg.dt)

            tsdata = tsr()  ## actually runs turbsim
            ptsm.write(tsdata, tsinput, fname=tsoutname)

        # here we provide the means to link turbsim to fast:
        self.tswind_file = tsoutname
        self.tswind_dir = run_dir

if __name__ == "__main__":

    case_inputs = {}
    case_inputs[("TMax")] = {'vals':[10.], 'group':0}
    case_inputs[("Vhub")] = {'vals':[10., 11., 12.], 'group':1}
    case_inputs[("Rho")] = {'vals':[.2, .25, .3], 'group':1}
    case_inputs[("RandSeed1")] = {'vals':[123,234], 'group':2}
    case_list, case_names = CaseGen_General(case_inputs, dir_matrix='', namebase='pyTurbsim_testing')

    filedict = {
        'ts_dir':"/Users/pgraf/work/wese/templates/turbsim/xloads/",
        'ts_file':"TurbSim.inp",
        'run_dir':"test_run_dir"}

    for idx in range(len(case_list)):
        case = case_list[idx]
        case_name = case_names[idx]
        pyturb = pyTurbsim_wrapper(filedict, case, case_name) # initialize runner with case variable inputs
        pyturb.ny = 20 # example of changing an attribute
        pyturb.execute() # run

        case['tswind_file'] = pyturb.tswind_file
        case['tswind_dir'] = pyturb.tswind_dir
        print("SUCCESS ")
        print("   ", case)
