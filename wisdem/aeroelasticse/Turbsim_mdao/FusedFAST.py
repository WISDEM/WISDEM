import os,glob,shutil, time

from openmdao.main.api import Component, Assembly, FileMetadata
from openmdao.lib.components.external_code import ExternalCode
from openmdao.main.datatypes.slot import Slot
from openmdao.main.datatypes.instance import Instance
from openmdao.main.datatypes.api import Array, Float, Str
import numpy as np


from runFAST import runFAST
from runTurbSim import runTurbSim

from FusedFASTrunCase import FASTRunCaseBuilder, FASTRunCase, FASTRunResult

from fusedwind.runSuite.runAero import openAeroCode
from fusedwind.runSuite.runCase import GenericRunCase, RunCase, RunResult, IECRunCaseBaseVT


import logging
logging.getLogger().setLevel(logging.DEBUG)


import pyts
#from pyts import api as api
#    from pyts import plot as pt
#import pyts.io.write as write
import pyts.runInput.main as ptsm
import pyts.io.input as ptsin
from pyts.base import tsGrid

from pyts.phaseModels.main import Rinker, Uniform


class openFAST(openAeroCode):
    
    code_input = Instance(FASTRunCase, iotype='in')

    def __init__(self, filedict):
        print("openFAST __init__")
        self.runfast = runFASText(filedict)
#        self.runts = runTurbSimext(filedict)
        self.runts = runTurbSimpy(filedict)

        ## this is repeated in runFASText, should consolodate
        self.basedir = os.path.join(os.getcwd(),"all_runs")
        if 'run_dir' in filedict:
            self.basedir = os.path.join(os.getcwd(),filedict['run_dir'])

        super(openFAST, self).__init__()

    def getRunCaseBuilder(self):
        return FASTRunCaseBuilder()

    def configure(self):
        print("openFAST configure", self.runfast, self.runts)
        self.add('tsrunner', self.runts)
        self.driver.workflow.add(['tsrunner'])
        self.add('runner', self.runfast)
        self.driver.workflow.add(['runner'])
        self.connect('code_input', 'runner.inputs')
        self.connect('code_input', 'tsrunner.inputs')
        self.connect('runner.outputs', 'outputs')        
        self.connect('tsrunner.tswind_file', 'runner.tswind_file')
        self.connect('tsrunner.tswind_dir', 'runner.tswind_dir')

    def execute(self):
        print("openFAST.execute(), case = ", self.inputs)
        run_case_builder = self.getRunCaseBuilder()
        dlc = self.inputs 
#        print("executing", dlc.case_name)
        self.code_input = run_case_builder.buildRunCase(dlc)
        super(openFAST, self).execute()

    def getResults(self, keys, results_dir, operation=max):
        myfast = self.runfast.rawfast        
        col = myfast.getOutputValues(keys, results_dir)
#        print("getting output for keys=", keys)
        vals = []
        for i in range(len(col)):
            c = col[i]
            try:
                val = operation(c)
            except:
                val = None
            vals.append(val)
        return vals

    def setOutput(self, output_params):
        channels = output_params['output_keys']
        if (not isinstance(channels, (list, tuple))):
            channels = [channels]                
        self.runfast.set_fast_outputs(channels)
        print("set FAST output:", output_params['output_keys'])


class runTurbSimext(Component):
    """ a component to run TurbSim.
        will try to make it aware of whether the wind file already exists"""
    
    inputs = Instance(IECRunCaseBaseVT, iotype='in')
    tswind_file = Str(iotype='out')
    tswind_dir = Str(iotype='out')

    def __init__(self, filedict):
        super(runTurbSimext,self).__init__()
        self.rawts = runTurbSim()
    
        self.rawts.ts_exe = filedict['ts_exe']
        self.rawts.ts_dir = filedict['ts_dir']
        self.rawts.ts_file = filedict['ts_file']
#        self.rawts.run_name = self.run_name

        self.basedir = os.path.join(os.getcwd(),"allts_runs")
        if 'run_dir' in filedict:
            self.basedir = os.path.join(os.getcwd(),filedict['run_dir'])
        if (not os.path.exists(self.basedir)):
            os.mkdir(self.basedir)

    def execute(self):
        case = self.inputs
        ws=case.fst_params['Vhub']
        rs = case.fst_params['RandSeed1'] if 'RandSeed1' in case.fst_params else None
        tmax = 2  ## should not be hard default ##
        if ('TMax' in case.fst_params):  ## Note, this gets set via "AnalTime" in input files--FAST peculiarity ? ##
            tmax = case.fst_params['TMax']

        # run TurbSim to generate the wind:        
        # for now, turbsim params we mess with are possibly: TMax, RandomSeed, Tmax.  These should generate
        # new runs, otherwise we should just use wind file we already have
            # for now, just differentiate by wind speed
        ts_case_name = "TurbSim-Vhub%.4f" % ws
        if rs != None:
            ts_case_name = "%s-Rseed%d" % (ts_case_name, rs)

        run_dir = os.path.join(self.basedir, ts_case_name)
        self._logger.info("running TurbSim in %s " % run_dir)
        print("running TurbSim in " , run_dir)
        self.rawts.run_dir = run_dir
        tsdict = dict({"URef": ws, "AnalysisTime":tmax, "UsableTime":tmax}.items() + case.fst_params.items())
        self.rawts.set_dict(tsdict)
        tsoutname = self.rawts.ts_file.replace("inp", "wnd")
        tsoutname = os.path.join(run_dir, tsoutname)
        tssumname = tsoutname.replace("wnd", "sum")
        reuse_run = False
        if (os.path.isfile(tsoutname) and os.path.isfile(tssumname)):
            # maybe there's an old results we can use:
            while (not reuse_run):
                ln = file(tssumname).readlines()
                if (ln != None and len(ln) > 0):
                    ln1 = ln[-1] # check last line2 lines (sometimes Turbsim inexplicably writes a final blank line!)
                    ln1 = ln1.split(".")
                    ln2 = ln[-2] # check last line2 lines (sometimes Turbsim inexplicably writes a final blank line!)
                    ln2 = ln2.split(".")
                    if ((len(ln1) > 0 and ln1[0] == "Processing complete") or (len(ln2) > 0 and ln2[0] == "Processing complete")):
                        print("re-using previous TurbSim output %s for ws = %f" % (tsoutname, ws))
                        reuse_run = True
                if (not reuse_run):
                    time.sleep(2)
                    print("waiting for ", tsoutname)
                    self._logger.info("waiting for %s" % tsoutname)
            self._logger.info("DONE waiting for %s" % tsoutname)
        
        if (not reuse_run):
            self.rawts.execute() 

        # here we link turbsim -> fast
        self.tswind_file = tsoutname
        self.tswind_dir = run_dir


class runTurbSimpy(Component):
    """ a component to run TurbSim.
        will try to make it aware of whether the wind file already exists"""
    
    inputs = Instance(IECRunCaseBaseVT, iotype='in')
    tswind_file = Str(iotype='out')
    tswind_dir = Str(iotype='out')

    def create_tsr(self, rho, U, tmax, dt):
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


    def add_phase_dist(self, tsr, rho, tmax):
#        tsr.cohere = pyts.cohereModels.main.none()
        tsr.cohere = pyts.cohereModels.main.nwtc()
        tsr.stress = pyts.stressModels.main.uniform(0,0,0)
        tsr.phase = Rinker(rho, self.mu)
        cg = tsr.grid
        tsr.grid = tsGrid(center=cg.center, ny=cg.n_y, nz=cg.n_z,
                          height=cg.height, width=cg.width,
                          time_sec=tmax, dt=cg.dt)
        
        return tsr
        
    def __init__(self, filedict):
        super(runTurbSimpy,self).__init__()

#        self.rawts.ts_exe = filedict['ts_exe']
        self.ts_dir = filedict['ts_dir']
        self.ts_file = filedict['ts_file']

        self.ustar = .8
##        self.U = 17.
        self.ny = 15
        self.nz = 15
        self.dt = 0.05  # these (should/have in past) come from TurbSim template file
        self.mid = self.nz/2

##        self.rho = 0.9999
        self.mu = np.pi

###        np.random.seed(1)  ## probably don't want this in this context

#        self.rawts.run_name = self.run_name


        self.basedir = os.path.join(os.getcwd(),"allts_runs")
        if 'run_dir' in filedict:
            self.basedir = os.path.join(os.getcwd(),filedict['run_dir'])
        if (not os.path.exists(self.basedir)):
            os.mkdir(self.basedir)

    def execute(self):
        case = self.inputs
        print("CASE", case.fst_params)
        ws=case.fst_params['Vhub']
        rho = case.fst_params['Rho']   #case.fst_params['rho'] ####### TODO: how does this get here?
        rs = case.fst_params['RandSeed1'] if 'RandSeed1' in case.fst_params else None
        tmax = 2  ## should not be hard default ##
        if ('TMax' in case.fst_params):  ## Note, this gets set via "AnalTime" in input files--FAST peculiarity ? ##
            tmax = case.fst_params['TMax']

        # run TurbSim to generate the wind:        
        # for now, turbsim params we mess with are possibly: TMax, RandomSeed, Tmax.  These should generate
        # new runs, otherwise we should just use wind file we already have
            # for now, just differentiate by wind speed
        ts_case_name = "TurbSim-Vhub%.4f" % ws
        if rs != None:
            ts_case_name = "%s-Rseed%d" % (ts_case_name, rs)

        run_dir = os.path.join(self.basedir, ts_case_name)
        if (not os.path.exists(run_dir)):
            os.mkdir(run_dir)

        self._logger.info("running TurbSim in %s " % run_dir)
        print("running TurbSim in " , run_dir)
#        self.rawts.run_dir = run_dir
        tsdict = dict({"URef": ws, "AnalysisTime":tmax, "UsableTime":tmax}.items() + case.fst_params.items())
#        self.rawts.set_dict(tsdict)
        print(case.fst_params)
        tsoutname = self.ts_file.replace("inp", "wnd")  #self.rawts.ts_file.replace("inp", "wnd")
        tsoutname = os.path.join(run_dir, tsoutname)
        tssumname = tsoutname.replace("wnd", "sum")
        reuse_run = False
        if (False and os.path.isfile(tsoutname) and os.path.isfile(tssumname)):
            # maybe there's an old results we can use:
            while (not reuse_run):
                ln = file(tssumname).readlines()
                if (ln != None and len(ln) > 0):
                    ln1 = ln[-1] # check last line2 lines (sometimes Turbsim inexplicably writes a final blank line!)
                    ln1 = ln1.split(".")
                    ln2 = ln[-2] # check last line2 lines (sometimes Turbsim inexplicably writes a final blank line!)
                    ln2 = ln2.split(".")
                    if ((len(ln1) > 0 and ln1[0] == "Processing complete") or (len(ln2) > 0 and ln2[0] == "Processing complete")):
                        print("re-using previous TurbSim output %s for ws = %f" % (tsoutname, ws))
                        reuse_run = True
                if (not reuse_run):
                    time.sleep(2)
                    print("waiting for ", tsoutname)
                    self._logger.info("waiting for %s" % tsoutname)
            self._logger.info("DONE waiting for %s" % tsoutname)
        
        if (not reuse_run):

            tsinput = ptsin.read(os.path.join(self.ts_dir, self.ts_file))
            tsr = ptsm.cfg2tsrun(tsinput)

            tsr = self.add_phase_dist(tsr, rho, tmax)
            #            tsdata=ptsm.run(tsinput)
            tsdata = tsr()  ## actually runs turbsim
            dphi_prob = tsr.phase.delta_phi_prob
            ptsm.write(tsdata, tsinput, fname=tsoutname)
            
            #fout = file("uhub.out", "w")
            #hubdat = tsdata.uturb[0, self.mid, self.mid, :]
            #for i in range(len(hubdat)):
            #    fout.write("%d  %e  %e\n" % ( i, tsdata.time[i], hubdat[i]))
            #fout.close()
            print("dphi prob is ", dphi_prob)
            fout = file(os.path.join(run_dir, "delta_phis_prob.out"), "w")
            fout.write("%e\n" % dphi_prob)
            fout.close()
            
#            tsr = self.create_tsr(rho, ws, tmax, self.dt)
#            tsdat = tsr()
#            print("writing data to ", tsdat, tsoutname)
#            tsdat.write_bladed(tsoutname)
            ### check for errors!!?? ###
            
        # here we link turbsim -> fast
        self.tswind_file = tsoutname
        self.tswind_dir = run_dir

class runFASText(Component):
    """ 
        this used to be an ExternalCode class to take advantage of file copying stuff.
        But now it relies on the global file system instead.
        it finally calls the real (openMDAO-free) FAST wrapper 
    """
    inputs = Instance(IECRunCaseBaseVT, iotype='in')
#    input = Instance(GenericRunCase, iotype='in')
    outputs = Instance(RunResult, iotype='out')  ## never used, never even set
    tswind_file = Str(iotype='in')
    tswind_dir = Str(iotype='in')

    ## just a template, meant to be reset by caller
    fast_outputs = ['WindVxi','RotSpeed', 'RotPwr', 'GenPwr', 'RootMxc1', 'RootMyc1', 'LSSGagMya', 'LSSGagMza', 'YawBrMxp', 'YawBrMyp','TwrBsMxt',
                    'TwrBsMyt', 'Fair1Ten', 'Fair2Ten', 'Fair3Ten', 'Anch1Ten', 'Anch2Ten', 'Anch3Ten'] 

    def __init__(self, filedict):
        super(runFASText,self).__init__()
        self.rawfast = runFAST()

        print("runFASText init(), filedict = ", filedict)

        # probably overridden by caller
        self.rawfast.setOutputs(self.fast_outputs)

        # if True, results will be copied back to basedir+casename.
        # In context of global file system, this is not necessary.  Instead, leave False and postprocess directly from run_dirs.
        self.copyback_files = False
 
        have_tags = all([tag in filedict for tag in ["fst_exe", "fst_dir", "fst_file", "ts_exe", "ts_dir", "ts_file"]])
        if (not have_tags):
            print("Failed to provide all necessary files/paths: fst_exe, fst_dir, fst_file, ts_exe, ts_dir, ts_file  needed to run FAST")
            raise ValueError("Failed to provide all necessary files/paths: fst_exe, fst_dir, fst_file, ts_exe, ts_dir, ts_file  needed to run FAST")

        self.rawfast.fst_exe = filedict['fst_exe']
        self.rawfast.fst_dir = filedict['fst_dir']
        self.rawfast.fst_file = filedict['fst_file']
        self.run_name = self.rawfast.fst_file.split(".")[0]
        self.rawfast.run_name = self.run_name

        self.basedir = os.path.join(os.getcwd(),"all_runs")
        if 'run_dir' in filedict:
            self.basedir = os.path.join(os.getcwd(),filedict['run_dir'])
        if (not os.path.exists(self.basedir)):
            os.mkdir(self.basedir)

    def set_fast_outputs(self,fst_out):
        self.fast_outputs = fst_out
        self.rawfast.setOutputs(self.fast_outputs)
                
    def execute(self):
        case = self.inputs

        ws=case.fst_params['Vhub']
        tmax = 2  ## should not be hard default ##
        if ('TMax' in case.fst_params):  ## Note, this gets set via "AnalTime" in input files--FAST peculiarity ? ##
            tmax = case.fst_params['TMax']

        # TurbSim has already been run to generate the wind, it's output is
        # connected as tswind_file
        self.rawfast.set_wind_file(self.tswind_file)

        run_dir = os.path.join(self.basedir, case.case_name)
        print("running FASTFASTFAST in " , run_dir, case.case_name)

        ### actually execute FAST (!!) 
        print("RUNNING FAST WITH RUN_DIR", run_dir)
        self.rawfast.run_dir = run_dir
        self.rawfast.set_dict(case.fst_params)
        # FAST object write its inputs in execute()
        self.rawfast.execute()
        ###

        # gather output directly
        self.output = FASTRunResult(self)


        ### special hack for getting phase difference probability file back in directory ultimate caller
        ### will know about:
        probfile = os.path.join(self.tswind_dir, "delta_phis_prob.out")
        if os.path.isfile(probfile):
            shutil.copy(probfile, os.path.join(run_dir, "delta_phis_prob.out"))
        ###
            
        # also, copy all the output and input back "home"
        if (self.copyback_files):
            self.results_dir = os.path.join(self.basedir, case.case_name)
            try:
                os.mkdir(self.results_dir)
            except:
                # print('error creating directory', results_dir)
                # print('it probably already exists, so no problem')
                pass

            # Is this supposed to do what we're doing by hand here?
            # self.copy_results_dirs(results_dir, '', overwrite=True)

            files = glob.glob( "%s.*" % os.path.join(self.rawfast.run_dir, self.rawfast.run_name))
            files += glob.glob( "%s.*" % os.path.join(self.rawts.run_dir, self.rawts.run_name))
            
            for filename in files:
#                print("wanting to copy %s to %s" % (filename, results_dir) ## for debugging, don't clobber stuff you care about!)
                shutil.copy(filename, self.results_dir)



class designFAST(openFAST):        
    """ base class for cases where we have parametric design (e.g. dakota),
    corresponding to a driver that are for use within a Driver that "has_parameters" """
    x = Array(iotype='in')   ## exact size of this gets filled in study.setup_cases(), which call create_x, below
    f = Float(iotype='out')
    # need some mapping back and forth
    param_names = []

    def __init__(self,geom,atm,filedict):
        super(designFAST, self).__init__(geom,atm,filedict)

    def create_x(self, size):
        """ just needs to exist and be right size to use has_parameters stuff """
        self.x = [0 for i in range(size)]

    def dlc_from_params(self,x):
        print(x, self.param_names, self.dlc.case_name)
        case = FASTRunCaseBuilder.buildRunCase_x(x, self.param_names, self.dlc)
        print(case.fst_params)
        return case

    def execute(self):
        # build DLC from x, if we're using it
        print("in design code. execute()", self.x)
        self.inputs = self.dlc_from_params(self.x)
        super(designFAST, self).execute()
        myfast = self.runfast.rawfast
        self.f = myfast.getMaxOutputValue('TwrBsMxt', directory=os.getcwd())



def designFAST_test():
    w = designFAST()

    ## sort of hacks to save this info
    w.param_names = ['Vhub']
    w.dlc = FASTRunCase("runAero-testcase", {}, None)
    print("set aerocode dlc")
    ##

    res = []
    for x in range(10,16,2):
        w.x = [x]
        w.execute()
        res.append([ w.dlc.case_name, w.param_names, w.x, w.f])
    for r in res:
        print(r)


def openFAST_test():
    # in real life these come from an input file:
    filedict = {'ts_exe' : "/Users/pgraf/opt/windcode-7.31.13/TurbSim/build/TurbSim_glin64",
                'ts_dir' : "/Users/pgraf/work/wese/fatigue12-13/from_gordie/SparFAST3.orig/TurbSim",
                'ts_file' : "TurbSim.inp",
                'fst_exe' : "/Users/pgraf/opt/windcode-7.31.13/build/FAST_glin64",
                'fst_dir' : "/Users/pgraf/work/wese/fatigue12-13/from_gordie/SparFAST3.orig",
                'fst_file' : "NRELOffshrBsline5MW_Floating_OC3Hywind.fst",
                'run_dir' : "run_dir"}

    w = openFAST(filedict)
    tmax = 5
    res = []

    case = 2
    if (case == 1):
        for x in [10,16,20]:
            dlc = GenericRunCase("runAero-testcase%d" % x, ['Vhub','AnalTime'], [x,tmax])
            w.inputs = dlc
            w.execute()
    elif case == 2:
        res = []
        vhub = 20
#        xs=[.10,.4,1.0]   # radians!, wave angle
#        xs=[10,20,30]   # m/s!, vhub
        xs = [0,30,60, 90]  # degrees, platform angle
        for x in xs:
            dlc = GenericRunCase("runAero-testcase%d" % x, ['PlatformDir','AnalTime', 'Vhub'], [x,tmax,vhub])
            w.inputs = dlc
            w.execute()
            results_dir = os.path.join(filedict['run_dir'],dlc.case_name)
            print("name", results_dir)
            rr = w.getResults(["RotPwr", "TwrBsMxt"], results_dir, operation=np.std)
            res.append(rr)
        for i in range(len(xs)):
            print(xs[i], res[i])
    elif case == 3:
        pass


if __name__=="__main__":
    openFAST_test()
#    designFAST_test()
