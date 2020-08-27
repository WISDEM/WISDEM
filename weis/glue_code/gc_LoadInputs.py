import numpy as np
import os
import wisdem.schema as sch
from weis.aeroelasticse.FAST_reader import InputReader_OpenFAST
from wisdem.glue_code.gc_LoadInputs import WindTurbineOntologyPython


class WindTurbineOntologyPythonWEIS(WindTurbineOntologyPython):
    # Pure python class inheriting the class WindTurbineOntologyPython from WISDEM and adding the WEIS options, namely the paths to the WEIS submodules (OpenFAST, ROSCO, TurbSim, XFoil) and initializing the control parameters.
    
    def __init__(self, fname_input_wt, fname_input_modeling, fname_input_analysis):

        self.modeling_options = sch.load_modeling_yaml(fname_input_modeling)
        self.modeling_options['fname_input_modeling'] = fname_input_modeling
        self.wt_init          = sch.load_geometry_yaml(fname_input_wt)
        self.analysis_options = sch.load_analysis_yaml(fname_input_analysis)
        self.defaults         = sch.load_default_geometry_yaml()
        self.set_run_flags()
        self.set_openmdao_vectors()
        self.set_openmdao_vectors_control()
        self.set_openfast_data()
        self.set_opt_flags()
    
    def set_openfast_data(self):
        # Openfast
        if self.modeling_options['Analysis_Flags']['OpenFAST'] == True:
            # Load Input OpenFAST model variable values
            fast                = InputReader_OpenFAST(FAST_ver=self.modeling_options['openfast']['file_management']['FAST_ver'])
            fast.FAST_InputFile = self.modeling_options['openfast']['file_management']['FAST_InputFile']
            if os.path.isabs(self.modeling_options['openfast']['file_management']['FAST_directory']):
                fast.FAST_directory = self.modeling_options['openfast']['file_management']['FAST_directory']
            else:
                fast.FAST_directory = os.path.join(os.path.dirname(self.modeling_options['fname_input_modeling']), self.modeling_options['openfast']['file_management']['FAST_directory'])
            if os.path.isabs(self.modeling_options['openfast']['file_management']['path2dll']):
                fast.path2dll = self.modeling_options['openfast']['file_management']['path2dll']
            else:
                fast.path2dll = os.path.join(os.path.dirname(self.modeling_options['fname_input_modeling']), self.modeling_options['openfast']['file_management']['path2dll'])
            fast.execute()
            self.modeling_options['openfast']['fst_vt']   = fast.fst_vt

            if os.path.isabs(self.modeling_options['openfast']['file_management']['Simulation_Settings_File']):
                path2settings = self.modeling_options['openfast']['file_management']['Simulation_Settings_File']
            else:
                path2settings = os.path.join(os.path.dirname(self.modeling_options['fname_input_modeling']), self.modeling_options['openfast']['file_management']['Simulation_Settings_File'])
            if os.path.exists(path2settings):
                self.modeling_options['openfast']['fst_settings'] = dict(sch.load_yaml(path2settings))
            else:
                print('WARNING: OpenFAST is called, but no file with settings is found.')
                self.modeling_options['openfast']['fst_settings'] = {}

            if os.path.isabs(self.modeling_options['xfoil']['path']):
                self.modeling_options['airfoils']['xfoil_path']   = self.modeling_options['xfoil']['path']
            else:
                self.modeling_options['airfoils']['xfoil_path'] = os.path.join(os.path.dirname(self.modeling_options['fname_input_modeling']), self.modeling_options['xfoil']['path'])
            if self.modeling_options['openfast']['analysis_settings']['Analysis_Level'] == 2 and self.modeling_options['openfast']['dlc_settings']['run_power_curve'] == False and self.modeling_options['openfast']['dlc_settings']['run_IEC'] == False:
                raise ValueError('WEIS is set to run OpenFAST, but both flags for power curve and IEC cases are set to False among the modeling options. Set at least one of the two to True to proceed.')
        else:
            self.modeling_options['openfast']['fst_vt']   = {}

            
    def set_openmdao_vectors_control(self):
        # Distributed aerodynamic control devices along blade
        self.modeling_options['blade']['n_te_flaps']      = 0
        if 'aerodynamic_control' in self.wt_init['components']['blade']:
            if 'te_flaps' in self.wt_init['components']['blade']['aerodynamic_control']:
                self.modeling_options['blade']['n_te_flaps'] = len(self.wt_init['components']['blade']['aerodynamic_control']['te_flaps'])
                self.modeling_options['airfoils']['n_tab']   = 3
            else:
                exit('A distributed aerodynamic control device is provided in the yaml input file, but not supported by wisdem.')
        
    def update_ontology_control(self, wt_opt):
        # Update controller
        if self.modeling_options['flags']['control']:
            self.wt_init['control']['tsr']      = float(wt_opt['pc.tsr_opt'])
            self.wt_init['control']['PC_omega'] = float(wt_opt['tune_rosco_ivc.PC_omega'])
            self.wt_init['control']['PC_zeta']  = float(wt_opt['tune_rosco_ivc.PC_zeta'])
            self.wt_init['control']['VS_omega'] = float(wt_opt['tune_rosco_ivc.VS_omega'])
            self.wt_init['control']['VS_zeta']  = float(wt_opt['tune_rosco_ivc.VS_zeta'])
            self.wt_init['control']['Flp_omega']= float(wt_opt['tune_rosco_ivc.Flp_omega'])
            self.wt_init['control']['Flp_zeta'] = float(wt_opt['tune_rosco_ivc.Flp_zeta'])
