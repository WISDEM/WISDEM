from weis.aeroelasticse.Turbsim_mdao.turbsim_vartrees import turbsiminputs
from weis.aeroelasticse.Turbsim_mdao.turbulence_spectrum import turb_specs
from weis.aeroelasticse.Turbsim_mdao.wind_profile_writer import write_wind
import os
import numpy as np
import random
from time import sleep
class TurbsimBuilder(turbsiminputs):
    def __init__(self):
         self.turbsim_vt = turbsiminputs()
         self.tsim_input_file = 'turbsim_default.in'
         self.tsim_turbulence_file = 'turbulence_default.in'
         self.tsim_profile_file = 'default.shear'
 
         # Turbulence file parameters
         self.wind_speed = 8.
         self.L_u = 2.54e+02
         self.L_v=1.635e+02
         self.L_w=4.7e+01
         self.sigma_u=1.325
         self.sigma_v=0.9
         self.sigma_w=0.7625
         self.turbulence_file_name = 'tsim_user_turbulence_default.inp'
         self.turbulence_template_file = 'TurbsimInputFiles/turbulence_user.inp'
         
         # profile file parameters
         self.profile_template = 'TurbsimInputFiles/shear.profile'
         self.shear_exponent = 0.7
         self.veer = 12.5
         self.turbsim_vt.metboundconds.ProfileFile = 'default.profile'

         self.run_dir = 'run%d'%np.random.uniform(0,1e10)

    def execute(self, write_specs=False, write_profile=False):
         if not os.path.exists(self.run_dir): 
            try:
               sleep(random.uniform(0, 1))
               if not os.path.exists(self.run_dir): 
                  os.makedirs(self.run_dir)
            except:
               pass

         # Write turbulence file
         if write_specs:
            self.turbsim_vt.metboundconds.UserFile = os.sep.join([self.run_dir, self.turbulence_file_name])
            turb_specs(V_ref=float(self.wind_speed), L_u=float(self.L_u), L_v=float(self.L_v), L_w=float(self.L_w), sigma_u=float(self.sigma_u), sigma_v=float(self.sigma_v), sigma_w=float(self.sigma_w), filename=self.turbsim_vt.metboundconds.UserFile, template_file=self.turbulence_template_file)
            self.turbsim_vt.metboundconds.UserFile = os.sep.join(['..', self.run_dir, self.turbulence_file_name])

         # Write profile file
         if write_profile:
            self.turbsim_vt.metboundconds.ProfileFile = os.sep.join([self.run_dir, self.turbsim_vt.metboundconds.ProfileFile])
            write_wind(V_ref=float(self.wind_speed), alpha=float(self.shear_exponent), Beta=float(self.veer), Z_hub=float(self.turbsim_vt.tmspecs.HubHt), filename=self.turbsim_vt.metboundconds.ProfileFile, template_file=self.profile_template)
            self.turbsim_vt.metboundconds.ProfileFile = os.sep.join(['..', self.turbsim_vt.metboundconds.ProfileFile])
            self.turbsim_vt.metboundconds.ProfileFile = os.sep.join(['..', self.run_dir, self.turbsim_vt.metboundconds.ProfileFile])

         tsinp = open(os.sep.join([self.run_dir, self.tsim_input_file]), 'w')
         tsinp.write("---------TurbSim v2.00.* Input File------------------------\n")
         tsinp.write(" Turbsim input file for {}\n".format(self.turbulence_file_name))
         tsinp.write("---------Runtime Options-----------------------------------\n")

         # runtime options
         tsinp.write('{!s:<12}   Echo            - Echo input data to <RootName>.ech (flag)\n'.format(self.turbsim_vt.runtime_options.echo))
         tsinp.write('{!s:<12}   RandSeed1       - First random seed  (-2147483648 to 2147483647)\n'.format(int(self.turbsim_vt.runtime_options.RandSeed1)))
         tsinp.write('{!s:<12}   RandSeed2       - Second random seed (-2147483648 to 2147483647) for intrinsic pRNG, or an alternative pRNG: "RanLux" or "RNSNLW"\n'.format(self.turbsim_vt.runtime_options.RandSeed2))
         tsinp.write('{!s:<12}   WrBHHTP         - Output hub-height turbulence parameters in binary form?  (Generates RootName.bin)\n'.format(self.turbsim_vt.runtime_options.WrBHHTP))
         tsinp.write('{!s:<12}   WrFHHTP         - Output hub-height turbulence parameters in formatted form?  (Generates RootName.dat)\n'.format(self.turbsim_vt.runtime_options.WrFHHTP))
         tsinp.write('{!s:<12}   WrADHH          - Output hub-height time-series data in AeroDyn form?  (Generates RootName.hh)\n'.format(self.turbsim_vt.runtime_options.WrADHH))
         tsinp.write('{!s:<12}   WrADFF          - Output full-field time-series data in TurbSim/AeroDyn form? (Generates RootName.bts)\n'.format(self.turbsim_vt.runtime_options.WrADFF))
         tsinp.write('{!s:<12}   WrBLFF          - Output full-field time-series data in BLADED/AeroDyn form?  (Generates RootName.wnd)\n'.format(self.turbsim_vt.runtime_options.WrBLFF))
         tsinp.write('{!s:<12}   WrADTWR         - Output tower time-series data? (Generates RootName.twr)\n'.format(self.turbsim_vt.runtime_options.WrADTWR))
         tsinp.write('{!s:<12}   WrFMTFF         - Output full-field time-series data in formatted (readable) form?  (Generates RootName.u, RootName.v, RootName.w)\n'.format(self.turbsim_vt.runtime_options.WrFMTFF))
         tsinp.write('{!s:<12}   WrACT           - Output coherent turbulence time steps in AeroDyn form? (Generates RootName.cts)\n'.format(self.turbsim_vt.runtime_options.WrACT))
         tsinp.write('{!s:<12}   Clockwise       - Clockwise rotation looking downwind? (used only for full-field binary files - not necessary for AeroDyn)\n'.format(self.turbsim_vt.runtime_options.Clockwise))
         tsinp.write('{!s:<12}   ScaleIEC        - Scale IEC turbulence models to exact target standard deviation? [0=no additional scaling; 1=use hub scale uniformly; 2=use individual scales]\n'.format(self.turbsim_vt.runtime_options.ScaleIEC))

         # Turbine/Model Specifications
         tsinp.write("\n")
         tsinp.write("--------Turbine/Model Specifications-----------------------\n")
         tsinp.write('{!s:<12}   NumGrid_Z       - Vertical grid-point matrix dimension\n'.format(self.turbsim_vt.tmspecs.NumGrid_Z))
         tsinp.write('{!s:<12}   NumGrid_Y       - Horizontal grid-point matrix dimension\n'.format(self.turbsim_vt.tmspecs.NumGrid_Y))
         tsinp.write('{!s:<12}   TimeStep        - Time step [seconds]\n'.format(self.turbsim_vt.tmspecs.TimeStep))
         tsinp.write('{!s:<12}   AnalysisTime    - Length of analysis time series [seconds] (program will add time if necessary: AnalysisTime = MAX(AnalysisTime, UsableTime+GridWidth/MeanHHWS) )\n'.format(self.turbsim_vt.tmspecs.AnalysisTime))
         tsinp.write('{!s:<12}   UsableTime      - Usable length of output time series [seconds] (program will add GridWidth/MeanHHWS seconds unless UsableTime is "ALL")\n'.format(self.turbsim_vt.tmspecs.UsableTime))
         tsinp.write('{!s:<12}   HubHt           - Hub height [m] (should be > 0.5*GridHeight)\n'.format(self.turbsim_vt.tmspecs.HubHt))
         tsinp.write('{!s:<12}   GridHeight      - Grid height [m]\n'.format(self.turbsim_vt.tmspecs.GridHeight))
         tsinp.write('{!s:<12}   GridWidth       - Grid width [m] (should be >= 2*(RotorRadius+ShaftLength))\n'.format(self.turbsim_vt.tmspecs.GridWidth))
         tsinp.write('{!s:<12}   VFlowAng        - Vertical mean flow (uptilt) angle [degrees]\n'.format(self.turbsim_vt.tmspecs.VFlowAng))
         tsinp.write('{!s:<12}   HFlowAng        - Horizontal mean flow (skew) angle [degrees]\n'.format(self.turbsim_vt.tmspecs.HFlowAng))

         # Meteorological Boundary Conditions
         tsinp.write("\n")
         tsinp.write("--------Meteorological Boundary Conditions-------------------\n")
         tsinp.write('{!s:<12}   TurbModel       - Turbulence model ("IECKAI","IECVKM","GP_LLJ","NWTCUP","SMOOTH","WF_UPW","WF_07D","WF_14D","TIDAL","API","USRINP","TIMESR", or "NONE")\n'.format(self.turbsim_vt.metboundconds.TurbModel))
         tsinp.write('{!s:<12}   UserFile        - Name of the file that contains inputs for user-defined spectra or time series inputs (used only for "USRINP" and "TIMESR" models)\n'.format(self.turbsim_vt.metboundconds.UserFile))
         tsinp.write('{!s:<12}   IECstandard     - Number of IEC 61400-x standard (x=1,2, or 3 with optional 61400-1 edition number (i.e. "1-Ed2") )\n'.format(self.turbsim_vt.metboundconds.IECstandard))
         tsinp.write('{!s:<12}   IECturbc        - IEC turbulence characteristic ("A", "B", "C" or the turbulence intensity in percent) ("KHTEST" option with NWTCUP model, not used for other models)\n'.format(self.turbsim_vt.metboundconds.IECturbc))
         tsinp.write('{!s:<12}   IEC_WindType    - IEC turbulence type ("NTM"=normal, "xETM"=extreme turbulence, "xEWM1"=extreme 1-year wind, "xEWM50"=extreme 50-year wind, where x=wind turbine class 1, 2, or 3)\n'.format(self.turbsim_vt.metboundconds.IEC_WindType))
         tsinp.write('{!s:<12}   ETMc            - IEC Extreme Turbulence Model "c" parameter [m/s]\n'.format(self.turbsim_vt.metboundconds.ETMc))
         tsinp.write('{!s:<12}   WindProfileType - Velocity profile type ("LOG";"PL"=power law;"JET";"H2L"=Log law for TIDAL model;"API";"USR";"TS";"IEC"=PL on rotor disk, LOG elsewhere; or "default")\n'.format(self.turbsim_vt.metboundconds.WindProfileType))
         tsinp.write('{!s:<12}   ProfileFile     - Name of the file that contains input profiles for WindProfileType="USR" and/or TurbModel="USRVKM" [-]\n'.format(self.turbsim_vt.metboundconds.ProfileFile))
         tsinp.write('{!s:<12}   RefHt           - Height of the reference velocity (URef) [m]\n'.format(self.turbsim_vt.metboundconds.RefHt))
         tsinp.write('{!s:<12}   URef            - Mean (total) velocity at the reference height [m/s] (or "default" for JET velocity profile) [must be 1-hr mean for API model; otherwise is the mean over AnalysisTime seconds]\n'.format(self.turbsim_vt.metboundconds.URef))
         tsinp.write('{!s:<12}   ZJetMax         - Jet height [m] (used only for JET velocity profile, valid 70-490 m)\n'.format(self.turbsim_vt.metboundconds.ZJetMax))
         tsinp.write('{!s:<12}   PLExp           - Power law exponent [-] (or "default")\n'.format(self.turbsim_vt.metboundconds.PLExp))
         tsinp.write('{!s:<12}   Z0              - Surface roughness length [m] (or "default")\n'.format(self.turbsim_vt.metboundconds.Z0))

         # Non-IEC Meteorological Boundary Conditions
         tsinp.write("\n")
         tsinp.write("--------Non-IEC Meteorological Boundary Conditions------------\n")
         tsinp.write('{!s:<12}   Latitude        - Site latitude [degrees] (or "default")\n'.format(self.turbsim_vt.noniecboundconds.Latitude))
         tsinp.write('{!s:<12}   RICH_NO         - Gradient Richardson number [-]\n'.format(self.turbsim_vt.noniecboundconds.RICH_NO))
         tsinp.write('{!s:<12}   UStar           - Friction or shear velocity [m/s] (or "default")\n'.format(self.turbsim_vt.noniecboundconds.UStar))
         tsinp.write('{!s:<12}   ZI              - Mixing layer depth [m] (or "default")\n'.format(self.turbsim_vt.noniecboundconds.ZI))
         tsinp.write('{!s:<12}   PC_UW           - Hub mean uw Reynolds stress [m^2/s^2] (or "default" or "none")\n'.format(self.turbsim_vt.noniecboundconds.PC_UW))
         tsinp.write('{!s:<12}   PC_UV           - Hub mean uv Reynolds stress [m^2/s^2] (or "default" or "none")\n'.format(self.turbsim_vt.noniecboundconds.PC_UV))
         tsinp.write('{!s:<12}   PC_VW           - Hub mean vw Reynolds stress [m^2/s^2] (or "default" or "none")\n'.format(self.turbsim_vt.noniecboundconds.PC_VW))

         # Spatial Coherence Parameters
         tsinp.write('\n')
         tsinp.write(
             '--------Spatial Coherence Parameters----------------------------\n')
         tsinp.write('{!s:<12}   SCMod1           - u-component coherence model ("GENERAL", "IEC", "API", "NONE", or "default")\n'.format(
             self.turbsim_vt.spatialcoherance.SCMod1))
         tsinp.write('{!s:<12}   SCMod2           - v-component coherence model ("GENERAL", "IEC", "NONE", or "default")\n'.format(
             self.turbsim_vt.spatialcoherance.SCMod2))
         tsinp.write('{!s:<12}   SCMod3           - w-component coherence model ("GENERAL", "IEC", "NONE", or "default")\n'.format(
             self.turbsim_vt.spatialcoherance.SCMod3))
         if not type(self.turbsim_vt.spatialcoherance.InCDec1) is str:
            tsinp.write('{:<5.2f}  {:<5.2f}   InCDec1        - u-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n'.format(
                float(self.turbsim_vt.spatialcoherance.InCDec1[0]), float(self.turbsim_vt.spatialcoherance.InCDec1[1])))
         else:
            tsinp.write('{!s:<12}   InCDec1        - u-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n'.format(
                self.turbsim_vt.spatialcoherance.InCDec1))
         if not type(self.turbsim_vt.spatialcoherance.InCDec2) is str:
            tsinp.write('{:<5.2f}  {:<5.2f}   InCDec2        - v-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n'.format(
                float(self.turbsim_vt.spatialcoherance.InCDec2[0]), float(self.turbsim_vt.spatialcoherance.InCDec2[1])))
         else:
            tsinp.write('{!s:<12}   InCDec2        - v-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n'.format(
                self.turbsim_vt.spatialcoherance.InCDec2))
         if not type(self.turbsim_vt.spatialcoherance.InCDec3) is str:
            tsinp.write('{:<5.2f}  {:<5.2f}   InCDec3        - w-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n'.format(
                float(self.turbsim_vt.spatialcoherance.InCDec3[0]), float(self.turbsim_vt.spatialcoherance.InCDec3[1])))
         else:
            tsinp.write('{!s:<12}   InCDec3        - w-component coherence parameters for general or IEC models [-, m^-1] (e.g. "10.0  0.3e-3" in quotes) (or "default")\n'.format(
                self.turbsim_vt.spatialcoherance.InCDec3))
         tsinp.write('{!s:<12}   CohExp           - Coherence exponent for general model [-] (or "default")\n'.format(
             self.turbsim_vt.spatialcoherance.CohExp))

         # Coherent Turbulence Scaling Parameters
         tsinp.write('\n')
         tsinp.write('--------Coherent Turbulence Scaling Parameters-------------------\n')
         tsinp.write('{!s:<12}   CTEventPath     - Name of the path where event data files are located\n'.format(self.turbsim_vt.coherentTurbulence.CTEventPath))
         tsinp.write('{!s:<12}   CTEventFile     - Type of event files ("LES", "DNS", or "RANDOM")\n'.format(self.turbsim_vt.coherentTurbulence.CTEventFile))
         tsinp.write('{!s:<12}   Randomize       - Randomize the disturbance scale and locations? (true/false)\n'.format(self.turbsim_vt.coherentTurbulence.Randomize))
         tsinp.write('{!s:<12}   DistScl         - Disturbance scale [-] (ratio of event dataset height to rotor disk). (Ignored when Randomize = true.)\n'.format(self.turbsim_vt.coherentTurbulence.DistScl))
         tsinp.write('{!s:<12}   CTLy            - Fractional location of tower centerline from right [-] (looking downwind) to left side of the dataset. (Ignored when Randomize = true.)\n'.format(self.turbsim_vt.coherentTurbulence.CTLy))
         tsinp.write('{!s:<12}   CTLz            - Fractional location of hub height from the bottom of the dataset. [-] (Ignored when Randomize = true.)\n'.format(self.turbsim_vt.coherentTurbulence.CTLz))
         tsinp.write('{!s:<12}   CTStartTime     - Minimum start time for coherent structures in RootName.cts [seconds]\n'.format(self.turbsim_vt.coherentTurbulence.CTStartTime))



if __name__=='__main__':
    s = TurbsimBuilder()
    s.turbsim_vt.metboundconds.UserFile = 'tsim_user_turbulence_default.inp'
    s.turbsim_vt.metboundconds.ProfileFile = 'default.profile'
    s.execute()
