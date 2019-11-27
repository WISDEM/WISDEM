import numpy as np
import os, sys
# import matplotlib.pyplot as plt 

from wisdem.aeroelasticse.Turbsim_mdao.turbsim_writer import TurbsimBuilder
from wisdem.aeroelasticse.Turbsim_mdao.turbsim_wrapper import Turbsim_wrapper
from wisdem.aeroelasticse.Turbsim_mdao.turbsim_vartrees import turbsiminputs

# from AeroelasticSE.Turbsim_mdao.pyturbsim_wrapper import pyTurbsim_wrapper

class pyIECWind_extreme():

    def __init__(self):

        self.Turbine_Class    = 'I'    # IEC Wind Turbine Class
        self.Turbulence_Class = 'B'    # IEC Turbulance Class
        self.Vert_Slope       = 0      # Vertical slope of the wind inflow (deg)
        self.TStart           = 30     # Time to start transient conditions (s)
        self.dt               = 0.05   # Transient wind time step (s)
        self.dir_change       = 'both' # '+','-','both': sign for transient events in EDC, EWS
        self.shear_orient     = 'both' # 'v','h','both': vertical or horizontal shear for EWS
        self.z_hub            = 90.    # wind turbine hub height (m)
        self.D                = 126.   # rotor diameter (m)
        
        self.T0               = 0.
        self.TF               = 630.

    def setup(self):
        # General turbulence parameters: 6.3
        # Sigma_1: logitudinal turbulence scale parameter

        # Setup
        if self.Turbine_Class == 'I':
            self.V_ref = 50.
        elif self.Turbine_Class == 'II':
            self.V_ref = 42.5
        elif self.Turbine_Class == 'III':
            self.V_ref = 37.5
        elif self.Turbine_Class == 'IV':
            self.V_ref = 30.
        self.V_ave = self.V_ref*0.2

        if self.Turbulence_Class == 'A+':
            self.I_ref = 0.18
        elif self.Turbulence_Class == 'A':
            self.I_ref = 0.16
        elif self.Turbulence_Class == 'B':
            self.I_ref = 0.14
        elif self.Turbulence_Class == 'C':
            self.I_ref = 0.12

        if self.z_hub > 60:
            self.Sigma_1 = 42
        else:
            self.Sigma_1 = 0.7*self.z_hub
            

    def NTM(self, V_hub):
        # Normal turbulence model: 6.3.1.3
        b = 5.6
        sigma_1 = self.I_ref*(0.75*V_hub + b)
        return sigma_1

    def ETM(self, V_hub):
        # Extreme turbulence model: 6.3.2.3
        c = 2
        sigma_1 = c*self.I_ref*(0.072*(self.V_ave/c + 3)*(V_hub/c - 4) + 10)
        return sigma_1

    def EWM(self, V_hub):
        # Extreme wind speed model: 6.3.2.1
                
        # Steady
        V_e50 = 1.4*self.V_ref
        V_e1 = 0.8*V_e50
        # Turb
        V_50 = self.V_ref
        V_1 = 0.8*V_50
        sigma_1 = 0.11*V_hub

        return sigma_1, V_e50, V_e1, V_50, V_1

    def EOG(self, V_hub_in):
        # Extreme operating guest: 6.3.2.2

        self.setup()

        T = 10.5
        t = np.linspace(0., T, num=(T/self.dt+1))

        # constants from standard
        alpha = 0.2

        # Flow angle adjustments
        V_hub = V_hub_in*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = V_hub_in*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = self.NTM(V_hub)
        __, __, V_e1, __, __ = self.EWM(V_hub)

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_dir = np.zeros_like(t)
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)

        V_gust = min([ 1.35*(V_e1 - V_hub), 3.3*(sigma_1/(1+0.1*(self.D/self.Sigma_1))) ])

        V_gust_t = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                V_gust_t[i] = 0. - 0.37*V_gust*np.sin(3*np.pi*ti/T)*(1-np.cos(2*np.pi*ti/T))
            else:
                V_gust_t[i] = 0.

        # Write Files
        self.fname_out = []
        self.fname_type = []
        fname = self.case_name + '_EOG_U%2.1f.wnd'%V_hub_in
        data = np.column_stack((t, V, V_dir, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust_t))
        hd = []
        hd.append('! Extreme operating guest\n')
        hd = self.heading_common(hd)
        hd = self.heading_variable(hd, V_hub_in=V_hub_in, sigma_1=sigma_1)
        self.write_wnd(fname, data, hd)
        self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
        self.fname_type.append(2)



    def EDC(self, V_hub_in):
        # Extreme direction change: 6.3.2.4

        self.setup()
        
        T = 6.
        t = np.linspace(0., T, num=(T/self.dt+1))

        # constants from standard
        alpha = 0.2

        # Flow angle adjustments
        V_hub = V_hub_in*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = V_hub_in*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = self.NTM(V_hub)

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)

        # Transcient
        Theta_e = 4.*np.arctan(sigma_1/(V_hub*(1.+0.01*(self.D/self.Sigma_1))))*180./np.pi
        if Theta_e > 180.:
            Theta_e = 180.

        Theta_p = np.zeros_like(t)
        Theta_n = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                Theta_p[i] = 0.5*Theta_e*(1-np.cos(np.pi*ti/T))
                Theta_n[i] = -1*0.5*Theta_e*(1-np.cos(np.pi*ti/T))
            else:
                Theta_p[i] = Theta_e
                Theta_n[i] = -1*Theta_e

        # Write Files
        self.fname_out = []
        self.fname_type = []
        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '+':
            ## Vert
            fname = self.case_name + '_EDC_P_U%2.1f.wnd'%V_hub_in
            data = np.column_stack((t, V, Theta_p, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust))
            hd = []
            hd.append('! Exteme Vertical Wind Shear, positive\n')
            hd = self.heading_common(hd)
            hd = self.heading_variable(hd, V_hub_in=V_hub_in, sigma_1=sigma_1)
            self.write_wnd(fname, data, hd)
            self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
            self.fname_type.append(2)


        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '-':
            ## Vert
            fname = self.case_name + '_EDC_N_U%2.1f.wnd'%V_hub_in
            data = np.column_stack((t, V, Theta_n, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust))
            hd = []
            hd.append('! Exteme Vertical Wind Shear, negative\n')
            hd = self.heading_common(hd)
            hd = self.heading_variable(hd, V_hub_in=V_hub_in, sigma_1=sigma_1)
            self.write_wnd(fname, data, hd)
            self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
            self.fname_type.append(2)

         
    def ECD(self, V_hub_in):
        # Extreme coherent gust with direction change: 6.3.2.5

        self.setup()
        
        T = 10.
        t = np.linspace(0., T, num=(T/self.dt+1))

        # constants from standard
        alpha = 0.2
        V_cg = 15 #m/s

        # Flow angle adjustments
        V_hub = V_hub_in*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = V_hub_in*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = self.NTM(V_hub)

        # Contant variables
        V_vert = np.zeros_like(t)+V_vert_mag
        shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        shear_vert_lin = np.zeros_like(t)
        V_gust = np.zeros_like(t)

        # Transcient
        if V_hub < 4:
            Theta_cg = 180
        else:
            Theta_cg = 720/V_hub

        Theta_p = np.zeros_like(t)
        Theta_n = np.zeros_like(t)
        V = np.zeros_like(t)
        for i, ti in enumerate(t):
            if ti<T:
                V[i] = V_hub + 0.5*V_cg*(1-np.cos(np.pi*ti/T))
                Theta_p[i] = 0.5*Theta_cg*(1-np.cos(np.pi*ti/T))
                Theta_n[i] = -1*0.5*Theta_cg*(1-np.cos(np.pi*ti/T))
            else:
                V[i] = V_hub+V_cg
                Theta_p[i] = Theta_cg
                Theta_n[i] = -1*Theta_cg

        # Write Files
        self.fname_out = []
        self.fname_type = []
        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '+':
            ## Vert
            fname = self.case_name + '_ECD_P_U%2.1f.wnd'%V_hub_in
            data = np.column_stack((t, V, Theta_p, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust))
            hd = []
            hd.append('! Exteme coherent gust with direction change, positive\n')
            hd = self.heading_common(hd)
            hd = self.heading_variable(hd, V_hub_in=V_hub_in)
            self.write_wnd(fname, data, hd)
            self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
            self.fname_type.append(2)


        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '-':
            ## Vert
            fname = self.case_name + '_ECD_N_U%2.1f.wnd'%V_hub_in
            data = np.column_stack((t, V, Theta_n, V_vert, shear_horz, shear_vert, shear_vert_lin, V_gust))
            hd = []
            hd.append('! Exteme coherent gust with direction change, negative\n')
            hd = self.heading_common(hd)
            hd = self.heading_variable(hd, V_hub_in=V_hub_in)
            self.write_wnd(fname, data, hd)
            self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
            self.fname_type.append(2)


    def EWS(self, V_hub_in):
        # Extreme wind shear: 6.3.2.6
        self.setup()

        T = 12
        t = np.linspace(0, T, num=(T/self.dt+1))

        # constants from standard
        alpha = 0.2
        Beta = 6.4

        # Flow angle adjustments
        V_hub = V_hub_in*np.cos(self.Vert_Slope*np.pi/180)
        V_vert_mag = V_hub_in*np.sin(self.Vert_Slope*np.pi/180)

        sigma_1 = self.NTM(V_hub)

        # Contant variables
        V = np.zeros_like(t)+V_hub
        V_dir = np.zeros_like(t)
        V_vert = np.zeros_like(t)+V_vert_mag
        # shear_horz = np.zeros_like(t)
        shear_vert = np.zeros_like(t)+alpha
        V_gust = np.zeros_like(t)

        # Transcient
        shear_lin_p = np.zeros_like(t)
        shear_lin_n = np.zeros_like(t)

        for i, ti in enumerate(t):
            shear_lin_p[i] = (2.5+0.2*Beta*sigma_1*(self.D/self.Sigma_1)**(1/4))*(1-np.cos(2*np.pi*ti/T))/V_hub
            shear_lin_n[i] = -1*(2.5+0.2*Beta*sigma_1*(self.D/self.Sigma_1)**(1/4))*(1-np.cos(2*np.pi*ti/T))/V_hub

        # Write Files
        self.fname_out = []
        self.fname_type = []
        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '+':
            if self.shear_orient.lower() == 'both' or self.shear_orient.lower() == 'v':
                ## Vert
                fname = self.case_name + '_EWS_V_P_U%2.1f.wnd'%V_hub_in
                data = np.column_stack((t, V, V_dir, V_vert, np.zeros_like(t), shear_vert, shear_lin_p, V_gust))
                hd = []
                hd.append('! Exteme Vertical Wind Shear, positive\n')
                hd = self.heading_common(hd)
                hd = self.heading_variable(hd, V_hub_in=V_hub_in)
                self.write_wnd(fname, data, hd)
                self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
                self.fname_type.append(2)

            if self.shear_orient.lower() == 'both' or self.shear_orient.lower() == 'h':
                # Horz
                fname = self.case_name + '_EWS_H_P_U%2.1f.wnd'%V_hub_in
                data = np.column_stack((t, V, V_dir, V_vert, shear_lin_p, shear_vert, np.zeros_like(t), V_gust))
                hd = []
                hd.append('! Exteme Horizontal Wind Shear, positive\n')
                hd = self.heading_common(hd)
                hd = self.heading_variable(hd, V_hub_in=V_hub_in)
                self.write_wnd(fname, data, hd)
                self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
                self.fname_type.append(2)

        if self.dir_change.lower() == 'both' or self.dir_change.lower() == '-':
            if self.shear_orient.lower() == 'both' or self.shear_orient.lower() == 'v':
                ## Vert
                fname = self.case_name + '_EWS_V_N_U%2.1f.wnd'%V_hub_in
                data = np.column_stack((t, V, V_dir, V_vert, np.zeros_like(t), shear_vert, shear_lin_n, V_gust))
                hd = []
                hd.append('! Exteme Vertical Wind Shear, negative\n')
                hd = self.heading_common(hd)
                hd = self.heading_variable(hd, V_hub_in=V_hub_in)
                self.write_wnd(fname, data, hd)
                self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
                self.fname_type.append(2)

            if self.shear_orient.lower() == 'both' or self.shear_orient.lower() == 'h':
                # Horz
                fname = self.case_name + '_EWS_H_N_U%2.1f.wnd'%V_hub_in
                data = np.column_stack((t, V, V_dir, V_vert, shear_lin_n, shear_vert, np.zeros_like(t), V_gust))
                hd = []
                hd.append('! Exteme Horizontal Wind Shear, negative\n')
                hd = self.heading_common(hd)
                hd = self.heading_variable(hd, V_hub_in=V_hub_in)
                self.write_wnd(fname, data, hd)
                self.fname_out.append(os.path.realpath(os.path.normpath(os.path.join(self.outdir, fname))))
                self.fname_type.append(2)

    def heading_common(self, hd):
        hd.append('! IEC Turbine Class %s, IEC Turbulence Category %s\n'%(self.Turbine_Class, self.Turbulence_Class))
        hd.append('! '+('%f'%(self.D)).ljust(14) + ' ' + 'Rotor_Diameter'.ljust(14) + ' - rotor diameter (m)\n')
        hd.append('! '+('%f'%(self.z_hub)).ljust(14) + ' ' + 'hub_height'.ljust(14) + ' - hub height of the wind turbine (m)\n')
        hd.append('! '+('%f'%(self.Sigma_1)).ljust(14) + ' ' + 'Sigma_1'.ljust(14) + ' - logitudinal turbulence scale parameter (m)\n')
        return hd
    
    def heading_variable(self, hd, V_hub_in=[], sigma_1=[]):
        if sigma_1:
            hd.append('! '+('%f'%(sigma_1)).ljust(14) + ' ' + 'sigma_1'.ljust(14) + ' - turbulence standard deviation\n')
        if V_hub_in:
            hd.append('! '+('%f'%(V_hub_in)).ljust(14) + ' ' + 'V_hub'.ljust(14) + ' - wind speed at hub height (m/s)\n')

        return hd

    def write_wnd(self, fname, data, hd):

        # Make sure directory exist
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

        # Move transcient event to user definted time
        data[:,0] += self.TStart
        data = np.vstack((data[0,:], data, data[-1,:]))
        data[0,0] = self.T0
        data[-1,0] = self.TF

        # Headers
        hd1 = ['Time', 'Wind', 'Wind', 'Vertical', 'Horiz.', 'Pwr. Law', 'Lin. Vert.', 'Gust']
        hd2 = ['', 'Speed', 'Dir', 'Speed', 'Shear', 'Vert. Shr', 'Shear', 'Speed']
        hd3 = ['(s)', '(m/s)', '(deg)', '(m/s)', '(-)', '(-)', '(-)', '(m/s)', ]

        self.fpath = os.path.join(self.outdir, fname)
        fid = open(self.fpath, 'w')

        fid.write('! Wind file generated by pyIECWind - IEC 61400-1 3rd Edition\n')
        for ln in hd:
            fid.write(ln)
        fid.write('! ---------------------------------------------------------------\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd1]) + '\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd2]) + '\n')
        fid.write('! '+''.join([('%s' % val).center(12) for val in hd3]) + '\n')
        for row in data:
            fid.write('  '+''.join([('%.6f' % val).center(12) for val in row]) + '\n')

        fid.close()

    def execute(self, Vtype, V_hub):

        if 'EOG' in Vtype:
            self.EOG(V_hub)
        if 'EDC' in Vtype:
            self.EDC(V_hub)
        if 'ECD' in Vtype:
            self.ECD(V_hub)
        if 'EWS' in Vtype:
            self.EWS(V_hub)

        return self.fname_out, self.fname_type
        

class pyIECWind_turb():

    def __init__(self):

        # Defaults
        self.seed             = np.random.uniform(1, 1e8)
        self.Turbulence_Class = 'B'  # IEC Turbulance Class
        self.z_hub            = 90.  # wind turbine hub height (m)
        self.D                = 126. # rotor diameter (m)
        self.PLExp            = 0.2
        self.AnalysisTime     = 720.
        self.debug_level      = 0
        self.overwrite        = True

    def setup(self):
        turbsim_vt = turbsiminputs()
        turbsim_vt.runtime_options.RandSeed1  = self.seed
        turbsim_vt.runtime_options.WrADTWR    = False
        turbsim_vt.tmspecs.AnalysisTime       = self.AnalysisTime
        turbsim_vt.tmspecs.HubHt              = self.z_hub
        turbsim_vt.tmspecs.GridHeight         = np.ceil(self.D*1.05)
        turbsim_vt.tmspecs.GridWidth          = np.ceil(self.D*1.05)
        turbsim_vt.tmspecs.NumGrid_Z          = 21
        turbsim_vt.tmspecs.NumGrid_Y          = 21
        turbsim_vt.tmspecs.HFlowAng           = 0.0
        turbsim_vt.tmspecs.VFlowAng           = 0.0
        turbsim_vt.metboundconds.TurbModel    = '"IECKAI"'
        turbsim_vt.metboundconds.UserFile     = '"unused"'
        turbsim_vt.metboundconds.IECturbc     = self.Turbulence_Class
        turbsim_vt.metboundconds.IEC_WindType = self.IEC_WindType
        turbsim_vt.metboundconds.ETMc         = '"default"'
        turbsim_vt.metboundconds.WindProfileType = '"PL"'
        turbsim_vt.metboundconds.ProfileFile  = '"unused"'
        turbsim_vt.metboundconds.RefHt        = self.z_hub
        turbsim_vt.metboundconds.URef         = self.Uref
        turbsim_vt.metboundconds.PLExp        = self.PLExp
        
        turbsim_vt.noniecboundconds.Latitude  = '"default"'
        turbsim_vt.noniecboundconds.RICH_NO   = 0.05
        turbsim_vt.noniecboundconds.UStar     = '"default"'
        turbsim_vt.noniecboundconds.ZI        = '"default"'
        turbsim_vt.noniecboundconds.PC_UW     = '"default"'
        turbsim_vt.noniecboundconds.PC_UV     = '"default"'
        turbsim_vt.noniecboundconds.PC_VW     = '"default"'
        
        
        
        return turbsim_vt

    def execute(self, IEC_WindType, Uref, ver='Turbsim'):
        self.IEC_WindType = IEC_WindType
        self.Uref = Uref

        turbsim_vt = self.setup()
        writer = TurbsimBuilder()
        if ver.lower() == 'turbsim':
            wrapper = Turbsim_wrapper()
        if ver.lower() =='pyturbsim':
            wrapper = pyTurbsim_wrapper()

        # if self.case_name[-3:] != '.in':
        #     self.case_name = self.case_name + '.in'
        # self.case_name += '_U%1.1f'%self.Uref + '_Seed%1.1f'%self.seed
        # self.case_name += '_U%d'%self.Uref + '_Seed%d.in'%self.seed

        case_name = self.case_name + '_' + IEC_WindType + '_U%1.6f'%self.Uref + '_Seed%1.1f'%self.seed
        
        tsim_input_file = case_name + '.in'
        wind_file_out   = case_name + '.bts'
        
        wind_file_out_abs = os.path.realpath(os.path.normpath(os.path.join(self.outdir, wind_file_out)))

        # If wind file already exists and overwriting is turned off, skip wind file write
        if os.path.exists(os.path.join(self.outdir, wind_file_out)) and not self.overwrite:
            return wind_file_out_abs, 3

        # Run wind file generation
        else:
            writer.turbsim_vt = turbsim_vt
            writer.run_dir = self.outdir
            writer.tsim_input_file = tsim_input_file
            writer.execute()

            wrapper.turbsim_input = os.path.realpath(os.path.join(writer.run_dir, writer.tsim_input_file))
            wrapper.run_dir = writer.run_dir
            wrapper.turbsim_exe = self.Turbsim_exe
            wrapper.debug_level = self.debug_level
            wrapper.execute()

            return wind_file_out_abs, 3


def example_ExtremeWind():

    iec = pyIECWind_extreme()
    iec.Turbine_Class = 'I'     # IEC Wind Turbine Class
    iec.Turbulence_Class = 'A'  # IEC Turbulance Class
    iec.dt = 0.05               # Transient wind time step (s)
    iec.dir_change = 'both'     # '+','-','both': sign for transient events in EDC, EWS
    iec.z_hub = 30.             # wind turbine hub height (m)
    iec.D = 42.                 # rotor diameter (m)

    iec.case_name = 'test'
    iec.outdir = 'temp'

    V_hub = 25
    iec.execute('EWS', V_hub)

def example_TurbulentWind():
    iec = pyIECWind_turb()
    
    iec.Turbulence_Class = 'A'  # IEC Turbulance Class
    iec.z_hub = 90.             # wind turbine hub height (m)
    iec.D = 126.                 # rotor diameter (m)
    iec.AnalysisTime = 30.

    iec.outdir = 'temp'
    iec.case_name = 'turbsim_testing'
    iec.Turbsim_exe = 'C:/Users/egaertne/WT_Codes/Turbsim_v2.00.07/bin/TurbSim_x64.exe'
    iec.debug_level = 1

    IEC_WindType = 'NTM'
    Uref = 10.

    iec.execute(IEC_WindType, Uref)


if __name__=="__main__":

    example_ExtremeWind()
    # example_TurbulentWind()
