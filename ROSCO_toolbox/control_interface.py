# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from ctypes import byref, cdll, c_int, POINTER, c_float, c_char_p, c_double, create_string_buffer, c_int32
import numpy as np
from numpy.ctypeslib import ndpointer

# Some useful constants
deg2rad = np.deg2rad(1)
rad2deg = np.rad2deg(1)
rpm2RadSec = 2.0*(np.pi)/60.0

class ControllerInterface():
    """
    Define interface to a given controller using the avrSWAP array

    Methods:
    --------
    call_discon
    call_controller
    show_control_values

    Parameters:
    -----------
    lib_name : str
                name of compiled dynamic library containing controller, (.dll,.so,.dylib)

    """

    def __init__(self, lib_name, param_filename='DISCON.IN'):
        """
        Setup the interface
        """
        self.lib_name = lib_name
        self.param_name = param_filename

        # Temp fixed parameters
        # PARAMETERS
        self.DT = 0.1
        self.num_blade = 3
        self.char_buffer = 500
        self.avr_size = 500

        # Initialize
        self.pitch = 0
        self.torque = 0
        # -- discon
        self.discon = cdll.LoadLibrary(self.lib_name)
        self.avrSWAP = np.zeros(self.avr_size)

        # Define some avrSWAP parameters
        self.avrSWAP[2] = self.DT
        self.avrSWAP[60] = self.num_blade
        self.avrSWAP[20] = 1 # HARD CODE initial rot speed = 1 rad/s
        self.avrSWAP[26] = 10 # HARD CODE initial wind speed = 10 m/s


        # Code this as first casll
        self.avrSWAP[0] = 0

        # Put some values in
        self.avrSWAP[58] = self.char_buffer
        self.avrSWAP[49] = len(self.param_name)
        self.avrSWAP[50] = self.char_buffer
        self.avrSWAP[51] = self.char_buffer

        # Initialize DISCON and related
        self.aviFAIL = c_int32() # 1
        self.accINFILE = self.param_name.encode('utf-8')
        self.avcOUTNAME = create_string_buffer(1000) # 'DEMO'.encode('utf-8')
        self.avcMSG = create_string_buffer(1000)
        self.discon.DISCON.argtypes = [POINTER(c_float), POINTER(c_int32), c_char_p, c_char_p, c_char_p] # (all defined by ctypes)

        # Run DISCON
        self.call_discon()

        # Code as not first run
        self.avrSWAP[0] = 1


    def call_discon(self):
        '''
        Call libdiscon.dll (or .so,.dylib,...)
        '''
        # Convert AVR swap to the c pointer
        c_float_p = POINTER(c_float)
        data = self.avrSWAP.astype(np.float32)
        p_data = data.ctypes.data_as(c_float_p)

        # Run DISCON
        self.discon.DISCON(p_data, byref(self.aviFAIL), self.accINFILE, self.avcOUTNAME, self.avcMSG)

        # Push back to avr swap
        self.avrSWAP = data


    def call_controller(self,t,dt,pitch,torque,genspeed,geneff,rotspeed,ws):
        '''
        Runs the controller. Passes current turbine state to the controller, and returns control inputs back
        
        Parameters:
        -----------
        t: float
           time, (s)
        dt: float
            timestep, (s)
        pitch: float
               blade pitch, (rad)
        genspeed: float
                  generator speed, (rad/s)
        geneff: float
                  generator efficiency, (rad/s)
        rotspeed: float
                  rotor speed, (rad/s)
        ws: float
            wind speed, (m/s)
        '''

        # Add states to avr
        self.avrSWAP[1] = t
        self.avrSWAP[2] = dt
        self.avrSWAP[3] = pitch
        self.avrSWAP[32] = pitch
        self.avrSWAP[33] = pitch
        self.avrSWAP[14] = genspeed*torque*geneff
        self.avrSWAP[22] = torque
        self.avrSWAP[19] = genspeed
        self.avrSWAP[20] = rotspeed
        self.avrSWAP[26] = ws

        # call controller
        self.call_discon()

        # return controller states
        self.pitch = self.avrSWAP[41]
        self.torque = self.avrSWAP[46]

        return(self.torque,self.pitch)

    def show_control_values(self):
        '''
        Show control values - should be obvious
        '''
        print('Pitch',self.pitch)
        print('Torque',self.torque)
