# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# speROSCO_cific language governing permissions and limitations under the License.

import numpy as np
from ROSCO_toolbox import turbine as ROSCO_turbine
import matplotlib.pyplot as plt
import sys

# Some useful constants
deg2rad = np.deg2rad(1)
rad2deg = np.rad2deg(1)
rpm2RadSec = 2.0*(np.pi)/60.0

class Sim():
    """
    Simple controller simulation interface for a wind turbine.
     - Currently runs a 1DOF simple rotor model based on an OpenFAST model

    Note: Due to the complex nature of the wind speed estimators implemented in ROSCO, 
    using them for simulations is known to cause problems for the simple simulator. 
    We suggesting using WE_Mode = 0 for the simulator


    Methods:
    --------
    sim_ws_series

    Parameters:
    -----------
    turbine: class
             Turbine class containing wind turbine information from OpenFAST model
    controller_int: class
                    Controller interface class to run compiled controller binary
    """

    def __init__(self, turbine, controller_int):
        """
        Setup the simulator
        """
        self.turbine = turbine
        self.controller_int = controller_int


    def sim_ws_series(self,t_array,ws_array,rotor_rpm_init=10,init_pitch=0.0, make_plots=True):
        '''
        Simulate simplified turbine model using a complied controller (.dll or similar).
            - currently a 1DOF rotor model

        Parameters:
        -----------
            t_array: float
                     Array of time steps, (s)
            ws_array: float
                      Array of wind speeds, (s)
            rotor_rpm_init: float, optional
                            initial rotor speed, (rpm)
            init_pitch: float, optional
                        initial blade pitch angle, (deg)
            make_plots: bool, optional
                        True: generate plots, False: don't. 
        '''

        print('Running simulation for %s wind turbine.' % self.turbine.TurbineName)

        # Store turbine data for conveniente
        dt = t_array[1] - t_array[0]
        R = self.turbine.rotor_radius
        GBRatio = self.turbine.Ng

        # Declare output arrays
        bld_pitch = np.ones_like(t_array) * init_pitch 
        rot_speed = np.ones_like(t_array) * rotor_rpm_init * rpm2RadSec # represent rot speed in rad / s
        gen_speed = np.ones_like(t_array) * rotor_rpm_init * GBRatio * rpm2RadSec # represent gen speed in rad/s
        aero_torque = np.ones_like(t_array) * 1000.0
        gen_torque = np.ones_like(t_array) # * trq_cont(turbine_dict, gen_speed[0])
        gen_power = np.ones_like(t_array) * 0.0

        
        # Loop through time
        for i, t in enumerate(t_array):
            if i == 0:
                continue # Skip the first run
            ws = ws_array[i]

            # Load current Cq data
            tsr = rot_speed[i-1] * self.turbine.rotor_radius / ws
            cq = self.turbine.Cq.interp_surface([bld_pitch[i-1]],tsr)
        
            # Update the turbine state
            #       -- 1DOF model: rotor speed and generator speed (scaled by Ng)
            aero_torque[i] = 0.5 * self.turbine.rho * (np.pi * R**2) * cq * R * ws**2
            rot_speed[i] = rot_speed[i-1] + (dt/self.turbine.J)*(aero_torque[i] * self.turbine.GenEff/100 - self.turbine.Ng * gen_torque[i-1])
            gen_speed[i] = rot_speed[i] * self.turbine.Ng

            # Call the controller
            gen_torque[i], bld_pitch[i] = self.controller_int.call_controller(t,dt,bld_pitch[i-1],gen_torque[i-1],gen_speed[i],self.turbine.GenEff/100,rot_speed[i],ws)

            # Calculate the power
            gen_power[i] = gen_speed[i] * gen_torque[i]

        # Save these values
        self.bld_pitch = bld_pitch
        self.rot_speed = rot_speed
        self.gen_speed = gen_speed
        self.aero_torque = aero_torque
        self.gen_torque = gen_torque
        self.gen_power = gen_power
        self.t_array = t_array
        self.ws_array = ws_array

        if make_plots:
            fig, axarr = plt.subplots(4,1,sharex=True,figsize=(6,10))

            ax = axarr[0]
            ax.plot(self.t_array,self.ws_array)
            ax.set_ylabel('Wind Speed (m/s)')
            ax.grid()
            ax = axarr[1]
            ax.plot(self.t_array,self.rot_speed)
            ax.set_ylabel('Rot Speed (rad/s)')
            ax.grid()
            ax = axarr[2]
            ax.plot(self.t_array,self.gen_torque)
            ax.set_ylabel('Gen Torque (N)')
            ax.grid()
            ax = axarr[3]
            ax.plot(self.t_array,self.bld_pitch*rad2deg)
            ax.set_ylabel('Bld Pitch (deg)')
            ax.set_xlabel('Time (s)')
            ax.grid()

        
