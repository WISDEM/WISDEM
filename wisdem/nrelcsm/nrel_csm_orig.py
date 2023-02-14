"""
aero_csm_component.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from math import pi, exp, gamma

import numpy as np

from wisdem.nrelcsm.csmPPI import PPI
from wisdem.commonse.utilities import hstack, smooth_abs, smooth_min

# Initialize ref and current YYYYMM
# Calling program can override these
#   e.g., ppi.ref_yr = 2003, etc.

ref_yr = 2002
ref_mon = 9
curr_yr = 2009
curr_mon = 12

ppi = PPI(ref_yr, ref_mon, curr_yr, curr_mon)


# NREL Cost and Scaling Model plant energy modules
##################################################


class aero_csm(object):
    def __init__(self):
        # Variables
        # machine_rating = Float(units = 'kW', iotype='in', desc= 'rated machine power in kW')
        # max_tip_speed = Float(units = 'm/s', iotype='in', desc= 'maximum allowable tip speed for the rotor')
        # rotor_diameter = Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine')
        # max_power_coefficient = Float(iotype='in', desc= 'maximum power coefficient of rotor for operation in region 2')
        # opt_tsr = Float(iotype='in', desc= 'optimum tip speed ratio for operation in region 2')
        # cut_in_wind_speed = Float(units = 'm/s', iotype='in', desc= 'cut in wind speed for the wind turbine')
        # cut_out_wind_speed = Float(units = 'm/s', iotype='in', desc= 'cut out wind speed for the wind turbine')
        # hub_height = Float(units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level')
        # altitude = Float(units = 'm', iotype='in', desc= 'altitude of wind plant')
        # air_density = Float(units = 'kg / (m * m * m)', iotype='in', desc= 'air density at wind plant site')  # default air density value is 0.0 - forces aero csm to calculate air density in model
        # max_efficiency = Float(iotype='in', desc = 'maximum efficiency of rotor and drivetrain - at rated power')
        # thrust_coefficient = Float(iotype='in', desc='thrust coefficient at rated power')

        # Outputs
        self.rated_wind_speed = 0.0  # Float(units = 'm / s', iotype='out', desc='wind speed for rated power')
        self.rated_rotor_speed = 0.0  # Float(units = 'rpm', iotype='out', desc = 'rotor speed at rated power')
        self.rotor_thrust = 0.0  # Float(iotype='out', units='N', desc='maximum thrust from rotor')
        self.rotor_torque = 0.0  # Float(iotype='out', units='N * m', desc = 'torque from rotor at rated power')
        self.power_curve = np.zeros(161)  # Array(iotype='out', units='kW', desc='total power before drivetrain losses')
        self.wind_curve = np.zeros(
            161
        )  # Array(iotype='out', units='m/s', desc='wind curve associated with power curve')

    def compute(
        self,
        machine_rating,
        max_tip_speed,
        rotor_diameter,
        max_power_coefficient,
        opt_tsr,
        cut_in_wind_speed,
        cut_out_wind_speed,
        hub_height,
        altitude,
        air_density,
        max_efficiency,
        thrust_coefficient,
    ):
        """
        Executes Aerodynamics Sub-module of the NREL _cost and Scaling Model to create a power curve based on a limited set of inputs.
        It then modifies the ideal power curve to take into account drivetrain efficiency losses through an interface to a drivetrain efficiency model.
        """

        # initialize input parameters
        self.hubHt = hub_height
        self.ratedPower = machine_rating
        self.maxTipSpd = max_tip_speed
        self.rotorDiam = rotor_diameter
        self.maxCp = max_power_coefficient
        self.maxTipSpdRatio = opt_tsr
        self.cutInWS = cut_in_wind_speed
        self.cutOutWS = cut_out_wind_speed

        if air_density == 0.0:
            # Compute air density
            ssl_pa = 101300  # std sea-level pressure in Pa
            gas_const = 287.15  # gas constant for air in J/kg/K
            gravity = 9.80665  # standard gravity in m/sec/sec
            lapse_rate = 0.0065  # temp lapse rate in K/m
            ssl_temp = 288.15  # std sea-level temp in K

            air_density = (
                ssl_pa
                * (1 - ((lapse_rate * (altitude + self.hubHt)) / ssl_temp)) ** (gravity / (lapse_rate * gas_const))
            ) / (gas_const * (ssl_temp - lapse_rate * (altitude + self.hubHt)))
        else:
            air_density = air_density

        # determine power curve inputs
        self.reg2pt5slope = 0.05

        # self.max_efficiency = self.drivetrain.getMaxEfficiency()
        self.ratedHubPower = self.ratedPower / max_efficiency  # RatedHubPower

        self.omegaM = self.maxTipSpd / (self.rotorDiam / 2.0)  # Omega M - rated rotor speed
        omega0 = self.omegaM / (1 + self.reg2pt5slope)  # Omega 0 - rotor speed at which region 2 hits zero torque
        Tm = self.ratedHubPower * 1000 / self.omegaM  # Tm - rated torque

        # compute rated rotor speed
        self.ratedRPM = (30.0 / pi) * self.omegaM

        # compute variable-speed torque constant k
        kTorque = (air_density * pi * self.rotorDiam**5 * self.maxCp) / (64 * self.maxTipSpdRatio**3)  # k

        b = -Tm / (self.omegaM - omega0)  # b - quadratic formula values to determine omegaT
        c = (Tm * omega0) / (self.omegaM - omega0)  # c

        # omegaT is rotor speed at which regions 2 and 2.5 intersect
        # add check for feasibility of omegaT calculation 09/20/2012
        omegaTflag = True
        if (b**2 - 4 * kTorque * c) > 0:
            omegaT = -(b / (2 * kTorque)) - (np.sqrt(b**2 - 4 * kTorque * c) / (2 * kTorque))  # Omega T

            windOmegaT = (omegaT * self.rotorDiam) / (2 * self.maxTipSpdRatio)  # Wind  at omegaT (M25)
            pwrOmegaT = kTorque * omegaT**3 / 1000  # Power at ometaT (M26)

        else:
            omegaTflag = False
            windOmegaT = self.ratedRPM
            pwrOmegaT = self.ratedPower

        # compute rated wind speed
        d = air_density * np.pi * self.rotorDiam**2.0 * 0.25 * self.maxCp
        self.ratedWindSpeed = 0.33 * ((2.0 * self.ratedHubPower * 1000.0 / (d)) ** (1.0 / 3.0)) + 0.67 * (
            (((self.ratedHubPower - pwrOmegaT) * 1000.0) / (1.5 * d * windOmegaT**2.0)) + windOmegaT
        )

        # set up for idealized power curve
        n = 161  # number of wind speed bins
        itp = [None] * n
        ws_inc = 0.25  # size of wind speed bins for integrating power curve
        Wind = []
        Wval = 0.0
        Wind.append(Wval)
        for i in range(1, n):
            Wval += ws_inc
            Wind.append(Wval)

        # determine idealized power curve
        self.idealPowerCurve(Wind, itp, kTorque, windOmegaT, pwrOmegaT, n, omegaTflag)

        # add a fix for rated wind speed calculation inaccuracies kld 9/21/2012
        ratedWSflag = False
        # determine power curve after losses
        mtp = [None] * n
        for i in range(0, n):
            mtp[i] = itp[i]  # * self.drivetrain.getdrivetrain_efficiency(itp[i],self.ratedHubPower)
            # print [Wind[i],itp[i],self.drivetrain.getdrivetrain_efficiency(itp[i],self.ratedHubPower),mtp[i]] # for testing
            if mtp[i] > self.ratedPower:
                if not ratedWSflag:
                    ratedWSflag = True
                mtp[i] = self.ratedPower

        self.rated_wind_speed = self.ratedWindSpeed
        self.rated_rotor_speed = self.ratedRPM
        self.power_curve = np.array(mtp)
        self.wind_curve = Wind

        # compute turbine load outputs
        self.rotor_torque = self.ratedHubPower / (self.ratedRPM * (pi / 30.0)) * 1000.0
        self.rotor_thrust = (
            air_density * thrust_coefficient * pi * rotor_diameter**2 * (self.ratedWindSpeed**2) / 8.0
        )

    def idealPowerCurve(self, Wind, ITP, kTorque, windOmegaT, pwrOmegaT, n, omegaTflag):
        """
        Determine the ITP (idealized turbine power) array
        """

        idealPwr = 0.0

        for i in range(0, n):
            if (Wind[i] >= self.cutOutWS) or (Wind[i] <= self.cutInWS):
                idealPwr = 0.0  # cut out
            else:
                if omegaTflag:
                    if Wind[i] > windOmegaT:
                        idealPwr = (self.ratedHubPower - pwrOmegaT) / (self.ratedWindSpeed - windOmegaT) * (
                            Wind[i] - windOmegaT
                        ) + pwrOmegaT  # region 2.5
                    else:
                        idealPwr = (
                            kTorque * (Wind[i] * self.maxTipSpdRatio / (self.rotorDiam / 2.0)) ** 3 / 1000.0
                        )  # region 2
                else:
                    idealPwr = (
                        kTorque * (Wind[i] * self.maxTipSpdRatio / (self.rotorDiam / 2.0)) ** 3 / 1000.0
                    )  # region 2

            ITP[i] = idealPwr
            # print [Wind[i],ITP[i]]

        return


def weibull(X, K, L):
    """
    Return Weibull probability at speed X for distribution with k=K, c=L

    Parameters
    ----------
    X : float
       wind speed of interest [m/s]
    K : float
       Weibull shape factor for site
    L : float
       Weibull scale factor for site [m/s]

    Returns
    -------
    w : float
      Weibull pdf value
    """
    w = (K / L) * ((X / L) ** (K - 1)) * exp(-((X / L) ** K))
    return w


class aep_calc_csm(object):
    def __init__(self):
        # Variables
        # power_curve = Array(iotype='in', units='kW', desc='total power after drivetrain losses')
        # wind_curve = Array(iotype='in', units='m/s', desc='wind curve associated with power curve')
        # hub_height = Float(iotype='in', units = 'm', desc='hub height of wind turbine above ground / sea level')
        # shear_exponent = Float(iotype='in', desc= 'shear exponent for wind plant') #TODO - could use wind model here
        # wind_speed_50m = Float(iotype='in', units = 'm/s', desc='mean annual wind speed at 50 m height')
        # weibull_k= Float(iotype='in', desc = 'weibull shape factor for annual wind speed distribution')
        # machine_rating = Float(iotype='in', units='kW', desc='machine power rating')

        # Parameters
        # soiling_losses = Float(0.0, iotype='in', desc = 'energy losses due to blade soiling for the wind plant - average across turbines')
        # array_losses = Float(0.06, iotype='in', desc = 'energy losses due to turbine interactions - across entire plant')
        # availability = Float(0.94287630736, iotype='in', desc = 'average annual availbility of wind turbines at plant')
        # turbine_number = Int(100, iotype='in', desc = 'total number of wind turbines at the plant')

        # Output
        gross_aep = 0.0  # Float(iotype='out', desc='Gross Annual Energy Production before availability and loss impacts', unit='kWh')
        net_aep = 0.0  # Float(units= 'kW * h', iotype='out', desc='Annual energy production in kWh')  # use PhysicalUnits to set units='kWh'
        power_array = 0.0  # Array(iotype='out', units='kW', desc='total power after drivetrain losses')
        capacity_factor = 0.0  # Float(iotype='out', desc='plant capacity factor')

    def compute(
        self,
        power_curve,
        wind_curve,
        hub_height,
        shear_exponent,
        wind_speed_50m,
        weibull_k,
        machine_rating,
        soiling_losses,
        array_losses,
        availability,
        turbine_number,
    ):
        """
        Executes AEP Sub-module of the NREL _cost and Scaling Model by convolving a wind turbine power curve with a weibull distribution.
        It then discounts the resulting AEP for availability, plant and soiling losses.
        """

        power_array = np.array([wind_curve, power_curve])

        hubHeightWindSpeed = ((hub_height / 50) ** shear_exponent) * wind_speed_50m
        K = weibull_k
        L = hubHeightWindSpeed / exp(np.log(gamma(1.0 + 1.0 / K)))

        turbine_energy = 0.0
        for i in range(0, power_array.shape[1]):
            X = power_array[0, i]
            result = power_array[1, i] * weibull(X, K, L)
            turbine_energy += result

        ws_inc = power_array[0, 1] - power_array[0, 0]
        self.gross_aep = turbine_energy * 8760.0 * turbine_number * ws_inc
        self.net_aep = self.gross_aep * (1.0 - soiling_losses) * (1.0 - array_losses) * availability
        self.capacity_factor = self.net_aep / (8760 * machine_rating)


class drivetrain_csm(object):
    """drivetrain losses from NREL cost and scaling model"""

    def __init__(self, drivetrain_type="geared"):
        self.drivetrain_type = drivetrain_type

        power = np.zeros(161)  # Array(iotype='out', units='kW', desc='total power after drivetrain losses')

    def compute(self, aero_power, aero_torque, aero_thrust, rated_power):
        if self.drivetrain_type == "geared":
            constant = 0.01289
            linear = 0.08510
            quadratic = 0.0

        elif self.drivetrain_type == "single_stage":
            constant = 0.01331
            linear = 0.03655
            quadratic = 0.06107

        elif self.drivetrain_type == "multi_drive":
            constant = 0.01547
            linear = 0.04463
            quadratic = 0.05790

        elif self.drivetrain_type == "pm_direct_drive":
            constant = 0.01007
            linear = 0.02000
            quadratic = 0.06899

        Pbar0 = aero_power / rated_power

        # handle negative power case (with absolute value)
        Pbar1, dPbar1_dPbar0 = smooth_abs(Pbar0, dx=0.01)

        # truncate idealized power curve for purposes of efficiency calculation
        Pbar, dPbar_dPbar1, _ = smooth_min(Pbar1, 1.0, pct_offset=0.01)

        # compute efficiency
        eff = 1.0 - (constant / Pbar + linear + quadratic * Pbar)

        self.power = aero_power * eff

    def provideJ(self):
        # gradients
        dPbar_dPa = dPbar_dPbar1 * dPbar1_dPbar0 / rated_power
        dPbar_dPr = -dPbar_dPbar1 * dPbar1_dPbar0 * aero_power / rated_power**2

        deff_dPa = dPbar_dPa * (constant / Pbar**2 - quadratic)
        deff_dPr = dPbar_dPr * (constant / Pbar**2 - quadratic)

        dP_dPa = eff + aero_power * deff_dPa
        dP_dPr = aero_power * deff_dPr

        self.J = hstack([np.diag(dP_dPa), dP_dPr])

        return self.J


class aep_csm(object):
    def __init__(self, drivetrain_type="geared"):
        self.aero = aero_csm()
        self.drivetrain = drivetrain_csm(drivetrain_type)
        self.aep = aep_calc_csm()

    def compute(
        self,
        machine_rating,
        max_tip_speed,
        rotor_diameter,
        max_power_coefficient,
        opt_tsr,
        cut_in_wind_speed,
        cut_out_wind_speed,
        hub_height,
        altitude,
        air_density,
        max_efficiency,
        thrust_coefficient,
        soiling_losses,
        array_losses,
        availability,
        turbine_number,
        shear_exponent,
        wind_speed_50m,
        weibull_k,
    ):
        self.aero.compute(
            machine_rating,
            max_tip_speed,
            rotor_diameter,
            max_power_coefficient,
            opt_tsr,
            cut_in_wind_speed,
            cut_out_wind_speed,
            hub_height,
            altitude,
            air_density,
            max_efficiency,
            thrust_coefficient,
        )

        self.drivetrain.compute(self.aero.power_curve, self.aero.rotor_torque, self.aero.rotor_thrust, machine_rating)

        self.aep.compute(
            self.drivetrain.power,
            self.aero.wind_curve,
            hub_height,
            shear_exponent,
            wind_speed_50m,
            weibull_k,
            machine_rating,
            soiling_losses,
            array_losses,
            availability,
            turbine_number,
        )


# NREL Cost and Scaling Model cost modules
##################################################

# Turbine Capital Costs
##################################################

##### Rotor


class blades_csm(object):
    """
    object to wrap python code for NREL cost and scaling model for a wind turbine blade
    """

    def __init__(self):
        """
        OpenMDAO object to wrap blade model of the NREL _cost and Scaling Model (csmBlades.py)

        """
        super(blades_csm, self).__init__()

        # Outputs
        self.blade_cost = 0.0  # Float(0.0, units='USD', iotype='out', desc='cost for a single wind turbine blade')
        self.blade_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='mass for a single wind turbine blade')

    def compute(self, rotor_diameter, year=2009, month=12, advanced_blade=False):
        """
        computes Blade model of the NREL _cost and Scaling Model to estimate wind turbine blade cost and mass.
        """

        # Variables
        self.rotor_diameter = (
            rotor_diameter  # = Float(126.0, units = 'm', iotype='in', desc= 'rotor diameter of the machine')
        )

        # Parameters
        self.year = year  # = Int(2009, iotype='in', desc = 'year of project start')
        self.month = month  # Int(12, iotype='in', desc = 'month of project start')
        self.advanced_blade = (
            advanced_blade  # Bool(False, iotype='in', desc = 'boolean for use of advanced blade curve')
        )

        if self.advanced_blade == True:
            massCoeff = 0.4948
            massExp = 2.5300
        else:
            massCoeff = 0.1452
            massExp = 2.9158

        self.blade_mass = massCoeff * (self.rotor_diameter / 2.0) ** massExp

        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon

        ppi_labor = ppi.compute("IPPI_BLL")

        if self.advanced_blade == True:
            ref_yr = ppi.ref_yr
            ppi.ref_yr = 2003
            ppi_mat = ppi.compute("IPPI_BLA")
            ppi.ref_yr = ref_yr
            slopeR3 = 0.4019376
            intR3 = -21051.045983
        else:
            ppi_mat = ppi.compute("IPPI_BLD")
            slopeR3 = 0.4019376
            intR3 = -955.24267

        laborCoeff = 2.7445
        laborExp = 2.5025

        bladeCostCurrent = (
            (slopeR3 * (self.rotor_diameter / 2.0) ** 3.0 + (intR3)) * ppi_mat
            + (laborCoeff * (self.rotor_diameter / 2.0) ** laborExp) * ppi_labor
        ) / (1.0 - 0.28)
        self.blade_cost = bladeCostCurrent

        # derivatives
        self.d_mass_d_diameter = massExp * (massCoeff * (self.rotor_diameter / 2.0) ** (massExp - 1)) * (1 / 2.0)
        self.d_cost_d_diameter = (
            3.0 * (slopeR3 * (self.rotor_diameter / 2.0) ** 2.0) * ppi_mat * (1 / 2.0)
            + (laborExp * laborCoeff * (self.rotor_diameter / 2.0) ** (laborExp - 1)) * ppi_labor * (1 / 2.0)
        ) / (1.0 - 0.28)

    def list_deriv_vars(self):
        inputs = ["rotor_diameter"]
        outputs = ["blade_mass", "blade_cost"]

        return inputs, outputs

    def provideJ(self):
        self.J = np.array([[self.d_mass_d_diameter], [self.d_cost_d_diameter]])

        return self.J


class hub_csm(object):
    """
    object to wrap python code for NREL cost and scaling model for a wind turbine hub
    """

    def __init__(self):
        """
        OpenMDAO object to wrap hub model of the NREL _cost and Scaling Model (csmHub.py)
        """
        super(hub_csm, self).__init__()

        # Outputs
        self.hub_system_cost = 0.0  # Float(0.0, units='USD', iotype='out', desc='hub system cost')
        self.hub_system_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='hub system mass')
        self.hub_cost = 0.0  # Float(0.0, units='USD', iotype='out', desc='hub cost')
        self.hub_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='hub mass')
        self.pitch_system_cost = 0.0  # Float(0.0, units='USD', iotype='out', desc='pitch system cost')
        self.pitch_system_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='pitch system mass')
        self.spinner_cost = 0.0  # Float(0.0, units='USD', iotype='out', desc='spinner / nose cone cost')
        self.spinner_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='spinner / nose cone mass')

    def compute(self, rotor_diameter, blade_mass, year=2009, month=12, blade_number=3):
        """
        computes hub model of the NREL _cost and Scaling model to compute hub system object masses and costs.
        """

        # Variables
        self.rotor_diameter = (
            rotor_diameter  # Float(126.0, units = 'm', iotype='in', desc= 'rotor diameter of the machine')
        )
        self.blade_mass = blade_mass  # Float(17650.67, units='kg', iotype='in', desc='mass of an individual blade')

        # Parameters
        self.year = year  # Int(2009, iotype='in', desc = 'year of project start')
        self.month = month  # Int(12, iotype='in', desc = 'month of project start')
        self.blade_number = blade_number  # Int(3, iotype='in', desc= 'number of rotor blades')

        # *** Pitch bearing and mechanism
        pitchBearingMass = 0.1295 * self.blade_mass * self.blade_number + 491.31  # slope*BldMass3 + int
        bearingHousingPct = 32.80 / 100.0
        massSysOffset = 555.0
        self.pitch_system_mass = pitchBearingMass * (1 + bearingHousingPct) + massSysOffset

        # *** Hub
        self.hub_mass = 0.95402537 * self.blade_mass + 5680.272238

        # *** NoseCone/Spinner
        self.spinner_mass = 18.5 * self.rotor_diameter + (-520.5)  # GNS

        self.hub_system_mass = self.hub_mass + self.pitch_system_mass + self.spinner_mass

        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon

        # *** Pitch bearing and mechanism
        bearingCost = 0.2106 * self.rotor_diameter**2.6576
        bearingCostEscalator = ppi.compute("IPPI_PMB")
        self.pitch_system_cost = bearingCostEscalator * (bearingCost + bearingCost * 1.28)

        # *** Hub
        hubCost2002 = self.hub_mass * 4.25  # $/kg
        hubCostEscalator = ppi.compute("IPPI_HUB")
        self.hub_cost = hubCost2002 * hubCostEscalator

        # *** NoseCone/Spinner
        spinnerCostEscalator = ppi.compute("IPPI_NAC")
        self.spinner_cost = spinnerCostEscalator * (5.57 * self.spinner_mass)

        self.hub_system_cost = self.hub_cost + self.pitch_system_cost + self.spinner_cost

        # derivatives
        self.d_hub_mass_d_diameter = 0.0
        self.d_pitch_mass_d_diameter = 0.0
        self.d_spinner_mass_d_diameter = 18.5
        self.d_system_mass_d_diameter = (
            self.d_hub_mass_d_diameter + self.d_pitch_mass_d_diameter + self.d_spinner_mass_d_diameter
        )

        self.d_hub_cost_d_diameter = 0.0
        self.d_pitch_cost_d_diameter = bearingCostEscalator * 2.28 * 2.6576 * (0.2106 * self.rotor_diameter**1.6576)
        self.d_spinner_cost_d_diameter = spinnerCostEscalator * (5.57 * self.d_spinner_mass_d_diameter)
        self.d_system_cost_d_diameter = (
            self.d_hub_cost_d_diameter + self.d_pitch_cost_d_diameter + self.d_spinner_cost_d_diameter
        )

        self.d_hub_mass_d_blade_mass = 0.95402537
        self.d_pitch_mass_d_blade_mass = 0.1295 * self.blade_number * (1 + bearingHousingPct)
        self.d_spinner_mass_d_blade_mass = 0.0
        self.d_system_mass_d_blade_mass = (
            self.d_hub_mass_d_blade_mass + self.d_pitch_mass_d_blade_mass + self.d_spinner_mass_d_blade_mass
        )

        self.d_hub_cost_d_blade_mass = self.d_hub_mass_d_blade_mass * 4.25 * hubCostEscalator
        self.d_pitch_cost_d_blade_mass = 0.0
        self.d_spinner_cost_d_blade_mass = 0.0
        self.d_system_cost_d_blade_mass = (
            self.d_hub_cost_d_blade_mass + self.d_pitch_cost_d_blade_mass + self.d_spinner_cost_d_blade_mass
        )

    def list_deriv_vars(self):
        inputs = ["rotor_diameter", "blade_mass"]
        outputs = [
            "hub_mass",
            "pitch_system_mass",
            "spinner_mass",
            "hub_system_mass",
            "hub_cost",
            "pitch_system_cost",
            "spinner_cost",
            "hub_system_cost",
        ]

        return inputs, outputs

    def provideJ(self):
        self.J = np.array(
            [
                [self.d_hub_mass_d_diameter, self.d_hub_mass_d_blade_mass],
                [self.d_pitch_mass_d_diameter, self.d_pitch_mass_d_blade_mass],
                [self.d_spinner_mass_d_diameter, self.d_spinner_mass_d_blade_mass],
                [self.d_system_mass_d_diameter, self.d_system_mass_d_blade_mass],
                [self.d_hub_cost_d_diameter, self.d_hub_cost_d_blade_mass],
                [self.d_pitch_cost_d_diameter, self.d_pitch_cost_d_blade_mass],
                [self.d_spinner_cost_d_diameter, self.d_spinner_cost_d_blade_mass],
                [self.d_system_cost_d_diameter, self.d_system_cost_d_blade_mass],
            ]
        )

        return self.J


##### Nacelle


class nacelle_csm(object):
    """
    object to wrap python code for NREL cost and scaling model for a wind turbine nacelle
    """

    def __init__(self):
        """
        OpenMDAO object to wrap nacelle mass-cost model based on the NREL _cost and Scaling model data (csmNacelle.py).
        """
        super(nacelle_csm, self).__init__()

        # Outputs
        self.nacelle_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='nacelle mass')
        self.lowSpeedShaft_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'low speed shaft mass')
        self.bearings_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'bearings system mass')
        self.gearbox_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'gearbox and housing mass')
        self.mechanicalBrakes_mass = (
            0.0  # Float(0.0, units='kg', iotype='out', desc= 'high speed shaft, coupling, and mechanical brakes mass')
        )
        self.generator_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'generator and housing mass')
        self.VSElectronics_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'variable speed electronics mass')
        self.yawSystem_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'yaw system mass')
        self.mainframeTotal_mass = (
            0.0  # Float(0.0, units='kg', iotype='out', desc= 'mainframe total mass including bedplate')
        )
        self.electronicCabling_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'electronic cabling mass')
        self.HVAC_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'HVAC system mass')
        self.nacelleCover_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'nacelle cover mass')
        self.controls_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'control system mass')

        self.nacelle_cost = 0.0  # Float(0.0, units='USD', iotype='out', desc='nacelle cost')
        self.lowSpeedShaft_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'low speed shaft _cost')
        self.bearings_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'bearings system _cost')
        self.gearbox_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'gearbox and housing _cost')
        self.mechanicalBrakes_cost = (
            0.0  # Float(0.0, units='kg', iotype='out', desc= 'high speed shaft, coupling, and mechanical brakes _cost')
        )
        self.generator_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'generator and housing _cost')
        self.VSElectronics_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'variable speed electronics _cost')
        self.yawSystem_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'yaw system _cost')
        self.mainframeTotal_cost = (
            0.0  # Float(0.0, units='kg', iotype='out', desc= 'mainframe total _cost including bedplate')
        )
        self.electronicCabling_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'electronic cabling _cost')
        self.HVAC_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'HVAC system _cost')
        self.nacelleCover_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'nacelle cover _cost')
        self.controls_cost = 0.0  # Float(0.0, units='kg', iotype='out', desc= 'control system _cost')

    def compute(
        self,
        rotor_diameter,
        rotor_mass,
        rotor_thrust,
        rotor_torque,
        machine_rating,
        drivetrain_design="geared",
        crane=True,
        advanced_bedplate=0,
        year=2009,
        month=12,
        offshore=True,
    ):
        """
        compute nacelle model of the NREL _cost and Scaling Model.
        """

        # Variables
        self.rotor_diameter = rotor_diameter  # = Float(126.0, units='m', iotype='in', desc = 'diameter of the rotor')
        self.rotor_mass = (
            rotor_mass  # Float(123193.3010, iotype='in', units='kg', desc = 'mass of rotor including blades and hub')
        )
        self.rotor_thrust = rotor_thrust  # Float(500930.0837, iotype='in', units='N', desc='maximum thurst from rotor')
        self.rotor_torque = (
            rotor_torque  # Float(4365248.7375, iotype='in', units='N * m', desc = 'torque from rotor at rated power')
        )
        self.machine_rating = machine_rating  # Float(5000.0, units='kW', iotype='in', desc = 'Machine rated power')

        # Parameters
        self.drivetrain_design = drivetrain_design  # Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
        self.crane = crane  # Bool(True, iotype='in', desc = 'boolean for presence of a service crane up tower')
        self.advanced_bedplate = (
            advanced_bedplate  # Int(0, iotype='in', desc= 'indicator for drivetrain bedplate design 0 - conventional')
        )
        self.year = year  # Int(2009, iotype='in', desc = 'year of project start')
        self.month = month  # Int(12, iotype='in', desc = 'month of project start')
        self.offshore = offshore  # Bool(True, iotype='in', desc = 'boolean for land or offshore wind project')

        # basic variable initialization
        if self.offshore == False:
            offshore = 0
        else:
            offshore = 1

        ppi.curr_yr = self.year
        ppi.curr_mon = self.month

        # Low Speed Shaft
        lenShaft = 0.03 * self.rotor_diameter
        mmtArm = lenShaft / 5
        bendLoad = 1.25 * 9.81 * self.rotor_mass
        bendMom = bendLoad * mmtArm
        hFact = 0.1
        hollow = 1 / (1 - (hFact) ** 4)
        outDiam = (
            (32.0 / np.pi)
            * hollow
            * 3.25
            * ((self.rotor_torque * 3.0 / 371000000.0) ** 2 + (bendMom / 71070000) ** 2) ** (0.5)
        ) ** (1.0 / 3.0)
        inDiam = outDiam * hFact

        self.lowSpeedShaft_mass = 1.25 * (np.pi / 4) * (outDiam**2 - inDiam**2) * lenShaft * 7860

        LowSpeedShaftCost2002 = 0.0998 * self.rotor_diameter**2.8873
        lssCostEsc = ppi.compute("IPPI_LSS")

        self.lowSpeedShaft_cost = LowSpeedShaftCost2002 * lssCostEsc

        d_mass_d_outD = 1.25 * (np.pi / 4) * (1 - 0.1**2) * 2 * outDiam * lenShaft * 7860
        d_outD_mult = (
            ((32.0 / np.pi) * hollow * 3.25) ** (1.0 / 3.0)
            * (1.0 / 6.0)
            * ((self.rotor_torque * 3.0 / 371000000.0) ** 2 + (bendMom / 71070000.0) ** 2) ** (-5.0 / 6.0)
        )
        d_outD_d_diameter = d_outD_mult * 2.0 * (bendMom / 71070000) * (1.0 / 71070000.0) * (bendLoad * 0.03 / 5)
        d_outD_d_mass = d_outD_mult * 2.0 * (bendMom / 71070000) * (1.0 / 71070000.0) * (mmtArm * 1.25 * 9.81)
        d_outD_d_torque = d_outD_mult * 2.0 * (self.rotor_torque * 3.0 / 371000000.0) * (3.0 / 371000000.0)
        self.d_lss_mass_d_r_diameter = (
            d_mass_d_outD * d_outD_d_diameter + 1.25 * (np.pi / 4) * (outDiam**2 - inDiam**2) * 7860 * 0.03
        )
        self.d_lss_mass_d_r_mass = d_mass_d_outD * d_outD_d_mass
        self.d_lss_mass_d_r_torque = d_mass_d_outD * d_outD_d_torque

        self.d_lss_cost_d_r_diameter = lssCostEsc * 2.8873 * 0.0998 * self.rotor_diameter**1.8873

        # Gearbox
        costCoeff = [None, 16.45, 74.101, 15.25697015, 0]
        costExp = [None, 1.2491, 1.002, 1.2491, 0]
        massCoeff = [None, 65.601, 81.63967335, 129.1702924, 0]
        massExp = [None, 0.759, 0.7738, 0.7738, 0]

        if self.drivetrain_design == "geared":
            drivetrain_design = 1
        elif self.drivetrain_design == "single_stage":
            drivetrain_design = 2
        elif self.drivetrain_design == "multi-drive":
            drivetrain_design = 3
        elif self.drivetrain_design == "pm_direct_drive":
            drivetrain_design = 4

        self.gearbox_mass = massCoeff[drivetrain_design] * (self.rotor_torque / 1000) ** massExp[drivetrain_design]

        gearboxCostEsc = ppi.compute("IPPI_GRB")
        Gearbox2002 = costCoeff[drivetrain_design] * self.machine_rating ** costExp[drivetrain_design]
        self.gearbox_cost = Gearbox2002 * gearboxCostEsc

        if drivetrain_design == 4:
            self.d_gearbox_mass_d_r_torque = 0.0
            self.d_gearbox_cost_d_rating = 0.0
        else:
            self.d_gearbox_mass_d_r_torque = (
                massExp[drivetrain_design]
                * massCoeff[drivetrain_design]
                * ((self.rotor_torque / 1000.0) ** (massExp[drivetrain_design] - 1))
                * (1 / 1000.0)
            )
            self.d_gearbox_cost_d_rating = (
                gearboxCostEsc
                * costExp[drivetrain_design]
                * costCoeff[drivetrain_design]
                * self.machine_rating ** (costExp[drivetrain_design] - 1)
            )

        # Generator
        costCoeff = [None, 65.000, 54.72533, 48.02963, 219.3333]  # $/kW - from 'Generators' worksheet
        massCoeff = [None, 6.4737, 10.50972, 5.343902, 37.68400]
        massExp = [None, 0.9223, 0.922300, 0.922300, 1.000000]

        if drivetrain_design < 4:
            self.generator_mass = massCoeff[drivetrain_design] * self.machine_rating ** massExp[drivetrain_design]
        else:  # direct drive
            self.generator_mass = massCoeff[drivetrain_design] * self.rotor_torque ** massExp[drivetrain_design]

        generatorCostEsc = ppi.compute("IPPI_GEN")
        GeneratorCost2002 = costCoeff[drivetrain_design] * self.machine_rating
        self.generator_cost = GeneratorCost2002 * generatorCostEsc

        if drivetrain_design < 4:
            self.d_generator_mass_d_r_torque = 0.0
            self.d_generator_mass_d_rating = (
                massExp[drivetrain_design]
                * massCoeff[drivetrain_design]
                * self.machine_rating ** (massExp[drivetrain_design] - 1)
            )
        else:
            self.d_generator_mass_d_r_torque = (
                massExp[drivetrain_design]
                * massCoeff[drivetrain_design]
                * self.rotor_torque ** (massExp[drivetrain_design] - 1)
            )
            self.d_generator_mass_d_rating = 0.0
        self.d_generator_cost_d_rating = generatorCostEsc * costCoeff[drivetrain_design]

        # Rest of the system

        # --- electrical connections
        self.electronicCabling_mass = 0.0

        # --- bearings
        self.bearings_mass = 0.00012266667 * (self.rotor_diameter**3.5) - 0.00030360 * (self.rotor_diameter**2.5)
        HousingMass = self.bearings_mass
        self.bearings_mass += HousingMass

        self.d_bearings_mass_d_r_diameter = 2 * (
            3.5 * 0.00012266667 * (self.rotor_diameter**2.5) - 0.00030360 * 2.5 * (self.rotor_diameter**1.5)
        )

        # --- mechanical brake
        mechBrakeCost2002 = 1.9894 * self.machine_rating + (-0.1141)
        self.mechanicalBrakes_mass = mechBrakeCost2002 * 0.10

        self.d_brakes_mass_d_rating = 0.10 * 1.9894

        # --- variable-speed electronics
        self.VSElectronics_mass = 0.0

        # --- yaw drive bearings
        self.yawSystem_mass = 1.6 * (0.0009 * self.rotor_diameter**3.314)

        self.d_yaw_mass_d_r_diameter = 3.314 * 1.6 * (0.0009 * self.rotor_diameter**2.314)

        # --- hydraulics, cooling
        self.HVAC_mass = 0.08 * self.machine_rating

        self.d_hvac_mass_d_rating = 0.08

        # --- bedplate ---
        if self.advanced_bedplate == 0:  # not an actual option in cost and scaling model
            BedplateWeightFac = 2.86  # modular
        elif self.advanced_bedplate == 1:  # test for mod-adv
            BedplateWeightFac = 2.40  # modular-advanced
        else:
            BedplateWeightFac = 0.71  # advanced

        # These RD functions from spreadsheet don't quite form a continuous composite function
        """if (self.rotor_diameter <= 15.0): # Removing for gradients - assuming large turbines only
            TowerTopDiam = 0.3
        elif (self.rotor_diameter <= 60.0):
            TowerTopDiam = (0.07042*self.rotor_diameter-0.715)
        else:"""
        TowerTopDiam = (12.29 * self.rotor_diameter + 2648) / 1000

        MassFromTorque = BedplateWeightFac * 0.00368 * self.rotor_torque
        MassFromThrust = 0.00158 * BedplateWeightFac * self.rotor_thrust * TowerTopDiam
        MassFromRotorWeight = 0.015 * BedplateWeightFac * self.rotor_mass * TowerTopDiam

        # Bedplate(Length|Area) added by GNS
        BedplateLength = 1.5874 * 0.052 * self.rotor_diameter
        BedplateArea = 0.5 * BedplateLength * BedplateLength
        MassFromArea = 100 * BedplateWeightFac * BedplateArea

        # mfmCoeff[1,4] for different drivetrain configurations
        mfmCoeff = [None, 22448, 1.29490, 1.72080, 22448]
        mfmExp = [None, 0, 1.9525, 1.9525, 0]

        # --- nacelle totals
        TotalMass = MassFromTorque + MassFromThrust + MassFromRotorWeight + MassFromArea

        if (drivetrain_design == 1) or (drivetrain_design == 4):
            self.bedplate_mass = TotalMass
        else:
            self.bedplate_mass = mfmCoeff[drivetrain_design] * (self.rotor_diameter ** mfmExp[drivetrain_design])

        NacellePlatformsMass = 0.125 * self.bedplate_mass

        # --- crane ---
        if self.crane:
            self.crane_mass = 3000.0
        else:
            self.crane_mass = 0.0

        # --- main frame ---
        self.mainframeTotal_mass = self.bedplate_mass + NacellePlatformsMass + self.crane_mass

        if (drivetrain_design == 1) or (drivetrain_design == 4):
            self.d_mainframe_mass_d_r_diameter = 1.125 * (
                (
                    (0.00158 * BedplateWeightFac * self.rotor_thrust * (12.29 / 1000.0))
                    + (0.015 * BedplateWeightFac * self.rotor_mass * (12.29 / 1000.0))
                    + (100 * BedplateWeightFac * 0.5 * (1.5874 * 0.052) ** 2.0 * (2 * self.rotor_diameter))
                )
            )
            self.d_mainframe_mass_d_r_mass = 1.125 * (0.015 * BedplateWeightFac * TowerTopDiam)
            self.d_mainframe_mass_d_r_thrust = 1.125 * (0.00158 * BedplateWeightFac * TowerTopDiam)
            self.d_mainframe_mass_d_r_torque = 1.125 * BedplateWeightFac * 0.00368
        else:
            self.d_mainframe_mass_d_r_diameter = (
                1.125
                * mfmCoeff[drivetrain_design]
                * (mfmExp[drivetrain_design] * self.rotor_diameter ** (mfmExp[drivetrain_design] - 1))
            )
            self.d_mainframe_mass_d_r_mass = 0.0
            self.d_mainframe_mass_d_r_thrust = 0.0
            self.d_mainframe_mass_d_r_torque = 0.0

        # --- nacelle cover ---
        nacelleCovCost2002 = 11.537 * self.machine_rating + (3849.7)
        self.nacelleCover_mass = nacelleCovCost2002 * 0.111111

        self.d_cover_mass_d_rating = 0.111111 * 11.537

        # --- control system ---
        self.controls_mass = 0.0

        # overall mass
        self.nacelle_mass = (
            self.lowSpeedShaft_mass
            + self.bearings_mass
            + self.gearbox_mass
            + self.mechanicalBrakes_mass
            + self.generator_mass
            + self.VSElectronics_mass
            + self.yawSystem_mass
            + self.mainframeTotal_mass
            + self.electronicCabling_mass
            + self.HVAC_mass
            + self.nacelleCover_mass
            + self.controls_mass
        )

        self.d_nacelle_mass_d_r_diameter = (
            self.d_lss_mass_d_r_diameter
            + self.d_bearings_mass_d_r_diameter
            + self.d_yaw_mass_d_r_diameter
            + self.d_mainframe_mass_d_r_diameter
        )
        self.d_nacelle_mass_d_r_mass = self.d_lss_mass_d_r_mass + self.d_mainframe_mass_d_r_mass
        self.d_nacelle_mass_d_r_thrust = self.d_mainframe_mass_d_r_thrust
        self.d_nacelle_mass_d_r_torque = (
            self.d_lss_mass_d_r_torque
            + self.d_gearbox_mass_d_r_torque
            + self.d_generator_mass_d_r_torque
            + self.d_mainframe_mass_d_r_torque
        )
        self.d_nacelle_mass_d_rating = (
            self.d_generator_mass_d_rating
            + self.d_brakes_mass_d_rating
            + self.d_hvac_mass_d_rating
            + self.d_cover_mass_d_rating
        )

        # Rest of System Costs
        # Cost Escalators - obtained from ppi tables
        bearingCostEsc = ppi.compute("IPPI_BRN")
        mechBrakeCostEsc = ppi.compute("IPPI_BRK")
        VspdEtronicsCostEsc = ppi.compute("IPPI_VSE")
        yawDrvBearingCostEsc = ppi.compute("IPPI_YAW")
        nacelleCovCostEsc = ppi.compute("IPPI_NAC")
        hydrCoolingCostEsc = ppi.compute("IPPI_HYD")
        mainFrameCostEsc = ppi.compute("IPPI_MFM")
        econnectionsCostEsc = ppi.compute("IPPI_ELC")

        # These RD functions from spreadsheet don't quite form a continuous composite function

        # --- electrical connections
        self.electronicCabling_cost = 40.0 * self.machine_rating  # 2002
        self.electronicCabling_cost *= econnectionsCostEsc

        self.d_electronics_cost_d_rating = 40.0 * econnectionsCostEsc

        # --- bearings
        bearingMass = 0.00012266667 * (self.rotor_diameter**3.5) - 0.00030360 * (self.rotor_diameter**2.5)
        HousingMass = bearingMass
        brngSysCostFactor = 17.6  # $/kg
        Bearings2002 = bearingMass * brngSysCostFactor
        Housing2002 = HousingMass * brngSysCostFactor
        self.bearings_cost = (Bearings2002 + Housing2002) * bearingCostEsc

        self.d_bearings_cost_d_r_diameter = bearingCostEsc * brngSysCostFactor * self.d_bearings_mass_d_r_diameter

        # --- mechanical brake
        mechBrakeCost2002 = 1.9894 * self.machine_rating + (-0.1141)
        self.mechanicalBrakes_cost = mechBrakeCostEsc * mechBrakeCost2002

        self.d_brakes_cost_d_rating = mechBrakeCostEsc * 1.9894

        # --- variable-speed electronics
        VspdEtronics2002 = 79.32 * self.machine_rating
        self.VSElectronics_cost = VspdEtronics2002 * VspdEtronicsCostEsc

        self.d_vselectronics_cost_d_rating = VspdEtronicsCostEsc * 79.32

        # --- yaw drive bearings
        YawDrvBearing2002 = 2 * (0.0339 * self.rotor_diameter**2.9637)
        self.yawSystem_cost = YawDrvBearing2002 * yawDrvBearingCostEsc

        self.d_yaw_cost_d_r_diameter = yawDrvBearingCostEsc * 2 * 2.9637 * (0.0339 * self.rotor_diameter**1.9637)

        # --- hydraulics, cooling
        self.HVAC_cost = 12.0 * self.machine_rating  # 2002
        self.HVAC_cost *= hydrCoolingCostEsc

        self.d_hvac_cost_d_rating = hydrCoolingCostEsc * 12.0

        # --- control system ---
        initControlCost = [35000, 55900]  # land, off-shore
        self.controls_cost = initControlCost[offshore] * ppi.compute("IPPI_CTL")

        # --- nacelle totals
        NacellePlatforms2002 = 8.7 * NacellePlatformsMass

        # --- nacelle cover ---
        nacelleCovCost2002 = 11.537 * self.machine_rating + (3849.7)
        self.nacelleCover_cost = nacelleCovCostEsc * nacelleCovCost2002

        self.d_cover_cost_d_rating = nacelleCovCostEsc * 11.537

        # --- crane ---

        if self.crane:
            self.crane_cost = 12000.0
        else:
            self.crane_cost = 0.0

        # --- main frame ---
        # mfmCoeff[1,4] for different drivetrain configurations
        mfmCoeff = [None, 9.4885, 303.96, 17.923, 627.28]
        mfmExp = [None, 1.9525, 1.0669, 1.6716, 0.8500]

        MainFrameCost2002 = mfmCoeff[drivetrain_design] * self.rotor_diameter ** mfmExp[drivetrain_design]
        BaseHardware2002 = MainFrameCost2002 * 0.7
        MainFrame2002 = MainFrameCost2002 + NacellePlatforms2002 + self.crane_cost + BaseHardware2002  # service crane
        self.mainframeTotal_cost = MainFrame2002 * mainFrameCostEsc

        self.d_mainframe_cost_d_r_diameter = mainFrameCostEsc * (
            1.7
            * mfmCoeff[drivetrain_design]
            * mfmExp[drivetrain_design]
            * self.rotor_diameter ** (mfmExp[drivetrain_design] - 1)
            + 8.7 * self.d_mainframe_mass_d_r_diameter * (0.125 / 1.125)
        )
        self.d_mainframe_cost_d_r_mass = mainFrameCostEsc * 8.7 * self.d_mainframe_mass_d_r_mass * (0.125 / 1.125)
        self.d_mainframe_cost_d_r_thrust = mainFrameCostEsc * 8.7 * self.d_mainframe_mass_d_r_thrust * (0.125 / 1.125)
        self.d_mainframe_cost_d_r_torque = mainFrameCostEsc * 8.7 * self.d_mainframe_mass_d_r_torque * (0.125 / 1.125)

        # overall system cost
        self.nacelle_cost = (
            self.lowSpeedShaft_cost
            + self.bearings_cost
            + self.gearbox_cost
            + self.mechanicalBrakes_cost
            + self.generator_cost
            + self.VSElectronics_cost
            + self.yawSystem_cost
            + self.mainframeTotal_cost
            + self.electronicCabling_cost
            + self.HVAC_cost
            + self.nacelleCover_cost
            + self.controls_cost
        )

        self.d_nacelle_cost_d_r_diameter = (
            self.d_lss_cost_d_r_diameter
            + self.d_bearings_cost_d_r_diameter
            + self.d_yaw_cost_d_r_diameter
            + self.d_mainframe_cost_d_r_diameter
        )
        self.d_nacelle_cost_d_r_mass = self.d_mainframe_cost_d_r_mass
        self.d_nacelle_cost_d_r_thrust = self.d_mainframe_cost_d_r_thrust
        self.d_nacelle_cost_d_r_torque = self.d_mainframe_cost_d_r_torque
        self.d_nacelle_cost_d_rating = (
            self.d_gearbox_cost_d_rating
            + self.d_generator_cost_d_rating
            + self.d_brakes_cost_d_rating
            + self.d_hvac_cost_d_rating
            + self.d_cover_cost_d_rating
            + self.d_electronics_cost_d_rating
            + self.d_vselectronics_cost_d_rating
        )

    def list_deriv_vars(self):
        inputs = ["rotor_diameter", "rotor_mass", "rotor_thrust", "rotor_torque", "machine_rating"]
        outputs = [
            "nacelle_mass",
            "lowSpeedShaft_mass",
            "bearings_mass",
            "gearbox_mass",
            "generator_mass",
            "mechanicalBrakes_mass",
            "yawSystem_mass",
            "electronicCabling_mass",
            "HVAC_mass",
            "VSElectronics_mass",
            "mainframeTotal_mass",
            "nacelleCover_mass",
            "controls_mass",
            "nacelle_cost",
            "lowSpeedShaft_cost",
            "bearings_cost",
            "gearbox_cost",
            "generator_cost",
            "mechanicalBrakes_cost",
            "yawSystem_cost",
            "electronicCabling_cost",
            "HVAC_cost",
            "VSElectronics_cost",
            "mainframeTotal_cost",
            "nacelleCover_cost",
            "controls_cost",
        ]

        return inputs, outputs

    def provideJ(self):
        self.J = np.array(
            [
                [
                    self.d_nacelle_mass_d_r_diameter,
                    self.d_nacelle_mass_d_r_mass,
                    self.d_nacelle_mass_d_r_thrust,
                    self.d_nacelle_mass_d_r_torque,
                    self.d_nacelle_mass_d_rating,
                ],
                [self.d_lss_mass_d_r_diameter, self.d_lss_mass_d_r_mass, 0.0, self.d_lss_mass_d_r_torque, 0.0],
                [self.d_bearings_mass_d_r_diameter, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, self.d_gearbox_mass_d_r_torque, 0.0],
                [0.0, 0.0, 0.0, self.d_generator_mass_d_r_torque, self.d_generator_mass_d_rating],
                [0.0, 0.0, 0.0, 0.0, self.d_brakes_mass_d_rating],
                [self.d_yaw_mass_d_r_diameter, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.d_hvac_mass_d_rating],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    self.d_mainframe_mass_d_r_diameter,
                    self.d_mainframe_mass_d_r_mass,
                    self.d_mainframe_mass_d_r_thrust,
                    self.d_mainframe_mass_d_r_torque,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, self.d_cover_mass_d_rating],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    self.d_nacelle_cost_d_r_diameter,
                    self.d_nacelle_cost_d_r_mass,
                    self.d_nacelle_cost_d_r_thrust,
                    self.d_nacelle_cost_d_r_torque,
                    self.d_nacelle_cost_d_rating,
                ],
                [self.d_lss_cost_d_r_diameter, 0.0, 0.0, 0.0, 0.0],
                [self.d_bearings_cost_d_r_diameter, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.d_gearbox_cost_d_rating],
                [0.0, 0.0, 0.0, 0.0, self.d_generator_cost_d_rating],
                [0.0, 0.0, 0.0, 0.0, self.d_brakes_cost_d_rating],
                [self.d_yaw_cost_d_r_diameter, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.d_electronics_cost_d_rating],
                [0.0, 0.0, 0.0, 0.0, self.d_hvac_cost_d_rating],
                [0.0, 0.0, 0.0, 0.0, self.d_vselectronics_cost_d_rating],
                [
                    self.d_mainframe_cost_d_r_diameter,
                    self.d_mainframe_cost_d_r_mass,
                    self.d_mainframe_cost_d_r_thrust,
                    self.d_mainframe_cost_d_r_torque,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, self.d_cover_cost_d_rating],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        return self.J


##### Tower


class tower_csm(object):
    """
    object to wrap python code for NREL cost and scaling model for a wind turbine tower
    """

    def __init__(self):
        """
        OpenMDAO object to wrap tower model based of the NREL _cost and Scaling Model data (csmTower.py).
        """
        super(tower_csm, self).__init__()

        # Outputs
        self.tower_cost = 0.0  # Float(0.0, units='USD', iotype='out', desc='cost for a tower')
        self.tower_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='mass for a turbine tower')

    def compute(self, rotor_diameter, hub_height, year=2009, month=12, advanced_tower=False):
        """
        computes the tower model of the NREL _cost and Scaling Model.
        """

        # Variables
        self.rotor_diameter = (
            rotor_diameter  # Float(126.0, units = 'm', iotype='in', desc= 'rotor diameter of the machine')
        )
        self.hub_height = hub_height  # Float(90.0, units = 'm', iotype='in', desc = 'hub height of machine')

        # Parameters
        self.year = year  # Int(2009, iotype='in', desc = 'year of project start')
        self.month = month  # Int(12, iotype='in', desc = 'month of project start')
        self.advanced_tower = advanced_tower  # Bool(False, iotype='in', desc = 'advanced tower configuration')

        windpactMassSlope = 0.397251147546925
        windpactMassInt = -1414.381881

        if self.advanced_tower:
            windpactMassSlope = 0.269380169
            windpactMassInt = 1779.328183

        self.tower_mass = (
            windpactMassSlope * np.pi * (self.rotor_diameter / 2.0) ** 2 * self.hub_height + windpactMassInt
        )

        ppi.curr_yr = curr_yr
        ppi.curr_mon = curr_mon

        twrCostEscalator = 1.5944
        twrCostEscalator = ppi.compute("IPPI_TWR")
        twrCostCoeff = 1.5  # $/kg

        self.towerCost2002 = self.tower_mass * twrCostCoeff
        self.tower_cost = self.towerCost2002 * twrCostEscalator

        # derivatives
        self.d_mass_d_diameter = (
            2 * windpactMassSlope * np.pi * (self.rotor_diameter / 2.0) * (1 / 2.0) * self.hub_height
        )
        self.d_mass_d_hheight = windpactMassSlope * np.pi * (self.rotor_diameter / 2.0) ** 2
        self.d_cost_d_diameter = twrCostCoeff * twrCostEscalator * self.d_mass_d_diameter
        self.d_cost_d_hheight = twrCostCoeff * twrCostEscalator * self.d_mass_d_hheight

    def list_deriv_vars(self):
        inputs = ["rotor_diameter", "hub_height"]
        outputs = ["tower_mass", "tower_cost"]

        return inputs, outputs

    def provideJ(self):
        self.J = np.array(
            [[self.d_mass_d_diameter, self.d_mass_d_hheight], [self.d_cost_d_diameter, self.d_cost_d_hheight]]
        )

        return self.J


##### Turbine


# -------------------------------------------------------
# Rotor mass adder
class rotor_mass_adder(object):
    def __init__(self):
        super(rotor_mass_adder, self).__init__()

        # Outputs
        self.rotor_mass = 0.0  # Float(units='kg', iotype='out', desc= 'overall rotor mass')

    def compute(self, blade_mass, hub_system_mass, blade_number=3):
        # Variables
        self.blade_mass = blade_mass  # Float(0.0, units='kg', iotype='in', desc='mass for a single wind turbine blade')
        self.hub_system_mass = hub_system_mass  # Float(0.0, units='kg', iotype='in', desc='hub system mass')

        # Parameters
        self.blade_number = blade_number  # Int(3, iotype='in', desc='blade numebr')

        self.rotor_mass = self.blade_mass * self.blade_number + self.hub_system_mass

        self.d_mass_d_blade_mass = self.blade_number
        self.d_mass_d_hub_mass = 1.0

    def list_deriv_vars(self):
        inputs = ["blade_mass", "hub_system_mass"]
        outputs = ["rotor_mass"]

        return inputs, outputs

    def provideJ(self):
        self.J = np.array([[self.d_mass_d_blade_mass, self.d_mass_d_hub_mass]])

        return self.J


# ------------------------------------------------------------------
class turbine_csm(object):
    def __init__(self):
        super(turbine_csm, self).__init__()

        # Outputs
        self.rotor_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='rotor mass')
        self.rotor_cost = 0.0  # Float(0.0, iotype='out', desc='rotor cost')
        self.turbine_mass = 0.0  # Float(0.0, units='kg', iotype='out', desc='turbine mass')
        self.turbine_cost = (
            0.0  # Float(0.0, iotype='out', desc='Overall wind turbine capial costs including transportation costs')
        )

    def compute(
        self,
        blade_cost,
        blade_mass,
        hub_system_cost,
        hub_system_mass,
        nacelle_mass,
        nacelle_cost,
        tower_cost,
        tower_mass,
        blade_number=3,
        offshore=True,
    ):
        """
        compute Turbine Capital _costs Model of the NREL _cost and Scaling Model.
        """

        # Variables
        self.blade_cost = (
            blade_cost  # Float(0.0, units='USD', iotype='in', desc='cost for a single wind turbine blade')
        )
        self.blade_mass = blade_mass  # Float(0.0, units='kg', iotype='in', desc='mass for a single wind turbine blade')
        self.hub_system_cost = hub_system_cost  # Float(0.0, units='USD', iotype='in', desc='hub system cost')
        self.hub_system_mass = hub_system_mass  # Float(0.0, units='kg', iotype='in', desc='hub system mass')
        self.nacelle_mass = nacelle_mass  # Float(0.0, units='kg', iotype='in', desc='nacelle mass')
        self.nacelle_cost = nacelle_cost  # Float(0.0, units='USD', iotype='in', desc='nacelle cost')
        self.tower_cost = tower_cost  # Float(0.0, units='USD', iotype='in', desc='cost for a tower')
        self.tower_mass = tower_mass  # Float(0.0, units='kg', iotype='in', desc='mass for a turbine tower')

        # Parameters (and ignored inputs)
        self.blade_number = blade_number  # Int(3, iotype='in', desc = 'number of rotor blades')
        self.offshore = offshore  # Bool(False, iotype='in', desc= 'boolean for offshore')

        # high level output assignment
        self.rotor_mass = self.blade_mass * self.blade_number + self.hub_system_mass
        self.rotor_cost = self.blade_cost * self.blade_number + self.hub_system_cost
        self.turbine_mass = self.rotor_mass + self.nacelle_mass + self.tower_mass
        self.turbine_cost = self.rotor_cost + self.nacelle_cost + self.tower_cost

        if self.offshore:
            self.turbine_cost *= 1.1

        # derivatives
        self.d_mass_d_blade_mass = self.blade_number
        self.d_mass_d_hub_mass = 1.0
        self.d_mass_d_nacelle_mass = 1.0
        self.d_mass_d_tower_mass = 1.0

        if self.offshore:
            self.d_cost_d_blade_cost = 1.1 * self.blade_number
            self.d_cost_d_hub_cost = 1.1
            self.d_cost_d_nacelle_cost = 1.1
            self.d_cost_d_tower_cost = 1.1
        else:
            self.d_cost_d_blade_cost = self.blade_number
            self.d_cost_d_hub_cost = 1.0
            self.d_cost_d_nacelle_cost = 1.0
            self.d_cost_d_tower_cost = 1.0

    def list_deriv_vars(self):
        inputs = [
            "blade_mass",
            "hub_system_mass",
            "nacelle_mass",
            "tower_mass",
            "blade_cost",
            "hub_system_cost",
            "nacelle_cost",
            "tower_cost",
        ]

        outputs = ["turbine_mass", "turbine_cost"]

        return inputs, outputs

    def provideJ(self):
        self.J = np.array(
            [
                [
                    self.d_mass_d_blade_mass,
                    self.d_mass_d_hub_mass,
                    self.d_mass_d_nacelle_mass,
                    self.d_mass_d_tower_mass,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    self.d_cost_d_blade_cost,
                    self.d_cost_d_hub_cost,
                    self.d_cost_d_nacelle_cost,
                    self.d_cost_d_tower_cost,
                ],
            ]
        )

        return self.J


# --------------------------------------------------------------------
class tcc_csm(object):
    def __init__(self):
        super(tcc_csm, self).__init__()  # will actually run the workflow

        # Outputs
        self.turbine_cost = (
            0.0  # Float(0.0, iotype='out', desc='Overall wind turbine capial costs including transportation costs')
        )
        self.rotor_cost = 0.0  # Float(0.0, iotype='out', desc='Rotor cost')
        self.nacelle_cost = 0.0  # Float(0.0, iotype='out', desc='Nacelle cost')
        self.tower_cost = 0.0  # Float(0.0, iotype='out', desc='Tower cost')

    def compute(
        self,
        rotor_diameter,
        machine_rating,
        hub_height,
        rotor_thrust,
        rotor_torque,
        year=2009,
        month=12,
        blade_number=3,
        offshore=True,
        advanced_blade=False,
        drivetrain_design="geared",
        crane=True,
        advanced_bedplate=0,
        advanced_tower=False,
    ):
        # Variables
        self.rotor_diameter = rotor_diameter  # Float(units = 'm', iotype='in', desc= 'rotor diameter of the machine')
        self.machine_rating = machine_rating  # Float(units = 'kW', iotype='in', desc = 'rated power of wind turbine')
        self.hub_height = (
            hub_height  # Float(units = 'm', iotype='in', desc= 'hub height of wind turbine above ground / sea level')
        )
        self.rotor_thrust = rotor_thrust  # Float(iotype='in', units='N', desc='maximum thurst from rotor')
        self.rotor_torque = rotor_torque  # Float(iotype='in', units='N * m', desc = 'torque from rotor at rated power')

        # Parameters
        self.year = year  # Int(2009, iotype='in', desc = 'year of project start')
        self.month = month  # Int(12, iotype='in', desc = 'month of project start')
        self.blade_number = blade_number  # Int(3, iotype='in', desc = 'number of rotor blades')
        self.offshore = offshore  # Bool(True, iotype='in', desc = 'boolean for offshore')
        self.advanced_blade = (
            advanced_blade  # Bool(False, iotype='in', desc = 'boolean for use of advanced blade curve')
        )
        self.drivetrain_design = drivetrain_design  # Enum('geared', ('geared', 'single_stage', 'multi_drive', 'pm_direct_drive'), iotype='in')
        self.crane = crane  # Bool(True, iotype='in', desc = 'boolean for presence of a service crane up tower')
        self.advanced_bedplate = (
            advanced_bedplate  # Int(0, iotype='in', desc= 'indicator for drivetrain bedplate design 0 - conventional')
        )
        self.advanced_tower = advanced_tower  # Bool(False, iotype='in', desc = 'advanced tower configuration')

        blade = blades_csm()
        blade.compute(rotor_diameter, year, month, advanced_blade)

        hub = hub_csm()
        hub.compute(rotor_diameter, blade.blade_mass, year, month, blade_number)

        rotor = rotor_mass_adder()
        rotor.compute(blade.blade_mass, hub.hub_system_mass, blade_number)

        nacelle = nacelle_csm()
        nacelle.compute(
            rotor_diameter,
            rotor.rotor_mass,
            rotor_thrust,
            rotor_torque,
            machine_rating,
            drivetrain_design,
            crane,
            advanced_bedplate,
            year,
            month,
            offshore,
        )

        tower = tower_csm()
        tower.compute(rotor_diameter, hub_height, year, month, advanced_tower)

        turbine = turbine_csm()
        turbine.compute(
            blade.blade_cost,
            blade.blade_mass,
            hub.hub_system_cost,
            hub.hub_system_mass,
            nacelle.nacelle_mass,
            nacelle.nacelle_cost,
            tower.tower_cost,
            tower.tower_mass,
            blade_number,
            offshore,
        )

        self.rotor_cost = turbine.rotor_cost
        self.rotor_mass = turbine.rotor_mass
        self.turbine_cost = turbine.turbine_cost
        self.turbine_mass = turbine.turbine_mass


# Balance of System Costs
##################################################


class bos_csm(object):
    def __init__(self):
        # Outputs
        # bos_breakdown = VarTree(BOSVarTree(), iotype='out', desc='BOS cost breakdown')
        # bos_costs = Float(iotype='out', desc='Overall wind plant balance of station/system costs up to point of comissioning')
        self.bos_costs = 0.0  # *= self.multiplier  # TODO: add to gradients
        self.bos_breakdown_development_costs = 0.0  # engPermits_costs * self.turbine_number
        self.bos_breakdown_preparation_and_staging_costs = (
            0.0  # (roadsCivil_costs + portStaging_costs) * self.turbine_number
        )
        self.bos_breakdown_transportation_costs = 0.0  # (transportation_costs * self.turbine_number)
        self.bos_breakdown_foundation_and_substructure_costs = 0.0  # foundation_cost * self.turbine_number
        self.bos_breakdown_electrical_costs = 0.0  # electrical_costs * self.turbine_number
        self.bos_breakdown_assembly_and_installation_costs = 0.0  # installation_costs * self.turbine_number
        self.bos_breakdown_soft_costs = 0.0  # 0.0
        self.bos_breakdown_other_costs = 0.0  # (pai_costs + scour_costs + suretyBond) * self.turbine_number

    def compute(
        self,
        machine_rating,
        rotor_diameter,
        hub_height,
        RNA_mass,
        turbine_cost,
        turbine_number=100,
        sea_depth=20.0,
        year=2009,
        month=12,
        multiplier=1.0,
    ):
        # for coding ease
        # Default Variables
        self.machine_rating = machine_rating  # Float(iotype='in', units='kW', desc='turbine machine rating')
        self.rotor_diameter = rotor_diameter  # Float(iotype='in', units='m', desc='rotor diameter')
        self.hub_height = hub_height  # Float(iotype='in', units='m', desc='hub height')
        self.RNA_mass = RNA_mass  # Float(iotype='in', units='kg', desc='Rotor Nacelle Assembly mass')
        self.turbine_cost = turbine_cost  # Float(iotype='in', units='USD', desc='Single Turbine Capital _costs')

        # Parameters
        self.turbine_number = turbine_number  # Int(iotype='in', desc='number of turbines in project')
        self.sea_depth = (
            sea_depth  # Float(20.0, units = 'm', iotype = 'in', desc = 'sea depth for offshore wind plant')
        )
        self.year = year  # Int(2009, iotype='in', desc='year for project start')
        self.month = month  # Int(12, iotype = 'in', desc= 'month for project start')
        self.multiplier = multiplier  # Float(1.0, iotype='in')

        lPrmtsCostCoeff1 = 9.94e-04
        lPrmtsCostCoeff2 = 20.31
        oPrmtsCostFactor = 37.0  # $/kW (2003)
        scourCostFactor = 55.0  # $/kW (2003)
        ptstgCostFactor = 20.0  # $/kW (2003)
        ossElCostFactor = 260.0  # $/kW (2003) shallow
        ostElCostFactor = 290.0  # $/kW (2003) transitional
        ostSTransFactor = 25.0  # $/kW (2003)
        ostTTransFactor = 77.0  # $/kW (2003)
        osInstallFactor = 100.0  # $/kW (2003) shallow & trans
        suppInstallFactor = 330.0  # $/kW (2003) trans additional
        paiCost = 60000.0  # per turbine

        suretyBRate = 0.03  # 3% of ICC
        suretyBond = 0.0

        # set variables
        if self.sea_depth == 0:  # type of plant # 1: Land, 2: < 30m, 3: < 60m, 4: >= 60m
            iDepth = 1
        elif self.sea_depth < 30:
            iDepth = 2
        elif self.sea_depth < 60:
            iDepth = 3
        else:
            iDepth = 4

        # initialize self.ppi index calculator
        if iDepth == 1:
            ref_yr = 2002
            ref_mon = 9
        else:
            ref_yr = 2003
            ref_mon = 9
        ppi.ref_yr = ref_yr
        ppi.ref_mon = ref_mon
        ppi.curr_yr = self.year
        ppi.curr_mon = self.month

        self.d_foundation_d_diameter = 0.0
        self.d_foundation_d_hheight = 0.0
        self.d_foundation_d_rating = 0.0
        # foundation costs
        if iDepth == 1:  # land
            fcCoeff = 303.23
            fcExp = 0.4037
            SweptArea = (self.rotor_diameter * 0.5) ** 2.0 * np.pi
            foundation_cost = fcCoeff * (self.hub_height * SweptArea) ** fcExp
            fndnCostEscalator = ppi.compute("IPPI_FND")
            self.d_foundation_d_diameter = (
                fndnCostEscalator
                * fcCoeff
                * fcExp
                * ((self.hub_height * (2.0 * 0.5 * (self.rotor_diameter * 0.5) * np.pi)) ** (fcExp - 1))
                * self.hub_height
            )
            self.d_foundation_d_hheight = (
                fndnCostEscalator * fcCoeff * fcExp * ((self.hub_height * SweptArea) ** (fcExp - 1)) * SweptArea
            )
        elif iDepth == 2:
            sscf = 300.0  # $/kW
            foundation_cost = sscf * self.machine_rating
            fndnCostEscalator = ppi.compute("IPPI_MPF")
            self.d_foundation_d_rating = fndnCostEscalator * sscf
        elif iDepth == 3:
            sscf = 450.0  # $/kW
            foundation_cost = sscf * self.machine_rating
            fndnCostEscalator = ppi.compute("IPPI_OAI")
            self.d_foundation_d_rating = fndnCostEscalator * sscf
        elif iDepth == 4:
            foundation_cost = 0.0
            fndnCostEscalator = 1.0

        foundation_cost *= fndnCostEscalator

        # cost calculations
        tpC1 = 0.00001581
        tpC2 = -0.0375
        tpInt = 54.7
        tFact = tpC1 * self.machine_rating * self.machine_rating + tpC2 * self.machine_rating + tpInt

        roadsCivil_costs = 0.0
        portStaging_costs = 0.0
        pai_costs = 0.0
        scour_costs = 0.0
        self.d_assembly_d_diameter = 0.0
        self.d_assembly_d_hheight = 0.0
        self.d_development_d_rating = 0.0
        self.d_preparation_d_rating = 0.0
        self.d_transport_d_rating = 0.0
        self.d_electrical_d_rating = 0.0
        self.d_assembly_d_rating = 0.0
        self.d_other_d_rating = 0.0
        if iDepth == 1:
            engPermits_costs = (lPrmtsCostCoeff1 * self.machine_rating * self.machine_rating) + (
                lPrmtsCostCoeff2 * self.machine_rating
            )
            ppi.ref_mon = 3
            engPermits_costs *= ppi.compute("IPPI_LPM")
            self.d_development_d_rating = ppi.compute("IPPI_LPM") * (
                2.0 * lPrmtsCostCoeff1 * self.machine_rating + lPrmtsCostCoeff2
            )
            ppi.ref_mon = 9

            elC1 = 3.49e-06
            elC2 = -0.0221
            elInt = 109.7
            eFact = elC1 * self.machine_rating * self.machine_rating + elC2 * self.machine_rating + elInt
            electrical_costs = self.machine_rating * eFact * ppi.compute("IPPI_LEL")
            self.d_electrical_d_rating = ppi.compute("IPPI_LEL") * (
                3.0 * elC1 * self.machine_rating**2.0 + 2.0 * elC2 * self.machine_rating + elInt
            )

            rcC1 = 2.17e-06
            rcC2 = -0.0145
            rcInt = 69.54
            rFact = rcC1 * self.machine_rating * self.machine_rating + rcC2 * self.machine_rating + rcInt
            roadsCivil_costs = self.machine_rating * rFact * ppi.compute("IPPI_RDC")
            self.d_preparation_d_rating = ppi.compute("IPPI_RDC") * (
                3.0 * rcC1 * self.machine_rating**2.0 + 2.0 * rcC2 * self.machine_rating + rcInt
            )

            iCoeff = 1.965
            iExp = 1.1736
            installation_costs = iCoeff * ((self.hub_height * self.rotor_diameter) ** iExp) * ppi.compute("IPPI_LAI")
            self.d_assembly_d_diameter = (
                iCoeff
                * ((self.hub_height * self.rotor_diameter) ** (iExp - 1))
                * self.hub_height
                * ppi.compute("IPPI_LAI")
            )
            self.d_assembly_d_hheight = (
                iCoeff
                * ((self.hub_height * self.rotor_diameter) ** (iExp - 1))
                * self.rotor_diameter
                * ppi.compute("IPPI_LAI")
            )

            transportation_costs = self.machine_rating * tFact * ppi.compute("IPPI_TPT")
            self.d_transport_d_rating = ppi.compute("IPPI_TPT") * (
                tpC1 * 3.0 * self.machine_rating**2.0 + tpC2 * 2.0 * self.machine_rating + tpInt
            )

        elif iDepth == 2:  # offshore shallow
            ppi.ref_yr = 2003
            pai_costs = paiCost * ppi.compute("IPPI_PAE")
            portStaging_costs = ptstgCostFactor * self.machine_rating * ppi.compute("IPPI_STP")  # 1.415538133
            self.d_preparation_d_rating = ptstgCostFactor * ppi.compute("IPPI_STP")
            engPermits_costs = oPrmtsCostFactor * self.machine_rating * ppi.compute("IPPI_OPM")
            self.d_development_d_rating = oPrmtsCostFactor * ppi.compute("IPPI_OPM")
            scour_costs = scourCostFactor * self.machine_rating * ppi.compute("IPPI_STP")  # 1.415538133#
            self.d_other_d_rating = scourCostFactor * ppi.compute("IPPI_STP")
            installation_costs = osInstallFactor * self.machine_rating * ppi.compute("IPPI_OAI")
            self.d_assembly_d_rating = osInstallFactor * ppi.compute("IPPI_OAI")
            electrical_costs = ossElCostFactor * self.machine_rating * ppi.compute("IPPI_OEL")
            self.d_electrical_d_rating = ossElCostFactor * ppi.compute("IPPI_OEL")
            ppi.ref_yr = 2002
            transportation_costs = self.machine_rating * tFact * ppi.compute("IPPI_TPT")
            self.d_transport_d_rating = ppi.compute("IPPI_TPT") * (
                tpC1 * 3.0 * self.machine_rating**2.0 + tpC2 * 2.0 * self.machine_rating + tpInt
            )
            ppi.ref_yr = 2003

        elif iDepth == 3:  # offshore transitional depth
            ppi.ref_yr = 2003
            turbInstall = osInstallFactor * self.machine_rating * ppi.compute("IPPI_OAI")
            supportInstall = suppInstallFactor * self.machine_rating * ppi.compute("IPPI_OAI")
            installation_costs = turbInstall + supportInstall
            self.d_assembly_d_rating = (osInstallFactor + suppInstallFactor) * ppi.compute("IPPI_OAI")
            pai_costs = paiCost * ppi.compute("IPPI_PAE")
            electrical_costs = ostElCostFactor * self.machine_rating * ppi.compute("IPPI_OEL")
            self.d_electrical_d_rating = ossElCostFactor * ppi.compute("IPPI_OEL")
            portStaging_costs = ptstgCostFactor * self.machine_rating * ppi.compute("IPPI_STP")
            self.d_preparation_d_rating = ptstgCostFactor * ppi.compute("IPPI_STP")
            engPermits_costs = oPrmtsCostFactor * self.machine_rating * ppi.compute("IPPI_OPM")
            self.d_development_d_rating = oPrmtsCostFactor * ppi.compute("IPPI_OPM")
            scour_costs = scourCostFactor * self.machine_rating * ppi.compute("IPPI_STP")
            self.d_other_d_rating = scourCostFactor * ppi.compute("IPPI_STP")
            ppi.ref_yr = 2002
            turbTrans = ostTTransFactor * self.machine_rating * ppi.compute("IPPI_TPT")
            self.d_transport_d_rating = ostTTransFactor * ppi.compute("IPPI_TPT")
            ppi.ref_yr = 2003
            supportTrans = ostSTransFactor * self.machine_rating * ppi.compute("IPPI_OAI")
            transportation_costs = turbTrans + supportTrans
            self.d_transport_d_rating += ostSTransFactor * ppi.compute("IPPI_OAI")

        elif iDepth == 4:  # offshore deep
            print("\ncsmBOS: Add costCat 4 code\n\n")

        bos_costs = (
            foundation_cost
            + transportation_costs
            + roadsCivil_costs
            + portStaging_costs
            + installation_costs
            + electrical_costs
            + engPermits_costs
            + pai_costs
            + scour_costs
        )

        self.d_other_d_tcc = 0.0
        if self.sea_depth > 0.0:
            suretyBond = suretyBRate * (self.turbine_cost + bos_costs)
            self.d_other_d_tcc = suretyBRate
            d_surety_d_rating = suretyBRate * (
                self.d_development_d_rating
                + self.d_preparation_d_rating
                + self.d_transport_d_rating
                + self.d_foundation_d_rating
                + self.d_electrical_d_rating
                + self.d_assembly_d_rating
                + self.d_other_d_rating
            )
            self.d_other_d_rating += d_surety_d_rating
        else:
            suretyBond = 0.0

        self.bos_costs = self.turbine_number * (bos_costs + suretyBond)
        self.bos_costs *= self.multiplier  # TODO: add to gradients

        self.bos_breakdown_development_costs = engPermits_costs * self.turbine_number
        self.bos_breakdown_preparation_and_staging_costs = (roadsCivil_costs + portStaging_costs) * self.turbine_number
        self.bos_breakdown_transportation_costs = transportation_costs * self.turbine_number
        self.bos_breakdown_foundation_and_substructure_costs = foundation_cost * self.turbine_number
        self.bos_breakdown_electrical_costs = electrical_costs * self.turbine_number
        self.bos_breakdown_assembly_and_installation_costs = installation_costs * self.turbine_number
        self.bos_breakdown_soft_costs = 0.0
        self.bos_breakdown_other_costs = (pai_costs + scour_costs + suretyBond) * self.turbine_number

        # derivatives
        self.d_development_d_rating *= self.turbine_number
        self.d_preparation_d_rating *= self.turbine_number
        self.d_transport_d_rating *= self.turbine_number
        self.d_foundation_d_rating *= self.turbine_number
        self.d_electrical_d_rating *= self.turbine_number
        self.d_assembly_d_rating *= self.turbine_number
        self.d_soft_d_rating = 0.0
        self.d_other_d_rating *= self.turbine_number
        self.d_cost_d_rating = (
            self.d_development_d_rating
            + self.d_preparation_d_rating
            + self.d_transport_d_rating
            + self.d_foundation_d_rating
            + self.d_electrical_d_rating
            + self.d_assembly_d_rating
            + self.d_soft_d_rating
            + self.d_other_d_rating
        )

        self.d_development_d_diameter = 0.0
        self.d_preparation_d_diameter = 0.0
        self.d_transport_d_diameter = 0.0
        # self.d_foundation_d_diameter
        self.d_electrical_d_diameter = 0.0
        # self.d_assembly_d_diameter
        self.d_soft_d_diameter = 0.0
        self.d_other_d_diameter = 0.0
        self.d_cost_d_diameter = (
            self.d_development_d_diameter
            + self.d_preparation_d_diameter
            + self.d_transport_d_diameter
            + self.d_foundation_d_diameter
            + self.d_electrical_d_diameter
            + self.d_assembly_d_diameter
            + self.d_soft_d_diameter
            + self.d_other_d_diameter
        )

        self.d_development_d_tcc = 0.0
        self.d_preparation_d_tcc = 0.0
        self.d_transport_d_tcc = 0.0
        self.d_foundation_d_tcc = 0.0
        self.d_electrical_d_tcc = 0.0
        self.d_assembly_d_tcc = 0.0
        self.d_soft_d_tcc = 0.0
        self.d_other_d_tcc *= self.turbine_number
        self.d_cost_d_tcc = (
            self.d_development_d_tcc
            + self.d_preparation_d_tcc
            + self.d_transport_d_tcc
            + self.d_foundation_d_tcc
            + self.d_electrical_d_tcc
            + self.d_assembly_d_tcc
            + self.d_soft_d_tcc
            + self.d_other_d_tcc
        )

        self.d_development_d_hheight = 0.0
        self.d_preparation_d_hheight = 0.0
        self.d_transport_d_hheight = 0.0
        # self.d_foundation_d_hheight
        self.d_electrical_d_hheight = 0.0
        # self.d_assembly_d_hheight
        self.d_soft_d_hheight = 0.0
        self.d_other_d_hheight = 0.0
        self.d_cost_d_hheight = (
            self.d_development_d_hheight
            + self.d_preparation_d_hheight
            + self.d_transport_d_hheight
            + self.d_foundation_d_hheight
            + self.d_electrical_d_hheight
            + self.d_assembly_d_hheight
            + self.d_soft_d_hheight
            + self.d_other_d_hheight
        )

        self.d_development_d_rna = 0.0
        self.d_preparation_d_rna = 0.0
        self.d_transport_d_rna = 0.0
        self.d_foundation_d_rna = 0.0
        self.d_electrical_d_rna = 0.0
        self.d_assembly_d_rna = 0.0
        self.d_soft_d_rna = 0.0
        self.d_other_d_rna = 0.0
        self.d_cost_d_rna = (
            self.d_development_d_rna
            + self.d_preparation_d_rna
            + self.d_transport_d_rna
            + self.d_foundation_d_rna
            + self.d_electrical_d_rna
            + self.d_assembly_d_rna
            + self.d_soft_d_rna
            + self.d_other_d_rna
        )

    def list_deriv_vars(self):
        inputs = ["machine_rating", "rotor_diameter", "turbine_cost", "hub_height", "RNA_mass"]
        outputs = [
            "bos_breakdown.development_costs",
            "bos_breakdown.preparation_and_staging_costs",
            "bos_breakdown.transportation_costs",
            "bos_breakdown.foundation_and_substructure_costs",
            "bos_breakdown.electrical_costs",
            "bos_breakdown.assembly_and_installation_costs",
            "bos_breakdown.soft_costs",
            "bos_breakdown.other_costs",
            "bos_costs",
        ]

        return inputs, outputs

    def provideJ(self):
        self.J = np.array(
            [
                [
                    self.d_development_d_rating,
                    self.d_development_d_diameter,
                    self.d_development_d_tcc,
                    self.d_development_d_hheight,
                    self.d_development_d_rna,
                ],
                [
                    self.d_preparation_d_rating,
                    self.d_preparation_d_diameter,
                    self.d_preparation_d_tcc,
                    self.d_preparation_d_hheight,
                    self.d_preparation_d_rna,
                ],
                [
                    self.d_transport_d_rating,
                    self.d_transport_d_diameter,
                    self.d_transport_d_tcc,
                    self.d_transport_d_hheight,
                    self.d_transport_d_rna,
                ],
                [
                    self.d_foundation_d_rating,
                    self.d_foundation_d_diameter,
                    self.d_foundation_d_tcc,
                    self.d_foundation_d_hheight,
                    self.d_foundation_d_rna,
                ],
                [
                    self.d_electrical_d_rating,
                    self.d_electrical_d_diameter,
                    self.d_electrical_d_tcc,
                    self.d_electrical_d_hheight,
                    self.d_electrical_d_rna,
                ],
                [
                    self.d_assembly_d_rating,
                    self.d_assembly_d_diameter,
                    self.d_assembly_d_tcc,
                    self.d_assembly_d_hheight,
                    self.d_assembly_d_rna,
                ],
                [
                    self.d_soft_d_rating,
                    self.d_soft_d_diameter,
                    self.d_soft_d_tcc,
                    self.d_soft_d_hheight,
                    self.d_soft_d_rna,
                ],
                [
                    self.d_other_d_rating,
                    self.d_other_d_diameter,
                    self.d_other_d_tcc,
                    self.d_other_d_hheight,
                    self.d_other_d_rna,
                ],
                [
                    self.d_cost_d_rating,
                    self.d_cost_d_diameter,
                    self.d_cost_d_tcc,
                    self.d_cost_d_hheight,
                    self.d_cost_d_rna,
                ],
            ]
        )

        return self.J


# Operational Expenditures
##################################################


class opex_csm(object):
    def __init__(self):
        # variables

        # Outputs
        self.avg_annual_opex = 0.0

        # self.opex_breakdown = VarTree(OPEXVarTree(),iotype='out')
        self.opex_breakdown_preventative_opex = 0.0
        self.opex_breakdown_corrective_opex = 0.0
        self.opex_breakdown_lease_opex = 0.0
        self.opex_breakdown_other_opex = 0.0

    def compute(self, sea_depth, year, month, turbine_number, machine_rating, net_aep):
        # initialize variables
        if sea_depth == 0:
            offshore = False
        else:
            offshore = True
        ppi.curr_yr = year
        ppi.curr_mon = month

        # O&M
        offshoreCostFactor = 0.0200  # $/kwH
        landCostFactor = 0.0070  # $/kwH
        if not offshore:  # kld - place for an error check - iShore should be in 1:4
            cost = net_aep * landCostFactor
            costEscalator = ppi.compute("IPPI_LOM")
        else:
            cost = net_aep * offshoreCostFactor
            ppi.ref_yr = 2003
            costEscalator = ppi.compute("IPPI_OOM")
            ppi.ref_yr = 2002

        self.opex_breakdown_preventative_opex = cost * costEscalator  # in $/year

        # LRC
        if not offshore:
            lrcCF = 10.70  # land based
            costlrcEscFactor = ppi.compute("IPPI_LLR")
        else:  # TODO: transition and deep water options if applicable
            lrcCF = 17.00  # offshore
            ppi.ref_yr = 2003
            costlrcEscFactor = ppi.compute("IPPI_OLR")
            ppi.ref_yr = 2002

        self.opex_breakdown_corrective_opex = machine_rating * lrcCF * costlrcEscFactor * turbine_number  # in $/yr

        # LLC
        if not offshore:
            leaseCF = 0.00108  # land based
            costlandEscFactor = ppi.compute("IPPI_LSE")
        else:  # TODO: transition and deep water options if applicable
            leaseCF = 0.00108  # offshore
            costlandEscFactor = ppi.compute("IPPI_LSE")

        self.opex_breakdown_lease_opex = net_aep * leaseCF * costlandEscFactor  # in $/yr

        # Other
        self.opex_breakdown_other_opex = 0.0

        # Total OPEX
        self.avg_annual_opex = (
            self.opex_breakdown_preventative_opex + self.opex_breakdown_corrective_opex + self.opex_breakdown_lease_opex
        )

    def compute_partials(self):
        # dervivatives
        self.d_corrective_d_aep = 0.0
        self.d_corrective_d_rating = lrcCF * costlrcEscFactor * self.turbine_number
        self.d_lease_d_aep = leaseCF * costlandEscFactor
        self.d_lease_d_rating = 0.0
        self.d_other_d_aep = 0.0
        self.d_other_d_rating = 0.0
        if not offshore:
            self.d_preventative_d_aep = landCostFactor * costEscalator
        else:
            self.d_preventative_d_aep = offshoreCostFactor * costEscalator
        self.d_preventative_d_rating = 0.0
        self.d_opex_d_aep = (
            self.d_preventative_d_aep + self.d_corrective_d_aep + self.d_lease_d_aep + self.d_other_d_aep
        )
        self.d_opex_d_rating = (
            self.d_preventative_d_rating + self.d_corrective_d_rating + self.d_lease_d_rating + self.d_other_d_rating
        )

        self.J = np.array(
            [
                [self.d_preventative_d_aep, self.d_preventative_d_rating],
                [self.d_corrective_d_aep, self.d_corrective_d_rating],
                [self.d_lease_d_aep, self.d_lease_d_rating],
                [self.d_other_d_aep, self.d_other_d_rating],
                [self.d_opex_d_aep, self.d_opex_d_rating],
            ]
        )

        return self.J


# NREL Cost and Scaling Model finance modules
##################################################


class fin_csm(object):
    def __init__(
        self,
        fixed_charge_rate=0.12,
        construction_finance_rate=0.0,
        tax_rate=0.4,
        discount_rate=0.07,
        construction_time=1.0,
        project_lifetime=20.0,
    ):
        """
        OpenMDAO component to wrap finance model of the NREL _cost and Scaling Model (csmFinance.py)

        """

        super(fin_csm, self).__init__()

        # Outputs
        self.coe = 0.0  # Float(iotype='out', desc='Levelized cost of energy for the wind plant')
        self.lcoe = 0.0  # Float(iotype='out', desc='_cost of energy - unlevelized')

        # parameters
        self.fixed_charge_rate = (
            fixed_charge_rate  # Float(0.12, iotype = 'in', desc = 'fixed charge rate for coe calculation')
        )
        self.construction_finance_rate = construction_finance_rate  # Float(0.00, iotype='in', desc = 'construction financing rate applied to overnight capital costs')
        self.tax_rate = tax_rate  # Float(0.4, iotype = 'in', desc = 'tax rate applied to operations')
        self.discount_rate = discount_rate  # Float(0.07, iotype = 'in', desc = 'applicable project discount rate')
        self.construction_time = (
            construction_time  # Float(1.0, iotype = 'in', desc = 'number of years to complete project construction')
        )
        self.project_lifetime = (
            project_lifetime  # Float(20.0, iotype = 'in', desc = 'project lifetime for LCOE calculation')
        )

    def compute(self, turbine_cost, turbine_number, bos_costs, avg_annual_opex, net_aep, sea_depth):
        """
        Executes finance model of the NREL _cost and Scaling model to determine overall plant COE and LCOE.
        """

        # Inputs
        self.turbine_cost = turbine_cost  # Float(iotype='in', desc = 'A Wind Turbine Capital _cost')
        self.turbine_number = turbine_number  # Int(iotype = 'in', desc = 'number of turbines at plant')
        self.bos_costs = bos_costs  # Float(iotype='in', desc='A Wind Plant Balance of Station _cost Model')
        self.avg_annual_opex = avg_annual_opex  # Float(iotype='in', desc='A Wind Plant Operations Expenditures Model')
        self.net_aep = net_aep  # Float(iotype='in', desc='A Wind Plant Annual Energy Production Model', units='kW*h')
        self.sea_depth = sea_depth

        if self.sea_depth > 0.0:
            offshore = True
        else:
            offshore = False

        if offshore:
            warrantyPremium = (self.turbine_cost * self.turbine_number / 1.10) * 0.15
            icc = self.turbine_cost * self.turbine_number + warrantyPremium + self.bos_costs
        else:
            icc = self.turbine_cost * self.turbine_number + self.bos_costs

        # compute COE and LCOE values
        self.coe = (icc * self.fixed_charge_rate / self.net_aep) + (self.avg_annual_opex) * (
            1 - self.tax_rate
        ) / self.net_aep

        amortFactor = (1 + 0.5 * ((1 + self.discount_rate) ** self.construction_time - 1)) * (
            self.discount_rate / (1 - (1 + self.discount_rate) ** (-1.0 * self.project_lifetime))
        )
        self.lcoe = (icc * amortFactor + self.avg_annual_opex) / self.net_aep

        # derivatives
        if offshore:
            self.d_coe_d_turbine_cost = (
                self.turbine_number * (1 + 0.15 / 1.10) * self.fixed_charge_rate
            ) / self.net_aep
        else:
            self.d_coe_d_turbine_cost = self.turbine_number * self.fixed_charge_rate / self.net_aep
        self.d_coe_d_bos_cost = self.fixed_charge_rate / self.net_aep
        self.d_coe_d_avg_opex = (1 - self.tax_rate) / self.net_aep
        self.d_coe_d_net_aep = -(icc * self.fixed_charge_rate + self.avg_annual_opex * (1 - self.tax_rate)) / (
            self.net_aep**2
        )

        if offshore:
            self.d_lcoe_d_turbine_cost = self.turbine_number * (1 + 0.15 / 1.10) * amortFactor / self.net_aep
        else:
            self.d_lcoe_d_turbine_cost = self.turbine_number * amortFactor / self.net_aep
        self.d_lcoe_d_bos_cost = amortFactor / self.net_aep
        self.d_lcoe_d_avg_opex = 1.0 / self.net_aep
        self.d_lcoe_d_net_aep = -(icc * amortFactor + self.avg_annual_opex) / (self.net_aep**2)

    def list_deriv_vars(self):
        inputs = ["turbine_cost", "bos_costs", "avg_annual_opex", "net_aep"]
        outputs = ["coe", "lcoe"]

        return inputs, outputs

    def provideJ(self):
        # Jacobian
        self.J = np.array(
            [
                [self.d_coe_d_turbine_cost, self.d_coe_d_bos_cost, self.d_coe_d_avg_opex, self.d_coe_d_net_aep],
                [self.d_lcoe_d_turbine_cost, self.d_lcoe_d_bos_cost, self.d_lcoe_d_avg_opex, self.d_lcoe_d_net_aep],
            ]
        )

        return self.J


"""if __name__=="__main__":


    ### TODO: Examples
"""
