#!/usr/bin/env python
# encoding: utf-8
"""
coe_mdao.py

Created by Andrew Ning on 2013-08-28.
Copyright (c) NREL. All rights reserved.
"""

import numpy as np
from math import pi
import algopy

from openmdao.main.api import Component, Assembly
from openmdao.main.datatypes.api import Float, Bool, Int
from openmdao.lib.drivers.sensitivity import SensitivityDriver
from openmdao.lib.differentiators.api import FiniteDifference, Analytic


class COE(Component):

    # variables
    TCC = Float(iotype='in', units='USD', desc='turbine capital costs for oen turbine')
    BOS = Float(iotype='in', units='USD', desc='costs for whole plant')
    OM = Float(iotype='in', units='USD', desc='for one turbine')
    LLC = Float(iotype='in', units='USD', desc='for one turbine')
    LRC = Float(iotype='in', units='USD', desc='for one turbine')
    AEP = Float(iotype='in', units='kW*h', desc='for one turbine')

    # parameters
    constructionRate = Float(iotype='in', desc='construction financing rate')
    fixedChargeRate = Float(iotype='in')
    taxRate = Float(iotype='in')
    nTurbines = Int(iotype='in')

    # outputs
    COE = Float(iotype='out', units='USD/kW/h', desc='cost of energy')


    def __init__(self):
        super(COE, self).__init__()

        self.derivatives.declare_first_derivative('COE', 'TCC')
        self.derivatives.declare_first_derivative('COE', 'BOS')
        self.derivatives.declare_first_derivative('COE', 'OM')
        self.derivatives.declare_first_derivative('COE', 'LLC')
        self.derivatives.declare_first_derivative('COE', 'LRC')
        self.derivatives.declare_first_derivative('COE', 'AEP')



    def execute(self):

        TCC = self.nTurbines * self.TCC
        BOS = self.BOS
        OM = self.nTurbines * self.OM
        LLC = self.nTurbines * self.LLC
        LRC = self.nTurbines * self.LRC
        AEP = self.nTurbines * self.AEP

        ICC = (TCC + BOS)*(1 + self.constructionRate)
        AOE = OM*(1-self.taxRate) + LLC + LRC

        self.COE = (ICC*self.fixedChargeRate + AOE) / AEP



    def calculate_first_derivatives(self):

        cr = self.constructionRate
        fcr = self.fixedChargeRate
        tr = self.taxRate
        AEP = self.AEP
        n = self.nTurbines

        self.derivatives.set_first_derivative('COE', 'TCC', (1.0+cr)*fcr/AEP)
        self.derivatives.set_first_derivative('COE', 'BOS', (1.0+cr)*fcr/n/AEP)
        self.derivatives.set_first_derivative('COE', 'OM', (1.0-tr)/AEP)
        self.derivatives.set_first_derivative('COE', 'LLC', 1.0/AEP)
        self.derivatives.set_first_derivative('COE', 'LRC', 1.0/AEP)
        self.derivatives.set_first_derivative('COE', 'AEP', -self.COE/AEP)



class TCC(Component):


    # variables
    bladeMass = Float(iotype='in', units='kg')
    hubMass = Float(iotype='in', units='kg')
    pitchSystemMass = Float(iotype='in', units='kg')
    spinnerMass = Float(iotype='in', units='kg')

    lssMass = Float(iotype='in', units='kg')
    bearingsMass = Float(iotype='in', units='kg')
    gearboxMass = Float(iotype='in', units='kg')
    mechBrakeMass = Float(iotype='in', units='kg')
    generatorMass = Float(iotype='in', units='kg')
    bedplateMass = Float(iotype='in', units='kg')
    yawSystemMass = Float(iotype='in', units='kg')

    towerMass = Float(iotype='in', units='kg')

    # parameters
    machineRating = Float(iotype='in', units='W')
    offshore = Bool(False, iotype='in')
    crane = Bool(False, iotype='in')
    nBlades = Int(3, iotype='in')
    advancedBlade = Bool(False, iotype='in')

    TCC = Float(iotype='out', units='USD', desc='turbine capital costs')


    def __init__(self):
        super(TCC, self).__init__()

        self.derivatives.declare_first_derivative('TCC', 'bladeMass')
        self.derivatives.declare_first_derivative('TCC', 'hubMass')
        self.derivatives.declare_first_derivative('TCC', 'pitchSystemMass')
        self.derivatives.declare_first_derivative('TCC', 'spinnerMass')
        self.derivatives.declare_first_derivative('TCC', 'lssMass')
        self.derivatives.declare_first_derivative('TCC', 'bearingsMass')
        self.derivatives.declare_first_derivative('TCC', 'gearboxMass')
        self.derivatives.declare_first_derivative('TCC', 'mechBrakeMass')
        self.derivatives.declare_first_derivative('TCC', 'generatorMass')
        self.derivatives.declare_first_derivative('TCC', 'bedplateMass')
        self.derivatives.declare_first_derivative('TCC', 'yawSystemMass')
        self.derivatives.declare_first_derivative('TCC', 'towerMass')


    def execute(self):

        # ------ rotor -----------

        if self.advancedBlade:
            slope = 13.0
            intercept = 5813.9
            ppi_mat = 1.07037339304
        else:
            slope = 8.0
            intercept = 21465.0
            ppi_mat = 1.03440640603

        bladeCost = (slope*self.bladeMass + intercept)*ppi_mat

        hubCost2002 = self.hubMass * 4.25  # $/kg
        hubCostEscalator = 1.3050397878
        hubCost = hubCost2002 * hubCostEscalator


        pitchSysCost2002 = 2.28 * 0.0808 * self.pitchSystemMass**1.4985  # new cost based on mass - x1.328 for housing proportion
        bearingCostEscalator = 1.32919380003
        pitchCost = bearingCostEscalator * pitchSysCost2002


        spinnerCostEscalator = 1.04209517363
        spinnerCost = spinnerCostEscalator * 5.57*self.spinnerMass

        rotorCost = bladeCost*self.nBlades + hubCost + pitchCost + spinnerCost

        # derivatives
        self.d_bladeMass = slope*ppi_mat*self.nBlades
        self.d_hubMass = hubCostEscalator * 4.25
        self.d_pitchSystemMass = bearingCostEscalator * 2.28 * 0.0808 * 1.4985 * self.pitchSystemMass**0.4985
        self.d_spinnerMass = spinnerCostEscalator * 5.57

        # ----- nacelle -------------
        lssCostEsc = 1.54583624913
        bearingCostEsc = 1.32328605201
        GearboxCostEsc = 1.33353510896
        mechBrakeCostEsc = 1.01968134958
        generatorCostEsc = 1.28663330951
        BedplateCostEsc = 1.3050397878
        yawDrvBearingCostEsc = 1.35582751798
        lssCostEsc = 1.54583624913
        bearingCostEsc = 1.32328605201
        GearboxCostEsc = 1.33353510896
        mechBrakeCostEsc = 1.01968134958
        generatorCostEsc = 1.28663330951
        BedplateCostEsc = 1.3050397878
        yawDrvBearingCostEsc = 1.35582751798
        BedplateCostEsc = 1.3050397878
        VspdEtronicsCostEsc = 1.27883310719
        nacelleCovCostEsc = 1.04209517363
        hydrCoolingCostEsc = 1.34722222222
        econnectionsCostEsc = 1.69098620946
        controlsCostEsc = 1.23169955442

        machineRating = self.machineRating / 1e3  # convert to kW


        # calculate component cost
        LowSpeedShaftCost2002 = 3.3602 * self.lssMass + 13587      # equation adjusted to be based on mass rather than rotor diameter using data from CSM
        lowSpeedShaftCost = LowSpeedShaftCost2002 * lssCostEsc

        brngSysCostFactor = 17.6  # $/kg                  # cost / unit mass from CSM
        Bearings2002 = self.bearingsMass * brngSysCostFactor
        mainBearingsCost = Bearings2002 * bearingCostEsc / 4.0   # div 4 to account for bearing cost mass differences CSM to Sunderland

        Gearbox2002 = 16.9 * self.gearboxMass - 25066          # for traditional 3-stage gearbox, use mass based cost equation from NREL CSM
        gearboxCost = Gearbox2002 * GearboxCostEsc

        mechBrakeCost2002 = 10 * self.mechBrakeMass                  # mechanical brake system cost based on $10 / kg multiplier from CSM model (inverse relationship)
        highSpeedShaftCost = mechBrakeCostEsc * mechBrakeCost2002

        GeneratorCost2002 = 19.697 * self.generatorMass + 9277.3
        generatorCost = GeneratorCost2002 * generatorCostEsc

        bedplateCost2002 = 0.9461 * self.bedplateMass + 17799                   # equation adjusted based on mass / cost relationships for components documented in NREL CSM
        bedplateCost = bedplateCost2002 * BedplateCostEsc

        YawDrvBearing2002 = 8.3221 * self.yawSystemMass + 2708.5          # cost / mass relationship derived from NREL CSM data
        yawSystemCost = YawDrvBearing2002 * yawDrvBearingCostEsc


        # electronic systems, hydraulics and controls
        econnectionsCost2002 = 40.0 * machineRating  # 2002
        econnectionsCost = econnectionsCost2002 * econnectionsCostEsc

        VspdEtronics2002 = 79.32 * machineRating
        vspdEtronicsCost = VspdEtronics2002 * VspdEtronicsCostEsc

        hydrCoolingCost2002 = 12.0 * machineRating  # 2002
        hydrCoolingCost = hydrCoolingCost2002 * hydrCoolingCostEsc

        if (not self.offshore):
            ControlsCost2002 = 35000.0  # initial approximation 2002
            controlsCost = ControlsCost2002 * controlsCostEsc
        else:
            ControlsCost2002 = 55900.0  # initial approximation 2002
            controlsCost = ControlsCost2002 * controlsCostEsc

        # mainframe system including bedplate, platforms, crane and miscellaneous hardware
        nacellePlatformsMass = 0.125 * self.bedplateMass
        NacellePlatforms2002 = 8.7 * nacellePlatformsMass

        if (self.crane):
            craneCost2002 = 12000.0
        else:
            craneCost2002 = 0.0

        # aggregation of mainframe components: bedplate, crane and platforms into single mass and cost
        BaseHardwareCost2002 = bedplateCost2002 * 0.7
        MainFrameCost2002 = NacellePlatforms2002 + craneCost2002 + BaseHardwareCost2002
        mainframeCost = MainFrameCost2002 * BedplateCostEsc + bedplateCost

        nacelleCovCost2002 = 11.537 * machineRating + 3849.7
        nacelleCovCost = nacelleCovCost2002 * nacelleCovCostEsc


        # aggregation of nacelle costs
        nacelleCost = lowSpeedShaftCost + mainBearingsCost + gearboxCost + highSpeedShaftCost + \
            generatorCost + yawSystemCost + econnectionsCost + vspdEtronicsCost + \
            hydrCoolingCost + mainframeCost + controlsCost + nacelleCovCost

        # derivatives
        self.d_lssMass = lssCostEsc * 3.3602
        self.d_bearingsMass = bearingCostEsc / 4.0 * brngSysCostFactor
        self.d_gearboxMass = GearboxCostEsc * 16.9
        self.d_mechBrakeMass = mechBrakeCostEsc * 10.0
        self.d_generatorMass = generatorCostEsc * 19.697
        self.d_bedplateMass = BedplateCostEsc * (0.9461 + 0.125*8.7 + 0.9461*0.7)
        self.d_yawSystemMass = yawDrvBearingCostEsc * 8.3221





        # ----- tower -------------

        twrCostEscalator = 1.51445578231
        twrCostCoeff = 1.5  # $/kg
        towerCost2002 = self.towerMass * twrCostCoeff
        towerCost = towerCost2002 * twrCostEscalator

        # derivative
        self.d_towerMass = twrCostEscalator * twrCostCoeff


        # ---- TCC ---
        turbineCost = rotorCost + nacelleCost + towerCost

        if self.offshore:  # add marinization
            marCoeff = 0.10  # 10%
            turbineCost *= (1 + marCoeff)

        self.TCC = turbineCost

        # derivative
        if self.offshore:
            self.d_multiplier = (1 + marCoeff)
        else:
            self.d_multiplier = 1.0



    def calculate_first_derivatives(self):

        self.derivatives.set_first_derivative('TCC', 'bladeMass', self.d_multiplier*self.d_bladeMass)
        self.derivatives.set_first_derivative('TCC', 'hubMass', self.d_multiplier*self.d_hubMass)
        self.derivatives.set_first_derivative('TCC', 'pitchSystemMass', self.d_multiplier*self.d_pitchSystemMass)
        self.derivatives.set_first_derivative('TCC', 'spinnerMass', self.d_multiplier*self.d_spinnerMass)
        self.derivatives.set_first_derivative('TCC', 'lssMass', self.d_multiplier*self.d_lssMass)
        self.derivatives.set_first_derivative('TCC', 'bearingsMass', self.d_multiplier*self.d_bearingsMass)
        self.derivatives.set_first_derivative('TCC', 'gearboxMass', self.d_multiplier*self.d_gearboxMass)
        self.derivatives.set_first_derivative('TCC', 'mechBrakeMass', self.d_multiplier*self.d_mechBrakeMass)
        self.derivatives.set_first_derivative('TCC', 'generatorMass', self.d_multiplier*self.d_generatorMass)
        self.derivatives.set_first_derivative('TCC', 'bedplateMass', self.d_multiplier*self.d_bedplateMass)
        self.derivatives.set_first_derivative('TCC', 'yawSystemMass', self.d_multiplier*self.d_yawSystemMass)
        self.derivatives.set_first_derivative('TCC', 'towerMass', self.d_multiplier*self.d_towerMass)



class BOS(Component):

    # variables
    diameter = Float(iotype='in', units='m')
    hubHeight = Float(iotype='in', units='m')
    TCC = Float(iotype='in', units='USD', desc='turbine capital costs for one turbine')
    towerTopMass = Float(iotype='in', units='kg')

    # parameters
    machineRating = Float(iotype='in', units='W')
    nTurbines = Int(iotype='in')

    # outputs
    BOS = Float(iotype='out', units='USD', desc='BOS for plant')


    def __init__(self):
        super(BOS, self).__init__()

        self.derivatives.declare_first_derivative('BOS', 'diameter')
        self.derivatives.declare_first_derivative('BOS', 'hubHeight')
        self.derivatives.declare_first_derivative('BOS', 'TCC')
        self.derivatives.declare_first_derivative('BOS', 'towerTopMass')




    def execute(self):

        rating = self.machineRating / 1e3  # convert to kW
        nTurb = self.nTurbines
        diameter = self.diameter
        hubHeight = self.hubHeight
        towerTopMass = self.towerTopMass / 1000  # convert to tonnes
        TCC = self.TCC / rating  # convert to $/kW
        totalMW = rating*nTurb/1000.0

        # inputs from user
        # rating = 2300  # kW
        # diameter = 100  # m
        # hubHeight = 80  # m
        # nTurb = 87
        # TCC = 1286  # $/kW
        # towerTopMass = 160  # tonnes

        # other inputs
        interconnectVoltage = 137  # kV
        distanceToInterconnect = 5  # mi
        siteTerrain = 'FlatToRolling'
        turbineLayout = 'Simple'
        soilCondition = 'Standard'
        constructionTime = 20  # months

        # advanced inputs
        deliveryAssistRequired = False
        buildingSize = 5000  # sqft
        windWeatherDelayDays = 80
        craneBreakdowns = 4
        accessRoadEntrances = 4
        performanceBond = False
        contingency = 3.0  # %
        warrantyManagement = 0.02  # %
        salesAndUseTax = 5  # %
        overhead = 6  # %
        profitMargin = 6  # %
        developmentFee = 5  # million
        turbineTransportation = 300  # miles

        # inputs that don't seem to be used
        # padMountTransformerRequired = True
        # rockTrenchingRequired = 10.0  # % of cable collector length
        # MVthermalBackfill = 0  # mi
        # MVoverheadCollector = 0  # mi


        # TODO: smoothness issues



        # ---- turbine and transport cost -----
        # mi = turbineTransportation
        # if rating < 2500 and hubHeight < 100:
        #     turbineTransportCost = 1349*mi**0.746 * nTurb
        # else:
        #     turbineTransportCost = 1867*mi**0.726 * nTurb



        # TODO: for now - my smoother version
        mi = turbineTransportation
        turbineTransportCost = 1867*mi**0.726 * nTurb




        # ---- engineering cost -------
        # engineeringCost = 7188.5*nTurb

        # multiplier = 16800
        # if totalMW > 300:
        #     engineeringCost += 20*multiplier
        # elif totalMW > 100:
        #     engineeringCost += 15*multiplier
        # elif totalMW > 30:
        #     engineeringCost += 10*multiplier
        # else:
        #     engineeringCost += 5*multiplier

        # multiplier = 161675
        # if totalMW > 200:
        #     engineeringCost += 2*multiplier
        # else:
        #     engineeringCost += multiplier

        # engineeringCost += 4000



        # TODO: for now - my smoother version
        engineeringCost = 7188.5*nTurb

        multiplier = 16800
        engineeringCost += 20*multiplier

        multiplier = 161675
        engineeringCost += 2*multiplier

        engineeringCost += 4000



        # ---- met mast and power performance engineering cost ----
        powerPerformanceCost = 200000

        if totalMW > 30:
            multiplier1 = 2
        else:
            multiplier1 = 1

        if totalMW > 100:
            multiplier2 = 4
        elif totalMW > 30:
            multiplier2 = 2
        else:
            multiplier2 = 1

        ## my smooth version (using cubic spline)
        hL = 85.0
        hU = 95.0

        poly1 = np.poly1d([-114.8, 30996.0, -2781030.0, 83175600.0])
        vL1 = 232600.0
        vU1 = 290000.0
        if hubHeight <= hL:
            value1 = vL1
            D1 = 0.0
        elif hubHeight >= hU:
            value1 = vU1
            D1 = 0.0
        else:
            value1 = poly1(hubHeight)
            D1 = poly1.deriv(1)(hubHeight)

        poly2 = np.poly1d([-48.4, 13068.0, -1172490.0, 35061600.0])
        vL2 = 92600
        vU2 = 116800

        if hubHeight <= hL:
            value2 = vL2
        elif hubHeight >= hU:
            value2 = vU2
        else:
            value2 = poly2(hubHeight)
            D2 = poly2.deriv(1)(hubHeight)

        powerPerformanceCost += multiplier1 * value1 + multiplier2 * value2
        ppc_deriv = multiplier1 * D1 + multiplier2 * D2

        # if hubHeight < 90:
        #     powerPerformanceCost += multiplier1 * 232600 + multiplier2 * 92600
        # else:
        #     powerPerformanceCost += multiplier1 * 290000 + multiplier2 * 116800




        # --- access road and site improvement cost, and electrical costs -----
        if turbineLayout == 'Simple':
            if siteTerrain == 'FlatToRolling':
                accessCost = 5972082
                electricalMaterialCost = 10975731
                electricalInstallationCost = 4368309
            elif siteTerrain == 'RidgeTop':
                accessCost = 6891018
                electricalMaterialCost = 11439093
                electricalInstallationCost = 6427965
            elif siteTerrain == 'Mountainous':
                accessCost = 7484975
                electricalMaterialCost = 11465572
                electricalInstallationCost = 7594765
            else:
                print 'error'  # TODO: handle error
        elif turbineLayout == 'Complex':
            if siteTerrain == 'FlatToRolling':
                accessCost = 7187138
                electricalMaterialCost = 12229923
                electricalInstallationCost = 6400995
            elif siteTerrain == 'RidgeTop':
                accessCost = 8262300
                electricalMaterialCost = 12694155
                electricalInstallationCost = 9313885
            elif siteTerrain == 'Mountainous':
                accessCost = 9055930
                electricalMaterialCost = 12796728
                electricalInstallationCost = 10386160
            else:
                print 'error'  # TODO: handle error
        else:
            print 'error'  # TODO: handle error



        # ---- site compound and security costs -----
        siteSecurityCost = 9825*accessRoadEntrances + 29850*constructionTime

        if totalMW > 100:
            multiplier = 10
        elif totalMW > 30:
            multiplier = 5
        else:
            multiplier = 3

        siteSecurityCost += multiplier * 30000

        if totalMW > 30:
            siteSecurityCost += 90000

        siteSecurityCost += totalMW * 60 + 62400




        # ---- control - O&M building cost -----
        buildingCost = buildingSize*125 + 176125



        # ----- turbine foundation cost -----
        foundationCost = rating*diameter*towerTopMass/1000.0 + 163794*nTurb**-0.12683 \
            + (hubHeight-80)*500

        if soilCondition == 'Bouyant':
            foundationCost += 20000

        foundationCost *= nTurb


        # --- turbine erection cost ----
        erectionCost = (37*rating + 250000*nTurb**-0.41 + (hubHeight-80)*500) * nTurb \
            + 20000*windWeatherDelayDays + 35000*craneBreakdowns + 180*nTurb + 1800
        if deliveryAssistRequired:
            erectionCost += 60000 * nTurb



        # ----- collector substation costs -----
        collectorCost = 11652*(interconnectVoltage + totalMW) + 11795*totalMW**0.3549 + 1234300



        # ---- transmission line and interconnection cost -----
        transmissionCost = (1176*interconnectVoltage + 218257)*distanceToInterconnect**0.8937 \
            + 18115*interconnectVoltage + 165944



        # --- project management ----
        if constructionTime < 28:
            projMgtCost = (53.333*constructionTime**2 - 3442*constructionTime + 209542)*(constructionTime+2)
        else:
            projMgtCost = (constructionTime+2)*155000


        # --- markup and contingency costs ----
        alpha_contingency = (contingency + warrantyManagement + salesAndUseTax + overhead + profitMargin)/100


        # ---- insurance bonds and permit costs -----
        insurance = (3.5 + 0.7 + 0.4 + 1.0) * TCC*totalMW
        if performanceBond:
            insurance += 10.0 * TCC*totalMW

        permit = foundationCost * 0.02
        permit += 20000

        insuranceCost = insurance + permit

        alpha_insurance = (3.5 + 0.7 + 0.4 + 1.0)/1000.0
        if performanceBond:
            alpha_insurance += 10.0/1000


        # ---- development cost ------
        developmentCost = developmentFee * 1e6


        # ------ total -----
        self.BOS = turbineTransportCost + insuranceCost + engineeringCost + powerPerformanceCost + \
            accessCost + siteSecurityCost + buildingCost + foundationCost + erectionCost + \
            electricalMaterialCost + electricalInstallationCost + collectorCost + \
            transmissionCost + projMgtCost + developmentCost


        # multiplier
        alpha = alpha_contingency + alpha_insurance

        self.BOS /= (1-alpha)

        # # if needed
        # contingencyCost = alpha_contingency*BOS
        # insuranceCost += alpha_insurance*BOS

        # return BOS


        # derivatives
        self.d_diameter = rating*towerTopMass/1000.0 * nTurb * (1 + 0.02)

        self.d_hubHeight = 500 * nTurb * (1 + 0.02) + 500 * nTurb + ppc_deriv

        self.d_TCC = (3.5 + 0.7 + 0.4 + 1.0)*totalMW
        if performanceBond:
            self.d_TCC += 10.0*totalMW
        self.d_TCC /= rating

        self.d_towerTopMass = rating*diameter/1000.0 * nTurb * (1 + 0.02)
        self.d_towerTopMass /= 1000

        self.d_mult = 1.0/(1-alpha)



    def calculate_first_derivatives(self):

        self.derivatives.set_first_derivative('BOS', 'diameter', self.d_mult * self.d_diameter)
        self.derivatives.set_first_derivative('BOS', 'hubHeight', self.d_mult * self.d_hubHeight)
        self.derivatives.set_first_derivative('BOS', 'TCC', self.d_mult * self.d_TCC)
        self.derivatives.set_first_derivative('BOS', 'towerTopMass', self.d_mult * self.d_towerTopMass)




class OM(Component):

    # variables
    AEP = Float(iotype='in', units='kW*h', desc='AEP for one turbine')

    # parameters
    machineRating = Float(iotype='in', units='W')
    offshore = Bool(False, iotype='in')

    # outputs
    OM = Float(iotype='out', units='USD')
    LRC = Float(iotype='out', units='USD')
    LLC = Float(iotype='out', units='USD')


    def __init__(self):
        super(OM, self).__init__()

        self.derivatives.declare_first_derivative('OM', 'AEP')
        self.derivatives.declare_first_derivative('LRC', 'AEP')
        self.derivatives.declare_first_derivative('LLC', 'AEP')


    def execute(self):

        machineRating = self.machineRating / 1e3  # convert to kW


        if not self.offshore:
            costEscalator = 1.09718172983
            costFactor = 0.0070  # $/kwH (land)

        else:
            costEscalator = 1.07115749526
            costFactor = 0.0200  # $/kwH (offshore)

        self.OM = self.AEP * costFactor * costEscalator
        self.d_OM = costFactor * costEscalator

        # levelized replacement cost ($/yr)

        if not self.offshore:
            lrcCF = 10.70  # land based
            costEscFactor = 1.09718172983
        else:
            lrcCF = 17.00  # offshore
            costEscFactor = 1.07115749526

        self.LRC = machineRating * lrcCF * costEscFactor  # in $/yr

        # returns lease cost ($/yr)

        # in CSM spreadsheet, land and offshore leases cost the same
        leaseCF = 0.00108  # land based and offshore
        costEscFactor = 1.09718172983

        self.LLC = self.AEP * leaseCF * costEscFactor  # in $/yr
        self.d_LLC = leaseCF * costEscFactor


    def calculate_first_derivatives(self):

        self.derivatives.set_first_derivative('OM', 'AEP', self.d_OM)
        self.derivatives.set_first_derivative('LRC', 'AEP', 0.0)
        self.derivatives.set_first_derivative('LLC', 'AEP', self.d_LLC)



class Hub(Component):
    """docstring for Hub"""

    # variables
    bladeMass = Float(iotype='in', units='kg', desc='mass of one blade')
    rootMoment = Float(iotype='in', units='N*m', desc='flapwise bending moment at blade root')
    rotorDiameter = Float(iotype='in', units='m', desc='rotor diameter')

    # parameters
    nBlades = Int(3, iotype='in')

    # outputs
    hubMass = Float(iotype='out', units='kg')
    pitchMass = Float(iotype='out', units='kg')
    spinnerMass = Float(iotype='out', units='kg')
    rotorMass = Float(iotype='out', units='kg')


    def __init__(self):
        super(Hub, self).__init__()

        self.derivatives.declare_first_derivative('hubMass', 'rootMoment')

        self.derivatives.declare_first_derivative('pitchMass', 'bladeMass')
        self.derivatives.declare_first_derivative('pitchMass', 'rootMoment')

        self.derivatives.declare_first_derivative('spinnerMass', 'rotorDiameter')

        self.derivatives.declare_first_derivative('rotorMass', 'bladeMass')
        self.derivatives.declare_first_derivative('rotorMass', 'rootMoment')
        self.derivatives.declare_first_derivative('rotorMass', 'rotorDiameter')



    def execute(self):

        # Sunderland method for calculating hub, pitch and spinner cone masses
        hubloadFact = 1              # default is 3 blade rigid hub (should depend on hub type)
        hubgeomFact = 1              # default is 3 blade (should depend on blade number)
        hubcontFact = 1.5            # default is 1.5 for full span pitch control (should depend on control method)
        hubmatldensity = 7860.0      # density of hub material (kg / m^3) - assuming BS1503-622 (same material as LSS)
        hubmatlstress = 371000000.0  # allowable stress of hub material (N / m^2)

        self.hubMass = 50.0 * hubgeomFact * hubloadFact * hubcontFact * self.nBlades * self.rootMoment * hubmatldensity / hubmatlstress


        # Sunderland method for calculating pitch system masses
        pitchmatldensity = 7860.0      # density of pitch system material (kg / m^3) - assuming BS1503-622 (same material as LSS)
        pitchmatlstress = 371000000.0  # allowable stress of hub material (N / m^2)
        hubpitchFact = 1.0             # default factor is 1.0 (0.54 for modern designs)

        self.pitchMass = hubpitchFact * (0.22 * self.bladeMass * self.nBlades + 12.6 * self.nBlades * self.rootMoment * pitchmatldensity / pitchmatlstress)

        # spinner mass comes from cost and scaling model
        self.spinnerMass = 18.5 * self.rotorDiameter - 520.5


        self.rotorMass = self.nBlades*self.bladeMass + self.hubMass + self.pitchMass + self.spinnerMass


        # derivatives
        self.d_hubMass_d_rootMoment = self.hubMass/self.rootMoment

        self.d_pitchMass_d_bladeMass = hubpitchFact * 0.22 * self.nBlades
        self.d_pitchMass_d_rootMoment = hubpitchFact * 12.6 * self.nBlades * pitchmatldensity / pitchmatlstress

        self.d_spinnerMass_d_rotorDiameter = 18.5

        self.d_rotorMass_d_bladeMass = self.nBlades + self.d_pitchMass_d_bladeMass
        self.d_rotorMass_d_rootMoment = self.d_hubMass_d_rootMoment + self.d_pitchMass_d_rootMoment
        self.d_rotorMass_d_rotorDiameter = self.d_spinnerMass_d_rotorDiameter


    def calculate_first_derivatives(self):

        self.derivatives.set_first_derivative('hubMass', 'rootMoment', self.d_hubMass_d_rootMoment)

        self.derivatives.set_first_derivative('pitchMass', 'bladeMass', self.d_pitchMass_d_bladeMass)
        self.derivatives.set_first_derivative('pitchMass', 'rootMoment', self.d_pitchMass_d_rootMoment)

        self.derivatives.set_first_derivative('spinnerMass', 'rotorDiameter', self.d_spinnerMass_d_rotorDiameter)

        self.derivatives.set_first_derivative('rotorMass', 'bladeMass', self.d_rotorMass_d_bladeMass)
        self.derivatives.set_first_derivative('rotorMass', 'rootMoment', self.d_rotorMass_d_rootMoment)
        self.derivatives.set_first_derivative('rotorMass', 'rotorDiameter', self.d_rotorMass_d_rotorDiameter)




class Nacelle(Component):

    # variables
    rotorDiameter = Float(iotype='in', units='m')
    rotorMass = Float(iotype='in', units='kg')
    rotorThrust = Float(iotype='in', units='N')
    rotorTorque = Float(iotype='in', units='N*m')
    rotorSpeed = Float(iotype='in', units='rpm')

    # parameters
    gearRatio = Float(iotype='in')
    machineRating = Float(iotype='in', units='W')
    towerTopDiameter = Float(iotype='in', units='m')
    crane = Bool(False, iotype='in')


    # outputs
    lssMass = Float(iotype='out', units='kg')
    bearingsMass = Float(iotype='out', units='kg')
    gearboxMass = Float(iotype='out', units='kg')
    mechBrakeMass = Float(iotype='out', units='kg')
    generatorMass = Float(iotype='out', units='kg')
    bedplateMass = Float(iotype='out', units='kg')
    yawSystemMass = Float(iotype='out', units='kg')

    totalMass = Float(iotype='out', units='kg')



    def __init__(self):
        super(Nacelle, self).__init__()

        self.derivatives.declare_first_derivative('lssMass', 'rotorDiameter')
        self.derivatives.declare_first_derivative('lssMass', 'rotorMass')
        self.derivatives.declare_first_derivative('lssMass', 'rotorTorque')

        self.derivatives.declare_first_derivative('bearingsMass', 'rotorDiameter')
        self.derivatives.declare_first_derivative('bearingsMass', 'rotorMass')
        self.derivatives.declare_first_derivative('bearingsMass', 'rotorTorque')
        self.derivatives.declare_first_derivative('bearingsMass', 'rotorSpeed')

        self.derivatives.declare_first_derivative('gearboxMass', 'rotorTorque')

        self.derivatives.declare_first_derivative('mechBrakeMass', 'rotorTorque')

        self.derivatives.declare_first_derivative('bedplateMass', 'rotorDiameter')
        self.derivatives.declare_first_derivative('bedplateMass', 'rotorMass')
        self.derivatives.declare_first_derivative('bedplateMass', 'rotorThrust')
        self.derivatives.declare_first_derivative('bedplateMass', 'rotorTorque')

        self.derivatives.declare_first_derivative('yawSystemMass', 'rotorDiameter')
        self.derivatives.declare_first_derivative('yawSystemMass', 'rotorMass')
        self.derivatives.declare_first_derivative('yawSystemMass', 'rotorThrust')
        self.derivatives.declare_first_derivative('yawSystemMass', 'rotorTorque')
        self.derivatives.declare_first_derivative('yawSystemMass', 'rotorSpeed')

        self.derivatives.declare_first_derivative('totalMass', 'rotorDiameter')
        self.derivatives.declare_first_derivative('totalMass', 'rotorMass')
        self.derivatives.declare_first_derivative('totalMass', 'rotorThrust')
        self.derivatives.declare_first_derivative('totalMass', 'rotorTorque')
        self.derivatives.declare_first_derivative('totalMass', 'rotorSpeed')


    def execute(self):

        self.ratedPower = self.machineRating/1e3  # convert to kW

        x = [self.rotorDiameter, self.rotorMass, self.rotorThrust, self.rotorTorque, self.rotorSpeed]

        self.lssMass = self.lss(x)
        self.bearingsMass = self.bearings(x)
        self.gearboxMass = self.gearbox(x)
        self.mechBrakeMass = self.mechBrake(x)
        self.generatorMass = self.generator(x)
        self.bedplateMass = self.bedplate(x)
        self.yawSystemMass = self.yawSystem(x)
        self.totalMass = self.total(x)


    def lss(self, x):

        rotorDiameter = x[0]
        rotorMass = x[1]
        # rotorThrust = x[2]
        rotorTorque = x[3]
        # rotorSpeed = x[4]


        # low-speed shaft
        ioratio = 0.100                                # constant value for inner/outer diameter ratio (should depend on LSS type)
        hollow = 1.0/(1-(ioratio)**4)                    # hollowness factor based on diameter ratio

        TQsafety = 3.0                                 # safety factor for design torque applied to rotor torque
        self.lss_designTQ = TQsafety * rotorTorque        # LSS design torque [Nm]

        lenFact = 0.03                                 # constant value for length as a function of rotor diameter (should depend on LSS type)
        length = lenFact * rotorDiameter          # LSS shaft length [m]
        maFact = 5                                     # moment arm factor from shaft lenght (should depend on shaft type)
        mmtArm = length / maFact                       # LSS moment arm [m] - from hub to first main bearing
        BLsafety = 1.25                                # saftey factor on bending load
        g = 9.81                                       # gravitational constant [m / s^2]
        designBL = BLsafety * g * rotorMass       # LSS design bending load [N]
        bendMom = designBL * mmtArm                    # LSS design bending moment [Nm]

        yieldst = 371000000.0                          # BS1503-622 yield stress [Pa] (should be adjusted depending on material type)
        endurstsp = 309000000.0                        # BS1503-625 specimen endurance limit [Pa] (should be adjusted depending on material type)
        endurFact = 0.23                               # factor for specimen to component endurance limit
                                                       # (0.75 surface condition * 0.65 size * 0.52 reliability * 1 temperature * 0.91 stress concentration)
        endurst = endurstsp * endurFact                # endurance limit [Pa] for LSS
        SOsafety = 3.25                                # Soderberg Line approach factor for safety
        self.lss_diameter = (32.0/pi*hollow*SOsafety*((self.lss_designTQ / yieldst)**2 + (bendMom/endurst)**2)**0.5)**(1./3.)  # outer diameter [m] computed by Westinghouse Code Formula based on Soderberg Line approach to fatigue design
        inDiam = self.lss_diameter * ioratio                    # inner diameter [m]

        massFact = 1.25                                    # mass weight factor (should depend on LSS/drivetrain type, currently from windpact modifications to account for flange weight)
        steeldens = 7860                                    # steel density [kg / m^3]

        lssMass = massFact*(pi/4)*(self.lss_diameter**2-inDiam**2)*length*steeldens      # mass of LSS [kg]

        self.lssMass_temp = lssMass

        return lssMass

        # # derivatives
        # k1 = (32.0/pi*hollow*SOsafety)**(1.0/3)
        # k2 = (TQsafety/yieldst)**2
        # k3 = (bendMom/endurst)**2
        # k4 = massFact*(pi/4)*length*steeldens*(1-ioratio**2)
        # k5 = k2*self.rotorTorque**2 + k3
        # self.d_lss_d_torque = k4*2*lss_diameter*k1*1.0/6*k5**(-5.0/6)*k2*2*self.rotorTorque


    def bearings(self, x):

        # rotorDiameter = x[0]
        # rotorMass = x[1]
        # rotorThrust = x[2]
        # rotorTorque = x[3]
        rotorSpeed = x[4]

        self.lss(x)


        # bearings
        g = 9.81  # gravitational constant [m / s^2]
        design1SL = (4.0 / 3.0) * self.lss_designTQ + self.lssMass_temp * (g / 2.0)  # front bearing static design load [N] based on default equation (should depend on LSS type)
        design2SL = (1.0 / 3.0) * self.lss_designTQ - self.lssMass_temp * (g / 2.0)  # rear bearing static design load [N] based on default equation (should depend on LSS type)
        design1DL = 2.29 * design1SL * (rotorSpeed ** 0.3)  # front bearing dynamic design load [N]
        design2DL = 2.29 * design2SL * (rotorSpeed ** 0.3)  # rear bearing dynamic design load [N]
        ratingDL = 17.96 * (self.lss_diameter * 1000.0) ** 1.9752  # basic dynamic load rating for a bearing given inside diameter based on catalogue regression

        massFact = 0.25                                 # bearing weight factor (should depend on drivetrain type) - using to adjust data closer to cost and scaling model estimates

        if (design1DL < ratingDL):
            b1mass = massFact * (26.13 * (10 ** (-6))) * ((self.lss_diameter * 1000.0) ** 2.77)
                                                           # bearing mass [kg] for single row bearing (based on catalogue data regression)
            h1mass = massFact * (67.44 * (10 ** (-6))) * ((self.lss_diameter * 1000.0) ** 2.64)
                                                           # bearing housing mass [kg] for single row bearing (based on catalogue data regression)
        else:
            b1mass = massFact * 1.7 * (26.13 * (10 ** (-6))) * ((self.lss_diameter * 1000.0) ** 2.77)
                                                          # bearing mass [kg] for double row bearing (based on catalogue data regression)
            h1mass = massFact * 1.5 * (67.44 * (10 ** (-6))) * ((self.lss_diameter * 1000.0) ** 2.64)
                                                          # bearing housing mass [kg] for double row bearing (based on catalogue data regression)

        if (design2DL < ratingDL):
            b2mass = massFact * (26.13 * (10 ** (-6))) * ((self.lss_diameter * 1000.0) ** 2.77)
                                                           # bearing mass [kg] for single row bearing (based on catalogue data regression)
            h2mass = massFact * (67.44 * (10 ** (-6))) * ((self.lss_diameter * 1000.0) ** 2.64)
                                                           # bearing housing mass [kg] for single row bearing (based on catalogue data regression)
        else:
            b2mass = massFact * 1.7 * (26.13 * (10 ** (-6))) * ((self.lss_diameter * 1000.0) ** 2.77)
                                                          # bearing mass [kg] for double row bearing (based on catalogue data regression)
            h2mass = massFact * 1.5 * (67.44 * (10 ** (-6))) * ((self.lss_diameter * 1000.0) ** 2.64)
                                                          # bearing housing mass [kg] for double row bearing (based on catalogue data regression)

        mainBearingMass = b1mass + h1mass
        secondBearingMass = b2mass + h2mass

        bearingsMass = mainBearingMass + secondBearingMass

        self.bearingsMass_temp = bearingsMass

        return bearingsMass


    def gearbox(self, x):

        # rotorDiameter = x[0]
        # rotorMass = x[1]
        # rotorThrust = x[2]
        rotorTorque = x[3]
        # rotorSpeed = x[4]


        # gearbox (eep only)
        overallweightFact = 1.00                          # default weight factor 1.0 (should depend on drivetrain design)

        U1 = (self.gearRatio/3.0)**0.5
        U2 = (self.gearRatio/3.0)**0.5
        U3 = 3.0
        mass = 0.0
        mass += self.__getEpicyclicStageWeight(rotorTorque, 1, U1, U2, U3)  # different than sunderland
        mass += self.__getEpicyclicStageWeight(rotorTorque, 2, U1, U2, U3)
        mass += self.__getParallelStageWeight(rotorTorque, 3, U1, U2, U3)

        gearboxMass = mass * overallweightFact

        self.gearboxMass_temp = gearboxMass

        return gearboxMass


    def mechBrake(self, x):

        # rotorDiameter = x[0]
        # rotorMass = x[1]
        # rotorThrust = x[2]
        rotorTorque = x[3]
        # rotorSpeed = x[4]

        # high-speed shaft

        hss_designTQ = rotorTorque / self.gearRatio               # design torque [Nm] based on rotor torque and Gearbox ratio
        massFact = 0.025                                 # mass matching factor default value
        hss_mass = massFact * hss_designTQ

        mechBrake_mass = 0.5 * hss_mass      # relationship derived from HSS multiplier for University of Sunderland model compared to NREL CSM for 750 kW and 1.5 MW turbines

        mechBrakeMass = mechBrake_mass + hss_mass

        self.mechBrakeMass_temp = mechBrakeMass

        return mechBrakeMass


    def generator(self, x):

        # rotorDiameter = x[0]
        # rotorMass = x[1]
        # # rotorThrust = x[2]
        # rotorTorque = x[3]
        # rotorSpeed = x[4]

        # generator (iDesign = 1 only)

        massCoeff = 6.4737
        massExp = 0.9223

        generatorMass = massCoeff * self.ratedPower**massExp

        self.generatorMass_temp = generatorMass

        return generatorMass


    def bedplate(self, x):

        rotorDiameter = x[0]
        rotorMass = x[1]
        rotorThrust = x[2]
        rotorTorque = x[3]
        # rotorSpeed = x[4]


        # bedplate (iDesign = 1 only)

        # bedplate sizing based on superposition of loads for rotor torque, thurst, weight         #TODO: only handles bedplate for a traditional drivetrain configuration
        bedplateWeightFact = 2.86                                   # toruqe weight factor for bedplate (should depend on drivetrain, bedplate type)

        torqueweightCoeff = 0.00368                   # regression coefficient multiplier for bedplate weight based on rotor torque
        MassFromTorque = bedplateWeightFact * torqueweightCoeff * rotorTorque

        thrustweightCoeff = 0.00158                                 # regression coefficient multiplier for bedplate weight based on rotor thrust
        MassFromThrust = bedplateWeightFact * thrustweightCoeff * rotorThrust * self.towerTopDiameter

        rotorweightCoeff = 0.015                                    # regression coefficient multiplier for bedplate weight based on rotor weight
        MassFromRotorWeight = bedplateWeightFact * rotorweightCoeff * rotorMass * self.towerTopDiameter

        # additional weight ascribed to bedplate area
        BPlengthFact = 1.5874                                       # bedplate length factor (should depend on drivetrain, bedplate type)
        nacellevolFact = 0.052                                      # nacelle volume factor (should depend on drivetrain, bedplate type)
        self.bedplateLength = (BPlengthFact * nacellevolFact * rotorDiameter)     # bedplate length [m] calculated as a function of rotor diameter
        width = (self.bedplateLength / 2.0)                              # bedplate width [m] assumed to be half of bedplate length
        area = self.bedplateLength * width                        # bedplate area [m^2]
        areaweightCoeff = 100                                       # regression coefficient multiplier for bedplate weight based on bedplate area
        MassFromArea = bedplateWeightFact * (areaweightCoeff * area)

        # total mass is calculated based on adding masses attributed to rotor torque, thrust, weight and bedplate area
        bedplateMass = MassFromTorque + MassFromThrust + MassFromRotorWeight + MassFromArea

        self.bedplateMass_temp = bedplateMass

        return bedplateMass


    def yawSystem(self, x):

        rotorDiameter = x[0]
        # rotorMass = x[1]
        rotorThrust = x[2]
        # rotorTorque = x[3]
        # rotorSpeed = x[4]

        self.lss(x)
        self.bearings(x)
        self.gearbox(x)
        self.mechBrake(x)
        self.generator(x)
        self.bedplate(x)

        # yaw mass

        econnectionsMass = 0.0
        vspdEtronicsMass = 0.0
        hydrCoolingMass = 0.08 * self.ratedPower

        # mainframe system including bedplate, platforms, crane and miscellaneous hardware
        nacellePlatformsMass = 0.125 * self.bedplateMass_temp

        if self.crane:
            craneMass = 3000.0
        else:
            craneMass = 0.0

        mainframeMass = self.bedplateMass_temp + craneMass + nacellePlatformsMass

        nacelleCovArea = 2 * self.bedplateLength**2              # this calculation is based on Sunderland
        nacelleCovMass = 84.1 * nacelleCovArea / 2

        self.aboveYawMass = self.lssMass_temp + self.bearingsMass_temp + self.gearboxMass_temp + self.mechBrakeMass_temp + \
            self.generatorMass_temp + mainframeMass + econnectionsMass + vspdEtronicsMass + \
            hydrCoolingMass + nacelleCovMass


        yawfactor = 0.41 * 2.4 * 10 **-3                   # should depend on rotor configuration: blade number and hub type
        weightMom = self.aboveYawMass * rotorDiameter                    # moment due to weight above yaw system
        thrustMom = rotorThrust * self.towerTopDiameter                  # moment due to rotor thrust
        yawSystemMass = (yawfactor * (0.4 * weightMom + 0.975 * thrustMom))

        self.yawSystemMass_temp = yawSystemMass

        return yawSystemMass


    def total(self, x):

        self.yawSystem(x)

        return self.aboveYawMass + self.yawSystemMass_temp



    def calculate_first_derivatives(self):

        x = algopy.UTPM.init_jacobian([self.rotorDiameter, self.rotorMass, self.rotorThrust, self.rotorTorque, self.rotorSpeed])

        d_lss = algopy.UTPM.extract_jacobian(self.lss(x))

        self.derivatives.set_first_derivative('lssMass', 'rotorDiameter', d_lss[0])
        self.derivatives.set_first_derivative('lssMass', 'rotorMass', d_lss[1])
        self.derivatives.set_first_derivative('lssMass', 'rotorTorque', d_lss[3])


        d_bgs = algopy.UTPM.extract_jacobian(self.bearings(x))

        self.derivatives.set_first_derivative('bearingsMass', 'rotorDiameter', d_bgs[0])
        self.derivatives.set_first_derivative('bearingsMass', 'rotorMass', d_bgs[1])
        self.derivatives.set_first_derivative('bearingsMass', 'rotorTorque', d_bgs[3])
        self.derivatives.set_first_derivative('bearingsMass', 'rotorSpeed', d_bgs[4])


        d_gbx = algopy.UTPM.extract_jacobian(self.gearbox(x))

        self.derivatives.set_first_derivative('gearboxMass', 'rotorTorque', d_gbx[3])


        d_mbk = algopy.UTPM.extract_jacobian(self.mechBrake(x))

        self.derivatives.set_first_derivative('mechBrakeMass', 'rotorTorque', d_mbk[3])


        d_bpt = algopy.UTPM.extract_jacobian(self.bedplate(x))

        self.derivatives.set_first_derivative('bedplateMass', 'rotorDiameter', d_bpt[0])
        self.derivatives.set_first_derivative('bedplateMass', 'rotorMass', d_bpt[1])
        self.derivatives.set_first_derivative('bedplateMass', 'rotorThrust', d_bpt[2])
        self.derivatives.set_first_derivative('bedplateMass', 'rotorTorque', d_bpt[3])


        d_yaw = algopy.UTPM.extract_jacobian(self.yawSystem(x))

        self.derivatives.set_first_derivative('yawSystemMass', 'rotorDiameter', d_yaw[0])
        self.derivatives.set_first_derivative('yawSystemMass', 'rotorMass', d_yaw[1])
        self.derivatives.set_first_derivative('yawSystemMass', 'rotorThrust', d_yaw[2])
        self.derivatives.set_first_derivative('yawSystemMass', 'rotorTorque', d_yaw[3])
        self.derivatives.set_first_derivative('yawSystemMass', 'rotorSpeed', d_yaw[4])

        d_total = algopy.UTPM.extract_jacobian(self.total(x))

        self.derivatives.set_first_derivative('totalMass', 'rotorDiameter', d_total[0])
        self.derivatives.set_first_derivative('totalMass', 'rotorMass', d_total[1])
        self.derivatives.set_first_derivative('totalMass', 'rotorThrust', d_total[2])
        self.derivatives.set_first_derivative('totalMass', 'rotorTorque', d_total[3])
        self.derivatives.set_first_derivative('totalMass', 'rotorSpeed', d_total[4])






    def __getParallelStageWeight(self, RotorTorque, stage, StageRatio1, StageRatio2, StageRatio3):

        '''
          This method calculates the stage weight for a parallel stage in a gearbox based on the input torque, stage number, and stage ratio for each individual stage.
        '''

        serviceFact = 1.00                                # default service factor for a gear stage is 1.75 based on full span VP (should depend on control type)
        applicationFact = 0.4                             # application factor ???
        stageweightFact = 8.029  # /2                           # stage weight factor applied to each Gearbox stage

        if (RotorTorque * serviceFact) < 200000.0:       # design factor for design and manufacture of Gearbox
            designFact = 925.0
        elif (RotorTorque * serviceFact) < 700000.0:
            designFact = 1000.0
        else:
            designFact = 1100.0                            # TODO: should be an exception for all 2 stage Gearboxes to have designFact = 1000

        if stage == 1:
            Qr = RotorTorque
            StageRatio = StageRatio1
        elif stage == 2:
            Qr = RotorTorque/StageRatio1
            StageRatio = StageRatio2
        elif stage == 3:
            Qr = RotorTorque/(StageRatio1*StageRatio2)
            StageRatio = StageRatio3

        gearFact = applicationFact / designFact          # Gearbox factor for design, manufacture and application of Gearbox

        gearweightFact = 1 + (1 / StageRatio) + StageRatio + (StageRatio ** 2)
                                                         # Gearbox weight factor for relationship of stage ratio required and relative stage volume

        stageWeight = stageweightFact * Qr * serviceFact * gearFact * gearweightFact
                                                         # forumula for parallel gearstage weight based on sunderland model

        return stageWeight


    def __getEpicyclicStageWeight(self, RotorTorque, stage, StageRatio1, StageRatio2, StageRatio3):
        '''
          This method calculates the stage weight for a epicyclic stage in a gearbox based on the input torque, stage number, and stage ratio for each individual stage
        '''

        serviceFact = 1.00                                # default service factor for a gear stage is 1.75 based on full span VP (should depend on control type)
        applicationFact = 0.4                             # application factor ???
        stageweightFact = 8.029/12                          # stage weight factor applied to each Gearbox stage
        OptWheels = 3.0                                    # default optional wheels (should depend on stage design)

        if (RotorTorque * serviceFact) < 200000.0:       # design factor for design and manufacture of Gearbox
            designFact = 850.0
        elif (RotorTorque * serviceFact) < 700000.0:
            designFact = 950.0
        else:
            designFact = 1100.0

        if stage == 1:
            Qr = RotorTorque
            StageRatio = StageRatio1
        elif stage == 2:
            Qr = RotorTorque/StageRatio1
            StageRatio = StageRatio2
        elif stage == 3:
            Qr = RotorTorque/(StageRatio1*StageRatio2)
            StageRatio = StageRatio3

        gearFact = applicationFact / designFact          # Gearbox factor for design, manufacture and application of Gearbox

        sunwheelratio = (StageRatio / 2.0) - 1             # sun wheel ratio for epicyclic Gearbox stage based on stage ratio
        gearweightFact = (1 / OptWheels) + (1 / (OptWheels * sunwheelratio)) + sunwheelratio + \
                         ((1 + sunwheelratio) / OptWheels) * ((StageRatio - 1.) ** 2)
                                                         # Gearbox weight factor for relationship of stage ratio required and relative stage volume

        stageWeight = stageweightFact * Qr * serviceFact * gearFact * gearweightFact
                                                         # forumula for epicyclic gearstage weight based on sunderland model

        return stageWeight



class RNAMass(Component):

    # in
    rotorMass = Float(iotype='in', units='kg')
    nacelleMass = Float(iotype='in', units='kg')

    # out
    rnaMass = Float(iotype='out', units='kg')

    def __init__(self):
        super(RNAMass, self).__init__()

        self.derivatives.declare_first_derivative('rnaMass', 'rotorMass')
        self.derivatives.declare_first_derivative('rnaMass', 'nacelleMass')

    def execute(self):

        self.rnaMass = self.rotorMass + self.nacelleMass

    def calculate_first_derivatives(self):

        self.derivatives.set_first_derivative('rnaMass', 'rotorMass', 1.0)
        self.derivatives.set_first_derivative('rnaMass', 'nacelleMass', 1.0)



class Turbine(Assembly):

    # variables
    rotorDiameter = Float(iotype='in', units='m')
    rotorThrust = Float(iotype='in', units='N')
    rotorTorque = Float(iotype='in', units='N*m')
    rotorSpeed = Float(iotype='in', units='rpm')

    bladeMass = Float(iotype='in', units='kg', desc='mass of one blade')
    towerMass = Float(iotype='in', units='kg')

    rootMoment = Float(iotype='in', units='N*m', desc='flapwise bending moment at blade root')
    AEP = Float(iotype='in', units='kW*h')


    # parameters
    hubHeight = Float(iotype='in', units='m')
    gearRatio = Float(iotype='in')
    machineRating = Float(iotype='in', units='W')
    towerTopDiameter = Float(iotype='in', units='m')
    crane = Bool(False, iotype='in')
    offshore = Bool(False, iotype='in')
    advancedBlade = Bool(False, iotype='in')
    constructionRate = Float(iotype='in', desc='construction financing rate')
    fixedChargeRate = Float(iotype='in')
    taxRate = Float(iotype='in')

    nBlades = Int(3, iotype='in')
    nTurbines = Int(50, iotype='in')


    def configure(self):

        self.add('derivs', SensitivityDriver())

        self.add('hub', Hub())
        self.add('nacelle', Nacelle())
        self.add('rna', RNAMass())
        self.add('om', OM())
        self.add('bos', BOS())
        self.add('tcc', TCC())
        self.add('coe', COE())

        # self.hub.force_execute = True
        # self.nacelle.force_execute = True
        # self.rna.force_execute = True
        # self.om.force_execute = True
        # self.bos.force_execute = True
        # self.tcc.force_execute = True
        # self.coe.force_execute = True

        self.driver.workflow.add(['hub', 'nacelle', 'rna', 'tcc', 'bos', 'om', 'coe'])

        # objective
        self.derivs.add_objective('coe.COE')

        # design variable
        self.derivs.add_parameter('rotorDiameter', low=0.0, high=1e9)
        self.derivs.add_parameter('rotorThrust', low=0.0, high=1e9)
        self.derivs.add_parameter('rotorTorque', low=0.0, high=1e9)
        self.derivs.add_parameter('rotorSpeed', low=0.0, high=1e9)
        self.derivs.add_parameter('bladeMass', low=0.0, high=1e9)
        self.derivs.add_parameter('towerMass', low=0.0, high=1e9)
        self.derivs.add_parameter('rootMoment', low=0.0, high=1e9)
        self.derivs.add_parameter('AEP', low=0.0, high=1e9)

        # self.derivs.differentiator = Analytic()
        # self.derivs.differentiator = FiniteDifference()
        self.derivs.add('differentiator', FiniteDifference())
        self.derivs.workflow.add(['hub', 'nacelle', 'rna', 'tcc', 'bos', 'om', 'coe'])

        # hub (in)
        self.connect('bladeMass', 'hub.bladeMass')
        self.connect('rootMoment', 'hub.rootMoment')
        self.connect('rotorDiameter', ['hub.rotorDiameter', 'nacelle.rotorDiameter', 'bos.diameter'])
        self.connect('nBlades', ['hub.nBlades', 'tcc.nBlades'])

        # nacelle (in)
        self.connect('hub.rotorMass', 'nacelle.rotorMass')
        self.connect('rotorThrust', 'nacelle.rotorThrust')
        self.connect('rotorTorque', 'nacelle.rotorTorque')
        self.connect('rotorSpeed', 'nacelle.rotorSpeed')
        self.connect('gearRatio', 'nacelle.gearRatio')
        self.connect('machineRating', ['nacelle.machineRating', 'tcc.machineRating',
                     'bos.machineRating', 'om.machineRating'])
        self.connect('towerTopDiameter', 'nacelle.towerTopDiameter')
        self.connect('crane', ['nacelle.crane', 'tcc.crane'])

        # tower top mass
        self.connect('hub.rotorMass', 'rna.rotorMass')
        self.connect('nacelle.totalMass', 'rna.nacelleMass')

        # TCC (in)
        self.connect('bladeMass', 'tcc.bladeMass')
        self.connect('hub.hubMass', 'tcc.hubMass')
        self.connect('hub.pitchMass', 'tcc.pitchSystemMass')
        self.connect('hub.spinnerMass', 'tcc.spinnerMass')

        self.connect('nacelle.lssMass', 'tcc.lssMass')
        self.connect('nacelle.bearingsMass', 'tcc.bearingsMass')
        self.connect('nacelle.gearboxMass', 'tcc.gearboxMass')
        self.connect('nacelle.mechBrakeMass', 'tcc.mechBrakeMass')
        self.connect('nacelle.generatorMass', 'tcc.generatorMass')
        self.connect('nacelle.bedplateMass', 'tcc.bedplateMass')
        self.connect('nacelle.yawSystemMass', 'tcc.yawSystemMass')
        self.connect('towerMass', 'tcc.towerMass')
        self.connect('offshore', ['tcc.offshore', 'om.offshore'])
        self.connect('advancedBlade', 'tcc.advancedBlade')

        # BOS
        self.connect('hubHeight', 'bos.hubHeight')
        self.connect('tcc.TCC', 'bos.TCC')
        self.connect('rna.rnaMass', 'bos.towerTopMass')
        self.connect('nTurbines', 'bos.nTurbines')

        # OM
        self.connect('AEP', 'om.AEP')

        # COE
        self.connect('tcc.TCC', 'coe.TCC')
        self.connect('bos.BOS', 'coe.BOS')
        self.connect('om.OM', 'coe.OM')
        self.connect('om.LLC', 'coe.LLC')
        self.connect('om.LRC', 'coe.LRC')
        self.connect('AEP', 'coe.AEP')

        self.connect('constructionRate', 'coe.constructionRate')
        self.connect('fixedChargeRate', 'coe.fixedChargeRate')
        self.connect('taxRate', 'coe.taxRate')
        self.connect('nTurbines', 'coe.nTurbines')


        # parameters
        self.create_passthrough('coe.COE')







def fd_check(comp, outputs, inputs, simplePrint=True, supplyAnalytic=[]):


    idx = 0

    for f in outputs:
        for x in inputs:

            # save center point
            comp.run()
            center = getattr(comp, f)

            # choose step size
            delta = getattr(comp, x) * 1e-6
            if delta == 0:
                delta = 1e-6

            # finite difference
            setattr(comp, x, getattr(comp, x)+delta)
            comp.run()
            fd = (getattr(comp, f) - center) / delta
            setattr(comp, x, getattr(comp, x)-delta)

            # analytic
            if len(supplyAnalytic) > 0:
                analytic = supplyAnalytic[idx]
                idx += 1

            else:
                comp.calc_derivatives(first=True)
                danalytic = comp.derivatives.first_derivatives

                # parse dictionary for analytic
                if f in danalytic:
                    danl = danalytic[f]
                    if x in danl:
                        analytic = danl[x]
                    else:
                        analytic = 0.0
                else:
                    analytic = 0.0



            # print results
            if not simplePrint:

                print 'd', f, 'wrt', x
                print 'fd, analytic =', fd, analytic

                if analytic == 0:
                    print 'diff =', fd - analytic
                else:
                    print '% error =', (fd - analytic)/analytic*100
                print

            else:

                if analytic == 0:
                    print fd - analytic
                else:
                    print (fd - analytic)/analytic*100





if __name__ == '__main__':

    coe = COE()

    nTurbines = 50
    coe.TCC = 285132158.931 / nTurbines
    coe.BOS = 103410574.109
    coe.OM = 9080318.68467 / nTurbines
    coe.LLC = 1400963.45421 / nTurbines
    coe.LRC = 2934961.12731 / nTurbines
    coe.AEP = 1182291272.5 / nTurbines

    coe.constructionRate = 0.03
    coe.fixedChargeRate = 0.12
    coe.taxRate = 0.4
    coe.nTurbines = nTurbines

    inputs = ['TCC', 'BOS', 'OM', 'LLC', 'LRC', 'AEP']
    outputs = ['COE']

    # fd_check(coe, outputs, inputs)



    tcc = TCC()
    tcc.bladeMass = 17152.9500422
    tcc.hubMass = 76544.1350962
    tcc.pitchSystemMass = 20441.9501062
    tcc.spinnerMass = 1810.5
    tcc.lssMass = 34268.6997029
    tcc.bearingsMass = 11021.9195465
    tcc.gearboxMass = 34190.7469405
    tcc.mechBrakeMass = 1687.56426346
    tcc.generatorMass = 16699.851325
    tcc.bedplateMass = 95544.3788875
    tcc.yawSystemMass = 13539.0835357
    tcc.towerMass = 349486.79362

    tcc.machineRating = 5e6
    tcc.offshore = False
    tcc.crane = True
    tcc.nBlades = 3
    tcc.advancedBlade = True

    inputs = ['bladeMass', 'hubMass', 'pitchSystemMass', 'spinnerMass', 'lssMass', 'bearingsMass', 'gearboxMass', 'mechBrakeMass', 'generatorMass', 'bedplateMass', 'yawSystemMass', 'towerMass']
    outputs = ['TCC']
    # fd_check(tcc, outputs, inputs)



    bos = BOS()

    bos.diameter = 126.0
    bos.hubHeight = 89.6
    bos.TCC = 285132158.931 / 50.0
    bos.towerTopMass = 359395.660756

    bos.nTurbines = 50
    bos.machineRating = 5e6

    inputs = ['diameter', 'hubHeight', 'TCC', 'towerTopMass']
    outputs = ['BOS']
    # fd_check(bos, outputs, inputs)


    om = OM()

    om.AEP = 23645825.4499

    om.offshore = False
    om.machineRating = 5e6

    inputs = ['AEP']
    outputs = ['OM']
    outputs = ['LLC']
    outputs = ['LRC']
    # fd_check(om, outputs, inputs)


    hub = Hub()

    hub.rootMoment = 11389434.6233
    hub.bladeMass = 17152.9500422
    hub.rotorDiameter = 126.0

    hub.nBlades = 3

    # hub.run()

    inputs = ['bladeMass', 'rootMoment', 'rotorDiameter']
    outputs= ['hubMass', 'pitchMass', 'spinnerMass', 'rotorMass']
    # fd_check(hub, outputs, inputs)


    # print hub.hubMass + hub.spinnerMass + hub.pitchMass
    # # 76544.1350962


    nac = Nacelle()
    nac.rotorDiameter = 126.0
    nac.rotorMass = 128002.985223
    nac.rotorThrust = 736609.076723
    nac.rotorTorque = 4365166.22816
    nac.rotorSpeed = 12.1260909022

    # parameters
    nac.gearRatio = 97.0
    nac.machineRating = 5e6
    nac.towerTopDiameter = 3.87
    nac.crane = True

    # nac.run()

    inputs = ['rotorDiameter', 'rotorMass', 'rotorThrust', 'rotorTorque', 'rotorSpeed']
    outputs= ['lssMass', 'bearingsMass', 'gearboxMass', 'mechBrakeMass', 'generatorMass', 'bedplateMass', 'yawSystemMass', 'totalMass']

    # fd_check(nac, outputs, inputs)


    rna = RNAMass()
    rotorMass = 17152.9500422*3
    nacelleMass = 17152.9500422*10

    # out
    rnaMass = Float(iotype='out', units='kg')

    inputs = ['rotorMass', 'nacelleMass']
    outputs= ['rnaMass']

    # fd_check(rna, outputs, inputs)




    turbine = Turbine()
    turbine.rotorDiameter = 126.0
    turbine.rotorThrust = 736609.076723
    turbine.rotorTorque = 4365166.22816
    turbine.rotorSpeed = 12.1260909022

    turbine.bladeMass = 17152.9500422
    turbine.towerMass = 349486.79362

    turbine.rootMoment = 11389434.6233
    turbine.AEP = 1182291272.5/50


    # parameters
    turbine.hubHeight = 89.6
    turbine.gearRatio = 97.0
    turbine.machineRating = 5e6
    turbine.towerTopDiameter = 3.87
    turbine.crane = True
    turbine.offshore = False
    turbine.advancedBlade = True
    turbine.constructionRate = 0.03
    turbine.fixedChargeRate = 0.12
    turbine.taxRate = 0.4

    turbine.nBlades = 3
    turbine.nTurbines = 50

    turbine.run()
    turbine.derivs.run()

    print "F: ", turbine.derivs.F
    print "dF: ", turbine.derivs.dF

    inputs = ['rotorDiameter', 'rotorThrust', 'rotorTorque', 'rotorSpeed', 'bladeMass', 'towerMass', 'rootMoment', 'AEP']
    outputs = ['COE']
    fd_check(turbine, outputs, inputs, supplyAnalytic=turbine.derivs.dF[0])



    # print turbine.coe.TCC*turbine.nTurbines, turbine.coe.BOS, turbine.coe.OM*turbine.nTurbines, \
    #     turbine.coe.LLC*turbine.nTurbines, turbine.coe.LRC*turbine.nTurbines, \
    #     turbine.coe.AEP*turbine.nTurbines, turbine.COE

    # 278961088.219 103367061.145 9080318.68467 1400963.45421 2934961.12731 1182291272.5 0.0482451967469
    # 278961088.219 103367061.145 9080318.68467 1400963.45421 2934961.1273  1182291272.5 0.0482451967467


    # turbine.calc_derivatives(first=True)
    # print turbine.derivatives.first_derivatives

