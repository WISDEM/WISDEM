"""
LCOE_csm_ssembly.py

Created by NWTC Systems Engineering Sub-Task on 2012-08-01.
Copyright (c) NREL. All rights reserved.
"""

from openmdao.main.datatypes.api import Int, Float

from fusedwind.plant_cost.fused_fin_asym import ExtendedFinancialAnalysis

# NREL cost and scaling model sub-assemblies
from TurbineSE.turbine import TurbineSE
from Turbine_CostsSE.Turbine_CostsSE.turbine_costsSE import Turbine_CostsSE
from Plant_CostsSE.Plant_BOS.NREL_CSM_BOS.nrel_csm_bos import bos_csm_assembly
from Plant_CostsSE.Plant_OM.NREL_CSM_OM.nrel_csm_om import om_csm_assembly
from Plant_FinanceSE.NREL_CSM_FIN.nrel_csm_fin import fin_csm_assembly
from Plant_AEPSE.Basic_AEP.basic_aep import aep_assembly


class lcoe_se_csm_assembly(ExtendedFinancialAnalysis):

    # parameters
    sea_depth = Float(0.0, units='m', iotype='in', desc='sea depth for offshore wind project')
    drivetrain_design = Int(1, iotype='in', desc='drivetrain design type 1 = 3-stage geared, 2 = single-stage geared, 3 = multi-generator, 4 = direct drive')
    turbine_number = Int(100, iotype='in', desc='total number of wind turbines at the plant')
    year = Int(2009, iotype='in', desc='year of project start')
    month = Int(12, iotype='in', desc='month of project start')
    array_losses = Float(0.059, iotype='in', desc='energy losses due to turbine interactions - across entire plant')
    other_losses = Float(0.0, iotype='in', desc='energy losses due to blade soiling, electrical, etc')
    availability = Float(0.94, iotype='in', desc='average annual availbility of wind turbines at plant')


    def configure(self):
        super(lcoe_se_csm_assembly, self).configure()

        self.replace('tcc_a', Turbine_CostsSE())
        self.replace('bos_a', bos_csm_assembly())
        self.replace('opex_a', om_csm_assembly())
        self.replace('aep_a', aep_assembly())
        self.replace('fin_a', fin_csm_assembly())

        self.add('turbine', TurbineSE(3))

        self.driver.workflow.add('turbine')

        # connect i/o to component and assembly inputs
        # turbine configuration
        # rotor
        self.connect('turbine.nBlades', 'tcc_a.blade_number')
        self.connect('turbine.rotor_diameter', 'bos_a.rotor_diameter')
        # drivetrain
        self.connect('turbine.machine_rating', ['bos_a.machine_rating', 'opex_a.machine_rating'])
        # self.connect('drivetrain_design', ['turbine.drivetrain_design', 'tcc_a.drivetrain_design'])
        self.connect('drivetrain_design', 'tcc_a.drivetrain_design')  # TODO: make drivetrain connection to turbine
        self.connect('turbine.crane', 'tcc_a.crane')
        # tower
        self.connect('turbine.hub_height', ['bos_a.hub_height'])
        # plant configuration
        # climate
        self.connect('sea_depth', ['bos_a.sea_depth', 'opex_a.sea_depth', 'fin_a.sea_depth'])
        # plant operation
        self.connect('turbine_number', ['aep_a.turbine_number', 'bos_a.turbine_number', 'opex_a.turbine_number', 'fin_a.turbine_number'])
        # financial
        self.connect('year', ['tcc_a.year', 'bos_a.year', 'opex_a.year'])
        self.connect('month', ['tcc_a.month', 'bos_a.month', 'opex_a.month'])

        # inter-model connections
        self.connect('turbine.AEP', 'aep_a.AEP_one_turbine')
        self.connect('turbine.blade_mass', 'tcc_a.blade_mass')
        self.connect('turbine.hub_mass', 'tcc_a.hub_mass')
        self.connect('turbine.pitch_system_mass', 'tcc_a.pitch_system_mass')
        self.connect('turbine.spinner_mass', 'tcc_a.spinner_mass')
        self.connect('turbine.low_speed_shaft_mass', 'tcc_a.low_speed_shaft_mass')
        self.connect('turbine.main_bearing_mass', 'tcc_a.main_bearing_mass')
        self.connect('turbine.second_bearing_mass', 'tcc_a.second_bearing_mass')
        self.connect('turbine.gearbox_mass', 'tcc_a.gearbox_mass')
        self.connect('turbine.high_speed_side_mass', 'tcc_a.high_speed_side_mass')
        self.connect('turbine.generator_mass', 'tcc_a.generator_mass')
        self.connect('turbine.bedplate_mass', 'tcc_a.bedplate_mass')
        self.connect('turbine.yaw_system_mass', 'tcc_a.yaw_system_mass')
        self.connect('aep_a.net_aep', 'opex_a.net_aep')

        # create passthroughs for key input variables of interest
        # turbine
        self.create_passthrough('tcc_a.advanced_blade')
        self.create_passthrough('tcc_a.offshore')  # todo connections

        # connect to outputs
        self.connect('other_losses', 'aep_a.other_losses')
        self.connect('array_losses', 'aep_a.array_losses')
        self.connect('availability', 'aep_a.availability')

        # create passthroughs for key output variables of interest
        self.create_passthrough('fin_a.fixed_charge_rate')
        self.create_passthrough('turbine.power_curve')
        self.create_passthrough('turbine.turbine_mass')



def example():

    lcoe = lcoe_se_csm_assembly()

    lcoe.turbine.crane = True
    lcoe.turbine.gear_configuration = 'eep'
    lcoe.turbine.gear_ratio = 97.0
    lcoe.drivetrain_design = 1
    lcoe.offshore = False
    lcoe.A = 8.35
    lcoe.k = 2.15

    lcoe.execute()

    print "COE: {0}".format(lcoe.coe)
    print "\n"
    print "AEP per turbine: {0}".format(lcoe.net_aep / lcoe.turbine_number)
    print "Turbine _cost: {0}".format(lcoe.turbine_cost)
    print "BOS costs per turbine: {0}".format(lcoe.bos_costs / lcoe.turbine_number)
    print "OnM costs per turbine: {0}".format(lcoe.avg_annual_opex / lcoe.turbine_number)

    #print "Turbine variable tree:"
    #lcoe.turbineVT.printVT()
    #print


def tipspeed():

    import os
    import numpy as np
    from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection  # TODO: can just pass file names and do this initialization inside of rotor
    from towerse.tower import TowerWithpBEAM
    from commonse.environment import PowerWind, TowerSoil

    lcoe = lcoe_se_csm_assembly()

    turbine = lcoe.turbine
    rotor = turbine.rotor
    nacelle = turbine.nacelle
    tower = turbine.tower

    # --- blade grid ---
    rotor.initial_aero_grid = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667, 0.43333333,
        0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724])
    rotor.initial_str_grid = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154, 0.0114340970275,
        0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455, 0.11111057,
        0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319, 0.36666667,
        0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333, 0.667358391486,
        0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724, 1.0])
    rotor.idx_cylinder_aero = 3
    rotor.idx_cylinder_str = 14
    rotor.hubFraction = 0.025

    # --- geometry -----
    rotor.hubHt = 90.0
    rotor.precone = 2.5
    rotor.tilt = 5.0
    rotor.yaw = 0.0
    rotor.nBlades = 3
    rotor.turbine_class = 'I'
    rotor.turbulence_class = 'B'

    # --- atmosphere ---
    rotor.rho = 1.225
    rotor.mu = 1.81206e-5
    rotor.shearExp = 0.2

    # --- operational conditions ---
    rotor.control.Vin = 3.0
    rotor.control.Vout = 25.0
    rotor.control.ratedPower = 5e6
    rotor.control.minOmega = 0.0
    rotor.control.maxOmega = 12.0
    rotor.control.tsr = 7.55
    rotor.control.pitch = 0.0

    rotor.pitch_extreme = 0.0
    rotor.azimuth_extreme = 0.0


    # --- airfoil files ---
    basepath = os.path.join('5MW_files', '5MW_AFFiles')

    # load all airfoils
    airfoil_types = [0]*8
    airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
    airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
    airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
    airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
    airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
    airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
    airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
    airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

    # place at appropriate radial stations
    af_idx = [0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7]

    n = len(af_idx)
    af = [0]*n
    for i in range(n):
        af[i] = airfoil_types[af_idx[i]]
    rotor.airfoil_files = af

    # --- aero analysis options ---
    rotor.npts_coarse_power_curve = 20
    rotor.npts_spline_power_curve = 200
    rotor.AEP_loss_factor = 1.0
    rotor.nSector = 4
    rotor.drivetrainType = 'geared'

    # --- materials and composite layup  ---
    basepath = os.path.join('5MW_files', '5MW_PrecompFiles')

    materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

    ncomp = len(rotor.initial_str_grid)
    upper = [0]*ncomp
    lower = [0]*ncomp
    webs = [0]*ncomp
    profile = [0]*ncomp

    rotor.leLoc = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    rotor.sector_idx_strain_spar = [2]*ncomp
    rotor.sector_idx_strain_te = [3]*ncomp
    web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899, 0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
    web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, 0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
    web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    rotor.chord_str_ref = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335, 3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643, 4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022, 4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502, 3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331, 2.05553464181, 1.82577817774, 1.5860853279, 1.4621])

    for i in range(ncomp):

        webLoc = []
        if web1[i] != -1:
            webLoc.append(web1[i])
        if web2[i] != -1:
            webLoc.append(web2[i])
        if web3[i] != -1:
            webLoc.append(web3[i])

        upper[i], lower[i], webs[i] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(i+1) + '.inp'), webLoc, materials)
        profile[i] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(i+1) + '.inp'))

    rotor.materials = materials
    rotor.upperCS = upper
    rotor.lowerCS = lower
    rotor.websCS = webs
    rotor.profile = profile
    # --------------------------------------


    # --- structural options ---
    rotor.g = 9.81
    rotor.nF = 5
    rotor.dynamic_amplication_tip_deflection = 1.2

    # --- rotor design variables ---
    rotor.r_aero = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333, 0.5,
        0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333, 0.97777724])
    rotor.r_max_chord = 0.23577
    rotor.chord_sub = [3.2612, 4.5709, 3.3178, 1.4621]
    rotor.theta_sub = [13.2783, 7.46036, 2.89317, -0.0878099]
    rotor.bladeLength = 63.0
    rotor.sparT = [1.0, 0.047754, 0.045376, 0.031085, 0.0061398]
    rotor.teT = [1.0, 0.09569, 0.06569, 0.02569, 0.00569]



    # ---- nacelle --------

    nacelle.gear_ratio = 87.965
    nacelle.drivetrain_design = 1  # TODO: sync with my drivetrainType variable
    nacelle.crane = True
    nacelle.bevel = 0
    nacelle.gear_configuration = 'epp'


    # ---- tower ------
    tower.replace('wind1', PowerWind())
    tower.replace('wind2', PowerWind())
    # tower.replace('wave1', LinearWaves())  # no waves
    tower.replace('soil', TowerSoil())
    tower.replace('tower1', TowerWithpBEAM())
    tower.replace('tower2', TowerWithpBEAM())

    # geometry
    tower.z = np.array([0.0, 0.5, 1.0])
    # tower.d = [6.0, 4.935, 3.87]
    tower.d[0:2] = [6.0, 4.935]
    turbine.tower_top_diameter = 3.87
    tower.t = [0.027*1.3, 0.023*1.3, 0.019*1.3]
    tower.n = [10, 10]
    tower.n_reinforced = 3

    tower.gamma_f = 1.35
    tower.gamma_m = 1.3
    tower.gamma_n = 1.0

    # damage
    tower.z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    tower.M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    tower.gamma_fatigue = 1.35*1.3*1.0

    # wind
    tower.wind1.shearExp = rotor.shearExp
    tower.wind2.shearExp = rotor.shearExp

    # soil
    tower.soil.rigid = 6*[True]


    lcoe.run()


    print "COE: {0}".format(lcoe.coe)
    print "AEP per turbine: {0}".format(lcoe.net_aep / lcoe.turbine_number)
    print "Turbine _cost: {0}".format(lcoe.turbine_cost)
    print "BOS costs per turbine: {0}".format(lcoe.bos_costs / lcoe.turbine_number)
    print "OnM costs per turbine: {0}".format(lcoe.avg_annual_opex / lcoe.turbine_number)


    # # outputs
    # print 'AEP =', rotor.AEP
    # print 'diameter =', rotor.diameter
    # print 'ratedConditions.V =', rotor.ratedConditions.V
    # print 'ratedConditions.Omega =', rotor.ratedConditions.Omega
    # print 'ratedConditions.pitch =', rotor.ratedConditions.pitch
    # print 'ratedConditions.T =', rotor.ratedConditions.T
    # print 'ratedConditions.Q =', rotor.ratedConditions.Q
    # print 'mass_one_blade =', rotor.mass_one_blade
    # print 'mass_all_blades =', rotor.mass_all_blades
    # print 'I_all_blades =', rotor.I_all_blades
    # print 'freq =', rotor.freq
    # print 'tip_deflection =', rotor.tip_deflection
    # print 'root_bending_moment =', rotor.root_bending_moment

    # # outputs
    # print 'tower mass =', tower.mass
    # print 'tower f1 =', tower.f1
    # print 'tower f2 =', tower.f2
    # print 'tower top_deflection1 =', tower.top_deflection1
    # print 'tower top_deflection2 =', tower.top_deflection2
    # print 'tower z_buckling =', tower.z_buckling


    eta_strain = 1.35*1.3*1.0
    eta_dfl = 1.35*1.1*1.0
    strain_ult_spar = 1.0e-2
    strain_ult_te = 2500*1e-6
    idx_strain = [0, 12, 14, 18, 22, 28]
    idx_buckling = [10, 12, 14, 20, 27, 31]
    freq_margin = 1.1
    min_ground_clearance = 20.0

    dt_min = 120.0
    tower_max_taper = 0.4
    idx_tower_stress = [0, 4, 8, 10, 12, 14]
    idx_tower_fatigue = [0, 3, 6, 9, 12, 15, 18, 20]

    print
    print 'rotor constraints:'
    print
    print 'tower strike =', rotor.tip_deflection * eta_dfl / turbine.max_tip_deflection - 1
    print 'ground strike =', min_ground_clearance/turbine.ground_clearance - 1
    print 'flap/edge freq = ', rotor.nBlades*rotor.ratedConditions.Omega/60.0*freq_margin - rotor.freq[0:2]
    print 'rotor strain sparU =', rotor.strainU_spar[idx_strain]*eta_strain/strain_ult_spar
    print 'rotor strain sparL =', rotor.strainL_spar[idx_strain]*eta_strain/strain_ult_spar
    print 'rotor strain teU =', rotor.strainU_te[idx_strain]*eta_strain/strain_ult_te
    print 'rotor strain teL =', rotor.strainL_te[idx_strain]*eta_strain/strain_ult_te
    print 'rotor buckling spar =', (rotor.strainU_spar[idx_buckling] - rotor.eps_crit_spar[idx_buckling]) / strain_ult_spar
    print 'rotor buckling te =', (rotor.strainU_te[idx_buckling] - rotor.eps_crit_te[idx_buckling]) / strain_ult_te

    print
    print 'tower constraints:'
    print
    print 'weldability =', (dt_min - tower.d/tower.t) / dt_min
    print 'manufactuability =', tower_max_taper - tower.d[-1] / tower.d[0]
    print 'freq_margin_tower =', rotor.ratedConditions.Omega/60.0*freq_margin - tower.f1
    print 'tower buckling1 =', tower.buckling1
    print 'tower buckling2 =', tower.buckling2
    print 'tower stress1 =', tower.stress1[idx_tower_stress]
    print 'tower stress2 =', tower.stress2[idx_tower_stress]
    print 'tower damage =', tower.damage[idx_tower_fatigue] - 1



    # idx_vm = [0, 4, 8, 10, 12]


    # # stress
    # stress_margin_tower = eta_tower*von_mises[::4][idx_vm] / sigma_y - 1


    import matplotlib.pyplot as plt
    plt.plot(rotor.V, rotor.P)
    plt.figure()
    plt.plot(rotor.spline.r_str, rotor.strainU_spar)
    plt.plot(rotor.spline.r_str[idx_strain], rotor.strainU_spar[idx_strain], 'x')
    plt.plot(rotor.spline.r_str, rotor.strainL_spar)
    plt.plot(rotor.spline.r_str[idx_strain], rotor.strainL_spar[idx_strain], 'x')
    plt.plot(rotor.spline.r_str, rotor.eps_crit_spar)
    plt.plot(rotor.spline.r_str[idx_buckling], rotor.eps_crit_spar[idx_buckling], 'x')
    plt.plot(rotor.spline.r_str, strain_ult_spar/eta_strain*np.ones_like(rotor.spline.r_str), 'k--')
    plt.plot(rotor.spline.r_str, -strain_ult_spar/eta_strain*np.ones_like(rotor.spline.r_str), 'k--')
    plt.ylim([-6e-3, 6e-3])
    plt.figure()
    plt.plot(rotor.spline.r_str, rotor.strainU_te)
    plt.plot(rotor.spline.r_str[idx_strain], rotor.strainU_te[idx_strain], 'x')
    plt.plot(rotor.spline.r_str, rotor.strainL_te)
    plt.plot(rotor.spline.r_str[idx_strain], rotor.strainL_te[idx_strain], 'x')
    plt.plot(rotor.spline.r_str, rotor.eps_crit_te)
    plt.plot(rotor.spline.r_str, strain_ult_te/eta_strain*np.ones_like(rotor.spline.r_str), 'k--')
    plt.plot(rotor.spline.r_str, -strain_ult_te/eta_strain*np.ones_like(rotor.spline.r_str), 'k--')
    plt.ylim([-6e-3, 6e-3])
    plt.figure()
    plt.plot(tower.stress1, tower.z_nodes)
    plt.plot(tower.stress1[idx_tower_stress], tower.z_nodes[idx_tower_stress], 'x')
    plt.plot(tower.stress2, tower.z_nodes)
    plt.plot(tower.stress2[idx_tower_stress], tower.z_nodes[idx_tower_stress], 'x')
    plt.plot(tower.buckling1, tower.z_buckling)
    plt.plot(tower.buckling2, tower.z_buckling)
    plt.plot(tower.damage - 1, tower.z_nodes)
    plt.plot(tower.damage[idx_tower_fatigue] - 1, tower.z_nodes[idx_tower_fatigue], 'x')
    plt.show()




if __name__=="__main__":

    tipspeed()