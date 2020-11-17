import numpy as np
import openmdao.api as om
import os

class PoseOptimization(object):
    def __init__(self, modeling_options, analysis_options):
        self.modeling    = modeling_options
        self.opt         = analysis_options

        
    def get_number_design_variables(self):
        # Determine the number of design variables
        n_DV = 0

        blade_opt   = self.opt['optimization_variables']['blade']
        tower_opt   = self.opt['optimization_variables']['tower']
        mono_opt    = self.opt['optimization_variables']['monopile']
        
        if blade_opt['aero_shape']['twist']['flag']:
            n_DV += blade_opt['aero_shape']['twist']['n_opt'] - 2
        if blade_opt['aero_shape']['chord']['flag']:
            n_DV += blade_opt['aero_shape']['chord']['n_opt'] - 3
        if blade_opt['aero_shape']['af_positions']['flag']:
            n_DV += self.modeling['blade']['n_af_span'] - blade_opt['aero_shape']['af_positions']['af_start'] - 1
        if blade_opt['structure']['spar_cap_ss']['flag']:
            n_DV += blade_opt['structure']['spar_cap_ss']['n_opt'] - 2
        if blade_opt['structure']['spar_cap_ps']['flag'] and not blade_opt['structure']['spar_cap_ps']['equal_to_suction']:
            n_DV += blade_opt['structure']['spar_cap_ps']['n_opt'] - 2
        if self.opt['optimization_variables']['control']['tsr']['flag']:
            n_DV += 1
        if self.opt['optimization_variables']['control']['servo']['pitch_control']['flag']:
            n_DV += 2
        if self.opt['optimization_variables']['control']['servo']['torque_control']['flag']:
            n_DV += 2
        if tower_opt['outer_diameter']['flag']:
            n_DV += self.modeling['tower']['n_height']
        if tower_opt['layer_thickness']['flag']:
            n_DV += (self.modeling['tower']['n_height'] - 1) * self.modeling['tower']['n_layers']
        if mono_opt['outer_diameter']['flag']:
            n_DV += self.modeling['monopile']['n_height']
        if mono_opt['layer_thickness']['flag']:
            n_DV += (self.modeling['monopile']['n_height'] - 1) * self.modeling['monopile']['n_layers']

        if self.opt['driver']['form'] == 'central':
            n_DV *= 2

        return n_DV

    
    def _get_step_size(self):
        # If a step size for the driver-level finite differencing is provided, use that step size. Otherwise use a default value.
        return (1.e-6 if not 'step_size' in self.opt['driver'] else self.opt['driver']['step_size'])

    
    def set_driver(self, wt_opt):
        folder_output = self.opt['general']['folder_output']

        step_size = self._get_step_size()

        # Solver has specific meaning in OpenMDAO
        wt_opt.model.approx_totals(method='fd', step=step_size, form=self.opt['driver']['form'])

        # Set optimization solver and options. First, Scipy's SLSQP
        if self.opt['driver']['solver'] == 'SLSQP':
            wt_opt.driver  = om.ScipyOptimizeDriver()
            wt_opt.driver.options['optimizer'] = self.opt['driver']['solver']
            wt_opt.driver.options['tol']       = self.opt['driver']['tol']
            wt_opt.driver.options['maxiter']   = self.opt['driver']['max_iter']

        # The next two optimization methods require pyOptSparse.
        elif self.opt['driver']['solver'] == 'CONMIN':
            try:
                from openmdao.api import pyOptSparseDriver
            except:
                raise ImportError('You requested the optimization solver CONMIN, but you have not installed the pyOptSparseDriver. Please do so and rerun.')
            wt_opt.driver = pyOptSparseDriver()
            wt_opt.driver.options['optimizer'] = self.opt['driver']['solver']
            wt_opt.driver.opt_settings['ITMAX']= self.opt['driver']['max_iter']

        elif self.opt['driver']['solver'] == 'SNOPT':
            try:
                from openmdao.api import pyOptSparseDriver
            except:
                raise ImportError('You requested the optimization solver SNOPT, but you have not installed the pyOptSparseDriver. Please do so and rerun.')
            wt_opt.driver = pyOptSparseDriver()
            try:
                wt_opt.driver.options['optimizer']                       = self.opt['driver']['solver']
            except:
                raise ImportError('You requested the optimization solver SNOPT, but you have not installed it within the pyOptSparseDriver. Please do so and rerun.')
            wt_opt.driver.opt_settings['Major optimality tolerance']  = float(self.opt['driver']['tol'])
            wt_opt.driver.opt_settings['Major iterations limit']      = int(self.opt['driver']['max_major_iter'])
            wt_opt.driver.opt_settings['Iterations limit']            = int(self.opt['driver']['max_minor_iter'])
            wt_opt.driver.opt_settings['Major feasibility tolerance'] = float(self.opt['driver']['tol'])
            wt_opt.driver.opt_settings['Summary file']                = os.path.join(folder_output, 'SNOPT_Summary_file.txt')
            wt_opt.driver.opt_settings['Print file']                  = os.path.join(folder_output, 'SNOPT_Print_file.txt')
            if 'hist_file_name' in self.opt['driver']:
                wt_opt.driver.hist_file = self.opt['driver']['hist_file_name']
            if 'verify_level' in self.opt['driver']:
                wt_opt.driver.opt_settings['Verify level'] = self.opt['driver']['verify_level']
            # wt_opt.driver.declare_coloring()
            if 'hotstart_file' in self.opt['driver']:
                wt_opt.driver.hotstart_file = self.opt['driver']['hotstart_file']

        else:
            raise ValueError('The optimizer ' + self.opt['driver']['solver'] + 'is not yet supported!')
        
        return wt_opt

    
    def set_objective(self, wt_opt):

        # Set merit figure. Each objective has its own scaling.
        if self.opt['merit_figure'] == 'AEP':
            wt_opt.model.add_objective('sse.AEP', ref = -1.e6)

        elif self.opt['merit_figure'] == 'blade_mass':
            wt_opt.model.add_objective('elastic.precomp.blade_mass', ref = 1.e4)

        elif self.opt['merit_figure'] == 'LCOE':
            wt_opt.model.add_objective('financese.lcoe', ref = 0.1)

        elif self.opt['merit_figure'] == 'blade_tip_deflection':
            wt_opt.model.add_objective('tcons.tip_deflection_ratio')

        elif self.opt['merit_figure'] == 'tower_mass':
            wt_opt.model.add_objective('towerse.tower_mass', scaler=1e-6)

        elif self.opt['merit_figure'] == 'mononpile_mass':
            wt_opt.model.add_objective('towerse.mononpile_mass', ref=1.e6)

        elif self.opt['merit_figure'] == 'structural_mass':
            wt_opt.model.add_objective('towerse.structural_mass', ref=1.e6)

        elif self.opt['merit_figure'] == 'tower_cost':
            wt_opt.model.add_objective('tcc.tower_cost', ref=1.e6)

        elif self.opt['merit_figure'] == 'Cp':
            if self.modeling['flags']['blade']:
                wt_opt.model.add_objective('sse.powercurve.Cp_regII', ref = -1.)
            else:
                wt_opt.model.add_objective('ccblade.CP', ref = -1.)
        else:
            raise ValueError('The merit figure ' + self.opt['merit_figure'] + ' is not supported.')
                
        return wt_opt

    
    def set_design_variables(self, wt_opt, wt_init):

        # Set optimization design variables.
        blade_opt    = self.opt['optimization_variables']['blade']
        tower_opt    = self.opt['optimization_variables']['tower']
        monopile_opt = self.opt['optimization_variables']['monopile']
        control_opt  = self.opt['optimization_variables']['control']

        if blade_opt['aero_shape']['twist']['flag']:
            indices        = range(2, blade_opt['aero_shape']['twist']['n_opt'])
            wt_opt.model.add_design_var('blade.opt_var.twist_opt_gain', indices = indices, lower=0., upper=1.)

        chord_options = blade_opt['aero_shape']['chord']
        if chord_options['flag']:
            indices  = range(3, chord_options['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.chord_opt_gain', indices = indices, lower=chord_options['min_gain'], upper=chord_options['max_gain'])

        if blade_opt['aero_shape']['af_positions']['flag']:
            n_af = self.modeling['blade']['n_af_span']
            indices  = range(blade_opt['aero_shape']['af_positions']['af_start'],n_af - 1)
            af_pos_init = wt_init['components']['blade']['outer_shape_bem']['airfoil_position']['grid']
            step_size = self._get_step_size()
            lb_af    = np.zeros(n_af)
            ub_af    = np.zeros(n_af)
            for i in range(1,indices[0]):
                lb_af[i]    = ub_af[i] = af_pos_init[i]
            for i in indices:
                lb_af[i]    = 0.5*(af_pos_init[i-1] + af_pos_init[i]) + step_size
                ub_af[i]    = 0.5*(af_pos_init[i+1] + af_pos_init[i]) - step_size
            lb_af[-1] = ub_af[-1] = 1.
            wt_opt.model.add_design_var('blade.opt_var.af_position', indices = indices, lower=lb_af[indices], upper=ub_af[indices])

        spar_cap_ss_options = blade_opt['structure']['spar_cap_ss']
        if spar_cap_ss_options['flag']:
            indices  = range(1,spar_cap_ss_options['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.spar_cap_ss_opt_gain', indices = indices, lower=spar_cap_ss_options['min_gain'], upper=spar_cap_ss_options['max_gain'])

        # Only add the pressure side design variables if we do set
        # `equal_to_suction` as False in the optimization yaml.
        spar_cap_ps_options = blade_opt['structure']['spar_cap_ps']
        if spar_cap_ps_options['flag'] and not spar_cap_ps_options['equal_to_suction']:
            indices  = range(1, spar_cap_ps_options['n_opt'] - 1)
            wt_opt.model.add_design_var('blade.opt_var.spar_cap_ps_opt_gain', indices = indices, lower=spar_cap_ps_options['min_gain'], upper=spar_cap_ps_options['max_gain'])

        if tower_opt['outer_diameter']['flag']:
            wt_opt.model.add_design_var('tower.diameter', lower=tower_opt['outer_diameter']['lower_bound'], upper=tower_opt['outer_diameter']['upper_bound'], ref=5.)

        if tower_opt['layer_thickness']['flag']:
            wt_opt.model.add_design_var('tower.layer_thickness', lower=tower_opt['layer_thickness']['lower_bound'], upper=tower_opt['layer_thickness']['upper_bound'], ref=1e-2)

        if monopile_opt['outer_diameter']['flag']:
            wt_opt.model.add_design_var('monopile.diameter', lower=monopile_opt['outer_diameter']['lower_bound'], upper=monopile_opt['outer_diameter']['upper_bound'], ref=5.)

        if monopile_opt['layer_thickness']['flag']:
            wt_opt.model.add_design_var('monopile.layer_thickness', lower=monopile_opt['layer_thickness']['lower_bound'], upper=monopile_opt['layer_thickness']['upper_bound'], ref=1e-2)

        # -- Control --
        if control_opt['tsr']['flag']:
            wt_opt.model.add_design_var('opt_var.tsr_opt_gain', lower=control_opt['tsr']['min_gain'],
                                                                upper=control_opt['tsr']['max_gain'])
        if control_opt['servo']['pitch_control']['flag']:
            wt_opt.model.add_design_var('control.PC_omega', lower=control_opt['servo']['pitch_control']['omega_min'],
                                                            upper=control_opt['servo']['pitch_control']['omega_max'])
            wt_opt.model.add_design_var('control.PC_zeta', lower=control_opt['servo']['pitch_control']['zeta_min'],
                                                           upper=control_opt['servo']['pitch_control']['zeta_max'])
        if control_opt['servo']['torque_control']['flag']:
            wt_opt.model.add_design_var('control.VS_omega', lower=control_opt['servo']['torque_control']['omega_min'],
                                                            upper=control_opt['servo']['torque_control']['omega_max'])
            wt_opt.model.add_design_var('control.VS_zeta', lower=control_opt['servo']['torque_control']['zeta_min'],
                                                           upper=control_opt['servo']['torque_control']['zeta_max'])
        
        return wt_opt

    
    def set_constraints(self, wt_opt):
        blade_opt   = self.opt['optimization_variables']['blade']

        # Set non-linear constraints
        blade_constraints = self.opt['constraints']['blade']
        if blade_constraints['strains_spar_cap_ss']['flag']:
            if blade_opt['structure']['spar_cap_ss']['flag']:
                wt_opt.model.add_constraint('rlds.constr.constr_max_strainU_spar', upper= 1.0)
            else:
                print('WARNING: the strains of the suction-side spar cap are set to be constrained, but spar cap thickness is not an active design variable. The constraint is not enforced.')

        if blade_constraints['strains_spar_cap_ps']['flag']:
            if blade_opt['structure']['spar_cap_ps']['flag'] or blade_opt['structure']['spar_cap_ps']['equal_to_suction']:
                wt_opt.model.add_constraint('rlds.constr.constr_max_strainL_spar', upper= 1.0)
            else:
                print('WARNING: the strains of the pressure-side spar cap are set to be constrained, but spar cap thickness is not an active design variable. The constraint is not enforced.')

        if blade_constraints['stall']['flag']:
            if blade_opt['aero_shape']['twist']['flag']:
                wt_opt.model.add_constraint('stall_check.no_stall_constraint', upper= 1.0)
            else:
                print('WARNING: the margin to stall is set to be constrained, but twist is not an active design variable. The constraint is not enforced.')

        if blade_constraints['tip_deflection']['flag']:
            if blade_opt['structure']['spar_cap_ss']['flag'] or blade_opt['structure']['spar_cap_ps']['flag']:
                wt_opt.model.add_constraint('tcons.tip_deflection_ratio', upper= blade_constraints['tip_deflection']['ratio'])
            else:
                print('WARNING: the tip deflection is set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced.')

        if blade_constraints['chord']['flag']:
            if blade_opt['aero_shape']['chord']['flag']:
                wt_opt.model.add_constraint('blade.pa.max_chord_constr', upper= 1.0)
            else:
                print('WARNING: the max chord is set to be constrained, but chord is not an active design variable. The constraint is not enforced.')

        if blade_constraints['frequency']['flap_above_3P']:
            if blade_opt['structure']['spar_cap_ss']['flag'] or blade_opt['structure']['spar_cap_ps']['flag']:
                wt_opt.model.add_constraint('rlds.constr.constr_flap_f_margin', upper= 0.0)
            else:
                print('WARNING: the blade flap frequencies are set to be constrained, but spar caps thickness is not an active design variable. The constraint is not enforced.')

        if blade_constraints['frequency']['edge_above_3P']:
            wt_opt.model.add_constraint('rlds.constr.constr_edge_f_margin', upper= 0.0)

        if blade_constraints['rail_transport']['flag']:
            if blade_constraints['rail_transport']['8_axle']:
                wt_opt.model.add_constraint('elastic.rail.constr_LV_8axle_horiz',   lower = 0.8, upper= 1.0)
                wt_opt.model.add_constraint('elastic.rail.constr_strainPS',         upper= 1.0)
                wt_opt.model.add_constraint('elastic.rail.constr_strainSS',         upper= 1.0)
            elif blade_constraints['rail_transport']['4_axle']:
                wt_opt.model.add_constraint('elastic.rail.constr_LV_4axle_horiz', upper= 1.0)
            else:
                raise ValueError('You have activated the rail transport constraint module. Please define whether you want to model 4- or 8-axle flatcars.')

        if self.opt['constraints']['blade']['moment_coefficient']['flag']:
            wt_opt.model.add_constraint('ccblade.CM', lower= self.opt['constraints']['blade']['moment_coefficient']['min'], upper= self.opt['constraints']['blade']['moment_coefficient']['max'])
        if self.opt['constraints']['blade']['match_cl_cd']['flag_cl'] or self.opt['constraints']['blade']['match_cl_cd']['flag_cd']:
            data_target = np.loadtxt(self.opt['constraints']['blade']['match_cl_cd']['filename'])
            eta_opt     = np.linspace(0., 1., self.opt['optimization_variables']['blade']['aero_shape']['twist']['n_opt'])
            target_cl   = np.interp(eta_opt, data_target[:,0], data_target[:,3])
            target_cd   = np.interp(eta_opt, data_target[:,0], data_target[:,4])
            eps_cl = 1.e-2
            if self.opt['constraints']['blade']['match_cl_cd']['flag_cl']:
                wt_opt.model.add_constraint('ccblade.cl_n_opt', lower = target_cl-eps_cl, upper = target_cl+eps_cl)
            if self.opt['constraints']['blade']['match_cl_cd']['flag_cd']:
                wt_opt.model.add_constraint('ccblade.cd_n_opt', lower = target_cd-eps_cl, upper = target_cd+eps_cl)
        if self.opt['constraints']['blade']['match_L_D']['flag_L'] or self.opt['constraints']['blade']['match_L_D']['flag_D']:
            data_target = np.loadtxt(self.opt['constraints']['blade']['match_L_D']['filename'])
            eta_opt     = np.linspace(0., 1., self.opt['optimization_variables']['blade']['aero_shape']['twist']['n_opt'])
            target_L   = np.interp(eta_opt, data_target[:,0], data_target[:,7])
            target_D   = np.interp(eta_opt, data_target[:,0], data_target[:,8])
        eps_L  = 1.e+2
        if self.opt['constraints']['blade']['match_L_D']['flag_L']:
            wt_opt.model.add_constraint('ccblade.L_n_opt', lower = target_L-eps_L, upper = target_L+eps_L)
        if self.opt['constraints']['blade']['match_L_D']['flag_D']:
            wt_opt.model.add_constraint('ccblade.D_n_opt', lower = target_D-eps_L, upper = target_D+eps_L)

        # Tower and monopile contraints
        tower_constraints = self.opt['constraints']['tower']
        monopile_constraints = self.opt['constraints']['monopile']
        if tower_constraints['height_constraint']['flag'] or monopile_constraints['height_constraint']['flag']:
            wt_opt.model.add_constraint('towerse.height_constraint',
                lower=tower_constraints['height_constraint']['lower_bound'],
                upper=tower_constraints['height_constraint']['upper_bound'])

        if tower_constraints['stress']['flag'] or monopile_constraints['stress']['flag']:
            for k in range(self.modeling['tower']['nLC']):
                kstr = '' if self.modeling['tower']['nLC'] == 0 else str(k+1)
                wt_opt.model.add_constraint('towerse.post'+kstr+'.stress', upper=1.0)

        if tower_constraints['global_buckling']['flag'] or monopile_constraints['global_buckling']['flag']:
            for k in range(self.modeling['tower']['nLC']):
                kstr = '' if self.modeling['tower']['nLC'] == 0 else str(k+1)
                wt_opt.model.add_constraint('towerse.post'+kstr+'.global_buckling', upper=1.0)

        if tower_constraints['shell_buckling']['flag'] or monopile_constraints['shell_buckling']['flag']:
            for k in range(self.modeling['tower']['nLC']):
                kstr = '' if self.modeling['tower']['nLC'] == 0 else str(k+1)
                wt_opt.model.add_constraint('towerse.post'+kstr+'.shell_buckling', upper=1.0)

        if tower_constraints['d_to_t']['flag'] or monopile_constraints['d_to_t']['flag']:
            wt_opt.model.add_constraint('towerse.constr_d_to_t',
                                        lower=tower_constraints['d_to_t']['lower_bound'],
                                        upper=tower_constraints['d_to_t']['upper_bound'])

        if tower_constraints['taper']['flag'] or monopile_constraints['taper']['flag']:
            wt_opt.model.add_constraint('towerse.constr_taper',
                                        lower=tower_constraints['taper']['lower_bound'])

        if tower_constraints['slope']['flag'] or monopile_constraints['slope']['flag']:
            wt_opt.model.add_constraint('towerse.slope', upper=1.0)

        if tower_constraints['frequency_1']['flag'] or monopile_constraints['frequency_1']['flag']:
            for k in range(self.modeling['tower']['nLC']):
                kstr = '' if self.modeling['tower']['nLC'] == 0 else str(k+1)
                wt_opt.model.add_constraint('towerse.post'+kstr+'.structural_frequencies', indices=[0],
                lower=tower_constraints['frequency_1']['lower_bound'],
                upper=tower_constraints['frequency_1']['upper_bound'])

        #control_constraints = self.opt['constraints']['control']
        return wt_opt

    
    def set_recorders(self, wt_opt):
        folder_output = self.opt['general']['folder_output']

        # Set recorder on the OpenMDAO driver level using the `optimization_log`
        # filename supplied in the optimization yaml
        if self.opt['recorder']['flag']:
            recorder = om.SqliteRecorder(os.path.join(folder_output, self.opt['recorder']['file_name']))
            wt_opt.driver.add_recorder(recorder)
            wt_opt.add_recorder(recorder)

            wt_opt.driver.recording_options['excludes'] = ['*_df']
            wt_opt.driver.recording_options['record_constraints'] = True
            wt_opt.driver.recording_options['record_desvars'] = True
            wt_opt.driver.recording_options['record_objectives'] = True
        
        return wt_opt


    def set_initial(self, wt_opt, wt_init):
        blade_opt   = self.opt['optimization_variables']['blade']

        if self.modeling['flags']['blade']:
            wt_opt['blade.opt_var.s_opt_twist']   = np.linspace(0., 1., blade_opt['aero_shape']['twist']['n_opt'])
            if blade_opt['aero_shape']['twist']['flag']:
                init_twist_opt = np.interp(wt_opt['blade.opt_var.s_opt_twist'], wt_init['components']['blade']['outer_shape_bem']['twist']['grid'], wt_init['components']['blade']['outer_shape_bem']['twist']['values'])
                lb_twist = np.array(blade_opt['aero_shape']['twist']['lower_bound'])
                ub_twist = np.array(blade_opt['aero_shape']['twist']['upper_bound'])
                wt_opt['blade.opt_var.twist_opt_gain']    = (init_twist_opt - lb_twist) / (ub_twist - lb_twist)
                if max(wt_opt['blade.opt_var.twist_opt_gain']) > 1. or min(wt_opt['blade.opt_var.twist_opt_gain']) < 0.:
                    print('Warning: the initial twist violates the upper or lower bounds of the twist design variables.')

            blade_constraints = self.opt['constraints']['blade']
            wt_opt['blade.opt_var.s_opt_chord']  = np.linspace(0., 1., blade_opt['aero_shape']['chord']['n_opt'])
            wt_opt['blade.ps.s_opt_spar_cap_ss'] = np.linspace(0., 1., blade_opt['structure']['spar_cap_ss']['n_opt'])
            wt_opt['blade.ps.s_opt_spar_cap_ps'] = np.linspace(0., 1., blade_opt['structure']['spar_cap_ps']['n_opt'])
            wt_opt['rlds.constr.max_strainU_spar'] = blade_constraints['strains_spar_cap_ss']['max']
            wt_opt['rlds.constr.max_strainL_spar'] = blade_constraints['strains_spar_cap_ps']['max']
            wt_opt['stall_check.stall_margin'] = blade_constraints['stall']['margin'] * 180. / np.pi
        
        return wt_opt


    def set_restart(self, wt_opt):
        if 'warmstart_file' in self.opt['driver']:

            # Directly read the pyoptsparse sqlite db file
            from pyoptsparse import SqliteDict
            db = SqliteDict(self.opt['driver']['warmstart_file'])

            # Grab the last iteration's design variables
            last_key = db['last']
            desvars = db[last_key]['xuser']

            # Obtain the already-setup OM problem's design variables
            if wt_opt.model._static_mode:
                design_vars = wt_opt.model._static_design_vars
            else:
                design_vars = wt_opt.model._design_vars

            # Get the absolute names from the promoted names within the OM model.
            # We need this because the pyoptsparse db has the absolute names for
            # variables but the OM model uses the promoted names.
            prom2abs = wt_opt.model._var_allprocs_prom2abs_list['output']
            abs2prom = {}
            for key in design_vars:
                abs2prom[prom2abs[key][0]] = key

            # Loop through each design variable
            for key in desvars:
                prom_key = abs2prom[key]

                # Scale each DV based on the OM scaling from the problem.
                # This assumes we're running the same problem with the same scaling
                scaler = design_vars[prom_key]['scaler']
                adder = design_vars[prom_key]['adder']

                if scaler is None:
                    scaler = 1.0
                if adder is None:
                    adder = 0.0

                scaled_dv = desvars[key] / scaler - adder

                # Special handling for blade twist as we only have the
                # last few control points as design variables
                if 'twist_opt_gain' in key:
                    wt_opt[key][2:] = scaled_dv
                else:
                    wt_opt[key][:] = scaled_dv
                    
        return wt_opt
