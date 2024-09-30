from ex_gomez_levy import *
from moobench.optlib_pymoo_proto import PymooOptimizationAlgorithmSingle
from moobench.optlib_scipy import ScipyOptimizationAlgorithm
import os
# test commit
# Prepare output directory for writing
out_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
isExist = os.path.exists(out_folder_path)
if not isExist:
    os.makedirs(out_folder_path)

op = EX_GomezLevy_OptProb()

#SciPy algorithms
if True:
    opt_ctrl = {}
    op.opt_algorithm = ScipyOptimizationAlgorithm('SLSQP_mi=1000','SLSQP',opt_ctrl)
    if True:
        sol = op.optimize()
        #sol = op.optimize([0.0,-0.5]) # start from solution [0.0,-0.5]
        op.print_output()
    else:
        sol = op.optimize_and_write(out_folder_path)

#Pymoo library
pop_size = 100
num_iter = 100
max_evaluations = pop_size * num_iter
termination = ('n_eval', max_evaluations)
init_ctrl_opt = {'termination': termination, 'tolfun': 1e-6, 'tolx': 1e-5}
if hf:
    mutation = {'name':'real_pm', 'eta':20, 'prob': 0.1}  # Check
    crossover = {'name':'real_sbx', 'eta':20, 'prob':0.95}  # Check
    selection = {'name':'random'}
    #selection = {'name': 'tournament'}

    if True:#ga
        ga_ctrl = {'pop_size': pop_size}
        ga_ctrl.update(init_ctrl_opt)
        ga_ctrl.update({'mutation': mutation, 'crossover': crossover, 'selection':selection})
        op.opt_algorithm = PymooOptimizationAlgorithmSingle('ga_default', 'ga', alg_ctrl=ga_ctrl)
        sol = op.optimize()
        op.print_output()
    if True:#de
        de_ctrl = {'pop_size': pop_size}
        de_ctrl.update(init_ctrl_opt)
        op.opt_algorithm = PymooOptimizationAlgorithmSingle('de_default', 'de', alg_ctrl=de_ctrl)
        sol = op.optimize()
        op.print_output()
    if True:  # cmes
        #uses internal pop_size
        # Valid options are ['AdaptSigma', 'CMA_active', 'CMA_active_injected', 'CMA_cmean', 'CMA_const_trace', '
        # CMA_diagonal', 'CMA_diagonal_decoding', 'CMA_eigenmethod', 'CMA_elitist', 'CMA_injections_threshold_keep_len',
        # 'CMA_mirrors', 'CMA_mirrormethod', 'CMA_mu', 'CMA_on', 'CMA_sampler', 'CMA_sampler_options', 'CMA_rankmu',
        # 'CMA_rankone', 'CMA_recombination_weights', 'CMA_dampsvec_fac', 'CMA_dampsvec_fade', 'CMA_teststds',
        # 'CMA_stds', 'CSA_dampfac', 'CSA_damp_mueff_exponent', 'CSA_disregard_length', 'CSA_clip_length_value',
        # 'CSA_squared', 'CSA_invariant_path', 'stall_sigma_change_on_divergence_iterations', 'BoundaryHandler',
        # 'bounds', 'conditioncov_alleviate', 'eval_final_mean', 'fixed_variables', 'ftarget', 'integer_variables',
        # 'is_feasible', 'maxfevals', 'maxiter', 'mean_shift_line_samples', 'mindx', 'minstd', 'maxstd',
        # 'maxstd_boundrange', 'pc_line_samples', 'popsize', 'popsize_factor', 'randn', 'scaling_of_variables',
        # 'seed', 'signals_filename', 'termination_callback', 'timeout', 'tolconditioncov', 'tolfacupx',
        # 'tolupsigma', 'tolflatfitness', 'tolfun', 'tolfunhist', 'tolfunrel', 'tolstagnation',
        # tolxstagnation', 'tolx', 'transformation', 'typical_x', 'updatecovwait', 'verbose', 'verb_append',
        # 'verb_disp', 'verb_disp_overwrite', 'verb_filenameprefix', 'verb_log', 'verb_log_expensive', 'verb_plot',
        # 'verb_time', 'vv']
        cmaes_ctrl = {}
        #cmaes_ctrl = {'maxiter': num_iter}
        cmaes_ctrl.update(init_ctrl_opt)
        op.opt_algorithm = PymooOptimizationAlgorithmSingle('cmaes_default', 'cmaes', alg_ctrl=cmaes_ctrl)
        sol = op.optimize()
        op.print_output()
    if True:
        brkga_ctrl = {'n_elites': 100, 'n_offsprings': 100,'n_mutants' : 100,'bias' : 0.7}
        brkga_ctrl.update(init_ctrl_opt)
        op.opt_algorithm = PymooOptimizationAlgorithmSingle('brkga_default', 'brkga', alg_ctrl=brkga_ctrl)
        sol = op.optimize()
        op.print_output()
if False:#nelder-mead
    def adaptive_params(problem):
        n = problem.n_var
        alpha = 1
        beta = 1 + 2 / n
        gamma = 0.75 - 1 / (2 * n)
        delta = 1 - 1 / n
        return alpha, beta, gamma, delta #reflection, expansion,contraction shrink

    nm_ctrl = {}
    nm_ctrl.update(init_ctrl_opt)
    #nm_ctrl.update({'func_params':adaptive_params})
    op.opt_algorithm = PymooOptimizationAlgorithmSingle('nelder-mead_default', 'nelder-mead', alg_ctrl=nm_ctrl)
    sol = op.optimize()
    op.print_output()
