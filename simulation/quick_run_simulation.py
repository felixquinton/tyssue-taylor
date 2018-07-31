"""Provides tools to run easily run an optimization process in a terminal.
To do so one must edit the if __name__ == '__main__' part of this script
and run it as a python script.
Results will be saved in the adress provided in .bashrc in the variable :
SAVE_DIR_ORGANOID_SIMULATION
"""
import time
import json
import os
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt

from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue.io.hdf5 import load_datasets, save_datasets
from tyssue.generation.shapes import AnnularSheet
from tyssue.config.draw import sheet_spec
from tyssue.draw.plt_draw import sheet_view, quick_edge_draw

from tyssue_taylor.adjusters.adjust_annular import (prepare_tensions,
                                                    adjust_tensions)
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model
from tyssue_taylor import version

def print_tensions(exp_organo, th_organo, to_save=False, path=None):
    """Print the tensions of an experimental organoid and the theoritical
    organoid on the same plot and save it in a PNG image.

    Parameters
    ----------
    exp_organo : class AnnularSheet
      the experimental organoid whose tensions will be ploted
    th_organo : class AnnularSheet
      the theoritical organoid which will be ploted behind exp_organo tensions
    to_save : bool
      if True the plot must be saved
    path : string
     path to the saved image.
    Returns
    ----------
    """
    draw_specs = sheet_spec()
    tension_max = np.max(exp_organo.edge_df.line_tension.values.copy())
    edge_color = 1/tension_max*exp_organo.edge_df.line_tension.values.copy()
    cmap = plt.cm.get_cmap('viridis')
    edge_cmap = cmap(edge_color)
    draw_specs['vert']['visible'] = False
    draw_specs['edge']['color'] = edge_cmap
    draw_specs['edge']['width'] = 0.25+3*edge_color
    fig, ax = quick_edge_draw(th_organo, lw=5, c='k', alpha=0.2)
    fig, ax = sheet_view(exp_organo, ax=ax, **draw_specs)
    fig.set_size_inches(12, 12)
    plt.xlabel('Size in µm')
    plt.ylabel('Size in µm')
    if to_save:
        plt.savefig(path)


def get_normal_from_seed(seed, mu, theta, shape):
    """Compute a normal pseudo random vector with given parameters wrt to a
    given random seed. Will returns the same vector for the same seed and
    set of parameters.

    Parameters
    ----------
    seed : int
      random seed
    mu : float
      mean of the normal distribution
    theta : float
      standard deviation of the normal distribution
    shape : tuple of int
      shape of the output vector
    Returns
    ----------
    A pseudo random vector from the normal distribution of given parameters, wrt
    to the given random seed.
    """
    np.random.seed(seed)
    return np.random.normal(mu, theta, shape)

def replicate_organo_from_file(path_hdf5, path_json):
    """Read and hdf5 file and a json file to create the theoritical and
     experimental organoids.

    Parameters
    ----------
    path_hdf5 : string
      path to the hdf5 file. Must contain the output from
      tyssue.io.hdf5.save_datasets.
    path_json : string
      path to the json file. Must contain the specs dictionnary of the organo.

    Returns
    ----------
    th_organo : class AnnularSheet
      the theoritical organoid from the files.
    exp_organo : class AnnularSheet
      the experimental organoid created with the random seed from the json file.
    """
    dsets = load_datasets(path_hdf5)
    th_organo = AnnularSheet('theoritical', dsets)
    with open(path_json, 'r') as file:
        dic = json.load(file)
        seed = dic['seed']
        theta = dic['theta']
        del dic['seed']
        del dic['theta']
        specs = dic
    th_organo.update_specs(specs, reset=True)
    exp_organo = th_organo.copy()
    random_sequence = get_normal_from_seed(seed, 1.0, theta, (exp_organo.Nv, 2))
    exp_organo.vert_df.loc[:, exp_organo.coords] *= random_sequence
    geom.update_all(exp_organo)
    return th_organo, exp_organo

def save_optimization_results(exp_organo, th_organo, opt_res, main_opt_options,
                              energy_opt_options, is_reg, is_lumen, dir_path,
                              seed, initial_opt_options=None):
    """Save the optimization results.

    Parameters
    ----------
    organo : class AnnularSheet
      the mesh after the optimization process.
    opt_res : dictionnary
      must contain following fields :
      all fields from scipy OptimizeResults or pyOpt res except for jac, grad,
      active_mask (if applicable)
      res_time : the solving time of the optimization process
      tension_error : the element wise error between thpipeline_test_theoritical_organo.hdf5e true tension vector
      and the optimization result
      git_revision : the git version of the source code used for the
      optimization
    main_opt_options : dictionnary
      the optimization options passed to the solver for the main optimization.
    energy_min_options : dictionnary
      the optimization options passed to the tyssue.solver.find_energy_min
      function
    is_reg : boolean
      indicates if the optimization includes a regularization module
    is_lumen : boolean
      indicates if the lumen volume is an optimization parameter
    initial_min_opt : dictionnary
      the optimization options passed to the search of the initial point if
      applicable.
    Returns
    ----------
    """
    reg_txt = 'nr'
    lum_txt = 'nl'
    if is_reg:
        reg_txt = 'r'
    if is_lumen:
        lum_txt = 'l'
    local_path = dir_path+'/simulation/'+main_opt_options['method']+\
                 '_'+reg_txt+'_'+lum_txt
    os.makedirs(local_path, exist_ok=True)

    for ind in opt_res:
        if isinstance(opt_res[ind], np.ndarray):
            opt_res[ind] = list(opt_res[ind])
        elif isinstance(opt_res[ind], np.int64):
            opt_res[ind] = np.float(opt_res[ind])
    if not opt_res.get('jac', None) is None:
        del opt_res['jac']
    if not opt_res.get('active_mask', None) is None:
        del opt_res['active_mask']
    save_datasets(local_path+'/exp_organo'+str(seed)+'.hdf5', exp_organo)
    with open(local_path+'/exp_organo_settings'+str(seed)+'.json',
              'w+') as outfile:
        json.dump(exp_organo.settings, outfile)
    with open(local_path+'/exp_organo_opt_res'+str(seed)+'.json',
              'w+') as outfile:
        json.dump(opt_res, outfile)
    with open(local_path+'/exp_organo_main_opt_options.json', 'w+') as outfile:
        json.dump(main_opt_options, outfile)
    if not initial_opt_options is None:
        initial_opt_options['bounds'] = list(initial_opt_options.get('bounds',
                                                                     None))
        if not initial_opt_options.get('verbose', None) is None:
            del initial_opt_options['verbose']
        with open(local_path+'/exp_organo_initial_opt_options.json',
                  'w+') as outfile:
            json.dump(initial_opt_options, outfile)
    with open(local_path+'/exp_organo_energy_opt_options.json',
              'w+') as outfile:
        json.dump(energy_opt_options, outfile)
    print_tensions(exp_organo, th_organo, True,
                   local_path+'/exp_organo'+str(seed)+'.png')
    #shutil.make_archive(local_path+'/exp_res'+str(seed), format='zip',
    #                    base_dir=local_path)



def run_nr_nl_optimization(organo, noisy, energy_min, main_min,
                           initial_min=None):
    """Run the optimization process when lumen volume is not a parameter and
    there is no regularization module.

    Parameters
    ----------
    organo: class AnnularSheet
    the theoritical mesh.
    noisy: class AnnularSheet
    the experimental mesh.
    main_min: dictionnary
      the optimization options passed to the solver for the main optimization.
    energy_min: dictionnary
      the optimization options passed to the tyssue.solver.find_energy_min
      function
    initial_min: dictionnary
      the optimization options passed to the search of the initial point if
      applicable.
    Returns
    noisy: class AnnularSheet
    the experimental organoid after the optimization process
    opt_res: dictionnary
    dictionnary containing relevant info on the optimization process. Fit to
    the requirements of save_optimization_resultsself.

    ----------
    """
    initial_guess = organo.edge_df.line_tension[:3*organo.Nf].values.copy()
    start = time.clock()
    res = adjust_tensions(noisy, initial_guess, {'dic':{}, 'weight':0},
                          energy_min, initial_min, **main_min)
    if main_min['method'] == 'PSQP':
        res_x = res['x']
    else:
        res_x = res.x
    noisy.edge_df.line_tension = prepare_tensions(noisy, res_x)
    Solver.find_energy_min(noisy, geom, model)
    res_time = time.clock()-start
    tension_error = np.divide((initial_guess-res_x), initial_guess)
    opt_res = res
    opt_res['res_time'] = res_time
    opt_res['tension_error'] = list(tension_error)
    opt_res['git_revision'] = version.git_revision

    return noisy, opt_res

def run_nr_l_optimization(organo, noisy, energy_min, main_min,
                          initial_min=None):
    """Run the optimization process when lumen volume IS a parameter and
    there is no regularization module.

    Parameters
    ----------
    organo: class AnnularSheet
    the theoritical mesh.
    noisy: class AnnularSheet
    the experimental mesh.
    main_min: dictionnary
     the optimization options passed to the solver for the main optimization.
    energy_min: dictionnary
     the optimization options passed to the tyssue.solver.find_energy_min
     function
    initial_min: dictionnary
     the optimization options passed to the search of the initial point if
     applicable.
    Returns
    noisy: class AnnularSheet
    the experimental organoid after the optimization process
    opt_res: dictionnary
    dictionnary containing relevant info on the optimization process. Fit to
    the requirements of save_optimization_resultsself.

    ----------
    """
    initial_guess = organo.edge_df.line_tension[:3*organo.Nf].values.copy()
    initial_guess = np.concatenate((initial_guess,
                                    [organo.settings['lumen_volume']]))
    start = time.clock()
    if not initial_min is None:
        if len(initial_min['bounds']) <= len(initial_guess):
            initial_min['bounds'][0].append(-1e-8)
            initial_min['bounds'][1].append(1e6)
    if main_min['method'] in ('bfgs', 'trf'):
        if len(main_min['bounds']) <= len(initial_guess):
            main_min['bounds'][0].append(-1e-8)
            main_min['bounds'][1].append(1e6)
    res = adjust_tensions(noisy, initial_guess, {'dic':{}, 'weight':0},
                          energy_min, initial_min, **main_min)
    if main_min['method'] == 'PSQP':
        res_x = res['x']
    else:
        res_x = res.x
    noisy.edge_df.line_tension = prepare_tensions(noisy, res_x)
    Solver.find_energy_min(noisy, geom, model)
    res_time = time.clock()-start
    tension_error = np.divide((initial_guess-res_x), initial_guess)
    opt_res = res
    opt_res['res_time'] = res_time
    opt_res['tension_error'] = list(tension_error)
    opt_res['git_revision'] = version.git_revision

    return noisy, opt_res


def run_r_nl_optimization(organo, noisy, energy_min, main_min,
                          initial_min=None):
    """Run the optimization process when lumen volume is not a parameter and
    there IS regularization module.

    Parameters
    ----------
    organo: class AnnularSheet
    the theoritical mesh.
    noisy: class AnnularSheet
    the experimental mesh.
    main_min: dictionnary
     the optimization options passed to the solver for the main optimization.
    energy_min: dictionnary
     the optimization options passed to the tyssue.solver.find_energy_min
     function
    initial_min: dictionnary
     the optimization options passed to the search of the initial point if
     applicable.
    Returns
    noisy: class AnnularSheet
    the experimental organoid after the optimization process
    opt_res: dictionnary
    dictionnary containing relevant info on the optimization process. Fit to
    the requirements of save_optimization_resultsself.

    ----------
    """
    initial_guess = organo.edge_df.line_tension[:3*organo.Nf].values.copy()
    start = time.clock()
    res = adjust_tensions(noisy, initial_guess,
                          {'dic':{'basal': True, 'apical': True}, 'weight':0.001},
                          energy_min, initial_min, **main_min)
    if main_min['method'] == 'PSQP':
        res_x = res['x']
    else:
        res_x = res.x
    noisy.edge_df.line_tension = prepare_tensions(noisy, res_x)
    Solver.find_energy_min(noisy, geom, model)
    res_time = time.clock()-start
    tension_error = np.divide((initial_guess-res_x), initial_guess)
    opt_res = res
    opt_res['res_time'] = res_time
    opt_res['tension_error'] = list(tension_error)
    opt_res['git_revision'] = version.git_revision


    return noisy, opt_res



def run_r_l_optimization(organo, noisy, energy_min, main_min,
                         initial_min=None):
    """Run the optimization process when lumen volume IS a parameter and
    there IS regularization module.

    Parameters
    ----------
    organo: class AnnularSheet
    the theoritical mesh.
    noisy: class AnnularSheet
    the experimental mesh.
    main_min: dictionnary
    the optimization options passed to the solver for the main optimization.
    energy_min: dictionnary
    the optimization options passed to the tyssue.solver.find_energy_min
    function
    initial_min: dictionnary
    the optimization options passed to the search of the initial point if
    applicable.
    Returns
    noisy: class AnnularSheet
    the experimental organoid after the optimization process
    opt_res: dictionnary
        print(ind, type(res[ind]))
    dictionnary containing relevant info on the optimization process. Fit to
    the requirements of save_optimization_resultsself.

    ----------
    """
    initial_guess = organo.edge_df.line_tension[:3*organo.Nf].values.copy()
    initial_guess = np.concatenate((initial_guess,
                                    [organo.settings['lumen_volume']]))
    if not initial_min is None:
        if len(initial_min['bounds']) <= len(initial_guess):
            initial_min['bounds'][0].append(-1e-8)
            initial_min['bounds'][1].append(1e6)
    if main_min['method'] in ('bfgs', 'trf'):
        if len(main_min['bounds']) <= len(initial_guess):
            main_min['bounds'][0].append(-1e-8)
            main_min['bounds'][1].append(1e6)
    start = time.clock()
    res = adjust_tensions(noisy, initial_guess,
                          {'dic':{'basal': True, 'apical': True}, 'weight':0.001},
                          energy_min, initial_min, **main_min)
    if main_min['method'] == 'PSQP':
        res_x = res['x']
    else:
        res_x = res.x
    noisy.edge_df.line_tension = prepare_tensions(noisy, res_x)
    Solver.find_energy_min(noisy, geom, model)
    res_time = time.clock()-start
    tension_error = np.divide((initial_guess-res_x), initial_guess)
    opt_res = res
    opt_res['res_time'] = res_time
    opt_res['tension_error'] = list(tension_error)
    opt_res['git_revision'] = version.git_revision

    return noisy, opt_res

if __name__ == '__main__':

    ASSET_PATH = os.environ.get('SAVE_DIR_ORGANOID_SIMULATION')
    with open(ASSET_PATH+'/assets/benchmark_instances/list_seed.json',
              'r') as inputfile:
        LIST_SEED = json.load(inputfile)['list']

    PARSER = argparse.ArgumentParser(description='Adjust line tensions')
    PARSER.add_argument('--psqp', action='store_true',
                        help='indicates if the solver must use PSQP.')
    PARSER.add_argument('--dist_psqp', action='store_true',
                        help='indicates if the solver must use PSQP.')
    PARSER.add_argument('--bfgs', action='store_true',
                        help='indicates if the solver must use BFGS.')
    PARSER.add_argument('--trf', action='store_true',
                        help='indicates if the solver must use TRF.')
    PARSER.add_argument('--lm', action='store_true',
                        help='indicates if the solver must use Levenberg-\
                        Marquartd.')
    PARSER.add_argument('--reg', action='store_true',
                        help='indicates if the objective must include a \
                        regularization module.')
    PARSER.add_argument('--lumen', action='store_true',
                        help='indicates if the optimization must include the \
                        lumen volume as an optimization parameter.')
    PARSER.add_argument('--nb_rep', type=int,
                        help='number of instances to load and solve')
    ARGS = vars(PARSER.parse_args())

    for SEED in LIST_SEED[:ARGS['nb_rep']]:
        print('Solving for random seed '+str(SEED)+'.')
        PATH_HFD5 = ASSET_PATH+'/assets/benchmark_instances/pipeline_test_'+\
                               'theoritical_organo'+str(SEED)+'.hdf5'
        PATH_JSON = ASSET_PATH+'/assets/benchmark_instances/pipeline_test_'+\
                               'theoritical_organo_specs'+str(SEED)+'.json'
        TH, EXP = replicate_organo_from_file(PATH_HFD5, PATH_JSON)
        with open(ASSET_PATH+'/assets/pipeline_test_energy_opt.json',
                  'r') as inputfile:
            ENER_MIN = json.load(inputfile)
        with open(ASSET_PATH+'/assets/pipeline_test_psqp_opt.json',
                  'r') as inputfile:
            PSQP_MIN = json.load(inputfile)
        with open(ASSET_PATH+'/assets/pipeline_test_trf_opt.json',
                  'r') as inputfile:
            TRF_MIN = json.load(inputfile)
        with open(ASSET_PATH+'/assets/pipeline_test_bfgs_opt.json',
                  'r') as inputfile:
            BFGS_MIN = json.load(inputfile)
        with open(ASSET_PATH+'/assets/pipeline_test_lm_opt.json',
                  'r') as inputfile:
            LM_MIN = json.load(inputfile)
        if ARGS['lm']+ARGS['bfgs']+ARGS['trf'] > 1:
            raise ValueError('Only one method allowed among TRF, BFGS and LM.')
        MAIN_MIN = TRF_MIN
        INIT_MIN = None
        if ARGS['psqp'] or ARGS['dist_psqp']:
            MAIN_MIN = PSQP_MIN
            if ARGS['dist_psqp']:
                MAIN_MIN['method'] = 'dist_PSQP'
            if ARGS['lm']:
                INIT_MIN = LM_MIN
            elif ARGS['bfgs']:
                INIT_MIN = BFGS_MIN
            else:
                INIT_MIN = TRF_MIN
        elif ARGS['lm']:
            MAIN_MIN = LM_MIN
        elif ARGS['trf']:
            MAIN_MIN = TRF_MIN
        elif ARGS['bfgs']:
            MAIN_MIN = BFGS_MIN
        if ARGS['reg']:
            if ARGS['lumen']:
                N, O_R = run_r_l_optimization(TH, EXP, ENER_MIN,
                                              MAIN_MIN, INIT_MIN)
            else:
                N, O_R = run_r_nl_optimization(TH, EXP, ENER_MIN,
                                               MAIN_MIN, INIT_MIN)
        else:
            if ARGS['lumen']:
                N, O_R = run_nr_l_optimization(TH, EXP, ENER_MIN,
                                               MAIN_MIN, INIT_MIN)
            else:
                N, O_R = run_nr_nl_optimization(TH, EXP, ENER_MIN,
                                                MAIN_MIN, INIT_MIN)
        save_optimization_results(N, TH, O_R, MAIN_MIN, ENER_MIN, ARGS['reg'],
                                  ARGS['lumen'], ASSET_PATH, SEED, INIT_MIN)
