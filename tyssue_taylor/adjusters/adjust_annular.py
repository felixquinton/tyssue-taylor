"""This module provides functions to build the optimization process
"""

import numpy as np
from tyssue.solvers.sheet_vertex_solver import Solver as solver
from tyssue import config
from scipy.optimize import minimize, least_squares
import pyOpt

from .cost_functions import _distance, _energy, distance_regularized
from ..models.annular import AnnularGeometry as geom
from ..models.annular import model

def adjust_scale(eptm, tensions_array):
    return None

def adjust_tensions(eptm, initial_guess, regularization,
                    energy_min_opt=None, initial_min_opt=None,
                    **main_min_opt):
    """Find the line tensions which minimize the distance to the epithelium

    Parameters
    ----------
    eptm : :class:`Epithelium` object
    initial_guess : vector of initial line tensions (size 3*Nf)
    regularization : dictionnary with fields :
                        'dic' : dictionnary with fields 'apical' and 'basal'
                        and boolean value indicating if the corresponding set
                        of edges should be regularized.
                        'weight' : float, weight of the regularization module
    energy_min_opt : scipy.optimize.minize option dictionnary for the energy
                     minimization.
    initial_min_opt : scipy.optimize.minimize option dictionnary for the
                      initial point search. Ignored if main_min_opt['method']
                      is not 'PSQP' or 'SLSQP'.
    main_min_opt : option dictionnary for the main optimization. Syntax depends
                   on the method. For bgfs and SLSQP use the scipy.optimize.minimize
                   syntax. For trf and lm use the scipy.optimize.least_squares
                   method. For PSQP the syntax is :
                   'lb' : lower bound of the Parameters
                   'ub' : upper bound of the Parameters
                   'method': PSQP
    """
    organo = eptm.copy()
    minimize_opt = config.solvers.minimize_spec()

    if energy_min_opt is not None:
        minimize_opt['minimize']['options'] = energy_min_opt.get(
            'options',
            minimize_opt['minimize']['options'])
    if main_min_opt['method'] == 'bfgs':
        return minimize(_obj_bfgs, initial_guess, **main_min_opt,
                        args=(organo, regularization, minimize_opt))
    elif main_min_opt['method'] in ('trf', 'lm'):
        return least_squares(_opt_dist, initial_guess, **main_min_opt,
                             args=(organo, regularization, False),
                             kwargs=minimize_opt)
    elif main_min_opt['method'] == 'SLSQP':
        return _slsqp_opt(organo, initial_guess, regularization,
                          minimize_opt, initial_min_opt,
                          **main_min_opt)
    elif main_min_opt['method'] == 'PSQP':
        return _psqp_ener_opt(organo, initial_guess, regularization,
                              minimize_opt, initial_min_opt,
                              **main_min_opt)
    elif main_min_opt['method'] == 'dist_PSQP':
        return _psqp_dist_opt(organo, initial_guess, regularization,
                              minimize_opt, initial_min_opt,
                              **main_min_opt)
    else:
        print(f"Unknown method : f{main_min_opt['method']}")
    return -1

def adjust_areas(eptm, initial_guess, opt_tensions,
                 energy_min_opt=None,
                 **main_min_opt):
    """Find the line tensions which minimize the distance to the epithelium

    Parameters
    ----------
    eptm : :class:`Epithelium` object
    area_guess : vector of initial prefered area (size 3*Nf)
    energy_min_opt : scipy.optimize.minize option dictionnary for the energy
                     minimization
    opt_tensions : table of tensions to be set in organo.
    main_min_opt : option dictionnary for the main optimization. Syntax depends
                   on the method.
                   For bgfs and SLSQP use the scipy.optimize.minimize syntax.
                   For trf and lm use the scipy.optimize.least_squares method.
                   For PSQP the syntax is:
                   'lb' : lower bound of the Parameters
                   'ub' : upper bound of the Parameters
                   'method': PSQP
    """
    organo = eptm.copy()
    minimize_opt = config.solvers.minimize_spec()

    if energy_min_opt is not None:
        minimize_opt['minimize']['options'] = energy_min_opt.get(
            'options',
            minimize_opt['minimize']['options'])
    if main_min_opt['method'] == 'bfgs':
        return -1
    #    return minimize(_obj_bfgs, initial_guess, **main_min_opt,
    #                    args=(organo, regularization, minimize_opt))
    elif main_min_opt['method'] in ('trf', 'lm'):
        return least_squares(_opt_dist, initial_guess,
                             **main_min_opt,
                             args=(organo, {'dic':{}, 'weight':0},
                                   False, opt_tensions),
                             kwargs=minimize_opt)
    elif main_min_opt['method'] == 'dist_PSQP':
        return _psqp_dist_opt(organo, initial_guess, {'dic':{}, 'weight':0},
                              minimize_opt, minimize_opt,
                              opt_tensions=opt_tensions,
                              **main_min_opt)
    else:
        print(f"Unknown method : f{main_min_opt['method']}")
    return -1

def _slsqp_opt(organo, initial_guess, regularization,
               minimize_opt, initial_min_opt,
               opt_tensions=None, **main_min_opt):
    print("Starting the search for the initial point")
    initial_point = least_squares(_opt_dist, initial_guess,
                                  **initial_min_opt,
                                  args=(organo, regularization, True),
                                  kwargs=(opt_tensions, minimize_opt))
    if initial_point.success:
        print(f"Initial point found with distance {initial_point.fun[0]}\n\
                Starting the energy minimization.")
        return minimize(_opt_ener, initial_point.x,
                        args=(organo),
                        constraints=({'type': 'ineq',
                                      'fun': _slsqp_cst,
                                      'args': (organo,
                                               initial_point.fun,
                                               regularization)}),
                        bounds=np.full((len(initial_point.x), 2), (0, 10)),
                        **main_min_opt)
    print(f"Initial point search failed with message :\
          \nf{initial_point.message}")
    return -1

def _psqp_ener_opt(organo, initial_guess, regularization,
                   minimize_opt, initial_min_opt,
                   opt_tensions=None,
                   **main_min_opt):
    print("Starting the search for the initial point")
    initial_point = least_squares(_opt_dist, initial_guess,
                                  **initial_min_opt,
                                  args=(organo, regularization, False),
                                  kwargs=minimize_opt)
    if initial_point.success:
        print(f"Initial point found with distance {initial_point.fun[0]}\n\
        Starting the energy minimization.")
        init_eptm = organo.copy()
        init_eptm.edge_df.line_tension = prepare_tensions(init_eptm,
                                                          initial_point.x)
        solver.find_energy_min(init_eptm, geom, model)
        initial_dist = _distance(init_eptm, organo)
        opt_prob = _create_pyopt_model(_wrap_obj_and_const,
                                       initial_point.x,
                                       'min_ener')
        psqp = pyOpt.PSQP()
        psqp.setOption('IPRINT', 2)
        psqp.setOption('TOLX', 1e-6)
        psqp.setOption('IFILE', main_min_opt.get('output_path', 'PSQP.out'))

        [fstr, xstr, inform] = psqp(opt_prob,
                                    sens_type='FD',
                                    disp_opts=True,
                                    sens_mode='pgc',
                                    pb_obj='min_ener',
                                    organo=organo,
                                    regularization=regularization,
                                    initial_dist=initial_dist,
                                    opt_tensions=opt_tensions,
                                    minimize_opt=minimize_opt)
        return {'fun': fstr, 'x': xstr, 'message': inform}
    print(f"Initial point search failed with message :\
          \nf{initial_point.message}")
    return -1

def _psqp_dist_opt(organo, initial_guess, regularization,
                   minimize_opt, energy_min_opt,
                   opt_tensions=None,
                   **main_min_opt):
    init_eptm = organo.copy()
    if opt_tensions is None:
        init_eptm.edge_df.line_tension = prepare_tensions(init_eptm,
                                                          initial_guess)
    else:
        init_eptm.edge_df.line_tension = prepare_tensions(init_eptm,
                                                          opt_tensions)
    solver.find_energy_min(init_eptm, geom, model)
    initial_ener = _opt_ener(initial_guess, init_eptm, opt_tensions,
                             **energy_min_opt)
    opt_prob = _create_pyopt_model(_wrap_obj_and_const,
                                   initial_guess,
                                   main_min_opt,
                                   'min_dist')
    psqp = pyOpt.PSQP()
    psqp.setOption('IPRINT', 2)
    psqp.setOption('TOLX', 1e-6)
    psqp.setOption('IFILE', main_min_opt.get('output_path', 'PSQP.out'))

    [fstr, xstr, inform] = psqp(opt_prob,
                                sens_type='FD',
                                disp_opts=True,
                                sens_mode='pgc',
                                pb_obj='min_dist',
                                organo=organo,
                                regularization=regularization,
                                initial_ener=initial_ener,
                                opt_tensions=opt_tensions,
                                minimize_opt=minimize_opt)
    return {'fun': fstr, 'x': xstr, 'message': inform}

def _cst_dist(tension_array, organo, initial_dist, regularization,
              opt_tensions=None, **minimize_opt):
    error = _opt_dist(tension_array, organo, regularization, True,
                      opt_tensions=opt_tensions, **minimize_opt)
    initial_dist_table = 2*np.maximum(initial_dist[:2*organo.Nf],
                                      np.full(2*organo.Nf, 0.1))
    cst = list(initial_dist_table-error)
    bounds = list(-tension_array)
    return  cst + bounds

def _slsqp_cst(tension_array, organo, initial_dist, regularization,
               **minimize_opt):
    return _cst_dist(tension_array, organo, initial_dist, regularization,
                     **minimize_opt)[:2*organo.Nf]

def _cst_ener(var_table, organo, initial_ener,
              opt_tensions, **minimize_opt):
    ener = _opt_ener(var_table, organo,
                     opt_tensions=opt_tensions,
                     **minimize_opt)
    bounds = list(-var_table)
    return  [initial_ener - ener] + bounds

def _opt_ener(var_table, organo,
              opt_tensions=None,
              **minimize_opt):
    tmp_organo = organo.copy()
    variables = {}
    if opt_tensions is None:
        variables[('edge', 'line_tension')] = prepare_tensions(
            tmp_organo, var_table[:3*tmp_organo.Nf])
    else:
        variables[('edge', 'line_tension')] = prepare_tensions(tmp_organo,
                                                               opt_tensions)
        variables[('face', 'prefered_area')] = var_table
    if len(var_table)%organo.Nf != 0:
        variables[('lumen_prefered_vol', None)] = var_table[-1]
    return _energy(tmp_organo, variables, solver, geom, model,
                   **minimize_opt)

def _opt_dist(var_table, organo, regularization, sum_obj,
              opt_tensions=None, **minimize_opt):
    tmp_organo = organo.copy()
    variables = {}
    if opt_tensions is None:
        variables[('edge', 'line_tension')] = prepare_tensions(
            tmp_organo, var_table[:3*tmp_organo.Nf])
    else:
        variables[('edge', 'line_tension')] = prepare_tensions(organo,
                                                               opt_tensions)
        variables[('face', 'prefered_area')] = var_table
    if len(var_table)%organo.Nf != 0:
        variables[('lumen_prefered_vol', None)] = var_table[-1]
    return distance_regularized(tmp_organo, organo, variables,
                                solver, geom, model,
                                to_regularize=regularization['dic'],
                                reg_weight=regularization['weight'],
                                sum_residuals=sum_obj,
                                **minimize_opt)

def _obj_bfgs(initial_guess, organo, regularization, minimize_opt):
    dist = np.sum(_opt_dist(initial_guess, organo, regularization, True,
                            **minimize_opt))
    #ener = _opt_ener(initial_guess, organo, **minimize_opt)
    return dist

def _wrap_obj_and_const(var_table, **kwargs):
    if kwargs['pb_obj'] == 'min_ener':
        fun = _opt_ener(var_table, kwargs['organo'],
                        opt_tensions=kwargs['opt_tensions'],
                        **kwargs['minimize_opt'])
        const = _cst_dist(var_table, kwargs['organo'],
                          kwargs['initial_dist'],
                          kwargs['regularization'],
                          opt_tensions=kwargs['opt_tensions'],
                          **kwargs['minimize_opt'])
        fail = 0
    elif kwargs['pb_obj'] == 'min_dist':
        fun = _opt_dist(var_table, kwargs['organo'],
                        kwargs['regularization'],
                        True,
                        opt_tensions=kwargs['opt_tensions'],
                        **kwargs['minimize_opt'])
        const = _cst_ener(var_table, kwargs['organo'],
                          kwargs['initial_ener'],
                          opt_tensions=kwargs['opt_tensions'],
                          **kwargs['minimize_opt'])
        fail = 0
    else:
        fun = lambda x: x
        const = lambda x: x
        fail = 1
    return fun, const, fail

def _create_pyopt_model(obj_fun, initial_guess, main_min_opt,
                        pb_obj='min_dist'):
    if pb_obj == 'min_ener':
        opt_prob = pyOpt.Optimization('Energy minimization problem', obj_fun)
        opt_prob.addObj('energy')
        opt_prob.addVarGroup('L', len(initial_guess), 'c',
                             value=initial_guess,
                             lower=main_min_opt['lb'],
                             upper=main_min_opt['ub'])
        opt_prob.addConGroup('distance', int(2/3*len(initial_guess)), 'i')
        opt_prob.addConGroup('bounds', len(initial_guess), 'i')
    elif pb_obj == 'min_dist':
        opt_prob = pyOpt.Optimization('Distance minimization problem', obj_fun)
        opt_prob.addObj('distance')
        opt_prob.addVarGroup('L', len(initial_guess), 'c',
                             value=initial_guess,
                             lower=main_min_opt['lb'],
                             upper=main_min_opt['ub'])
        opt_prob.addCon('ener', 'i')
        opt_prob.addConGroup('bounds', len(initial_guess), 'i')
    return opt_prob

def prepare_tensions(organo, tension_array):
    """Match the tension in a reduced array to an organo dataset

    Parameters
    ----------
    organo : :class:`Epithelium` object
    tension_array : vector of initial line tensions (size 3*Nf)

    Return
    ----------
    tensions : np.ndarray of size 4*Nf
    the tensions array properly organised to fit into an organo dataset
    """
    tensions = organo.edge_df.line_tension.values
    # apical and basal edges
    tensions[:2*organo.Nf] = tension_array[:2*organo.Nf]

    tensions[2*organo.Nf:3*organo.Nf] = tension_array[2*organo.Nf:3*organo.Nf]
    tensions[3*organo.Nf:4*organo.Nf] = np.roll(
        tension_array[2*organo.Nf:3*organo.Nf], -1)
    return tensions

def set_init_point(r_in, r_out, nb_cells, alpha):
    """Define the initial point as proposed in the doc (in french)
    https://www.sharelatex.com/read/zdxptpnrryhc

    Parameters
    ----------
    r_in : float, the radius of the apical ring
    r_out : float, the radius of the basal ring
    Nf : int, the number of cells in the mesh
    alpha : float, the multiplicative coefficient between the mean area
      of the cells and the mean prefered area of the cells, i.e
      organo.face_df.prefered_area.mean() = alpha * organo.face_df.area.mean()

    Return
    ----------
    initial_point : np.ndarray of size 3*Nf
    the initial point for the optimization problems, according to the doc.
    """
    initial_point = np.zeros(3*nb_cells)
    area = (r_out**2-r_in**2)/2*np.sin(2*np.pi/nb_cells)
    initial_point[:nb_cells] = np.full(nb_cells,
                                       2*np.cos(np.pi/nb_cells)*
                                       area*(alpha-1)*(r_out-r_in))
    initial_point[2*nb_cells:] = np.full(nb_cells,
                                         np.sin(2*np.pi/nb_cells)/2*
                                         area*(alpha-1)*r_out)
    """
    Initial point with the Moore-Penrose pseudo inverse used to solve
    the underdetermination of the system. Gives very large lateral tensions.
    The mesh obtained does not represent an organoid.
    """
    """
    area = (r_out**2-r_in**2)/2*np.sin(2*np.pi/Nf)
    initial_point[:Nf] = np.full(Nf,
                                 ((1-alpha)*area*r_in*
                                  (np.sin(np.pi/Nf)**2+4)*
                                  np.sin(2*np.pi/Nf)*
                                  np.sin(np.pi/Nf)**2)/
                                 (np.sin(np.pi/Nf)**4+
                                  8*np.sin(np.pi/Nf)**2) +
                                 (4*(alpha-1)*area*r_out*
                                  np.sin(2*np.pi/Nf)*
                                  np.sin(np.pi/Nf)**2)/
                                 (np.sin(np.pi/Nf)**4+
                                  8*np.sin(np.pi/Nf)**2))
    initial_point[Nf:2*Nf] = np.full(Nf,
                                     ((alpha-1)*area*r_out*
                                      (np.sin(np.pi/Nf)**2+4)*
                                      np.sin(2*np.pi/Nf)*
                                      np.sin(np.pi/Nf)**2)/
                                     (np.sin(np.pi/Nf)**4+
                                      8*np.sin(np.pi/Nf)**2) +
                                     (4*(1-alpha)*area*r_in*
                                      np.sin(2*np.pi/Nf)*
                                      np.sin(np.pi/Nf)**2)/
                                     (np.sin(np.pi/Nf)**4+
                                      8*np.sin(np.pi/Nf)**2))
    initial_point[2*Nf:] = np.full(Nf,
                                   ((1-alpha)*area*r_in*np.sin(2*np.pi/Nf) *
                                    (8-2*(np.sin(np.pi/Nf)**2+4))) /
                                   (np.sin(np.pi/Nf)**4+
                                    8*np.sin(np.pi/Nf)**2) +
                                   ((alpha-1)*area*r_out*np.sin(2*np.pi/Nf) *
                                    (2*(np.sin(np.pi/Nf)**2+4)-8)) /
                                   (np.sin(np.pi/Nf)**4+
                                    8*np.sin(np.pi/Nf)**2))
    """
    return initial_point
