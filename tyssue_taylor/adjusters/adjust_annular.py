"""This module provides functions to build the optimization process
"""

import numpy as np
from tyssue.solvers.sheet_vertex_solver import Solver as solver
from tyssue import config
from scipy.optimize import minimize, least_squares
import pyOpt

from .cost_functions import _energy, distance_regularized
from ..models.annular import AnnularGeometry as geom
from ..models.annular import model


def adjust_tensions(organo, initial_guess, regularization,
                    energy_min_opt=None, initial_min_opt=None,
                    **main_min_opt):
    """Find the line tensions which minimize the distance to the epithelium

    Parameters
    ----------
    organo : :class:`Epithelium` object
    initial_guess : vector of initial line tensions (size 3*Nf)
    regularization : dictionnary with fields :
                        'dic' : dictionnary with fields 'apical' and 'basal'
                        and boolean value indicating if the corresponding set
                        of edges should be regularized.
                        'weight' : float, weight of the regularization module
    energy_min_opt : scipy.optimize.minize option dictionnary for the energy
                     minimization
    initial_min_opt : scipy.optimize.minimize option dictionnary for the
                      initial point search. Ignored if main_min_opt['method']
                      is not 'PSQP' or 'SLSQP'
    main_min_opt : option dictionnary for the main optimization. Syntax depends
                   on the method. For bgfs and SLSQP use the scipy.optimize.minimize
                   syntax. For trf and lm use the scipy.optimize.least_squares
                   method. For PSQP the syntax is:
                   'lb': lower bound of the Parameters
                   'ub': upper bound of the Parameters
                   'method': PSQP
    """
    minimize_opt = config.solvers.minimize_spec()

    if energy_min_opt is not None:
        minimize_opt.update(energy_min_opt)

    if main_min_opt['method'] == 'bfgs':
        return minimize(_obj_bfgs, initial_guess, **main_min_opt,
                        args=(organo, regularization, minimize_opt))
    elif main_min_opt['method'] in ('trf', 'lm'):
        return least_squares(_opt_dist, initial_guess, **main_min_opt,
                             args=(organo, regularization),
                             kwargs=minimize_opt)
    elif main_min_opt['method'] == 'SLSQP':
        print("Starting the search for the initial point")
        initial_point = least_squares(_opt_dist, initial_guess,
                                      **initial_min_opt,
                                      args=(organo, regularization),
                                      kwargs=minimize_opt)
        if initial_point.success:
            print(f"Initial point found with distance {initial_point.fun[0]}\n\
                    Starting the energy minimization.")
            initial_guess = initial_point.x
            initial_dist = initial_point.fun
            return minimize(_opt_ener, initial_guess,
                            args=(organo),
                            constraints=[{'type': 'ineq', 'fun': _cst_dist,
                                          'args': (organo, initial_dist,
                                                   regularization)}],
                            **main_min_opt)
        print(f"Initial point search failed with message :\
              \nf{initial_point.message}")
    elif main_min_opt['method'] == 'PSQP':
        #print("Starting the search for the initial point")
        initial_point = least_squares(_opt_dist, initial_guess,
                                      **initial_min_opt,
                                      args=(organo, regularization),
                                      kwargs=minimize_opt)
        if initial_point.success:
            #print(f"Initial point found with distance {initial_point.fun[0]}\n\
            #Starting the energy minimization.")
            initial_guess = initial_point.x
            initial_dist = initial_point.fun
            opt_prob = _create_pyOpt_model(_wrap_obj_and_const, initial_guess,
                                           main_min_opt)
            psqp = pyOpt.PSQP()
            psqp.setOption('IPRINT', 2)

            [fstr, xstr, inform] = psqp(opt_prob, sens_type='FD',
                                        sens_mode='pgc',
                                        organo=organo,
                                        regularization=regularization,
                                        initial_dist=initial_dist,
                                        minimize_opt=minimize_opt)
            return {'fun': fstr, 'x': xstr, 'message': inform}
        print(f"Initial point search failed with message :\
              \nf{initial_point.message}")

    else:
        print(f"Unknown method : f{main_min_opt['method']}")



def _opt_dist(tension_array, organo, regularization,
              **minimize_opt):

    tmp_organo = organo.copy()
    variables = {}
    tensions = prepare_tensions(tmp_organo, tension_array[:3*tmp_organo.Nf])
    variables[('edge', 'line_tension')] = tensions
    if len(tension_array) > 3*tmp_organo.Nf:
        lumen_volume = tension_array[3*tmp_organo.Nf]
        variables[('lumen_prefered_vol', None)] = lumen_volume
    error = distance_regularized(tmp_organo, organo, variables,
                                 regularization['dic'],
                                 regularization['weight'],
                                 solver, geom, model,
                                 **minimize_opt)
    return error


def _cst_dist(tension_array, organo, initial_dist, regularization,
              **minimize_opt):

    tmp_organo = organo.copy()
    variables = {}
    tensions = prepare_tensions(organo, tension_array[:3*tmp_organo.Nf])
    variables[('edge', 'line_tension')] = tensions
    if len(tension_array) > 3*tmp_organo.Nf:
        lumen_volume = tension_array[3*tmp_organo.Nf]
        variables[('lumen_prefered_vol', None)] = lumen_volume
    error = distance_regularized(tmp_organo, organo, variables,
                                 regularization['dic'],
                                 regularization['weight'],
                                 solver, geom, model,
                                 **minimize_opt)
    return  initial_dist - error.sum()

def _opt_ener(tension_array, organo, **minimize_opt):

    tmp_organo = organo.copy()
    variables = {}
    tensions = prepare_tensions(organo, tension_array[:3*tmp_organo.Nf])
    variables[('edge', 'line_tension')] = tensions
    if len(tension_array) > 3*tmp_organo.Nf:
        lumen_volume = tension_array[3*tmp_organo.Nf]
        variables[('lumen_prefered_vol', None)] = lumen_volume
    energy_min = _energy(tmp_organo, variables, solver, geom, model,
                         **minimize_opt)
    return energy_min

def _wrap_obj_and_const(tension_array, **kwargs):
    fun = _opt_ener(tension_array, kwargs['organo'], **kwargs['minimize_opt'])
    const = -_cst_dist(tension_array, kwargs['organo'], kwargs['initial_dist'],
                       kwargs['regularization'], **kwargs['minimize_opt'])
    fail = 0
    return fun, const, fail

def _create_pyOpt_model(obj_fun, initial_guess, main_min_opt):
    opt_prob = pyOpt.Optimization('Energy minimization problem', obj_fun)
    opt_prob.addObj('energy')
    opt_prob.addVarGroup('L', len(initial_guess), 'c',
                         value=initial_guess, lower=main_min_opt['lb'],
                         upper=main_min_opt['ub'])
    opt_prob.addCon('distance', 'i')
    return opt_prob


def _obj_bfgs(initial_guess, organo, regularization, minimize_opt):
    return np.sum(_opt_dist(initial_guess, organo, regularization,
                            **minimize_opt))

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
    Nf = organo.Nf
    tensions = organo.edge_df.line_tension.values
    # apical and basal edges
    tensions[: 2*Nf] = tension_array[: 2*Nf]

    tensions[2*Nf: 3*Nf] = tension_array[2*Nf:3*Nf]
    tensions[3*Nf: 4*Nf] = np.roll(tension_array[2*Nf:3*Nf], -1)
    return tensions

def set_init_point(r_in, r_out, Nf, alpha):
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
    initial_point = np.zeros(3*Nf)
    area = (r_out**2-r_in**2)/2*np.sin(2*np.pi/Nf)
    initial_point[:Nf] = np.full(Nf,
                                 2*np.cos(np.pi/Nf)*area*(alpha-1)*(r_out-r_in))
    initial_point[2*Nf:] = np.full(Nf,
                                   np.sin(2*np.pi/Nf)/2*area*(alpha-1)*r_out)
    return initial_point
