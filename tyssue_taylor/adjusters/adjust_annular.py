import numpy as np

from .cost_functions import energy, distance_regularized
from ..models.annular import AnnularGeometry as geom
from ..models.annular import model
from tyssue.solvers.sheet_vertex_solver import Solver as solver
from tyssue import config
from scipy.optimize import minimize, least_squares
import pyOpt


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
    distance_min_opt : scipy.optimize.minize option dictionnary for the
                        ditance minimization
    """
    minimize_opt = config.solvers.minimize_spec()

    if energy_min_opt is not None:
        minimize_opt.update(energy_min_opt)

    if main_min_opt['method'] in ('trf', 'lm'):
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
            psqp.setOption('IPRINT', 0)

            [fstr, xstr, inform] = psqp(opt_prob,sens_type='FD',
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
    energy_min = energy(tmp_organo, variables, solver, geom, model,
                        **minimize_opt)
    return energy_min

def _wrap_obj_and_const(tension_array, **kwargs):
    f = _opt_ener(tension_array, kwargs['organo'], **kwargs['minimize_opt'])
    g = -_cst_dist(tension_array, kwargs['organo'], kwargs['initial_dist'],
                   kwargs['regularization'], **kwargs['minimize_opt'])
    fail = 0
    return f, g, fail

def _create_pyOpt_model(obj_fun, initial_guess, main_min_opt):
    opt_prob = pyOpt.Optimization('Energy minimization problem', obj_fun)
    opt_prob.addObj('energy')
    opt_prob.addVarGroup('L', len(initial_guess), 'c',
                         value=initial_guess, lower=main_min_opt['lb'],
                         upper=main_min_opt['ub'])
    opt_prob.addCon('distance', 'i')
    return opt_prob


def prepare_tensions(organo, tension_array):
    Nf = organo.Nf
    tensions = organo.edge_df.line_tension.values
    # apical and basal edges
    tensions[: 2*Nf] = tension_array[: 2*Nf]

    tensions[2*Nf: 3*Nf] = tension_array[2*Nf:3*Nf]
    tensions[3*Nf: 4*Nf] = np.roll(tension_array[2*Nf:3*Nf], 1)
    return tensions
