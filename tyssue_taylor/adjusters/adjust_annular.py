"""This module provides functions to build the optimization process
"""

import numpy as np

from tyssue.solvers.sheet_vertex_solver import Solver as solver
from tyssue import config
from scipy.optimize import minimize, least_squares

from tyssue_taylor.adjusters.cost_functions import distance_regularized
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model


def adjust_parameters(eptm, initial_guess,
                      parameters=[('edge', 'line_tension'),
                                  ('face', 'prefered_area')],
                      energy_min_opt=None,
                      iprint_file=None,
                      COPY_OR_SYM='copy',
                      **main_min_opt):
    """Find the line tensions which minimize the distance to the epithelium

    Parameters
    ----------
    eptm : :class:`Epithelium` object
    initial_guess : vector of initial parameters. Length depends of the
      parameters to optimize.
      !! MUST BE ORDERED ACCORDING TO parameters !!
    parameters : list of string couples. Each couple indicates the dataframe
      and the dataframe's column that contains the optimization parameters.
    energy_min_opt : scipy.optimize.minize option dictionnary for the energy
                     minimization.
    iprint_file : string. Path to a csv or txt file to print the objective
                    function evaluations during the optimization process.
    main_min_opt : option dictionnary for the main optimization. Syntax depends
                   on the method. For bgfs and SLSQP use the
                   scipy.optimize.minimize syntax. For trf and lm use the
                   scipy.optimize.least_squares method.
                   For PSQP the syntax is :
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
    if main_min_opt['method'] in ('trf', 'lm'):
        return least_squares(_opt_dist, initial_guess, **main_min_opt,
                             args=(organo, parameters,
                                   iprint_file, COPY_OR_SYM),
                             kwargs=minimize_opt)
    else:
        print(f"Unknown method : f{main_min_opt['method']}")
    return -1


def _opt_dist(var_table, organo, parameters,
              iprint_file=None, COPY_OR_SYM='copy',
              **minimize_opt):
    tmp_organo = organo.copy()
    tmp_organo.get_extra_indices()
    split_inds = np.cumsum([organo.datasets[elem][column].size
                            for elem, column in parameters])
    last_tensions_ind = tmp_organo.sgle_edges.shape[0]
    var_table = np.r_[
        prepare_tensions(tmp_organo, var_table[:last_tensions_ind]),
        var_table[last_tensions_ind:]]
    splitted_var = np.split(var_table, split_inds[:-1])
    variables = _prepare_params(tmp_organo, splitted_var, parameters)
    return distance_regularized(tmp_organo, organo, variables,
                                solver, geom, model,
                                coords=tmp_organo.coords,
                                IPRINT=iprint_file,
                                COPY_OR_SYM='copy',
                                **minimize_opt)


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
    tensions[:2*organo.Nf] = tension_array[:2*organo.Nf]

    tensions[2*organo.Nf:3*organo.Nf] = tension_array[2*organo.Nf:3*organo.Nf]
    tensions[3*organo.Nf:4*organo.Nf] = np.roll(
        tension_array[2*organo.Nf:3*organo.Nf], -1)
    return tensions


def _prepare_params(organo, splitted_var, parameters):
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
    variables = {}
    for ind, (elem, param) in enumerate(parameters):
        if param == 'line_tension':
            variables[(elem, param)] = prepare_tensions(
                organo, splitted_var[ind])
        elif (param == 'prefered_area' and
              len(splitted_var[ind]) % organo.Nf != 0):
            variables[('lumen_prefered_vol', None)] = splitted_var[ind][-1]
            variables[(elem, param)] = splitted_var[ind][:-1]
        else:
            try:
                variables[(elem, param)] = splitted_var[ind]
            except KeyError as key_error:
                print('Called key is unknown too the datasets: ', key_error)
            except IndexError as ind_error:
                print('Not enougth indices in parameters vector: ', key_error)
    return variables
