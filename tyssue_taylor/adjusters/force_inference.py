"""Build the force inference problem
This module provides functions to define a force inference problem :
for each vertex of the mesh, identify adjacent vertices
for each edges, identify adjacent faces
build the M coefficient matrix that will be inverted.
We use this paper : Mechanical Stress inference
for Two Dimensional Cell Arrays, K.Chiou et al., 2012
To use force inference, please call the infer_forces function.
The doc-string of infer_forces is given below :
    Uses the functions defined above to compute the initial
    guess given by the force inference method with Moore-Penrose
    pseudo-inverse.
    *****************
    Parameters:
    organo :  :class:`Epithelium` object
    method : string
     one of 'MP' (default) for Moore-Penrose pseudo-inverse method, 'QP' to
     solve the model with quadratic programming (which ensure non negative
     tensions) or 'NNLS' which runs the non-negative least squares algorithm
     from Lawson C., Hanson R.J., (1987) Solving Least Squares Problems.
    init_method : string
     argument to define the initialization method for the QP method.
     One of 'simple'(default) : initialize with vector of zeros.
            'moore-penrose' : initialize with the Moore-Penrose initial point.
    compute_pressions : boolean
     If True, the method computes tensions and pressions. If False, the method
     computes only tensions.
    verbose : boolean
     If True, print the inital point.
    *****************
    Returns
    dic with key :  tensions : the vector of tensions
                    pressions : the vector of pressions if computed
    *****************
"""
import itertools
import numpy as np
import pandas as pd
import argparse

from scipy import sparse
from scipy.optimize import nnls

from tyssue.dynamics.planar_gradients import area_grad
from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model, lumen_area_grad
from tyssue_taylor.models.display import create_organo
from tyssue_taylor.adjusters.adjust_annular import (set_init_point,
                                                    prepare_tensions)


def _coef_matrix(organo, sup_param='', no_scale=False):
    organo.get_extra_indices()
    u_ij = organo.edge_df.eval('dx / length')
    v_ij = organo.edge_df.eval('dy / length')
    uv_ij = np.concatenate((u_ij, v_ij))

    coef_shape = (2*organo.Nv+(not no_scale), organo.Ne)
    srce_rows = np.concatenate([
        organo.edge_df.srce,              # x lines
        organo.edge_df.srce + organo.Nv   # y lines
    ])

    trgt_rows = np.concatenate([
        organo.edge_df.trgt,              # x lines
        organo.edge_df.trgt + organo.Nv   # y lines
    ])

    cols = np.r_[:organo.Ne, :organo.Ne]  # [0 ... Ne, 0 ... Ne]

    coef_srce = sparse.coo_matrix((uv_ij, (srce_rows, cols)),
                                  shape=coef_shape)
    # Inversing the coefs for trgt edges so that the force is applied away from
    # the vertex.
    coef_trgt = sparse.coo_matrix((-uv_ij, (trgt_rows, cols)),
                                  shape=coef_shape)
    # Ones every where on the last line
    if no_scale:
        coef = coef_srce + coef_trgt
    else:
        coef_sum_t = sparse.coo_matrix((np.ones(organo.Ne),
                                        (np.ones(organo.Ne)*2*organo.Nv,
                                         np.arange(organo.Ne))),
                                       shape=coef_shape)
        coef = coef_srce + coef_trgt + coef_sum_t

    # As tensions are equal for edge pairs, we can solve for only
    # the single edges. An index over only one half-edge per edge
    # can be obtained with:

    coef = coef[:, organo.sgle_edges].toarray()
    if sup_param == 'areas':
        coef = np.c_[coef, _areas_coefs(organo, no_scale)]
    elif sup_param == 'pressions':
        coef = np.hstack((coef, _pression_coefs(organo, no_scale)))
    return coef


def _pression_coefs(organo, no_scale):
    organo.get_extra_indices()
    beta_x = 0.5*organo.edge_df.dy[organo.sgle_edges].copy()
    beta_y = -0.5*organo.edge_df.dx[organo.sgle_edges].copy()
    # cell to cell coefficients are placed on 4 stacked diagonal matrix
    coef_lat_pres = np.vstack((
        np.diag(beta_x[np.intersect1d(organo.sgle_edges.values,
                                      organo.lateral_edges.values)]),
        -np.diag(beta_x[np.intersect1d(organo.sgle_edges.values,
                                       organo.lateral_edges.values)]),
        np.diag(beta_y[np.intersect1d(organo.sgle_edges.values,
                                      organo.lateral_edges.values)]),
        -np.diag(beta_y[np.intersect1d(organo.sgle_edges.values,
                                       organo.lateral_edges.values)])))
    coef_api_pres = np.concatenate((
        np.add(beta_x[organo.apical_edges],
               np.roll(beta_x[organo.apical_edges], 1)),
        np.zeros(organo.Nf),
        np.add(beta_y[organo.apical_edges],
               np.roll(beta_y[organo.apical_edges], 1)),
        np.zeros(organo.Nf)))
    # exterior to cell coefficients in a columnar vector
    coef_bas_pres = np.concatenate((
        np.zeros(organo.Nf),
        np.add(beta_x[organo.basal_edges],
               np.roll(beta_x[organo.basal_edges], 1)),
        np.zeros(organo.Nf),
        np.add(beta_y[organo.basal_edges],
               np.roll(beta_y[organo.basal_edges], 1))))
    coef_bas_pres = np.zeros(coef_bas_pres.shape)
    pres_coef = np.hstack((coef_lat_pres,
                           np.reshape(coef_api_pres, (2*organo.Nv, 1)),
                           np.reshape(coef_bas_pres, (2*organo.Nv, 1))))
    if not no_scale:
        pres_coef = np.vstack((pres_coef, np.zeros(organo.Nf+2)))
    return pres_coef


def _areas_coefs(organo, no_scale):
    grad_srce, grad_trgt = area_grad(organo)
    grad_lumen_srce, grad_lumen_trgt = lumen_area_grad(organo)
    grouped_srce = grad_srce.groupby(organo.edge_df.srce)
    grouped_trgt = grad_trgt.groupby(organo.edge_df.trgt)
    grouped_lumen_srce = grad_lumen_srce.groupby(organo.edge_df.srce)
    grouped_lumen_trgt = grad_lumen_trgt.groupby(organo.edge_df.trgt)
    area_coefs = np.zeros((2*organo.Nv, organo.Nf+1))
    for vertex in range(organo.Nv):
        adj_srce = grouped_srce.get_group(
            list(grouped_srce.groups.keys())[vertex])
        adj_trgt = grouped_trgt.get_group(
            list(grouped_trgt.groups.keys())[vertex])[::-1]
        adj_lumen_srce = grouped_lumen_srce.get_group(
            list(grouped_lumen_srce.groups.keys())[vertex])
        adj_lumen_trgt = grouped_lumen_trgt.get_group(
            list(grouped_lumen_trgt.groups.keys())[vertex])[::-1]
        coefs_cols = organo.edge_df.loc[adj_srce.index, 'face']
        area_coefs[vertex][coefs_cols] = (adj_srce.gx.values +
                                          adj_trgt.gx.values)
        area_coefs[organo.Nv+vertex][coefs_cols] = (adj_srce.gy.values +
                                                    adj_trgt.gy.values)
        area_coefs[vertex][organo.Nf] = np.sum((adj_lumen_srce.gx,
                                               adj_lumen_trgt.gx))
        area_coefs[organo.Nv+vertex][organo.Nf] = np.sum((adj_lumen_srce.gy,
                                                          adj_lumen_trgt.gy))
    area_elasticity = np.tile(np.hstack([organo.face_df.area_elasticity,
                                         organo.settings['lumen_elasticity']]),
                              (2*organo.Nv, 1))
    area_coefs = np.multiply(area_coefs, area_elasticity)
    if not no_scale:
        area_coefs = np.vstack((area_coefs, np.zeros(organo.Nf+1)))
    return area_coefs


def _right_side(organo, coefs):
    res = np.zeros(coefs.shape[0])
    res[-1] = (organo.edge_df.line_tension.mean() /
               (organo.face_df.area_elasticity.mean() *
                organo.face_df.prefered_area.mean()**1.5))
    return res


def _moore_penrose_inverse(organo, sup_param, no_scale):
    coefs = _coef_matrix(organo, sup_param, no_scale)
    inv = np.linalg.pinv(coefs)
    system_sol = np.dot(inv, _right_side(organo, coefs))
    return system_sol


def _nnls_model(organo, sup_param, no_scale, verbose):
    coefs = _coef_matrix(organo, sup_param, no_scale)
    res, _ = nnls(coefs, _right_side(organo, coefs))
    if verbose:
        print(res)
    return res


def _linear_algebra(organo, sup_param, no_scale, mult=None):
    coefs = _coef_matrix(organo, sup_param, no_scale)
    if mult is not None:
        coefs = np.multiply(coefs, mult)
    system_sol = np.linalg.solve(coefs, _right_side(organo, coefs))
    return system_sol


def infer_forces(organo, method='MP', init_method='simple',
                 sup_param='', no_scale=False, mult=None, verbose=False):
    """Uses the functions defined above to compute the initial
    guess given by the force inference method with Moore-Penrose
    pseudo-inverse.
    *****************
    Parameters:
    organo :  :class:`Epithelium` object
    method : string
     one of 'MP' (default) for Moore-Penrose pseudo-inverse method, 'QP' to
     solve the model with quadratic programming (which ensure non negative
     tensions) or 'NNLS' which runs the non-negative least squares algorithm
     from Lawson C., Hanson R.J., (1987) Solving Least Squares Problems.
     Update 12/10/2018 : the problem can be solve using basic linear algebra
     ig there is the same number of equations and variables. Use 'LINALG'.
    init_method : string
     argument to define the initialization method for the QP method.
     One of 'simple'(default) : initialize with vector of zeros.
            'moore-penrose' : initialize with the Moore-Penrose initial point.
    sup_param : string, one of '', 'pressions' or 'areas'
     '': computes only tensions
     'pressions': compute tensions and pressions
     'areas': computes tensions and the difference between cells area and cells
      prefered area.
    verbose : boolean
     If True, print the inital point.
    *****************
    Returns
    dic with key :  tensions : the vector of tensions
                    pressions : the vector of pressions if computed
    *****************
    """
    if method == 'MP':
        system_sol = _moore_penrose_inverse(organo, sup_param, no_scale)
    elif method == 'NNLS':
        system_sol = _nnls_model(organo, sup_param, no_scale, verbose)
    elif method == 'LINALG':
        system_sol = _linear_algebra(organo, sup_param, no_scale, mult=mult)
    if sup_param == 'pressions':
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)],
                   'pressions': system_sol[int(organo.Ne*0.75):]}
    elif sup_param == 'areas':
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)],
                   'areas': system_sol[int(organo.Ne*0.75):]}
    else:
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)]}
    return dic_res


def _print_solving_results(organo, fi_res, coefs, constant,
                           sup_param=None):
    """Used to print the results when the module is run from the console.
    """
    print('Ideal scale factor: ',
          np.sum(ORGANO.edge_df.line_tension[:3*ORGANO.Nf]))
    print('A :\n', coefs)
    print('b :', constant)
    vect_res = np.array(fi_res['tensions'])
    if sup_param is not None:
        vect_res = np.hstack((fi_res['tensions'], fi_res[sup_param]))
    vect_param = organo.edge_df.line_tension[:3*organo.Nf]
    if sup_param == 'areas':
        vect_param = np.concatenate((vect_param,
                                     organo.face_df.area -
                                     organo.face_df.prefered_area,
                                     [organo.settings['lumen_volume'] -
                                      organo.settings['lumen_prefered_vol']]))
    print('t :', vect_param)
    print('Ax*-b: ', np.dot(coefs, vect_res)-constant)
    print('||Ax*-b||: ', np.linalg.norm(np.dot(coefs, vect_res) - constant))
    if COEFS.shape[1] == vect_param.shape[0]:
        print('At-b: ', np.dot(coefs, vect_param) - constant)
        print('||At-b||: ', np.linalg.norm(np.dot(coefs, vect_param) -
                                           constant))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Adjust line tensions')
    PARSER.add_argument('--nnls', action='store_true',
                        help='indicates if the solver must use NNLS.')
    PARSER.add_argument('--mp', action='store_true',
                        help='indicates if the solver must use Moore-Penrose.')
    PARSER.add_argument('--linalg', action='store_true',
                        help='indicates if the solver must use LINALG.')
    PARSER.add_argument('--pressions', action='store_true',
                        help='indicates if the force inference\
                        must compute pressions.')
    PARSER.add_argument('--areas', action='store_true',
                        help='indicates if the force inference\
                        must compute areas.')
    ARGS = vars(PARSER.parse_args())
    SUP_PARAM = None
    if ARGS['pressions']:
        SUP_PARAM = 'pressions'
    elif ARGS['areas']:
        SUP_PARAM = 'areas'
    METHOD = 'LINALG'
    if ARGS['nnls']:
        METHOD = 'NNLS'
    elif ARGS['mp']:
        METHOD = 'MP'
    NF, R_IN, R_OUT = (3, 1, 4)
    ORGANO = create_organo(NF, R_IN, R_OUT, rot=np.pi/12)
    ORGANO.edge_df.loc[:NF, 'line_tension'] *= 2
    ORGANO.edge_df.loc[NF:2*NF-1, 'line_tension'] = 0
    geom.update_all(ORGANO)
    Solver.find_energy_min(ORGANO, geom, model)
    COEFS = _coef_matrix(ORGANO, sup_param=SUP_PARAM)
    CONSTANT = _right_side(ORGANO, COEFS)
    RES_INFERENCE = infer_forces(ORGANO, method=METHOD,
                                 sup_param=SUP_PARAM, no_scale=False)
    _print_solving_results(ORGANO, RES_INFERENCE, COEFS, CONSTANT, SUP_PARAM)
    print(RES_INFERENCE)
