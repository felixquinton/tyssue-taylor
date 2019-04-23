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
import numpy as np
import pandas as pd

from scipy import sparse
from scipy.optimize import nnls, minimize_scalar

from tyssue.dynamics.planar_gradients import area_grad
from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model, lumen_area_grad
from tyssue_taylor.models.display import create_organo
from tyssue_taylor.adjusters.adjust_annular import prepare_tensions
from tyssue_taylor.adjusters.cost_functions import _distance


def infer_forces(organo,  sup_param="areas", t_sum=0.01, verbose=False):
    """Uses the functions defined above to compute the initial
    guess given by the force inference method with Moore-Penrose
    pseudo-inverse.
    *****************
    Parameters:
    organo :  :class:`Epithelium` object
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
                    areas : the vector of areas if computed
    *****************
    """
    system_sol = _nnls_model(organo, sup_param, t_sum, verbose)
    if sup_param == "areas":
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)],
                   'areas': system_sol[int(organo.Ne*0.75):]}
    else:
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)]}
    return dic_res


def opt_sum_lambda(eptm, method="golden"):
    """Solve for the optimal total tension in the mesh.
    *****************
    Parameters:
    eptm :  :class:`Epithelium` object
    method : string, one of 'golden' or 'bounded'
     the method used to optimize the sum of tensions.
    *****************
    Returns
    float : the optimal sum of tensions.
    *****************
    """
    tmp_eptm = eptm.copy()
    cst_opt_result = minimize_scalar(_opt_cst_obj,
                                     args=(tmp_eptm.copy()),
                                     method=method)
    return cst_opt_result.x


def _opt_cst_obj(sum_tensions, eptm):
    th_eptm = eptm.copy()
    exp_eptm = _tmp_infer_forces(th_eptm,
                                 sum_tensions).copy()
    dist = np.sum(_distance(th_eptm, exp_eptm))
    print(dist)
    return dist


def _tmp_infer_forces(eptm,
                      t_sum,
                      sup_param="areas"):
    tmp_organo = eptm.copy()
    fi_params = infer_forces(eptm, sup_param, t_sum, verbose=False)

    tmp_organo.edge_df.loc[:, 'line_tension'] = (
        prepare_tensions(tmp_organo, fi_params['tensions']))
    tmp_organo.face_df.loc[:, 'prefered_area'] = (
        tmp_organo.face_df.area.values + fi_params['areas'][:-1])
    tmp_organo.settings['lumen_prefered_vol'] = (
        tmp_organo.settings['lumen_volume'] + fi_params['areas'][-1])

    Solver.find_energy_min(tmp_organo, geom, model)

    return tmp_organo


def _coef_matrix(organo, sup_param='', t_sum=0.01):
    organo.get_extra_indices()
    u_ij = organo.edge_df.eval('dx / length')
    v_ij = organo.edge_df.eval('dy / length')
    uv_ij = np.concatenate((u_ij, v_ij))

    coef_shape = (2*organo.Nv, organo.Ne)
    srce_rows = np.concatenate([
        organo.edge_df.srce,
        organo.edge_df.srce + organo.Nv
    ])

    trgt_rows = np.concatenate([
        organo.edge_df.trgt,
        organo.edge_df.trgt + organo.Nv
    ])

    cols = np.r_[:organo.Ne, :organo.Ne]

    coef_srce = sparse.coo_matrix((uv_ij, (srce_rows, cols)),
                                  shape=coef_shape)
    coef_trgt = sparse.coo_matrix((-uv_ij, (trgt_rows, cols)),
                                  shape=coef_shape)
    coef = coef_srce + coef_trgt

    coef = coef[:, :organo.Nf*3].toarray()
    if sup_param == 'areas':
        coef = np.c_[coef, _areas_coefs(organo)]

    coef = np.r_[coef, _t_per_cell_coefs(organo,
                                         _get_sim_param(organo.Nf,
                                                        organo.settings['R_in'],
                                                        organo.settings['R_out'],
                                                        t_sum),
                                         _infer_pol(organo),
                                         sup_param=sup_param)]
    return coef


def _areas_coefs(organo):
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
    return area_coefs


def _t_per_cell_coefs(eptm,
                      params_in_sym_mesh,
                      polar_coefs,
                      sup_param='areas'):
    """
    Discussion sur l'estimation de la polarisation. Voir _infer_pol
    """
    coefs_tables = np.c_[polar_coefs, np.ones((eptm.Nf, 3))].flatten()
    base_col_ind = np.arange(eptm.Nf)
    col_inds = np.c_[base_col_ind,
                     base_col_ind + eptm.Nf,
                     base_col_ind + 2*eptm.Nf,
                     (base_col_ind + 1) % (eptm.Nf) + 2*eptm.Nf].flatten()
    row_inds = np.repeat(np.arange(eptm.Nf), 4)
    if sup_param == 'areas':
        matrix_shape = (eptm.Nf, 4*eptm.Nf+1)
    else:
        matrix_shape = (eptm.Nf, 3*eptm.Nf)
    coef_matrix = sparse.coo_matrix((coefs_tables, (row_inds, col_inds)),
                                    shape=matrix_shape)

    return coef_matrix.toarray()


def _right_side(eptm, t_sum=0.01):
    params_in_sym_mesh = _get_sim_param(eptm.Nf,
                                        eptm.settings['R_in'],
                                        eptm.settings['R_out'],
                                        sum_lbda=t_sum)
    far_cste = 4*((params_in_sym_mesh[1:].sum() + params_in_sym_mesh[-1])/3 /
                  (abs(eptm.face_df.area.mean() +
                       params_in_sym_mesh[0]))**1.5)
    far_table = np.full(int(eptm.Nf), far_cste)
    return np.r_[np.zeros(2*eptm.Nv), far_table]


def _get_sim_param(Nf, Ra, Rb, sum_lbda=0.01):
    sin_t = np.sin(np.pi/Nf)
    sin_2t = np.sin(2*np.pi/Nf)

    dAdRa = - sin_2t * Ra
    dAdRb = sin_2t * Rb
    dladRa = dlbdRb = 2*sin_t

    denom = dAdRa+dAdRb+dAdRb*dladRa

    area_dif = -dladRa*sum_lbda/denom
    lambda_a = sum_lbda*(dAdRa+dAdRb)/denom
    lambda_h = dladRa*dAdRb*sum_lbda/denom

    return np.array((area_dif, lambda_a, lambda_h))


def _infer_pol(eptm):
    """
    Remarque : il ne faut pas faire real_dist / sym_dist mais l'inverse.
    Lorsque les cellules s'allongent le centre recule.
    Lorsque les cellules s'élargissent en basal et se retrécissent
    en apical le centroïde avance.
    Le premier effet est plus fort donc les cellules polarizée ont
    un centroïde plus loin de l'edge apical.
    Du moins dans ce mesh...
    """
    non_lateral_edges = np.concatenate((eptm.apical_edges,
                                        eptm.basal_edges))
    polar_coefs = np.ones(eptm.Nf)

    cell_centers = np.c_[eptm.face_df.loc[:, eptm.coords]]
    api_vs = np.c_[eptm.vert_df.loc[eptm.apical_verts, eptm.coords]]

    a = ((np.roll(api_vs[:, 1], 1)-api_vs[:, 1]) /
         (np.roll(api_vs[:, 0], 1)-api_vs[:, 0]))
    b = api_vs[:, 1] - a*api_vs[:, 0]

    dist = np.divide(np.abs(np.multiply(a, cell_centers[:, 0]) -
                            cell_centers[:, 1] + b),
                     np.sqrt(a**2 + 1))

    sym_height = (eptm.settings['R_out'] -
                  eptm.settings['R_in']) * np.cos(np.pi/eptm.Nf)
    sin_t = np.sin(np.pi/eptm.Nf)
    sym_la = 2*eptm.settings['R_in']*sin_t
    sym_lb = 2*eptm.settings['R_out']*sin_t
    sym_dist = sym_height/3 * (sym_la + 2*sym_lb)/(sym_la + sym_lb)

    polar_coefs = dist
    return polar_coefs


def _nnls_model(organo, sup_param, t_sum, verbose):
    res, _ = nnls(_coef_matrix(organo, sup_param, t_sum),
                  _right_side(organo, t_sum))
    if verbose:
        print(res)
    return res


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
    SUP_PARAM = None
    SUP_PARAM = 'areas'
    METHOD = 'NNLS'
    NF, R_IN, R_OUT = (3, 1, 2)
    ORGANO = create_organo(NF, R_IN, R_OUT)
    ORGANO.edge_df.loc[:NF, 'line_tension'] *= 2
    ORGANO.edge_df.loc[NF:2*NF-1, 'line_tension'] = 0
    geom.update_all(ORGANO)
    print(ORGANO.vert_df.loc[:, ('x', 'y')])
    # Solver.find_energy_min(ORGANO, geom, model)
    COEFS = _coef_matrix(ORGANO, sup_param=SUP_PARAM)
    CONSTANT = _right_side(ORGANO, COEFS)
    RES_INFERENCE = infer_forces(ORGANO, method=METHOD,
                                 sup_param=SUP_PARAM, no_scale=False)
    print(ORGANO.vert_df.loc[:, ('x', 'y')])
    print(ORGANO.edge_df.loc[:, ('srce', 'trgt', 'length')])
    _print_solving_results(ORGANO, RES_INFERENCE, COEFS, CONSTANT, SUP_PARAM)
    pd.DataFrame(COEFS).to_csv('A_symetric.csv')
    pd.DataFrame(CONSTANT).to_csv('b_symetric.csv')
    print(RES_INFERENCE)
