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
from scipy.optimize import minimize, nnls

from tyssue.generation import generate_ring
from tyssue.dynamics.planar_gradients import area_grad
from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model, lumen_area_grad
from tyssue_taylor.adjusters.adjust_annular import (set_init_point,
                                                    prepare_tensions)


def _adj_edges(organo):
    """Identify the adjacents edges for a given vertex
    *****************
    Parameters :
    organo : :class:`Epithelium` object
    vertex : int in the range 0, 3*Nf-1
    *****************
    Return :
    adj_edges : DataFrame containing the edges adjacent to vertex
    """
    srcs = np.reshape(organo.get_orbits('srce', 'trgt').index.labels[1],
                      (organo.Nv, 2))
    trgts = np.reshape(organo.get_orbits('trgt', 'srce').index.labels[1][::2],
                       (organo.Nv, 1))
    adj_edges = np.hstack((srcs, trgts))
    return adj_edges


def _adj_faces(organo):
    """Identify the couple of faces separated by the edges adjacent
    to a given vertex.
    *****************
    Parameters :
    organo : :class:`Epithelium` object
    vertex : int in the range 0, 3*Nf-1
    *****************
    Return :
    faces : dic with keys being the edges connected to vertex and
     containing the corresponding adjacent faces' indices
    REMARK : indice -1 stands for the lumen and -2 for the exterior
    """
    edges = organo.edge_df.iloc[_adj_edges(organo).flatten()]
    faces = np.zeros((edges.shape[0], 2), dtype=int)
    apical_inds = np.squeeze(np.argwhere(edges.segment == 'apical'))
    basal_inds = np.squeeze(np.argwhere(edges.segment == 'basal'))
    lateral_inds = np.squeeze(np.argwhere(edges.segment == 'lateral'))
    faces[apical_inds] = np.vstack((edges.face.iloc[apical_inds],
                                    -np.ones(apical_inds.shape))).T
    faces[basal_inds] = np.vstack((edges.face.iloc[basal_inds],
                                   -2*np.ones(basal_inds.shape))).T
    faces[lateral_inds] = np.vstack((edges.face.iloc[lateral_inds],
                                     np.roll(edges.face.iloc[lateral_inds],
                                             organo.Nf))).T
    return faces


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
    pres_coef = np.hstack((coef_lat_pres,
                           np.reshape(coef_api_pres, (2*organo.Nv, 1)),
                           np.reshape(coef_bas_pres, (2*organo.Nv, 1))))
    if not no_scale:
        pres_coef = np.vstack((pres_coef, np.zeros(organo.Nf+2)))
    return pres_coef


def _indices_areas_coefs(organo):
    """
    Build the indices to give to sparse.coo_matrix in order to built the
    area_coefs matrix
    """
    nb_api = organo.apical_edges.shape[0]
    nb_basal = organo.apical_edges.shape[0]
    apical_verts = organo.edge_df.srce[organo.apical_edges].values
    basal_verts = organo.edge_df.srce[organo.basal_edges].values
    apical_rows = np.c_[(apical_verts - 1) % nb_api,  # cell on the left
                        apical_verts % nb_api,  # cell on the right
                        np.full(nb_api, nb_api)].flatten()  # lumen
    # apical_rows = np.c_[apical_rows, ]
    basal_rows = np.c_[(basal_verts - 1) % nb_basal,  # cell on the left
                       basal_verts % nb_basal,  # cell on the right
                       np.full(nb_basal, nb_basal)].flatten()  # exterior
    apical_cols = np.repeat(np.arange(organo.apical_edges.shape[0]), 3)
    basal_cols = np.repeat(np.arange(organo.basal_edges.shape[0]), 3)
    return {'api_rows': apical_rows, 'api_cols': apical_cols,
            'bas_rows': basal_rows, 'bas_cols': basal_cols}


def _coefs_areas_coefs(organo,
                       grad_srce, grad_trgt,
                       grad_lumen_srce, grad_lumen_trgt):
    """
    Computes the coefficents for the matrix areas_coefs.
    """
    per_vertex_grad_x = (organo.sum_srce(grad_srce).gx +
                         organo.sum_trgt(grad_trgt).gx).values
    per_vertex_grad_y = (organo.sum_srce(grad_srce).gy +
                         organo.sum_trgt(grad_trgt).gy).values
    per_vertex_lumen_grad_x = (organo.sum_srce(grad_lumen_srce).gx +
                               organo.sum_trgt(grad_lumen_trgt).gx).values
    per_vertex_lumen_grad_y = (organo.sum_srce(grad_lumen_srce).gy +
                               organo.sum_trgt(grad_lumen_trgt).gy).values
    grad_factors = np.tile(organo.face_df.area_elasticity, 2)
    coefs_x_apical = np.multiply(grad_factors, per_vertex_grad_x)
    coefs_y_apical = np.multiply(grad_factors, per_vertex_grad_y)
    coefs_x_basal = np.multiply(grad_factors, per_vertex_grad_x)
    coefs_y_basal = np.multiply(grad_factors, per_vertex_grad_y)
    coefs_x_apical = np.insert(coefs_x_apical,
                               np.arange(1,
                                         2*organo.apical_edges.shape[0]+1,
                                         2),
                               per_vertex_lumen_grad_x[organo.apical_edges])
    coefs_y_apical = np.insert(coefs_y_apical,
                               np.arange(1,
                                         2*organo.apical_edges.shape[0]+1,
                                         2),
                               per_vertex_lumen_grad_y[organo.apical_edges])
    coefs_x_basal = np.insert(coefs_x_basal,
                              np.arange(1,
                                        2*organo.basal_edges.shape[0]+1,
                                        2),
                              per_vertex_lumen_grad_x[organo.basal_edges])
    coefs_y_basal = np.insert(coefs_y_basal,
                              np.arange(1,
                                        2*organo.basal_edges.shape[0]+1,
                                        2),
                              per_vertex_lumen_grad_y[organo.basal_edges])
    coefs_x_apical = np.divide(np.ones(coefs_x_apical.shape),
                               coefs_x_apical,
                               out=np.zeros_like(coefs_x_apical),
                               where=coefs_x_apical != 0)
    coefs_y_apical = np.divide(np.ones(coefs_y_apical.shape),
                               coefs_y_apical,
                               out=np.zeros_like(coefs_y_apical),
                               where=coefs_y_apical != 0)
    coefs_x_basal = np.divide(np.ones(coefs_x_basal.shape),
                              coefs_x_basal,
                              out=np.zeros_like(coefs_x_basal),
                              where=coefs_x_basal != 0)
    coefs_y_basal = np.divide(np.ones(coefs_y_basal.shape),
                              coefs_y_basal,
                              out=np.zeros_like(coefs_y_basal),
                              where=coefs_y_basal != 0)
    return {'api_x': -coefs_x_apical, 'api_y': -coefs_y_apical,
            'bas_x': -coefs_x_basal, 'bas_y': -coefs_y_basal}


def _areas_coefs(organo, no_scale):
    ind_dic = _indices_areas_coefs(organo)  # coo_matrix indices
    grad_srce, grad_trgt = area_grad(organo)  # cell's areas gradient
    grad_lumen_srce, grad_lumen_trgt = area_grad(organo)  # lumen gradient
    coef_shape = (organo.apical_edges.shape[0], organo.Nf+1)
    coefs = _coefs_areas_coefs(organo,
                               grad_srce, grad_trgt,
                               grad_lumen_srce, grad_lumen_trgt)
    # print(coefs)
    coef_x_apical = sparse.coo_matrix((coefs['api_x'],
                                       (ind_dic['api_cols'],
                                        ind_dic['api_rows'])),
                                      shape=(coef_shape))
    coef_x_basal = sparse.coo_matrix((coefs['api_y'],
                                      (ind_dic['bas_cols'],
                                       ind_dic['bas_rows'])),
                                     shape=(coef_shape))
    coef_y_apical = sparse.coo_matrix((coefs['bas_x'],
                                       (ind_dic['api_cols'],
                                        ind_dic['api_rows'])),
                                      shape=(coef_shape))
    coef_y_basal = sparse.coo_matrix((coefs['bas_y'],
                                      (ind_dic['bas_cols'],
                                       ind_dic['bas_rows'])),
                                     shape=(coef_shape))
    area_coefs = np.vstack((coef_x_apical.toarray(),
                            coef_x_basal.toarray(),
                            coef_y_apical.toarray(),
                            coef_y_basal.toarray()))
    if not no_scale:
        area_coefs = np.vstack((area_coefs, np.zeros(organo.Nf+1)))
    return area_coefs


def _right_side(organo, coefs):
    res = np.zeros(coefs.shape[0])
    res[-organo.Nf-1:-1] = organo.face_df.area
    res[-1] = organo.Ne*3/4
    # print(res)
    return res


def _moore_penrose_inverse(organo, sup_param, no_scale):
    coefs = _coef_matrix(organo, sup_param, no_scale)
    inv = np.linalg.pinv(coefs)
    # constant stands for the right side of equation (8) of
    # the referenced paper
    system_sol = np.dot(inv, _right_side(organo, coefs))
    return system_sol


def _nnls_model(organo, sup_param, no_scale, verbose):
    coefs = _coef_matrix(organo, sup_param, no_scale)
    res, _ = nnls(coefs, _right_side(organo, coefs))
    if verbose:
        print(res)
    return res


def _linear_algebra(organo, sup_param, no_scale):
    coefs = _coef_matrix(organo, sup_param, no_scale)
    system_sol = np.linalg.solve(coefs, _right_side(organo, coefs))
    return system_sol


def _qp_obj(params, coefs, constant):
    coefxparam = np.dot(coefs, params)
    dotminusconstant = coefxparam - constant
    return np.dot(dotminusconstant, dotminusconstant)


def _qp_model(organo, init_method, sup_param, no_scale, verbose):
    coefs = _coef_matrix(organo, sup_param, no_scale)
    bounds = [(0, None)]*int(organo.Ne*0.75)+[(0, None)]*(organo.Nf+1)
    if init_method == 'simple':
        init_point = np.zeros(int(organo.Ne*0.75)+organo.Nf+1)
    elif init_method == 'moore-penrose':
        init_point = _moore_penrose_inverse(organo, sup_param, no_scale)
        print('The initial point was obtained using the Moore-Penrose \
              pseudo-inverse to solve the linear system proposed in the paper.\
              Initial point : \n', init_point)
    res = minimize(_qp_obj,
                   init_point,
                   args=(coefs, _right_side(organo, coefs)),
                   method='L-BFGS-B',
                   bounds=bounds)
    if verbose:
        print('\n\n\n', res)
    return res.x


def infer_forces(organo, method='MP', init_method='simple',
                 sup_param='', no_scale=False, verbose=False):
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
    elif method == 'QP':
        system_sol = _qp_model(organo, init_method, sup_param,
                               no_scale, verbose)
    elif method == 'NNLS':
        system_sol = _nnls_model(organo, sup_param, no_scale, verbose)
    elif method == 'LINALG':
        system_sol = _linear_algebra(organo, sup_param, no_scale)
    if sup_param == 'pressions':
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)],
                   'pressions': system_sol[int(organo.Ne*0.75):]}
    elif sup_param == 'areas':
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)],
                   'areas': system_sol[int(organo.Ne*0.75):]}
    else:
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)]}
    return dic_res


if __name__ == "__main__":
    NF = 3
    ORGANO = generate_ring(NF, 1, 2)
    geom.update_all(ORGANO)
    ALPHA = 1 + 1/(20*(ORGANO.settings['R_out']-ORGANO.settings['R_in']))
    np.random.seed(1553)
    # Model parameters or specifications
    SPECS = {
        'face': {
            'is_alive': 1,
            'prefered_area':  list(ALPHA*ORGANO.face_df.area.values),
            'area_elasticity': 1., },
        'edge': {
            'ux': 0.,
            'uy': 0.,
            'uz': 0.,
            'line_tension': 0.1,
            'is_active': 1
            },
        'vert': {
            'adhesion_strength': 0.,
            'x_ecm': 0.,
            'y_ecm': 0.,
            'is_active': 1
            },
        'settings': {
            'lumen_elasticity': 0.1,
            'lumen_prefered_vol': ORGANO.settings['lumen_volume'],
            'lumen_volume': ORGANO.settings['lumen_volume']
            }
        }

    ORGANO.update_specs(SPECS, reset=True)
    geom.update_all(ORGANO)

    SYMETRIC_TENSIONS = np.multiply(set_init_point(ORGANO.settings['R_in'],
                                                   ORGANO.settings['R_out'],
                                                   ORGANO.Nf, ALPHA),
                                    np.random.normal(1, 0.002,
                                                     int(ORGANO.Ne*0.75)))
    SIN_MUL = 1+(np.sin(np.linspace(0, 2*np.pi, ORGANO.Nf, endpoint=False)))**2
    ORGANO.face_df.prefered_area *= np.random.normal(1.0, 0.05, ORGANO.Nf)
    ORGANO.edge_df.line_tension = prepare_tensions(ORGANO, SYMETRIC_TENSIONS)
    ORGANO.edge_df.loc[:ORGANO.Nf-1, 'line_tension'] *= SIN_MUL

    ORGANO.vert_df[['x_ecm', 'y_ecm']] = ORGANO.vert_df[['x', 'y']]
    ORGANO.vert_df.loc[ORGANO.basal_verts, 'adhesion_strength'] = 0.01

    NEW_TENSIONS = ORGANO.edge_df.line_tension

    ORGANO.edge_df.loc[:, 'line_tension'] = NEW_TENSIONS

    RES = Solver.find_energy_min(ORGANO, geom, model)
    COEFS = _coef_matrix(ORGANO, 'areas')
    CONSTANT = _right_side(ORGANO, COEFS)
    DF_COEFS = pd.DataFrame(COEFS)
    import matplotlib.pyplot as plt
    for i in COEFS:
        print(i)
    # DF_COEFS.to_csv('A_'+str(NF)+'cells.csv', index=False)
    DF_CONSTANT = pd.DataFrame(CONSTANT)
    # DF_CONSTANT.to_csv('b_'+str(NF)+'cells.csv', index=False)
    # RES_INFERENCE = _linear_model(ORGANO, 'areas')
    # RES_INFERENCE = _qp_model(ORGANO, 'simple', True, 0)
    RES_INFERENCE = infer_forces(ORGANO, method='NNLS',
                                 sup_param='areas', no_scale=False)
    print('Mean tension :', RES_INFERENCE['tensions'].mean())
    TO_VECT_RES = np.concatenate((RES_INFERENCE['tensions'],
                                  RES_INFERENCE['areas']))
    TO_VECT_RES = np.array(RES_INFERENCE['tensions'])
    print(RES_INFERENCE)
    DF_RES_INFERENCE = pd.DataFrame(TO_VECT_RES)
    # DF_RES_INFERENCE.to_csv('x*_'+str(NF)+'nnls_cells.csv')
