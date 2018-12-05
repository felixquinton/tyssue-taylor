"""Build the force inference problem
This module provides functions to define a force inference problem :
for each vertex of the mesh, identify adjacent vertices
for each edges, identify adjacent faces
build the M coefficient matrix that will be inverted.
We use this paper : Mechanical Stress inference
for Two Dimensional Cell Arrays, K.Chiou et al., 2012
//////\\\\\\
IN PROGRESS
\\\\\\//////
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
from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model
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

def _coef_matrix(organo, compute_pressions=False):
    organo.get_extra_indices()
    u_ij = organo.edge_df.eval('dx / length')
    v_ij = organo.edge_df.eval('dy / length')
    uv_ij = np.concatenate((u_ij, v_ij))

    coef_shape = (2*organo.Nv+1, organo.Ne)
    srce_rows = np.concatenate([
        organo.edge_df.srce,              ## x lines
        organo.edge_df.srce + organo.Nv   ## y lines
    ])

    trgt_rows = np.concatenate([
        organo.edge_df.trgt,              ## x lines
        organo.edge_df.trgt + organo.Nv   ## y lines
    ])

    cols = np.r_[:organo.Ne, :organo.Ne] ## [0 ... Ne, 0 ... Ne]

    coef_srce = sparse.coo_matrix((uv_ij, (srce_rows, cols)), shape=coef_shape)
    coef_trgt = sparse.coo_matrix((-uv_ij, (trgt_rows, cols)), shape=coef_shape)
    # Ones every where on the last line
    coef_sum_t = sparse.coo_matrix((np.ones(organo.Ne),
                                    (np.ones(organo.Ne)*2*organo.Nv,
                                     np.arange(organo.Ne))), shape=coef_shape)

    coef = coef_srce + coef_trgt + coef_sum_t

    # As tensions are equal for edge pairs, we can solve for only
    # the single edges. An index over only one half-edge per edge
    # can be obtained with:

    coef = coef[:, organo.sgle_edges].toarray()
    if compute_pressions:
        coef = np.hstack((coef, _pression_coefs(organo)))
        print(coef)
    return coef

def _pression_coefs(organo):
    beta_x = 0.5*organo.edge_df.dy[organo.sgle_edges].copy()
    beta_y = -0.5*organo.edge_df.dx[organo.sgle_edges].copy()
    #cell to cell coefficients are placed on 4 stacked diagonal matrix
    coef_lat_pres = np.vstack((
        np.diag(beta_x[np.intersect1d(organo.sgle_edges.values,
                                      organo.lateral_edges.values)]),
        -np.diag(beta_x[np.intersect1d(organo.sgle_edges.values,
                                       organo.lateral_edges.values)]),
        np.diag(beta_y[np.intersect1d(organo.sgle_edges.values,
                                      organo.lateral_edges.values)]),
        -np.diag(beta_y[np.intersect1d(organo.sgle_edges.values,
                                       organo.lateral_edges.values)])))
    #lumen to cell coefficients in a columnar vector
    coef_api_pres = np.concatenate((
        np.add(beta_x[organo.apical_edges],
               np.roll(beta_x[organo.apical_edges], 1)),
        np.zeros(organo.Nf),
        np.add(beta_y[organo.apical_edges],
               np.roll(beta_y[organo.apical_edges], 1)),
        np.zeros(organo.Nf)))
    #exterior to cell coefficients in a columnar vector
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
    return np.vstack((pres_coef, np.zeros(organo.Nf+2)))

def _moore_penrose_inverse(organo):
    coefs = _coef_matrix(organo)
    inv = np.linalg.pinv(coefs)
    #constant stands for the right side of equation (8) of the referenced paper
    constant = np.zeros(coefs.shape[0])
    constant[-1] = int(organo.Ne*0.75)
    system_sol = np.dot(inv, constant)
    return system_sol

def _nnls_model(organo, compute_pressions, verbose):
    coefs = _coef_matrix(organo, compute_pressions)
    constant = np.zeros(coefs.shape[0])
    constant[-1] = 0.01*int(organo.Ne*0.75)
    res, _ = nnls(coefs, constant)
    if verbose:
        print(res)
    return res

def _qp_obj(params, coefs, constant):
    coefxparam = np.dot(coefs, params)
    dotminusconstant = coefxparam - constant
    return np.dot(dotminusconstant, dotminusconstant)

def _qp_model(organo, init_method, verbose):
    coefs = _coef_matrix(organo)
    constant = np.zeros(coefs.shape[0])
    constant[-1] = 0.01*int(organo.Ne*0.75)
    bounds = [(0, None)]*int(organo.Ne*0.75)+[(None, None)]*(organo.Nf+2)
    if init_method == 'simple':
        init_point = np.zeros(int(organo.Ne*0.75)+organo.Nf+2)
    elif init_method == 'moore-penrose':
        init_point = _moore_penrose_inverse(organo)
        print('The initial point was obtained using the Moore-Penrose \
              pseudo-inverse to solve the linear system proposed in the paper.\
              Initial point : \n', init_point)
    res = minimize(_qp_obj,
                   init_point,
                   args=(coefs, constant),
                   method='L-BFGS-B',
                   bounds=bounds)
    if verbose:
        print('\n\n\n', res)
    return res.x

def infer_forces(organo, method='MP', init_method='simple',
                 compute_pressions=True, verbose=False):
    """Uses the functions defined above to compute the initial
    guess given by the force inference method with Moore-Penrose
    pseudo-inverse.
    *****************
    Parameters:
    organo :  :class:`Epithelium` object
    method : string
     one of 'MP' (default) for Moore-Penrose pseudo-inverse method, 'QP' to solve
     the model with quadratic programming (which ensure non negative tensions)
     or 'NNLS' which runs the non-negative least squares algorithm from
     Lawson C., Hanson R.J., (1987) Solving Least Squares Problems.
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
    if method == 'MP':
        system_sol = _moore_penrose_inverse(organo)
    elif method == 'QP':
        system_sol = _qp_model(organo, init_method, verbose)
    elif method == 'NNLS':
        system_sol = _nnls_model(organo, compute_pressions, verbose)
    if compute_pressions:
        dic_res = {'tensions': system_sol[:int(organo.Ne*0.75)],
                   'pressions': system_sol[int(organo.Ne*0.75):]}
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
        'face':{
            'is_alive': 1,
            'prefered_area':  list(ALPHA*ORGANO.face_df.area.values),
            'area_elasticity': 1.,},
        'edge':{
            'ux': 0.,
            'uy': 0.,
            'uz': 0.,
            'line_tension': 0.1,
            'is_active': 1
            },
        'vert':{
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
    COEFS = _coef_matrix(ORGANO, True)
    CONSTANT = np.zeros(COEFS.shape[0])
    CONSTANT[-1] = 0.01*int(ORGANO.Ne*0.75)
    DF_COEFS = pd.DataFrame(COEFS)
    DF_COEFS.to_csv('A_'+str(NF)+'cells.csv', index=False)
    DF_CONSTANT = pd.DataFrame(CONSTANT)
    DF_CONSTANT.to_csv('b_'+str(NF)+'cells.csv', index=False)
    RES_INFERENCE = infer_forces(ORGANO, 'NNLS', compute_pressions=False)
    DF_RES_INFERENCE = pd.DataFrame(RES_INFERENCE)
    DF_RES_INFERENCE.to_csv('x*_'+str(NF)+'cells.csv')
    print(RES_INFERENCE)
    print(ORGANO.edge_df)
