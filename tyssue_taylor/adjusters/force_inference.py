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

from scipy.optimize import minimize, nnls

from tyssue.generation import generate_ring
from tyssue_taylor.models.annular import AnnularGeometry as geom

def _adj_edges(organo, vertex):
    """Identify the adjacents edges for a given vertex
    *****************
    Parameters :
    organo : :class:`Epithelium` object
    vertex : int in the range 0, 3*Nf-1
    *****************
    Return :
    adj_edges : DataFrame containing the edges adjacent to vertex
    """
    is_source = organo.edge_df[organo.edge_df.srce == vertex]
    is_target = organo.edge_df[organo.edge_df.trgt == vertex]
    adj_edges = pd.concat([is_source,
                           is_target[is_target.segment != 'lateral']])
    return adj_edges

def _adj_faces(organo, vertex):
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
    edges = _adj_edges(organo, vertex)
    faces = {}
    for index, edge in edges.iterrows():
        if edge.segment == 'apical':
            faces[index] = (edge.face, -1)
        elif edge.segment == 'basal':
            faces[index] = (edge.face, -2)
        else:
            lat_index = index
    faces[lat_index] = tuple(faces[key][0] for key in faces)
    return faces

def _collect_data(organo):
    """Create a dictionnay with for each vertex, the adjacent edges
    and corresponding faces.
    """
    data = {}
    for ind, _ in organo.vert_df.iterrows():
        data[ind] = _adj_faces(organo, ind)
    return data

def _coef_matrix(organo, compute_pressions=True):
    """Write the coefficient matrix for the linear system
    *****************
    Parameters:
    organo :  :class:`Epithelium` object
    *****************
    Returns
    coefs : np.ndarray containing the coefficients for the tensions
     and pressions
    *****************
    The problem is formulated as M*phi=C (see Mechanical Stress inference
    for Two Dimensional Cell Arrays, K.Chiou et al., 2012).
    The parameter vector phi contains the Ne linear tensions and the Nf
    pressions.
    We could set 2*Nv constraints but K.Chiou et al., 2012 discard
    three of them because of symetries.
    We add two additional constraints as recommended in the supplementary text
    S1 : the mean tension must be equal to a constant. We chose e*0.01 as we
    want the average tensions to be around 0.01. Also, the pression of the
    exterior is set to 0.
    """
    data = _collect_data(organo)
    Ne = int(organo.Ne*0.75)
    #in the coefs, we add a cell for the interior and for the exterior
    if compute_pressions:
        coefs = np.zeros((2*organo.Nv, Ne+(organo.Nf+2)))
    else:
        coefs = np.zeros((2*organo.Nv, Ne))
    for vertex in data:
        for edge in data[vertex]:
            edge_vertices = (organo.edge_df.srce[edge],
                             organo.edge_df.trgt[edge])
            if edge_vertices[0] != vertex:
                edge_vertices = (organo.edge_df.trgt[edge],
                                 organo.edge_df.srce[edge])
            #coef for the first term in equation (1)
            edge_ind = edge

            if edge_ind >= Ne:
                edge_ind -= int(organo.Ne*0.25)
            coefs[vertex][edge_ind] = ((organo.vert_df.x[edge_vertices[1]] -
                                        organo.vert_df.x[edge_vertices[0]]) /
                                       organo.edge_df.length[edge])
            coefs[organo.Nv+vertex][edge_ind] = ((organo.vert_df.y[edge_vertices[1]] -
                                                  organo.vert_df.y[edge_vertices[0]]) /
                                                 organo.edge_df.length[edge])
            if compute_pressions:
                for ind, face in enumerate(data[vertex][edge]):
                    coord_dif_y = (organo.vert_df.y[edge_vertices[1]] -
                                   organo.vert_df.y[edge_vertices[0]])
                    coord_dif_x = (organo.vert_df.x[edge_vertices[1]] -
                                   organo.vert_df.x[edge_vertices[0]])
                    if face >= 0:
                        #coef for the second term in equation (1)
                        coefs[organo.Nv+vertex][Ne+face] = ((1-2*ind) *
                                                            coord_dif_y/2)
                        coefs[organo.Nv+vertex][Ne+face] = (-(1-2*ind) *
                                                            coord_dif_x/2)
                    elif face == -1: #if the face is the interior
                        coefs[organo.Nv+vertex][face] = ((1-2*ind) *
                                                         coord_dif_y/2)
                        coefs[organo.Nv+vertex][face] = (-(1-2*ind) *
                                                         coord_dif_x/2)
                    else: #if the face is the interior
                        coefs[organo.Nv+vertex][face] = ((1-2*ind) *
                                                         coord_dif_y/2)
                        coefs[organo.Nv+vertex][face] = (-(1-2*ind) *
                                                         coord_dif_x/2)

    #coefs = np.delete(coefs, (Ne-2, Ne-1, Ne+organo.Nf+1), axis=0)
    #coefs = np.append(coefs, [[0]*Ne+[0]*(organo.Nf)+[1, 0],
    #                          [1]*Ne+[0]*(organo.Nf+2)], axis=0)
    if compute_pressions:
        coefs = coefs[:-1, :]
        coefs = np.append(coefs, [[0]*Ne+[0]*(organo.Nf)+[1, 0],
                                  [1]*Ne+[0]*(organo.Nf+2)], axis=0)
    else:
        coefs = np.append(coefs, [[1]*Ne], axis=0)
    return coefs

def _moore_penrose_inverse(organo):
    coefs = _coef_matrix(organo)
    inv = np.linalg.pinv(coefs)
    #C stands for the right side of equation (8) of the referenced paper
    C = np.zeros(coefs.shape[0])
    C[-1] = int(organo.Ne*0.75)
    system_sol = np.dot(inv, C)
    return system_sol

def _nnls_model(organo, compute_pressions, verbose):
    coefs = _coef_matrix(organo, compute_pressions)
    C = np.zeros(coefs.shape[0])
    C[-1] = 0.01*int(organo.Ne*0.75)
    res, _ = nnls(coefs, C)
    if verbose:
        print(res)
    return res


def _qp_obj(x, coefs, C):
    coefxparam = np.dot(coefs, x)
    dotminusC = coefxparam - C
    return np.dot(dotminusC, dotminusC)

def _qp_model(organo, init_method, verbose):
    coefs = _coef_matrix(organo)
    C = np.zeros(coefs.shape[0])
    C[-1] = 0.01*int(organo.Ne*0.75)
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
                   args=(coefs, C),
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
    ORGANO = generate_ring(3, 1, 2)
    geom.update_all(ORGANO)
    ALPHA = 1 + 1/(20*(ORGANO.settings['R_out']-ORGANO.settings['R_in']))


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
    print(infer_forces(ORGANO, 'NNLS', compute_pressions=True))
    print(infer_forces(ORGANO, 'NNLS', compute_pressions=False))
