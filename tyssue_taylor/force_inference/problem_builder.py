"""Build the force inference problem
This module provides functions to define a force inference problem :
for each vertex of the mesh, identify adjacent vertices
for each edges, identify adjacent faces
build the M coefficient matrix that will be inverted.
//////\\\\\\
IN PROGRESS
\\\\\\//////
"""
import numpy as np
import pandas as pd

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
    faces : dic with keys being the vertices connected to vertex and
     containing the corresponding adjacent faces' indices
    /!\ REMARK : indice -1 stands for the lumen and -2 for the exterior
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
    faces[lat_index] = tuple(faces.keys())
    return faces

def _collect_data(organo):
    """Create a dictionnay with for each vertex, the adjacent edges
    and corresponding faces.
    """
    data = {}
    for vertex in organo.vert_df:
        data[vertex] = _adj_faces(organo, vertex)
    return data

def coef_matrix(organo):
    """Write the coefficient matrix for the linear system
    *****************
    Parameters:
    organo :  :class:`Epithelium` object
    *****************
    returns
    coefs : np.ndarray containing the coefficients for the tensions
     and pressions
    """
    data = _collect_data(organo)
    coefs = np.zeros((organo.Nv-2, organo.Ne+organo.Nf))
