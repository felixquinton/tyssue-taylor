"""Test the force inference module
"""
import numpy as np
import pandas as pd

from tyssue.generation import generate_ring
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.adjusters.force_inference import (_adj_edges,
                                                     _adj_faces,
                                                     _collect_data,
                                                     _coef_matrix,
                                                     infer_forces)


def test_adj_edges():
    organo = _create_org()
    answer = [(0, 11, 2),
              (1, 9, 0),
              (2, 10, 1),
              (5, 6, 3),
              (3, 7, 4),
              (4, 8, 5)]
    for ind, _ in organo.vert_df.iterrows():
        assert tuple(_adj_edges(organo, ind).index) == answer[ind]

def test_adj_faces():
    organo = _create_org()
    answer = [{0: (0, -1), 2: (2, -1), 11: (0, 2)},
              {1: (1, -1), 0: (0, -1), 9: (1, 0)},
              {2: (2, -1), 1: (1, -1), 10: (2, 1)},
              {5: (2, -2), 3: (0, -2), 6: (2, 0)},
              {3: (0, -2), 4: (1, -2), 7: (0, 1)},
              {4: (1, -2), 5: (2, -2), 8: (1, 2)}]
    for ind, _ in organo.vert_df.iterrows():
        assert _adj_faces(organo, ind) == answer[ind]


def test_collect_data():
    organo = _create_org()
    answer = {0: {0: (0, -1), 2: (2, -1), 11: (0, 2)},
              1: {1: (1, -1), 0: (0, -1), 9: (1, 0)},
              2: {2: (2, -1), 1: (1, -1), 10: (2, 1)},
              3: {5: (2, -2), 3: (0, -2), 6: (2, 0)},
              4: {3: (0, -2), 4: (1, -2), 7: (0, 1)},
              5: {4: (1, -2), 5: (2, -2), 8: (1, 2)}}

    for ind, _ in organo.vert_df.iterrows():
        assert _collect_data(organo)[ind] == answer[ind]

def test_coef_matrix():
    organo = _create_org()
    answer = [[-0.866, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0 ,0.5, 0, 0],
              [0.866, 0.866, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
              [0, -0.866, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0],
              [0, 0, 0, -0.866, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0.866, 0.866, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, -0.866, 0, 0, 0, -0.5, 0, 0, 0, 0, 0],
              [0.5, 0, 1, 0, 0, 0, 0, 0, 0, -0.25, 0, 0.25, 0, 0],
              [-0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, -0.5, 0.5, 0, 0, 0.75],
              [0, -0.5, -1, 0, 0, 0, 0, 0, 0, 0, 0.25, -0.25, 0, -0.75],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
    matrix = _coef_matrix(organo)
    for line_ind, line in enumerate(matrix):
        for element_ind, coef in enumerate(line):
            assert round(coef, 3) == answer[line_ind][element_ind]

def test_infer_forces():
    organo = _create_org()
    init_param = infer_forces(organo)
    assert 'tensions' in init_param
    assert 'pressions' in init_param
    assert len(init_param['tensions']) == int(organo.Ne*0.75)
    assert len(init_param['pressions']) == organo.Nf+2
    assert round(np.mean(init_param['tensions']), 8) == 0.01
    assert round(init_param['pressions'][-2], 8) == 0

def _create_org():
    organo = generate_ring(3, 1, 2)
    geom.update_all(organo)
    alpha = 1 + 1/(20*(organo.settings['R_out']-organo.settings['R_in']))


    # Model parameters or specifications
    specs = {
        'face':{
            'is_alive': 1,
            'prefered_area':  list(alpha*organo.face_df.area.values),
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
            'lumen_prefered_vol': organo.settings['lumen_volume'],
            'lumen_volume': organo.settings['lumen_volume']
            }
        }

    organo.update_specs(specs, reset=True)
    geom.update_all(organo)
    return organo
