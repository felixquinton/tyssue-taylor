"""Test the force inference module
"""
import numpy as np
import pandas as pd

from tyssue.generation import generate_ring
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.adjusters.force_inference import (_coef_matrix,
                                                     _areas_coefs,
                                                     _infer_pol,
                                                     _get_sim_param,
                                                     _right_side,
                                                     _t_per_cell_coefs,
                                                     infer_forces,
                                                     _tmp_infer_forces,
                                                     _opt_cst_obj,
                                                     opt_sum_lambda)


def test_infer_pol():
    organo = generate_ring(3, 1, 2)
    pol_organo = organo.copy()
    pol_organo.vert_df.loc[pol_organo.basal_verts, ('x', 'y')] *= 1.1
    geom.update_all(pol_organo)
    for val in _infer_pol(pol_organo):
        assert round(val, 9) == 0.9


def test_get_sim_param():
    Nf = 3
    Ra = 1
    Rb = 2
    res = _get_sim_param(Nf, Ra, Rb)
    assert round(res[0], 6) == -0.00448
    assert round(res[1], 6) == 0.00224
    assert round(res[2], 6) == 0.00776

    res2 = _get_sim_param(Nf, Ra, Rb, sum_lbda=0.02)
    assert round(res[0], 6) == 0.5*round(res2[0], 6)
    assert round(res[1], 6) == 0.5*round(res2[1], 6)
    assert round(res[2], 6) == 0.5*round(res2[2], 6)


def test_right_side():
    organo = generate_ring(3, 1, 2)
    pol_organo = organo.copy()
    pol_organo.vert_df.loc[pol_organo.basal_verts, ('x', 'y')] *= 1.1
    geom.update_all(pol_organo)
    cst = _right_side(pol_organo)
    assert isinstance(cst, np.ndarray)
    assert np.array_equal(cst.shape, (15,))
    assert np.array_equal(cst[:12], np.zeros(12))
    assert not np.isnan(cst[12:]).any()


def test_t_per_cell_coefs():
    organo = generate_ring(3, 1, 2)
    t_per_cell = _t_per_cell_coefs(organo,
                                   _get_sim_param(3, 1, 2),
                                   _infer_pol(organo))
    assert isinstance(t_per_cell, np.ndarray)
    assert np.array_equal(t_per_cell.shape, (3, 13))
    assert np.array_equal(t_per_cell[:, :3], np.eye(3))
    assert np.array_equal(t_per_cell[:, 3:6], np.eye(3))
    assert np.array_equal(t_per_cell[:, 6:9], np.roll(np.eye(3)
                                                      + np.roll(np.eye(3), 1,
                                                                axis=0),
                                                      1, axis=1))
    assert np.array_equal(t_per_cell[:, 9:], np.zeros((3, 4)))


def test_areas_coefs():
    organo = _create_org()
    ar_coefs = _areas_coefs(organo)
    assert isinstance(ar_coefs, np.ndarray)
    assert np.array_equal(ar_coefs.shape, (12, 4))


def test_coef_matrix():
    organo = _create_org()
    coef_mat = _coef_matrix(organo)
    ar_coefs = _areas_coefs(organo)
    t_per_cell = _t_per_cell_coefs(organo,
                                   _get_sim_param(3, 1, 2),
                                   _infer_pol(organo))
    assert isinstance(coef_mat, np.ndarray)
    assert np.array_equal(coef_mat.shape, (15, 13))
    assert np.array_equal(coef_mat[:12, 9:], ar_coefs)
    assert np.array_equal(coef_mat[12:], t_per_cell)


def test_tmp_infer_forces():
    organo = _create_org()
    _tmp_infer_forces(organo, 0.01)


def test_opt_cst_obj():
    organo = _create_org()
    dist = _opt_cst_obj(0.02, organo)
    assert dist > 0


def test_opt_sum_lambda():
    organo = _create_org()
    res = opt_sum_lambda(organo)
    assert type(res) == float
    assert res > 0


def test_infer_forces():
    organo = _create_org()
    dic = infer_forces(organo)
    assert isinstance(dic['tensions'], np.ndarray)
    assert isinstance(dic['areas'], np.ndarray)
    assert np.array_equal(dic['tensions'].shape, (9,))
    assert np.array_equal(dic['areas'].shape, (4,))


def _create_org():
    organo = generate_ring(3, 1, 2)
    geom.update_all(organo)
    alpha = 1 + 1/(20*(organo.settings['R_out']-organo.settings['R_in']))
    specs = {
        'face': {
            'is_alive': 1,
            'prefered_area':  list(alpha*organo.face_df.area.values),
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
            'lumen_prefered_vol': organo.settings['lumen_volume'],
            'lumen_volume': organo.settings['lumen_volume']
            }
        }

    organo.update_specs(specs, reset=True)
    geom.update_all(organo)
    return organo
