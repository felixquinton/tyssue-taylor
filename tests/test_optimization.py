#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:35:28 2018

@author: fquinton
"""
import os
import numpy as np

from tyssue.generation import generate_ring
from tyssue import config

from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.adjusters.adjust_annular import (prepare_tensions,
                                                    _prepare_params,
                                                    _opt_dist,
                                                    adjust_parameters)
CURRENT_DIR = os.path.abspath(__file__)


def test_prepare_tensions():
    organo = generate_ring(3, 1, 2)
    organo.edge_df.loc[:, 'line_tension'] = np.ones(12)
    tension_array = organo.edge_df.loc[:, 'line_tension'][:3*organo.Nf]
    tension_array[0] += 1
    tension_array[3*organo.Nf-1] += 1
    tensions = prepare_tensions(organo, tension_array)
    assert len(tensions) == 4*organo.Nf
    assert np.all(np.equal(tensions[:3*organo.Nf],
                           tension_array[:3*organo.Nf]))
    assert np.all(np.equal(np.roll(tensions[3*organo.Nf:], 1),
                           tension_array[2*organo.Nf:]))


def test_prepare_params():
    organo = _create_org()
    var_table = np.r_[np.ones(organo.Nf*4), -np.ones(organo.Nf+1)]
    parameters = [('edge', 'line_tension'), ('face', 'prefered_area')]
    tmp_organo = organo.copy()
    split_inds = np.cumsum([organo.datasets[elem][column].size
                            for elem, column in parameters])
    splitted_var = np.split(var_table, split_inds[:-1])
    res = _prepare_params(organo, splitted_var, parameters)
    assert len(res) == 3
    assert np.array_equal(res[('edge', 'line_tension')],
                          np.ones(organo.Nf*4))
    assert np.array_equal(res[('face', 'prefered_area')],
                          -np.ones(organo.Nf))
    assert res[('lumen_prefered_vol', None)] == -1


def test_opt_dist():
    organo = _create_org()
    var_table = np.zeros(organo.Nf*4+1)
    minimize_opt = config.solvers.minimize_spec()
    lm_opt = {'method': 'lm', 'xtol': 1e-7, 'ftol': 1e-6, 'verbose': 1}
    parameters = [('edge', 'line_tension'), ('face', 'prefered_area')]
    error = _opt_dist(var_table, organo, parameters, **lm_opt)
    assert isinstance(error, np.ndarray)
    assert len(list(error)) == organo.Nv + organo.Nf*3


def test_adjust_parameters():
    organo = _create_org()
    var_table = np.zeros(organo.Nf*4+1)
    lm_opt = {'method': 'lm', 'xtol': 1e-7, 'ftol': 1e-6, 'verbose': 1}
    res = adjust_parameters(organo, var_table, **lm_opt)
    assert res.success


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
