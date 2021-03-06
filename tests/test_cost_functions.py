#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:35:28 2018

@author: fquinton
"""
import os
import numpy as np

from tyssue.generation import generate_ring
from tyssue.solvers.sheet_vertex_solver import Solver

from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model
from tyssue_taylor.adjusters.cost_functions import (_tension_bounds,
                                                    _distance,
                                                    distance_regularized)
CURRENT_DIR = os.path.abspath(__file__)


def test_tension_bounds():
    organo = generate_ring(3, 1, 2)
    organo.edge_df.loc[:, 'line_tension'] = np.ones(12)
    organo.edge_df.loc[(0, 1), 'line_tension'] = [-1, 2*1e3]
    penalties = _tension_bounds(organo)
    assert len(penalties) == 3*organo.Nf
    assert penalties[0] == 1e3
    assert penalties[1] == 0
    assert np.argwhere(penalties != 0).squeeze() == np.array(0)


def test_distance():
    exp_organo = generate_ring(3, 1, 2)
    th_organo = generate_ring(3, 1, 2)
    exp_organo.vert_df.loc[0, 'x'] += 1
    distance = _distance(exp_organo, th_organo)
    assert np.all(np.equal(distance,
                           np.concatenate(([1], np.zeros(exp_organo.Nv-1)))))


def test_distance_regularized():
    exp_organo = generate_ring(3, 1, 2)
    th_organo = generate_ring(3, 1, 2)
    geom.update_all(exp_organo)
    geom.update_all(th_organo)
    specs = {
        'face': {
            'is_alive': 1,
            'prefered_area':  1.1*th_organo.face_df.area,
            'area_elasticity': 1, },
        'edge': {
            'ux': 0.,
            'uy': 0.,
            'uz': 0.,
            'line_tension': 0.01,
            'is_active': 1
            },
        'vert': {
            'adhesion_strength': 0.,
            'x_ecm': 0.,
            'y_ecm': 0.,
            'is_active': 1
            },
        'settings': {
            'lumen_elasticity': 0.01,
            'lumen_prefered_vol': th_organo.settings['lumen_volume'],
            'lumen_volume': th_organo.settings['lumen_volume']
            }
        }
    exp_organo.update_specs(specs, reset=True)
    th_organo.update_specs(specs, reset=True)
    param_tension_only = {('edge', 'line_tension'): np.ones(12)}
    param_with_lumen = {('edge', 'line_tension'): np.ones(12),
                        ('lumen_prefered_vol', None):
                        th_organo.settings['lumen_volume']}
    for basal_reg, apical_reg in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for variables in (param_with_lumen, param_tension_only):
            to_regularize = {'dic': {'apical': basal_reg, 'basal': apical_reg}}
            reg_weight = 0.01
            energy_opt = {'options': {'gtol': 1e-1, 'ftol': 1e-1}}

            exp_organo.vert_df.loc[0, 'x'] += 1
            res = distance_regularized(exp_organo, th_organo, variables,
                                       Solver, geom, model, **energy_opt)
            assert isinstance(res, np.ndarray)
            assert res.all() >= 0
            assert res.any() > 0
