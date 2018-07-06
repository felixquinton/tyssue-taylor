#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:35:28 2018

@author: fquinton
"""
import os
import numpy as np
import pyOpt

from tyssue.generation import generate_ring
from tyssue.solvers.sheet_vertex_solver import Solver

from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model
from tyssue_taylor.adjusters.adjust_annular import (prepare_tensions,
                                                    _opt_dist,
                                                    _cst_dist,
                                                    _opt_ener,
                                                    _wrap_obj_and_const,
                                                    _create_pyOpt_model,
                                                    set_init_point)
CURRENT_DIR = os.path.abspath(__file__)

def test_prepare_tensions():
    organo = generate_ring(3, 1, 2)
    organo.edge_df.loc[:, 'line_tension'] = np.ones(12)
    tension_array = organo.edge_df.loc[:, 'line_tension'][:3*organo.Nf]
    tension_array[0] += 1
    tension_array[3*organo.Nf-1] += 1
    tensions = prepare_tensions(organo, tension_array)
    assert len(tensions) == 4*organo.Nf
    assert np.all(np.equal(tensions[:3*organo.Nf], tension_array[:3*organo.Nf]))
    assert np.all(np.equal(np.roll(tensions[3*organo.Nf:], -1),
                           tension_array[2*organo.Nf:]))

def test_opt_dist():
    organo = generate_ring(3, 1, 2)
    geom.update_all(organo)
    specs = {
        'face':{
            'is_alive': 1,
            'prefered_area':  1.1*organo.face_df.area,
            'area_elasticity': 1,},
        'edge':{
            'ux': 0.,
            'uy': 0.,
            'uz': 0.,
            'line_tension': 0.01,
            'is_active': 1
            },
        'vert':{
            'adhesion_strength': 0.,
            'x_ecm': 0.,
            'y_ecm': 0.,
            'is_active': 1
            },
        'settings': {
            'lumen_elasticity': 0.01,
            'lumen_prefered_vol': organo.settings['lumen_volume'],
            'lumen_volume': organo.settings['lumen_volume']
            }
        }
    organo.update_specs(specs, reset=True)
    tension_array = organo.edge_df.loc[:, 'line_tension'][:3*organo.Nf]
    regularization = {'dic':{'apical' : True, 'basal': True},
                      'weight': 0.01}
    energy_opt = {'options': {'gtol': 1e-1, 'ftol': 1e-1}}
    error = _opt_dist(tension_array, organo, regularization, **energy_opt)
    assert isinstance(error, np.ndarray)
    tension_array = np.concatenate(
        (organo.edge_df.loc[:, 'line_tension'][:3*organo.Nf],
         [organo.settings['lumen_prefered_vol']]))
    error = _opt_dist(tension_array, organo, regularization, **energy_opt)
    assert isinstance(error, np.ndarray)


def test_cst_dist():
    organo = generate_ring(3, 1, 2)
    geom.update_all(organo)
    specs = {
        'face':{
            'is_alive': 1,
            'prefered_area':  1.1*organo.face_df.area,
            'area_elasticity': 1,},
        'edge':{
            'ux': 0.,
            'uy': 0.,
            'uz': 0.,
            'line_tension': 0.01,
            'is_active': 1
            },
        'vert':{
            'adhesion_strength': 0.,
            'x_ecm': 0.,
            'y_ecm': 0.,
            'is_active': 1
            },
        'settings': {
            'lumen_elasticity': 0.01,
            'lumen_prefered_vol': organo.settings['lumen_volume'],
            'lumen_volume': organo.settings['lumen_volume']
            }
        }
    organo.update_specs(specs, reset=True)
    tension_array = organo.edge_df.loc[:, 'line_tension'][:3*organo.Nf]
    regularization = {'dic':{'apical' : True, 'basal': True},
                      'weight': 0.01}
    initial_dist = 1
    energy_opt = {'options': {'gtol': 1e-1, 'ftol': 1e-1}}
    error = _cst_dist(tension_array, organo, initial_dist,
                      regularization, **energy_opt)
    assert isinstance(error, float)
    tension_array = np.concatenate(
        (organo.edge_df.loc[:, 'line_tension'][:3*organo.Nf],
         [organo.settings['lumen_prefered_vol']]))
    error = _cst_dist(tension_array, organo, initial_dist,
                      regularization, **energy_opt)
    assert isinstance(error, float)

def test_opt_ener():
    organo = generate_ring(3, 1, 2)
    geom.update_all(organo)
    specs = {
        'face':{
            'is_alive': 1,
            'prefered_area':  1.1*organo.face_df.area,
            'area_elasticity': 1,},
        'edge':{
            'ux': 0.,
            'uy': 0.,
            'uz': 0.,
            'line_tension': 0.01,
            'is_active': 1
            },
        'vert':{
            'adhesion_strength': 0.,
            'x_ecm': 0.,
            'y_ecm': 0.,
            'is_active': 1
            },
        'settings': {
            'lumen_elasticity': 0.01,
            'lumen_prefered_vol': organo.settings['lumen_volume'],
            'lumen_volume': organo.settings['lumen_volume']
            }
        }
    organo.update_specs(specs, reset=True)
    tension_array = organo.edge_df.loc[:, 'line_tension'][:3*organo.Nf]
    energy_opt = {'options': {'gtol': 1e-1, 'ftol': 1e-1}}
    ener = _opt_ener(tension_array, organo, **energy_opt)
    assert isinstance(ener, float)
    assert ener > 0
    tension_array = np.concatenate(
        (organo.edge_df.loc[:, 'line_tension'][:3*organo.Nf],
         [organo.settings['lumen_prefered_vol']]))
    ener = _opt_ener(tension_array, organo, **energy_opt)
    assert isinstance(ener, float)
    assert ener > 0

def test_wrap_obj_and_const():
    organo = generate_ring(3, 1, 2)
    geom.update_all(organo)
    specs = {
        'face':{
            'is_alive': 1,
            'prefered_area':  1.1*organo.face_df.area,
            'area_elasticity': 1,},
        'edge':{
            'ux': 0.,
            'uy': 0.,
            'uz': 0.,
            'line_tension': 0.01,
            'is_active': 1
            },
        'vert':{
            'adhesion_strength': 0.,
            'x_ecm': 0.,
            'y_ecm': 0.,
            'is_active': 1
            },
        'settings': {
            'lumen_elasticity': 0.01,
            'lumen_prefered_vol': organo.settings['lumen_volume'],
            'lumen_volume': organo.settings['lumen_volume']
            }
        }
    organo.update_specs(specs, reset=True)
    tension_array = np.ones(9)
    energy_opt = {'options': {'gtol': 1e-1, 'ftol': 1e-1}}
    regularization = {'dic':{'apical' : True, 'basal': True},
                      'weight': 0.01}
    kwargs = {'organo': organo, 'minimize_opt': energy_opt, 'initial_dist': 1,
              'regularization': regularization}
    res = _wrap_obj_and_const(tension_array, **kwargs)
    assert res[0] > 0
    assert res[1] > 0
    assert res[2] == 0

def test_create_pyOpt_model():
    obj_fun = _opt_ener
    initial_guess = np.ones(9)
    main_min_opt = {'lb': 0, 'ub': 100, 'method': 'PSQP'}
    assert isinstance(_create_pyOpt_model(obj_fun, initial_guess, main_min_opt),
                      pyOpt.pyOpt_optimization.Optimization)

def test_set_init_point():
    r_in = 1
    r_out = 2
    Nf = 10
    alpha = 1.01
    init_point = set_init_point(r_in, r_out, Nf, alpha)
    assert np.all(np.equal(init_point[Nf:2*Nf], np.zeros(Nf)))
    assert np.all(np.greater(init_point[:Nf], np.zeros(Nf)))
    assert np.all(np.greater(init_point[2*Nf:], np.zeros(Nf)))
    assert np.all(np.greater(init_point[:Nf], init_point[2*Nf:]))
