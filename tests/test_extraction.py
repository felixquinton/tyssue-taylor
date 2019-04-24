#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:35:28 2018

@author: fquinton
"""
import os
import numpy as np

from tyssue.generation.shapes import AnnularSheet

from tyssue_taylor.segmentation.segment2D import (extract_membranes,
                                                  _recognize_in_from_out,
                                                  _star_convex_polynoms,
                                                  _quick_del_art,
                                                  _quick_clockwise,
                                                  _card_coords,
                                                  generate_ring_from_image)
CURRENT_DIR = os.path.abspath(__file__)


def test_recognize_in_from_out():
    retained_contours = np.array([[[[1, 0]], [[0, 1]]], [[[2, 0]], [[0, 2]]]])
    centers = np.array([[0, 0], [0, 0]])
    radii = np.array([1, 2])
    inside, outside, res_dic = _recognize_in_from_out(retained_contours,
                                                      centers, radii)
    for table in (inside, outside):
        assert not np.any(np.isnan(table))
        assert not np.any(np.isinf(table))
        assert table.shape[1] == 2
        assert table.shape[0] > 1
    assert res_dic['rIn'] > 0
    assert res_dic['rOut'] > res_dic['rIn']
    assert not np.any(np.isnan(res_dic['center_inside']))
    assert not np.any(np.isinf(res_dic['center_inside']))
    assert inside[0][0] == 1
    assert outside[0][0] == 2


def test_extract_membranes():
    gp_dir = os.sep.join(CURRENT_DIR.split(os.sep)[:-2])
    brightfield_path = gp_dir+'/assets/sample_image_actin_surligned.tif'
    assert os.path.isfile(brightfield_path)
    membrane_dic = extract_membranes(brightfield_path)
    for table in (membrane_dic['inside'], membrane_dic['outside'],
                  membrane_dic['raw_inside'], membrane_dic['raw_outside']):
        assert isinstance(table, np.ndarray)
        assert not np.any(np.isnan(table))
        assert not np.any(np.isinf(table))
        assert table.shape[1] == 2
        assert table.shape[0] > 1
    assert membrane_dic['rIn'] > 0
    assert membrane_dic['rOut'] > membrane_dic['rIn']
    assert membrane_dic['img_shape'][0] > membrane_dic['rOut']
    assert not np.any(np.isnan(membrane_dic['center_inside']))
    assert not np.any(np.isinf(membrane_dic['center_inside']))


def test_quick_del_art():
    centers = np.array([[1, 0], [0, 1], [0.1, 0.1]])
    rho, phi = _card_coords(centers, (0, 0))
    res = _quick_del_art(centers, rho, 1)
    assert [0.1, 0.1] not in res
    assert [1, 0] in res
    assert [0, 1] in res
    assert len(res) == 2


def test_quick_clockwise():
    centers = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    org_center = (0, 0)
    rho, phi = _card_coords(centers, org_center)
    res = _quick_clockwise(centers, phi, rho, 1)
    assert len(res) == 4
    assert np.array_equal(res[0], [0, -1])
    assert np.array_equal(res[1], [1,  0])
    assert np.array_equal(res[2], [0,  1])
    assert np.array_equal(res[3], [-1,  0])


def test_star_convex_polynoms():
    gp_dir = os.sep.join(CURRENT_DIR.split(os.sep)[:-2])
    dapi_path = gp_dir+'/assets/sample_image_dapi.tiff'
    model_path = gp_dir+'/assets/star_convex_polynoms/models'
    brightfield_path = gp_dir+'/assets/sample_image_actin_surligned.tif'
    assert os.path.isfile(dapi_path)
    assert os.path.isfile(brightfield_path)
    membrane_dic = extract_membranes(brightfield_path, 2, 9)
    res = _star_convex_polynoms(dapi_path, membrane_dic, model_path)
    assert isinstance(res, np.ndarray)
    assert res.shape[1] == 2
    assert res.shape[0] == 206


def test_generate_ring_from_image():
    gp_dir = os.sep.join(CURRENT_DIR.split(os.sep)[:-2])
    brightfield_path = gp_dir+'/assets/sample_image_actin_surligned.tif'
    dapi_path = gp_dir+'/assets/sample_image_dapi.tiff'
    model_path = gp_dir+'/assets/star_convex_polynoms/models'
    organo, inners, outers, centers = generate_ring_from_image(
        brightfield_path,
        dapi_path,
        scp_model_path=model_path,
        threshold=2, blur=9,
        rol_window_inside=100,
        rol_window_outside=20)
    for table in (inners, outers):
        assert isinstance(table, np.ndarray)
        assert not np.any(np.isnan(table))
        assert not np.any(np.isinf(table))
        assert table.shape[1] == 2
        assert table.shape[0] > 1
    assert isinstance(organo, AnnularSheet)
    assert not np.any(np.isnan(organo.vert_df.loc[:, ('x', 'y')]))
    assert not np.any(np.isnan(organo.edge_df.loc[:, ('trgt', 'srce', 'face')]))
