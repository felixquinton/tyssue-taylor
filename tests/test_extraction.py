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
                                                  _delete_artifact,
                                                  _arrange_centers_clockwise,
                                                  extract_nuclei,
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


def test_extract_membranes():
    gp_dir = os.sep.join(CURRENT_DIR.split(os.sep)[:-2])
    brightfield_path = gp_dir+'/assets/sample_image_brightfield.tiff'
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

def test_delete_artifact():
    centers = np.array([[1, 0], [0, 1], [100, 100]])
    raw_inside = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    res = _delete_artifact(centers, raw_inside)
    assert [100, 100] not in res
    assert [1, 0] in res
    assert [0, 1] in res
    assert len(res) == 2

def test_arrange_centers_clockwise():
    centers = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    org_center = (0, 0)
    res = _arrange_centers_clockwise(centers, org_center)
    assert isinstance(res, list)
    assert len(res) == 4
    assert res.index((1, 0)) == (res.index((0, 1)) - 1) % 4
    assert res.index((0, -1)) == (res.index((1, 0)) - 1) % 4
    assert res.index((-1, 0)) == (res.index((0, -1)) - 1) % 4
    assert res.index((0, 1)) == (res.index((-1, 0)) - 1) % 4

def test_extract_nuclei():
    gp_dir = os.sep.join(CURRENT_DIR.split(os.sep)[:-2])
    dapi_path = gp_dir+'/assets/CELLPROFILER_sample_image_dapi.tiff.csv'
    assert os.path.isfile(dapi_path)
    center_inside = (631.47, 429.66)
    raw_inside = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    img_shape = (3, 3)
    res = extract_nuclei(dapi_path, center_inside, raw_inside, img_shape)
    assert isinstance(res, np.ndarray)
    assert res.shape[1] == 2
    assert res.shape[0] == 136

def test_generate_ring_from_image():
    gp_dir = os.sep.join(CURRENT_DIR.split(os.sep)[:-2])
    brightfield_path = gp_dir+'/assets/sample_image_brightfield.tiff'
    dapi_path = gp_dir+'/assets/CELLPROFILER_sample_image_dapi.tiff.csv'
    organo, inners, outers = generate_ring_from_image(brightfield_path,
                                                      dapi_path)
    for table in (inners, outers):
        assert isinstance(table, np.ndarray)
        assert not np.any(np.isnan(table))
        assert not np.any(np.isinf(table))
        assert table.shape[1] == 2
        assert table.shape[0] > 1
    assert isinstance(organo, AnnularSheet)
    assert not np.any(np.isnan(organo.vert_df.loc[:, ('x', 'y')]))
    assert not np.any(np.isnan(organo.edge_df.loc[:, ('trgt', 'srce', 'face')]))
