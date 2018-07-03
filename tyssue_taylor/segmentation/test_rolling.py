#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:56:25 2018

@author: fquinton
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import cv2 as cv

from tyssue.generation import generate_ring
from tyssue.draw.plt_draw import quick_edge_draw
from tyssue.solvers.sheet_vertex_solver import Solver
from tyssue.config.draw import sheet_spec
from tyssue.draw.plt_draw import sheet_view

from tyssue_taylor.adjusters.adjust_annular import prepare_tensions, adjust_tensions
from tyssue_taylor.models.annular import AnnularGeometry as geom
from tyssue_taylor.models.annular import model
from tyssue_taylor.adjusters.cost_functions import _distance
from tyssue_taylor.segmentation.segment2D import generate_ring_from_image

brigthfield_path = 'images/Images/r01c01f03p55-ch3sk1fk1fl1.tiff'
dapi_path = 'images/CELLPROFILER_r01c01f03p55-ch1sk1fk1fl1.tiff.csv'
organo, centers, org_center, inner_vs = generate_ring_from_image(brigthfield_path, dapi_path, 28, 9)