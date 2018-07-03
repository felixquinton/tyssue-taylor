#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:35:28 2018

@author: fquinton
"""
from scipy.interpolate import splrep
import numpy as np
x = np.concatenate((np.linspace(-180, 180, 720),np.linspace(180, -180, 720)))
y = np.concatenate((np.linspace(-180, 180, 720),np.linspace(-180, 180, 720)))
spl = splrep(x, y, per=True)
